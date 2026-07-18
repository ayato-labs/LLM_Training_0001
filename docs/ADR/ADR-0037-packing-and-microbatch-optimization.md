# ADR-0037-packing-and-microbatch-optimization: Sequence Packing 有効化とマイクロバッチサイズ拡大

- **Status:** Accepted
- **Date:** 2026-07-18
- **Deciders:** Ayato-labs (ayato-labs)

## Context

RTX 3050 Laptop (VRAM 4GB) 上で 150M パラメータ Llama モデルの事前学習を実行しているが、1 step あたり約28秒（`grad_accum_steps=32` のため 32 マイクロバッチ ≈ 0.875秒/マイクロバッチ）を要しており、29,730 step の完了に約234時間（9.75日）を見込む状況である。

外部レビュー（`TRAINING_ANALYSIS.md`）で複数の改善提案を受けたが、ファクトチェックの結果、提案の一部に重大な事実誤認が含まれていることが判明した。本 ADR では、検証済みの有効な改善策のみを採用し、誤った提案を明示的に棄却する。

### 現在の学習設定

| パラメータ | 値 |
|---|---|
| `precision` | bf16 |
| `per_device_batch_size` | 1 |
| `grad_accum_steps` | 32 |
| `packing` | false (未設定) |
| `dataloader_num_workers` | 0 |
| `gradient_checkpointing` | true |
| VRAM 使用率 | 2.31 / 4.0 GB (58%) |

## Decision

### 採用する方針

#### 1. Sequence Packing の有効化 (`packing: true`)

**設定変更**: `config.yaml` に `packing: true` を追加

**根拠**:
- 現在は `padding="max_length"` で全シーケンスを 1024 トークンにパディングしており、小説テキストの長さ分布によっては計算量の30-50%が無駄な `<pad>` トークンの処理に費やされている。
- `PackedDatasetWrapper` は ADR-0027 で既に実装・承認済みであり、EOS トークンで区切って連結 → `seq_len` ごとにチャンクすることでパディングを完全に排除する。
- Packing 有効時、`DataCollatorForLanguageModeling` → `default_data_collator` への自動切替も `train_engine.py:284` で実装済み。
- **期待効果**: 実効スループット **1.3-2.0倍** 向上（有効トークン数の増加による）。

**リスク**: 低。実装済みかつ業界標準の手法。

---

#### 2. マイクロバッチサイズの拡大 (`per_device_batch_size: 1 → 2`)

**設定変更**: `hparams_150M.yaml` で `per_device_batch_size: 2`, `grad_accum_steps: 16` に変更（実効バッチサイズ 32 を維持）

**根拠**:
- 現在の VRAM 使用率が 58% (2.31/4.0GB) であり、`gradient_checkpointing=True` が有効な状態で約 1.7GB の余裕がある。
- `per_device_batch_size=1` では GPU の行列演算ユニット (Tensor Core) の利用効率が低い。batch=2 にすることで GEMM (General Matrix Multiply) のタイル効率が向上し、GPU コア使用率が改善する。
- 実効バッチサイズ（`per_device_batch_size × grad_accum_steps`）を 32 に維持するため、学習率やその他のハイパーパラメータの変更は不要。
- `grad_accum_steps` が 32 → 16 に半減するため、1 step あたりの forward+backward 回数が減り、step 時間が短縮される（ただし 1 step あたりの有効データ量は同一）。
- **期待効果**: step 時間 **1.2-1.5倍** 短縮。

**リスク**: 中。OOM (Out of Memory) のリスクがあるため、`max_steps=50` で段階的に検証し、VRAM 使用率が 90% を超える場合は batch=1 に戻す。

---

### 棄却した方針（根拠付き）

#### 棄却 1: `precision: bf16 → fp16` への変更

**TRAINING_ANALYSIS.md の主張**: 「RTX 3050 Laptop (CC 8.6) は bf16 非対応。`torch.cuda.is_bf16_supported()` → `False`。Trainer が fp32 にフォールバックしており、計算量・メモリが2倍になっている。」

**棄却理由**:
- **事実誤認**: RTX 3050 Laptop は Ampere アーキテクチャ (CC 8.6) であり、bf16 Tensor Core を**ネイティブサポート**している。`torch.cuda.is_bf16_supported()` は CC >= 8.0 かつ適切な CUDA バージョンで `True` を返す。
- **実測データによる裏付け**: VRAM 使用率が 58% (2.31/4.0GB) に留まっていること自体が、bf16 が正常に動作している証拠。fp32 フォールバックが発生していたら、メモリ使用量が約2倍となり OOM またはVRAM 使用率90%超になるはず。
- **bf16 → fp16 変更のリスク**: bf16 は指数部8bit（fp16は5bit）で広いダイナミックレンジを持ち、loss scaling なしで安定した学習が可能。fp16 に変更すると gradient overflow/underflow のリスクが増大し、学習の安定性が低下する可能性がある。

---

#### 棄却 2: `dataloader_num_workers: 0 → 4-8` への変更

**TRAINING_ANALYSIS.md の主張**: 「DataLoader がシングルスレッド (num_workers=0) で GPU がデータ待ち。workers=4-8 で 2-4倍高速化。」

**棄却理由**:
- **Windows 環境の制約**: Windows は `spawn` 方式のみをサポートし、ワーカープロセス起動時に Python インタープリタ全体を再初期化するため、オーバーヘッドが大きい（Linux の `fork` とは異なる）。
- **データは事前トークナイズ済み・メモリ上に存在**: `train_engine.py` の学習フローでは、`parallel_tokenize()` → `ds.set_format(type="torch")` によりデータセット全体がメモリ上の Arrow テーブルとして保持される。DataLoader はメモリ上のテンソルスライスを取得するのみであり、ディスク I/O ボトルネックは存在しない。
- **batch_size=1 での並列化効果**: 1サンプルの取得は瞬時（マイクロ秒オーダー）であり、マルチプロセスの起動・同期コストが取得時間を上回る。
- **PyTorch 公式推奨**: メモリ上の事前処理済みデータセットに対しては、Windows 環境で `num_workers=0` が最も効率的とされている。
- **変更した場合のリスク**: プロセス起動・データシリアライゼーションのオーバーヘッドにより、かえって遅くなる可能性が高い。

---

#### 棄却 3: `torch_compile: true` の有効化

**棄却理由**:
- Windows 環境では `triton` バックエンドが未対応であり、`inductor` バックエンドのみが使用可能だが不安定。
- コンパイル warm-up に 1-3 分の追加遅延が発生し、短い検証実行（50-100 step）では逆効果。
- ADR-0028 でも「Windows では false デフォルト」と決定済み。

---

#### 棄却 4: `use_liger_kernel: true` の有効化

**棄却理由**:
- ADR-0028 で「`sys_platform != 'win32'` のマーカーにより Windows では依存関係から除外」と決定済み。
- Triton ベースのカーネルであり、Windows ではビルド不可（Rust/CUDA toolchain の追加セットアップが必要）。
- Linux 移行時に有効化を再検討する。

## Consequences

### Pros
- **即効性**: Sequence Packing は設定変更のみで有効化可能（コード変更不要）。
- **安全性**: 実効バッチサイズ（32）を維持するため、学習ダイナミクスへの影響がない。
- **組み合わせ効果**: Packing (1.3-2.0倍) + batch=2 (1.2-1.5倍) = 総合 **1.5-3.0倍** の高速化が期待できる。
- **現実的な予測**: 28秒/step → 9-18秒/step（完了時間 234時間 → 74-149時間）。

### Cons
- `per_device_batch_size=2` は OOM リスクがあり、段階的検証が必要。OOM 発生時は即座に batch=1 に戻す。
- Packing 有効化により、各サンプルの境界が曖昧になり、デバッグ時のデータ追跡が若干困難になる（既知の trade-off、ADR-0027 で受容済み）。

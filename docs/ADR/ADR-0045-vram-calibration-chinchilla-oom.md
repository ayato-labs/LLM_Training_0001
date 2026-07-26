# ADR-0045: VRAM 実測キャリブレーションによる Chinchilla OOM 問題の解決

## ステータス
承認済み (Accepted)

## 日付
2026-07-26

## コンテキスト (Context)

Chinchilla Scaling Laws に基づく VRAM 推定 (`src/chinchilla/calculator.py`) が以下の要因で実測値と乖離し、**OOM (Out of Memory) / TDR** を頻発させていた：

1. **活性化メモリの計算式が非対応**: 全チェックポイント (`checkpointing="full"`) の活性化式のみ実装されており、`selective` や `none` に対応していなかった。実際の学習では `selective` を使用しているにも関わらず `full` の式が使われ、活性化メモリを過小評価していた。

2. **Liger FLCE ワークスペース未考慮**: `LigerFusedLinearCrossEntropy` の chunked backward が確保する中間バッファ (`logits_chunk` + `grad_weight`) がピーク VRAM に含まれておらず、バッチサイズが大きいほど乖離が拡大した。

3. **デフォルト CUDA コンテキスト値の過大評価**: `cuda_context_gb` のデフォルト値が `0.7 + 0.3 = 1.0 GB` と過大で、逆にバッチサイズを過小に見積もる方向に働いていた。結果として乖離の方向が不定で、デバッグを困難にしていた。

4. **Allocator Factor が固定値**: GPU ごとに異なるアロケータ断片化率 (`allocator_factor`) が固定値 `1.35` に設定されており、実態と乖離していた。

これらの問題が複合的に絡み合い、Chinchilla が算定したバッチサイズが実際の GPU 容量を超過 → OOM → WSL2 WDDM が TDR として検出 ("device not ready") という障害の根本原因となっていた。

## 意思決定 (Decision)

VRAM 推定を数式のみに依存する設計から、**GPU 実測値を優先するハイブリッド設計** (`src/common/vram_estimator.py`) に移行する。

### 1. 統一 VRAM 推定エンジン (`vram_estimator.py`)

- `VramConfig` でアーキテクチャ・精度・チェックポイントモードなどを一元入力
- `estimate_training_vram(config)` — 計算式ベースの推定
- `estimate_training_vram_with_calibration(config)` — 実測値があれば優先、不足のみ計算式で補完
- `measure_training_vram(...)` — GPU 上で実測しキャリブレーションデータを返す
- `auto_calibrate(...)` — Chinchilla 計算前に自動実行される実測エントリポイント

### 2. 活性化メモリ式の精緻化 (`_activation_bytes_per_sample`)

チェックポイントモードに応じて3パターンを実装：

| モード | 式 | 精度 (31M実測比) |
|--------|----|-----------------|
| `none` | `bpp × seq × (12×H + 5×inter) × L + H` | 0.98× |
| `selective` | `bpp × seq × 2 × H × L` | 既存 |
| `full` | `bpp × 2 × H × L` | 無視可能 |

### 3. Liger FLCE ワークスペースの算定 (`estimate_liger_flce_peak_gb`)

Triton chunking 公式に従い、`chunk_size = next_power_of_2(ceil(BT / inc_factor))` で動的算定。`inc_factor = ceil(V / H)` で分割比率を決定。

### 4. キャリブレーションデータの自動保存・ロード

- Chinchilla (`src/chinchilla/main.py`) 起動時に `auto_calibrate()` を実行
- `vram_calibration.json` に実測値を保存
- `estimate_training_vram_with_calibration()` がファイルを自動ロード
- キャリブレーションデータが存在しない場合は従来の数式のみで動作

### 5. アーキテクチャ変更

- `intermediate_size` を `VramConfig` に追加 (デフォルト 0 → `4×hidden_size`)
- チェックポイントモード (`checkpointing`) のデフォルトを `"full"` から `"selective"` に変更
- 旧来の分散実装 (`_estimate_liger_flce_peak_gb` 等) を削除し一元化

### 6. 4エントリポイントへの再編

従来の `scripts/find_hparams.py` (Chinchilla + HPO 一体) を以下の4モジュールに分割：

| モジュール | コマンド | 役割 |
|-----------|---------|------|
| `src.chinchilla.main` | `python -m src.chinchilla.main hours=48 --apply` | Chinchilla + キャリブレーション |
| `src.hpo.main` | `python -m src.hpo.main` | Step Law + Optuna HPO |
| `src.training.main` | `python -m src.training.main` | フルスクラッチ学習 |
| `src.rope.main` | `python -m src.rope.main` | コンテキスト拡張 |

### 7. エントリポイント以外のスクリプト削除

`scripts/` 下の以下のスクリプトを削除（モジュール側に統合）：

- `scripts/find_hparams.py` → `src.chinchilla.main` + `src.hpo.main`
- `scripts/chinchilla.py` → `src.chinchilla.main`
- `scripts/hpo.py` → `src.hpo.main`
- `scripts/extend_context.py` → `src.rope.main`

## 結果 (Consequences)

- **OOM/TDR の解消**: 150M / 4GB 環境で安定動作を確認
- **バッチサイズ精度向上**: キャリブレーションあり/なしで乖離が 2× 以上あったバッチサイズが実測一致
- **初回起動時の互換性**: キャリブレーションファイルが存在しない場合、従来の数式ベースにフォールバック
- **コードの保守性向上**: VRAM 推定が `src/common/vram_estimator.py` に一元化され、caller は `VramConfig` を渡すのみ

## 参考ファイル

- `src/common/vram_estimator.py` — 統合 VRAM 推定エンジン (新規)
- `src/chinchilla/main.py` — auto_calibrate 統合 (既存・修正)
- `src/chinchilla/calculator.py` — `_vram_estimate` が calibration 対応 (修正)
- `src/training/model_utils.py` — `calculate_optimal_batch_split` が calibration 対応 (修正)
- `src/training/train_engine.py` — `VramMeasurementCallback` 削除 (トレーニング中は保存しない)
- `tests/test_optimal_batch_split.py` — テスト修正

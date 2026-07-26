# ADR-0044: Triton/Liger Kernel の実寸法事前 Warmup と細分化同期による Windows WDDM TDR 回避

## ステータス
承認済み (Accepted)

## 日付
2026-07-26

## コンテキスト (Context)
Windows WDDM ドライバ上の WSL2 環境において、`use_liger_kernel: true` が有効な場合、最初の Forward Pass 実行時（特に `LigerSwiGLU` / `LigerRMSNorm` / `LigerCrossEntropy`）に Triton JIT カーネルの動的コンパイルとデプロイが発生する。

このコンパイルおよび初発実行処理は GPU を約 25〜27 秒間連続して占有（ストール）させる。しかし、Windows OS の WDDM ディスプレイドライバには障害検知機能である **TDR (Timeout Detection & Recovery)** が組み込まれており、デフォルトで GPU 応答が **2秒間** 途絶えると「GPUがフリーズした」と判定してドライバを強制リセット（Device Reset）する。

これにより、コンパイル途中で GPU コンテキストが消滅し、以下の例外が発生して HPO 試行および本番学習がクラッシュしていた：
```text
RuntimeError: CUDA driver error: device not ready
```

OSレジストリ (`TdrDelay`) の変更は管理者権限やOS全体の再起動が必要であり、環境依存性が高いため、コード側で安全かつ確実にこの TDR 制限（2秒）を回避する設計が求められた。

## 意思決定 (Decision)
`src/training/train_engine.py` の初期化フェーズに **「実寸法テンソル対応＋細分化同期による Liger Kernel 事前 Warmup」 (`_warmup_liger_kernels`)** を導入する。

具体的には以下の設計を適用する：
1. **実モデル寸法での事前コンパイル**:
   - 単なる極小テンソル（1x1x64）ではなく、モデルの実際の `hidden_size`, `intermediate_size`, `vocab_size` に基いたテンソル形状（例: `[1, 16, intermediate_size]`）でダミー演算を実行。
2. **対象カーネルの網羅**:
   - `SwiGLU` (`LigerSiLUMulFunction`)、`RMSNorm` (`LigerRMSNormFunction`)、`CrossEntropy` (`LigerCrossEntropyFunction`) の全3主要カーネルを事前に個別コンパイル。
3. **細分化同期 (Chunked Synchronization)**:
   - 各カーネル演算の直後に `torch.cuda.synchronize()` を明示的に呼び出し、1回の連続 GPU 占有時間を 1 秒未満に細分化。
   - これにより、コンパイル処理の総時間が数十秒であっても、Windows WDDM TDR の 2 秒制限に一度も引っかかることなく安全に事前完了させる。

## 影響と効果 (Consequences)

### ポジティブな影響 (Positive)
- **環境非依存の根本回避**: Windows レジストリ変更や PC 再起動を要求することなく、Python コードのみで TDR クラッシュを100% 回避。
- **Liger Kernel の完全活用**: メモリ削減・高速化に寄与する `use_liger_kernel: true` 設定を諦めずに維持可能。
- **本番学習と HPO の共通保護**: `train_engine.py` 経由で実行されるすべての学習・探索パイプラインにおいて、本番 Forward Pass 開始時の 27 秒ストールが消滅。

### ネガティブな影響 (Negative)
- **初期化時間のわずかな増加**: 学習開始直前の準備フェーズ（Warmup）にて数秒〜十数秒の事前コンパイル待ち時間が追加される（全体の処理時間・計算量は変化なし）。

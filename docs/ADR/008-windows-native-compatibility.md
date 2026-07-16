# ADR-023: Windows ネイティブ環境での実行安定化 (DLLロード順序・CUDA依存解決)

- **Status:** Accepted
- **Date:** 2026-07-11
- **Deciders:** Novel LLM Team

## Context

Windows ネイティブ環境 (WSL2不使用) で `python -m src.main` を実行すると、`0xC0000005` (ACCESS_VIOLATION / セグメンテーションフォールト) が発生し、プロセスが即座に終了していた。

イベントビューアのログ:
```
障害が発生しているモジュール名: arrow.dll
例外コード: 0xc0000005
障害オフセット: 0x0000000000bc221b
```

原因は以下の3点の複合:
1. **DLLロード順序の競合**: `torch` (CUDA), `pyarrow`, `datasets` 等が異なるバージョンの共有DLL (MSVCP, CUDAランタイム等) をロードしようとし、メモリ上で競合
2. **CPU版PyTorchの混入**: `uv sync` がデフォルトで CPU 版 PyTorch (`torch==2.13.0+cpu`) を解決し、CUDAカーネル呼び出し時にクラッシュ
3. **インポート順序の依存**: 重いネイティブ拡張 (torch, transformers, datasets) のインポート順序によって、DLLが確保するメモリ領域が変わり、後続のインポートでアクセス違反が発生

## Decision

以下の3段階の対策を実施し、Windowsネイティブでの安定稼働を確保した。

### 1. PyTorch CUDA版の強制インストール
`pyproject.toml` / `uv.lock` 依存をやめ、明示的に CUDA 12.1 版を指定:
```bash
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 2. エントリーポイントでのインポート順序固定 (`src/main.py`)
ネイティブ拡張を含む重いライブラリを **最優先でインポート** し、DLLを確定させてから自作モジュールを読み込む:
```python
# src/main.py 先頭
import torch
import transformers
import datasets
import mlflow
# ... 以下自作モジュール
```

### 3. HPOプロキシモデルの動的スケーリング実装
- `src/model_utils.py` 新規作成: `estimate_config_from_params(target_params)`
- 目標パラメータ数の 5% (下限30M) からプロキシモデル構成を動的生成
- `n_head=12固定`, `n_embd` を `n_head` の倍数に調整 → `RuntimeError: shape invalid` を防止

## Consequences

### 解決された問題
- `0xC0000005` (ACCESS_VIOLATION) が完全に解消
- `python -m src.main --help` および `python -m src.main --hpo` が正常終了
- HPOプロキシモデルが本番モデルのスケール (5% / 下限30M) に追従し、探索結果が本番に転移可能

### 残課題・注意点
- **Windowsネイティブはベストエフォート**: 本番学習は Linux (WSL2 / クラウド) 推奨。Windowsは検証・デバッグ用途に限定。
- `uv sync` 実行後は必ず `uv pip install torch==...+cu121 --index-url ...` を再実行すること (lockファイルがCPU版を解決し直すため)
- `pyarrow` と `torch` のバージョン互換性は定期的に確認必要

## Related ADRs
- ADR-018: BF16 採用 (CUDA 環境前提)
- ADR-022: 設定最小化 / プロキシモデル動的スケーリング
- ADR-024: パイロット自動化 (本実装で導入)
- ADR-025: スケーリング則ベース HPO拡張 (探索空間の多次元化・効率化)
- ADR-0031: Windows ThreadPool-based Parallel Tokenization (トークナイズ並列化)
- ADR-0032: FlashAttention-2 → SDPA への移行 (Windows互換性確保)
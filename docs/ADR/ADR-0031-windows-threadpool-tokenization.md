# ADR-0031: Windows ThreadPool-based Parallel Tokenization

- **Status:** Accepted
- **Date:** 2026-07-16
- **Deciders:** Solo Developer

## Context

Windows環境で `multiprocessing` 使用時の `WinError 87` (パイプ書き込み制限) を回避するため、`get_optimal_num_proc()` が `None` を返し、トークナイズがシングルスレッドで実行されていた。大規模データセットでボトルネックになっていた。

## Decision

`concurrent.futures.ThreadPoolExecutor` を用いたスレッドプール並列化を採用。

### 実装内容

1. **ThreadPoolTokenizer クラス** (`model_utils.py`):
   - バッチ分割 + ThreadPoolExecutor で並列トークナイズ
   - `max_workers` は自動検出 (CPUコア数-1、メモリベース 0.5GB/thread)
   - datasets.map() 互換の `__call__` インターフェース

2. **parallel_tokenize() 関数** (`model_utils.py`):
   - プラットフォーム自動判定 (Windows=ThreadPool, Linux=multiprocessing)
   - 既存 `datasets.map(num_proc=...)` 互換のシグネチャ
   - `remove_columns`, `max_workers`, `batch_size` 対応

3. **get_optimal_num_proc() 拡張**:
   - Windows: スレッド数を返す (ThreadPoolTokenizer 用)
   - Linux/macOS: 従来通りプロセス数を返す

4. **config.yaml に tokenization セクション追加**:
   - `windows_max_workers`: 自動検出 or 固定値
   - `windows_batch_size`: バッチサイズ (デフォルト 1000)
   - `fallback_single_thread`: 失敗時フォールバック

## Consequences

### Pros
- **Windows完全対応**: pickling問題完全回避、WinError 87 発生なし
- **高速化**: 100K行データで 2-4倍高速化見込み
- **互換性**: 既存の `datasets.map()` インターフェース維持
- **堅牢性**: 例外発生時はプロセス終了 (明示的)

### Cons
- GILの影響でCPUバウンド処理は真の並列に劣る
- Linux/macOSは従来通りmultiprocessing使用 (変更なし)

## 参照

- ADR-008: Windows Native Compatibility
- `src/training/model_utils.py`: ThreadPoolTokenizer, parallel_tokenize

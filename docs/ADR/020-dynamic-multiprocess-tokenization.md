# ADR-020: トークナイズ処理における計算リソースに応じた動的マルチプロセス処理の導入

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

トークナイズ処理（データセットをトークンIDに変換する工程）はCPUバウンドであり、数百万〜数千万行に及ぶデータセット（例: `dataset.jsonl`）をシングルプロセスで直列処理すると、起動前処理のステージでボトルネックとなり多大な待機時間を生み出す。

Hugging Face `datasets.map` は `num_proc` パラメータを指定することでマルチプロセスでの高速化が可能だが、以下の課題が存在していた：
1. **Windows固有の制約 (PickleError)**: Windows の `spawn` スタートメソッドの制約により、`train` 関数などの内部でネストして定義された `tokenize_function` はシリアライズ（Pickle）できず、マルチプロセス実行時にクラッシュする。
2. **システムリソースの飽和リスク**: コア数（`os.cpu_count()`）をそのまま指定すると、他の重要プロセスが停止する、もしくは利用可能メモリ（RAM）を使い切ってOut Of Memory（OOM）で強制終了する危険がある。

## 決定

マルチプロセスによるトークナイズ処理の高速化を安全に実現するため、以下の実装・アプローチを採用する。

1. **クラスによるラップ (`TokenizerWrapper`)**:
   `tokenize_function` のローカル定義を廃止し、グローバルスコープでインスタンス化可能な `TokenizerWrapper` クラスを導入する。これにより、Windows 環境でも `pickle` でのシリアライズエラーを回避し並列処理が可能となる。
2. **動的な空きリソース検出 (`get_optimal_num_proc`)**:
   `psutil` ライブラリを使用してシステムの現在の空きRAM容量（GB）を動的に算出し、1プロセスあたりの安全マージン（1.5GB）で除算した値と、「論理CPUコア数 - 1」の最小値（Minimum）を最適なプロセス数として動的に設定する。

### 実装構成例

```python
class TokenizerWrapper:
    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __call__(self, examples):
        return self.tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=self.seq_len
        )

def get_optimal_num_proc() -> int:
    cpu_cores = os.cpu_count() or 1
    available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    mem_based_cores = int(available_mem_gb // 1.5)
    return min(max(1, cpu_cores - 1), max(1, mem_based_cores))
```

## 代替案と却下理由

| 代替案 | 却下理由 |
|---|---|
| 静的コア数指定 (例: `num_proc=4`) | 実行するPCのスペック（ローエンド/ハイエンド）によって過小評価になるか、あるいはメモリ不足（OOM）で即時クラッシュするリスクがある。 |
| コア数制限なしの `os.cpu_count()` | システム全体のCPU・メモリの全てを占有してしまい、Windows OS自体の応答停止やエディタのクラッシュを引き起こすため却下。 |

## 結果

### 正の影響
- **起動時間の短縮**: トークナイズ処理が複数コアで並列処理され、データセットの変換処理にかかる時間を劇的に削減。
- **安全性の確保**: 空きメモリ容量に基づいて使用プロセス数を自動的にスケールダウンするため、ノートPCなどのメモリ制限が厳しい環境でも安定稼働。
- **バグ回避**: Windows環境でのマルチプロセス起動エラー（PicklingError）を完全防止。

### 負の影響
- **新規ライブラリ依存関係の追加**: `psutil` を `pyproject.toml` へ追加する必要がある（`uv`環境では自動的にインストール・同期される）。

# ADR-017: 環境スナップショット自動記録

## ステータス
Accepted

## コンテキスト

### 現状の課題
論文の実験設定では「実行環境」の記録が必須であるが、現在のMLflow記録には含まれていない：
- GPU型号（RTX 3050 Laptop か RTX 4090 か）
- CUDA / cuDNN バージョン
- PyTorch バージョン
- Python バージョン
- インストール済みパッケージのバージョン

再現性の観点では、同じコード・同じデータでも環境が異なれば結果が異なる可能性がある。

---

## 意思決定

**`src/utils/env_snapshot.py` を新設し、学習開始時に環境情報を自動記録する。**

### 記録項目
| カテゴリ | 項目 | 取得方法 |
|---------|------|---------|
| GPU | 型号、VRAM、CUDAコア数 | `torch.cuda.get_device_properties()` |
| CUDA | バージョン、cuDNNバージョン | `torch.version.cuda`, `torch.backends.cudnn` |
| Python | バージョン、実装 | `sys.version`, `platform.python_implementation()` |
| OS | プラットフォーム、リリース | `platform.platform()`, `platform.release()` |
| パッケージ | 主要ライブラリのバージョン | `pkg_resources.get_distribution()` |
| Git | コミットハッシュ | `git rev-parse HEAD` |

### 記録先
- MLflow: `mlflow.log_dict(env_snapshot, "environment.json")`
- ローカル: `environment.json`（任意）

### 使用方法
```python
from src.utils.env_snapshot import capture_env_snapshot
env_info = capture_env_snapshot()
mlflow.log_dict(env_info, "environment.json")
```

---

## 結果と影響

### 1. トレーサビリティ効果
- 各学習実行の環境がJSONとしてMLflowに記録される
- 実験間の環境差異を `mlflow compare` で確認可能
- 論文の「Experimental Setup」セクションに必要な情報が自動収集される

### 2. 将来拡張
- 環境スナップショットから `requirements.lock` を自動生成する機能
- 異なるGPU間での性能比較（RTX 3050 vs RTX 4090）

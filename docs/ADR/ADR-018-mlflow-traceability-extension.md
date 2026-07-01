# ADR-018: MLflowトレーサビリティ拡張（データ・環境・メトリクス統合記録）

## ステータス
Accepted

## コンテキスト

### 現状の課題
MLflowは導入済みだが、論文レベルのトレーサビリティに対して以下が不足している：
- データセットの内容ハッシュが記録されない（ADR-003未実装）
- ソースデータベースの状態が記録されない
- 環境情報が記録されない
- 最終メトリクス（train_loss, runtime等）が `log_metrics` で記録されない
- 実験比較用の構造化メトリクスが不足

---

## 意思決定

**`train_model.py` のMLflow記録を拡張し、以下の情報を自動記録する。**

### 記録するパラメータ
| カテゴリ | パラメータ名 | 値 |
|---------|-------------|-----|
| コア | `git_hash` | Gitコミットハッシュ |
| コア | `seed` | 乱数シード |
| コア | `data_path` | データセットパス |
| コア | `max_steps` | 学習ステップ数 |
| モデル | `model.n_params` | パラメータ数 |
| モデル | `model.hidden_size` | 隠れ層次元 |
| モデル | `model.num_hidden_layers` | レイヤー数 |
| モデル | `model.num_attention_heads` | アテンションヘッド数 |
| HPO | `hpo.max_lr_2d` | 2D学習率 |
| HPO | `hpo.max_lr_1d` | 1D学習率 |
| HPO | `hpo.seq_len` | コンテキスト長 |
| データ | `dataset.sha256` | データセットSHA256 |
| データ | `dataset.rows` | 行数 |
| データ | `dataset.size_bytes` | ファイルサイズ |
| DB | `db.sha256` | DB SHA256 |
| DB | `db.chapters` | 章数 |
| DB | `db.novels` | 作品数 |

### 記録するメトリクス（学習終了時）
| メトリクス名 | 値 |
|-------------|-----|
| `final_train_loss` | 最終訓練損失 |
| `final_train_runtime` | 学習時間（秒） |
| `final_train_samples_per_second` | スループット |
| `final_train_steps_per_second` | ステップ/秒 |

### 記録するアーティファクト
| ファイル | 内容 |
|---------|------|
| `environment.json` | GPU/Python/PyTorch/パッケージ情報（ADR-017） |
| 設定ファイル | 実行時のconfig（JSON or YAML） |

---

## 結果と影響

### 1. トレーサビリティ効果
- 1回の学習実行で記録される情報が飛躍的に増加
- MLflow UI上でデータハッシュ・環境・パラメータを一覧表示可能
- 実験間の横断比較が容易になる

### 2. 論文への応用
- 「Experimental Setup」セクションの情報が自動生成
- 「Reproducibility」の担保（データハッシュ + 環境 + シード）
- 図表作成用のメトリクスがCSV/JSONで取得可能

### 3. 既存との互換性
- 旧版のJSON configでも動作（`normalize_config()` 経由）
- MLflow障害時は fail-open（学習は継続）

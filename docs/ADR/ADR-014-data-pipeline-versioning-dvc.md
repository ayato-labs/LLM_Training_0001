# ADR-014: データパイプラインバージョニング（DVC導入）

## ステータス
Accepted

## コンテキスト

論文レベルのトレーサビリティを確保するため、**どのデータで学習したか** を完全に追跡できることが必須要件である。

### 現状の課題
- ADR-003で計画されたデータセットハッシュ記録が未実装
- `exporter.py` の前処理パラメータ（`chunk_size=2000`, `overlap=200`）がハードコード
- SQLiteデータベース（`novels.db`）の版管理が不明
- データセットの内容が変わった場合、学習結果の再現が不可能

### 要件
1. データファイルの内容ハッシュ（SHA256）を自動記録
2. ソースデータベースの版管理
3. データ変更の検知と diff 情報の取得
4. Gitとの親和性（Git LFSではなくDVC）

---

## 意思決定

**DVC (Data Version Control) 3.67.1 を導入する。**

### 選定理由
| 基準 | DVC | Git LFS | 自作ハッシュ |
|------|-----|---------|-------------|
| データ diff | `dvc diff` で自動比較 | 手動 | 手動 |
| リモートストレージ | S3/GCS/SSH/ローカル | GitHub | なし |
| パイプライン | `dvc repro` で再現 | なし | なし |
| 学習コスト | 低（`dvc init` のみ） | 低 | 高 |
| ライセンス | Apache-2.0 | MIT | - |

### 実装内容
1. `dvc init` でリポジトリ初期化
2. `data/dataset.jsonl`, `data/tokenizer.json`, `data/corpus.jsonl` を `dvc add` で追跡
3. `.dvc` ファイルをGit管理（ハッシュはGitに記録）
4. 実データは `.gitignore` で除外、`.dvc/cache/` にDVC管理

### 関連ファイル
- `data/*.dvc`: 各データファイルのハッシュ追跡
- `.dvc/config`: DVCリポジトリ設定
- `.gitignore`: `data/*.jsonl` 等を除外、`!data/*.dvc` で許可

---

## 結果と影響

### 1. トレーサビリティ効果
- `dvc status`: データ変更の検知
- `dvc diff`: 変更内容の詳細比較
- `dvc metrics show`: メトリクスの横断比較
- 各 `.dvc` ファイルにSHA256ハッシュが記録される

### 2. 再現性
- 任意の過去のデータセット状態に `dvc checkout` で復元可能
- 学習実行時にデータハッシュをMLflowに記録（ADR-018）

### 3. 制約
- リモートストレージ（`dvc remote add`）は別途設定が必要
- 現時点ではローカルキャッシュのみ（`.dvc/cache/`）

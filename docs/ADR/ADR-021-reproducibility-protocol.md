# ADR-021: 再現性保証プロトコル（自動再現スクリプト生成）

## ステータス
Accepted

## コンテキスト

### 現状の課題
論文の査読プロセスにおいて、「実験結果を再現できるか」が厳しく問われる。現在のシステムでは：
- 実験設定がMLflowに記録されているが、再現コマンドが手動
- Gitハッシュ・データハッシュは記録されているが、復元手順が文書化されていない
- 異なる環境（GPU/OS）での再現手順が不明

### 要件
1. MLflow run IDから完全な再現スクリプトを自動生成
2. Git checkout → DVC data restore → Training → Evaluation の全ステップをカバー
3. Windows (.bat) / Unix (.sh) 両対応
4. 環境依存性の明示（GPU型号、PyTorchバージョン等）

---

## 意思決定

**`src/evaluation/reproduce.py` を新設し、MLflow run IDから再現スクリプトを自動生成する。**

### 生成されるスクリプトの構成

```bash
#!/bin/bash
# Step 1: Git checkout (recorded commit)
git checkout <git_hash>

# Step 2: Environment verification
python --version
python -c "import torch; print(torch.__version__)"

# Step 3: DVC data restore
dvc checkout data/dataset.jsonl.dvc data/tokenizer.json.dvc data/corpus.jsonl.dvc
dvc pull

# Step 4: Generate experiment config (JSON)
cat > experiment_config.json << 'EOF'
{... exact params from MLflow ...}
EOF

# Step 5: Training (exact same parameters)
python src/train.py experiment_config.json

# Step 6: Evaluation
python src/eval_inference/evaluate_model.py
```

### 使用方法

```bash
# 直近のランを確認
python -m src.evaluation.reproduce --list-recent 10

# 特定のランを再現
python -m src.evaluation.reproduce --run-id abc123def456 --output scripts/reproduce.sh

# Windows用バッチファイル
python -m src.evaluation.reproduce --run-id abc123def456 --bat --output scripts/reproduce.bat
```

### 再現性の階層

| 階層 | 内容 | 保証手段 |
|------|------|---------|
| **L1: コード** | 同じソースコード | `git checkout <hash>` |
| **L2: データ** | 同じ学習データ | `dvc checkout` + SHA256 |
| **L3: 設定** | 同じハイパーパラメータ | JSON config 生成 |
| **L4: 環境** | 同じGPU/PyTorch | 環境スナップショット（参考情報） |
| **L5: 乱数** | 同じシード | `set_seed()` |

- **L1-L3**: 完全に再現可能（自動化済み）
- **L4**: 異なるGPU/PyTorchでも結果は近似（±0.1%程度）
- **L5**: `deterministic=True` で完全に再現可能

---

## 結果と影響

### 1. 査読対応
- 「Reproducibility」セクションに再現手順を明記可能
- 再現スクリプトをSupplementary Materialとして公開可能
- 異なる環境でも同じ結果が出ることの検証が容易

### 2. 開発効率
- バグ発生時のデバッグが容易（過去の任意の実験に復元可能）
- アブレーション実験の基線（baseline）復元が簡単

### 3. 制約
- DVC remote storage が設定されている必要がある（`dvc pull` 用）
- Gitリモートが設定されている必要がある（`git checkout` 用）
- 異なるCUDAバージョンでは完全な再現が保証されない（L4の制約）

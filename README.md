# Novel LLM Training System

このシステムは、特定の小説データセットのみを学習させることで、個人の好みを完璧に反映した「唯一無二の小説生成特化LLM」を構築するための学習パイプラインです。最新のスケーリング則（Step Law）に基づいた効率的なパラメータ最適化を組み込んでいます。

## 概要

本プロジェクトは、汎用的な知能ではなく、特定の文学的文体と物語構造を模倣・生成する「特化型文芸AI」の構築を目的とします。

## ディレクトリ構成

- `tools/data_import/`: データソース（SQLite）から学習用JSONL形式への変換ツール。
- `src/`: モデルの学習実行用スクリプト群。
- `LLM_Hyperparameter_Optimization/`: 最新の計算理論に基づく学習パラメータ（学習率、バッチサイズ）の自動算出ロジック。
- `docs/`: 概念的要件定義書など、プロジェクトの設計思想を記述。
- `data/`: 学習用データセット（`dataset.jsonl`）。
- `models/`: 学習済みモデルの保存先。
- `logs/`: 学習ログ。

## 使い方

### 1. データセットの準備と変換
データベースファイル (`novels.db`) を `tools/data_import/` に配置し、学習用データセットへ変換します。

```bash
python tools/data_import/export_to_training.py
```
これにより、`data/dataset.jsonl` が生成されます。

### 2. 学習パラメータの最適化
データ量とモデルサイズに基づき、最適な学習パラメータを算出します。

```python
# python等で以下のロジックを実行
from LLM_Hyperparameter_Optimization.src.step_law import compute_hpo_for_target
hpo = compute_hpo_for_target(n_params=100_000_000, n_tokens=5_000_000, seq_len=512)
print(hpo)
```

### 3. 学習の実行
算出したパラメータを `config/config.yaml` に適用し、学習を実行します。

```bash
# 学習実行
.venv\Scripts\python.exe src\train_model.py config\config.yaml
```

## 注意事項

- 本システムは文体適応に特化しています。汎用的な対話機能は実装していません。
- 学習データの著作権、投稿先の利用規約には十分注意し、最終的な出力結果は必ず人間が確認してください。

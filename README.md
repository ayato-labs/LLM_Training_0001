# Novel LLM Training System

> 論文が書けるレベルのトレーサビリティを備えた、小説特化型LLM事前学習パイプライン

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## 概要

本プロジェクトは、**特定の小特定の小説コーパスのみを学習**し、文体・会話率・感情トーンをメタデータとして制御できるLLMを構築する。最新のスケーリング則（Step Law, arXiv:2503.04715）に基づくハイパーパラメータ自動算出と、**完全な実験再現性**を保証するトレーサビリティスタックを搭載している。

### 特徴

- **条件付き事前学習**: メタデータプレフィックス（会話率・感情・ジャンル）で文体を制御
- **Step Law HPO**: 代理モデル探索から本番モデルへの最適学習率外挿
- **Muon/AdamW分離最適化**: 2DパラメータにMuon、1DにAdamWを適用
- **論文レベルトレーサビリティ**: DVC/Hydra/MLflow/シード固定/環境記録を統合
- **自動再現スクリプト生成**: MLflow run IDから再現シェルスクリプトを生成

## クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/your-username/Novel_LLM.git
cd Novel_LLM/LLM_Training

# 仮想環境作成
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 依存関係インストール
pip install -r requirements.txt
# または
pip install -e .
```

### 2. データ準備

```bash
# DVCからデータを復元
dvc pull
# または手動で data/ ディレクトリに配置
```

### 3. 学習実行

```bash
# Hydraモード（推奨）
python main.py --config-name=config

# 設定オーバーライド
python main.py --config-name=config seed=123 training.seq_len=1024

# 複数シード実行
python main.py --config-name=config -m seed=42,123,456

# レジューム
python main.py --resume
```

### 4. 評価・比較

```bash
# 推論テスト実行
python src/eval_inference/evaluate_model.py

# MLflowラン比較
python -m src.evaluation.compare_runs --top 20

# 再現スクリプト生成
python -m src.evaluation.reproduce --run-id <run_id> --output scripts/reproduce.sh
```

## ディレクトリ構成

```
LLM_Training/
├── configs/                    # Hydra設定ファイル
│   ├── config.yaml            # メイン設定
│   ├── multi_seed.yaml        # 複数シード実行用
│   ├── scaling_3b.yaml        # 3Bモデル用（RTX 5090）
│   └── scaling_7b.yaml        # 7Bモデル用（RTX 5090×2）
├── src/
│   ├── training/
│   │   └── train_model.py     # 学習コアロジック
│   ├── eval_inference/
│   │   ├── evaluate_model.py  # 推論テスト（MLflow記録対応）
│   │   └── inference.py       # 推論API
│   ├── evaluation/
│   │   ├── statistics.py      # 統計分析（t検定/Cohen's d）
│   │   ├── compare_runs.py    # MLflowラン比較CLI
│   │   ├── report_generator.py # レポート自動生成
│   │   └── reproduce.py       # 再現スクリプト生成
│   ├── preprocessing/
│   │   └── exporter.py        # SQLite→JSONL変換
│   └── utils/
│       ├── set_seed.py        # 乱数シード固定
│       ├── env_snapshot.py    # 環境スナップショット
│       └── drive_uploader.py  # Google Driveバックアップデーモン
├── docs/
│   └── ADR/                   # Architectural Decision Records
│       ├── ADR-013〜022.md    # 技術的意思決定記録
├── data/                       # 学習データ（DVC管理）
├── models/                     # 学習済みモデル
├── logs/                       # 学習ログ・評価レポート
├── mlruns/                     # MLflow実験データ
└── main.py                     # エントリーポイント
```

## トレーサビリティ

本プロジェクトは以下のトレーサビリティ機能を搭載している：

| 機能 | ツール | ADR |
|------|--------|-----|
| データ版管理 | DVC (SHA256) | ADR-014 |
| 設定管理 | Hydra + OmegaConf | ADR-015 |
| 乱数シード固定 | `set_seed()` | ADR-016 |
| 環境記録 | `env_snapshot.py` | ADR-017 |
| 実験追跡 | MLflow | ADR-018 |
| ストレージ最適化 | `drive_uploader.py` | ADR-019 |
| 評価プロトコル | 統計分析 + レポート生成 | ADR-020 |
| 再現性保証 | `reproduce.py` | ADR-021 |
| 推論出力記録 | JSON + MLflow artifact | ADR-022 |

## スケーリング

| モデルサイズ | 必要VRAM | 推奨GPU | 設定ファイル |
|-------------|---------|---------|-------------|
| 150M | 4GB | RTX 3050 | `configs/config.yaml` |
| 3B | 24GB | RTX 5090 | `configs/scaling_3b.yaml` |
| 7B | 48GB | RTX 5090×2 | `configs/scaling_7b.yaml` |

```bash
# 3Bモデルで学習
python main.py --config-name=scaling_3b

# 7Bモデルで学習
python main.py --config-name=scaling_7b
```

## ライセンス

MIT License

## 引用

```bibtex
@software{novel_llm,
  title={Novel LLM Training System},
  year={2026},
  url={https://github.com/your-username/Novel_LLM}
}
```

## 謝辞

- [Step Law (arXiv:2503.04715)](https://arxiv.org/abs/2503.04715) - スケーリング則
- [Muon Optimizer](https://github.com/KellerJordan/Muon) - 2D最適化
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Llama実装
- [Hydra](https://hydra.cc/) - 設定管理
- [DVC](https://dvc.org/) - データ版管理
- [MLflow](https://mlflow.org/) - 実験追跡

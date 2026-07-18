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

本プロジェクトは `uv` を使用して環境管理を行います。Windows ネイティブ（検証用）および WSL2/Linux（本番推奨）の双方に対応したランチャースクリプトを用意しています。

### 1. 環境構築 (Setup)

依存関係（CUDA対応PyTorch、liger-kernel、bitsandbytesなど）を一括インストールします。

* **Windows**:
  ```cmd
  setup.bat
  ```
* **WSL2 / Linux**:
  ```bash
  # uv のインストール（未導入の場合のみ）
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env

  # 依存関係の同期（仮想環境の自動構築）
  uv sync
  ```

---

### 2. 学習実行 (Training)

150Mモデルの事前学習を実行します。デフォルト設定および HPO 探索済みの最適パラメータが適用されます。

* **Windows**:
  ```cmd
  # 通常起動
  run_train.bat

  # 学習再開 (最新のチェックポイントからレジューム)
  run_train.bat --resume

  # デバッグ用軽量実行 (100ステップ限定)
  run_train.bat --max-steps 100
  ```
* **WSL2 / Linux**:
  ```bash
  # 通常起動
  uv run python -m src.training.main

  # 学習再開 (最新のチェックポイントからレジューム)
  uv run python -m src.training.main resume_from_checkpoint=true

  # デバッグ用軽量実行 (100ステップ限定)
  uv run python -m src.training.main training.max_steps=100
  ```

---

### 3. ハイパーパラメータ探索 (HPO)

代理（プロキシ）モデルを用いて最適な学習率やパラメータを探索し、本番モデルへ Step Law に基づいて自動スケーリング転移します。

* **Windows**:
  ```cmd
  run_hpo.bat 150M data/dataset.jsonl configs/hparams_150M.yaml 150 4 1024 --sync-config
  ```
* **WSL2 / Linux**:
  ```bash
  uv run python -m scripts.find_hparams \
    --proxy-model-size 150M \
    --target-model-size 150M \
    --data-path data/dataset.jsonl \
    --output configs/hparams_150M.yaml \
    --n-trials 150 \
    --seq-len 1024 \
    --sync-config
  ```

---

### 4. TensorBoardの起動 (Monitoring)

学習の進捗ロスやハイパーパラメータメトリクスをブラウザで可視化します。

* **Windows**:
  ```cmd
  launch_tensorboard.bat
  ```
* **WSL2 / Linux**:
  ```bash
  uv run tensorboard --logdir=models/output --port=6006
  ```
  起動後、ブラウザで [http://localhost:6006](http://localhost:6006) にアクセスします。

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
│       └── env_snapshot.py    # 環境スナップショット
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

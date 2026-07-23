# Novel LLM Training System

> 単一GPU（Single GPU）環境下での限界突破効率化と論文レベルのトレーサビリティを備えた、超高効率LLM事前学習パイプライン

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## 概要

本プロジェクトは、**単一GPU（Single GPU / エントリークラスGPU）環境下で最大効率のLLMフルスクラッチ事前学習**を実現するためのパイプラインである。最新のスケーリング則（Step Law, arXiv:2503.04715 / チンチラの法則）に基づくハイパーパラメータ・アーキテクチャ自動算定、選択的 Attention Checkpointing、Muon/AdamW 分離最適化、および**完全な実験再現性**を保証するトレーサビリティスタックを搭載している。

### 特徴

- **条件付き事前学習**: メタデータプレフィックス（会話率・感情・ジャンル）で文体を制御
- **Step Law HPO**: 代理モデル探索から本番モデルへの最適学習率外挿
- **Universal Chinchilla Calculator**: チンチラの法則と GPU プロファイリングによる目標時間内での最適構成自律算定
- **選択的 Attention Checkpointing**: SwiGLU (MLP) の重い再計算を回避し、Attention のみ再計算することで高速化と VRAM 節約を両立
- **事後長文拡張 (Long-Context Extension)**: YaRN / Dynamic NTK による `seq_len: 4096~8192` への追加事前学習
- **Muon/AdamW分離最適化**: 2DパラメータにMuon、1DにAdamWを適用
- **軽量トレーサビリティ**: TensorBoard/Hydra/シード固定/環境スナップショットによる完全追跡

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

### 4. チンチラの法則に基づく最適モデル規模の自動算定 (Chinchilla Calculator)

目標学習時間（時間 / 日数）と GPU 性能プロファイリングに基づき、最も Loss が低くなる最適なモデルパラメータ数と構造を自動算定します。

* **Windows**:
  ```cmd
  uv run python -m src.chinchilla.main hours=48
  ```
* **WSL2 / Linux**:
  ```bash
  uv run python -m src.chinchilla.main days=3
  ```

---

### 5. 事後長文拡張 (Long-Context Extension)

事前学習（`seq_len=1024`）完了後、モデル重みを維持しながら RoPE Scaling (YaRN/Dynamic NTK) を注入し、コンテキストウィンドウを `seq_len=4096~8192` へ高速追加学習（Continued Pretraining）します。

* **Windows**:
  ```cmd
  uv run python -m src.context_extension.main target_seq_len=4096
  ```
* **WSL2 / Linux**:
  ```bash
  uv run python -m src.context_extension.main base_model_path=models/output/checkpoint-latest target_seq_len=4096
  ```

---

### 6. TensorBoardの起動 (Monitoring)

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

### 7. 評価・推論テスト

```bash
# 推論・テキスト生成テストの実行
python -m src.eval_inference.evaluate_model
```

## ディレクトリ構成

```
LLM_Training/
├── configs/                    # Hydra設定ファイル
│   ├── config.yaml            # メイン設定
│   ├── extension_config.yaml  # 事後長文拡張用設定 (NEW)
│   ├── multi_seed.yaml        # 複数シード実行用
│   ├── scaling_3b.yaml        # 3Bモデル用（RTX 5090）
│   └── scaling_7b.yaml        # 7Bモデル用（RTX 5090×2）
├── src/
│   ├── training/              # 事前学習エンジン & コールバック
│   ├── chinchilla/            # チンチラ法則自動算定モジュール (NEW)
│   │   ├── calculator.py      # プロファイリング・最適逆算コア
│   │   └── main.py            # CLI エントリーポイント
│   ├── context_extension/     # 事後長文拡張モジュール (NEW)
│   │   ├── extension_engine.py# RoPE Scaling 注入・Continued Pretraining
│   │   └── main.py            # CLI エントリーポイント
│   ├── hpo/                   # Step Law HPO モジュール
│   ├── eval_inference/        # 推論テスト
│   ├── evaluation/            # レポート生成
│   └── preprocessing/         # データ前処理
├── docs/                      # ドキュメント & 工夫記録
├── data/                      # 学習データ（DVC管理）
├── models/                    # 学習済みモデル
└── logs/                      # 学習ログ
```

## トレーサビリティ

本プロジェクトは以下の軽量トレーサビリティ機能を搭載している：

| 機能 | ツール | 設計仕様 |
|------|--------|-----|
| データ版管理 | DVC (SHA256) | ADR-014 |
| 設定管理 | Hydra + OmegaConf | ADR-015 |
| 乱数シード固定 | `set_seed()` | ADR-016 |
| 環境記録 | `env_snapshot.py` | ADR-017 |
| 実験追跡 | TensorBoard | ADR-016 (軽量化) |
| 損失監視 | `PeriodicEvaluationCallback` | 自律発散防止 |

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
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 実験追跡 (MLflow代替)
- [DVC](https://dvc.org/) - データ版管理

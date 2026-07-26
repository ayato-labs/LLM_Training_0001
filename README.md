# Novel LLM Training System

> 単一GPU環境向けLLMフルスクラッチ事前学習フレームワーク。
> スケーリング則・自動VRAM実測・Step Law HPO による効率的な学習を実現する。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## 概要

本プロジェクトは**単一GPU環境でLLMをフルスクラッチ事前学習するためのフレームワーク**である。以下の4つのサブシステムで構成される：

| サブシステム | モジュール | 役割 |
|-------------|-----------|------|
| Chinchilla | `src.chinchilla.main` | VRAM実測 + 最適モデル規模・バッチサイズ自動算定 |
| HPO | `src.hpo.main` | Step Law + Optuna によるハイパーパラメータ探索 |
| Training | `src.training.main` | Hydra設定駆動の事前学習エンジン |
| Context Extension | `src.rope.main` | 事後コンテキストウィンドウ拡張 (RoPE) |

### 特徴

- **Chinchilla Scaling Laws**: 目標時間・VRAM・データセットから最適パラメータ数を自律算定
- **VRAM実測キャリブレーション**: GPU上で実測し推定精度を向上、OOM/TDR防止
- **Step Law HPO**: 代理モデル探索 → 本番モデルへの学習率外挿 (arXiv:2503.04715)
- **自動バッチ分割**: VRAM上限に応じて per_device / grad_accum を最適配分
- **選択的 Attention Checkpointing**: VRAM効率と速度のバランス
- **Muon/AdamW分離最適化**: 2D=Muon, 1D=AdamW
- **軽量トレーサビリティ**: Hydra + TensorBoard + DVC + シード固定

## クイックスタート

### 1. 環境構築

```bash
# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 依存関係の同期
uv sync
```

### 2. 学習の流れ（標準的な4ステップ）

```bash
# Step 1: Chinchilla — モデル規模・バッチサイズを自動算定
uv run python -m src.chinchilla.main hours=48 --apply

# Step 2: HPO — 最適ハイパーパラメータを探索
uv run python -m src.hpo.main

# Step 3: フルスクラッチ学習
uv run python -m src.training.main

# Step 4: コンテキスト拡張（任意）
uv run python -m src.rope.main checkpoint=models/output/checkpoint-latest new_max=8192
```

各ステップは独立して実行可能。Chinchilla・HPO は初回のみ、もしくは設定変更時に再実行する。

## コマンドリファレンス

### Chinchilla

GPU実測キャリブレーション → チンチラ最適モデル規模算定 → バッチ分割。

```bash
# 48時間学習に最適な構成を算定し適用
uv run python -m src.chinchilla.main hours=48 --apply

# 3日間、VRAM 4GB制限で
uv run python -m src.chinchilla.main days=3 vram=4.0 --apply

# コンテキスト長トレードオフ比較
uv run python -m src.chinchilla.main hours=24 seq_len=2048
```

`--apply` で `configs/chinchilla_config.yaml` に反映される。

### HPO (Hyperparameter Optimization)

Step Law 事前分布 + Optuna TPE による5次元探索。試行回数は自動算出。

```bash
# デフォルト実行（自動試行回数）
uv run python -m src.hpo.main

# 試行回数を明示
uv run python -m src.hpo.main n_trials=100

# シーケンス長指定
uv run python -m src.hpo.main seq_len=2048
```

結果は `configs/hpo_config.yaml` に保存される。

### Training

Hydra 設定駆動の事前学習。

```bash
# 通常起動
uv run python -m src.training.main

# 学習再開
uv run python -m src.training.main resume_from_checkpoint=true

# デバッグ用（100ステップ）
uv run python -m src.training.main training.max_steps=100
```

### Context Extension

学習済みモデルのコンテキストウィンドウを事後拡張する。

```bash
uv run python -m src.rope.main checkpoint=models/output/checkpoint-latest new_max=8192 method=ntk
```

## ディレクトリ構成

```
LLM_Training/
├── configs/                    # Hydra設定ファイル
│   ├── config.yaml            # メイン設定
│   ├── chinchilla_config.yaml # Chinchilla算定結果
│   └── hpo_config.yaml        # HPO探索結果
├── src/
│   ├── chinchilla/            # チンチラ法則算定
│   │   ├── calculator.py      # プロファイリング・最適逆算コア
│   │   └── main.py            # CLIエントリーポイント
│   ├── hpo/                   # Step Law + Optuna HPO
│   │   ├── hpo_manager.py     # Optuna目的関数・探索空間
│   │   ├── step_law.py        # Step Lawスケーリング則
│   │   └── main.py            # CLIエントリーポイント
│   ├── training/              # 事前学習エンジン
│   │   ├── train_engine.py    # 学習ループ
│   │   ├── config.py          # Hydra設定ロード
│   │   ├── model_utils.py     # モデル構築・バッチ分割
│   │   └── main.py            # CLIエントリーポイント
│   ├── rope/                  # コンテキスト拡張
│   │   └── main.py            # CLIエントリーポイント
│   └── common/                # 共通ユーティリティ
│       └── vram_estimator.py  # VRAM推定・実測キャリブレーション
├── scripts/                   # 補助スクリプト
├── tests/                     # 単体テスト
├── docs/                      # ADR・設計ドキュメント
├── data/                      # 学習データ（DVC管理）
└── models/                    # チェックポイント・出力
```

## トレーサビリティ

| 機能 | ツール |
|------|--------|
| データ版管理 | DVC (SHA256) |
| 設定管理 | Hydra + OmegaConf |
| 乱数シード固定 | `set_seed()` |
| 環境記録 | `env_snapshot.py` |
| 実験追跡 | TensorBoard |
| 損失監視 | `PeriodicEvaluationCallback` |

## スケーリング

| モデルサイズ | 必要VRAM | 推奨GPU |
|-------------|---------|---------|
| 150M | 4GB | RTX 3050 |
| 3B | 24GB | RTX 5090 |
| 7B | 48GB | RTX 5090×2 |

```bash
# 3Bモデル
uv run python -m src.training.main --config-name=scaling_3b

# 7Bモデル
uv run python -m src.training.main --config-name=scaling_7b
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

- [Step Law (arXiv:2503.04715)](https://arxiv.org/abs/2503.04715)
- [Muon Optimizer](https://github.com/KellerJordan/Muon)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Hydra](https://hydra.cc/)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

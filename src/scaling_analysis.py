import json
from pathlib import Path

import yaml


def calculate_model_params(config):
    """
    Llama系モデルのパラメータ数を概算 (おおよその式: 12 * layers * hidden^2)
    より正確にするにはembeddingやnorm層を含める必要がある
    """
    layers = config["model"]["num_hidden_layers"]
    hidden = config["model"]["hidden_size"]
    # 概算式: 簡易的に構成要素から算出
    params = 12 * layers * (hidden**2)
    return params


def get_experiment_data(model_dir):
    """実験ディレクトリから設定と最終Lossを抽出"""
    model_dir = Path(model_dir)
    if not (model_dir / "config.yaml").exists() or not (model_dir / "trainer_state.json").exists():
        return None

    with open(model_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(model_dir / "trainer_state.json") as f:
        trainer_state = json.load(f)

    loss = trainer_state.get("best_loss") or trainer_state.get("train_loss_history", [{}])[-1].get(
        "loss"
    )

    return {"params": calculate_model_params(config), "loss": loss}


def estimate_optimal_compute(target_loss, alpha=0.34, beta=0.28):
    """
    Kaplan et al. のスケール則モデルに基づく目標Lossへの必要Compute(FLOPs)概算
    Loss ~ N^-alpha * D^-beta
    疎結合にするため、シンプルにNとDの比率で計算
    """
    # 簡易モデル: 訓練計算量 C = 6 * N * D
    # 専門家として、この関数は後ほど実験データでパラメータをフィッティング可能にする
    return f"Estimate for target loss {target_loss} requires fitting on at least 3+ experiments."

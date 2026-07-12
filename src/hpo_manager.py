"""HPO Library: Optuna objective & search space definition.
main.py からは import されない。scripts/find_hparams.py からのみ使用。
"""

import optuna
import torch
from src.model_utils import estimate_model_size
from src.logger import logger
# Note: Since train_model.py still exists, we reuse train from there.
# If this is not correct, we need to adapt it. Assuming train() can accept config dict.
from src.train_model import train as proxy_train


def create_search_space(step_law_hpo: dict, vram_gb: float) -> dict:
    """Step Law結果を中心とした探索空間定義"""
    lr_center = step_law_hpo["max_lr_2d"]
    return {
        "max_lr_2d": (lr_center * 0.5, lr_center * 2.0, "log"),
        "max_lr_1d": (lr_center * 5, lr_center * 20, "log"),
        "batch_size_seqs": [8, 16, 32],
        "warmup_ratio": (0.01, 0.1),
        "weight_decay": (0.01, 0.3),
        "beta2": (0.9, 0.99),
        "grad_clip": (0.5, 2.0),
    }


def objective(trial: optuna.Trial, arch: dict, data_path: str, seq_len: int, vram_gb: float) -> float:
    """Proxy training objective (short run, small data fraction)"""
    
    # Sample hyperparams
    hpo = {}
    space = create_search_space({}, vram_gb)  # baseline
    for param, spec in space.items():
        if isinstance(spec, list):
            hpo[param] = trial.suggest_categorical(param, spec)
        elif spec[2] == "log":
            hpo[param] = trial.suggest_float(param, spec[0], spec[1], log=True)
        else:
            hpo[param] = trial.suggest_float(param, spec[0], spec[1])
    
    # Build minimal config for proxy training
    config = {
        **arch,
        "hpo": hpo,
        "data_path": data_path,
        "seq_len": seq_len,
        "max_steps": 50,  # Proxy steps
        "data_fraction": 0.005,
        "precision": "bf16",
        "vram_limit_gb": vram_gb,
        "seed": 42,
    }
    
    try:
        # Quick proxy training (reuse train_model logic but minimal)
        loss = proxy_train(config)
        return loss if not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss))) else 1e9
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 1e9

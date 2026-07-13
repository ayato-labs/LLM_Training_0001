"""HPO Library: Optuna objective & search space definition.
main.py からは import されない。scripts/find_hparams.py からのみ使用。
"""

import optuna
import torch

from src.logger import logger

# Note: Since train_model.py still exists, we reuse train from there.
# If this is not correct, we need to adapt it. Assuming train() can accept config dict.
from src.train_model import train as proxy_train


from transformers import TrainerCallback

class OptunaPruningCallback(TrainerCallback):
    """Callback to prune trials in Optuna based on intermediate loss value."""
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            self.trial.report(loss, step=state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def create_search_space(step_law_hpo: dict, vram_gb: float) -> dict:
    """Step Law結果を中心とした探索空間定義 (4次元)"""
    lr_center = step_law_hpo["max_lr_2d"]
    return {
        "max_lr_2d": (lr_center * 0.5, lr_center * 2.0, "log"),
        "max_lr_1d": (lr_center * 5, lr_center * 20, "log"),
        "batch_size_seqs": [8, 16, 32],
        "weight_decay": (0.01, 0.3, ""),
    }


def objective(
    trial: optuna.Trial,
    arch: dict,
    tokenized_dataset,
    seq_len: int,
    vram_gb: float,
    step_law_hpo: dict,
) -> float:
    """Proxy training objective (short run, small data fraction, 50 steps)"""

    # Sample hyperparams (4D)
    hpo = {}
    space = create_search_space(step_law_hpo, vram_gb)
    for param, spec in space.items():
        if isinstance(spec, list):
            hpo[param] = trial.suggest_categorical(param, spec)
        elif spec[2] == "log":
            hpo[param] = trial.suggest_float(param, spec[0], spec[1], log=True)
        else:
            hpo[param] = trial.suggest_float(param, spec[0], spec[1])

    # Fixed values for fixed dimensions
    hpo["warmup_ratio"] = 0.03
    hpo["beta2"] = 0.95
    hpo["grad_clip"] = 1.0

    # Build config matching normalize_config expectations
    config = {
        "model_params": {
            "n_params": arch["n_params"],
            "hidden_size": arch["hidden"],
            "num_hidden_layers": arch["layers"],
            "num_attention_heads": arch["heads"],
            "num_key_value_heads": arch["kv_heads"],
            "intermediate_size": arch["ffn"],
            "rope_theta": 10000.0,
            "vocab_size": 64000,
        },
        "hpo": hpo,
        "seq_len": seq_len,
        "max_steps": 50,  # 50 steps proxy run
        "data_fraction": 0.001,  # Tiny fraction for speed
        "precision": "bf16",
        "vram_limit_gb": vram_gb,
        "seed": 42,
    }

    try:
        # Quick proxy training with pruning callback
        pruning_callback = OptunaPruningCallback(trial)
        loss = proxy_train(
            config,
            tokenized_datasets=tokenized_dataset,
            extra_callbacks=[pruning_callback],
        )
        return (
            loss
            if not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)))
            else 1e9
        )
    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        raise
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 1e9
    finally:
        # HPO試行終了時にストレージを圧迫する中間生成物・モデルを完全に消去
        import shutil
        from pathlib import Path

        output_dir = Path("models/output")
        if output_dir.exists():
            try:
                logger.info(f"Cleaning up HPO trial output files in {output_dir}")
                # ディレクトリの中身を全削除（次の試行に影響しないようにする）
                for item in output_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up HPO output: {e}")

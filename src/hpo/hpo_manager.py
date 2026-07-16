"""HPO Library: Optuna objective & search space definition.
main.py からは import されない。scripts/find_hparams.py からのみ使用。
"""

import optuna
import torch

from src.common.logger import logger

# Use the unified training engine
from src.training.train_engine import train as proxy_train


from transformers import TrainerCallback

class OptunaPruningCallback(TrainerCallback):
    """Optunaの途中損失値に基づいてトライアルをプルーンするコールバック。"""
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            self.trial.report(loss, step=state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def create_search_space(step_law_hpo: dict, vram_gb: float, n_params: int = 150_000_000) -> dict:
    """Step Law結果を中心とした探索空間定義 (5次元、モデルサイズ依存の動的範囲)

    小モデルほどStep Law誤差が大きく→範囲を広めに
    大モデルほどStep Law信頼度が高く→範囲を絞り込む
    """
    lr_2d_center = step_law_hpo["max_lr_2d"]
    lr_1d_center = step_law_hpo["max_lr_1d"]

    # モデルサイズに応じたLR探索範囲の倍率 (小→大で狭くなる)
    if n_params < 100_000_000:          # <100M
        lr_low, lr_high = 0.6, 1.7     # ±40~70%
        wd_low, wd_high = 0.03, 0.25
    elif n_params < 500_000_000:        # 100M-500M
        lr_low, lr_high = 0.7, 1.5     # ±30~50%
        wd_low, wd_high = 0.04, 0.22
    elif n_params < 2_000_000_000:      # 500M-2B
        lr_low, lr_high = 0.75, 1.4    # ±25~40%
        wd_low, wd_high = 0.05, 0.20
    else:                               # ≥2B
        lr_low, lr_high = 0.8, 1.3     # ±20~30%
        wd_low, wd_high = 0.06, 0.18

    return {
        "max_lr_2d": (lr_2d_center * lr_low, lr_2d_center * lr_high, "log"),
        "max_lr_1d": (lr_1d_center * lr_low, lr_1d_center * lr_high, "log"),
        "batch_size_seqs": [8, 16, 32],
        "weight_decay": (wd_low, wd_high, ""),
        "warmup_ratio": (0.01, 0.1, ""),  # 1%〜10% の範囲で探索 (据え置き)
    }


def objective(
    trial: optuna.Trial,
    arch: dict,
    tokenized_dataset,
    seq_len: int,
    vram_gb: float,
    step_law_hpo: dict,
) -> float:
    """プロキシ学習目的関数（短時間実行、小データ割合、50ステップ）。"""

    # Print visible progress banner
    total_trials = trial.study.user_attrs.get("n_trials", "?")
    logger.info("=" * 60)
    logger.info(f"  [HPO Progress] Trial {trial.number + 1} / {total_trials} started...")
    logger.info("=" * 60)

    # Sample hyperparams (5D)
    hpo = {}
    space = create_search_space(step_law_hpo, vram_gb, n_params=arch["n_params"])
    for param, spec in space.items():
        if isinstance(spec, list):
            hpo[param] = trial.suggest_categorical(param, spec)
        elif spec[2] == "log":
            hpo[param] = trial.suggest_float(param, spec[0], spec[1], log=True)
        else:
            hpo[param] = trial.suggest_float(param, spec[0], spec[1])

    # Fixed values for fixed dimensions
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
            "rope_theta": arch.get("rope_theta", 500000.0),
            "vocab_size": 64000,
        },
        "hpo": hpo,
        "seq_len": seq_len,
        "max_steps": 50,  # 50 steps proxy run
        "data_fraction": 0.001,  # Tiny fraction for speed
        "precision": "bf16",
        "vram_limit_gb": vram_gb,
        "seed": 42,
        "tokenizer_path": "data/tokenizer.json",
        "output_dir": "models/output",
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
        import gc
        import time
        import stat
        from pathlib import Path

        def _handle_remove_readonly(func, path, exc):
            """Windows読み取り専用ファイル対応"""
            import os
            os.chmod(path, stat.S_IWRITE)
            func(path)

        def _cleanup_with_retry(output_path: Path, max_retries: int = 3):
            """リトライ付きクリーンアップ (Windowsファイルロック対応)"""
            for attempt in range(max_retries):
                try:
                    gc.collect()
                    time.sleep(0.5 * (attempt + 1))  # ファイルロック解放待ち
                    for item in output_path.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item, onerror=_handle_remove_readonly)
                            else:
                                item.unlink()
                        except Exception:
                            pass  # 個別ファイル削除失敗は無視
                    return  # 成功
                except Exception:
                    if attempt < max_retries - 1:
                        continue
                    # 最大リトライ回数到達後は無視

        # Windowsのファイルロックを解放するために明示的にガベージコレクションを実行
        gc.collect()

        output_dir = Path("models/output")
        if output_dir.exists():
            logger.info(f"Cleaning up HPO trial output files in {output_dir}")
            _cleanup_with_retry(output_dir)

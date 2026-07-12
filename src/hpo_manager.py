import gc
import time

import torch

from src.config import detect_vram
from src.logger import get_logger, logger
from src.model_utils import estimate_config_from_params
from src.trainer import train_model


def _cleanup_vram():
    """VRAMを確実に解放"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _get_batch_size_candidates(vram_gb: float) -> list[int]:
    """VRAM容量に基づいてバッチサイズ候補を動的に決定"""
    if vram_gb >= 24:
        return [4, 8, 16, 32]
    elif vram_gb >= 16:
        return [2, 4, 8, 16]
    elif vram_gb >= 12:
        return [2, 4, 8]
    elif vram_gb >= 8:
        return [1, 2, 4, 8]
    else:  # 4GB or less
        return [1, 2, 4]


def compute_scaling_priors(target_params: int, n_tokens: int, vram_gb: float = 4.0) -> dict:
    """
    スケーリング則に基づくハイパーパラメータの事前分布を計算。

    参考: Chinchilla (Hoffmann et al., 2022), Kaplan et al. (2020),
    MuTransfer (Yang et al., 2021), μP (Yang et al., 2022)
    """
    # === Learning Rate Scaling (Chinchilla / Kaplan) ===
    # 最適 LR は model_size^(-0.5) 〜 model_size^(-0.3) でスケール
    # 基準: 150M で lr ≈ 3e-4 (max_lr_2d)
    base_lr_2d = 3e-4 * (150_000_000 / target_params) ** 0.35
    base_lr_1d = 3e-3 * (150_000_000 / target_params) ** 0.35

    # === Batch Size Scaling (Linear Scaling Rule) ===
    # 最適 batch_size ∝ model_size^0.5 (Chinchilla)
    # ただし VRAM制約で上限あり
    optimal_batch = int(16 * (target_params / 150_000_000) ** 0.5)
    optimal_batch = min(max(optimal_batch, 4), 64)

    # VRAMに基づくバッチサイズ候補の決定
    batch_size_candidates = _get_batch_size_candidates(vram_gb)
    batch_size_prior = min(optimal_batch, max(batch_size_candidates))

    # === Warmup Steps ===
    # 最適 warmup_steps ≈ 0.01 * total_steps 〜 0.1 * total_steps
    warmup_ratio = 0.03

    # === Weight Decay ===
    # 最適 weight decay は model_size にほぼ依存しないが、大きいモデルでは少し小さく
    weight_decay = 0.1 * (150_000_000 / target_params) ** 0.1

    # === Beta2 (Adam) ===
    # 大きいモデルでは 0.95-0.99 が安定
    beta2 = 0.95 if target_params > 100_000_000 else 0.99

    # === Gradient Clipping ===
    # μP 理論では clip=1.0 がスケール不変で最適
    grad_clip = 1.0

    return {
        # LR (log-uniform prior around theoretical optimum)
        "lr_2d_center": base_lr_2d,
        "lr_1d_center": base_lr_1d,
        "lr_2d_range": (base_lr_2d * 0.1, base_lr_2d * 10.0),
        "lr_1d_range": (base_lr_1d * 0.1, base_lr_1d * 10.0),

        # Batch size (discrete) - VRAMベースで動的決定
        "batch_size_candidates": batch_size_candidates,
        "batch_size_prior": batch_size_prior,

        # Warmup
        "warmup_ratio_range": (0.005, 0.2),
        "warmup_ratio_prior": warmup_ratio,

        # Weight decay
        "weight_decay_range": (0.001, 0.5),
        "weight_decay_prior": weight_decay,

        # Beta2
        "beta2_range": (0.95, 0.999),
        "beta2_prior": beta2,

        # Gradient clipping
        "grad_clip_range": (0.25, 4.0),
        "grad_clip_prior": grad_clip,

        # Other
        "seq_len": 1024,
        "model_params": target_params,
        "n_tokens": n_tokens,
    }


def objective(trial, config: dict):
    trial_start_time = time.time()
    trial_logger = get_logger(f"hpo.trial_{trial.number}")

    # VRAMクリーンアップ: 前のTrialのメモリを確実に解放
    _cleanup_vram()

    try:
        # VRAM取得（configから、なければ自動検出）
        vram_gb = config.get("vram_limit_gb", detect_vram())

        # プロキシモデル構成を動的に計算（本番と同じアーキテクチャでサイズのみ縮小）
        proxy_config = estimate_config_from_params(
            max(int(config["model_params"]["n_params"] * 0.05), 5_000_000)
        )

        # スケーリング則ベースの事前分布取得（本番サイズで計算）
        priors = compute_scaling_priors(config["model_params"]["n_params"], 1_400_000_000, vram_gb)

        # === 探索空間: スケーリング則で絞った範囲 ===

        # 1. Learning Rate (2D/1D 分離) - μP理論に基づく log-uniform
        lr_2d = trial.suggest_float(
            "lr_2d",
            3e-4 * 0.1, 3e-4 * 10.0,
            log=True
        )
        lr_1d = trial.suggest_float(
            "lr_1d",
            3e-3 * 0.1, 3e-3 * 10.0,
            log=True
        )

        # 2. Weight Decay (log-uniform)
        weight_decay = trial.suggest_float(
            "weight_decay",
            0.001, 0.5,
            log=True
        )

        # 3. Beta2 (Adam) - categorical で離散的に
        beta2 = trial.suggest_categorical(
            "beta2", [0.95, 0.98, 0.99, 0.999]
        )

        # 4. Warmup Ratio (log-uniform)
        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            0.005, 0.2,
            log=True
        )

        # 5. Gradient Clipping
        grad_clip = trial.suggest_float("grad_clip", 0.25, 4.0)

        # 6. Batch Size (離散) - VRAMに基づく動的候補
        batch_size = trial.suggest_categorical(
            "batch_size",
            priors["batch_size_candidates"]
        )

        trial_logger = get_logger(f"hpo.trial_{trial.number}")

        trial_logger.info(
            f"Trial {trial.number} STARTED | "
            f"lr_2d={lr_2d:.2e}, lr_1d={lr_1d:.2e}, "
            f"wd={weight_decay:.4f}, beta2={beta2}, warmup={warmup_ratio:.3f}, "
            f"grad_clip={grad_clip:.1f}, bs={batch_size}"
        )

        # プロキシモデル設定（動的に計算）

        # プロキシモデル設定（小さなモデル）

        trial_start = time.time()
        trial_logger = get_logger(f"hpo.trial_{trial.number}")
        trial_logger.info(f"Trial {trial.number} TRAINING STARTED (max_steps=50, pilot_mode)")

        try:
            # 本番と同じ train_model を使用（モデルサイズのみ縮小）
            loss = train_model({
                "model_name": "modern_gpt",
                "model_config": {
                    "n_layer": proxy_config.n_layer,
                    "n_embd": proxy_config.n_embd,
                    "n_head": proxy_config.n_head,
                    "n_kv_head": proxy_config.n_kv_head,
                    "vocab_size": proxy_config.vocab_size,
                },
                "lr": lr_2d,
                "lr_1d": lr_1d,
                "weight_decay": weight_decay,
                "beta2": beta2,
                "warmup_ratio": warmup_ratio,
                "grad_clip": grad_clip,
                "optimizer": "normuon",
                "batch_size_seqs": batch_size,
                "max_steps": 50,
                "pilot_mode": True,
            }, trial=trial)

            trial_duration = time.time() - trial_start
            time.time() - trial_start_time

            logger.info(
                f"Trial {trial.number} COMPLETED | "
                f"loss={loss:.6f} | "
                f"trial_time={trial_duration:.1f}s | "
                f"total_time={time.time() - trial_start_time:.1f}s"
            )

            # Optuna callback for pruning info
            if trial.should_prune():
                logger.warning(f"Trial {trial.number} PRUNED by MedianPruner")

            return loss

        except Exception as e:
            trial_duration = time.time() - trial_start_time
            logger.error(
                f"Trial {trial.number} FAILED after {trial_duration:.1f}s: {e}",
                exc_info=True
            )
            raise

    except Exception as e:
        trial_duration = time.time() - trial_start_time
        logger.error(
            f"Trial {trial.number} FAILED after {trial_duration:.1f}s: {e}",
            exc_info=True
        )
        raise

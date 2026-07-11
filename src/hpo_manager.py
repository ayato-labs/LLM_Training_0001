import optuna
import math

from src.logger import logger
from src.step_law import compute_hpo_for_target
from src.trainer import train_model
from src.model_utils import estimate_config_from_params


def compute_scaling_priors(target_params: int, n_tokens: int) -> dict:
    """
    スケーリング則に基づくハイパーパラメータの事前分布を計算。
    
    参考: Chinchilla (Hoffmann et al., 2022), Kaplan et al. (2020), 
    MuTransfer (Yang et al., 2021), μP (Yang et al., 2022)
    """
    # パラメータ数とデータ量の対数
    log_params = math.log10(target_params)
    log_tokens = math.log10(n_tokens)
    
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
    
    # === Warmup Steps ===
    # 最適 warmup_steps ≈ 0.01 * total_steps 〜 0.1 * total_steps
    # total_steps ≈ n_tokens / (batch_size * seq_len)
    est_total_steps = n_tokens / (16 * 1024)  # 概算
    warmup_steps = int(est_total_steps * 0.03)
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
        "lr_2d_range": (base_lr_2d * 0.3, base_lr_2d * 3.0),
        "lr_1d_range": (base_lr_1d * 0.3, base_lr_1d * 3.0),
        
        # Batch size (discrete)
        "batch_size_candidates": [8, 16, 32, 64],
        "batch_size_prior": optimal_batch,
        
        # Warmup
        "warmup_ratio_range": (0.01, 0.1),
        "warmup_ratio_prior": warmup_ratio,
        
        # Weight decay
        "weight_decay_range": (0.01, 0.3),
        "weight_decay_prior": weight_decay,
        
        # Beta2
        "beta2_range": (0.95, 0.999),
        "beta2_prior": beta2,
        
        # Gradient clipping
        "grad_clip_range": (0.5, 2.0),
        "grad_clip_prior": grad_clip,
        
        # Other
        "seq_len": 1024,
        "model_params": target_params,
        "n_tokens": n_tokens,
    }


def objective(trial, config):
    try:
        target_params = config["model_params"].get("n_params", 150_000_000)
        n_tokens = 1_400_000_000
        
        # スケーリング則ベースの事前分布取得
        priors = compute_scaling_priors(target_params, n_tokens)
        config_model = estimate_config_from_params(target_params)
        
        # === 探索空間: スケーリング則で絞った範囲 ===
        
        # 1. Learning Rate (2D/1D 分離) - μP理論に基づく log-uniform
        lr_2d = trial.suggest_float(
            "lr_2d", 
            priors["lr_2d_range"][0], 
            priors["lr_2d_range"][1], 
            log=True
        )
        lr_1d = trial.suggest_float(
            "lr_1d", 
            priors["lr_1d_range"][0], 
            priors["lr_1d_range"][1], 
            log=True
        )
        
        # 2. Weight Decay (log-uniform)
        weight_decay = trial.suggest_float(
            "weight_decay",
            priors["weight_decay_range"][0],
            priors["weight_decay_range"][1],
            log=True
        )
        
        # 3. Beta2 (Adam) - categorical で離散的に
        beta2 = trial.suggest_categorical(
            "beta2", [0.95, 0.98, 0.99, 0.999]
        )
        
        # 3. Warmup Ratio (log-uniform)
        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            priors["warmup_ratio_range"][0],
            priors["warmup_ratio_range"][1],
            log=True
        )
        
        # 4. Gradient Clipping
        grad_clip = trial.suggest_float(
            "grad_clip",
            priors["grad_clip_range"][0],
            priors["grad_clip_range"][1],
        )
        
        # 4. Batch Size (離散) - GPUメモリ制約で上限あり
        batch_size = trial.suggest_categorical(
            "batch_size", 
            [b for b in priors["batch_size_candidates"] if b <= 16]  # VRAM 4GB制約
        )
        
        logger.debug(
            f"Trial {trial.number}: lr_2d={lr_2d:.2e}, lr_1d={lr_1d:.2e}, "
            f"wd={weight_decay:.4f}, beta2={beta2}, warmup={warmup_ratio:.3f}, "
            f"grad_clip={grad_clip:.1f}, bs={batch_size}"
        )
        
        hpo_config = {
            "model_name": "modern_gpt",
            "model_config": config_model.__dict__,
            "lr": lr_2d,  # 2D params (attention, ffn weights)
            "lr_1d": lr_1d,  # 1D params (bias, norm, embeddings)
            "weight_decay": weight_decay,
            "beta2": beta2,
            "warmup_ratio": warmup_ratio,
            "grad_clip": grad_clip,
            "optimizer": "normuon",
            "batch_size_seqs": batch_size,
        }
        
        loss = train_model(hpo_config, trial=trial)
        logger.info(f"Trial {trial.number}: loss={loss:.6f}")
        return loss
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("Starting HPO study with scaling-law priors")
    
    # TPE Sampler with multivariate=True でパラメータ間相関を学習
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=5,
        n_ei_candidates=24,
        seed=42,
    )
    
    # MedianPruner で明らかに悪い試行を早期打ち切り
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=10,
        interval_steps=2,
    )
    
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="llm_scaling_law_hpo",
    )
    
    # 設定をダミーで作成
    config = {"model_params": {"n_params": 150_000_000}}
    
    study.optimize(
        lambda trial: objective(trial, {"model_params": {"n_params": 150_000_000}}),
        n_trials=30,
        timeout=7200,  # 2時間で打ち切り
        n_jobs=1,
    )
    
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best value: {study.best_value:.6f}")
    
    # 重要度分析
    try:
        importances = optuna.importance.get_param_importances(study)
        logger.info(f"Param importances: {importances}")
    except Exception as e:
        logger.warning(f"Could not compute importances: {e}")

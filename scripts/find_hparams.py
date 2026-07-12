#!/usr/bin/env python3
"""
Offline Hyperparameter Search Script
分離された探索フェーズ。学習パイプライン(main.py)からは完全独立。

Usage:
  python -m scripts.find_hparams --model-size 150M --data-path data/dataset.jsonl --output configs/hparams_150M.yaml --n-trials 20
"""

import argparse
import yaml
from pathlib import Path
import json

import optuna
from src.step_law import compute_hpo_for_target
from src.hpo_manager import objective, create_search_space
from src.config import _detect_vram as detect_vram
from src.logger import logger, log_exceptions, log_function_call


def parse_args():
    p = argparse.ArgumentParser(description="Offline HPO for LLM Training")
    p.add_argument("--model-size", required=True, choices=["50M", "150M", "3B", "7B"],
                   help="Target model size (determines architecture)")
    p.add_argument("--data-path", required=True, help="Path to training dataset (JSONL)")
    p.add_argument("--output", required=True, help="Output YAML path (e.g., configs/hparams_150M.yaml)")
    p.add_argument("--n-trials", type=int, default=20, help="Optuna trials")
    p.add_argument("--vram-gb", type=float, help="Override VRAM detection")
    p.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    return p.parse_args()


@log_function_call(log_args=True)
def get_base_config(model_size: str) -> dict:
    """モデルサイズごとのベースアーキテクチャ設定"""
    try:
        configs = {
            "50M":  {"n_params": 50_000_000,  "hidden": 640, "layers": 10, "heads": 10, "kv_heads": 10, "ffn": 2560},
            "150M": {"n_params": 150_000_000, "hidden": 768, "layers": 12, "heads": 12, "kv_heads": 3,  "ffn": 3072},
            "3B":   {"n_params": 3_000_000_000, "hidden": 2560, "layers": 28, "heads": 20, "kv_heads": 20, "ffn": 10240},
            "7B":   {"n_params": 7_000_000_000, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 32, "ffn": 11008},
        }
        return configs[model_size]
    except KeyError:
        logger.error(f"Invalid model size: {model_size}")
        raise

@log_function_call(log_args=True)
def estimate_tokens(data_path: str) -> int:
    """データセットのトークン数概算"""
    try:
        count = 0
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                count += 1
        logger.debug(f"Estimated tokens from {data_path}: {count * 1024}")
        return count * 1024  # 1行≒1024トークン想定
    except FileNotFoundError:
        logger.exception(f"Dataset file not found: {data_path}")
        raise
    except Exception as e:
        logger.exception(f"Error estimating tokens: {e}")
        raise


@log_exceptions
def main():
    args = parse_args()
    logger.info("Starting HPO search", extra={"args": vars(args)})
    
    # 1. ベース設定構築
    arch = get_base_config(args.model_size)
    n_tokens = estimate_tokens(args.data_path)
    vram = args.vram_gb or detect_vram()
    
    logger.info(f"HPO Search initialized", extra={
        "model_size": args.model_size, 
        "tokens": n_tokens, 
        "vram": vram, 
        "trials": args.n_trials
    })
    
    # 2. Step Law で初期値・探索空間取得
    try:
        step_law_hpo = compute_hpo_for_target(
            n_params=arch["n_params"],
            n_tokens=n_tokens,
            seq_len=args.seq_len
        )
        logger.debug(f"Step Law HPO result: {step_law_hpo}")
    except Exception as e:
        logger.exception(f"Error computing Step Law HPO: {e}")
        raise
    
    # 3. Optuna Study
    try:
        search_space = create_search_space(step_law_hpo, vram)
        logger.debug(f"Search space: {search_space}")
        
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        logger.info("Starting Optuna optimization")
        study.optimize(
            lambda trial: objective(trial, arch, args.data_path, args.seq_len, vram, step_law_hpo),
            n_trials=args.n_trials,
            timeout=3600,  # 1時間制限
        )
        logger.info("Optuna optimization completed")
    except Exception as e:
        logger.exception(f"Error during Optuna study: {e}")
        raise
    
    best = study.best_params
    logger.info("Best parameters found", extra={"best_params": best, "best_value": study.best_value})
    
    # 4. 派生パラメータ計算（main.pyで再計算不要な値も含める）
    try:
        per_device = min(best["batch_size_seqs"], 
                         1 if vram <= 4.5 else (4 if vram <= 8.5 else 8))
        grad_accum = max(1, best["batch_size_seqs"] // per_device)
        
        output = {
            "training": {
                **best,
                "per_device_batch_size": per_device,
                "grad_accum_steps": grad_accum,
            }
        }
    except Exception as e:
        logger.exception(f"Error calculating derived parameters: {e}")
        raise
    
    # 5. YAML出力
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {args.output}")
    except Exception as e:
        logger.exception(f"Error saving configuration to {args.output}: {e}")
        raise


if __name__ == "__main__":
    main()

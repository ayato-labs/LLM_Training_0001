#!/usr/bin/env python3
"""
Main training entry point for Novel LLM Scratch Training.

Usage:
    python -m src.main [config_file] [--resume]
    
    config_file: Path to JSON config file (default: configs/experiment_config.json)
    --resume: Resume from latest checkpoint
"""
import sys
import argparse
import json
import optuna
from pathlib import Path
from src.logger import logger, log_exceptions

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_model import train, compute_dataset_fingerprint
from src.step_law import compute_hpo_for_target
from src.hpo_manager import objective


def load_config(config_path: str) -> dict:
    """Load JSON config file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_step_law(config: dict) -> dict:
    """Apply Step Law to optimize hyperparameters based on dataset size."""
    n_params = config["model"]["n_params"]
    train_path = PROJECT_ROOT / config["data"]["train_data_path"]
    
    stats = compute_dataset_fingerprint(str(train_path))
    if "error" in stats:
        logger.warning(f"Could not compute fingerprint: {stats['error']}")
        return config
    
    # 1チャンク=1024トークンと仮定してトークン数を推定
    n_tokens = stats["line_count"] * 1024
    seq_len = config["training"]["seq_len"]
    
    logger.info(f"Applying Step Law for {n_params} params and {n_tokens} tokens...")
    hpo = compute_hpo_for_target(n_params=n_params, n_tokens=n_tokens, seq_len=seq_len)
    
    # Configを更新
    config["hpo"] = hpo
    return config


@log_exceptions
def main():
    parser = argparse.ArgumentParser(description="Novel LLM Scratch Training")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiment_config.json",
        help="Path to JSON config file (default: configs/experiment_config.json)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run Optuna HPO study"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_steps from config"
    )
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Apply HPO Optimization (Default)
    print("Starting HPO study with proxy models...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.get("hpo_trials", 10))
    
    # Update config with HPO results
    best_params = study.best_params
    print(f"Best params found: {best_params}")
    if "hpo" not in config:
        config["hpo"] = {}
    config["hpo"].update(best_params)
    
    # Apply Step Law (as fallback/prior)
    config = apply_step_law(config)
    
    # Apply CLI overrides
    if args.resume:
        config["resume"] = True
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps

    # Run training
    print("Starting training...")
    train(config)


if __name__ == "__main__":
    main()
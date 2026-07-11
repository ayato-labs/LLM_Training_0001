#!/usr/bin/env python3
"""
Main training entry point.

Usage:
    python -m src.main [config_file] [--resume] [--max-steps N]

    config_file: Path to YAML config (default: configs/config.yaml)
"""

import argparse
from pathlib import Path

import optuna

from src.config import load_config, resolve_config_path
from src.hpo_manager import objective
from src.logger import log_exceptions, logger
from src.step_law import compute_hpo_for_target
from src.train_model import compute_dataset_fingerprint, train

PROJECT_ROOT = Path(__file__).parent.parent


def apply_step_law(config: dict) -> dict:
    """Apply Step Law to compute optimal hyperparameters from dataset size."""
    n_params = config["model_params"]["n_params"]
    train_path = PROJECT_ROOT / config["data_path"]

    stats = compute_dataset_fingerprint(str(train_path))
    if "error" in stats:
        logger.warning(f"Could not compute fingerprint: {stats['error']}")
        return config

    n_tokens = stats["line_count"] * 1024
    seq_len = config["hpo"]["seq_len"]

    logger.info(f"Applying Step Law for {n_params} params and {n_tokens} tokens...")
    hpo = compute_hpo_for_target(n_params=n_params, n_tokens=n_tokens, seq_len=seq_len)

    config["hpo"].update(hpo)
    return config


@log_exceptions
def main():
    parser = argparse.ArgumentParser(description="Novel LLM Scratch Training")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config (default: configs/config.yaml)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO study")
    parser.add_argument("--max-steps", type=int, help="Override max_steps")
    args = parser.parse_args()

    # Load config
    config_path = resolve_config_path(args.config)
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # HPO
    logger.info("Starting HPO study with proxy models...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.get("hpo_trials", 10))

    best_params = study.best_params
    logger.info(f"Best params found: {best_params}")
    config["hpo"].update(best_params)

    # Step Law
    config = apply_step_law(config)

    # CLI overrides
    if args.resume:
        config["resume"] = True
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    logger.info("Starting training...")
    train(config)


if __name__ == "__main__":
    main()

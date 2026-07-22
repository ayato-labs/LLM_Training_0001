#!/usr/bin/env python3
"""
LLM Training Entry Point

Usage:
    uv run python -m src.main [OVERRIDES...]

Examples:
    uv run python -m src.main
    uv run python -m src.main training.max_steps=100
    uv run python -m src.main +experiment=debug

    # 学習再開コマンド
    uv run python -m src.training.main resume_from_checkpoint=models/output/checkpoint-latest
"""

import hydra
from omegaconf import DictConfig

from src.common.logger import log_exceptions
from src.training.config import load_config
from src.training.train_engine import train


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
@log_exceptions
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    train(config)


if __name__ == "__main__":
    main()

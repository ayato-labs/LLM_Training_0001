#!/usr/bin/env python3
"""
LLM Training Entry Point

Usage:
    python -m src.main [OVERRIDES...]

Examples:
    python -m src.main
    python -m src.main training.max_steps=100
    python -m src.main +experiment=debug
"""

import hydra
from omegaconf import DictConfig

from src.training.config import load_config
from src.common.logger import log_exceptions
from src.training.train_engine import train


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
@log_exceptions
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    train(config)


if __name__ == "__main__":
    main()

"""Context Extension Main Entrypoint

Hydra CLI を経由して長文拡張エンジン (run_context_extension) を起動。

使用例:
    python -m src.context_extension.main
    python -m src.context_extension.main target_seq_len=8192 rope_scaling.type=yarn
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.common.logger import logger
from src.context_extension.extension_engine import run_context_extension


@hydra.main(version_base=None, config_path="../../configs", config_name="extension_config")
@logger.catch(reraise=True)
def main(cfg: DictConfig) -> None:
    logger.info("=== Long-Context Extension CLI ===")
    result = run_context_extension(cfg)
    logger.info(f"Long-Context Extension completed successfully. Output: {result}")


if __name__ == "__main__":
    main()

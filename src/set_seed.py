"""
Random seed management for deterministic training.
ADR-017: Complete reproducibility via fixed seeds.

Usage:
    from src.set_seed import set_seed
    set_seed(42)  # must be called before model initialization
"""
import os
import random
import numpy as np
import torch
from src.logger import logger


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Fix all random seeds and optionally enable deterministic algorithms.

    Args:
        seed: The seed value to use.
        deterministic: If True, enables torch.use_deterministic_algorithms(True)
                       and cudnn.benchmark=False for full reproducibility.
                       May cause ~10-20% performance degradation.

    Returns:
        The seed value used (for logging).
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        logger.info(f"All random seeds fixed to {seed} (deterministic={deterministic})")
        return seed
    except Exception as e:
        logger.error(f"Failed to set seed {seed}: {e}", exc_info=True)
        raise

"""
乱数シード管理：決定論的学習のため
ADR-017: 固定シードによる完全な再現性確保

Usage:
    from src.set_seed import set_seed
    set_seed(42)  # モデル初期化前に必ず呼ぶ
"""

import os
import random

import numpy as np
import torch

from src.common.logger import logger


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    すべての乱数シードを固定し、オプションで決定論的アルゴリズムを有効化。

    Args:
        seed: 使用するシード値。
        deterministic: True の場合、torch.use_deterministic_algorithms(True)
                       と cudnn.benchmark=False を設定し完全な再現性を確保。
                       約 10-20% の性能低下の可能性あり。

    Returns:
        使用したシード値（ログ用）。
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

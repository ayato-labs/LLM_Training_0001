"""Common Configuration Utility Functions

configs/base_config.yaml や configs/scaling_config.yaml などの設定ファイルを安全に読み込む共通ヘルパー。
"""

from pathlib import Path
from typing import Any

import yaml

from src.common.logger import logger


def load_base_config_yaml(config_path: str = "configs/base_config.yaml") -> dict[str, Any]:
    """configs/base_config.yaml を安全に読み込んで辞書形式で返す共通ヘルパー関数。"""
    path_obj = Path(config_path)
    if not path_obj.exists():
        return {}

    try:
        with open(path_obj, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to read base_config from {config_path}: {e}")
        return {}

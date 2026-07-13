import gc
import os
import sys

import torch

# 1. VRAMキャッシュの解放
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# 2. train.py があるディレクトリの親（プロジェクトルート）を基準にパスを解決
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# カレントディレクトリをプロジェクトルートに変更して相対パスのズレを防止
os.chdir(BASE_DIR)

from src.training.config import load_config, resolve_config_path  # noqa: E402
from src.training.train_model import train  # noqa: E402

if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = resolve_config_path(config_arg)
    config = load_config(config_path)
    train(config)

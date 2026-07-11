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

from src.train_model import train  # noqa: E402

if __name__ == "__main__":
    # 引数からconfigパスを取得、デフォルトはルートのexperiment_config.json
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment_config.json"

    # config_pathが絶対パスでない場合、BASE_DIR基準で絶対パス化
    if not os.path.isabs(config_path):
        config_path = os.path.join(BASE_DIR, config_path)

    train(config_path)

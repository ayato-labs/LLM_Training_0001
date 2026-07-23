import glob
import hashlib
import json
from pathlib import Path

# 開発段位のみ使えるアーキテクチャを改良したけど、途中から学習を続けたいっていう特殊事例のみの想定

def update_hashes():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print(f"Config path {config_path} not found.")
        return

    with open(config_path, "rb") as f:
        current_config_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"Current Config Hash: {current_config_hash}")

    hash_files = glob.glob("**/hashes.json", recursive=True)
    if not hash_files:
        print("No hashes.json files found under current directory.")
        return

    for hf in hash_files:
        path = Path(hf)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            old_hash = data.get("config_hash")
            data["config_hash"] = current_config_hash
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Updated {path}: {old_hash} -> {current_config_hash}")
        except Exception as e:
            print(f"Failed to update {path}: {e}")


if __name__ == "__main__":
    update_hashes()

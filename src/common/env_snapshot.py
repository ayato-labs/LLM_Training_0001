"""Environment Snapshot: 再現性のための最小限メタデータ取得"""

import importlib.metadata
import platform
import subprocess
import sys

import torch


def capture_env_snapshot() -> dict:
    """再現性のための最小限メタデータ取得"""
    snap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "none",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }

    # Git hash
    try:
        snap["git_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        snap["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        snap["git_hash"] = "unknown"

    # Key packages
    for pkg in ["transformers", "datasets", "accelerate", "tokenizers"]:
        try:
            snap[pkg] = importlib.metadata.version(pkg)
        except Exception:
            snap[pkg] = "not_installed"

    return snap

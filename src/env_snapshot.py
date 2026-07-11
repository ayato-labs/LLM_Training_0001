"""
Environment snapshot recorder for experiment traceability.
Captures GPU, CUDA, PyTorch, and system info.

Usage:
    from src.utils.env_snapshot import capture_env_snapshot
    env_info = capture_env_snapshot()
    mlflow.log_dict(env_info, "environment.json")
"""
import os
import sys
import platform
import subprocess
import datetime
import json


def _get_gpu_info() -> dict:
    """Capture GPU properties via PyTorch CUDA API."""
    info = {"available": False}
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        info["available"] = True
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["major"] = props.major
        info["minor"] = props.minor
        info["multi_processor_count"] = props.multi_processor_count
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except Exception as e:
        info["error"] = str(e)
    return info


def _get_git_info() -> dict:
    """Capture current git commit hash."""
    info = {"hash": "unknown"}
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        info["hash"] = result.decode("ascii").strip()
    except Exception:
        pass
    return info


def _get_pip_packages() -> dict:
    """Capture key package versions."""
    packages = {}
    key_pkgs = [
        "torch", "transformers", "datasets", "accelerate", "tokenizers",
        "mlflow", "hydra-core", "omegaconf", "dvc", "scipy", "numpy"
    ]
    try:
        import pkg_resources
        for pkg in key_pkgs:
            try:
                packages[pkg] = pkg_resources.get_distribution(pkg).version
            except pkg_resources.DistributionNotFound:
                packages[pkg] = "not installed"
    except Exception:
        pass
    return packages


def capture_env_snapshot() -> dict:
    """
    Capture a complete environment snapshot.

    Returns:
        Dictionary with all environment metadata suitable for MLflow logging.
    """
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os": platform.system(),
        "os_release": platform.release(),
        "gpu": _get_gpu_info(),
        "git": _get_git_info(),
        "packages": _get_pip_packages(),
    }
    return snapshot


def save_snapshot(path: str = "environment.json") -> str:
    """Capture and save environment snapshot to a JSON file."""
    snapshot = capture_env_snapshot()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"[Env] Environment snapshot saved to {path}")
    return path

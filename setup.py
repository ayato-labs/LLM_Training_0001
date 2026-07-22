#!/usr/bin/env python3
"""
Virtual Environment Setup (GPU / CUDA Support)

Cross-platform replacement for setup.bat.
Unified under uv management.

Usage:
    uv run python setup.py
"""

import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def _venv_python() -> Path:
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    return VENV_DIR / scripts / "python.exe" if sys.platform == "win32" else VENV_DIR / scripts / "python"


def _step(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {msg}")
    print(f"{'=' * 60}")


def check_uv() -> None:
    _step("Check uv installation")
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[ERROR] uv not found. Please install uv:")
        print("  Windows: winget install astral-sh.uv")
        print("  macOS:   brew install uv")
        print("  Linux:   curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  Docs:    https://docs.astral.sh/uv/")
        input("Press Enter to exit...")
        sys.exit(1)
    print("[OK] uv is available")


def remove_venv() -> None:
    _step("Remove existing virtual environment")
    if VENV_DIR.exists():
        print(f"[INFO] Removing {VENV_DIR}...")
        shutil.rmtree(VENV_DIR)
    else:
        print("[SKIP] No existing .venv found")


def create_venv() -> None:
    _step("Create virtual environment (uv venv)")
    subprocess.run(["uv", "venv", str(VENV_DIR), "--python", "3.12"], check=True)
    print("[OK] Virtual environment created")


def sync_dependencies() -> None:
    _step("Install dependency packages (uv sync)")
    print("  torch will be installed with CUDA 12.4 support (cu124)")
    subprocess.run(["uv", "sync"], cwd=ROOT, check=True)
    print("[OK] Dependencies installed")


def verify_torch() -> None:
    _step("Verify CUDA / torch installation")
    python = _venv_python()
    code = """
import torch
cuda_ok = torch.cuda.is_available()
print(f"torch       : {torch.__version__}")
print(f"CUDA build  : {torch.version.cuda}")
print(f"CUDA avail  : {cuda_ok}")
if cuda_ok:
    print(f"GPU name    : {torch.cuda.get_device_name(0)}")
else:
    print("GPU name    : N/A")
"""
    result = subprocess.run([str(python), "-c", code])
    if result.returncode != 0:
        print("[WARN] Failed to verify torch. Please check the installation status.")


def main() -> None:
    check_uv()
    remove_venv()
    create_venv()
    sync_dependencies()
    verify_torch()

    print(f"\n{'=' * 60}")
    print(" Setup Completed Successfully")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Model Utilities: モデル初期化・パラメータ推定"""

from transformers import LlamaConfig


def create_model_config(config: dict, tokenizer) -> LlamaConfig:
    """config dict から LlamaConfig 生成（hidden_size調整込み）"""
    mp = config["model_params"]
    hidden = mp["hidden_size"]
    heads = mp["num_attention_heads"]
    # hidden_size を heads の倍数に調整
    adjusted_hidden = (hidden // heads) * heads

    return LlamaConfig(
        vocab_size=mp["vocab_size"],
        hidden_size=adjusted_hidden,
        intermediate_size=mp["intermediate_size"],
        num_hidden_layers=mp["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=mp["num_key_value_heads"],
        rope_theta=mp["rope_theta"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,  # training用
    )


def estimate_model_size(model) -> int:
    """モデルパラメータ数推定（埋め込み込み）"""
    return sum(p.numel() for p in model.parameters())


import os
import psutil
import hashlib
from pathlib import Path
import torch
from src.common.logger import logger


class TokenizerWrapper:
    """Windowsマルチプロセッシングのピクリング問題をサポートするためのラッパー。"""

    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __call__(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
        )


def get_optimal_num_proc() -> int:
    """CPU論理コアと利用可能なメモリを検出し最適なnum_procを計算。"""
    cpu_cores = os.cpu_count() or 1
    try:
        available_mem_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        available_mem_gb = 8.0  # 検出失敗時のフォールバック

    # 1プロセスあたり1.5GBを見積もる
    mem_based_cores = int(available_mem_gb // 1.5)
    optimal_cores = min(max(1, cpu_cores - 1), max(1, mem_based_cores))
    logger.info(
        f"Resource Auto-Adjustment: Cores={cpu_cores}, Available RAM={available_mem_gb:.1f}GB -> num_proc={optimal_cores}"
    )
    return optimal_cores


def compute_file_hash(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        return ""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_vram() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0


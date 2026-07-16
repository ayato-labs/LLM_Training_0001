"""Model Utilities: モデル初期化・パラメータ推定"""

from transformers import LlamaConfig


def create_model_config(config: dict, tokenizer) -> LlamaConfig:
    """config dict から LlamaConfig 生成（hidden_size調整込み）"""
    mp = config["model_params"]
    hidden = mp["hidden_size"]
    heads = mp["num_attention_heads"]
    # hidden_size を heads の倍数に調整
    adjusted_hidden = (hidden // heads) * heads

    attn_implementation = config.get("attn_implementation") or mp.get("attn_implementation", "flash_attention_2")

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
        attn_implementation=attn_implementation,
    )


def estimate_model_size(model) -> int:
    """モデルパラメータ数推定（埋め込み込み）"""
    return sum(p.numel() for p in model.parameters())


class PackedDatasetWrapper:
    """Sequence packing wrapper to eliminate padding tokens by concatenating texts with EOS and chunking."""
    
    def __init__(self, dataset, seq_len: int, eos_token_id: int):
        self.dataset = dataset
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id

    def __call__(self):
        def pack_function(examples):
            input_ids = examples["input_ids"]
            attention_mask = examples.get("attention_mask", None)
            
            packed_input_ids = []
            packed_attention_masks = []
            
            current_ids = []
            current_mask = []
            
            for i in range(len(input_ids)):
                ids = input_ids[i]
                mask = attention_mask[i] if attention_mask is not None else [1] * len(ids)
                
                # Append EOS token if it's not already at the end
                if len(ids) == 0 or ids[-1] != self.eos_token_id:
                    ids = ids + [self.eos_token_id]
                    mask = mask + [1]
                
                current_ids.extend(ids)
                current_mask.extend(mask)
                
                while len(current_ids) >= self.seq_len:
                    packed_input_ids.append(current_ids[:self.seq_len])
                    packed_attention_masks.append(current_mask[:self.seq_len])
                    current_ids = current_ids[self.seq_len:]
                    current_mask = current_mask[self.seq_len:]
            
            result = {
                "input_ids": packed_input_ids,
                "attention_mask": packed_attention_masks,
                "labels": [list(ids) for ids in packed_input_ids],
            }
            return result

        packed_ds = self.dataset.map(
            pack_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc=f"Packing dataset to seq_len={self.seq_len}",
        )
        return packed_ds


import os
import psutil
import hashlib
from pathlib import Path
import torch
from src.common.logger import logger


class TokenizerWrapper:
    """Windowsマルチプロセッシングのピクリング問題をサポートするためのラッパー。"""

    def __init__(self, tokenizer, seq_len: int, padding: bool = True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.padding = padding

    def __call__(self, examples):
        if self.padding:
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
            )
        else:
            return self.tokenizer(
                examples["text"],
                truncation=False,
            )


def get_optimal_num_proc() -> int:
    """CPU論理コアと利用可能なメモリを検出し最適なnum_procを計算。"""
    import sys
    if sys.platform == "win32":
        logger.info(
            "Resource Auto-Adjustment: Windows detected. Forcing num_proc=None to avoid WinError 87 pipe writing limitations."
        )
        return None

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


def compute_dataset_fingerprint(dataset_path: str) -> dict:
    path = Path(dataset_path)
    if not path.exists():
        return {"error": f"File not found: {dataset_path}"}

    import datetime
    stat = path.stat()
    line_count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            line_count += 1

    return {
        "path": str(path.resolve()),
        "sha256": compute_file_hash(str(path)),
        "size_bytes": stat.st_size,
        "line_count": line_count,
        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def compute_db_fingerprint(db_path: str) -> dict:
    path = Path(db_path)
    if not path.exists():
        return {"error": f"Database not found: {db_path}"}

    import sqlite3
    import datetime

    stat = path.stat()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chapters")
        chapter_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM novels")
        novel_count = cursor.fetchone()[0]
        conn.close()
    except Exception:
        chapter_count = -1
        novel_count = -1

    return {
        "path": str(path.resolve()),
        "sha256": compute_file_hash(str(path)),
        "size_bytes": stat.st_size,
        "chapter_count": chapter_count,
        "novel_count": novel_count,
        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def get_checkpoints(output_dir=None):
    """ステップ番号でソートされた有効なチェックポイントディレクトリを一覧表示。"""
    import re
    target_dir = Path(output_dir) if output_dir else Path("models/output")
    if not target_dir.exists():
        return []

    checkpoints = []
    for item in target_dir.iterdir():
        if item.is_dir() and re.match(r"^checkpoint-\d+$", item.name):
            step = int(item.name.split("-")[1])
            checkpoints.append((step, item))
    return sorted(checkpoints, key=lambda x: x[0])


def cleanup_old_checkpoints(keep=2, output_dir=None):
    """ローカルの古いチェックポイントディレクトリを削除し、最新の`keep`個のみを保持。"""
    import shutil
    checkpoints = get_checkpoints(output_dir=output_dir)
    if len(checkpoints) <= keep:
        return

    to_remove = checkpoints[:-keep]
    for _step, path in to_remove:
        logger.info(f"Cleaning up old checkpoint: {path.name}")
        shutil.rmtree(path)
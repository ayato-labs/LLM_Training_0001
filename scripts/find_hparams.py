#!/usr/bin/env python3
"""
Offline Hyperparameter Search Script
分離された探索フェーズ。学習パイプライン(main.py)からは完全独立。

Usage:
  python -m scripts.find_hparams --model-size 150M --data-path data/dataset.jsonl --output configs/hparams_150M.yaml --n-trials 20
"""

import argparse
import yaml
from pathlib import Path
import json

import optuna
from src.step_law import compute_hpo_for_target
from src.hpo_manager import objective, create_search_space
from src.config import detect_vram
from src.logger import logger


def parse_args():
    p = argparse.ArgumentParser(description="Offline HPO for LLM Training")
    p.add_argument("--model-size", required=True, choices=["50M", "150M", "3B", "7B"],
                   help="Target model size (determines architecture)")
    p.add_argument("--data-path", required=True, help="Path to training dataset (JSONL)")
    p.add_argument("--output", required=True, help="Output YAML path (e.g., configs/hparams_150M.yaml)")
    p.add_argument("--n-trials", type=int, default=20, help="Optuna trials")
    p.add_argument("--vram-gb", type=float, help="Override VRAM detection")
    p.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    return p.parse_args()


def get_base_config(model_size: str) -> dict:
    """モデルサイズごとのベースアーキテクチャ設定"""
    configs = {
        "50M":  {"n_params": 50_000_000,  "hidden": 640, "layers": 10, "heads": 10, "kv_heads": 10, "ffn": 2560},
        "150M": {"n_params": 150_000_000, "hidden": 768, "layers": 12, "heads": 12, "kv_heads": 3,  "ffn": 3072},
        "3B":   {"n_params": 3_000_000_000, "hidden": 2560, "layers": 28, "heads": 20, "kv_heads": 20, "ffn": 10240},
        "7B":   {"n_params": 7_000_000_000, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 32, "ffn": 11008},
    }
    return configs[model_size]


def estimate_tokens(data_path: str) -> int:
    """データセットトークン数概算"""
    count = 0
    with open(data_path) as f:
        for line in f:
            count += 1
    return count * 1024  # 1行≒1024トークン想定


def main():
    args = parse_args()
    
    # 1. ベース設定構築
    arch = get_base_config(args.model_size)
    n_tokens = estimate_tokens(args.data_path)
    vram = args.vram_gb or detect_vram()
    
    logger.info(f"HPO Search: {args.model_size} | Tokens: {n_tokens:,} | VRAM: {vram}GB | Trials: {args.n_trials}")
    
    # 2. Step Law で初期値・探索空間取得
    step_law_hpo = compute_hpo_for_target(
        n_params=arch["n_params"],
        n_tokens=n_tokens,
        seq_len=args.seq_len
    )
    
    # 3. Optuna Study
    search_space = create_search_space(step_law_hpo, vram)
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, arch, args.data_path, args.seq_len, vram),
        n_trials=args.n_trials,
        timeout=3600,  # 1時間制限
    )
    
    best = study.best_params
    logger.info(f"Best params: {best}")
    logger.info(f"Best value (loss): {study.best_value:.4f}")
    
    # 4. 派生パラメータ計算（main.pyで再計算不要な値も含める）
    per_device = min(best["batch_size_seqs"], 
                     1 if vram <= 4.5 else (4 if vram <= 8.5 else 8))
    grad_accum = max(1, best["batch_size_seqs"] // per_device)
    
    # Step Law由来の max_lr_1d (AdamW用) を明示的に保存
    # best には max_lr_1d も含まれているはず（探索空間に定義済み）
    output = {
        "training": {
            **best,
            "per_device_batch_size": per_device,
            "grad_accum_steps": grad_accum,
        }
    }
    
    # 5. YAML出力
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

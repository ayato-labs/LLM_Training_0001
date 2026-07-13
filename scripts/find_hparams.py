#!/usr/bin/env python3
"""
Offline Hyperparameter Search Script
分離された探索フェーズ。学習パイプライン(main.py)からは完全独立。

Usage:
  python -m scripts.find_hparams --model-size 150M --data-path data/dataset.jsonl --output configs/hparams_150M.yaml --n-trials 20
"""

import argparse
from pathlib import Path

import optuna
import yaml
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from src.training.config import _detect_vram as detect_vram
from src.hpo.hpo_manager import create_search_space, objective
from src.common.logger import log_exceptions, log_function_call, logger
from src.hpo.step_law import compute_hpo_for_target
from src.training.train_model import TokenizerWrapper, get_optimal_num_proc


def parse_args():
    p = argparse.ArgumentParser(description="Offline HPO for LLM Training")
    p.add_argument(
        "--model-size",
        choices=["50M", "150M", "3B", "7B"],
        help="Target model size (optional, overrides config.yaml architecture if specified)",
    )
    p.add_argument("--data-path", required=True, help="Path to training dataset (JSONL)")
    p.add_argument(
        "--output", required=True, help="Output YAML path (e.g., configs/hparams_150M.yaml)"
    )
    p.add_argument("--n-trials", type=int, default=100, help="Optuna trials")
    p.add_argument("--vram-gb", type=float, help="Override VRAM detection")
    p.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    return p.parse_args()


@log_function_call(log_args=True)
def get_base_config(model_size: str = None) -> dict:
    """ベース設定構築、または configs/config.yaml から動的に取得"""
    # 1. まず config.yaml からのロードを試みる
    try:
        from omegaconf import OmegaConf

        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            model = cfg.get("model", {})
            llama = model.get("llama", {})
            if llama:
                logger.info("Loaded architecture settings from configs/config.yaml")
                return {
                    "n_params": model.get("target_params", 150_000_000),
                    "hidden": llama.get("hidden_size", 768),
                    "layers": llama.get("num_hidden_layers", 12),
                    "heads": llama.get("num_attention_heads", 12),
                    "kv_heads": llama.get("num_key_value_heads", 3),
                    "ffn": llama.get("intermediate_size", 3072),
                }
    except Exception as e:
        logger.warning(
            f"Could not load architecture from config.yaml: {e}. Falling back to default list."
        )

    # 2. 指定された model_size、またはデフォルトのフォールバック
    size_key = model_size or "150M"
    configs = {
        "50M": {
            "n_params": 50_000_000,
            "hidden": 640,
            "layers": 10,
            "heads": 10,
            "kv_heads": 10,
            "ffn": 2560,
        },
        "150M": {
            "n_params": 150_000_000,
            "hidden": 768,
            "layers": 12,
            "heads": 12,
            "kv_heads": 3,
            "ffn": 3072,
        },
        "3B": {
            "n_params": 3_000_000_000,
            "hidden": 2560,
            "layers": 28,
            "heads": 20,
            "kv_heads": 20,
            "ffn": 10240,
        },
        "7B": {
            "n_params": 7_000_000_000,
            "hidden": 4096,
            "layers": 32,
            "heads": 32,
            "kv_heads": 32,
            "ffn": 11008,
        },
    }
    try:
        return configs[size_key]
    except KeyError:
        logger.error(f"Invalid model size: {size_key}")
        raise


@log_function_call(log_args=True)
def estimate_tokens(data_path: str) -> int:
    """データセットのトークン数概算"""
    try:
        count = 0
        with open(data_path, encoding="utf-8") as f:
            for _line in f:
                count += 1
        logger.debug(f"Estimated tokens from {data_path}: {count * 1024}")
        return count * 1024  # 1行≒1024トークン想定
    except FileNotFoundError:
        logger.exception(f"Dataset file not found: {data_path}")
        raise
    except Exception as e:
        logger.exception(f"Error estimating tokens: {e}")
        raise


@log_exceptions
def main():
    args = parse_args()
    logger.info("Starting HPO search", extra={"args": vars(args)})

    # 1. ベース設定構築
    arch = get_base_config(args.model_size)
    n_tokens = estimate_tokens(args.data_path)
    vram = args.vram_gb or detect_vram()

    logger.info(
        "HPO Search initialized",
        extra={
            "model_size": args.model_size,
            "tokens": n_tokens,
            "vram": vram,
            "trials": args.n_trials,
        },
    )

    # 2. Tokenize dataset once
    logger.info(f"Loading and tokenizing dataset from {args.data_path} once...")
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"

        dataset = load_dataset("json", data_files=str(args.data_path))
        num_proc = get_optimal_num_proc()
        tokenize_function = TokenizerWrapper(tokenizer, args.seq_len)
        all_cols = dataset["train"].column_names
        cols_to_remove = [c for c in all_cols if c not in ["input_ids", "attention_mask", "labels"]]

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=cols_to_remove,
            keep_in_memory=True,
        )

        # Apply data_fraction = 0.001 (0.1%) for HPO pilot run speed
        for split in tokenized_dataset:
            n_samples = int(len(tokenized_dataset[split]) * 0.001)
            # Ensure at least 1 sample exists
            n_samples = max(1, n_samples)
            if n_samples < len(tokenized_dataset[split]):
                tokenized_dataset[split] = tokenized_dataset[split].select(range(n_samples))
                logger.info(f"Split '{split}' sampled to {n_samples} samples for HPO.")

        tokenized_dataset.set_format("torch")
        logger.info("Dataset tokenized and cached in memory successfully.")
    except Exception as e:
        logger.exception(f"Error pre-tokenizing dataset: {e}")
        raise

    # 3. Step Law で初期値・探索空間取得
    try:
        step_law_hpo = compute_hpo_for_target(
            n_params=arch["n_params"], n_tokens=n_tokens, seq_len=args.seq_len
        )
        logger.debug(f"Step Law HPO result: {step_law_hpo}")
    except Exception as e:
        logger.exception(f"Error computing Step Law HPO: {e}")
        raise

    # 4. Optuna Study with MedianPruner
    try:
        search_space = create_search_space(step_law_hpo, vram)
        logger.debug(f"Search space: {search_space}")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=15,
                interval_steps=5,
            )
        )
        study.set_user_attr("n_trials", args.n_trials)
        logger.info("Starting Optuna optimization")
        study.optimize(
            lambda trial: objective(trial, arch, tokenized_dataset, args.seq_len, vram, step_law_hpo),
            n_trials=args.n_trials,
            timeout=28800,  # 8 hours timeout
        )
        logger.info("Optuna optimization completed")
    except Exception as e:
        logger.exception(f"Error during Optuna study: {e}")
        raise

    best = study.best_params
    logger.info(
        "Best parameters found", extra={"best_params": best, "best_value": study.best_value}
    )

    # 5. 派生パラメータ計算（main.pyで再計算不要な値も含める）
    try:
        per_device = min(best["batch_size_seqs"], 1 if vram <= 4.5 else (4 if vram <= 8.5 else 8))
        grad_accum = max(1, best["batch_size_seqs"] // per_device)

        # Include fixed parameters in the output configuration
        output = {
            "training": {
                **best,
                "warmup_ratio": 0.03,
                "beta2": 0.95,
                "grad_clip": 1.0,
                "per_device_batch_size": per_device,
                "grad_accum_steps": grad_accum,
            }
        }
    except Exception as e:
        logger.exception(f"Error calculating derived parameters: {e}")
        raise

    # 6. YAML出力
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {args.output}")
    except Exception as e:
        logger.exception(f"Error saving configuration to {args.output}: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Offline Hyperparameter Search Script
分離された探索フェーズ。学習パイプライン(main.py)からは完全独立。

Usage:
  python -m scripts.find_hparams --data-path data/dataset.jsonl --output configs/hparams_150M.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna
import yaml
from datasets import disable_caching, load_dataset
from transformers import PreTrainedTokenizerFast

# Linuxの /tmp (tmpfs RAMディスク) の容量不足による [Errno 28] OOM を回避するため、
# 一時ディレクトリを十分な空き容量のあるローカルディスク（ext4）上に強制指定
os.environ["TMPDIR"] = str(Path("models/output/tmp").resolve())
Path("models/output/tmp").mkdir(parents=True, exist_ok=True)

# PyTorch CUDA メモリアロケータ設定（断片化対策・OOM緩和）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

disable_caching()

from src.common.logger import log_exceptions, log_function_call, logger
from src.hpo.hpo_manager import create_search_space, objective
from src.hpo.step_law import compute_hpo_for_target
from src.training.config import _detect_vram as detect_vram
from src.training.model_utils import (
    calculate_optimal_batch_split,
    compute_file_hash,
    get_optimal_num_proc,
    parallel_tokenize,
)


def _estimate_vram_from_arch(
    arch: dict, batch_seqs: int, seq_len: int, target_vram: float
) -> float:
    """アーキテクチャ情報から理論 VRAM 使用量を簡易推定 (GB)"""
    n_params = arch["n_params"]
    model_bytes = n_params * 6
    activation = batch_seqs * seq_len * arch["hidden"] * arch["layers"] * 2
    cuda_overhead = 0.7 + (0.3 if "linux" in sys.platform else 0.0)
    return (model_bytes + activation) / (1024**3) + cuda_overhead


def parse_args():
    p = argparse.ArgumentParser(description="Offline HPO for LLM Training")
    p.add_argument(
        "--proxy-model-size",
        type=str,
        default=None,
        help="Optional proxy model size for HPO (e.g. 50M, 150M, 1.5B). Auto-calculated via 10%% golden ratio if omitted.",
    )
    p.add_argument(
        "--target-model-size",
        type=str,
        default=None,
        help="Optional target model size (e.g. 150M, 3B, 7B). Auto-resolved from config.yaml if omitted.",
    )
    p.add_argument(
        "--target-vram-gb",
        type=float,
        help="Target model VRAM (GB) for batch size calculation (defaults to proxy VRAM)",
    )
    p.add_argument("--data-path", required=True, help="Path to training dataset (JSONL)")
    p.add_argument(
        "--output",
        default="configs/hpo_config.yaml",
        help="Output YAML path (defaults to configs/hpo_config.yaml)",
    )

    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optuna trials count (optional: dynamically calculated based on proxy model size if omitted)",
    )
    p.add_argument("--vram-gb", type=float, help="Override VRAM detection for proxy")
    p.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    return p.parse_args()


def calculate_dynamic_n_trials(search_space_dim: int = 5) -> int:
    """
    探索空間の次元数 (自由度: 5次元: lr_2d, lr_1d, batch_size, weight_decay, warmup_ratio) に基づき
    理論的に必要な Optuna 試行回数を自律計算する。
    
    1次元あたり 30 試行を基本基準とし、Optuna の MedianPruner (枝刈り) と併用することで
    過剰適合や手動ハードコードを完全に排除。
    """
    n_trials = search_space_dim * 30

    logger.info(
        f"[Dynamic HPO Trial Allocation] Search space dimensions: {search_space_dim} -> "
        f"Auto-calculated n_trials: {n_trials} trials (30 trials/dim)"
    )
    return n_trials



def determine_optimal_proxy_size(target_params: int) -> str:
    """
    ターゲットモデルの 10% 規模（下限 50M ガード）を厳密計算し、
    動的アーキテクチャ生成用のパラメータ表記文字列（例: "50M", "350M", "700M" 等）を自動返却する。
    """
    proxy_params = max(50_000_000, int(target_params * 0.10))
    proxy_size_str = f"{int(proxy_params / 1e6)}M"

    logger.info(
        f"[Golden Ratio Proxy Selection] Target parameters: {target_params:,} -> "
        f"Auto-calculated proxy size: '{proxy_size_str}' ({proxy_params:,} params, Min bound: 50M)"
    )
    return proxy_size_str




@log_function_call(log_args=True)
def get_base_config(model_size: str = None) -> dict:
    """ベース設定構築、または configs/config.yaml から動的に取得"""
    if model_size is None:
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
                        "rope_theta": llama.get("rope_theta", 500000.0),
                    }
        except Exception as e:
            logger.warning(
                f"Could not load architecture from config.yaml: {e}. Falling back to default list."
            )

    configs = {
        "50M": {
            "n_params": 50_000_000,
            "hidden": 640,
            "layers": 10,
            "heads": 10,
            "kv_heads": 10,
            "ffn": 2560,
            "rope_theta": 500000.0,
        },
        "150M": {
            "n_params": 150_000_000,
            "hidden": 768,
            "layers": 12,
            "heads": 12,
            "kv_heads": 3,
            "ffn": 3072,
            "rope_theta": 500000.0,
        },
        "3B": {
            "n_params": 3_000_000_000,
            "hidden": 2560,
            "layers": 28,
            "heads": 20,
            "kv_heads": 20,
            "ffn": 10240,
            "rope_theta": 500000.0,
        },
        "7B": {
            "n_params": 7_000_000_000,
            "hidden": 4096,
            "layers": 32,
            "heads": 32,
            "kv_heads": 32,
            "ffn": 11008,
            "rope_theta": 500000.0,
        },
    }

    size_key = model_size or "150M"
    if size_key in configs:
        return configs[size_key]

    try:
        size_upper = model_size.upper().strip()
        if size_upper.endswith("M"):
            n_params = int(float(size_upper[:-1]) * 1e6)
        elif size_upper.endswith("B"):
            n_params = int(float(size_upper[:-1]) * 1e9)
        elif size_upper.endswith("K"):
            n_params = int(float(size_upper[:-1]) * 1e3)
        else:
            n_params = int(size_upper)

        hidden = max(128, (int((n_params / 1.5e5) ** 0.5) * 64))
        layers = max(4, int(hidden / 64))
        heads = max(4, hidden // 64)
        kv_heads = max(1, heads // 4)
        ffn = int(hidden * 4)

        logger.info(
            f"Dynamically generated custom architecture for '{model_size}': "
            f"n_params={n_params:,}, hidden={hidden}, layers={layers}, heads={heads}"
        )
        return {
            "n_params": n_params,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "kv_heads": kv_heads,
            "ffn": ffn,
            "rope_theta": 500000.0,
        }
    except Exception as e:
        logger.error(f"Failed to parse custom model size '{model_size}': {e}")
        raise ValueError(f"Invalid or unparseable model size: {model_size}")


@log_function_call(log_args=True)
def estimate_tokens(data_path: str) -> int:
    """データセットのトークン数概算"""
    try:
        count = 0
        with open(data_path, encoding="utf-8") as f:
            for _line in f:
                count += 1
        logger.debug(f"Estimated tokens from {data_path}: {count * 1024}")
        return count * 1024
    except FileNotFoundError:
        logger.exception(f"Dataset file not found: {data_path}")
        raise
    except Exception as e:
        logger.exception(f"Error estimating tokens: {e}")
        raise


def sync_config_yaml(target_arch: dict, target_size: str, hparams_output: str) -> None:
    """config.yaml をターゲットアーキテクチャに同期する"""
    from omegaconf import OmegaConf

    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        logger.warning(f"config.yaml not found at {config_path}, skipping sync")
        return

    cfg = OmegaConf.load(config_path)
    cfg.model.target_params = target_arch["n_params"]
    cfg.model.llama.hidden_size = target_arch["hidden"]
    cfg.model.llama.num_hidden_layers = target_arch["layers"]
    cfg.model.llama.num_attention_heads = target_arch["heads"]
    cfg.model.llama.num_key_value_heads = target_arch["kv_heads"]
    cfg.model.llama.intermediate_size = target_arch["ffn"]
    cfg.model.llama.rope_theta = target_arch.get("rope_theta", 500000.0)

    hparams_name = f"hparams_{target_size}"
    if cfg.defaults and isinstance(cfg.defaults[0], dict):
        if list(cfg.defaults[0].keys())[0] == "hparams_150M":
            cfg.defaults[0] = {hparams_name: ""}
    elif (
        cfg.defaults and isinstance(cfg.defaults[0], str) and cfg.defaults[0].startswith("hparams_")
    ):
        cfg.defaults[0] = hparams_name

    OmegaConf.save(cfg, config_path)
    logger.info(
        f"Updated config.yaml for {target_size}: "
        f"target_params={target_arch['n_params']}, "
        f"hparams={hparams_name}"
    )


@log_exceptions
def main():
    args = parse_args()
    logger.info("Starting HPO search", extra={"args": vars(args)})

    target_size = args.target_model_size
    target_arch = get_base_config(target_size)

    if args.proxy_model_size:
        proxy_size = args.proxy_model_size
    else:
        proxy_size = determine_optimal_proxy_size(target_arch["n_params"])

    proxy_arch = get_base_config(proxy_size)

    n_tokens = estimate_tokens(args.data_path)
    proxy_vram = args.vram_gb or detect_vram()
    target_vram = args.target_vram_gb or proxy_vram

    # n_trials が指定されていない場合は、探索自由度（次元数）に応じた理論試行回数を算出
    if args.n_trials:
        n_trials = args.n_trials
    else:
        n_trials = calculate_dynamic_n_trials(search_space_dim=5)


    logger.info(
        "HPO Search initialized",
        extra={
            "proxy_model_size": proxy_size,
            "target_model_size": target_size or "from_config",
            "tokens": n_tokens,
            "proxy_vram": proxy_vram,
            "target_vram": target_vram,
            "trials": n_trials,
        },
    )


    logger.info(f"Loading and tokenizing dataset from {args.data_path} once...")
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"

        dataset = load_dataset("json", data_files=str(args.data_path))

        for split in dataset:
            n_samples = int(len(dataset[split]) * 0.001)
            n_samples = max(1, n_samples)
            if n_samples < len(dataset[split]):
                dataset[split] = dataset[split].select(range(n_samples))
                logger.info(f"Split '{split}' raw dataset sampled to {n_samples} samples for HPO.")

        all_cols = dataset["train"].column_names
        cols_to_remove = [c for c in all_cols if c not in ["input_ids", "attention_mask", "labels"]]

        num_proc = get_optimal_num_proc()
        logger.info(f"Tokenizing dataset with parallelism={num_proc}")

        tokenized_dataset = parallel_tokenize(
            dataset,
            tokenizer,
            seq_len=args.seq_len,
            padding=True,
            remove_columns=cols_to_remove,
            max_workers=num_proc,
            batch_size=1000,
        )

        tokenized_dataset.set_format("torch")
        logger.info("Dataset tokenized and cached in memory successfully.")
    except Exception as e:
        logger.exception(f"Error pre-tokenizing dataset: {e}")
        raise

    try:
        step_law_hpo = compute_hpo_for_target(
            n_params=proxy_arch["n_params"], n_tokens=n_tokens, seq_len=args.seq_len
        )
        logger.debug(f"Proxy Step Law HPO result: {step_law_hpo}")
    except Exception as e:
        logger.exception(f"Error computing Step Law HPO: {e}")
        raise

    try:
        search_space = create_search_space(
            step_law_hpo, proxy_vram, n_params=proxy_arch["n_params"]
        )
        logger.debug(f"Search space: {search_space}")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=15,
                interval_steps=5,
            ),
        )
        study.set_user_attr("n_trials", n_trials)
        logger.info(f"Starting Optuna optimization with n_trials={n_trials}")
        study.optimize(
            lambda trial: objective(
                trial, proxy_arch, tokenized_dataset, args.seq_len, proxy_vram, step_law_hpo
            ),
            n_trials=n_trials,
            timeout=86400,
        )

        logger.info("Optuna optimization completed")
    except Exception as e:
        logger.exception(f"Error during Optuna study: {e}")
        raise

    best = study.best_params
    logger.info(
        "Best proxy parameters found", extra={"best_params": best, "best_value": study.best_value}
    )

    try:
        n_proxy = proxy_arch["n_params"]
        n_target = target_arch["n_params"]
        scaling_ratio = (n_target / n_proxy) ** -0.713
        logger.info(
            f"Applying Step Law scaling ratio from "
            f"{proxy_size}({n_proxy}) to {target_size}({n_target}): "
            f"{scaling_ratio:.6f}"
        )

        scaled_best = {}
        for k, v in best.items():
            if k in ["max_lr_2d", "max_lr_1d"]:
                scaled_best[k] = round(float(v * scaling_ratio), 6)
            else:
                scaled_best[k] = v

        target_batch_seqs = scaled_best.get("batch_size_seqs", 16)
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=target_batch_seqs,
            vram_gb=target_vram,
            n_params=target_arch["n_params"],
            seq_len=args.seq_len,
            hidden_size=target_arch["hidden"],
            num_layers=target_arch["layers"],
            precision="bf16",
            selective_checkpointing=True,
        )

        output = {
            "training": {
                **scaled_best,
                "warmup_ratio": scaled_best.get("warmup_ratio", 0.03),
                "beta2": 0.95,
                "grad_clip": 1.0,
                "per_device_batch_size": per_device,
                "grad_accum_steps": grad_accum,
            }
        }
    except Exception as e:
        logger.exception(f"Error calculating scaled derived parameters: {e}")
        raise

    # 6. YAML出力 (configs/hpo_config.yaml 形式に統一)
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        # Hydraレイヤー合成用ヘッダーコメントを追加
        hpo_yaml_content = (
            "# @package _global_\n"
            "# HPO 探索 (scripts.find_hparams / Step Law + Optuna) 成果物 (最適ハイパーパラメータ)\n\n"
            + yaml.dump(output, default_flow_style=False, sort_keys=False)
        )

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(hpo_yaml_content)
        logger.info(f"Configuration saved to {args.output}")
    except Exception as e:
        logger.exception(f"Error saving configuration to {args.output}: {e}")
        raise

    try:
        trials_df = study.trials_dataframe()
        csv_path = Path(args.output).with_suffix(".trials.csv")
        trials_df.to_csv(csv_path, index=False)
        logger.info(f"All {len(trials_df)} trials saved to {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save trials CSV: {e}")

    final_batch_seqs = output["training"].get("batch_size_seqs", 16)
    estimated_vram = _estimate_vram_from_arch(
        target_arch, final_batch_seqs, args.seq_len, target_vram
    )

    logger.info("=" * 60)
    logger.info("  [HPO Final Config] 本番学習で使用される最終設定")
    logger.info("=" * 60)
    for k, v in output["training"].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  estimated_vram_gb: {estimated_vram:.2f}")
    logger.info(f"  step_law_scaling_ratio: {scaling_ratio:.6f}")
    logger.info(f"  best_proxy_loss: {study.best_value:.4f}")

    logger.info("  [Scaling Trace] Step Law -> Proxy HPO -> Target Scaled")
    for k in ["max_lr_2d", "max_lr_1d", "warmup_ratio", "batch_size_seqs", "weight_decay"]:
        sl = step_law_hpo.get(k)
        proxy = best.get(k)
        final = output["training"].get(k)
        logger.info(f"    {k}: StepLaw={sl} -> Proxy={proxy} -> Final={final}")
    logger.info("=" * 60)

    try:
        git_commit = subprocess.getoutput("git rev-parse HEAD 2>/dev/null || echo unknown")
        data_hash = compute_file_hash(args.data_path) if Path(args.data_path).exists() else None
        meta = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
            "seed": 42,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_complete": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "best_value": study.best_value,
            "proxy_model_size": proxy_size,
            "target_model_size": target_size,
            "n_tokens": n_tokens,
            "dataset_hash": data_hash,
            "proxy_vram_gb": proxy_vram,
            "target_vram_gb": target_vram,
            "estimated_final_vram_gb": estimated_vram,
            "scaling_ratio": scaling_ratio,
        }
        json_path = Path(args.output).with_suffix(".meta.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Run metadata saved to {json_path}")
    except Exception as e:
        logger.warning(f"Failed to save run metadata: {e}")



if __name__ == "__main__":
    main()

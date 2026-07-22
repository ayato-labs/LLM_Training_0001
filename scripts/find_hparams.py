#!/usr/bin/env python3
"""
Offline Hyperparameter Search Script
分離された探索フェーズ。学習パイプライン(main.py)からは完全独立。

Usage:
  python -m scripts.find_hparams --model-size 150M \\
    --data-path data/dataset.jsonl \\
    --output configs/hparams_150M.yaml --n-trials 20
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
# expandable_segments: True で小さな割り当てをまとめて管理し、VRAM断片化を抑制
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

disable_caching()


from src.common.logger import log_exceptions, log_function_call, logger
from src.hpo.hpo_manager import create_search_space, objective
from src.hpo.step_law import compute_hpo_for_target
from src.training.config import _detect_vram as detect_vram
from src.training.model_utils import compute_file_hash, get_optimal_num_proc, parallel_tokenize


def _estimate_vram_from_arch(
    arch: dict, batch_seqs: int, seq_len: int, target_vram: float
) -> float:
    """アーキテクチャ情報から理論 VRAM 使用量を簡易推定 (GB)"""
    n_params = arch["n_params"]
    # bf16前提: weights 2B + grads 2B + optimizer 8bit 2B = 6B/param
    model_bytes = n_params * 6
    # アクティベーション (gradient_checkpointing 前提)
    activation = batch_seqs * seq_len * arch["hidden"] * arch["layers"] * 2
    # システムオーバーヘッド
    cuda_overhead = 0.7 + (0.3 if "linux" in sys.platform else 0.0)
    return (model_bytes + activation) / (1024**3) + cuda_overhead


def parse_args():
    p = argparse.ArgumentParser(description="Offline HPO for LLM Training")
    p.add_argument(
        "--proxy-model-size",
        choices=["50M", "150M", "3B", "7B"],
        default="150M",
        help="Model size to run HPO proxy search on (defaults to 150M)",
    )
    p.add_argument(
        "--target-model-size",
        choices=["50M", "150M", "3B", "7B"],
        help="Final target model size (defaults to proxy size if not specified)",
    )
    p.add_argument(
        "--target-vram-gb",
        type=float,
        help="Target model VRAM (GB) for batch size calculation (defaults to proxy VRAM)",
    )
    p.add_argument("--data-path", required=True, help="Path to training dataset (JSONL)")
    p.add_argument(
        "--output", required=True, help="Output YAML path (e.g., configs/hparams_150M.yaml)"
    )
    p.add_argument("--n-trials", type=int, default=150, help="Optuna trials (5D: 150推奨)")
    p.add_argument("--vram-gb", type=float, help="Override VRAM detection for proxy")
    p.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    p.add_argument(
        "--sync-config",
        action="store_true",
        help=(
            "Also update configs/config.yaml with target architecture "
            "(for target_model_size != proxy)"
        ),
    )
    return p.parse_args()


@log_function_call(log_args=True)
def get_base_config(model_size: str = None) -> dict:
    """ベース設定構築、または configs/config.yaml から動的に取得"""
    # 1. model_size が明示的に指定されていない場合のみ config.yaml からのロードを試みる
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


def sync_config_yaml(target_arch: dict, target_size: str, hparams_output: str) -> None:
    """config.yaml をターゲットアーキテクチャに同期する"""
    from omegaconf import OmegaConf

    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        logger.warning(f"config.yaml not found at {config_path}, skipping sync")
        return

    cfg = OmegaConf.load(config_path)

    # model.target_params 更新
    cfg.model.target_params = target_arch["n_params"]

    # llama アーキテクチャ更新
    cfg.model.llama.hidden_size = target_arch["hidden"]
    cfg.model.llama.num_hidden_layers = target_arch["layers"]
    cfg.model.llama.num_attention_heads = target_arch["heads"]
    cfg.model.llama.num_key_value_heads = target_arch["kv_heads"]
    cfg.model.llama.intermediate_size = target_arch["ffn"]
    cfg.model.llama.rope_theta = target_arch.get("rope_theta", 500000.0)

    # hparams ファイル名も更新 (config.yaml の defaults には hparams_<size>.yaml を期待)
    hparams_name = f"hparams_{target_size}"
    if cfg.defaults and isinstance(cfg.defaults[0], dict):
        # 既存の defaults が dict 形式の場合はキーを更新
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

    # 1. ベース設定構築
    proxy_size = args.proxy_model_size
    target_size = args.target_model_size or proxy_size

    proxy_arch = get_base_config(proxy_size)
    target_arch = get_base_config(target_size)

    n_tokens = estimate_tokens(args.data_path)
    proxy_vram = args.vram_gb or detect_vram()
    target_vram = args.target_vram_gb or proxy_vram

    logger.info(
        "HPO Search initialized",
        extra={
            "proxy_model_size": proxy_size,
            "target_model_size": target_size,
            "tokens": n_tokens,
            "proxy_vram": proxy_vram,
            "target_vram": target_vram,
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

        # Apply data_fraction = 0.001 (0.1%) before tokenization to save massive RAM & time
        for split in dataset:
            n_samples = int(len(dataset[split]) * 0.001)
            n_samples = max(1, n_samples)
            if n_samples < len(dataset[split]):
                dataset[split] = dataset[split].select(range(n_samples))
                logger.info(f"Split '{split}' raw dataset sampled to {n_samples} samples for HPO.")

        all_cols = dataset["train"].column_names
        cols_to_remove = [c for c in all_cols if c not in ["input_ids", "attention_mask", "labels"]]

        # プラットフォーム自動判定で並列トークナイズ (Windows: ThreadPool, Linux: multiprocessing)
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

    # 3. Step Law で初期値・探索空間取得 (プロキシモデルに基づき算出)
    try:
        step_law_hpo = compute_hpo_for_target(
            n_params=proxy_arch["n_params"], n_tokens=n_tokens, seq_len=args.seq_len
        )
        logger.debug(f"Proxy Step Law HPO result: {step_law_hpo}")
    except Exception as e:
        logger.exception(f"Error computing Step Law HPO: {e}")
        raise

    # 4. Optuna Study with MedianPruner
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
        study.set_user_attr("n_trials", args.n_trials)
        logger.info("Starting Optuna optimization")
        study.optimize(
            lambda trial: objective(
                trial, proxy_arch, tokenized_dataset, args.seq_len, proxy_vram, step_law_hpo
            ),
            n_trials=args.n_trials,
            timeout=86400,  # 24時間 (安全弁。フル本番は数日)
        )
        logger.info("Optuna optimization completed")
    except Exception as e:
        logger.exception(f"Error during Optuna study: {e}")
        raise

    best = study.best_params
    logger.info(
        "Best proxy parameters found", extra={"best_params": best, "best_value": study.best_value}
    )

    # 5. 派生パラメータ計算とスケーリング転移の実施
    try:
        # プロキシモデルからターゲットモデルへの学習率スケーリング転移 (N_target / N_proxy)^(-0.713)
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

        # ターゲット用のデバイスあたりバッチサイズと勾配累積ステップ数算出 (ターゲットVRAM基準)
        target_batch_seqs = scaled_best.get("batch_size_seqs", 16)
        per_device = min(
            target_batch_seqs, 1 if target_vram <= 4.5 else (4 if target_vram <= 8.5 else 8)
        )
        grad_accum = max(1, target_batch_seqs // per_device)

        # Include fixed parameters in the final output configuration
        # warmup_ratio は HPO 結果を尊重（探索されていない場合のみフォールバック）
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

    # 6. YAML出力
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {args.output}")
    except Exception as e:
        logger.exception(f"Error saving configuration to {args.output}: {e}")
        raise

    # 7. 全試行 CSV 保存
    try:
        trials_df = study.trials_dataframe()
        csv_path = Path(args.output).with_suffix(".trials.csv")
        trials_df.to_csv(csv_path, index=False)
        logger.info(f"All {len(trials_df)} trials saved to {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save trials CSV: {e}")

    # 8. VRAM 推定値の算出
    final_batch_seqs = output["training"].get("batch_size_seqs", 16)
    estimated_vram = _estimate_vram_from_arch(
        target_arch, final_batch_seqs, args.seq_len, target_vram
    )

    # 9. 最終スケーリング値サマリ出力
    logger.info("=" * 60)
    logger.info("  [HPO Final Config] 本番学習で使用される最終設定")
    logger.info("=" * 60)
    for k, v in output["training"].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  estimated_vram_gb: {estimated_vram:.2f}")
    logger.info(f"  step_law_scaling_ratio: {scaling_ratio:.6f}")
    logger.info(f"  best_proxy_loss: {study.best_value:.4f}")

    # 10. スケーリング前後対比ログ
    logger.info("  [Scaling Trace] Step Law -> Proxy HPO -> Target Scaled")
    for k in ["max_lr_2d", "max_lr_1d", "warmup_ratio", "batch_size_seqs", "weight_decay"]:
        sl = step_law_hpo.get(k)
        proxy = best.get(k)
        final = output["training"].get(k)
        logger.info(f"    {k}: StepLaw={sl} -> Proxy={proxy} -> Final={final}")
    logger.info("=" * 60)

    # 11. メタ情報 JSON 保存
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

    # 12. config.yaml 同期 (--sync-config 指定時)
    if args.sync_config:
        try:
            sync_config_yaml(target_arch, target_size, args.output)
            logger.info(f"Synced configs/config.yaml with {target_size} architecture")
        except Exception as e:
            logger.warning(f"Failed to sync config.yaml: {e}")


if __name__ == "__main__":
    main()

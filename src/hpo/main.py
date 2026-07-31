"""HPO (Hyperparameter Optimization) CLI Entrypoint

Reads configs/chinchilla_config.yaml and runs Step Law + Optuna search.

Usage:
    python -m src.hpo.main n_trials=50
    python -m src.hpo.main proxy=50M seq_len=1024 --apply
"""

import sys
from datetime import datetime
from pathlib import Path

import optuna
import yaml
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from src.common.logger import logger
from src.common.vram_estimator import detect_vram
from src.hpo.hpo_manager import (
    calculate_dynamic_n_trials,
    create_search_space,
    objective,
)
from src.hpo.step_law import compute_hpo_for_target
from src.scaling_laws.calculator import (
    detect_seq_len_from_config,
    generate_universal_architecture,
    load_scaling_config,
)
from src.training.model_utils import (
    get_optimal_num_proc,
    parallel_tokenize,
)


def load_target_arch(path: str = "configs/scaling_config.yaml") -> tuple[dict, int]:
    cfg = load_scaling_config(path)
    model_cfg = cfg.get("model", {})
    llama = model_cfg.get("llama", {})
    arch = {
        "n_params": model_cfg.get("target_params", 86523648),
        "hidden": llama.get("hidden_size", 768),
        "layers": llama.get("num_hidden_layers", 10),
        "heads": llama.get("num_attention_heads", 12),
        "kv_heads": llama.get("num_key_value_heads", 3),
        "ffn": llama.get("intermediate_size", 2048),
        "vocab_size": llama.get("vocab_size", 32000),
    }
    batch_size = cfg.get("training", {}).get("batch_size_seqs", 16)
    return arch, batch_size


def main():
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            args[k.strip().lower()] = v.strip()
        elif arg.startswith("--"):
            args[arg.lstrip("-").lower()] = "true"

    # ---- Load target architecture ----
    scaling_cfg_path = (
        "configs/scaling_config.yaml"
        if Path("configs/scaling_config.yaml").exists()
        else "configs/chinchilla_config.yaml"
    )
    chinchilla_cfg = args.get("scaling_config", args.get("chinchilla_config", scaling_cfg_path))
    target_arch, chinchilla_batch_size = load_target_arch(chinchilla_cfg)
    target_size = f"{target_arch['n_params'] / 1e6:.0f}M"

    # ---- Proxy model ----
    if "proxy" in args:
        proxy_size = args["proxy"]
    else:
        # プロキシはターゲットの約10% (下限 50M)。ターゲットが 50M 未満の場合は
        # ターゲットサイズをそのまま使用 (プロキシがターゲットより大きくなるのを防ぐ)
        proxy_n = min(
            int(target_arch["n_params"]),
            max(50_000_000, int(target_arch["n_params"] * 0.1)),
        )
        proxy_size = f"{proxy_n // 1_000_000}M"
    proxy_n = int(proxy_size.replace("M", "")) * 1_000_000
    proxy_arch = generate_universal_architecture(proxy_n)

    # ---- Resources ----
    data_path = args.get("data_path", "data/dataset.jsonl")
    # 行数ではなく実トークン数を算定 (Step Law はトークン数前提のため)
    from src.scaling_laws.proxy_benchmark import detect_data_path_and_tokens

    _, est_tokens = detect_data_path_and_tokens(data_path=data_path)
    if est_tokens is None or est_tokens <= 0:
        n_tokens = sum(1 for _ in open(data_path, encoding="utf-8"))
        logger.warning(
            f"Token estimation failed for {data_path}. Falling back to line count ({n_tokens:,} lines)."
        )
    else:
        n_tokens = est_tokens
        logger.info(f"Estimated dataset tokens: {n_tokens:,.0f}")
    proxy_vram = float(args.get("vram_gb", 0)) or detect_vram()
    seq_len = int(args.get("seq_len", "0")) or detect_seq_len_from_config()

    # ---- Data prep ----
    logger.info(f"Loading dataset from {data_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
    for attr, val in [
        ("unk_token", "<unk>"),
        ("bos_token", "<s>"),
        ("eos_token", "</s>"),
        ("pad_token", "<pad>"),
    ]:
        setattr(tokenizer, attr, val)

    dataset = load_dataset("json", data_files=str(data_path))
    for split in dataset:
        n_samples = max(1, int(len(dataset[split]) * 0.001))
        if n_samples < len(dataset[split]):
            dataset[split] = dataset[split].select(range(n_samples))

    cols_to_remove = [
        c
        for c in dataset["train"].column_names
        if c not in ["input_ids", "attention_mask", "labels"]
    ]
    num_proc = get_optimal_num_proc()
    tokenized_dataset = parallel_tokenize(
        dataset,
        tokenizer,
        seq_len=seq_len,
        padding=True,
        remove_columns=cols_to_remove,
        max_workers=num_proc,
        batch_size=1000,
    )
    tokenized_dataset.set_format("torch")

    # ---- Step Law ----
    step_law_hpo = compute_hpo_for_target(
        n_params=proxy_arch["n_params"], n_tokens=n_tokens, seq_len=seq_len
    )
    step_law_hpo["batch_size_seqs"] = chinchilla_batch_size

    # ---- Optuna ----
    search_space = create_search_space(step_law_hpo, proxy_vram, n_params=proxy_arch["n_params"])
    user_n_trials = int(args.get("n_trials", "0"))
    n_trials = (
        user_n_trials
        if user_n_trials > 0
        else calculate_dynamic_n_trials(search_space_dim=len(search_space))
    )

    logger.info(
        f"Proxy={proxy_size}, Target={target_size}, Tokens={n_tokens}, VRAM={proxy_vram} GB, SpaceDim={len(search_space)}, Trials={n_trials}"
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5),
    )
    study.set_user_attr("n_trials", n_trials)
    study.optimize(
        lambda trial: objective(
            trial, proxy_arch, tokenized_dataset, seq_len, proxy_vram, step_law_hpo
        ),
        n_trials=n_trials,
    )

    best = study.best_params

    # ---- Scale to target ----
    scaling_ratio = (target_arch["n_params"] / proxy_arch["n_params"]) ** -0.713
    scaled_best = {
        k: round(float(v * scaling_ratio), 6) if k in ["max_lr_2d", "max_lr_1d"] else v
        for k, v in best.items()
    }

    output = {
        "training": {
            **scaled_best,
        }
    }

    # ---- Save ----
    output_path = args.get("output", "configs/hpo_config.yaml")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ====================================================================\n")
        f.write("# HPO 探索結果 (src.hpo.main により自動生成)\n")
        f.write(f"# タイムスタンプ      : {datetime.now().isoformat()}\n")
        f.write(f"# プロキシモデル規模  : ~{proxy_size} (ターゲット: ~{target_size})\n")
        f.write(f"# 総試行数            : {len(study.trials)} trials (枝刈り数: {n_pruned})\n")
        f.write(f"# 最良損失 (Best Loss): {study.best_value:.4f}\n")
        f.write("# ====================================================================\n\n")
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    logger.info(f"HPO config saved to {output_path}")

    print("\n===== HPO Result =====")
    print(f"  Best loss: {study.best_value:.4f}")
    for k, v in output["training"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

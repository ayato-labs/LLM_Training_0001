"""Scaling Laws CLI Entry Point

Usage:
    uv run python -m src.scaling_laws.main [target_hours=24] [--apply]

Examples:
    uv run python -m src.scaling_laws.main
    uv run python -m src.scaling_laws.main target_hours=12 --apply
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.common.logger import logger
from src.scaling_laws.calculator import calculate_chinchilla_scaling


def main():
    target_hours = 24.0
    apply = False

    for arg in sys.argv[1:]:
        if arg.startswith("target_hours="):
            try:
                target_hours = float(arg.split("=")[1])
            except ValueError:
                pass
        elif arg in ("apply=true", "--apply", "-a"):
            apply = True

    logger.info(f"Calculating Scaling Laws & Compute-Optimal Architecture for target_hours={target_hours}...")
    res = calculate_chinchilla_scaling(target_hours=target_hours)

    arch = res["recommended_architecture"]
    data_info = res["dataset_info"]
    hpo_sim = res["hpo_simulation"]

    print("\n" + "=" * 70)
    print(f" Scaling Laws & Compute-Optimal Architecture Report ({target_hours:.1f} Hours Target)")
    print("=" * 70)
    print(f"  GPU Hardware         : {res['gpu_info']['device_name']} (Total VRAM: {res['gpu_info']['total_vram_gb']} GB)")
    print(f"  Measured Throughput  : {res['measured_throughput_tps']} tokens/sec (Source: {res['throughput_source']})")
    print(f"  Total Tokens (D)     : {res['computable_tokens']:,} tokens ({res['computable_tokens_million']:.2f}M)")
    print(f"  Context Length       : {res['seq_len']} tokens")

    print("\n[Recommended Model Architecture (Pure Chinchilla Golden Ratio: D = 20N)]")
    print(f"  Model Parameters (N) : {arch['n_params']:,} (~{arch['n_params']/1e6:.2f}M)")
    print(f"  Hidden Size          : {arch['hidden_size']}")
    print(f"  Num Layers           : {arch['num_hidden_layers']}")
    print(f"  Attention Heads      : {arch['num_attention_heads']} (KV Heads: {arch['num_key_value_heads']})")
    print(f"  Intermediate Size    : {arch['intermediate_size']}")

    print("\n[VRAM & Physical Micro-Batch Split Safety]")
    print(f"  Real Free VRAM       : {res['true_free_vram_gb']} GB")
    print(f"  Estimated Peak VRAM  : {res['estimated_peak_vram_gb']} GB / {res['vram_limit_gb']} GB")
    print(f"  Physical Micro-Batch : per_device_batch_size={res['optimal_batch_size']}, grad_accum_steps={res['grad_accum_steps']} -> Total Batch={res['batch_size_seqs']}")

    print("\n[Dataset Audit]")
    print(f"  Data Path            : {data_info['data_path']}")
    print(f"  Actual Tokens        : {data_info['actual_tokens_str']}")
    print(f"  Sufficiency Ratio    : {data_info['data_sufficiency_ratio_pct']}%")

    if data_info["is_data_shortage"]:
        capped_arch = res["data_capped_architecture"]
        print("\n  [WARN] Dataset tokens are insufficient for pure compute-optimal training!")
        print(f"         Recommended Data-Capped Model Size: {capped_arch['n_params']:,} (~{capped_arch['n_params']/1e6:.2f}M)")

    print("\n[HPO Simulation]")
    print(f"  Proxy Model Size     : {hpo_sim['proxy_params']:,} (~{hpo_sim['proxy_params']/1e6:.2f}M)")
    print(f"  Estimated Search Time: ~{hpo_sim['expected_with_median_pruner_minutes']} mins (MedianPruner) / ~{hpo_sim['worst_case_no_pruning_minutes']} mins (Worst-Case)")
    print("=" * 70 + "\n")

    if apply:
        output_yaml = Path("configs/scaling_config.yaml")
        output_yaml.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "model": {
                "target_params": arch["n_params"],
                "llama": {
                    "hidden_size": arch["hidden_size"],
                    "num_hidden_layers": arch["num_hidden_layers"],
                    "num_attention_heads": arch["num_attention_heads"],
                    "num_key_value_heads": arch["num_key_value_heads"],
                    "intermediate_size": arch["intermediate_size"],
                    "rope_theta": arch["rope_theta"],
                    "vocab_size": arch["vocab_size"],
                    "attn_implementation": arch["attn_implementation"],
                    "tie_word_embeddings": arch["tie_word_embeddings"],
                },
            },
            "training": {
                "batch_size_seqs": res["batch_size_seqs"],
            },
            "metadata": {
                "target_hours": target_hours,
                "gpu": res["gpu_info"]["device_name"],
                "measured_throughput_tps": res["measured_throughput_tps"],
                "estimated_peak_vram_gb": res["estimated_peak_vram_gb"],
                "computable_tokens": res["computable_tokens"],
                "dataset": {
                    "data_path": data_info["data_path"],
                    "actual_tokens": data_info["actual_tokens_str"],
                    "sufficiency_ratio_pct": data_info["data_sufficiency_ratio_pct"],
                    "is_data_shortage": data_info["is_data_shortage"],
                },
                "hpo_simulation": hpo_sim,
            },
        }

        with open(output_yaml, "w", encoding="utf-8") as f:
            f.write("# @package _global_\n")
            f.write("# -----------------------------------------------------------------------------\n")
            f.write("# スケーリング法則 (src.scaling_laws) による自動算定成果物\n")
            f.write(f"# タイムスタンプ: {datetime.now().isoformat()}\n")
            f.write("# -----------------------------------------------------------------------------\n\n")
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Scaling laws configuration successfully saved to {output_yaml}")


if __name__ == "__main__":
    main()

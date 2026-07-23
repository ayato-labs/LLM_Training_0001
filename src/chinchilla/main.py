"""Chinchilla Scaling Laws CLI Entrypoint (Universal Edition)

目標時間 (hours=48 または days=3) から、チンチラ法則および
自動プロファイリング / ログ解析に基づいた汎用最適モデル構造を算出表示する。

使用例:
    python -m src.chinchilla.main hours=48
    python -m src.chinchilla.main days=3
    python -m src.chinchilla.main hours=24 benchmark=true
"""

import sys
from pathlib import Path

from src.common.logger import logger
from src.chinchilla.calculator import (
    calculate_chinchilla_scaling,
    calculate_context_sensitivity_comparison,
)


def print_banner():
    print("=" * 68)
    print("   Universal Chinchilla Scaling Laws Calculator")
    print("=" * 68)


def main():
    print_banner()

    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            args[k.strip().lower()] = v.strip()
        elif arg.startswith("--"):
            args[arg.lstrip("-").lower()] = "true"

    # 1. 目標時間の指定処理 (hours または days)
    target_hours = 48.0
    if "days" in args:
        try:
            target_hours = float(args["days"]) * 24.0
        except ValueError:
            pass
    elif "hours" in args:
        try:
            target_hours = float(args["hours"])
        except ValueError:
            pass

    # 2. 手動スループット指定 or ベンチマーク指定 or コンテキスト長指定
    user_tps = None
    if "throughput" in args or "tps" in args:
        try:
            user_tps = float(args.get("throughput") or args.get("tps"))
        except ValueError:
            pass

    user_seq_len = None
    if "seq_len" in args or "seqlen" in args:
        try:
            user_seq_len = int(args.get("seq_len") or args.get("seqlen"))
        except ValueError:
            pass

    force_bench = args.get("benchmark") == "true" or args.get("bench") == "true"

    # 3. 汎用逆算 ＆ 3段コンテキストトレードオフ比較の実行
    comp_res = calculate_context_sensitivity_comparison(
        target_hours=target_hours,
        user_throughput_tps=user_tps,
        user_seq_len=user_seq_len,
        force_benchmark=force_bench,
    )

    base_seq_len = comp_res["base_seq_len"]
    results = comp_res["comparison_results"]
    target_res = results[1]  # 中央がターゲット基準

    gpu = target_res["gpu_info"]
    arch = target_res["recommended_architecture"]

    print(f"  [GPU Detected]        : {gpu['device_name']} ({gpu['total_vram_gb']} GB VRAM)")
    print(f"  [Throughput Source]   : {target_res['throughput_source']}")
    print(f"  [Measured Throughput] : {target_res['measured_throughput_tps']:,.1f} tokens/sec")
    print(f"  [Target Time]         : {target_res['target_hours']:.1f} hours ({target_res['target_hours']/24.0:.2f} days)")
    print("-" * 68)
    print("  [Target Architecture Configuration]")
    print(f"    - Target Parameters     : ~{arch['n_params']/1e6:.1f}M ({arch['n_params']:,} params)")
    print(f"    - Layers (hidden_layers) : {arch['num_hidden_layers']}")
    print(f"    - Hidden Dim             : {arch['hidden_size']}")
    print(f"    - Attention Heads        : {arch['num_attention_heads']} (head_dim = {arch['head_dim']})")
    print(f"    - Key-Value Heads        : {arch['num_key_value_heads']} (GQA {arch['num_attention_heads']//arch['num_key_value_heads']}:1)")
    print(f"    - Intermediate Size (FFN): {arch['intermediate_size']}")
    print("=" * 68)
    print("  [Context Window Trade-off Comparison (-1 Step / Target / +1 Step)]")
    print("-" * 68)
    print(f"  {'Metric / Context Window':<26} | {'-1 Step (' + str(results[0]['seq_len']) + ')':<12} | {'Target (' + str(results[1]['seq_len']) + ')':<12} | {'+1 Step (' + str(results[2]['seq_len']) + ')':<12}")
    print(f"  {'-'*26}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    vram_str_0 = f"{results[0]['estimated_peak_vram_gb']} GB {'[SAFE]' if results[0]['is_vram_safe'] else '[WARN]'}"
    vram_str_1 = f"{results[1]['estimated_peak_vram_gb']} GB {'[SAFE]' if results[1]['is_vram_safe'] else '[WARN]'}"
    vram_str_2 = f"{results[2]['estimated_peak_vram_gb']} GB {'[SAFE]' if results[2]['is_vram_safe'] else '[WARN]'}"

    steps_str_0 = f"{results[0]['estimated_total_steps']:,} steps"
    steps_str_1 = f"{results[1]['estimated_total_steps']:,} steps"
    steps_str_2 = f"{results[2]['estimated_total_steps']:,} steps"

    time_str_0 = f"~{results[0]['estimated_sec_per_step']}s/it"
    time_str_1 = f"~{results[1]['estimated_sec_per_step']}s/it"
    time_str_2 = f"~{results[2]['estimated_sec_per_step']}s/it"

    print(f"  {'Est Peak VRAM (Reserved)':<26} | {vram_str_0:<12} | {vram_str_1:<12} | {vram_str_2:<12}")
    print(f"  {'Total Steps Required':<26} | {steps_str_0:<12} | {steps_str_1:<12} | {steps_str_2:<12}")
    print(f"  {'Step Processing Time':<26} | {time_str_0:<12} | {time_str_1:<12} | {time_str_2:<12}")
    print("=" * 68)


if __name__ == "__main__":
    main()

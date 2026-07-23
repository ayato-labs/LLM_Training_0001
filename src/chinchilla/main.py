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
from src.chinchilla.calculator import calculate_chinchilla_scaling


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

    # 2. 手動スループット指定 or ベンチマーク指定
    user_tps = None
    if "throughput" in args or "tps" in args:
        try:
            user_tps = float(args.get("throughput") or args.get("tps"))
        except ValueError:
            pass

    force_bench = args.get("benchmark") == "true" or args.get("bench") == "true"

    # 3. 汎用逆算実行
    res = calculate_chinchilla_scaling(
        target_hours=target_hours,
        user_throughput_tps=user_tps,
        force_benchmark=force_bench,
    )

    gpu = res["gpu_info"]
    arch = res["recommended_architecture"]

    print(f"  [GPU Detected]        : {gpu['device_name']} ({gpu['total_vram_gb']} GB VRAM)")
    print(f"  [Throughput Source]   : {res['throughput_source']}")
    print(f"  [Measured Throughput] : {res['measured_throughput_tps']:,.1f} tokens/sec")
    print(f"  [Target Time]         : {res['target_hours']:.1f} hours ({res['target_hours']/24.0:.2f} days)")
    print("-" * 68)
    print("  [Scaling Analysis Results]")
    print(f"    - Total Computable Tokens : {res['computable_tokens_million']:,.1f} Million tokens ({res['computable_tokens_million']*1e6:,.0f} tokens)")
    print(f"    - Pure Chinchilla Optimal : ~{res['chinchilla_pure_optimal_n_million']:,.1f} Million params")
    print(f"    - Total Steps (seq_len=1K): {res['estimated_total_steps']:,} steps (~{res['estimated_sec_per_step']}s / step)")
    print("-" * 68)
    print("  [Recommended Architecture Configuration]")
    print(f"    - Target Parameters     : ~{arch['n_params']/1e6:.1f}M ({arch['n_params']:,} params)")
    print(f"    - Layers (hidden_layers) : {arch['num_hidden_layers']}")
    print(f"    - Hidden Dim             : {arch['hidden_size']}")
    print(f"    - Attention Heads        : {arch['num_attention_heads']} (head_dim = {arch['head_dim']})")
    print(f"    - Key-Value Heads        : {arch['num_key_value_heads']} (GQA {arch['num_attention_heads']//arch['num_key_value_heads']}:1)")
    print(f"    - Intermediate Size (FFN): {arch['intermediate_size']}")
    print("-" * 68)
    status_str = "SAFE (Fits in VRAM)" if res["is_vram_safe"] else "WARNING (Close to VRAM Limit)"
    print(f"  [Estimated Peak VRAM]   : {res['estimated_peak_vram_gb']} GB / {res['vram_limit_gb']} GB [{status_str}]")
    print("=" * 68)


if __name__ == "__main__":
    main()

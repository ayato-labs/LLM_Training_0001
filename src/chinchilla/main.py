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
import yaml

from src.common.logger import logger

from src.chinchilla.calculator import (
    calculate_chinchilla_scaling,
    detect_seq_len_from_config,
)
from src.common.vram_estimator import auto_calibrate


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

    # 0. VRAM calibration (base_config.yaml から seq_len を自動取得)
    cal_label = args.get("cal_label", "proxy")
    cal_seq_len = int(args.get("cal_seq_len", "0")) or detect_seq_len_from_config()
    auto_calibrate(
        hidden_size=512, intermediate_size=1280, num_layers=5,
        seq_len=cal_seq_len, n_params=0,
        n_samples=5, label=cal_label,
    )

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

    # 3. Chinchilla Scaling計算の実行
    target_res = calculate_chinchilla_scaling(
        target_hours=target_hours,
        user_throughput_tps=user_tps,
        user_seq_len=user_seq_len,
        force_benchmark=force_bench,
    )

    gpu = target_res["gpu_info"]
    arch = target_res["recommended_architecture"]

    print(f"  [GPU Detected]        : {gpu['device_name']} ({gpu['total_vram_gb']} GB VRAM)")
    print(f"  [Throughput Source]   : {target_res['throughput_source']}")
    print(f"  [Measured Throughput] : {target_res['measured_throughput_tps']:,.1f} tokens/sec")
    print(f"  [Target Time]         : {target_res['target_hours']:.1f} hours ({target_res['target_hours']/24.0:.2f} days)")
    # データセットのトークン充足度判定・警告表示
    ds_info = target_res["dataset_info"]
    print("  [Dataset Token Sufficiency Audit]")
    print(f"    - Dataset Path          : {ds_info['data_path']}")
    if ds_info["actual_tokens_million"] is not None:
        print(f"    - Dataset Tokens        : ~{ds_info['actual_tokens_million']}M tokens")
        print(f"    - Computable Tokens (D) : ~{target_res['computable_tokens_million']}M tokens")
        print(f"    - Sufficiency Ratio     : {ds_info['sufficiency_ratio']}%")
    else:
        print("    - Dataset Tokens        : (Dataset file not found - Skipped audit)")

    if ds_info["is_data_shortage"]:
        capped_a = target_res["data_capped_architecture"]
        print("-" * 68)
        print("  ⚠️ [WARN / DATA SHORTAGE DETECTED]")
        print("    Target compute requires more tokens than available in dataset!")
        print("    Training target model with insufficient dataset may cause Overfitting.")
        print("    - Overfitting-Safe Model: ~" + f"{capped_a['n_params']/1e6:.1f}M params "
              + f"(Layers: {capped_a['num_hidden_layers']}, Hidden: {capped_a['hidden_size']})")
        print("    - Ideal Compute-Optimal : ~" + f"{arch['n_params']/1e6:.1f}M params "
              + f"(Layers: {arch['num_hidden_layers']}, Hidden: {arch['hidden_size']}) [Sufficient Data Assumed]")
    print("-" * 68)

    print("  [Target Architecture Configuration (Ideal Compute-Optimal)]")
    print(f"    - Target Parameters     : ~{arch['n_params']/1e6:.1f}M ({arch['n_params']:,} params)")
    print(f"    - Layers (hidden_layers) : {arch['num_hidden_layers']}")
    print(f"    - Hidden Dim             : {arch['hidden_size']}")
    print(f"    - Attention Heads        : {arch['num_attention_heads']} (head_dim = {arch['head_dim']})")
    print(f"    - Key-Value Heads        : {arch['num_key_value_heads']} (GQA {arch['num_attention_heads']//arch['num_key_value_heads']}:1)")
    print(f"    - Intermediate Size (FFN): {arch['intermediate_size']}")
    print("=" * 68)

    # 4. HPO 探索時間シミュレーションの表示
    hpo_sim = target_res["hpo_simulation"]
    print("  [HPO Search Time Simulation (Proxy ~10% Size)]")
    print(f"    - Proxy Model Size       : ~{hpo_sim['proxy_params']/1e6:.1f}M params")
    print(f"    - Number of Trials       : {hpo_sim['n_trials']} trials (5D Space)")
    print(f"    - Worst-Case (No Prune)  : ~{hpo_sim['worst_case_no_pruning_minutes']:.1f} minutes")
    print(f"    - Expected (With Pruning): ~{hpo_sim['expected_with_median_pruner_minutes']:.1f} minutes (MedianPruner)")
    print("=" * 68)

    # 5. --apply フラグ指定時に configs/chinchilla_config.yaml 単一ファイルへ成果物・メタデータを完全統合
    should_apply = args.get("apply") == "true" or "--apply" in sys.argv
    if should_apply:
        config_path = Path("configs/chinchilla_config.yaml")
        try:
            # データ量情報文字列の準備
            ds_text = (
                f"~{ds_info['actual_tokens_million']}M tokens"
                if ds_info['actual_tokens_million'] is not None
                else "Unknown"
            )
            data_warn_comment = (
                f"# ⚠️ [WARN] データ不足検知 (充足率: {ds_info['sufficiency_ratio']}%). "
                f"過学習防止モデル規模: ~{target_res['data_capped_architecture']['n_params']/1e6:.1f}M\n"
                if ds_info['is_data_shortage'] else ""
            )

            yaml_content = f"""# @package _global_
# -----------------------------------------------------------------------------
# チンチラの法則 (src.chinchilla) による目標時間逆算成果物
# -----------------------------------------------------------------------------
{data_warn_comment}
# [モデル構造設定] 目標時間 ({target_res['target_hours']}時間) で Loss が最も低くなる黄金比率アーキテクチャ
model:
  target_params: {arch['n_params']}  # 推定パラメータ数 (~{arch['n_params']/1e6:.1f}M)
  llama:
    hidden_size: {arch['hidden_size']}  # 隠れ層ベクトル次元
    num_hidden_layers: {arch['num_hidden_layers']}  # Transformerレイヤー数
    num_attention_heads: {arch['num_attention_heads']}  # Attention Head数 (head_dim = {arch.get('head_dim', 64)})
    num_key_value_heads: {arch['num_key_value_heads']}  # Key/Value Head数 (GQA 4:1)
    intermediate_size: {arch['intermediate_size']}  # SwiGLU FFN中間層次元
    rope_theta: {arch.get('rope_theta', 10000.0)}  # RoPE位置エンコーディング基準周波数
    vocab_size: {arch.get('vocab_size', 32000)}  # 語彙数
    attn_implementation: "{arch.get('attn_implementation', 'sdpa')}"  # PyTorch SDPA バックエンド
    tie_word_embeddings: {str(arch.get('tie_word_embeddings', True)).lower()}  # 埋め込み層とLM Headの重み共有

# [学習実行制御] VRAM容量限界まで計算速度（Tokens/sec）を最大化する物理最速バッチサイズ
training:
  batch_size_seqs: {target_res['optimal_batch_size']}  # GPU最高速度バッチサイズ (b_max)

# [計算環境・データセット監査および HPO 探索時間シミュレーション (参考メタデータ)]
metadata:
  target_hours: {target_res['target_hours']}  # 目標学習時間 (hours)
  gpu: "{gpu['device_name']}"  # 検出されたGPU型番
  measured_throughput_tps: {target_res['measured_throughput_tps']}  # 実測スループット (tokens/sec)
  estimated_peak_vram_gb: {target_res['estimated_peak_vram_gb']}  # 訓練時推定 Peak VRAM (GB)
  computable_tokens: {target_res['computable_tokens']}  # Chinchilla理論計算可能トークン数 D=20N
  dataset:
    data_path: "{ds_info['data_path']}"  # 使用データセットパス
    actual_tokens: "{ds_text}"  # データセット内実測トークン数
    sufficiency_ratio_pct: {ds_info['sufficiency_ratio']}  # 必要トークン数に対するデータ充足率 (%)
    is_data_shortage: {str(ds_info['is_data_shortage']).lower()}  # データ不足警告フラグ
  hpo_simulation:
    proxy_params: {hpo_sim['proxy_params']}  # 探索に使用するプロキシモデル規模 (~10%)
    n_trials: {hpo_sim['n_trials']}  # 5次元探索の標準試行回数 (30 trials/dim)
    worst_case_no_pruning_minutes: {hpo_sim['worst_case_no_pruning_minutes']}  # 枝刈りゼロ時の最悪総探索時間 (分)
    expected_with_median_pruner_minutes: {hpo_sim['expected_with_median_pruner_minutes']}  # MedianPruner適用時の期待探索時間 (分)
"""

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            logger.info(
                f"Successfully applied Chinchilla recommended architecture to {config_path} "
                f"(target_params={arch['n_params']:,}, computable_tokens={target_res['computable_tokens']:,})"
            )
            print(f"\n[APPLIED] Successfully saved Chinchilla recommendation to {config_path}!")
        except Exception as e:
            logger.error(f"Failed to apply Chinchilla config to {config_path}: {e}")
    else:
        print("\n[NOTE] Run with 'apply=true' or '--apply' to automatically update configs/chinchilla_config.yaml")







if __name__ == "__main__":
    main()


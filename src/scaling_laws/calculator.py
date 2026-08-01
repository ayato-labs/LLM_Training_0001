"""Unified Scaling Laws Calculator & Orchestrator

Chinchilla (Hoffmann et al.), Critical Batch Size (Kaplan/OpenAI),
および Dynamic Proxy Benchmark を統括するファサード (Orchestrator) モジュール。
"""

from pathlib import Path
from typing import Any

import yaml

from src.common.logger import logger
from src.common.vram_estimator import (
    VramConfig,
    VramEstimate,
    estimate_training_vram_with_calibration,
)


def load_scaling_config(config_path: str = "configs/scaling_config.yaml") -> dict[str, Any]:
    """configs/scaling_config.yaml を SSOT としてロードする共通関数。"""
    path_obj = Path(config_path)
    if not path_obj.exists():
        return {}

    try:
        with open(path_obj, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.warning(f"Could not load scaling config from {config_path}: {e}")
        return {}


from src.hpo.hpo_manager import calculate_dynamic_n_trials
from src.scaling_laws.chinchilla_law import (
    calculate_compute_optimal_n_d_from_flops,
    generate_universal_architecture,
)
from src.scaling_laws.critical_batch_law import (
    select_decoupled_batch_split,
)
from src.scaling_laws.proxy_benchmark import (
    detect_data_path_and_tokens,
    detect_gpu_info,
    detect_real_available_vram_gb,
    detect_seq_len_from_config,
    extract_throughput_from_recent_logs,
    run_quick_proxy_benchmark,
)


def _vram_estimate(
    n_params: int,
    seq_len: int,
    batch_size: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    vocab_size: int = 32000,
    intermediate_size: int = 0,
    vram_cap: float = 4.0,
    checkpointing: str = "full",
    use_liger_kernel: bool = True,
    optimizer_type: str = "adamw_bnb_8bit",
) -> VramEstimate:
    eff_intermediate = intermediate_size if intermediate_size > 0 else 4 * hidden_size
    return estimate_training_vram_with_calibration(
        VramConfig(
            n_params=n_params,
            hidden_size=hidden_size,
            intermediate_size=eff_intermediate,
            num_layers=num_layers,
            vocab_size=vocab_size,
            micro_batch_size=batch_size,
            seq_len=seq_len,
            precision="bf16",
            optimizer_type=optimizer_type,
            use_liger_kernel=use_liger_kernel,
            checkpointing=checkpointing,
            total_vram_gb=vram_cap,
        )
    )


def _max_n_within_vram(
    seq_len: int,
    true_free_vram_gb: float,
    checkpointing: str,
    use_liger: bool,
    optim_type: str,
    upper_bound_n: int,
    safe_ratio: float = 0.90,
) -> int:
    """VRAM制約 (safe_ratio × 空きVRAM) 以内に収まる最大のモデル規模 N を二分探索で算出。

    micro_batch_size=1 (勾配蓄積でどこまでも分割可能) を前提とするため、
    これがVRAMに収まらないモデルは物理的に実行不可能。
    select_decoupled_batch_split と同じ safe_ratio (0.90) を使い、
    二分探索が承認したアーキが後段で拒否されないことを保証する。
    """
    safe_limit_gb = true_free_vram_gb * safe_ratio

    def estimate_for(n: int) -> float:
        arch = generate_universal_architecture(int(n))
        return _vram_estimate(
            arch["n_params"],
            seq_len,
            1,
            arch["hidden_size"],
            arch["num_hidden_layers"],
            arch["vocab_size"],
            arch.get("intermediate_size", 0),
            checkpointing=checkpointing,
            use_liger_kernel=use_liger,
            optimizer_type=optim_type,
        ).breakdown.total_estimated_gb

    lo, hi = 1, max(1, int(upper_bound_n))
    best_n = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if estimate_for(mid) <= safe_limit_gb:
            best_n = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # 線形微調整: 量子化アーキテクチャの階段に沿って +2% 刻みで詰め、
    # VRAM 上限を最大限活用できる最大 N に収束させる (best_n は既に feasible)。
    n_try = best_n + max(1, int(best_n * 0.02))
    for _ in range(30):
        if n_try > upper_bound_n or estimate_for(n_try) > safe_limit_gb:
            break
        best_n = n_try
        n_try += max(1, int(best_n * 0.02))

    if estimate_for(best_n) > safe_limit_gb:
        raise ValueError(
            f"VRAM infeasibility: even the smallest model exceeds the safe VRAM limit "
            f"({safe_limit_gb:.2f} GB) on this GPU (free VRAM: {true_free_vram_gb:.2f} GB). "
            f"Free up VRAM or lower the target."
        )
    return best_n


def find_max_safe_batch_size(
    arch_dict: dict,
    seq_len: int,
    true_free_vram_gb: float,
    max_search_bs: int = 64,
    checkpointing: str = "full",
) -> int:
    micro_bs, _, _ = select_decoupled_batch_split(
        arch_dict, seq_len, true_free_vram_gb, checkpointing=checkpointing
    )
    return micro_bs


def calculate_chinchilla_scaling(
    target_hours: float = 48.0,
    user_throughput_tps: float | None = None,
    user_seq_len: int | None = None,
    user_vram_limit_gb: float | None = None,
    force_benchmark: bool = False,
    reference_n_params: int | None = None,
) -> dict[str, Any]:
    """与えられた目標学習時間 (target_hours) から可習得トークン数 D = 20N および
    最適な Chinchilla アーキテクチャを完全自律算定するメイン・ファンクション。
    base_config.yaml の設定（gradient_checkpointing 等）を考慮する。
    """
    from src.common.config_utils import load_base_config_yaml

    base_cfg = load_base_config_yaml()
    use_grad_ckpt = base_cfg.get("gradient_checkpointing", True)
    ckpt_mode = "full" if use_grad_ckpt else "none"
    use_liger = base_cfg.get("use_liger_kernel", True)
    optim_type = base_cfg.get("training", {}).get("optim", base_cfg.get("optim", "adamw_bnb_8bit"))

    gpu_info = detect_gpu_info()
    seq_len = user_seq_len or detect_seq_len_from_config()

    # スループットの実測・抽出
    tps = user_throughput_tps
    tp_source = "user_input"

    if tps is None and not force_benchmark:
        # ログからのスループット抽出には実測 seq_len/batch_size が必要
        tps = extract_throughput_from_recent_logs(seq_len=seq_len, batch_size=16)
        if tps is not None:
            tp_source = "recent_logs"

    ref_n = reference_n_params or 27_450_000

    if tps is None:
        logger.info(f"Running GPU proxy benchmark for seq_len={seq_len}, batch_size=16...")
        bench_tps, bench_n = run_quick_proxy_benchmark(seq_len=seq_len, batch_size=16)
        tps = bench_tps
        ref_n = bench_n
        tp_source = "proxy_benchmark"
        logger.info(
            f"GPU proxy benchmark result: {bench_tps:.1f} tokens/sec (reference params={bench_n:,})"
        )

    # 1. 時間予算から総計算量 C (FLOPs) を算定
    #    トークン/sec はモデル規模に依存するため、ハードウェアの実効FLOPsレート
    #    (実測tps x プロキシ規模 x 1トークンあたりFLOPs) をモデル非依存の不変量として使う。
    #    Gradient Checkpointing 有効時は 8N FLOPs/token、無効時は 6N FLOPs/token
    seconds = target_hours * 3600.0
    flop_multiplier = 8.0 if use_grad_ckpt else 6.0
    eff_flops_per_sec = flop_multiplier * ref_n * tps
    total_compute_flops = eff_flops_per_sec * seconds

    # 2. チンチラ基本法則 (C = 6ND, D = 20N → N = sqrt(C/120)) による
    #    純粋 Compute-Optimal モデル規模 N, D の導出
    chinchilla_n_pure, chinchilla_d_pure = calculate_compute_optimal_n_d_from_flops(
        total_compute_flops
    )

    # 3. VRAM 物理制約によるモデル規模キャップ (bs=1 で収まる最大 N を探索)
    true_free_vram_gb = detect_real_available_vram_gb()
    max_n_by_vram = _max_n_within_vram(
        seq_len,
        true_free_vram_gb,
        ckpt_mode,
        use_liger,
        optim_type,
        int(chinchilla_n_pure),
    )
    vram_capped_arch = generate_universal_architecture(max_n_by_vram)
    pure_arch_ref = generate_universal_architecture(int(chinchilla_n_pure))
    # 量子化後アーキ同士で比較 (生の N 比較は int 切捨てによる偽陽性が出る)
    vram_shortage_warn = vram_capped_arch["n_params"] < pure_arch_ref["n_params"]

    # 4. 最終推奨規模: データは追加収集可能なため警告のみ。
    #    純粋 Chinchilla 値と VRAM 実測値からの実行可能上限の最小値を採用
    target_n = int(min(chinchilla_n_pure, max_n_by_vram))
    arch = generate_universal_architecture(target_n)
    requested_n_params = target_n
    arch_deviation_pct = round((arch["n_params"] - target_n) / target_n * 100.0, 1)

    # 5. 推奨モデルの学習計画 D (target_hours 内に処理可能なトークン数) を導出。
    #    この D は 20N と等価であり、予測学習時間は target_hours に自己整合する。
    d_plan = total_compute_flops / (flop_multiplier * arch["n_params"])

    # 6. データセット実効トークン数の正確な事前監査 (計画 D に対する充足率)
    data_path, actual_dataset_tokens = detect_data_path_and_tokens()

    data_shortage_warn = False
    data_capped_arch = None
    data_sufficiency_ratio = 100.0

    max_allowed_n_by_data = float("inf")

    if actual_dataset_tokens is not None:
        data_sufficiency_ratio = round((actual_dataset_tokens / d_plan) * 100.0, 1)

        if d_plan > actual_dataset_tokens:
            data_shortage_warn = True
            max_allowed_n_by_data = actual_dataset_tokens / 20.0
            data_capped_arch = generate_universal_architecture(int(max_allowed_n_by_data))

    # 7. 最適アーキテクチャの生成と 2段階分離型バッチサイズの自動決定
    arch = generate_universal_architecture(target_n)

    # 2段階分離型バッチ設計 (Critical Batch Size ＋ VRAM Physical Micro-batch)
    optimal_micro_bs, grad_accum_steps, actual_b_total = select_decoupled_batch_split(
        arch, seq_len, true_free_vram_gb, checkpointing=ckpt_mode
    )

    est_vram = _vram_estimate(
        arch["n_params"],
        seq_len,
        optimal_micro_bs,
        arch["hidden_size"],
        arch["num_hidden_layers"],
        arch["vocab_size"],
        arch.get("intermediate_size", 0),
        checkpointing=ckpt_mode,
        use_liger_kernel=use_liger,
        optimizer_type=optim_type,
    ).breakdown.total_estimated_gb

    # 純粋な Compute-Optimal アーキテクチャの VRAM 必要量 (bs=1 前提・報告用)
    pure_arch = pure_arch_ref
    pure_est_vram = _vram_estimate(
        pure_arch["n_params"],
        seq_len,
        1,
        pure_arch["hidden_size"],
        pure_arch["num_hidden_layers"],
        pure_arch["vocab_size"],
        pure_arch.get("intermediate_size", 0),
        checkpointing=ckpt_mode,
        use_liger_kernel=use_liger,
        optimizer_type=optim_type,
    ).breakdown.total_estimated_gb

    # 推奨モデルの実効スループット・予測学習時間 (FLOPs レートと自己整合)
    projected_target_tps = eff_flops_per_sec / (flop_multiplier * arch["n_params"])
    predicted_training_hours = d_plan / projected_target_tps / 3600.0
    optimal_budget_tokens = 20 * arch["n_params"]
    optimal_budget_training_hours = optimal_budget_tokens / projected_target_tps / 3600.0

    # HPO シミュレーション時間の導出
    # hpo/main.py と同じ規則 (約10%、下限 50M、ターゲット未満にキャップ) でプロキシ規模を算定
    proxy_n = min(int(arch["n_params"]), max(50_000_000, int(arch["n_params"] * 0.10)))
    proxy_arch = generate_universal_architecture(proxy_n)
    n_trials = calculate_dynamic_n_trials(search_space_dim=4)
    proxy_step_time_sec = 0.50
    worst_case_minutes = (n_trials * 50 * proxy_step_time_sec) / 60.0
    expected_median_minutes = worst_case_minutes * 0.35

    effective_tflops = round(eff_flops_per_sec / 1e12, 4)
    vram_limit_gb = gpu_info.get("total_vram_gb", 4.0)

    return {
        "gpu_info": gpu_info,
        "target_hours": target_hours,
        "seq_len": seq_len,
        "measured_throughput_tps": round(tps, 1),
        "throughput_source": tp_source,
        "reference_model_params": ref_n,
        "effective_tflops": effective_tflops,
        "total_compute_flops": round(total_compute_flops),
        "computable_tokens": round(d_plan),
        "computable_tokens_million": round(d_plan / 1e6, 2),
        "dataset_info": {
            "data_path": data_path,
            "actual_dataset_tokens": actual_dataset_tokens,
            "actual_tokens_str": f"~{actual_dataset_tokens / 1e6:.2f}M tokens"
            if actual_dataset_tokens
            else "Unknown",
            "data_sufficiency_ratio_pct": data_sufficiency_ratio,
            "is_data_shortage": data_shortage_warn,
        },
        "chinchilla_pure_optimal_n_million": round(chinchilla_n_pure / 1e6, 2),
        "chinchilla_pure_optimal_d_million": round(chinchilla_d_pure / 1e6, 2),
        "requested_n_params": requested_n_params,
        "arch_deviation_pct": arch_deviation_pct,
        "recommended_architecture": arch,
        "data_capped_architecture": data_capped_arch,
        "is_vram_capped": vram_shortage_warn,
        "vram_capped_architecture": vram_capped_arch,
        "pure_optimal_estimated_peak_vram_gb": round(pure_est_vram, 4),
        "true_free_vram_gb": true_free_vram_gb,
        "estimated_peak_vram_gb": est_vram,
        "vram_limit_gb": vram_limit_gb,
        "is_vram_safe": est_vram <= vram_limit_gb,
        "optimal_batch_size": optimal_micro_bs,
        "per_device_batch_size": optimal_micro_bs,
        "grad_accum_steps": grad_accum_steps,
        "batch_size_seqs": actual_b_total,
        "projected_target_tps": round(projected_target_tps, 1),
        "predicted_training_hours": round(predicted_training_hours, 1),
        "optimal_budget_tokens": round(optimal_budget_tokens),
        "optimal_budget_training_hours": round(optimal_budget_training_hours, 1),
        "hpo_simulation": {
            "proxy_params": proxy_arch["n_params"],
            "n_trials": n_trials,
            "worst_case_no_pruning_minutes": round(worst_case_minutes, 1),
            "expected_with_median_pruner_minutes": round(expected_median_minutes, 1),
        },
    }

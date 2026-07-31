"""Critical Batch Size Scaling Law (McCandlish/Kaplan/OpenAI 2018-2020)

モデル規模 N における限界バッチサイズ (B_crit) のスケーリング計算、
および VRAM 物理制限と GPU MFU (Roofline Model) による 2段階完全分離型バッチ分割モジュール。
"""

from pathlib import Path

import yaml

from src.common.logger import logger


def calculate_critical_batch_size(
    n_params: int,
    seq_len: int,
    config_path: str = "configs/scaling_config.yaml",
) -> int:
    """Kaplan / OpenAI スケーリング法則 (Critical Batch Size Theory) に基づき、
    モデル規模 n_params において最小ステップ・最高収束効率を与える理論最適トータルバッチシーケンス数を算出。

    configs/scaling_config.yaml からモデル規模と基準バッチサイズを SSOT として動的ロード。
    """
    ref_n = 86_523_648
    ref_batch_seqs = 64

    # scaling_config.yaml から動的ロード (SSOT原則)
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                if (
                    "model" in cfg
                    and isinstance(cfg["model"], dict)
                    and "target_params" in cfg["model"]
                ):
                    ref_n = int(cfg["model"]["target_params"])
                if (
                    "training" in cfg
                    and isinstance(cfg["training"], dict)
                    and "batch_size_seqs" in cfg["training"]
                ):
                    ref_batch_seqs = int(cfg["training"]["batch_size_seqs"])
        except Exception as e:
            logger.warning(
                f"Could not load scaling_config.yaml for Critical Batch Size scaling ({e}). Using default reference values."
            )

    ref_tokens = ref_batch_seqs * seq_len

    # べき乗則スケーリング: B_crit = B_ref * (N / N_ref)^0.3
    scale_factor = (n_params / ref_n) ** 0.3
    target_tokens = ref_tokens * scale_factor
    target_seqs = max(1, round(target_tokens / seq_len))
    return target_seqs


def calculate_tensor_core_mfu(test_bs: int, seq_len: int, arch_dict: dict) -> float:
    """物理 Roofline モデルおよび Tensor Core 行列アライメントに基づき MFU 効率係数を動的計算。

    1. Tensor Core 行列アライメント: ワープ境界 (8 または 16 の倍数) で演算器充填率が最大化。
    2. L2 キャッシュ収容性: アクティベーションサイズと L2 キャッシュ容量の物理比率。
    """
    # 1. Tensor Core Warp Alignment (8/16 の倍数)
    if test_bs % 16 == 0:
        alignment_efficiency = 1.00
    elif test_bs % 8 == 0:
        alignment_efficiency = 0.90
    else:
        alignment_efficiency = 0.70

    # 2. L2 Cache vs Activation Memory Pressure (Roofline)
    hidden_size = arch_dict.get("hidden_size", 768)
    activation_bytes = test_bs * seq_len * hidden_size * 2.0  # bfloat16 / float16

    l2_capacity_bytes = 4 * 1024 * 1024  # 4MB L2 Baseline
    if activation_bytes > l2_capacity_bytes:
        cache_factor = max(0.80, l2_capacity_bytes / activation_bytes)
    else:
        cache_factor = 1.00

    return round(alignment_efficiency * cache_factor, 4)


def select_decoupled_batch_split(
    arch_dict: dict,
    seq_len: int,
    true_free_vram_gb: float,
    checkpointing: str = "selective",
    override_total_batch_size: int | None = None,
) -> tuple[int, int, int]:
    """2段階完全分離型バッチ設計 (Decoupled Batch Architecture):
    1. Critical Batch Size スケーリング法則で理論最適トータルバッチ B_total を算定。
    2. VRAM 90% 容量以内で PyTorch 自動微分・勾配蓄積オーバーヘッドを最小化する最速マイクロバッチ b_micro を選定。
       (VRAM に収まる場合は GA=1 となる最大バッチが最速。VRAM 超過時のみ最小の GA へ動的分割)
    3. 勾配蓄積数 GA = max(1, round(B_total / b_micro)) を算定。

    返り値: (b_micro, grad_accum_steps, b_total_actual)

    例外: ValueError — micro_batch_size=1 でも VRAM 安全上限を超過する実現不可能な場合
    """
    from src.common.vram_estimator import VramConfig, estimate_training_vram_with_calibration

    n_params = arch_dict["n_params"]

    # Step 1: スケーリング法則による理論最適トータルバッチサイズ算定
    target_b_total = override_total_batch_size or calculate_critical_batch_size(n_params, seq_len)

    # Step 2: VRAM 物理限界を満たす最速マイクロバッチ選定 (256, 128, 64, 32, 16, 8, 4, 2, 1)
    safe_limit_gb = true_free_vram_gb * 0.90

    max_bs = max(256, target_b_total)
    candidates = []
    curr = 1
    while curr <= max_bs:
        if curr <= target_b_total:
            candidates.append(curr)
        curr *= 2
    if target_b_total not in candidates:
        candidates.append(target_b_total)
    candidates.sort(reverse=True)

    best_micro_bs = None

    for test_bs in candidates:
        est = estimate_training_vram_with_calibration(
            VramConfig(
                n_params=arch_dict["n_params"],
                hidden_size=arch_dict["hidden_size"],
                intermediate_size=arch_dict.get("intermediate_size", 4 * arch_dict["hidden_size"]),
                num_layers=arch_dict["num_hidden_layers"],
                vocab_size=arch_dict["vocab_size"],
                micro_batch_size=test_bs,
                seq_len=seq_len,
                precision="bf16",
                optimizer_type="adamw_bnb_8bit",
                use_liger_kernel=True,
                checkpointing=checkpointing,
                total_vram_gb=true_free_vram_gb,
            )
        )
        est_vram = est.breakdown.total_estimated_gb

        if est_vram <= safe_limit_gb:
            best_micro_bs = test_bs
            break

    if best_micro_bs is None:
        raise ValueError(
            f"VRAM infeasibility: even micro_batch_size=1 exceeds the safe VRAM limit "
            f"({safe_limit_gb:.2f} GB = {true_free_vram_gb:.2f} GB free x 0.90) "
            f"for this architecture (n_params={n_params:,}). "
            f"The model itself cannot be loaded on this GPU. Reduce model size or free up VRAM."
        )

    best_micro_bs = min(best_micro_bs, target_b_total)

    # Step 3: 自動結合 (Gradient Accumulation 計算)
    grad_accum_steps = max(1, round(target_b_total / best_micro_bs))
    actual_b_total = best_micro_bs * grad_accum_steps

    return best_micro_bs, grad_accum_steps, actual_b_total

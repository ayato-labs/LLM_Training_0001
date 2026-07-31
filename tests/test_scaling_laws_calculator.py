import pytest

from src.scaling_laws.calculator import calculate_chinchilla_scaling
from src.scaling_laws.critical_batch_law import select_decoupled_batch_split


def test_scaling_laws_outputs_computable_tokens_not_max_steps():
    """Scaling Laws は max_steps ではなく computable_tokens を返す"""
    result = calculate_chinchilla_scaling(
        target_hours=24.0,
        user_throughput_tps=14338.6,
        user_seq_len=512,
    )

    # 理論値が含まれる
    assert "computable_tokens" in result
    assert result["computable_tokens"] > 0
    assert "computable_tokens_million" in result

    # max_steps は出力されない
    assert "estimated_total_steps" not in result
    assert "reference_total_steps" not in result


def test_scaling_laws_computable_tokens_invariant():
    """computable_tokens が安定して計算されること"""
    result = calculate_chinchilla_scaling(target_hours=24, user_throughput_tps=14338.6)
    assert result["computable_tokens"] > 0


def test_scaling_laws_pure_optimal_follows_chinchilla_from_flops():
    """純粋 Compute-Optimal は C = 6ND, D = 20N → N = sqrt(C/120) に従う"""
    result = calculate_chinchilla_scaling(target_hours=24, user_throughput_tps=14338.6)
    n_pure = result["chinchilla_pure_optimal_n_million"] * 1e6
    d_pure = result["chinchilla_pure_optimal_d_million"] * 1e6

    assert abs(n_pure - (result["total_compute_flops"] / 120.0) ** 0.5) / n_pure < 0.01
    assert abs(d_pure - 20 * n_pure) / (20 * n_pure) < 0.01


def test_scaling_laws_training_plan_is_self_consistent():
    """FLOPs ベース導出により、推奨モデルの予測学習時間は target_hours と自己整合する"""
    result = calculate_chinchilla_scaling(target_hours=24, user_throughput_tps=14338.6)
    assert result["predicted_training_hours"] == pytest.approx(24.0, rel=0.02)
    assert result["computable_tokens"] == pytest.approx(
        result["projected_target_tps"] * 24.0 * 3600.0, rel=0.02
    )


def test_scaling_laws_output_has_required_keys():
    """必須キーが含まれる"""
    result = calculate_chinchilla_scaling(target_hours=24, user_throughput_tps=14338.6)

    required_keys = [
        "gpu_info",
        "target_hours",
        "seq_len",
        "measured_throughput_tps",
        "throughput_source",
        "reference_model_params",
        "effective_tflops",
        "total_compute_flops",
        "computable_tokens",
        "computable_tokens_million",
        "dataset_info",
        "chinchilla_pure_optimal_n_million",
        "chinchilla_pure_optimal_d_million",
        "recommended_architecture",
        "data_capped_architecture",
        "is_vram_capped",
        "vram_capped_architecture",
        "pure_optimal_estimated_peak_vram_gb",
        "true_free_vram_gb",
        "estimated_peak_vram_gb",
        "vram_limit_gb",
        "is_vram_safe",
        "optimal_batch_size",
        "projected_target_tps",
        "predicted_training_hours",
        "optimal_budget_tokens",
        "optimal_budget_training_hours",
        "hpo_simulation",
    ]

    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_scaling_laws_predicted_training_time_consistent():
    """予測学習時間は D / projected_tps と整合する (1000h・VRAMキャップ時)"""
    result = calculate_chinchilla_scaling(
        target_hours=1000.0,
        user_throughput_tps=55520.6,
        user_seq_len=512,
    )

    expected_hours = result["computable_tokens"] / result["projected_target_tps"] / 3600.0
    assert result["predicted_training_hours"] > 0
    assert abs(result["predicted_training_hours"] - expected_hours) < max(
        1.0, expected_hours * 0.01
    )
    assert result["predicted_training_hours"] == pytest.approx(1000.0, rel=0.02)


def test_scaling_laws_recommended_architecture_fits_vram():
    """推奨アーキテクチャは常に現在の VRAM に収まる (回帰: 32GB要求の実現不可能な推奨を防止)"""
    result = calculate_chinchilla_scaling(
        target_hours=1000.0,
        user_throughput_tps=55520.6,
        user_seq_len=512,
    )

    assert result["estimated_peak_vram_gb"] <= result["vram_limit_gb"]
    assert result["is_vram_safe"] is True


def test_scaling_laws_vram_cap_applied_when_pure_optimal_infeasible():
    """純粋な Compute-Optimal が VRAM 実行可能上限を超える場合、
    VRAM-Capped アーキテクチャが推奨値として適用される"""
    result = calculate_chinchilla_scaling(
        target_hours=3000.0,
        user_throughput_tps=55520.6,
        user_seq_len=512,
    )

    pure_n = result["chinchilla_pure_optimal_n_million"] * 1e6
    recommended_n = result["recommended_architecture"]["n_params"]

    if result["is_vram_capped"]:
        assert result["pure_optimal_estimated_peak_vram_gb"] > result["true_free_vram_gb"] * 0.90
        # 推奨は VRAM 実行可能上限そのもの (データ不足による縮小は受けない)
        assert recommended_n < pure_n
        assert recommended_n == result["vram_capped_architecture"]["n_params"]


def test_scaling_laws_data_shortage_is_warning_only():
    """データ不足は警告のみでモデルサイズを縮小しない (データは追加収集可能)"""
    result = calculate_chinchilla_scaling(
        target_hours=1000.0,
        user_throughput_tps=55520.6,
        user_seq_len=512,
    )

    assert result["dataset_info"]["is_data_shortage"] is True
    assert (
        result["recommended_architecture"]["n_params"]
        > result["data_capped_architecture"]["n_params"]
    )


def test_universal_architecture_within_15_percent():
    """generate_universal_architecture の量子化誤差は ±15% 以内 (回帰: -47% の大幅下振れ)"""
    from src.scaling_laws.chinchilla_law import generate_universal_architecture

    for target in (37_000_000, 87_150_000, 617_290_000):
        arch = generate_universal_architecture(target)
        err = (arch["n_params"] - target) / target
        assert abs(err) <= 0.15, (
            f"target={target/1e6:.2f}M -> {arch['n_params']/1e6:.2f}M ({err:+.1%})"
        )


def test_scaling_laws_recommended_not_data_capped_at_20h():
    """20h で recommended が data_capped と偶然一致しないこと (回帰: 46.28M の一致問題)"""
    result = calculate_chinchilla_scaling(
        target_hours=20.0,
        user_throughput_tps=14338.6,
        user_seq_len=512,
    )

    if result["dataset_info"]["is_data_shortage"]:
        assert (
            result["recommended_architecture"]["n_params"]
            != result["data_capped_architecture"]["n_params"]
        )


def test_scaling_laws_arch_deviation_reported():
    """量子化前の要求 N と実アーキの乖離が結果に含まれる"""
    result = calculate_chinchilla_scaling(
        target_hours=24.0,
        user_throughput_tps=14338.6,
        user_seq_len=512,
    )

    assert "requested_n_params" in result
    assert "arch_deviation_pct" in result
    assert abs(result["arch_deviation_pct"]) <= 15.0


def test_decoupled_batch_split_raises_on_infeasible_architecture():
    """bs=1 でも VRAM 上限を超過するアーキテクチャは黙って (1, GA) を返さず ValueError を送出する"""
    infeasible_arch = {
        "n_params": 3_838_967_808,
        "hidden_size": 3328,
        "num_hidden_layers": 26,
        "intermediate_size": 8832,
        "vocab_size": 32000,
    }
    with pytest.raises(ValueError, match="VRAM infeasibility"):
        select_decoupled_batch_split(
            arch_dict=infeasible_arch,
            seq_len=512,
            true_free_vram_gb=3.16,
            checkpointing="full",
        )

from src.scaling_laws.calculator import calculate_chinchilla_scaling


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


def test_scaling_laws_theoretical_d_equals_20n():
    """D = 20 × N の関係が成り立つ"""
    result = calculate_chinchilla_scaling(target_hours=24, user_throughput_tps=14338.6)
    N_pure = result["chinchilla_pure_optimal_n_million"] * 1e6
    D = result["computable_tokens"]
    assert abs(D - 20 * N_pure) / D < 0.01  # 1%以内


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
        "computable_tokens",
        "computable_tokens_million",
        "dataset_info",
        "chinchilla_pure_optimal_n_million",
        "recommended_architecture",
        "data_capped_architecture",
        "true_free_vram_gb",
        "estimated_peak_vram_gb",
        "vram_limit_gb",
        "is_vram_safe",
        "optimal_batch_size",
        "projected_target_tps",
        "hpo_simulation",
    ]

    for key in required_keys:
        assert key in result, f"Missing key: {key}"

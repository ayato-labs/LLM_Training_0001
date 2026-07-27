from src.common.constants import MAX_LR_2D, clip_learning_rates
from src.training.optimizers.muon import get_optimal_newton_schulz_steps


def run_tests():
    print(f"MAX_LR_2D: {MAX_LR_2D}")

    # Test clipping
    lr_2d, lr_1d = clip_learning_rates(0.05, 0.002, source="Test")
    print(f"Clipped LRs (0.05, 0.002) -> 2D: {lr_2d}, 1D: {lr_1d}")
    assert lr_2d == MAX_LR_2D, "2D LR should be clipped to MAX_LR_2D"
    assert lr_1d == 0.0010, "1D LR should be clipped to MAX_LR_1D (0.0010)"

    # Test Newton-Schulz steps
    ns_small = get_optimal_newton_schulz_steps(hidden_size=768, n_params=150_000_000)
    ns_large = get_optimal_newton_schulz_steps(hidden_size=2048, n_params=3_000_000_000)
    print(f"150M model NS steps: {ns_small}")
    print(f"3B model NS steps: {ns_large}")
    assert ns_small == 3
    assert ns_large == 5
    print("ALL VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    run_tests()

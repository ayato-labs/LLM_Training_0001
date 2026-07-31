from tests.test_scaling_laws_calculator import (
    test_scaling_laws_computable_tokens_invariant,
    test_scaling_laws_output_has_required_keys,
    test_scaling_laws_outputs_computable_tokens_not_max_steps,
    test_scaling_laws_theoretical_d_equals_20n,
)

if __name__ == "__main__":
    test_scaling_laws_outputs_computable_tokens_not_max_steps()
    test_scaling_laws_computable_tokens_invariant()
    test_scaling_laws_theoretical_d_equals_20n()
    test_scaling_laws_output_has_required_keys()
    print("ALL SCALING LAWS TESTS PASSED CLEANLY!")

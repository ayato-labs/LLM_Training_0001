"""Calculate search space dimensions for HPO analysis."""
from src.hpo.step_law import compute_hpo_for_target
from src.hpo.hpo_manager import create_search_space

# Replicate the actual run parameters
n_params = 150_000_000
n_tokens = 587932 * 1024  # line_count * 1024
seq_len = 1024

step_law_hpo = compute_hpo_for_target(n_params, n_tokens, seq_len)
print("=== Step Law Initial Values ===")
for k, v in step_law_hpo.items():
    print(f"  {k}: {v}")

space = create_search_space(step_law_hpo, 4.0, n_params=n_params)
print("\n=== Search Space ===")
for k, v in space.items():
    if isinstance(v, list):
        print(f"  {k}: categorical {v} (|options|={len(v)})")
    elif v[2] == "log":
        print(f"  {k}: log-uniform [{v[0]:.6f}, {v[1]:.6f}] (ratio={v[1]/v[0]:.1f}x)")
    else:
        print(f"  {k}: uniform [{v[0]:.6f}, {v[1]:.6f}] (range={v[1]-v[0]:.6f})")

# Dimensionality analysis
n_continuous = sum(1 for v in space.values() if not isinstance(v, list))
n_categorical = sum(1 for v in space.values() if isinstance(v, list))
total_dims = n_continuous + n_categorical
print(f"\n=== Dimensionality ===")
print(f"  Continuous dims: {n_continuous}")
print(f"  Categorical dims: {n_categorical}")
print(f"  Total dims: {total_dims}")

# Recommended trials
import math
# Rule of thumb: 10-20 trials per dimension for TPE
rec_min = total_dims * 10
rec_good = total_dims * 20
print(f"\n=== Recommended Trials ===")
print(f"  Minimum (10x dims): {rec_min}")
print(f"  Good    (20x dims): {rec_good}")
print(f"  Actual configured:  20")
print(f"  Actual completed:   9 (timeout 3600s)")

# Time analysis
trial_time_sec = 400  # ~6.8 min per trial from logs
print(f"\n=== Time Analysis ===")
print(f"  Time per trial:     ~{trial_time_sec}s ({trial_time_sec/60:.1f} min)")
print(f"  Max trials in 1h:   {3600 // trial_time_sec}")
print(f"  Time for {rec_min} trials: {rec_min * trial_time_sec / 3600:.1f}h")
print(f"  Time for {rec_good} trials: {rec_good * trial_time_sec / 3600:.1f}h")

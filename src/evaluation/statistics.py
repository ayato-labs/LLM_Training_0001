"""
Statistical analysis utilities for multi-seed experiment evaluation.
Used for computing confidence intervals, significance tests, and summary statistics.

ADR-020: Evaluation protocol standardization.
"""
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


def compute_summary(values: list[float]) -> dict:
    """
    Compute summary statistics for a list of values.

    Returns:
        dict with mean, std, sem, ci_95_low, ci_95_high, min, max, n
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"error": "empty values"}
    if n == 1:
        return {
            "mean": float(arr[0]),
            "std": 0.0,
            "sem": 0.0,
            "ci_95_low": float(arr[0]),
            "ci_95_high": float(arr[0]),
            "min": float(arr[0]),
            "max": float(arr[0]),
            "n": 1,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(n)
    # 95% CI using t-distribution
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_half = t_crit * sem

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "sem": round(sem, 6),
        "ci_95_low": round(mean - ci_half, 6),
        "ci_95_high": round(mean + ci_half, 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "n": n,
    }


def paired_t_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    Paired t-test between two conditions (e.g., baseline vs. proposed).

    H0: mean(values_a) == mean(values_b)
    """
    arr_a = np.array(values_a, dtype=np.float64)
    arr_b = np.array(values_b, dtype=np.float64)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Sample sizes differ: {len(arr_a)} vs {len(arr_b)}")
    if len(arr_a) < 2:
        return {"error": "need at least 2 paired samples"}

    t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
    diff_mean = float(np.mean(arr_a - arr_b))
    diff_std = float(np.std(arr_a - arr_b, ddof=1))
    n = len(arr_a)

    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "mean_diff": round(diff_mean, 6),
        "std_diff": round(diff_std, 6),
        "n": n,
        "interpretation": (
            "Significant difference" if p_value < alpha else "No significant difference"
        ),
    }


def welch_t_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    Welch's t-test (unequal variance) for unpaired samples.
    """
    arr_a = np.array(values_a, dtype=np.float64)
    arr_b = np.array(values_b, dtype=np.float64)

    if len(arr_a) < 2 or len(arr_b) < 2:
        return {"error": "need at least 2 samples per group"}

    t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)

    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "mean_a": round(float(np.mean(arr_a)), 6),
        "mean_b": round(float(np.mean(arr_b)), 6),
        "n_a": len(arr_a),
        "n_b": len(arr_b),
        "interpretation": (
            "Significant difference" if p_value < alpha else "No significant difference"
        ),
    }


def cohen_d(values_a: list[float], values_b: list[float]) -> dict:
    """
    Cohen's d effect size for two independent groups.
    """
    arr_a = np.array(values_a, dtype=np.float64)
    arr_b = np.array(values_b, dtype=np.float64)

    n_a, n_b = len(arr_a), len(arr_b)
    var_a, var_b = np.var(arr_a, ddof=1), np.var(arr_b, ddof=1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return {"d": 0.0, "magnitude": "negligible"}

    d = float((np.mean(arr_a) - np.mean(arr_b)) / pooled_std)

    # Magnitude interpretation (Cohen, 1988)
    abs_d = abs(d)
    if abs_d < 0.2:
        mag = "negligible"
    elif abs_d < 0.5:
        mag = "small"
    elif abs_d < 0.8:
        mag = "medium"
    else:
        mag = "large"

    return {"d": round(d, 4), "magnitude": mag}


def compare_experiment_groups(
    group_a: dict[str, list[float]],
    group_b: dict[str, list[float]],
    alpha: float = 0.05,
) -> dict:
    """
    Compare two experiment groups across multiple metrics.

    Args:
        group_a: {metric_name: [values_per_seed]}
        group_b: {metric_name: [values_per_seed]}
        alpha: significance level

    Returns:
        dict with per-metric statistical tests
    """
    results = {}
    common_metrics = set(group_a.keys()) & set(group_b.keys())

    for metric in sorted(common_metrics):
        vals_a = group_a[metric]
        vals_b = group_b[metric]

        results[metric] = {
            "summary_a": compute_summary(vals_a),
            "summary_b": compute_summary(vals_b),
            "paired_t_test": paired_t_test(vals_a, vals_b, alpha),
            "cohen_d": cohen_d(vals_a, vals_b),
        }

    return results

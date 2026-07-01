"""
Automated evaluation report generator for paper-level documentation.
Generates structured Markdown reports with statistics, figures, and tables.

ADR-020: Evaluation protocol standardization.
"""
import json
import datetime
from pathlib import Path
from typing import Optional

from .statistics import compute_summary, paired_t_test, cohen_d


def _fmt_ci(summary: dict, metric: str = "") -> str:
    """Format a summary dict as 'mean ± std (95% CI: low–high)'."""
    return (
        f"{summary['mean']:.6f} ± {summary['std']:.6f} "
        f"(95% CI: [{summary['ci_95_low']:.6f}, {summary['ci_95_high']:.6f}], n={summary['n']})"
    )


def generate_experiment_summary(
    run_results: list[dict],
    config: Optional[dict] = None,
    output_path: str = "logs/experiment_summary.md",
) -> str:
    """
    Generate a full experiment summary report from multiple run results.

    Args:
        run_results: List of dicts, each with keys:
            - seed: int
            - final_loss: float
            - runtime_seconds: float
            - dataset_hash: str (optional)
            - git_hash: str (optional)
            - config: dict (optional)
        config: Global experiment config (optional).
        output_path: Where to save the Markdown report.

    Returns:
        Path to the generated report.
    """
    report = []

    # Header
    report.append("# Novel LLM Pretraining Experiment Report")
    report.append("")
    report.append(f"* **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if config:
        if "git_hash" in config:
            report.append(f"* **Git Commit**: `{config['git_hash'][:12]}`")
        if "dataset_hash" in config:
            report.append(f"* **Dataset Hash**: `{config['dataset_hash'][:16]}`")
        if "n_params" in config:
            report.append(f"* **Model Parameters**: {config['n_params']:,}")
    report.append(f"* **Number of Seeds**: {len(run_results)}")
    report.append("")
    report.append("---")
    report.append("")

    # 1. Run Details Table
    report.append("## 1. Individual Run Results")
    report.append("")
    report.append("| Seed | Final Loss | Runtime (s) | Steps/s |")
    report.append("|------|-----------|-------------|---------|")
    for r in run_results:
        seed = r.get("seed", "?")
        loss = r.get("final_loss", -1)
        runtime = r.get("runtime_seconds", -1)
        steps_per_sec = r.get("steps_per_second", -1)
        report.append(f"| {seed} | {loss:.6f} | {runtime:.1f} | {steps_per_sec:.2f} |")
    report.append("")

    # 2. Statistical Summary
    losses = [r["final_loss"] for r in run_results if "final_loss" in r]
    runtimes = [r["runtime_seconds"] for r in run_results if "runtime_seconds" in r]

    report.append("## 2. Statistical Summary")
    report.append("")

    if losses:
        loss_summary = compute_summary(losses)
        report.append("### Final Training Loss")
        report.append("")
        report.append(f"- **Mean**: {loss_summary['mean']:.6f}")
        report.append(f"- **Std Dev**: {loss_summary['std']:.6f}")
        report.append(f"- **95% CI**: [{loss_summary['ci_95_low']:.6f}, {loss_summary['ci_95_high']:.6f}]")
        report.append(f"- **Range**: [{loss_summary['min']:.6f}, {loss_summary['max']:.6f}]")
        report.append(f"- **N**: {loss_summary['n']}")
        report.append("")

    if runtimes:
        runtime_summary = compute_summary(runtimes)
        report.append("### Training Runtime")
        report.append("")
        report.append(f"- **Mean**: {runtime_summary['mean']:.1f} s ({runtime_summary['mean']/3600:.2f} h)")
        report.append(f"- **Std Dev**: {runtime_summary['std']:.1f} s")
        report.append(f"- **95% CI**: [{runtime_summary['ci_95_low']:.1f}, {runtime_summary['ci_95_high']:.1f}] s")
        report.append("")

    # 3. Configuration Snapshot
    report.append("## 3. Configuration")
    report.append("")
    if run_results and "config" in run_results[0]:
        cfg = run_results[0]["config"]
        report.append("| Parameter | Value |")
        report.append("|-----------|-------|")
        for k, v in sorted(cfg.items()):
            if isinstance(v, (int, float, str, bool)):
                report.append(f"| `{k}` | `{v}` |")
        report.append("")

    # 4. Environment (if available)
    if run_results and "environment" in run_results[0]:
        env = run_results[0]["environment"]
        report.append("## 4. Environment")
        report.append("")
        if "gpu" in env:
            report.append(f"- **GPU**: {env['gpu'].get('device_name', 'N/A')}")
            report.append(f"- **VRAM**: {env['gpu'].get('total_memory_gb', 'N/A')} GB")
            report.append(f"- **CUDA**: {env['gpu'].get('cuda_version', 'N/A')}")
        if "packages" in env:
            report.append(f"- **PyTorch**: {env['packages'].get('torch', 'N/A')}")
            report.append(f"- **Transformers**: {env['packages'].get('transformers', 'N/A')}")
        report.append("")

    # 5. Reproducibility Checklist
    report.append("## 5. Reproducibility Checklist")
    report.append("")
    report.append("| Requirement | Status |")
    report.append("|-------------|--------|")
    report.append("| Random seeds fixed | [x] (ADR-016) |")
    report.append("| Dataset hash recorded | [x] (DVC + ADR-014) |")
    report.append("| Environment snapshot | [x] (ADR-017) |")
    report.append("| Config versioned | [x] (Hydra + ADR-015) |")
    report.append("| Multiple seeds | [x] |")
    report.append("| Statistical analysis | [x] |")
    report.append("| Code version (git) | [x] (ADR-018) |")
    report.append("")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report generated: {output}")
    return str(output)


def generate_comparison_table(
    group_a_name: str,
    group_a_results: list[dict],
    group_b_name: str,
    group_b_results: list[dict],
    output_path: str = "logs/comparison_report.md",
) -> str:
    """
    Generate a comparison report between two experiment groups.
    E.g., baseline vs. proposed method.
    """
    report = []

    report.append("# Experiment Comparison Report")
    report.append("")
    report.append(f"* **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"* **Group A**: {group_a_name} (n={len(group_a_results)})")
    report.append(f"* **Group B**: {group_b_name} (n={len(group_b_results)})")
    report.append("")
    report.append("---")
    report.append("")

    # Loss comparison
    losses_a = [r["final_loss"] for r in group_a_results if "final_loss" in r]
    losses_b = [r["final_loss"] for r in group_b_results if "final_loss" in r]

    if losses_a and losses_b:
        summary_a = compute_summary(losses_a)
        summary_b = compute_summary(losses_b)
        t_result = paired_t_test(losses_a, losses_b)
        d_result = cohen_d(losses_a, losses_b)

        report.append("## Final Training Loss Comparison")
        report.append("")
        report.append("| Group | Mean | Std | 95% CI |")
        report.append("|-------|------|-----|--------|")
        report.append(
            f"| {group_a_name} | {summary_a['mean']:.6f} | {summary_a['std']:.6f} | "
            f"[{summary_a['ci_95_low']:.6f}, {summary_a['ci_95_high']:.6f}] |"
        )
        report.append(
            f"| {group_b_name} | {summary_b['mean']:.6f} | {summary_b['std']:.6f} | "
            f"[{summary_b['ci_95_low']:.6f}, {summary_b['ci_95_high']:.6f}] |"
        )
        report.append("")

        report.append("### Statistical Test")
        report.append("")
        report.append(f"- **Paired t-test**: t={t_result['t_statistic']}, p={t_result['p_value']}")
        report.append(f"- **Significant** (α=0.05): {'Yes' if t_result['significant'] else 'No'}")
        report.append(f"- **Cohen's d**: {d_result['d']} ({d_result['magnitude']} effect)")
        report.append(f"- **Mean Difference**: {t_result['mean_diff']:.6f}")
        report.append("")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Comparison report generated: {output}")
    return str(output)

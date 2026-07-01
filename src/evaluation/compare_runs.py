"""
MLflow run comparison CLI tool.
Queries MLflow for experiment runs and generates comparison reports.

Usage:
    python -m src.evaluation.compare_runs --top 5
    python -m src.evaluation.compare_runs --seed-group
    python -m src.evaluation.compare_runs --output logs/comparison.md
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import mlflow


def get_runs(experiment_name: str = "LLM_Training", max_results: int = 100) -> list[dict]:
    """Fetch runs from MLflow tracking server."""
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.", file=sys.stderr)
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    results = []
    for _, row in runs.iterrows():
        run_data = {
            "run_id": row.get("run_id", ""),
            "start_time": str(row.get("start_time", "")),
            "status": row.get("status", ""),
            "metrics": {},
            "params": {},
        }
        # Extract metrics
        for col in row.index:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                run_data["metrics"][metric_name] = row[col]
            elif col.startswith("params."):
                param_name = col.replace("params.", "")
                run_data["params"][param_name] = row[col]
        results.append(run_data)

    return results


def group_by_seed(runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs by seed parameter."""
    groups = {}
    for run in runs:
        seed = run.get("params", {}).get("seed", "unknown")
        if seed not in groups:
            groups[seed] = []
        groups[seed].append(run)
    return groups


def generate_markdown_report(runs: list[dict], title: str = "MLflow Run Comparison") -> str:
    """Generate a Markdown comparison table from MLflow runs."""
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"* **Total runs**: {len(runs)}")
    lines.append("")

    # Summary table
    lines.append("## Runs Summary")
    lines.append("")
    lines.append("| Run ID | Seed | Final Loss | Runtime (s) | Status |")
    lines.append("|--------|------|-----------|-------------|--------|")

    for run in runs:
        run_id = run["run_id"][:8]
        seed = run["params"].get("seed", "?")
        loss = run["metrics"].get("final_train_loss", -1)
        runtime = run["metrics"].get("final_train_runtime", -1)
        status = run["status"]
        lines.append(f"| {run_id} | {seed} | {loss:.6f} | {runtime:.1f} | {status} |")

    lines.append("")

    # Per-seed statistics
    from .statistics import compute_summary
    losses = [r["metrics"]["final_train_loss"] for r in runs if "final_train_loss" in r.get("metrics", {})]
    if losses:
        summary = compute_summary(losses)
        lines.append("## Aggregate Statistics")
        lines.append("")
        lines.append(f"- **Mean Loss**: {summary['mean']:.6f} ± {summary['std']:.6f}")
        lines.append(f"- **95% CI**: [{summary['ci_95_low']:.6f}, {summary['ci_95_high']:.6f}]")
        lines.append(f"- **N**: {summary['n']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MLflow Run Comparison")
    parser.add_argument("--experiment", default="LLM_Training", help="Experiment name")
    parser.add_argument("--top", type=int, default=20, help="Number of recent runs to fetch")
    parser.add_argument("--seed-group", action="store_true", help="Group results by seed")
    parser.add_argument("--output", default="logs/mlflow_comparison.md", help="Output file path")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of Markdown")
    args = parser.parse_args()

    runs = get_runs(args.experiment, max_results=args.top)
    if not runs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(runs)} runs from MLflow.")

    if args.json:
        output = Path(args.json_output if hasattr(args, "json_output") else args.output.replace(".md", ".json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON saved to {output}")
    else:
        if args.seed_group:
            groups = group_by_seed(runs)
            for seed, seed_runs in sorted(groups.items()):
                report = generate_markdown_report(seed_runs, title=f"Seed={seed}")
                print(f"\n--- Seed {seed} ({len(seed_runs)} runs) ---\n")
                print(report)
        else:
            report = generate_markdown_report(runs)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to {output}")


if __name__ == "__main__":
    main()

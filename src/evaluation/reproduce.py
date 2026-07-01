"""
Reproducibility script generator.
Given an MLflow run ID, generates a complete shell script to reproduce
the exact same experiment from scratch.

Usage:
    python -m src.evaluation.reproduce --run-id <run_id>
    python -m src.evaluation.reproduce --run-id <run_id> --output scripts/reproduce.sh
    python -m src.evaluation.reproduce --list-recent 10
"""
import argparse
import json
import os
import sys
import subprocess
import datetime
from pathlib import Path

import mlflow


def get_run_info(run_id: str) -> dict:
    """Fetch complete run information from MLflow."""
    mlflow.set_tracking_uri("file:./mlruns")

    run = mlflow.get_run(run_id)
    if run is None:
        raise ValueError(f"Run '{run_id}' not found in MLflow.")

    info = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "start_time": run.info.start_time,
        "status": run.info.status,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
        "artifact_uri": run.info.artifact_uri,
    }
    return info


def list_recent_runs(n: int = 10) -> list[dict]:
    """List the N most recent runs with their key parameters."""
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name("LLM_Training")
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=n,
        order_by=["start_time DESC"],
    )

    results = []
    for _, row in runs.iterrows():
        results.append({
            "run_id": row.get("run_id", ""),
            "start_time": str(row.get("start_time", "")),
            "status": row.get("status", ""),
            "seed": row.get("params.seed", "?"),
            "loss": row.get("metrics.final_train_loss", -1),
            "git_hash": row.get("params.git_hash", "?"),
            "dataset_hash": row.get("params.dataset.sha256", "?")[:16] if row.get("params.dataset.sha256") else "?",
        })
    return results


def generate_reproduce_script(run_info: dict, output_path: str = None) -> str:
    """
    Generate a shell script to reproduce the experiment.

    The script includes:
    1. Git checkout to the recorded commit
    2. Environment setup (optional)
    3. DVC data restore
    4. Training with the exact same parameters
    5. Evaluation
    """
    params = run_info.get("params", {})
    git_hash = params.get("git_hash", "unknown")
    seed = params.get("seed", "42")
    data_path = params.get("data_path", "data/dataset.jsonl")

    # Build config dict
    model_params = {
        "n_params": params.get("model.n_params", "150000000"),
        "hidden_size": params.get("model.hidden_size", "2496"),
        "num_hidden_layers": params.get("model.num_hidden_layers", "2"),
        "num_attention_heads": params.get("model.num_attention_heads", "39"),
    }

    hpo_params = {}
    for k, v in params.items():
        if k.startswith("hpo."):
            hpo_key = k.replace("hpo.", "")
            hpo_params[hpo_key] = v

    lines = []
    lines.append("#!/bin/bash")
    lines.append("# ============================================================")
    lines.append(f"# Reproduce Experiment: Run ID = {run_info['run_id']}")
    lines.append(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Original: {datetime.datetime.fromtimestamp(run_info['start_time']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("# ============================================================")
    lines.append("set -euo pipefail")
    lines.append("")

    # Step 1: Git checkout
    if git_hash and git_hash != "unknown":
        lines.append("# Step 1: Checkout the exact code version")
        lines.append(f"echo 'Checking out git commit: {git_hash[:12]}'")
        lines.append(f"git checkout {git_hash}")
        lines.append("")

    # Step 2: Environment check
    lines.append("# Step 2: Verify environment")
    lines.append("echo 'Python version:'")
    lines.append("python --version")
    lines.append("echo 'PyTorch version:'")
    lines.append("python -c \"import torch; print(torch.__version__)\"")
    lines.append("echo 'CUDA available:'")
    lines.append("python -c \"import torch; print(torch.cuda.is_available())\"")
    lines.append("")

    # Step 3: DVC data restore
    lines.append("# Step 3: Restore data from DVC")
    lines.append("echo 'Restoring data from DVC cache...'")
    lines.append("dvc checkout data/dataset.jsonl.dvc data/tokenizer.json.dvc data/corpus.jsonl.dvc")
    lines.append("dvc pull  # requires remote storage configuration")
    lines.append("")

    # Step 4: Generate config file
    lines.append("# Step 4: Generate experiment config")
    run_config = {
        "model_params": model_params,
        "hpo": hpo_params,
        "data_path": data_path,
        "max_steps": params.get("max_steps", "-1"),
        "resume": False,
    }
    config_json = json.dumps(run_config, indent=4)

    lines.append("cat > experiment_config.json << 'CONFIGEOF'")
    lines.append(config_json)
    lines.append("CONFIGEOF")
    lines.append("")

    # Step 5: Run training
    lines.append("# Step 5: Run training")
    lines.append(f"echo 'Starting training with seed={seed}...'")
    lines.append(f"python src/train.py experiment_config.json")
    lines.append("")

    # Step 6: Evaluation
    lines.append("# Step 6: Run evaluation")
    lines.append("python src/eval_inference/evaluate_model.py")
    lines.append("")

    lines.append("echo 'Reproduction complete.'")

    script_content = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(script_content)
        print(f"Reproduce script saved to: {out}")
    else:
        print(script_content)

    return script_content


def generate_reproduce_bat(run_info: dict, output_path: str = None) -> str:
    """Generate a Windows batch script variant."""
    params = run_info.get("params", {})
    git_hash = params.get("git_hash", "unknown")
    seed = params.get("seed", "42")
    data_path = params.get("data_path", "data/dataset.jsonl")

    model_params = {
        "n_params": params.get("model.n_params", "150000000"),
        "hidden_size": params.get("model.hidden_size", "2496"),
        "num_hidden_layers": params.get("model.num_hidden_layers", "2"),
        "num_attention_heads": params.get("model.num_attention_heads", "39"),
    }

    hpo_params = {}
    for k, v in params.items():
        if k.startswith("hpo."):
            hpo_params[k.replace("hpo.", "")] = v

    lines = []
    lines.append("@echo off")
    lines.append("REM ============================================================")
    lines.append(f"REM Reproduce Experiment: Run ID = {run_info['run_id']}")
    lines.append(f"REM Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("REM ============================================================")
    lines.append("")

    # Step 1: Git checkout
    if git_hash and git_hash != "unknown":
        lines.append("REM Step 1: Checkout the exact code version")
        lines.append(f"echo Checking out git commit: {git_hash[:12]}")
        lines.append(f"git checkout {git_hash}")
        lines.append("")

    # Step 2: DVC restore
    lines.append("REM Step 2: Restore data")
    lines.append("echo Restoring data from DVC...")
    lines.append("dvc checkout data\\dataset.jsonl.dvc data\\tokenizer.json.dvc data\\corpus.jsonl.dvc")
    lines.append("dvc pull")
    lines.append("")

    # Step 3: Generate config
    lines.append("REM Step 3: Generate experiment config")
    run_config = {
        "model_params": model_params,
        "hpo": hpo_params,
        "data_path": data_path,
        "max_steps": params.get("max_steps", "-1"),
        "resume": False,
    }
    config_json = json.dumps(run_config, indent=4)

    # Write config via echo commands (Windows compatible)
    lines.append("echo {> experiment_config.json")
    lines.append("echo   \"model_params\": {> _tmp.json")
    lines.append(f'echo     "n_params": {model_params["n_params"]},>> _tmp.json')
    lines.append(f'echo     "hidden_size": {model_params["hidden_size"]},>> _tmp.json')
    lines.append(f'echo     "num_hidden_layers": {model_params["num_hidden_layers"]},>> _tmp.json')
    lines.append(f'echo     "num_attention_heads": {model_params["num_attention_heads"]}>> _tmp.json')
    lines.append("echo   },>> _tmp.json")

    # Simpler approach: write the whole JSON with PowerShell
    lines.append("powershell -Command \"")
    lines.append(f"$config = @{{")
    lines.append(f"  model_params = @{{")
    for k, v in model_params.items():
        lines.append(f'    {k} = {v}')
    lines.append(f"  }}")
    lines.append(f"  hpo = @{{")
    for k, v in hpo_params.items():
        try:
            float(v)
            lines.append(f"    {k} = {v}")
        except ValueError:
            lines.append(f'    {k} = "{v}"')
    lines.append(f"  }}")
    lines.append(f'  data_path = "{data_path}"')
    lines.append(f'  max_steps = "{params.get("max_steps", "-1")}"')
    lines.append(f"  resume = $false")
    lines.append(f"}}")
    lines.append(f"$config | ConvertTo-Json -Depth 5 | Set-Content experiment_config.json -Encoding UTF8\"")
    lines.append("")

    # Step 4: Run training
    lines.append("REM Step 4: Run training")
    lines.append(f"echo Starting training with seed={seed}...")
    lines.append("python src\\train.py experiment_config.json")
    lines.append("")

    # Step 5: Evaluation
    lines.append("REM Step 5: Run evaluation")
    lines.append("python src\\eval_inference\\evaluate_model.py")
    lines.append("")

    lines.append("echo Reproduction complete.")

    script_content = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(script_content)
        print(f"Reproduce script saved to: {out}")

    return script_content


def main():
    parser = argparse.ArgumentParser(description="Experiment Reproducibility Script Generator")
    parser.add_argument("--run-id", help="MLflow run ID to reproduce")
    parser.add_argument("--list-recent", type=int, metavar="N", help="List N most recent runs")
    parser.add_argument("--output", help="Output script path (default: stdout)")
    parser.add_argument("--bat", action="store_true", help="Generate Windows .bat instead of .sh")
    args = parser.parse_args()

    if args.list_recent:
        runs = list_recent_runs(args.list_recent)
        print(f"{'Run ID':<40} {'Seed':<8} {'Loss':<12} {'Git Hash':<14} {'Status':<10}")
        print("-" * 90)
        for r in runs:
            print(f"{r['run_id']:<40} {r['seed']:<8} {r['loss']:<12.6f} {r['git_hash']:<14} {r['status']:<10}")
        return

    if not args.run_id:
        parser.print_help()
        sys.exit(1)

    run_info = get_run_info(args.run_id)

    if args.bat:
        generate_reproduce_bat(run_info, output_path=args.output)
    else:
        generate_reproduce_script(run_info, output_path=args.output)


if __name__ == "__main__":
    main()

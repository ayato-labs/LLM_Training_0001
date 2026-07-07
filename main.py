"""
Novel LLM Pretraining Pipeline Orchestrator
Hydra-compatible entry point with legacy JSON config support.

Usage:
    # Hydra mode (new)
    python main.py --config-name=config
    python main.py --config-name=config training.seq_len=1024 seed=123

    # Legacy mode (backward compat)
    python main.py --legacy --resume
"""
import json
import subprocess
import sys
import os
import argparse
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from LLM_Hyperparameter_Optimization.src.step_law import compute_hpo_for_target

# DataPreprocessing のパス
DATAPREPROCESSING_DIR = PROJECT_ROOT.parent / "DataPreprocessing"
DATAPREPROCESSING_VENV_PYTHON = DATAPREPROCESSING_DIR / ".venv" / "Scripts" / "python.exe"
DEFAULT_DB_PATH = r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Novel_Data_Collection\novels.db"


def run_preprocessing_pipeline(db_path=None, output_path=None):
    """DataPreprocessing パイプラインを subprocess で実行"""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if output_path is None:
        output_path = DATAPREPROCESSING_DIR / "data" / "dataset.jsonl"

    if not DATAPREPROCESSING_VENV_PYTHON.exists():
        raise FileNotFoundError(f"DataPreprocessing venv python not found: {DATAPREPROCESSING_VENV_PYTHON}")

    cmd = [
        str(DATAPREPROCESSING_VENV_PYTHON),
        "-m", "src.cli", "pipeline",
        "--db", str(db_path),
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, cwd=str(DATAPREPROCESSING_DIR), capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"DataPreprocessing pipeline failed:\n{result.stderr}")
    return output_path


# ============================================================
# Legacy config loading (backward compat)
# ============================================================
def load_legacy_config():
    """Load training_config.py constants for legacy mode."""
    try:
        import training_config as config
        return config
    except ImportError:
        # Fallback defaults
        class _Cfg:
            TARGET_PARAMS = 150_000_000
            PROXY_MIN_PARAMS = 30_000_000
            TARGET_TOKENS = 30_000_000
            SEQ_LEN = 512
            MAX_STEPS = -1
            NUM_EPOCHS = 3
            DATA_PATH = Path("data/dataset.jsonl")
            VRAM_LIMIT_GB = 32.0
        return _Cfg()


# ============================================================
# Llama dimension estimation
# ============================================================
def estimate_llama_dimensions(n_params, kv_ratio=4):
    """
    Chinchilla scaling law: N ≈ 12 * L * H^2, H = r * L (r ≈ 64)
    GQA: kv_heads = num_heads // kv_ratio (default: 4 = GQA-8)
    """
    r = 64
    L_opt_raw = (n_params / (12 * (r ** 2))) ** (1/3)
    L_opt = max(8, int(round(L_opt_raw / 2.0) * 2))

    best_L = L_opt
    best_H = 128
    min_diff = float('inf')

    for L in [L_opt - 2, L_opt, L_opt + 2]:
        if L < 8:
            continue
        H_raw = int((n_params / (12 * L)) ** 0.5)
        H = max(64, (H_raw // 64) * 64)
        est = 12 * L * (H ** 2)
        diff = abs(est - n_params)
        if diff < min_diff:
            min_diff = diff
            best_L = L
            best_H = H

    best_heads = max(2, best_H // 64)
    # Ensure heads is divisible by kv_ratio (GQA)
    best_heads = (best_heads // kv_ratio) * kv_ratio
    best_hidden = (best_H // best_heads) * best_heads
    best_kv_heads = max(1, best_heads // kv_ratio)
    return best_hidden, best_L, best_heads, best_kv_heads


# ============================================================
# Experiment runner (proxy model exploration)
# ============================================================
def run_experiment_dynamic(params, tokens, lr, steps, proxy_hidden, proxy_layers, proxy_heads, proxy_kv_heads, seq_len=1024):
    """Run short training with specified params, return final loss."""
    hpo = compute_hpo_for_target(n_params=params, n_tokens=tokens, seq_len=seq_len)
    hpo['max_lr_2d'] = lr

    run_config = {
        "model_params": {
            "n_params": params,
            "hidden_size": proxy_hidden,
            "num_hidden_layers": proxy_layers,
            "num_attention_heads": proxy_heads,
            "num_key_value_heads": proxy_kv_heads,
        },
        "hpo": hpo,
        "data_path": str(config.DATA_PATH),
        "max_steps": steps,
    }

    config_path = Path("experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    subprocess.run([sys.executable, "src/train.py", str(config_path)], check=True)

    with open("last_run_result.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("train_loss", float("inf"))


# ============================================================
# Orchestrator (legacy mode)
# ============================================================
def orchestrate_legacy(args):
    """Run full pipeline: preprocess → HPO → train."""
    config = load_legacy_config()

    # Background uploader
    uploader_script = Path(__file__).resolve().parent / "src" / "utils" / "drive_uploader.py"
    if uploader_script.exists():
        print("Orchestrator: Starting Google Drive uploader in the background...")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = open(log_dir / "drive_uploader.log", "a", encoding="utf-8")
        subprocess.Popen([sys.executable, str(uploader_script)], stdout=log_file, stderr=log_file)

    # Resume mode
    if args.resume:
        config_path = Path("current_run_config.json")
        if not config_path.exists():
            print("Error: 'current_run_config.json' not found.", file=sys.stderr)
            sys.exit(1)
        print("Orchestrator: Resuming from checkpoint...")
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
        run_config["resume"] = True
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)
        subprocess.run([sys.executable, "src/train.py", "current_run_config.json"], check=True)
        return

    # Preprocessing
    print("Orchestrator: Running preprocessing...")
    run_preprocessing_pipeline()

    # Hardware constraints
    n_tokens = config.TARGET_TOKENS
    target_hidden, target_layers, target_heads, target_kv_heads = estimate_llama_dimensions(config.TARGET_PARAMS)
    proxy_params = max(int(config.TARGET_PARAMS * 0.05), getattr(config, "PROXY_MIN_PARAMS", 30_000_000))
    proxy_params = min(proxy_params, config.TARGET_PARAMS)
    proxy_hidden, proxy_layers, proxy_heads, proxy_kv_heads = estimate_llama_dimensions(proxy_params)

    print(f"Orchestrator: Target {config.TARGET_PARAMS} params (H:{target_hidden}, L:{target_layers}, Heads:{target_heads}, KV:{target_kv_heads})")
    print(f"Orchestrator: Proxy {proxy_params} params (H:{proxy_hidden}, L:{proxy_layers}, Heads:{proxy_heads}, KV:{proxy_kv_heads})")

    # Proxy exploration
    base_hpo = compute_hpo_for_target(n_params=proxy_params, n_tokens=n_tokens, seq_len=config.SEQ_LEN)
    base_lr = base_hpo['max_lr_2d']
    candidates = [base_lr * 0.5, base_lr, base_lr * 2.0]

    best_lr = base_lr
    min_loss = float("inf")

    print("Orchestrator: Starting Dynamic Proxy Exploration...")
    for lr in candidates:
        print(f"Testing LR: {lr}")
        loss = run_experiment_dynamic(
            proxy_params, n_tokens, lr, 50,
            proxy_hidden, proxy_layers, proxy_heads, proxy_kv_heads, seq_len=config.SEQ_LEN
        )
        print(f"Loss: {loss}")
        if loss < min_loss:
            min_loss = loss
            best_lr = lr

    print(f"Best LR found: {best_lr}")

    # Extrapolate to production
    final_hpo = compute_hpo_for_target(n_params=config.TARGET_PARAMS, n_tokens=n_tokens, seq_len=config.SEQ_LEN)
    scaling_factor = best_lr / base_lr
    final_hpo['max_lr_2d'] *= scaling_factor
    final_hpo['max_lr_1d'] *= scaling_factor

    # Write final config
    run_config = {
        "model_params": {
            "n_params": config.TARGET_PARAMS,
            "hidden_size": target_hidden,
            "num_hidden_layers": target_layers,
            "num_attention_heads": target_heads,
            "num_key_value_heads": target_kv_heads,
        },
        "hpo": final_hpo,
        "data_path": str(config.DATA_PATH),
        "max_steps": getattr(config, "MAX_STEPS", -1),
        "resume": False,
    }

    with open(Path("current_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    print("Orchestrator: Launching Final Training...")
    subprocess.run([sys.executable, "src/train.py", "current_run_config.json"], check=True)


# ============================================================
# Hydra entry point (new traceability-first mode)
# ============================================================
def run_hydra(cfg):
    """Hydra-driven orchestrator: preprocess → train with full traceability."""
    from omegaconf import DictConfig, OmegaConf

    print("=== Hydra Mode ===")
    print(OmegaConf.to_yaml(cfg))

    # Background uploader
    uploader_script = Path(__file__).resolve().parent / "src" / "utils" / "drive_uploader.py"
    if uploader_script.exists():
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = open(log_dir / "drive_uploader.log", "a", encoding="utf-8")
        subprocess.Popen([sys.executable, str(uploader_script)], stdout=log_file, stderr=log_file)

    # Resume mode
    if cfg.get("resume", False):
        config_path = Path("current_run_config.json")
        if not config_path.exists():
            print("Error: 'current_run_config.json' not found.", file=sys.stderr)
            sys.exit(1)
        print("Orchestrator: Resuming from checkpoint...")
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
        run_config["resume"] = True
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)
        subprocess.run([sys.executable, "src/train.py", "current_run_config.json"], check=True)
        return

    # Preprocessing
    if cfg.get("run_preprocessing", True):
        print("Orchestrator: Running preprocessing...")
        run_preprocessing_pipeline()

    # Model dimension estimation
    target_params = cfg.model.target_params
    target_hidden, target_layers, target_heads, target_kv_heads = estimate_llama_dimensions(target_params)
    print(f"Orchestrator: Target {target_params} params (H:{target_hidden}, L:{target_layers}, Heads:{target_heads}, KV:{target_kv_heads})")

    # HPO via Step Law
    n_tokens = cfg.training.target_tokens
    final_hpo = compute_hpo_for_target(n_params=target_params, n_tokens=n_tokens, seq_len=cfg.training.seq_len)

    # Build Hydra-compatible config for train_model.py
    run_config = OmegaConf.create({
        "model_params": {
            "n_params": target_params,
            "hidden_size": target_hidden,
            "num_hidden_layers": target_layers,
            "num_attention_heads": target_heads,
            "num_key_value_heads": target_kv_heads,
        },
        "hpo": final_hpo,
        "data_path": cfg.data.dataset_path,
        "tokenizer_path": cfg.data.tokenizer_path,
        "max_steps": cfg.training.max_steps,
        "num_epochs": cfg.training.num_epochs,
        "seed": cfg.seed,
        "_hydra_cfg": OmegaConf.to_container(cfg),
    })

    with open("current_run_config.json", "w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(run_config, resolve=True), f, indent=4)

    print("Orchestrator: Launching Final Training...")
    subprocess.run([sys.executable, "src/train.py", "current_run_config.json"], check=True)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Novel LLM Training Pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--legacy", action="store_true", help="Use legacy config (training_config.py)")
    parser.add_argument("--config-name", default="config", help="Hydra config name")
    args, unknown = parser.parse_known_args()

    if args.legacy or args.resume:
        orchestrate_legacy(args)
    else:
        # Hydra mode
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf

            with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
                cfg = compose(config_name=args.config_name, overrides=unknown)
            run_hydra(cfg)
        except ImportError:
            print("Hydra not installed. Falling back to legacy mode.")
            orchestrate_legacy(args)


if __name__ == "__main__":
    main()

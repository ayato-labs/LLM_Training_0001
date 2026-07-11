"""
Config loader: YAML → internal dict.
VRAM detection, precision, and defaults are handled in code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect_vram() -> float:
    """Auto-detect GPU VRAM. Returns 4.0 as fallback."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return round(props.total_memory / (1024**3), 2)
    except Exception:
        pass
    return 4.0


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return as dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config_arg: str | None) -> Path:
    """Resolve config path from CLI argument."""
    if config_arg:
        p = Path(config_arg)
        if not p.is_absolute():
            p = PROJECT_ROOT / config_arg
    else:
        p = PROJECT_ROOT / "configs" / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return p


def load_config(config_path: Path) -> dict:
    """
    Load YAML config and normalize to internal format.

    Returns dict with keys:
        model_params, hpo, data_path, tokenizer_path,
        max_steps, num_epochs, seed, vram_limit_gb, precision
    """
    raw = load_yaml(config_path)
    return normalize_config(raw)


def normalize_config(raw: dict) -> dict:
    """Convert YAML config to internal dict format."""
    # Model params
    model = raw.get("model", {})
    llama = model.get("llama", {})
    model_params = {
        "n_params": model.get("target_params", 150_000_000),
        "hidden_size": llama.get("hidden_size", 768),
        "num_hidden_layers": llama.get("num_hidden_layers", 12),
        "num_attention_heads": llama.get("num_attention_heads", 12),
        "num_key_value_heads": llama.get("num_key_value_heads", 3),
        "intermediate_size": llama.get("intermediate_size", 3072),
        "rope_theta": llama.get("rope_theta", 10000.0),
        "vocab_size": llama.get("vocab_size", 64000),
    }

    # Training / HPO
    t = raw.get("training", {})
    hpo = {
        "seq_len": t.get("seq_len", 1024),
        "max_lr_2d": t.get("max_lr_2d", 3e-4),
        "max_lr_1d": t.get("max_lr_1d", 3e-3),
        "batch_size_seqs": t.get("batch_size_seqs", 16),
        "warmup_ratio": t.get("warmup_ratio", 0.03),
        "min_lr": t.get("min_lr", 1e-5),
    }

    # Hardware (auto-detected)
    vram_limit_gb = detect_vram()
    precision = "bf16"

    return {
        "model_params": model_params,
        "hpo": hpo,
        "data_path": raw.get("data", {}).get("dataset_path", "data/dataset.jsonl"),
        "tokenizer_path": raw.get("data", {}).get("tokenizer_path", "data/tokenizer.json"),
        "max_steps": t.get("max_steps", -1),
        "num_epochs": t.get("num_epochs", 3),
        "seed": raw.get("seed", 42),
        "vram_limit_gb": vram_limit_gb,
        "precision": precision,
    }

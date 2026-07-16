"""Config Loader: Hydra DictConfig → 内部dict 正規化"""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def load_config(cfg: DictConfig) -> dict:
    """
    Hydra合成済みDictConfigをフラットなdictに変換・検証。
    必須キー: model_params, training, data_path, tokenizer_path, output_dir, seed
    """
    # 構造化設定への変換
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # 正規化・デフォルト値補完
    normalized = _normalize_config(container)

    # VRAM自動検出（上書き可能）
    if "vram_limit_gb" not in normalized or normalized["vram_limit_gb"] is None:
        normalized["vram_limit_gb"] = _detect_vram()

    # Precision既定値
    normalized.setdefault("precision", "bf16")

    # 出力ディレクトリ解決
    normalized["output_dir"] = str(Path(normalized.get("output_dir", "models/output")).resolve())

    return normalized


def _normalize_config(raw: dict) -> dict:
    """ネストしたYAML構造を学習パイプライン用フラットdictに変換"""

    # model_params 抽出
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

    # training 抽出（hparams_*.yaml で上書きされる前提）
    t = raw.get("training", {})
    training = {
        "seq_len": t.get("seq_len", 1024),
        "max_lr_2d": t.get("max_lr_2d", 3e-4),
        "max_lr_1d": t.get("max_lr_1d", 3e-3),
        "batch_size_seqs": t.get("batch_size_seqs", 16),
        "warmup_ratio": t.get("warmup_ratio", 0.03),
        "weight_decay": t.get("weight_decay", 0.1),
        "beta2": t.get("beta2", 0.95),
        "grad_clip": t.get("grad_clip", 1.0),
        "per_device_batch_size": t.get("per_device_batch_size", 1),
        "grad_accum_steps": t.get("grad_accum_steps", 1),
        "max_steps": t.get("max_steps", -1),
        "num_epochs": t.get("num_epochs", 3),
        "save_steps": t.get("save_steps", 1000),
        "eval_steps": t.get("eval_steps", 1000),
        "logging_steps": t.get("logging_steps", 10),
    }

    # トップレベルマージ
    return {
        **raw,  # seed, data_path, tokenizer_path, max_steps, num_epochs, etc.
        "model_params": model_params,
        "hpo": training,  # 互換性のため hpo キーも残す
        **training,  # フラットにも展開（TrainingArguments直渡し用）
    }


def _detect_vram() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0

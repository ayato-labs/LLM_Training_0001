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

    # 整合性チェック: config.yaml と hparams_*.yaml のアーキテクチャ一致確認
    _validate_config_consistency(normalized)

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
        "attn_implementation": llama.get("attn_implementation", "sdpa"),
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
        "warmup_steps": t.get("warmup_steps", 0),
    }

    # トップレベルマージ
    return {
        **raw,  # seed, data_path, tokenizer_path, max_steps, num_epochs, etc.
        "model_params": model_params,
        "hpo": training,  # 互換性のため hpo キーも残す
        **training,  # フラットにも展開（TrainingArguments直渡し用）
    }


def _validate_config_consistency(config: dict) -> None:
    """config.yaml と hparams の整合性チェック"""
    from src.common.logger import logger

    # 使用される hparams ファイル名を特定 (config.yaml の defaults から)
    defaults = config.get("defaults", [])
    hparams_name = None
    for d in defaults:
        if isinstance(d, dict) and list(d.keys())[0].startswith("hparams_"):
            hparams_name = list(d.keys())[0]
            break
        elif isinstance(d, str) and d.startswith("hparams_"):
            hparams_name = d
            break

    if not hparams_name:
        logger.warning("No hparams_* found in config.yaml defaults, skipping consistency check")
        return

    # 期待されるターゲットパラメータ数 (config.yaml の model.target_params)
    expected_n_params = config.get("model", {}).get("target_params", 150_000_000)

    # hparams ファイル名からモデルサイズを推定 (hparams_150M -> 150M)
    import re
    match = re.search(r"hparams_(\d+(?:\.\d+)?[MBK]?)", hparams_name)
    if not match:
        logger.debug(f"Could not parse model size from hparams name: {hparams_name}")
        return

    size_str = match.group(1)
    # 簡易的なパース (150M -> 150_000_000)
    size_map = {
        "50M": 50_000_000,
        "150M": 150_000_000,
        "3B": 3_000_000_000,
        "7B": 7_000_000_000,
    }
    expected_from_hparams = size_map.get(size_str)
    if not expected_from_hparams:
        logger.debug(f"Unknown model size in hparams: {size_str}")
        return

    # 不一致チェック (10% 許容)
    if expected_n_params != expected_from_hparams:
        ratio = expected_n_params / expected_from_hparams
        if ratio < 0.9 or ratio > 1.1:
            logger.warning(
                f"Config consistency check: model.target_params ({expected_n_params:,}) "
                f"does not match hparams implied size ({expected_from_hparams:,}) "
                f"from {hparams_name}. Ratio: {ratio:.2f}"
            )
            logger.warning("Consider running HPO with --sync-config or manually updating config.yaml")
        else:
            logger.debug(f"Config consistency OK: target_params={expected_n_params:,} matches hparams {hparams_name}")
    else:
        logger.debug(f"Config consistency OK: target_params={expected_n_params:,} matches hparams {hparams_name}")


def _detect_vram() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0

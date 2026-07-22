"""Config Loader: Hydra DictConfig → 内部dict 正規化"""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def _resolve_wsl_paths(config_obj):
    """Linux(WSL)環境下で、Windows形式の絶対パスをWSL形式に再帰的に解決."""
    import re
    import sys

    if sys.platform == "win32":
        return config_obj

    if isinstance(config_obj, dict):
        return {k: _resolve_wsl_paths(v) for k, v in config_obj.items()}
    elif isinstance(config_obj, list):
        return [_resolve_wsl_paths(x) for x in config_obj]
    elif isinstance(config_obj, str):
        if re.match(r"^[a-zA-Z]:[/\\]", config_obj):
            drive = config_obj[0].lower()
            converted = "/mnt/" + drive + config_obj[2:].replace("\\", "/")
            return converted
    return config_obj


def load_config(cfg: DictConfig) -> dict:
    """
    Hydra合成済みDictConfigをフラットなdictに変換・検証。
    必須キー: model_params, training, data_path, tokenizer_path, output_dir, seed
    """
    # 構造化設定への変換
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # WSL環境下でのWindows絶対パスの自動解決
    container = _resolve_wsl_paths(container)

    # 正規化・デフォルト値補完
    normalized = _normalize_config(container)

    # Linux以外の環境では、Triton制約や安定性のために
    # コンパイルとLiger Kernelを自動無効化
    import sys

    if sys.platform != "linux":
        from src.common.logger import logger

        if normalized.get("torch_compile"):
            logger.info("Non-Linux OS detected. Forcing 'torch_compile' to False.")
            normalized["torch_compile"] = False
            if "hpo" in normalized:
                normalized["hpo"]["torch_compile"] = False
        if normalized.get("use_liger_kernel"):
            logger.info("Non-Linux OS detected. Forcing 'use_liger_kernel' to False.")
            normalized["use_liger_kernel"] = False
            if "hpo" in normalized:
                normalized["hpo"]["use_liger_kernel"] = False

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
    raw_lr_2d = t.get("max_lr_2d", 3e-4)
    raw_lr_1d = t.get("max_lr_1d", 3e-3)

    # 安全上限による自動クリッピング (案C)
    from src.common.constants import clip_learning_rates

    clipped_lr_2d, clipped_lr_1d = clip_learning_rates(raw_lr_2d, raw_lr_1d, source="Config")

    training = {
        "seq_len": t.get("seq_len", 1024),
        "max_lr_2d": clipped_lr_2d,
        "max_lr_1d": clipped_lr_1d,
        "batch_size_seqs": t.get("batch_size_seqs", 16),
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
        "packing": t.get("packing", False),
        "torch_compile": t.get("torch_compile", False),
        "use_liger_kernel": t.get("use_liger_kernel", False),
        "dataloader_pin_memory": t.get("dataloader_pin_memory", True),
        "dataloader_num_workers": t.get("dataloader_num_workers", 0),
        "allow_tf32": t.get("allow_tf32", True),
        "torch_empty_cache_steps": t.get("torch_empty_cache_steps", 100),
        "dataloader_prefetch_factor": t.get("dataloader_prefetch_factor", 2),
        "optim": t.get("optim", "adamw_torch_fused"),
        # LRスケジューラ設定 (Step Law / Muon 推奨)
        "lr_scheduler_type": t.get("lr_scheduler_type", "constant_cosine"),
        "warmup_ratio": t.get("warmup_ratio", 0.03),
        "constant_ratio": t.get("constant_ratio", 0.1),
        "warmup_steps": t.get("warmup_steps", 0),
        "constant_steps": t.get("constant_steps", 0),
        "num_cycles": t.get("num_cycles", 0.5),
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
            logger.warning(
                "Consider running HPO with --sync-config or manually updating config.yaml"
            )
        else:
            logger.debug(
                "Config consistency OK: "
                f"target_params={expected_n_params:,} "
                f"matches hparams {hparams_name}"
            )
    else:
        logger.debug(
            "Config consistency OK: "
            f"target_params={expected_n_params:,} "
            f"matches hparams {hparams_name}"
        )


def _detect_vram() -> float:
    """VRAM検出: ローカルのtorch.cuda使用版（config内部用）。

    サブプロセス呼出しは不要（config読み込み時点では既にCUDA初期化済みの場合があるため）。
    """
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0

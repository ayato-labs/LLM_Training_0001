"""Config Loader: Hydra DictConfig → 内部dict 正規化"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.common.vram_estimator import detect_vram


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
        normalized["vram_limit_gb"] = detect_vram()

    # Precision既定値
    normalized.setdefault("precision", "bf16")

    # 出力ディレクトリ解決
    normalized["output_dir"] = str(Path(normalized.get("output_dir", "models/output")).resolve())

    # 整合性チェック: config.yaml と hparams_*.yaml のアーキテクチャ一致確認
    _validate_config_consistency(normalized)

    return normalized


def _normalize_config(raw: dict) -> dict:
    """ネストしたYAML構造を学習パイプライン用フラットdictに変換"""
    import copy

    raw = copy.deepcopy(raw)

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
        "vocab_size": llama.get("vocab_size", 32000),
        "attn_implementation": llama.get("attn_implementation", "sdpa"),
        "tie_word_embeddings": llama.get("tie_word_embeddings", True),
    }

    # training 抽出（hparams_*.yaml で上書きされる前提）
    t = raw.get("training", {})
    raw_lr_2d = t.get("max_lr_2d", 3e-4)
    raw_lr_1d = t.get("max_lr_1d", 3e-3)

    # 安全上限による自動クリッピング (案C)
    from src.common.constants import clip_learning_rates

    clipped_lr_2d, clipped_lr_1d = clip_learning_rates(raw_lr_2d, raw_lr_1d, source="Config")

    # トータルバッチサイズ (batch_size_seqs) の決定
    # 設定ファイルで未指定の場合は Step Law 推奨値を使用
    if "batch_size_seqs" in t:
        total_batch = t["batch_size_seqs"]
    else:
        from src.hpo.step_law import step_law_optimal_batch

        n_tokens_est = raw.get("n_tokens", 10_000_000)
        seq_len = t.get("seq_len", 1024)
        calculated_tokens = step_law_optimal_batch(n_tokens_est)
        total_batch = max(1, calculated_tokens // seq_len)

    configured_per_device = t.get("per_device_batch_size")

    if configured_per_device is None or configured_per_device == 1:
        from src.training.model_utils import calculate_optimal_batch_split

        n_params = model_params.get("n_params", 150_000_000)
        vram_gb = raw.get("vram_limit_gb", 4.0) or 4.0
        seq_len = t.get("seq_len", 1024)
        per_device_batch, grad_accum = calculate_optimal_batch_split(
            total_batch_size=total_batch,
            vram_gb=vram_gb,
            n_params=n_params,
            seq_len=seq_len,
            hidden_size=model_params.get("hidden_size", 768),
            num_layers=model_params.get("num_hidden_layers", 12),
            precision=raw.get("precision", "bf16"),
            selective_checkpointing=t.get("selective_checkpointing", True),
            optimizer_type=t.get("optim", "adamw_bnb_8bit"),
        )

    else:
        per_device_batch = configured_per_device
        grad_accum = t.get("grad_accum_steps", max(1, total_batch // per_device_batch))

    # === max_steps 動的計算 (Chinchilla対応) ===
    # Chinchilla から metadata.computable_tokens を取得
    metadata = raw.get("metadata", {})
    computable_tokens = metadata.get("computable_tokens")

    # batch_size_seqs を取得（t または raw["training"] から）
    batch_size_seqs = t.get("batch_size_seqs")
    if batch_size_seqs is None:
        batch_size_seqs = total_batch

    seq_len = t.get("seq_len", 1024)

    if computable_tokens and computable_tokens > 0:
        # 理論トークン数 / (バッチ × シーケンス長) = ステップ数
        calculated_max_steps = int(computable_tokens / (batch_size_seqs * seq_len))
        # 設定ファイルに max_steps が明示されていない場合のみ上書き
        if "max_steps" not in t or t["max_steps"] == -1:
            t["max_steps"] = calculated_max_steps
            from src.common.logger import logger

            logger.info(
                f"[Chinchilla] Dynamic max_steps calculated: {calculated_max_steps:,} "
                f"(D={computable_tokens:,}, batch={batch_size_seqs}, seq_len={seq_len})"
            )

    training = {
        "seq_len": seq_len,
        "max_lr_2d": clipped_lr_2d,
        "max_lr_1d": clipped_lr_1d,
        "batch_size_seqs": total_batch,
        "weight_decay": t.get("weight_decay", 0.1),
        "beta2": t.get("beta2", 0.95),
        "grad_clip": t.get("grad_clip", 1.0),
        "per_device_batch_size": per_device_batch,
        "grad_accum_steps": grad_accum,
        "max_steps": t.get("max_steps", -1),
        "num_epochs": t.get("num_epochs", 3),
        "save_steps": t.get("save_steps", 1000),
        "eval_steps": t.get("eval_steps", 1000),
        "logging_steps": t.get("logging_steps", 200),
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
        "warmup_steps": t.get("warmup_steps", 50),
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
    """新 Hydra defaults 構造 (base_config / scaling_config / hpo_config) の整合性チェック"""
    from src.common.logger import logger

    # base_config に model.target_params があるか確認
    base_cfg = config.get("base_config", {})
    expected_n_params = base_cfg.get("model", {}).get("target_params")

    # scaling_config に llama 設定があるか確認
    scaling_cfg = config.get("scaling_config", {})
    llama_cfg = scaling_cfg.get("model", {}).get("llama", {})

    if expected_n_params is None and not llama_cfg:
        logger.warning(
            "Config consistency check: neither base_config.model.target_params "
            "nor scaling_config.model.llama found. Skipping check."
        )
        return

    # scaling_config から推定パラメータ数を計算
    if llama_cfg:
        hidden = llama_cfg.get("hidden_size", 0)
        layers = llama_cfg.get("num_hidden_layers", 0)
        vocab = llama_cfg.get("vocab_size", 0)
        inter = llama_cfg.get("intermediate_size", 0)
        if hidden and layers and vocab and inter:
            # LLaMA 概算: attention(4H^2) + FFN(3HI) + embedding (tied)
            estimated = layers * (4 * hidden * hidden + 3 * hidden * inter) + vocab * hidden
            if expected_n_params is not None:
                ratio = estimated / expected_n_params
                if ratio < 0.8 or ratio > 1.2:
                    logger.warning(
                        f"Config consistency check: scaling_config estimated params "
                        f"({estimated:,}) differs from base_config target_params "
                        f"({expected_n_params:,}) by ratio {ratio:.2f}"
                    )
                else:
                    logger.debug(
                        f"Config consistency OK: estimated={estimated:,} "
                        f"target={expected_n_params:,} ratio={ratio:.2f}"
                    )
            else:
                logger.debug(
                    f"Config consistency: scaling_config estimated params={estimated:,} "
                    f"(no base_config target_params to compare)"
                )

    # hpo_config に training セクションがあるか確認
    hpo_cfg = config.get("hpo_config", {})
    training_cfg = hpo_cfg.get("training", {})
    if not training_cfg:
        logger.debug("Config consistency: no hpo_config.training section found")
    else:
        logger.debug(f"Config consistency: hpo_config.training keys: {list(training_cfg.keys())}")

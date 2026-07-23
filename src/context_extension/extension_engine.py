"""Long-Context Extension Engine (事後長文拡張エンジン)

事前学習済み LLM に対して RoPE Scaling (YaRN/Dynamic NTK) を動的注入し、
選択的 Attention Checkpointing の下で長文コンテキスト適応追加学習 (Continued Pretraining) を実行する。
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.common.logger import logger
from src.training.callbacks import PeriodicEvaluationCallback
from src.training.config import _normalize_config
from src.training.model_utils import (
    PackedDatasetWrapper,
    apply_selective_attention_checkpointing,
    get_optimal_num_proc,
    parallel_tokenize,
)
from src.training.trainer.dual_optimizer_trainer import DualOptimizerTrainer


def load_merged_config(cfg: DictConfig | dict) -> dict:
    """ベース config.yaml および HPO 成果物 (last_run_result.json) を自動マージ・継承"""
    raw_cfg = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg

    # 1. ベース configs/config.yaml のロード
    base_config_path = Path("configs/config.yaml")
    base_cfg = {}
    if base_config_path.exists():
        try:
            base_cfg = OmegaConf.to_container(OmegaConf.load(base_config_path), resolve=True)
            logger.info(f"Loaded base configuration from {base_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load base config.yaml: {e}")

    # 2. HPO成果物 last_run_result.json が存在する場合は統合
    hpo_result_path = Path("last_run_result.json")
    hpo_override = {}
    if hpo_result_path.exists():
        try:
            with open(hpo_result_path, "r", encoding="utf-8") as f:
                hpo_data = json.load(f)
                if "best_params" in hpo_data:
                    hpo_override = hpo_data["best_params"]
                    logger.info(f"Loaded best HPO parameters from {hpo_result_path}: {hpo_override}")
        except Exception as e:
            logger.warning(f"Could not parse {hpo_result_path}: {e}")

    # 3. 設定のマージ（ベース < HPO成果物 < 長文拡張オーバーライド）
    merged = {**base_cfg, **raw_cfg}
    if hpo_override:
        if "hpo" not in merged:
            merged["hpo"] = {}
        merged["hpo"].update(hpo_override)

    # 4. コンテキスト長拡張用の調整 (seq_len と LR)
    target_seq_len = merged.get("target_seq_len", 4096)
    merged["seq_len"] = target_seq_len

    # LRファクターの調整 (拡張時は事前学習LRより抑えて破綻を防ぐ)
    lr_factor = merged.get("extension_lr_factor", 0.2)
    if "hpo" in merged:
        if "max_lr_2d" in merged["hpo"]:
            merged["hpo"]["max_lr_2d"] *= lr_factor
        if "max_lr_1d" in merged["hpo"]:
            merged["hpo"]["max_lr_1d"] *= lr_factor

    res = _normalize_config(merged)
    res["seq_len"] = target_seq_len
    res["target_seq_len"] = target_seq_len
    return res


def prepare_model_and_tokenizer_for_extension(config: dict) -> tuple[Any, Any]:
    """ベースモデルをロードし、RoPE Scaling と Selective Checkpointing を動的注入"""
    base_model_path = config.get("base_model_path", "models/output/checkpoint-latest")
    target_seq_len = config.get("target_seq_len", 4096)
    rope_scaling_cfg = config.get("rope_scaling", {"type": "yarn", "factor": 4.0})

    logger.info(f"Loading base model for context extension from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_path", "data/tokenizer.json"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. モデルロード
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if config.get("precision") == "bf16" else torch.float32,
        trust_remote_code=True,
    )

    # 2. RoPE Scaling (YaRN / Dynamic NTK / Linear) の動的注入
    original_max_len = rope_scaling_cfg.get("original_max_position_embeddings", 1024)
    scaling_type = rope_scaling_cfg.get("type", "yarn")
    scaling_factor = rope_scaling_cfg.get("factor", float(target_seq_len) / float(original_max_len))

    model.config.rope_scaling = {
        "type": scaling_type,
        "factor": scaling_factor,
        "original_max_position_embeddings": original_max_len,
    }
    model.config.max_position_embeddings = target_seq_len
    logger.info(
        f"Injected RoPE Scaling ({scaling_type}, factor={scaling_factor:.2f}) "
        f"target_seq_len: {original_max_len} -> {target_seq_len}"
    )

    # 3. 選択的 Gradient Checkpointing (Attention のみ再計算) の適用
    selective_ckpt = config.get("selective_checkpointing", True)
    if selective_ckpt:
        count = apply_selective_attention_checkpointing(model)
        logger.info(
            f"Applied Selective Attention Checkpointing to {count} layers "
            f"for long-context extension ({target_seq_len} tokens)."
        )

    return model, tokenizer


def run_context_extension(config_input: DictConfig | dict) -> dict:
    """事後長文拡張 Continued Pretraining のメインパイプライン"""
    config = load_merged_config(config_input)

    target_seq_len = config.get("seq_len", 4096)
    max_steps = config.get("extension_max_steps", 2000)
    output_dir = config.get("extension_output_dir", "models/output_context_extension")

    logger.info("=== Starting Long-Context Extension Pipeline ===")
    logger.info(f"Target Sequence Length: {target_seq_len} | Max Steps: {max_steps}")

    # 1. モデルとトークナイザーの準備
    model, tokenizer = prepare_model_and_tokenizer_for_extension(config)

    # 2. 長文用データセットのロードと Packing
    data_files = {"train": config.get("data_path", "data/dataset.jsonl")}
    raw_dataset = load_dataset("json", data_files=data_files)["train"]
    num_proc = get_optimal_num_proc(config)
    tokenized_ds = parallel_tokenize(raw_dataset, tokenizer, num_proc=num_proc)

    packer = PackedDatasetWrapper(
        tokenized_ds,
        seq_len=target_seq_len,
        eos_token_id=tokenizer.eos_token_id,
    )
    packed_dataset = packer()
    logger.info(f"Packed dataset for seq_len={target_seq_len}: {len(packed_dataset)} samples")

    # 3. TrainingArguments の構築
    per_device_batch = config.get("per_device_batch_size", 1)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    selective_ckpt = config.get("selective_checkpointing", True)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.get("hpo", {}).get("max_lr_2d", 6e-5),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=not selective_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=max_steps,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        bf16=(config.get("precision") == "bf16"),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        dataloader_persistent_workers=(config.get("dataloader_num_workers", 0) > 0),
        report_to="none",
    )

    # 4. Trainer の構築と実行
    callbacks = [
        PeriodicEvaluationCallback(
            eval_every_n_steps=config.get("eval_steps", 500),
            divergence_threshold=config.get("divergence_threshold"),
        )
    ]

    trainer = DualOptimizerTrainer(
        model=model,
        args=args,
        train_dataset=packed_dataset,
        callbacks=callbacks,
        split_optimizer_config=config,
    )

    logger.info("Starting Continued Pretraining for Long-Context Extension...")
    train_result = trainer.train()

    # 5. 完成モデルの保存
    final_save_path = Path(output_dir) / "final_extended_model"
    trainer.save_model(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))
    logger.info(f"Long-Context extended model saved to: {final_save_path}")

    return {
        "status": "success",
        "extended_model_path": str(final_save_path),
        "target_seq_len": target_seq_len,
        "total_steps": train_result.global_step,
    }

#!/usr/bin/env python3
"""
LLM Training Entry Point

Usage:
    python -m src.main [OVERRIDES...]

Examples:
    python -m src.main
    python -m src.main training.max_steps=100
    python -m src.main +experiment=debug
"""

from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

from src.config import load_config
from src.drive_uploader import DriveUploadCallback
from src.env_snapshot import capture_env_snapshot
from src.logger import logger, log_exceptions
from src.model_utils import (
    create_model_config,
    estimate_model_size,
    generate_deepspeed_config,
)
from src.set_seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
@log_exceptions
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    logger.info(f"Config resolved: {config}")

    env_snap = capture_env_snapshot()
    logger.debug(f"Env snapshot: {env_snap}")

    set_seed(config["seed"], deterministic=True)

    # Local tokenizer.json を直接読み込み（HF Hub経由させない）
    from tokenizers import Tokenizer as HFTokenizer
    tokenizer_path = Path(config["tokenizer_path"])
    if tokenizer_path.suffix == ".json" and tokenizer_path.exists():
        hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # dummy for interface
        tokenizer._tokenizer = hf_tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

    tokenizer.unk_token = "ᠨ"
    tokenizer.bos_token = "ᠠ"
    tokenizer.eos_token = "ᠡ"
    tokenizer.pad_token = "<pad>"
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})

    train_ds, eval_ds = load_and_tokenize_datasets(config, tokenizer)

    model = create_model(config, tokenizer)
    ds_config = resolve_deepspeed_config(config, model)
    args = build_training_args(config, ds_config)

    callbacks = [DriveUploadCallback(upload_interval_steps=config["drive_upload_interval"])]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )

    logger.info("*** Starting Training ***")
    trainer.train()

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    logger.info(f"Model saved to {config['output_dir']}")
    callbacks[0].force_final_upload(trainer.state.global_step)


def load_and_tokenize_datasets(config: dict, tokenizer):
    data_files = {"train": config["data_path"]}
    if config.get("val_data_path"):
        data_files["validation"] = config["val_data_path"]

    ds = load_dataset("json", data_files=data_files)
    remove_columns = [c for c in ds["train"].column_names if c in {"text", "metadata"}]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config["seq_len"],
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_columns)

    if config.get("data_fraction", 1.0) < 1.0:
        for split in ds:
            n = int(len(ds[split]) * config["data_fraction"])
            ds[split] = ds[split].select(range(n))

    ds.set_format(type="torch")
    return ds["train"], ds.get("validation")


def create_model(config: dict, tokenizer):
    model_config = create_model_config(config, tokenizer)
    model = LlamaForCausalLM(model_config)
    model.resize_token_embeddings(len(tokenizer))
    return model


def build_training_args(config: dict, ds_config_path: str | None):
    """TrainingArguments 構築"""
    return TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["max_lr_1d"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        gradient_checkpointing=True,
        max_steps=config["max_steps"],
        num_train_epochs=config["num_epochs"] if config["max_steps"] == -1 else 0,
        lr_scheduler_type="cosine",
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        adam_beta2=config["beta2"],
        max_grad_norm=config["grad_clip"],
        bf16=config["precision"] == "bf16",
        fp16=config["precision"] == "fp16",
        deepspeed=ds_config_path if ds_config_path else None,
        save_strategy="steps",
        save_steps=config["save_steps"],
        eval_strategy="steps" if config.get("val_data_path") else "no",
        eval_steps=config["eval_steps"] if config.get("val_data_path") else None,
        logging_steps=config["logging_steps"],
        report_to=["tensorboard"],
        load_best_model_at_end=bool(config.get("val_data_path")),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config["seed"],
        remove_unused_columns=False,
    )


def resolve_deepspeed_config(config: dict, model) -> str | None:
    vram_gb = config.get("vram_limit_gb") or detect_vram()
    n_params = estimate_model_size(model)
    return generate_deepspeed_config(n_params, vram_gb, config["precision"])


def detect_vram() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0


if __name__ == "__main__":
    main()

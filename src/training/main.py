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

from src.training.config import load_config
from src.training.drive_uploader import DriveUploadCallback
from src.common.env_snapshot import capture_env_snapshot
from src.common.logger import log_exceptions, logger
from src.training.model_utils import (
    create_model_config,
)
from src.common.set_seed import set_seed
from src.training.train_model import ProgressBarFormatCallback


@hydra.main(version_base=None, config_path="../configs", config_name="config")
@log_exceptions
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    logger.info(f"Config resolved: {config}")

    env_snap = capture_env_snapshot()
    logger.debug(f"Env snapshot: {env_snap}")

    set_seed(config["seed"], deterministic=True)

    # Local tokenizer.json を直接読み込み（HF Hub経由させない）
    tokenizer_path = Path(config["tokenizer_path"])
    if tokenizer_path.suffix == ".json" and tokenizer_path.exists():
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

    # ADR-021: Use SP-native token names (IDs 0-3 in vocab)
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    train_ds, eval_ds = load_and_tokenize_datasets(config, tokenizer)

    model = create_model(config, tokenizer)
    args = build_training_args(config)

    callbacks = [
        DriveUploadCallback(upload_interval_steps=config["drive_upload_interval"]),
        ProgressBarFormatCallback(),
    ]

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


def build_training_args(config: dict):
    """TrainingArguments 構築"""
    precision = config.get("precision", "bf16")
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
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
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


def detect_vram() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0


if __name__ == "__main__":
    main()

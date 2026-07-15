import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainingArguments,
)

from src.common.logger import logger
from src.common.set_seed import set_seed
from src.common.env_snapshot import capture_env_snapshot
from src.training.model_utils import (
    create_model_config,
    TokenizerWrapper,
    get_optimal_num_proc,
    compute_file_hash,
    compute_dataset_fingerprint,
    compute_db_fingerprint,
)
from src.training.trainer import CustomTrainer
from src.training.callbacks import (
    ProgressBarFormatCallback,
    HashSaveCallback,
    DriveUploadCallback,
    DetailedLoggingCallback,
)


def train(config: dict, tokenized_datasets=None, extra_callbacks=None):
    """
    Unified training orchestration flow.
    """
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=True)

    # Env snapshot & dataset fingerprint
    env_snapshot = capture_env_snapshot()
    logger.debug(f"Env snapshot: {env_snapshot}")

    config_path = Path("configs/config.yaml")
    data_path_str = config.get("data_path", "data/dataset.jsonl")
    current_config_hash = compute_file_hash(str(config_path))
    current_data_hash = compute_file_hash(data_path_str)

    data_fingerprint = compute_dataset_fingerprint(data_path_str)
    db_path_str = config.get("db_path", "../Novel_Data_Collection/novels.db")
    db_fingerprint = compute_db_fingerprint(db_path_str)

    logger.info(
        "Training initialization",
        extra={
            "seed": seed,
            "data_path": data_path_str,
            "dataset_fingerprint": data_fingerprint,
            "db_fingerprint": db_fingerprint,
        },
    )

    # Resume from checkpoint resolution & validation
    resume_checkpoint = config.get("resume_from_checkpoint") or config.get("resume")
    if resume_checkpoint is True or (isinstance(resume_checkpoint, str) and "checkpoint-latest" in resume_checkpoint):
        from src.training.drive_uploader import get_checkpoints
        checkpoints = get_checkpoints()
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1][1])
            logger.info(f"Resolved checkpoint-latest to: {resume_checkpoint}")
        else:
            resume_checkpoint = None
            logger.warning("No checkpoint found to resume from. Starting from scratch.")

    if isinstance(resume_checkpoint, str) and Path(resume_checkpoint).exists():
        checkpoint_path = Path(resume_checkpoint)
        hash_file = checkpoint_path / "hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, "r") as f:
                    saved_hashes = json.load(f)
                saved_config_hash = saved_hashes.get("config_hash")
                saved_data_hash = saved_hashes.get("data_hash")
                
                if saved_config_hash != current_config_hash or saved_data_hash != current_data_hash:
                    logger.error("Configuration or training dataset has changed since the checkpoint was saved!")
                    logger.error(f"Saved Config Hash: {saved_config_hash} | Current Config Hash: {current_config_hash}")
                    logger.error(f"Saved Data Hash: {saved_data_hash} | Current Data Hash: {current_data_hash}")
                    raise ValueError("Cannot resume training: config.yaml or training dataset does not match the checkpoint.")
                else:
                    logger.info("Configuration and dataset hashes match. Verification successful.")
            except Exception as e:
                logger.error(f"Failed to verify checkpoint hashes: {e}")
                raise
        else:
            logger.warning(f"No hashes.json found in checkpoint {resume_checkpoint}. Proceeding without verification.")
    else:
        if not isinstance(resume_checkpoint, str):
            resume_checkpoint = None

    # Tokenizer loading
    tokenizer_path = Path(config.get("tokenizer_path", "data/tokenizer.json"))
    if tokenizer_path.suffix == ".json" and tokenizer_path.exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    # Dataset loading & tokenization
    if tokenized_datasets is None:
        logger.info("Starting dataset loading...")
        data_files = {"train": data_path_str}
        if config.get("val_data_path"):
            data_files["validation"] = config["val_data_path"]

        ds = load_dataset("json", data_files=data_files)
        remove_columns = [c for c in ds["train"].column_names if c in {"text", "metadata"}]

        seq_len = config.get("seq_len", 1024)
        tokenize_fn = TokenizerWrapper(tokenizer, seq_len)
        num_proc = get_optimal_num_proc()
        logger.info(f"Tokenizing dataset with num_proc={num_proc} (calculated dynamically from available RAM and CPUs)")

        ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_columns, num_proc=num_proc)

        data_fraction = config.get("data_fraction", 1.0)
        if data_fraction < 1.0:
            for split in ds:
                n = int(len(ds[split]) * data_fraction)
                n = max(1, n)
                ds[split] = ds[split].select(range(n))
                logger.info(f"Split '{split}' sampled to {n} samples (fraction={data_fraction})")

        ds.set_format(type="torch")
        train_ds = ds["train"]
        eval_ds = ds.get("validation")
    else:
        train_ds = tokenized_datasets["train"]
        eval_ds = tokenized_datasets.get("validation")

    # Model creation
    model_config = create_model_config(config, tokenizer)
    model = LlamaForCausalLM(model_config)
    model.resize_token_embeddings(len(tokenizer))

    # TrainingArguments construction
    precision = config.get("precision", "bf16")
    output_dir = config.get("output_dir", "models/output")
    hpo_config = config.get("hpo", config)

    max_steps = config.get("max_steps", -1)
    num_epochs = config.get("num_epochs", 3) if max_steps == -1 else 0

    per_device_batch = config.get("per_device_batch_size", 1)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    
    if "batch_size_seqs" in hpo_config and "per_device_batch_size" not in config:
        target_total_batch_seqs = hpo_config.get("batch_size_seqs", 16)
        vram_limit = config.get("vram_limit_gb", 4.0)
        if vram_limit <= 4.5:
            max_batch = 1
        elif vram_limit <= 8.5:
            max_batch = 4
        else:
            max_batch = 8
        per_device_batch = min(target_total_batch_seqs, max_batch)
        grad_accum_steps = max(1, target_total_batch_seqs // per_device_batch)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=hpo_config.get("max_lr_2d", 3e-4),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=hpo_config.get("warmup_ratio", 0.03),
        weight_decay=hpo_config.get("weight_decay", 0.1),
        adam_beta2=hpo_config.get("beta2", 0.95),
        max_grad_norm=hpo_config.get("grad_clip", 1.0),
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.get("eval_steps", 1000) if eval_ds is not None else None,
        logging_steps=config.get("logging_steps", 10),
        report_to=["tensorboard"],
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        seed=seed,
        remove_unused_columns=False,
    )

    # Callbacks configuration
    drive_cb = DriveUploadCallback(upload_interval_steps=config.get("drive_upload_interval", 1000))
    callbacks = [
        ProgressBarFormatCallback(),
        HashSaveCallback(config_hash=current_config_hash, data_hash=current_data_hash),
        DetailedLoggingCallback(log_every_n_steps=1),
    ]

    if max_steps == -1 or config.get("enable_drive_upload_hpo", False):
        callbacks.append(drive_cb)

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        additional_config=config,
        callbacks=callbacks,
    )

    for cb in callbacks:
        if isinstance(cb, DetailedLoggingCallback):
            cb.trainer = trainer

    logger.info("*** Starting Unified Training Pipeline ***")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    if max_steps == -1:
        with open("last_run_result.json", "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    if drive_cb in callbacks:
        drive_cb.force_final_upload(trainer.state.global_step)

    return train_result.training_loss

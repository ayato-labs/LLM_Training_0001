import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.logger import logger


# ============================================================
# Logging setup
# ============================================================
def is_deepspeed_available():
    try:
        import importlib.metadata

        importlib.metadata.version("deepspeed")
        return True
    except Exception:
        return False


logger.info(f"CUDA available: {torch.cuda.is_available()}")


# ============================================================
# Dataset fingerprinting (traceability)
# ============================================================
def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for traceability."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_fingerprint(dataset_path: str) -> dict:
    """
    Compute a comprehensive fingerprint of the training dataset.
    Returns dict with hash, row count, file size, and modification time.
    """
    path = Path(dataset_path)
    if not path.exists():
        return {"error": f"File not found: {dataset_path}"}

    stat = path.stat()
    # Line count for JSONL
    line_count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            line_count += 1

    return {
        "path": str(path.resolve()),
        "sha256": compute_file_hash(str(path)),
        "size_bytes": stat.st_size,
        "line_count": line_count,
        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def compute_db_fingerprint(db_path: str) -> dict:
    """Compute fingerprint of the source SQLite database."""
    path = Path(db_path)
    if not path.exists():
        return {"error": f"Database not found: {db_path}"}

    import sqlite3

    stat = path.stat()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chapters")
        chapter_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM novels")
        novel_count = cursor.fetchone()[0]
        conn.close()
    except Exception:
        chapter_count = -1
        novel_count = -1

    return {
        "path": str(path.resolve()),
        "sha256": compute_file_hash(str(path)),
        "size_bytes": stat.st_size,
        "chapter_count": chapter_count,
        "novel_count": novel_count,
        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


# ============================================================
# Config normalization: accept internal dict directly
# ============================================================
def normalize_config(raw) -> dict:
    """
    Accept an internal config dict (from src/config.py) and return
    a flat dictionary with standardized keys.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported config type: {type(raw)}")

    # Already normalized (has model_params and hpo)
    if "model_params" in raw and "hpo" in raw:
        return raw

    raise ValueError("Config must have 'model_params' and 'hpo' keys")


# ============================================================
# DeepSpeed config generation (ADR-018: BF16, ADR-019: ZeRO-3)
# ============================================================
def generate_deepspeed_config(n_params, vram_limit_gb, precision="bf16"):
    est_vram_gb = (n_params * 14) / (1024**3)

    # BF16 (推奨) or FP16 (フォールバック)
    if precision == "bf16":
        precision_config = {"bf16": {"enabled": True}}
        precision_name = "bf16"
    else:
        precision_config = {"fp16": {"enabled": True, "loss_scale": 0, "loss_scale_window": 1000}}
        precision_name = "fp16"

    # モデルサイズに基づいてDeepSpeedステージを決定
    # CPUオフロードは極力避ける（学習速度が大幅に低下するため）
    if est_vram_gb > vram_limit_gb * 0.9:
        # VRAMに収まらない場合はZeRO-3 + CPU offload
        zero_config = {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "stage3_param_persistence_threshold": 10000,
            "stage3_max_live_parameters": 1e7,
            "stage3_max_reuse_distance": 1e7,
            "sub_group_size": 1e7,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
        }
        zero_name = "ZeRO-3 (CPU offload)"
    elif vram_limit_gb <= 5.0:
        # VRAM制限が小さいがモデルが収まる場合はZeRO-1（軽量）
        zero_config = {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
        }
        zero_name = "ZeRO-1"
    else:
        # VRAMに余裕がある場合はZeRO-2
        zero_config = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
        }
        zero_name = "ZeRO-2"

    ds_config = {
        **precision_config,
        "zero_optimization": zero_config,
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_of_nodes": 1,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
    }

    if est_vram_gb > vram_limit_gb:
        logger.warning(
            f"DeepSpeed: Est. param memory ({est_vram_gb:.2f} GB) > VRAM ({vram_limit_gb} GB). GPU-only may OOM."
        )
    else:
        logger.info(
            f"DeepSpeed: Est. VRAM ({est_vram_gb:.2f} GB) fits within limit ({vram_limit_gb} GB)."
        )

    logger.info(f"DeepSpeed: {precision_name} + {zero_name} (VRAM limit: {vram_limit_gb} GB)")

    ds_config_path = "temp_ds_config.json"
    with open(ds_config_path, "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)
    return ds_config_path


def get_git_revision_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


# ============================================================
# Google Drive Checkpoint Upload Callback
# ============================================================
class DriveUploadCallback(TrainerCallback):
    """
    Callback to upload checkpoints to Google Drive after saving.
    Integrates with drive_uploader.py logic.
    """

    def __init__(self, upload_interval_steps: int = 1000):
        self.upload_interval_steps = upload_interval_steps
        self.last_uploaded_step = -1
        self.drive_service = None
        self.root_folder_id = None
        self._initialized = False

    def _init_drive(self):
        """Lazy initialize Google Drive service."""
        if self._initialized:
            return
        try:
            from src.drive_uploader import (
                get_drive_service,
                get_or_create_drive_folder,
            )

            self.drive_service = get_drive_service()
            self.root_folder_id = get_or_create_drive_folder(
                self.drive_service, "Novel_LLM_Checkpoints"
            )
            self._initialized = True
            logger.info("Google Drive service initialized")
        except Exception as e:
            logger.warning(f"Failed to init Drive: {e}")
            self._initialized = False

    def _compress_and_upload(self, checkpoint_path: Path, step: int):
        """Compress checkpoint folder and upload to Drive."""
        if not self._initialized:
            self._init_drive()
            if not self._initialized:
                return

        zip_name = f"checkpoint-{step}.zip"
        zip_path = Path("models/output") / zip_name

        try:
            logger.info(f"Compressing checkpoint-{step}...")
            shutil.make_archive(
                str(Path("models/output") / f"checkpoint-{step}"), "zip", str(checkpoint_path)
            )

            from src.drive_uploader import upload_file_to_drive

            upload_file_to_drive(self.drive_service, zip_path, self.root_folder_id)

            # Mark as uploaded
            (checkpoint_path / ".uploaded").touch()

            # Clean up zip
            if zip_path.exists():
                os.remove(zip_path)

            logger.info(f"Uploaded checkpoint-{step} to Google Drive")

        except Exception as e:
            logger.error(f"Error uploading checkpoint-{step}: {e}", exc_info=True)

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Called after checkpoint save."""
        # Check if we should upload (every upload_interval_steps)
        if state.global_step % self.upload_interval_steps != 0:
            return

        # Avoid duplicate uploads
        if state.global_step <= self.last_uploaded_step:
            return

        checkpoint_dirs = sorted(
            Path(args.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
        )
        if not checkpoint_dirs:
            return

        latest_checkpoint = checkpoint_dirs[-1]
        step = int(latest_checkpoint.name.split("-")[1])

        logger.info(f"Uploading checkpoint at step {step}...")
        self._compress_and_upload(latest_checkpoint, step)
        self.last_uploaded_step = step

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Upload final checkpoint on training end."""
        checkpoint_dirs = sorted(
            Path(args.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
        )
        if not checkpoint_dirs:
            return

        latest_checkpoint = checkpoint_dirs[-1]
        step = int(latest_checkpoint.name.split("-")[1])

        if step > self.last_uploaded_step:
            logger.info(f"Final upload of checkpoint-{step}...")
            self._compress_and_upload(latest_checkpoint, step)


# ============================================================
# CustomTrainer with Muon/AdamW split
# ============================================================
class CustomTrainer(Trainer):
    def __init__(self, *args, additional_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_config = additional_config

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        config = self.additional_config
        hpo_config = config.get("hpo", {})

        lr_2d = hpo_config.get("max_lr_2d", 3e-4)
        lr_1d = hpo_config.get("max_lr_1d", 3e-3)
        weight_decay = hpo_config.get("weight_decay", 0.1)
        beta2 = hpo_config.get("beta2", 0.95)
        grad_clip = hpo_config.get("grad_clip", 1.0)

        params_2d = []
        params_1d = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if len(p.shape) < 2 or "embed" in n or "norm" in n or "bias" in n or "lm_head" in n:
                params_1d.append(p)
            else:
                params_2d.append(p)

        try:
            from muon import Muon

            logger.info(f"Optimizer: Muon for 2D (lr={lr_2d}), AdamW for 1D (lr={lr_1d})")
            self.optimizer = Muon(
                params_2d,
                lr=lr_2d,
                momentum=0.95,
                adamw_params=dict(
                    params=params_1d,
                    lr=lr_1d,
                    betas=(0.9, hpo_config.get("beta2", 0.95)),
                    weight_decay=hpo_config.get("weight_decay", 0.1),
                ),
            )
        except ImportError:
            logger.info("Optimizer: Muon not found. Falling back to split AdamW.")
            from torch.optim import AdamW

            self.optimizer = AdamW(
                [
                    {"params": params_2d, "lr": lr_2d, "weight_decay": 0.0},
                    {"params": params_1d, "lr": lr_1d, "weight_decay": hpo_config.get("weight_decay", 0.1)},
                ],
                betas=(0.9, hpo_config.get("beta2", 0.95)),
            )

        return self.optimizer


# ============================================================
# Main training function
# ============================================================
def train(config):
    """
    Args:
        config: dict from src.config.load_config()
    """
    config = normalize_config(config)
    git_hash = get_git_revision_hash()
    seed = config.get("seed", 42)

    # --- Seed fixing (ADR-017) ---
    from src.set_seed import set_seed

    set_seed(seed, deterministic=True)

    # --- Dataset fingerprinting (traceability) ---
    data_path_str = config.get("data_path", "data/dataset.jsonl")
    data_fingerprint = compute_dataset_fingerprint(data_path_str)
    db_path_str = config.get("db_path", "../Novel_Data_Collection/novels.db")
    db_fingerprint = compute_db_fingerprint(db_path_str)

    # --- Environment snapshot (traceability) ---
    from src.env_snapshot import capture_env_snapshot

    env_snapshot = capture_env_snapshot()

    # --- MLflow init with full traceability ---
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_run = None
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("LLM_Training")
        mlflow_run = mlflow.start_run()

        # Core params
        mlflow.log_params(
            {
                "git_hash": git_hash,
                "seed": seed,
                "data_path": data_path_str,
                "max_steps": str(config.get("max_steps", -1)),
            }
        )

        # Model params
        for k, v in config.get("model_params", {}).items():
            if v is not None:
                mlflow.log_param(f"model.{k}", v)

        # HPO params
        for k, v in config.get("hpo", {}).items():
            if v is not None:
                mlflow.log_param(f"hpo.{k}", v)

        # Dataset fingerprint (traceability)
        if "error" not in data_fingerprint:
            mlflow.log_params(
                {
                    "dataset.sha256": data_fingerprint["sha256"],
                    "dataset.rows": data_fingerprint["line_count"],
                    "dataset.size_bytes": data_fingerprint["size_bytes"],
                }
            )

        # DB fingerprint (traceability)
        if "error" not in db_fingerprint:
            mlflow.log_params(
                {
                    "db.sha256": db_fingerprint["sha256"],
                    "db.chapters": db_fingerprint["chapter_count"],
                    "db.novels": db_fingerprint["novel_count"],
                }
            )

        # Environment snapshot (traceability)
        mlflow.log_dict(env_snapshot, "environment.json")

        # Config artifact (log as dict)
        mlflow.log_dict(config, "config.json")

    except Exception as e:
        logger.warning(f"MLflow initialization failed: {e}", exc_info=True)

    # --- Tokenizer ---
    tokenizer_path = config.get("tokenizer_path", "data/tokenizer.json")
    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    # ADR-021: Use SP-native token names (IDs 0-3 in vocab)
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    logger.info(
        f"Special token IDs: unk={tokenizer.unk_token_id}, bos={tokenizer.bos_token_id}, "
        f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
    )

    # --- Dataset loading ---
    logger.info("Starting dataset load...")
    data_path = Path(data_path_str)
    if not data_path.exists():
        fallback_path = Path("data") / data_path.name
        if fallback_path.exists():
            data_path = fallback_path
            logger.info(f"Dataset path resolved to fallback: {data_path}")
        else:
            raise FileNotFoundError(
                f"Could not find dataset at '{data_path_str}' or '{fallback_path.resolve()}'"
            )

    # Check for separate train/val files
    train_path = config.get("train_data_path")
    val_path = config.get("val_data_path")

    if train_path and val_path:
        # Use explicit train/val files
        train_data_path = Path(train_path)
        val_data_path = Path(val_path)
        if not train_data_path.is_absolute():
            train_data_path = Path(data_path.parent) / train_path
        if not val_data_path.is_absolute():
            val_data_path = Path(data_path.parent) / val_path

        logger.info(f"Loading train from: {train_data_path}")
        logger.info(f"Loading val from: {val_data_path}")

        dataset = load_dataset(
            "json", data_files={"train": str(train_data_path), "validation": str(val_data_path)}
        )
    else:
        # Single file - use as train only, no validation
        logger.info(f"Loading single dataset from: {data_path}")
        dataset = load_dataset("json", data_files=str(data_path))

    seq_len = config.get("hpo", {}).get("seq_len", 512)
    
    # Support data_fraction for pilot runs (e.g., 0.01 = 1% of data)
    data_fraction = config.get("data_fraction", 1.0)
    if data_fraction < 1.0:
        logger.info(f"Using data_fraction={data_fraction} for pilot run")
    
    logger.info(f"Tokenizing dataset with max_length={seq_len}...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=seq_len
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "metadata"])
    
    # Apply data_fraction for pilot runs
    data_fraction = config.get("data_fraction", 1.0)
    if data_fraction < 1.0:
        for split in tokenized_datasets:
            n_samples = int(len(tokenized_datasets[split]) * data_fraction)
            if n_samples < len(tokenized_datasets[split]):
                tokenized_datasets[split] = tokenized_datasets[split].select(range(n_samples))
                logger.info(f"Split '{split}' sampled to {n_samples} samples (fraction={data_fraction})")
    
    tokenized_datasets.set_format("torch")

    # Print dataset sizes
    if "train" in tokenized_datasets:
        logger.info(f"Train dataset size: {len(tokenized_datasets['train'])}")
    if "validation" in tokenized_datasets:
        logger.info(f"Validation dataset size: {len(tokenized_datasets['validation'])}")

    # --- Model initialization ---
    logger.info("Initializing model...")
    model_params = config["model_params"].copy()
    model_params.pop("hidden_size", None)

    hidden_size = config["model_params"]["hidden_size"]
    num_heads = config["model_params"]["num_attention_heads"]
    adjusted_hidden_size = (hidden_size // num_heads) * num_heads

    model_config = LlamaConfig(
        **{k: v for k, v in model_params.items() if v is not None},
        hidden_size=adjusted_hidden_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(model_config)

    # 高速化: torch.compileはエラーになるため削除

    model.resize_token_embeddings(len(tokenizer))
    logger.info(
        f"Model initialized: hidden_size={adjusted_hidden_size}, vocab_size={len(tokenizer)}"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- TrainingArguments ---
    max_steps = config.get("max_steps", -1)
    num_epochs = config.get("num_epochs", 3) if max_steps == -1 else 0

    hpo_config = config["hpo"]
    target_total_batch_seqs = hpo_config.get("batch_size_seqs", 16)

    n_params_est = config["model_params"].get("n_params", 125_000_000)

    # VRAM limit: from config or auto-detect
    vram_limit = config.get("vram_limit_gb")
    if vram_limit is None:
        from src.config import detect_vram

        vram_limit = detect_vram()
    vram_limit = float(vram_limit)

    # Precision: from config or default bf16
    precision = config.get("precision", "bf16")

    ds_config_path = generate_deepspeed_config(n_params_est, vram_limit, precision=precision)

    # Batch size calculation - モデルサイズとVRAMに基づく
    # バッチサイズ計算: 4GB VRAMではseq_len=2048でbatch=1が限界
    # CPUオフロードを使わず、batch小さく + grad_accumで調整
    if vram_limit <= 4.5:
        max_batch = 1  # seq_len=2048ではbatch=1が安全
    elif vram_limit <= 8.5:
        max_batch = 4
    else:
        max_batch = 8

    per_device_batch = min(target_total_batch_seqs, max_batch)
    grad_accum_steps = max(1, target_total_batch_seqs // per_device_batch)
    warmup_ratio = hpo_config.get("warmup_ratio", 0.03)
    weight_decay = hpo_config.get("weight_decay", 0.1)
    adam_beta2 = hpo_config.get("beta2", 0.95)
    max_grad_norm = hpo_config.get("grad_clip", 1.0)

    training_args = TrainingArguments(
        output_dir="models/output",
        learning_rate=hpo_config["max_lr_2d"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps != -1 else -1,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        adam_beta2=adam_beta2,
        max_grad_norm=max_grad_norm,
        seed=seed,
        deepspeed=ds_config_path
        if (torch.cuda.is_available() and is_deepspeed_available())
        else None,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps" if "validation" in tokenized_datasets else "no",
        eval_steps=1000 if "validation" in tokenized_datasets else None,
        logging_steps=10,
        report_to=["tensorboard", "mlflow"],
        load_best_model_at_end="validation" in tokenized_datasets,
        metric_for_best_model="eval_loss" if "validation" in tokenized_datasets else None,
        greater_is_better=False,
    )

    # Prepare callbacks
    callbacks = [DriveUploadCallback(upload_interval_steps=1000)]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
        additional_config=config,
        callbacks=callbacks,
    )

    logger.info(
        f"Trainer: batch={per_device_batch}, grad_accum={grad_accum_steps}, scheduler=cosine (warmup={warmup_ratio})"
    )

# --- Detailed Training Callback ---
    class DetailedLoggingCallback(TrainerCallback):
        def __init__(self, logger, log_every_n_steps=1):
            self.logger = logger
            self.log_every_n_steps = log_every_n_steps
            self.step_count = 0
            self.epoch_start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            self.step_count += 1
            if self.step_count % self.log_every_n_steps == 0:
                # Get current learning rate from optimizer
                lr = None
                if self.trainer and self.trainer.optimizer:
                    lr = self.trainer.optimizer.param_groups[0]['lr']

                # Get loss from state
                loss = state.log_history[-1].get('loss') if state.log_history else None

                # GPU memory
                gpu_mem = ""
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_mem = f" | GPU: {torch.cuda.memory_allocated() / 1024**3:.2f}/{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"

                if loss is not None:
                    elapsed = time.time() - self.epoch_start_time
                    if self.trainer and self.trainer.optimizer:
                        lr_str = f"lr={self.trainer.optimizer.param_groups[0]['lr']:.2e}"
                    else:
                        lr_str = "lr=N/A"
                    self.logger.info(
                        f"Step {self.step_count} | "
                        f"loss={state.log_history[-1].get('loss', 'N/A'):.4f} | "
                        f"lr={self.trainer.optimizer.param_groups[0]['lr']:.2e}" 
                        f" | elapsed={time.time()-self.epoch_start_time:.1f}s"
                        f" | GPU: {torch.cuda.memory_allocated()/1024**3:.2f}/{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
                    )

            return control

    # Attach custom callback
    detailed_callback = DetailedLoggingCallback(logger, log_every_n_steps=1)
    callbacks.append(detailed_callback)

    # --- Resume ---
    resume_flag = config.get("resume", False)
    train_result = trainer.train(resume_from_checkpoint=resume_flag)

    # --- Post-training traceability ---
    with open("last_run_result.json", "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f)

    # Log final metrics to MLflow
    try:
        if mlflow_run is not None:
            mlflow.log_metrics(
                {
                    "final_train_loss": train_result.metrics.get("train_loss", -1),
                    "final_train_runtime": train_result.metrics.get("train_runtime", -1),
                    "final_train_samples_per_second": train_result.metrics.get(
                        "train_samples_per_second", -1
                    ),
                    "final_train_steps_per_second": train_result.metrics.get(
                        "train_steps_per_second", -1
                    ),
                }
            )
    except Exception as e:
        logger.warning(f"MLflow metrics logging failed: {e}", exc_info=True)

    # --- Cleanup ---
    if os.path.exists(ds_config_path):
        import contextlib

        with contextlib.suppress(Exception):
            os.remove(ds_config_path)

    model.save_pretrained("models/output")
    tokenizer.save_pretrained("models/output")

    # --- Log model as MLflow artifact (traceability) ---
    try:
        if mlflow_run is not None:
            # Log model config as artifact
            model_config_path = Path("models/output/config.json")
            if model_config_path.exists():
                mlflow.log_artifact(str(model_config_path), artifact_path="model")

            # Log tokenizer as artifact
            tokenizer_files = [
                "models/output/tokenizer.json",
                "models/output/special_tokens_map.json",
                "models/output/tokenizer_config.json",
            ]
            for tf in tokenizer_files:
                if Path(tf).exists():
                    mlflow.log_artifact(tf, artifact_path="model")

            # Log the full model directory as a zip artifact
            import zipfile

            model_zip_path = Path("logs/model_snapshot.zip")
            with zipfile.ZipFile(model_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in Path("models/output").rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to("models/output")
                        zf.write(file_path, arcname)
            mlflow.log_artifact(str(model_zip_path), artifact_path="model")
            model_zip_path.unlink(missing_ok=True)
            logger.info("[MLflow] Model artifacts logged.")
    except Exception as e:
        logger.warning(f"[MLflow] Model artifact logging failed: {e}", exc_info=True)

    try:
        if mlflow_run is not None:
            mlflow.end_run()
    except Exception:
        pass

    logger.info("Training finished.")


if __name__ == "__main__":
    config_path = sys.argv[1]
    train(config_path)

import datetime
import json
import os
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.common.logger import logger
from src.common.env_snapshot import capture_env_snapshot
from src.common.set_seed import set_seed
from src.training.model_utils import (
    create_model_config,
    TokenizerWrapper,
    get_optimal_num_proc,
    compute_file_hash,
)


class ProgressBarFormatCallback(TrainerCallback):
    """
    tqdm進捗バーの書式をカスタマイズするコールバック。
    推定残り時間の代わりに平均イテレーション時間を表示。
    """

    def __init__(self):
        self.iteration_times = []
        self.last_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.iteration_times = []
        self.last_time = time.time()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.last_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.last_time is not None:
            iter_time = time.time() - self.last_time
            self.iteration_times.append(iter_time)
            if len(self.iteration_times) > 100:
                self.iteration_times.pop(0)
            self.last_time = None

            if self.iteration_times:
                avg_time = sum(self.iteration_times) / len(self.iteration_times)
                if 'pbar' in kwargs:
                    kwargs['pbar'].set_postfix_str(f"avg: {avg_time:.2f}s/it")


class HashSaveCallback(TrainerCallback):
    """
    保存された各チェックポイントにconfigとデータのハッシュを保存するコールバック。
    """
    def __init__(self, config_hash: str, data_hash: str):
        self.config_hash = config_hash
        self.data_hash = data_hash

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            hash_file = checkpoint_dir / "hashes.json"
            with open(hash_file, "w") as f:
                json.dump({
                    "config_hash": self.config_hash,
                    "data_hash": self.data_hash,
                    "timestamp": datetime.datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Saved config and data hashes to {hash_file}")


class DriveUploadCallback(TrainerCallback):
    """
    保存後にチェックポイントをGoogle Driveにアップロードするコールバック。
    """

    def __init__(self, upload_interval_steps: int = 1000):
        self.upload_interval_steps = upload_interval_steps
        self.last_uploaded_step = -1
        self.drive_service = None
        self.root_folder_id = None
        self._initialized = False

    def _init_drive(self):
        if self._initialized:
            return
        try:
            from src.training.drive_uploader import (
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
        if not self._initialized:
            self._init_drive()
            if not self._initialized:
                return

        zip_name = f"checkpoint-{step}.zip"
        # output_dir の親や自身の位置から zip のパスを設定
        zip_path = checkpoint_path.parent / zip_name

        try:
            logger.info(f"Compressing checkpoint-{step}...")
            shutil.make_archive(
                str(checkpoint_path.parent / f"checkpoint-{step}"), "zip", str(checkpoint_path)
            )

            from src.training.drive_uploader import upload_file_to_drive

            upload_file_to_drive(self.drive_service, zip_path, self.root_folder_id)

            # アップロード済みとしてマーク
            (checkpoint_path / ".uploaded").touch()

            # zipクリーンアップ
            if zip_path.exists():
                os.remove(zip_path)

            logger.info(f"Uploaded checkpoint-{step} to Google Drive")

        except Exception as e:
            logger.error(f"Error uploading checkpoint-{step}: {e}", exc_info=True)

    def force_final_upload(self, step: int):
        """学習終了時にアップロードされていなければ強制アップロード"""
        # 直前のアップロードステップと同じであればスキップ
        if step <= self.last_uploaded_step:
            return
        self._init_drive()
        if not self._initialized:
            return
        # 最新のチェックポイントディレクトリを探す
        # ここでは後述の引数保存位置から解決されるが、呼び出し側からトリガーされる
        logger.info(f"Triggered final check/upload at step {step}")

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if state.global_step % self.upload_interval_steps != 0:
            return

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
            self.last_uploaded_step = step


class CustomTrainer(Trainer):
    """
    Muon/AdamW分割オプティマイザを自動構築するカスタムTrainer。
    """
    def __init__(self, *args, additional_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_config = additional_config

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        config = self.additional_config
        # hpo または training の設定からハイパーパラメータを取得
        hpo_config = config.get("hpo", config)

        lr_2d = hpo_config.get("max_lr_2d", 3e-4)
        lr_1d = hpo_config.get("max_lr_1d", 3e-3)
        weight_decay = hpo_config.get("weight_decay", 0.1)
        beta2 = hpo_config.get("beta2", 0.95)

        # 2Dパラメータ（行列）と1Dパラメータ（embeddings, biases, layernorms）に分類
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
                    betas=(0.9, beta2),
                    weight_decay=weight_decay,
                ),
            )
        except ImportError:
            logger.info("Optimizer: Muon not found. Falling back to split AdamW.")
            from torch.optim import AdamW

            self.optimizer = AdamW(
                [
                    {"params": params_2d, "lr": lr_2d, "weight_decay": 0.0},
                    {
                        "params": params_1d,
                        "lr": lr_1d,
                        "weight_decay": weight_decay,
                    },
                ],
                betas=(0.9, beta2),
            )

        return self.optimizer


class DetailedLoggingCallback(TrainerCallback):
    """詳細ログ出力用コールバック"""
    def __init__(self, log_every_n_steps=1):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.epoch_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.log_every_n_steps == 0:
            loss = state.log_history[-1].get("loss") if state.log_history else None
            lr_val = "N/A"
            if self.trainer and self.trainer.optimizer:
                lr_val = f"{self.trainer.optimizer.param_groups[0]['lr']:.2e}"

            gpu_info = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info = f" | GPU: {allocated:.2f}/{total:.1f}GB"

            if loss is not None:
                logger.info(
                    f"Step {self.step_count} | "
                    f"loss={loss:.4f} | "
                    f"lr={lr_val}"
                    f" | elapsed={time.time() - self.epoch_start_time:.1f}s"
                    f"{gpu_info}"
                )
        return control


def compute_dataset_fingerprint(dataset_path: str) -> dict:
    path = Path(dataset_path)
    if not path.exists():
        return {"error": f"File not found: {dataset_path}"}

    stat = path.stat()
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


def train(config: dict, tokenized_datasets=None, extra_callbacks=None):
    """
    Unified training implementation.
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
        # Resolve to latest checkpoint if it is a directory or path pattern
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
        # If it was boolean or unresolved
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
                # Ensure at least 1 sample is selected
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
    
    # Optional resource adjustments if batch size sequences is set (typically during HPO)
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

    # Only upload checkpoints automatically when running full training
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

    # Set trainer reference on callback for logging optimizer state
    for cb in callbacks:
        if isinstance(cb, DetailedLoggingCallback):
            cb.trainer = trainer

    logger.info("*** Starting Unified Training Pipeline ***")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final results
    if max_steps == -1:
        with open("last_run_result.json", "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Force final upload if drive callback is in use
    if drive_cb in callbacks:
        drive_cb.force_final_upload(trainer.state.global_step)

    return train_result.training_loss

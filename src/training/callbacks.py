import datetime
import json
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.common.logger import logger


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
        if step <= self.last_uploaded_step:
            return
        self._init_drive()
        if not self._initialized:
            return
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


class DetailedLoggingCallback(TrainerCallback):
    """詳細ログ出力用コールバック"""
    def __init__(self, log_every_n_steps=1):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.epoch_start_time = time.time()
        self.trainer = None  # Reference injected after trainer instantiation

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

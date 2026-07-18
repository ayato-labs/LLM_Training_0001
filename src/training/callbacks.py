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



class DetailedLoggingCallback(TrainerCallback):
    """詳細ログ出力用コールバック"""
    def __init__(self, log_every_n_steps=1):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.epoch_start_time = time.time()
        self.start_step = 0
        self.trainer = None  # Reference injected after trainer instantiation
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.start_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count = state.global_step
        current_time = time.time()
        
        if self.step_count % self.log_every_n_steps == 0:
            loss = state.log_history[-1].get("loss") if state.log_history else None
            lr_val = "N/A"
            if self.trainer and self.trainer.optimizer:
                lr_val = f"{self.trainer.optimizer.param_groups[0]['lr']:.2e}"

            # 進捗割合とETAの算出
            total_steps = state.max_steps
            progress_str = f"Step {self.step_count}"
            eta_str = ""
            speed_str = ""
            
            elapsed_time = current_time - self.epoch_start_time
            steps_in_session = self.step_count - self.start_step
            if steps_in_session > 0:
                steps_per_sec = steps_in_session / elapsed_time
                speed_str = f" | {1.0 / steps_per_sec:.2f}s/it"
                
                if total_steps and total_steps > 0:
                    pct = (self.step_count / total_steps) * 100
                    progress_str = f"Step {self.step_count}/{total_steps} ({pct:.1f}%)"
                    
                    remaining_steps = total_steps - self.step_count
                    remaining_time = remaining_steps * (elapsed_time / steps_in_session)
                    
                    # hh:mm:ss 形式にフォーマット
                    hrs, remainder = divmod(int(remaining_time), 3600)
                    mins, secs = divmod(remainder, 60)
                    if hrs > 0:
                        eta_str = f" | ETA={hrs}h{mins}m"
                    elif mins > 0:
                        eta_str = f" | ETA={mins}m{secs}s"
                    else:
                        eta_str = f" | ETA={secs}s"

            gpu_info = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info = f" | GPU: {allocated:.2f}/{total:.1f}GB"
                if total > 0 and (allocated / total) > 0.95:
                    logger.warning(
                        f"High VRAM usage detected: {allocated:.2f}/{total:.1f}GB ({allocated/total*100:.1f}%). "
                        "CPU offloading or Unified Memory paging may be active, which can severely degrade training speed."
                    )

            if loss is not None:
                logger.info(
                    f"{progress_str} | "
                    f"loss={loss:.4f} | "
                    f"lr={lr_val}"
                    f"{speed_str}"
                    f" | elapsed={elapsed_time:.1f}s"
                    f"{eta_str}"
                    f"{gpu_info}"
                )
        return control


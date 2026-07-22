import datetime
import json
import math
import time
from pathlib import Path

import torch
from transformers import (
    TrainerCallback,
)

from src.common.logger import logger


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
                json.dump(
                    {
                        "config_hash": self.config_hash,
                        "data_hash": self.data_hash,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
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

                # 直近インターバルの速度（local）を算出（コンパイル等の初期遅延による影響を排除）
                if not hasattr(self, "last_logged_step") or self.last_logged_step is None:
                    self.last_logged_step = self.start_step
                    self.last_logged_time = self.epoch_start_time

                local_steps = self.step_count - self.last_logged_step
                local_elapsed = current_time - self.last_logged_time

                local_speed_str = ""
                if local_steps > 0 and local_elapsed > 0:
                    local_speed = local_steps / local_elapsed
                    local_speed_str = f" ({1.0 / local_speed:.2f}s/it local)"

                self.last_logged_step = self.step_count
                self.last_logged_time = current_time

                speed_str = f" | {1.0 / steps_per_sec:.2f}s/it{local_speed_str}"

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
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info = (
                    f" | GPU: {allocated:.2f}/{total:.1f}GB (peak_reserved={peak_reserved:.2f}GB)"
                )

                # 4GB未満のエントリーGPU等で、CUDAコンテクスト（0.7GB）込みの
                # 総量が物理VRAMの上限に迫っている場合に警告
                estimated_total_vram = peak_reserved + 0.7
                if total > 0 and estimated_total_vram > total and (allocated / total) <= 0.95:
                    logger.warning(
                        "Silent VRAM Paging Warning: "
                        f"Peak reserved memory ({peak_reserved:.2f}GB) "
                        f"plus estimated CUDA context/OS overhead (~0.7GB) "
                        f"is {estimated_total_vram:.2f}GB, "
                        f"which exceeds physical VRAM ({total:.1f}GB). "
                        "The Windows WDDM driver has likely silently "
                        "paged CUDA memory to system RAM, "
                        "which will severely degrade steps speed "
                        "(up to 5x-10x slower)."
                    )
                elif total > 0 and (allocated / total) > 0.95:
                    logger.warning(
                        f"High VRAM usage detected: "
                        f"{allocated:.2f}/{total:.1f}GB "
                        f"({allocated / total * 100:.1f}%). "
                        "CPU offloading or Unified Memory paging may be "
                        "active, which can severely degrade training speed."
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


class PeriodicEvaluationCallback(TrainerCallback):
    """
    学習中の定期評価コールバック

    機能:
    - perplexity 計算 (eval_dataset使用時)
    - 生成サンプル出力 (指定プロンプト)
    - TensorBoard 記録
    - 早期発散検知 (loss > threshold)

    Step Law / Muon 環境での学習安定性監視用
    """

    def __init__(
        self,
        eval_every_n_steps: int = 500,
        eval_prompts: list[str] | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        divergence_threshold: float = 10.0,
        log_generations: bool = True,
    ):
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_prompts = eval_prompts or [
            "＜|start_of_story|＞",
            "昔々、あるところに",
            "彼は剣を構え、",
            "「待って、」彼女は言った。",
        ]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.divergence_threshold = divergence_threshold
        self.log_generations = log_generations
        self.trainer = None
        self.tokenizer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.trainer is not None:
            self.tokenizer = self.trainer.tokenizer

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        # 発散検知
        if state.log_history:
            last_loss = state.log_history[-1].get("loss")
            if last_loss and last_loss > self.divergence_threshold:
                logger.error(
                    f"DIVERGENCE DETECTED at step {step}: loss={last_loss:.4f} "
                    f"> threshold={self.divergence_threshold}. Consider early stopping."
                )
                control.should_training_stop = True
                return control

        # 定期評価実行
        if step > 0 and step % self.eval_every_n_steps == 0:
            self._run_periodic_evaluation(args, state, control)

        return control

    def _run_periodic_evaluation(self, args, state, control):
        """定期評価の実行"""
        logger.info(f"=== Periodic Evaluation at Step {state.global_step} ===")

        # 1. Perplexity 計算 (eval_datasetがある場合)
        if self.trainer is not None and self.trainer.eval_dataset is not None:
            try:
                eval_results = self.trainer.evaluate()
                eval_loss = eval_results.get("eval_loss")
                if eval_loss is not None:
                    perplexity = math.exp(min(eval_loss, 20))  # overflow防止
                    logger.info(f"  Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")

                    # TensorBoard記録
                    if self.trainer.tb_writer:
                        self.trainer.tb_writer.add_scalar("eval/loss", eval_loss, state.global_step)
                        self.trainer.tb_writer.add_scalar(
                            "eval/perplexity", perplexity, state.global_step
                        )
            except Exception as e:
                logger.warning(f"Periodic evaluation failed: {e}")

        # 2. 生成サンプル出力
        if self.log_generations and self.trainer is not None and self.tokenizer is not None:
            self._generate_samples(state.global_step)

        logger.info("=== Periodic Evaluation Complete ===")

    def _generate_samples(self, step: int):
        """生成サンプルの出力"""
        if not self.eval_prompts:
            return

        model = self.trainer.model
        model.eval()

        try:
            for i, prompt in enumerate(self.eval_prompts[:3]):  # 最大3サンプル
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # プロンプト部分を除いてデコード
                generated = outputs[0][inputs["input_ids"].shape[1] :]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)

                logger.info(f"  [Gen {i + 1}] Prompt: '{prompt[:40]}...' -> '{text[:80]}...'")

                # TensorBoardにテキスト記録
                if self.trainer.tb_writer:
                    self.trainer.tb_writer.add_text(
                        f"generation/sample_{i + 1}",
                        f"Prompt: {prompt}\nGenerated: {text}",
                        step,
                    )

        except Exception as e:
            logger.warning(f"Generation failed: {e}")
        finally:
            model.train()

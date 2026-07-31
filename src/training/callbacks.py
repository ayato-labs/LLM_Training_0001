import datetime
import json
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
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
    """進捗バー + 定期ログ出力用コールバック

    - tqdm でリアルタイム進捗バー表示（loss, lr, speed, GPU）
    - log_every_n_steps ごとに INFO ログ出力（メトリクス永続化）
    """

    def __init__(self, log_every_n_steps=200):
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.step_count = 0
        self.epoch_start_time = time.time()
        self.start_step = 0
        self.trainer = None
        self.last_step_time = None
        self._pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.start_step = state.global_step
        self.step_count = state.global_step

        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        self._pbar = tqdm(
            total=total,
            initial=self.step_count,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            disable=total is None,
        )

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count = state.global_step
        current_time = time.time()

        # ---- loss / lr 取得 ----
        loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss = entry["loss"]
                    break
        lr_val = "N/A"
        if self.trainer and self.trainer.optimizer:
            lr_val = f"{self.trainer.optimizer.param_groups[0]['lr']:.2e}"

        # ---- 速度 / ETA / GPU 計算 ----
        total_steps = state.max_steps
        elapsed_time = current_time - self.epoch_start_time
        steps_in_session = self.step_count - self.start_step

        speed_str = ""
        eta_str = ""
        gpu_info = ""

        if steps_in_session > 0:
            steps_per_sec = steps_in_session / elapsed_time

            # local speed (直近インターバル)
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

            speed_str = f" | {1.0 / (steps_in_session / elapsed_time):.2f}s/it{local_speed_str}"

            if total_steps and total_steps > 0:
                pct = (self.step_count / total_steps) * 100
                remaining_steps = total_steps - self.step_count
                remaining_time = remaining_steps * (elapsed_time / max(1, steps_in_session))
                hrs, remainder = divmod(int(remaining_time), 3600)
                mins, secs = divmod(remainder, 60)
                if hrs > 0:
                    eta_str = f" | ETA={hrs}h{mins}m"
                elif mins > 0:
                    eta_str = f" | ETA={mins}m{secs}s"
                else:
                    eta_str = f" | ETA={secs}s"

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f" | GPU: {allocated:.2f}/{total:.1f}GB (peak={peak_reserved:.2f}GB)"

        # ---- tqdm 更新 ----
        postfix = {}
        if loss is not None:
            postfix["loss"] = f"{loss:.4f}"
        postfix["lr"] = lr_val
        if loss is not None:
            self._pbar.set_postfix(postfix)

        self._pbar.n = self.step_count
        self._pbar.refresh()

        # ---- logging_steps ごとに INFO ログ出力 ----
        if self.step_count % self.log_every_n_steps == 0 and loss is not None:
            progress_str = f"Step {self.step_count}"
            if total_steps and total_steps > 0:
                pct = (self.step_count / total_steps) * 100
                progress_str = f"Step {self.step_count}/{total_steps} ({pct:.1f}%)"
            elapsed_time = current_time - self.epoch_start_time

            breakdown_str = ""
            if self.trainer and hasattr(self.trainer, "_last_fwd_bwd_ms"):
                data_ms = getattr(self.trainer, "_last_data_fetch_ms", 0.0)
                h2d_ms = getattr(self.trainer, "_last_h2d_ms", 0.0)
                fwd_ms = getattr(self.trainer, "_last_fwd_bwd_ms", 0.0)
                total_ms = max(1.0, data_ms + h2d_ms + fwd_ms)
                breakdown_str = (
                    f" | Breakdown: DataFetch={data_ms:.1f}ms ({data_ms / total_ms * 100:.1f}%), "
                    f"H2D={h2d_ms:.1f}ms ({h2d_ms / total_ms * 100:.1f}%), "
                    f"FwdBwd={fwd_ms:.1f}ms ({fwd_ms / total_ms * 100:.1f}%)"
                )

            logger.info(
                f"{progress_str} | "
                f"loss={loss:.4f} | "
                f"lr={lr_val}"
                f"{speed_str if 'speed_str' in locals() else ''}"
                f"{breakdown_str}"
                f" | elapsed={elapsed_time:.1f}s"
                f"{eta_str if 'eta_str' in locals() else ''}"
                f"{gpu_info}"
            )

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._pbar:
            self._pbar.close()
            self._pbar = None


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
        divergence_threshold: float | None = None,
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
        self.user_divergence_threshold = divergence_threshold
        self.divergence_threshold = divergence_threshold or 15.0
        self.log_generations = log_generations
        self.trainer = None
        self.tokenizer = None
        self.min_loss: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_step = state.global_step
        if self.trainer is not None:
            self.tokenizer = self.trainer.tokenizer

        # ユーザー明示指定がない場合、語彙数 V から初期理論上限 ln(V) * 1.5 を動的算定
        if self.user_divergence_threshold is None:
            vocab_size = None
            if self.tokenizer and hasattr(self.tokenizer, "vocab_size"):
                vocab_size = self.tokenizer.vocab_size
            elif (
                self.trainer
                and hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "config")
            ):
                vocab_size = getattr(self.trainer.model.config, "vocab_size", None)

            if vocab_size and vocab_size > 0:
                theory_init_loss = math.log(vocab_size)
                self.divergence_threshold = max(12.0, theory_init_loss * 1.5)
                logger.info(
                    f"Dynamic divergence threshold computed from vocab_size ({vocab_size}): "
                    f"ln({vocab_size})={theory_init_loss:.2f} -> threshold={self.divergence_threshold:.2f}"
                )
            else:
                self.divergence_threshold = 15.0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        # 専門家仕様の多重自律発散検知
        if state.log_history and len(state.log_history) > 0:
            last_loss = state.log_history[-1].get("loss")
            if last_loss is not None:
                # 1. NaN / Inf 判定
                if math.isnan(last_loss) or math.isinf(last_loss):
                    logger.error(
                        f"NUMERICAL INSTABILITY DETECTED at step {step}: loss is {last_loss}. Stopping training."
                    )
                    control.should_training_stop = True
                    return control

                # 2. 最小 Loss の追跡
                if self.min_loss is None or last_loss < self.min_loss:
                    self.min_loss = last_loss

                # 3. 初期理論上限超過判定 (Warmup完了後からの初期過大発散防止)
                # warmup_steps が設定されている場合はその完了後、未指定時は 20 ステップを猶予
                warmup_grace = getattr(args, "warmup_steps", 20) or 20
                if step > warmup_grace and last_loss > self.divergence_threshold:
                    logger.error(
                        f"DIVERGENCE DETECTED at step {step}: loss={last_loss:.4f} "
                        f"> dynamic_threshold={self.divergence_threshold:.4f}. Stopping training."
                    )
                    control.should_training_stop = True
                    return control

                # 4. 過去最小 Loss からの相対スパイク判定 (学習途中の局所発散検知)
                # warmup 完了およびセッション開始から一定ステップ経過するまでは過渡現象を猶予する
                steps_since_start = step - getattr(self, "start_step", 0)
                grace_steps = max(warmup_grace * 2, 50)
                if (
                    step > grace_steps
                    and steps_since_start > grace_steps
                    and self.min_loss is not None
                ):
                    spike_limit = max(self.min_loss * 2.5, self.min_loss + 4.0)
                    if last_loss > spike_limit:
                        logger.error(
                            f"LOSS SPIKE DETECTED at step {step} ({steps_since_start} steps since session start): "
                            f"loss={last_loss:.4f} > spike_limit={spike_limit:.4f} (min_loss={self.min_loss:.4f}). Stopping training."
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
        #    HF 標準評価 (eval_strategy="steps") が同じ eval_steps で実行されるため、
        #    trainer.evaluate() を再呼び出しせず log_history の結果を参照する (二重評価防止)。
        if self.trainer is not None and self.trainer.eval_dataset is not None:
            eval_loss = None
            eval_step = None
            for entry in reversed(self.trainer.state.log_history):
                if "eval_loss" in entry:
                    eval_loss = entry["eval_loss"]
                    eval_step = entry.get("step")
                    break

            if eval_loss is not None:
                perplexity = math.exp(min(eval_loss, 20))  # overflow防止
                eval_step_str = f" (step {eval_step})" if eval_step is not None else ""
                logger.info(
                    f"  Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}{eval_step_str}"
                )

                # TensorBoard記録
                if self.trainer.tb_writer:
                    self.trainer.tb_writer.add_scalar(
                        "eval/loss", eval_loss, state.global_step
                    )
                    self.trainer.tb_writer.add_scalar(
                        "eval/perplexity", perplexity, state.global_step
                    )

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

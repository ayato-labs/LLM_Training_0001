"""HPO Library: Optuna objective & search space definition.
main.py からは import されない。scripts/find_hparams.py からのみ使用。
"""

import time
import warnings

import optuna
import torch
from transformers import TrainerCallback

from src.common.logger import logger

warnings.filterwarnings("ignore", message=".*use_return_dict.*")

# Use the unified training engine
from src.training.train_engine import train as proxy_train


class OptunaPruningCallback(TrainerCallback):
    """Optunaの途中損失値に基づいてトライアルをプルーンするコールバック。"""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            self.trial.report(loss, step=state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def determine_optimal_proxy_size(target_n_params: int) -> str:
    """ターゲットモデルサイズに対する最良のプロキシモデルサイズ (約10%) を決定

    下限 50M を適用するが、ターゲットが 50M 未満の場合はターゲットサイズをそのまま
    使用する (プロキシがターゲットより大きくなるのを防ぐ)。
    """
    ratio = 0.1
    proxy_n = min(int(target_n_params), max(50_000_000, int(target_n_params * ratio)))
    return f"{proxy_n // 1_000_000}M"


def calculate_dynamic_n_trials(search_space_dim: int = 4) -> int:
    """探索パラメータ空間の自由度 (次元数: len(search_space)) から理論的最適試行回数を自律計算 (D * 30 trials)"""
    return max(30, search_space_dim * 30)


def create_search_space(step_law_hpo: dict, vram_gb: float, n_params: int = 150_000_000) -> dict:
    """Step Law理論予測値を中心とした共通対数探索空間定義 (4次元: max_lr_2d, max_lr_1d, weight_decay, warmup_ratio)"""
    lr_2d_center = step_law_hpo["max_lr_2d"]
    lr_1d_center = step_law_hpo["max_lr_1d"]

    # Step Law 中心値に対する対数探索倍率 [0.5x, 2.0x] ＋ Weight Decay ＋ Warmup
    lr_low, lr_high = 0.5, 2.0
    wd_low, wd_high = 0.01, 0.20

    return {
        "max_lr_2d": (lr_2d_center * lr_low, lr_2d_center * lr_high, "log"),
        "max_lr_1d": (lr_1d_center * lr_low, lr_1d_center * lr_high, "log"),
        "weight_decay": (wd_low, wd_high, ""),
        "warmup_ratio": (0.01, 0.05, ""),
    }


def _run_training_process(config, tokenized_dataset, queue):
    import time

    start_time = time.perf_counter()
    logger.info("[HPO Subprocess Diag] Subprocess started at t=0.000s")
    try:
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                logger.info(
                    f"[HPO Subprocess Diag t={(time.perf_counter() - start_time) * 1000:.2f}ms] "
                    f"CUDA Device: {props.name} | Total VRAM: {props.total_memory / (1024**3):.2f}GB"
                )
            except Exception as cuda_e:
                logger.warning(f"[HPO Subprocess Diag] GPU Query Warning: {cuda_e}")

        from transformers import TrainerCallback

        class SubprocessPruningCallback(TrainerCallback):
            def __init__(self, queue, start_time):
                self.queue = queue
                self.start_time = start_time
                self.last_step_time = start_time

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    loss = logs["loss"]
                    now = time.perf_counter()
                    step_duration = (now - self.last_step_time) * 1000
                    total_elapsed = (now - self.start_time) * 1000
                    self.last_step_time = now
                    logger.info(
                        f"[HPO Subprocess Diag t={total_elapsed:.2f}ms] "
                        f"Step {state.global_step} logged: loss={loss:.4f} (step_duration={step_duration:.2f}ms)"
                    )
                    self.queue.put(("report", state.global_step, loss))

        pruning_callback = SubprocessPruningCallback(queue, start_time)
        logger.info(
            f"[HPO Subprocess Diag t={(time.perf_counter() - start_time) * 1000:.2f}ms] Calling proxy_train()..."
        )
        train_start = time.perf_counter()
        loss = proxy_train(
            config,
            tokenized_datasets=tokenized_dataset,
            extra_callbacks=[pruning_callback],
        )
        train_duration = (time.perf_counter() - train_start) * 1000
        logger.info(
            f"[HPO Subprocess Diag t={(time.perf_counter() - start_time) * 1000:.2f}ms] "
            f"proxy_train() completed in {train_duration:.2f}ms"
        )
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        queue.put(("success", loss))
    except Exception as e:
        import traceback

        fail_time = (time.perf_counter() - start_time) * 1000
        cuda_status = "Unknown"
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                cuda_status = f"Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB"
            except Exception as c_err:
                cuda_status = f"CUDA Query Failed ({c_err})"

        logger.error(
            f"[HPO Subprocess Diag CRASH t={fail_time:.2f}ms] "
            f"Subprocess failed! CUDA Status: [{cuda_status}] | Error: {e}"
        )
        queue.put(
            (
                "error",
                f"t={fail_time:.2f}ms | CUDA Status: [{cuda_status}] | {e}\n{traceback.format_exc()}",
            )
        )


def objective(
    trial: optuna.Trial,
    arch: dict,
    tokenized_dataset,
    seq_len: int,
    vram_gb: float,
    step_law_hpo: dict,
) -> float:
    """プロキシ学習目的関数（短時間実行、小データ割合、50ステップ）。"""

    # Print visible progress banner
    total_trials = trial.study.user_attrs.get("n_trials", "?")
    logger.info("=" * 60)
    logger.info(f"  [HPO Progress] Trial {trial.number + 1} / {total_trials} started...")
    logger.info("=" * 60)

    # Sample hyperparams (5D)
    hpo = {}
    space = create_search_space(step_law_hpo, vram_gb, n_params=arch["n_params"])
    for param, spec in space.items():
        if isinstance(spec, list):
            hpo[param] = trial.suggest_categorical(param, spec)
        elif spec[2] == "log":
            hpo[param] = trial.suggest_float(param, spec[0], spec[1], log=True)
        else:
            hpo[param] = trial.suggest_float(param, spec[0], spec[1])

    # Fixed values for fixed dimensions
    hpo["beta2"] = 0.95
    hpo["grad_clip"] = 1.0

    # LRスケジューラ設定 (proxy用) — train_engine は config["hpo"] を参照するため
    # トップレベルではなく hpo サブ辞書側に格納する (Step Law推奨: Constant+Cosine)
    hpo["lr_scheduler_type"] = "constant_cosine"
    hpo["warmup_steps"] = 2
    hpo["constant_steps"] = 10
    hpo["num_cycles"] = 0.5

    from pathlib import Path

    from omegaconf import OmegaConf

    # SSOT 優先順位: 1. configs/scaling_config.yaml 2. configs/base_config.yaml 3. configs/config.yaml
    scaling_path = Path("configs/scaling_config.yaml")
    base_config_path = Path("configs/base_config.yaml")
    config_path = Path("configs/config.yaml")
    base_cfg = {}

    target_cfg_path = (
        scaling_path
        if scaling_path.exists()
        else (base_config_path if base_config_path.exists() else config_path)
    )
    if target_cfg_path.exists():
        try:
            base_cfg = OmegaConf.to_container(OmegaConf.load(target_cfg_path), resolve=True)
            logger.info(
                f"[SSOT] HPO objective loaded hardware & model defaults from {target_cfg_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load {target_cfg_path} in HPO objective: {e}")

    # Chinchilla / base_config で算定された最高速度バッチサイズを適用
    batch_size_seqs = base_cfg.get("training", {}).get("batch_size_seqs") or step_law_hpo.get(
        "batch_size_seqs", 16
    )
    hpo["batch_size_seqs"] = batch_size_seqs

    # Build config matching normalize_config expectations
    config = {
        "model_params": {
            "n_params": arch["n_params"],
            "hidden_size": arch["hidden_size"],
            "num_hidden_layers": arch["num_hidden_layers"],
            "num_attention_heads": arch["num_attention_heads"],
            "num_key_value_heads": arch["num_key_value_heads"],
            "intermediate_size": arch["intermediate_size"],
            "rope_theta": arch.get("rope_theta", 500000.0),
            "vocab_size": arch.get("vocab_size", 32000),
            "attn_implementation": base_cfg.get("model", {})
            .get("llama", {})
            .get("attn_implementation", "sdpa"),
        },
        "hpo": hpo,
        "seq_len": seq_len,
        "max_steps": 50,  # 50 steps proxy run
        "logging_steps": 10,  # HPO 50ステッププロキシ試行用の明示的10ステップログ間隔（Optuna MedianPruner判定用）
        "data_fraction": 0.001,  # Tiny fraction for speed
        "precision": "bf16",
        "vram_limit_gb": vram_gb,
        "use_liger_kernel": base_cfg.get("use_liger_kernel", True),  # config.yamlから再利用
        "torch_compile": base_cfg.get("torch_compile", False),  # config.yamlから再利用
        "seed": 42,
        "tokenizer_path": "data/tokenizer.json",
        "output_dir": "models/output",
    }

    # Chinchilla 物理 VRAM 自動分解エンジン: トータルバッチサイズを物理 VRAM 安全上限へ動的自動分割
    from src.training.model_utils import calculate_optimal_batch_split

    per_device_batch, grad_accum = calculate_optimal_batch_split(
        total_batch_size=hpo["batch_size_seqs"],
        vram_gb=vram_gb,
        n_params=arch["n_params"],
        seq_len=seq_len,
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_hidden_layers"],
        vocab_size=arch.get("vocab_size", 32000),
        precision="bf16",
        selective_checkpointing=False,  # 全層標準 Gradient Checkpointing
    )

    config["per_device_batch_size"] = per_device_batch
    config["grad_accum_steps"] = grad_accum
    logger.info(
        f"[Optuna HPO] Trial {trial.number}: Chinchilla dynamic VRAM allocation -> "
        f"per_device_batch_size={per_device_batch}, grad_accum_steps={grad_accum} "
        f"(Total batch={hpo['batch_size_seqs']})"
    )

    # VRAM Estimator ガードレール (Zero-Cost Pre-Pruning)
    # 不完全な設定で VRAM の 90% を超える推定値の場合、GPU プロセスを起動せずに事前枝刈り
    from src.common.vram_estimator import (
        VramCalibration,
        VramConfig,
        estimate_training_vram_with_calibration,
    )

    cal = VramCalibration.load()
    vram_est = estimate_training_vram_with_calibration(
        VramConfig(
            n_params=arch["n_params"],
            hidden_size=arch["hidden_size"],
            intermediate_size=arch.get("intermediate_size", 0),
            num_layers=arch["num_hidden_layers"],
            vocab_size=arch.get("vocab_size", 32000),
            micro_batch_size=per_device_batch,
            seq_len=seq_len,
            precision="bf16",
            optimizer_type="adamw_bnb_8bit",
            checkpointing="full",
            use_liger_kernel=config.get("use_liger_kernel", True),
            torch_compile=config.get("torch_compile", False),
            total_vram_gb=vram_gb,
        ),
        calibration=cal,
    )

    if not vram_est.breakdown.is_safe or vram_est.breakdown.total_estimated_gb > (vram_gb * 0.90):
        logger.warning(
            f"[Optuna HPO Pre-Pruning] Trial {trial.number} pre-pruned by VRAM Estimator! "
            f"Estimated: {vram_est.breakdown.total_estimated_gb:.2f} GB / Limit: {vram_gb:.2f} GB "
            f"(Util: {vram_est.breakdown.util_pct:.1f}% > 90.0% Safety Threshold)"
        )
        raise optuna.TrialPruned()

    try:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_training_process,
            args=(config, tokenized_dataset, queue),
        )
        p.start()

        loss = 1e9
        pruned = False
        error_msg = None

        while p.is_alive() or not queue.empty():
            try:
                msg = queue.get(timeout=0.5)
                if msg[0] == "report":
                    step, val = msg[1], msg[2]
                    trial.report(val, step=step)
                    if trial.should_prune():
                        logger.info(
                            f"[Optuna HPO] Trial {trial.number}: Pruned by early stopping (枝刈り発動 - 設定学習率の効率不良を静かに打ち切り)"
                        )
                        try:
                            queue.close()
                            queue.cancel_join_thread()
                        except Exception:
                            pass
                        p.terminate()
                        p.join(timeout=1.0)
                        pruned = True
                        break
                elif msg[0] == "success":
                    loss = msg[1]
                elif msg[0] == "error":
                    error_msg = msg[1]
            except Exception:  # queue.Empty
                continue

        if pruned:
            logger.info(f"Trial {trial.number} pruned.")
            raise optuna.TrialPruned()

        p.join(timeout=600)  # 10 min timeout to prevent indefinite hang on TDR
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            logger.warning(f"Trial {trial.number}: subprocess timed out and was terminated")

        if error_msg:
            logger.warning(f"Trial subprocess failed with exception:\n{error_msg}")
            if "CUDA driver error" in error_msg or "device not ready" in error_msg:
                logger.error(
                    "=============================================================================\n"
                    "  CUDA driver error detected - likely Windows TDR (Timeout Detection & Recovery)\n"
                    "  \n"
                    "  FIX: Increase TdrDelay in Windows Registry:\n"
                    "    HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\TdrDelay = 30\n"
                    "    (DWORD, then restart WSL2)\n"
                    "  \n"
                    "  WORKAROUND: Set use_liger_kernel: false in config.yaml\n"
                    "============================================================================="
                )
            time.sleep(5.0)  # GPU driver recovery cooldown after failure
            return 1e9

        if p.exitcode != 0:
            logger.warning(f"Trial subprocess crashed with exit code {p.exitcode}")
            time.sleep(5.0)  # GPU driver recovery cooldown after failure
            return 1e9

        return (
            loss
            if not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)))
            else 1e9
        )
    finally:
        # HPO試行終了時に GPU/CPU メモリを確実に解放
        import gc
        import os
        import shutil
        import stat
        from pathlib import Path

        def _handle_remove_readonly(func, path, exc):
            """Windows読み取り専用ファイル対応"""
            os.chmod(path, stat.S_IWRITE)
            func(path)

        def _cleanup_with_retry(output_path: Path, max_retries: int = 3):
            """リトライ付きクリーンアップ (Windowsファイルロック対応)

            `runs/` (TensorBoard ログ) は削除しない。
            別の HPO インスタンスや進行中の学習が共有する可能性があり、
            削除すると EventFileWriter スレッドが FileNotFoundError で落ちる。
            """
            for attempt in range(max_retries):
                try:
                    gc.collect()
                    time.sleep(0.5 * (attempt + 1))
                    for item in output_path.iterdir():
                        if item.name in ("tmp", "runs"):
                            continue
                        try:
                            if item.is_dir():
                                shutil.rmtree(item, onerror=_handle_remove_readonly)
                            else:
                                item.unlink()
                        except Exception:
                            pass
                    return
                except Exception:
                    if attempt < max_retries - 1:
                        continue

        # === CPUメモリのみクリーンアップ (親プロセスではCUDAコンテキストを作成しない) ===
        gc.collect()

        # === ディスク上の中間生成物削除 ===
        output_dir = Path("models/output")
        if output_dir.exists():
            logger.info(f"Cleaning up HPO trial output files in {output_dir}")
            _cleanup_with_retry(output_dir)

import json
import os
import warnings
from pathlib import Path

import torch
import torch._dynamo
from datasets import disable_caching, load_dataset

warnings.filterwarnings("ignore", message=".*use_return_dict.*")

# Linuxの /tmp (tmpfs RAMディスク) の容量不足による [Errno 28] OOM を回避するため、
# 一時ディレクトリを十分な空き容量のあるローカルディスク（ext4）上に強制指定
os.environ["TMPDIR"] = str(Path("models/output/tmp").resolve())
# PyTorch CUDA Caching Allocator のフラグメンテーションと不要な Reserved メモリ拡大を防止
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

disable_caching()

from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainingArguments,
)

from src.common.logger import logger
from src.common.set_seed import set_seed
from src.training.callbacks import (
    DetailedLoggingCallback,
    HashSaveCallback,
    PeriodicEvaluationCallback,
)
from src.training.model_utils import (
    PackedDatasetWrapper,
    compute_config_hash,
    compute_dataset_fingerprint,
    compute_file_hash,
    create_model_config,
    get_optimal_num_proc,
    parallel_tokenize,
)


def _verify_flash_attention(precision: str) -> None:
    """SDPA経由でFlashAttentionバックエンドが正常にディスパッチ可能かを明示的に検証する（サイレントフォールバック防止）。"""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Skipping FlashAttention verification.")
        return

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    try:
        # Llamaモデルの標準ヘッド構造を模したテンソルを作成
        q = torch.randn(1, 12, 1024, 64, dtype=dtype, device="cuda")
        k = torch.randn(1, 12, 1024, 64, dtype=dtype, device="cuda")
        v = torch.randn(1, 12, 1024, 64, dtype=dtype, device="cuda")

        # FLASH_ATTENTIONのみを有効化した状態で実行テスト
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        logger.info(
            "Verification SUCCESS: Native FlashAttention backend is active "
            f"and verified via SDPA (dtype={precision})."
        )
    except Exception as e:
        logger.warning(
            f"Verification WARNING: FlashAttention backend could not be dispatched via SDPA. "
            f"PyTorch will silently fall back to slower math kernels. Error: {e}"
        )


def _warmup_liger_kernels(config: dict) -> None:
    """プロキシ/本番モデルの実寸法テンソルで Liger Triton JIT カーネルを個別コンパイル・分割同期する。"""
    import time

    if not torch.cuda.is_available():
        return

    t0 = time.perf_counter()
    model_params = config.get("model_params", {})
    hidden_size = model_params.get("hidden_size", 768)
    intermediate_size = model_params.get("intermediate_size", 3072)
    vocab_size = model_params.get("vocab_size", 64000)
    precision = config.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    logger.info(
        f"[Liger Warmup] Pre-compiling Liger Triton JIT kernels for model dimensions "
        f"(hidden={hidden_size}, intermediate={intermediate_size}, vocab={vocab_size}, precision={precision})..."
    )

    # Step 1: SwiGLU Triton Kernel Warmup
    try:
        t_step = time.perf_counter()
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction

        # gate_proj & up_proj output shape: [batch, seq_len, intermediate_size]
        a = torch.randn(1, 16, intermediate_size, dtype=dtype, device="cuda")
        b = torch.randn(1, 16, intermediate_size, dtype=dtype, device="cuda")
        _ = LigerSiLUMulFunction.apply(a, b)
        torch.cuda.synchronize()
        del a, b, _
        logger.info(
            f"[Liger Warmup 1/3] SwiGLU kernel JIT compiled in {(time.perf_counter() - t_step) * 1000:.2f}ms."
        )
    except Exception as swiglu_e:
        logger.warning(f"[Liger Warmup 1/3] SwiGLU warmup warning: {swiglu_e}")

    # Step 2: RMSNorm Triton Kernel Warmup
    try:
        t_step = time.perf_counter()
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        x = torch.randn(1, 16, hidden_size, dtype=dtype, device="cuda")
        w = torch.ones(hidden_size, dtype=dtype, device="cuda")
        _ = LigerRMSNormFunction.apply(x, w, 1e-6)
        torch.cuda.synchronize()
        del x, w, _
        logger.info(
            f"[Liger Warmup 2/3] RMSNorm kernel JIT compiled in {(time.perf_counter() - t_step) * 1000:.2f}ms."
        )
    except Exception as rms_e:
        logger.warning(f"[Liger Warmup 2/3] RMSNorm warmup warning: {rms_e}")

    # Step 3: CrossEntropy / LCE Triton Kernel Warmup
    try:
        t_step = time.perf_counter()
        from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

        # logits shape: [batch_seq, vocab_size], target shape: [batch_seq]
        logits = torch.randn(16, vocab_size, dtype=dtype, device="cuda")
        target = torch.randint(0, vocab_size, (16,), dtype=torch.long, device="cuda")
        # LigerCrossEntropyFunction.forward(ctx, _input, target, weight=None, ignore_index=-100, ...)
        _ = LigerCrossEntropyFunction.apply(logits, target, None, -100)
        torch.cuda.synchronize()
        del logits, target, _
        logger.info(
            f"[Liger Warmup 3/3] CrossEntropy kernel JIT compiled in {(time.perf_counter() - t_step) * 1000:.2f}ms."
        )
    except Exception as ce_e:
        logger.warning(f"[Liger Warmup 3/3] CrossEntropy warmup warning: {ce_e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[Liger Warmup] All Liger Triton kernels successfully pre-compiled & synchronized in {elapsed_ms:.2f}ms."
    )


def _check_and_warn_tdr_delay() -> None:
    """WSL2 / Windows 環境において Windows TdrDelay 及び TdrDdiDelay レジストリ設定をチェックし、不足時にアクション案内ログを出力する。"""
    import shutil
    import subprocess
    import sys

    # WSL2 または Windows 環境のチェック
    is_wsl = False
    if sys.platform == "linux":
        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    is_wsl = True
        except Exception:
            pass

    if not (is_wsl or sys.platform == "win32"):
        return  # Pure Linux環境等ではチェック不要

    reg_cmd = "/mnt/c/Windows/System32/reg.exe" if is_wsl else "reg"
    if not is_wsl and not shutil.which("reg"):
        return

    def _query_reg_key(key_name: str) -> int | None:
        try:
            res = subprocess.run(
                [
                    reg_cmd,
                    "query",
                    r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
                    "/v",
                    key_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if key_name in line and "REG_DWORD" in line:
                        parts = line.split()
                        raw_val = parts[-1]
                        return int(raw_val, 16) if raw_val.startswith("0x") else int(raw_val)
        except Exception:
            pass
        return None

    tdr_delay_val = _query_reg_key("TdrDelay")
    tdr_ddi_delay_val = _query_reg_key("TdrDdiDelay")

    # TdrDelay <= 3秒 または TdrDdiDelay <= 10秒 の場合に案内ログを出力
    need_warning = False
    if tdr_delay_val is None or tdr_delay_val <= 3:
        need_warning = True
    if tdr_ddi_delay_val is None or tdr_ddi_delay_val <= 10:
        need_warning = True

    if need_warning:
        delay_str = (
            f"{tdr_delay_val} sec" if tdr_delay_val is not None else "Not Set (Default 2 sec)"
        )
        ddi_str = (
            f"{tdr_ddi_delay_val} sec"
            if tdr_ddi_delay_val is not None
            else "Not Set (Default 5 sec)"
        )
        logger.warning(
            "\n"
            "================================================================================\n"
            "  [Windows WDDM TDR Alert] GPU Timeout Protection Check\n"
            "  ------------------------------------------------------------------------------\n"
            f"  Current Settings:\n"
            f"    - TdrDelay:    {delay_str} (Recommended: 60 sec)\n"
            f"    - TdrDdiDelay: {ddi_str} (Recommended: 60 sec)\n"
            "  \n"
            "  Liger Kernel / Triton JIT compilation on WSL2 may take 10-30 seconds on the\n"
            "  first forward pass, triggering an OS-level GPU driver reset\n"
            "  ('CUDA driver error: device not ready').\n"
            "  \n"
            "  RECOMMENDED ACTION STEPS:\n"
            "  1. Clear potentially corrupted Triton cache in WSL2 terminal:\n"
            "     rm -rf ~/.triton/cache\n"
            "  \n"
            "  2. Apply full TDR registry fix in Windows PowerShell (Administrator):\n"
            '     Start-Process powershell -Verb RunAs -ArgumentList \'-Command reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f; reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 60 /f\'\n'
            "  \n"
            "  3. REBOOT Windows (Restart PC) to apply registry changes to GPU driver.\n"
            "================================================================================"
        )


def _estimate_vram_and_warn(config: dict) -> None:
    """統一 VRAM 推定エンジン (src.common.vram_estimator) に委譲して見積もり・警告。"""
    if not torch.cuda.is_available():
        return

    try:
        from src.common.vram_estimator import VramConfig, estimate_training_vram

        mp = config.get("model_params", {})
        cfg = VramConfig(
            n_params=mp.get("n_params", 150_000_000),
            hidden_size=mp.get("hidden_size", 768),
            intermediate_size=mp.get("intermediate_size", 0),
            num_layers=mp.get("num_hidden_layers", 12),
            vocab_size=mp.get("vocab_size", 32000),
            micro_batch_size=config.get("per_device_batch_size", 1),
            seq_len=config.get("seq_len", 1024),
            precision=config.get("precision", "bf16"),
            optimizer_type=config.get("optim", "adamw_torch_fused"),
            use_liger_kernel=config.get("use_liger_kernel", True),
            torch_compile=config.get("torch_compile", False),
            total_vram_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3),
        )
        est = estimate_training_vram(cfg)
        logger.info(f"VRAM Consumption Estimation:\n{est.breakdown.log_string()}")
        if not est.breakdown.is_safe:
            logger.warning(
                f"VRAM Allocation Danger Warning: "
                f"Estimated Peak VRAM ({est.breakdown.total_estimated_gb:.2f} GB) "
                f"exceeds your physical GPU VRAM ({cfg.total_vram_gb:.2f} GB).\n"
                "The training is very likely to trigger OS-level silent "
                "memory paging (swapping) or Out-Of-Memory (OOM) errors.\n"
                "It is highly recommended to set 'torch_compile: false' "
                "or reduce 'per_device_batch_size'."
            )
    except Exception as e:
        logger.debug(f"Failed to calculate VRAM estimation: {e}")


def train(config: dict, tokenized_datasets=None, extra_callbacks=None):
    """
    統一された学習オーケストレーションフロー。
    """
    import time

    t_start = time.perf_counter()
    logger.info("[Train Engine Diag t=0.000ms] train() entry point reached.")

    # 0. Windows / WSL2 環境下の TdrDelay 設定状態をチェック・警告表示
    _check_and_warn_tdr_delay()

    # 0. 解決されたハイパーパラメータ/設定値の全出力
    logger.info(f"Resolved Configuration:\n{json.dumps(config, indent=2, ensure_ascii=False)}")

    # 0.5 理論的VRAM使用量の見積もりと警告の出力
    _estimate_vram_and_warn(config)

    # 1. 乱数シードの設定
    seed = config.get("seed", 42)
    deterministic_val = config.get("deterministic", False)
    set_seed(seed, deterministic=deterministic_val)
    logger.info(
        f"[Train Engine Diag t={(time.perf_counter() - t_start) * 1000:.2f}ms] Seed set successfully."
    )

    # 1.5 TF32 (TensorFloat-32) の有効化（Ampere世代以降のGPUでの行列演算高速化）
    if config.get("allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 (TF32) enabled for matmul and cudnn.")

    # 1.6 torch.compile と Liger Kernel の同時使用に対する警告（グラフブレイクによる減速リスク）
    if config.get("torch_compile", False) and config.get("use_liger_kernel", False):
        logger.warning(
            "Performance Warning: Both 'torch_compile' and "
            "'use_liger_kernel' are enabled. "
            "Combining Triton custom autograd functions in Liger "
            "with torch.compile Dynamo tracing often causes severe "
            "graph breaks, triggering eager fallbacks and degrading "
            "step speed. It is highly recommended to disable one of "
            "them: Use Liger Kernel alone for memory savings, or "
            "torch.compile alone for compute optimization."
        )

    # 1.65 torch.compile コンパイラ最適化設定 (Graph break の抑制)
    torch._dynamo.config.capture_scalar_outputs = True

    # 1.7 FlashAttention のディスパッチ検証（サイレントフォールバックの防止）
    _verify_flash_attention(config.get("precision", "bf16"))

    # 1.8 Liger Kernel Triton JIT 事前 Warmup (初回 Forward Pass での長時間GPUストール防止)
    if config.get("use_liger_kernel", False):
        _warmup_liger_kernels(config)

    # 2. 現在の合成設定とデータセットのハッシュ値の算出（整合性チェック用）
    #    configs/config.yaml は Hydra defaults のポインタのため、適用される合成設定全体をハッシュ化する
    logger.info("Computing merged config hash (base_config + scaling_config + hpo_config)...")
    current_config_hash = compute_config_hash(config)
    data_path_str = config.get("data_path", "data/dataset.jsonl")
    logger.info(f"Computing data hash: {data_path_str}")
    current_data_hash = compute_file_hash(data_path_str)

    # 4. データセットおよびデータベースのフィンガープリントを取得
    logger.info(f"Computing dataset fingerprint: {data_path_str}")
    data_fingerprint = compute_dataset_fingerprint(data_path_str)

    logger.info(
        "Training initialization",
        extra={
            "seed": seed,
            "data_path": data_path_str,
            "dataset_fingerprint": data_fingerprint,
        },
    )

    # 5. チェックポイントからの再開（Resume）処理の解決
    resume_checkpoint = config.get("resume_from_checkpoint") or config.get("resume")

    # "checkpoint-latest" が指定された場合、または True の場合は最新のチェックポイントを自動探索
    if resume_checkpoint is True or (
        isinstance(resume_checkpoint, str) and "checkpoint-latest" in resume_checkpoint
    ):
        from src.training.model_utils import get_checkpoints

        checkpoints = get_checkpoints(sort_by="mtime")
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1][1])
            logger.info(f"Resolved checkpoint-latest to: {resume_checkpoint}")
        else:
            resume_checkpoint = None
            logger.warning("No checkpoint found to resume from. Starting from scratch.")

    # 6. チェックポイントデータの整合性検証（ハッシュ値による構成変更チェック）
    if isinstance(resume_checkpoint, str) and Path(resume_checkpoint).exists():
        checkpoint_path = Path(resume_checkpoint)
        hash_file = checkpoint_path / "hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file) as f:
                    saved_hashes = json.load(f)
                saved_config_hash = saved_hashes.get("config_hash")
                saved_data_hash = saved_hashes.get("data_hash")

                # チェックポイント保存時と現在の設定・データが異なる場合はエラーとして学習を中断する
                if saved_config_hash != current_config_hash or saved_data_hash != current_data_hash:
                    logger.error(
                        "Configuration or training dataset has changed "
                        "since the checkpoint was saved!"
                    )
                    logger.error(
                        f"Saved Config Hash: {saved_config_hash} | "
                        f"Current Config Hash: {current_config_hash}"
                    )
                    logger.error(
                        f"Saved Data Hash: {saved_data_hash} | "
                        f"Current Data Hash: {current_data_hash}"
                    )
                    raise ValueError(
                        "Cannot resume training: config.yaml or training "
                        "dataset does not match the checkpoint."
                    )
                else:
                    logger.info("Configuration and dataset hashes match. Verification successful.")
            except Exception as e:
                logger.error(f"Failed to verify checkpoint hashes: {e}")
                raise
        else:
            logger.warning(
                f"No hashes.json found in checkpoint {resume_checkpoint}. "
                "Proceeding without verification."
            )
    else:
        if not isinstance(resume_checkpoint, str):
            resume_checkpoint = None

    # 7. トークナイザーの読み込み（高速なRust実装版を優先）
    tokenizer_path = Path(config.get("tokenizer_path", "data/tokenizer.json"))
    if tokenizer_path.suffix == ".json" and tokenizer_path.exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # 特殊トークンの明示的な設定
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    # 8. データセットのロードとトークン化処理（前処理が渡されていない場合のみ実施）
    if tokenized_datasets is None:
        logger.info("Starting dataset loading...")

        # 物理パス（シンボリックリンクを含む）の解決状況を特定しログ出力
        resolved_train_path = Path(data_path_str).resolve()
        is_symlink = Path(data_path_str).is_symlink()
        symlink_msg = " (symlink)" if is_symlink else ""
        logger.info(
            f"Loading train dataset from: {data_path_str}{symlink_msg} "
            f"-> Resolved physical path: {resolved_train_path}"
        )

        if str(resolved_train_path).startswith("/mnt/"):
            logger.warning(
                f"Performance Alert: Resolved dataset path "
                f"'{resolved_train_path}' is located under WSL mount "
                "directory '/mnt/'. Accessing Windows filesystems from "
                "WSL2 is extremely slow (10x-20x slower). "
                "It is strongly recommended to copy the dataset file "
                "directly into WSL2 storage (e.g. ~/dataset.jsonl)."
            )

        data_files = {"train": data_path_str}
        if config.get("val_data_path"):
            val_path_str = config["val_data_path"]
            resolved_val_path = Path(val_path_str).resolve()
            logger.info(
                f"Loading validation dataset from: {val_path_str} "
                f"-> Resolved physical path: {resolved_val_path}"
            )
            data_files["validation"] = val_path_str

        # JSONLファイルからデータセットをロード
        ds = load_dataset("json", data_files=data_files)
        remove_columns = [c for c in ds["train"].column_names if c in {"text", "metadata"}]

        packing = config.get("packing", False)
        seq_len = config.get("seq_len", 1024)

        # プラットフォーム自動判定で並列トークナイズ (Windows: ThreadPool, Linux: multiprocessing)
        num_proc = get_optimal_num_proc()
        logger.info(f"Tokenizing dataset with parallelism={num_proc}")

        ds = parallel_tokenize(
            ds,
            tokenizer,
            seq_len=seq_len,
            padding=not packing,
            remove_columns=remove_columns,
            max_workers=num_proc,
            batch_size=config.get("tokenization", {}).get("batch_size", 1000),
        )

        # Sequence Packingの適用
        if packing:
            logger.info("Packing dataset enabled. Eliminating pad tokens...")
            packed_ds = {}
            for split in ds:
                wrapper = PackedDatasetWrapper(ds[split], seq_len, tokenizer.eos_token_id)
                packed_ds[split] = wrapper()
            from datasets import DatasetDict

            ds = DatasetDict(packed_ds)

        # デバッグや軽量テスト用途のデータ一部抽出 (data_fraction < 1.0 の場合)
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
        # 外部から提供されたトークン化済みデータを設定
        train_ds = tokenized_datasets["train"]
        eval_ds = tokenized_datasets.get("validation")

    # 9. モデルの設定と初期化 (Llamaアーキテクチャ)
    model_config = create_model_config(config, tokenizer)
    model = LlamaForCausalLM(model_config)

    # トークナイザーの語彙数に合わせて埋め込み層のサイズを調整
    model.resize_token_embeddings(len(tokenizer))

    # 9.5 CUDAカーネル事前コンパイル (Windows TDR防止)
    # WARNING: Pre-compilation is skipped for WSL2 environments because Triton/FlashAttention
    # kernel compilation can exceed Windows TDR timeout (default: 2s), triggering GPU reset.
    # The actual fix is to increase TdrDelay via Windows Registry:
    #   HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay = 30 (DWORD)
    # then restart WSL2.

    # Liger Kernel 互換性重視のため、全層フル Gradient Checkpointing (use_reentrant=False) に一本化
    use_gradient_checkpointing = config.get("gradient_checkpointing", True)
    if use_gradient_checkpointing:
        logger.info(
            "Enabled Standard Full Gradient Checkpointing (use_reentrant=False) for 100% Liger Kernel compatibility."
        )

    # 10. TrainingArguments (学習パラメータ) の構築
    precision = config.get("precision", "bf16")
    output_dir = config.get("output_dir", "models/output")
    hpo_config = config.get("hpo", config)
    if not isinstance(hpo_config, dict):
        hpo_config = config
    else:
        # hpo サブ辞書に無いスケジューラ系キーはトップレベルからフォールバック
        # (hpo/main.py 等から直接渡される config はトップレベルに設定を持つため)
        hpo_config = dict(hpo_config)
        for _scheduler_key in (
            "lr_scheduler_type",
            "warmup_steps",
            "warmup_ratio",
            "constant_steps",
            "constant_ratio",
            "num_cycles",
        ):
            if _scheduler_key not in hpo_config and config.get(_scheduler_key) is not None:
                hpo_config[_scheduler_key] = config[_scheduler_key]

    max_steps = config.get("max_steps", -1)

    # configs/scaling_config.yaml または HPO からのマイクロバッチ解決
    batch_size_seqs = config.get(
        "batch_size_seqs", config.get("training", {}).get("batch_size_seqs", 64)
    )
    per_device_batch = config.get(
        "per_device_batch_size", config.get("training", {}).get("per_device_batch_size", None)
    )
    grad_accum_steps = config.get(
        "grad_accum_steps", config.get("training", {}).get("grad_accum_steps", None)
    )

    if per_device_batch is None or grad_accum_steps is None:
        from src.training.model_utils import calculate_optimal_batch_split

        vram_cap = config.get("vram_limit_gb", 4.0)
        per_device_batch, grad_accum_steps = calculate_optimal_batch_split(
            total_batch_size=batch_size_seqs,
            vram_gb=vram_cap,
            n_params=model_config.vocab_size * model_config.hidden_size
            + model_config.num_hidden_layers * 12 * (model_config.hidden_size**2),
            seq_len=model_config.max_position_embeddings,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_hidden_layers,
            vocab_size=model_config.vocab_size,
            precision=precision,
            selective_checkpointing=False,
        )

    logger.info(
        f"[Train Engine] Resolved Training Batch Split: "
        f"per_device_batch_size={per_device_batch}, grad_accum_steps={grad_accum_steps} "
        f"(Total Batch Seqs={per_device_batch * grad_accum_steps})"
    )

    if max_steps > 0:
        import math

        steps_per_epoch = max(1, len(train_ds) // (per_device_batch * grad_accum_steps))
        num_epochs = max(100, math.ceil(max_steps / steps_per_epoch))
    else:
        num_epochs = config.get("num_epochs", 3)

    # Pagedオプティマイザが選択されている場合、性能低下の可能性を警告
    optim_selected = config.get("optim", "adamw_torch_fused")
    if "paged" in optim_selected:
        logger.warning(
            f"Paged optimizer '{optim_selected}' is selected. "
            "If GPU VRAM limits are exceeded, optimizer states will be "
            "paged to CPU system memory, which will severely degrade "
            "training speed."
        )

    # LRスケジューラ設定解決 (Step Law推奨: Constant+Cosine)
    scheduler_type = hpo_config.get("lr_scheduler_type", "cosine")  # HF標準: cosine / linear 等
    custom_scheduler = scheduler_type in ("constant_cosine", "step_law")
    if custom_scheduler:
        # カスタムスケジューラ使用時は HF 標準の cosine をベースに設定
        # （後で create_scheduler で上書き）
        hf_scheduler_type = "cosine"
    else:
        hf_scheduler_type = scheduler_type

    warmup_steps = hpo_config.get("warmup_steps", 0)
    warmup_ratio = hpo_config.get("warmup_ratio", 0.03)
    constant_steps = hpo_config.get("constant_steps", 0)
    constant_ratio = hpo_config.get("constant_ratio", 0.1)
    if warmup_steps == 0 and warmup_ratio > 0:
        # epochモード時の比率解決
        estimated_total_steps = max_steps if max_steps > 0 else 10000
        warmup_steps = int(estimated_total_steps * warmup_ratio)
    if constant_steps == 0 and constant_ratio > 0:
        estimated_total_steps = max_steps if max_steps > 0 else 10000
        constant_steps = int(estimated_total_steps * constant_ratio)

    # dataloader_num_workers の動的パフォーマンス適合設定
    user_num_workers = config.get("dataloader_num_workers")
    if user_num_workers is not None and user_num_workers > 0:
        dataloader_num_workers = user_num_workers
    else:
        import os

        cpu_cores = os.cpu_count() or 2
        # 非同期バックグラウンド並列フェッチ (Windows Native/WSL2/Linux 全対応)
        # spawn + persistent_workers + pin_memory により CUDA IPC 衝突なしに高速読み込み
        dataloader_num_workers = min(4, max(2, cpu_cores // 2))

    logger.info(
        f"DataLoader Auto-Optimization: num_workers={dataloader_num_workers}, "
        f"persistent_workers={dataloader_num_workers > 0}, pin_memory=True"
    )

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=hpo_config.get("max_lr_2d", 3e-4),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # 安定性とコンパイラ互換性のための非再帰方式
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type=hf_scheduler_type,  # constant_cosine/step_law 時は cosine (後で上書き)
        warmup_steps=warmup_steps,
        weight_decay=hpo_config.get("weight_decay", 0.1),
        adam_beta2=hpo_config.get("beta2", 0.95),
        max_grad_norm=hpo_config.get("grad_clip", 1.0),
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 2),  # ローカルチェックポイント保持数制限
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.get("eval_steps", 1000) if eval_ds is not None else None,
        logging_steps=config.get("logging_steps", 200),
        report_to=["tensorboard"],  # TensorBoardによる学習監視の有効化
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        seed=seed,
        remove_unused_columns=False,  # カスタムデータコレーター利用時のカラム自動削除防止
        optim=config.get("optim", "adamw_torch_fused"),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        torch_empty_cache_steps=config.get("torch_empty_cache_steps", 100),
        dataloader_prefetch_factor=config.get("dataloader_prefetch_factor", 2)
        if dataloader_num_workers > 0
        else None,
        torch_compile=config.get("torch_compile", False),
        use_liger_kernel=config.get("use_liger_kernel", False),
        disable_tqdm=True,  # tqdm進捗バー無効化（独自ログのみ使用）
    )

    # 10.5 cuDNN最速カーネルサーチの有効化 & Tensor Core medium 精度指定
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark & Tensor Core medium precision.")

    # 11. コールバックの設定
    callbacks = [
        HashSaveCallback(
            config_hash=current_config_hash, data_hash=current_data_hash
        ),  # hashes.jsonの自動作成
        DetailedLoggingCallback(
            log_every_n_steps=config.get("logging_steps", 200)
        ),  # 詳細なステップ別メトリクスのログ出力
        PeriodicEvaluationCallback(
            eval_every_n_steps=config.get("eval_steps", 1000),
            eval_prompts=config.get("eval_prompts"),
            max_new_tokens=config.get("eval_max_new_tokens", 64),
            temperature=config.get("eval_temperature", 0.8),
            top_p=config.get("eval_top_p", 0.95),
            divergence_threshold=config.get("divergence_threshold"),
            log_generations=config.get("eval_log_generations", True),
        ),  # 定期評価: perplexity + 生成サンプル + 発散検知
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # 12. Trainerインスタンスの生成 (Muon(2D) + AdamW(1D) 分離最適化)
    from transformers import default_data_collator

    from src.training.trainer.dual_optimizer_trainer import DualOptimizerTrainer

    # Newton-Schulz直交化ステップ判定用モデルパラメーター情報を追加
    hpo_config_with_model = dict(hpo_config)
    hpo_config_with_model["hidden_size"] = model_config.hidden_size
    hpo_config_with_model["n_params"] = config.get("model_params", {}).get("n_params", 150_000_000)

    trainer = DualOptimizerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator
        if config.get("packing", False)
        else DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
        split_optimizer_config=hpo_config_with_model,
    )

    # 既定のPrinterCallbackを削除して重複ログを防止
    # （disable_tqdm=True時に標準出力へ辞書をprintする）
    from transformers.trainer_callback import PrinterCallback

    trainer.remove_callback(PrinterCallback)

    # 詳細ログ出力コールバックにTrainerオブジェクトの参照を渡す
    for cb in callbacks:
        if isinstance(cb, DetailedLoggingCallback):
            cb.trainer = trainer

    # 13. 学習プロセスの実行
    from transformers import TrainerCallback

    class DiagFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug(
                f"[Step Diag t={(time.perf_counter() - t_start) * 1000:.2f}ms] Step {state.global_step + 1} BEGIN"
            )

        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.debug(
                f"[Step Diag t={(time.perf_counter() - t_start) * 1000:.2f}ms] Step {state.global_step + 1} END"
            )

    trainer.add_callback(DiagFirstStepCallback())
    logger.info(
        f"[Train Engine Diag t={(time.perf_counter() - t_start) * 1000:.2f}ms] Starting trainer.train()..."
    )
    train_exec_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_exec_duration = (time.perf_counter() - train_exec_start) * 1000
    logger.info(
        f"[Train Engine Diag t={(time.perf_counter() - t_start) * 1000:.2f}ms] "
        f"trainer.train() finished cleanly in {train_exec_duration:.2f}ms."
    )

    # 通常学習の場合、最終結果メトリクスをJSONファイルに保存
    if max_steps == -1:
        with open("last_run_result.json", "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f)

    # 14. 学習済みモデル・トークナイザーの保存処理
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return train_result.training_loss

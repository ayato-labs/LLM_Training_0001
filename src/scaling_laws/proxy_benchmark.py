"""Dynamic GPU Proxy Benchmark & Physical Profiling

指定された GPU 上でリアルタイムにプロキシモデルを実行し、
実効スループット (tokens/sec) および 物理 VRAM 空き容量を精度高く動的実測プロファイリング。
"""

import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from src.common.config_utils import load_base_config_yaml
from src.common.logger import logger


def detect_data_path_and_tokens() -> tuple[str, float | None]:
    """configs/base_config.yaml から data_path と tokenizer_path を取得し、
    本番と同一の PreTrainedTokenizerFast と同一の text 抽出ロジックで
    データセットの総トークン数を正確に算定する。

    訓練パイプライン (train_engine.py:469-470, model_utils.py:203-212) と同様に:
      1. JSONL の各行から "text" フィールドのみを抽出
      2. PreTrainedTokenizerFast で tokenize
      3. 全行の文字数をカウント + 均一サンプリングで token/char 比率を推定
    """
    cfg = load_base_config_yaml()
    data_path = str(cfg.get("data_path", "data/dataset.jsonl"))
    tokenizer_path = str(cfg.get("tokenizer_path", "data/tokenizer.json"))

    path_obj = Path(data_path)
    if not path_obj.exists():
        return data_path, None

    try:
        from transformers import PreTrainedTokenizerFast

        tokenizer = None
        if Path(tokenizer_path).exists():
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        if tokenizer is None:
            file_size_bytes = path_obj.stat().st_size
            est_tokens = file_size_bytes / 3.5
            return data_path, float(est_tokens)

        logger.info(f"Counting dataset tokens from {data_path} (this may take a few minutes)...")

        # --- First pass: count total chars of all "text" fields ---
        total_text_chars = 0
        record_count = 0
        with open(path_obj, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                total_text_chars += len(text)
                record_count += 1

        # --- Second pass: uniform sampling for token/char ratio ---
        sample_interval = max(1, record_count // 2000)  # ~2000 samples
        sample_chars = 0
        sample_tokens = 0
        sample_count = 0

        with open(path_obj, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % sample_interval == 0:
                    data = json.loads(line)
                    text = data.get("text", "")
                    sample_chars += len(text)
                    sample_tokens += len(tokenizer.encode(text))
                    sample_count += 1

        if sample_chars == 0:
            return data_path, 0.0

        ratio = sample_tokens / sample_chars
        total_tokens = total_text_chars * ratio

        logger.info(
            f"Dataset: {record_count:,} chapters, "
            f"{total_text_chars:,} chars, "
            f"~{total_tokens:,.0f} tokens "
            f"(sampled {sample_count:,} chapters, token/char={ratio:.4f})"
        )

        return data_path, float(total_tokens)

    except Exception as e:
        logger.warning(
            f"Could not estimate dataset tokens via PreTrainedTokenizerFast from {data_path}: {e}"
        )
        return data_path, None


def detect_seq_len_from_config() -> int:
    """configs/base_config.yaml から事前学習の基本 seq_len を一元取得"""
    cfg = load_base_config_yaml()
    if "training" in cfg and "seq_len" in cfg["training"]:
        return int(cfg["training"]["seq_len"])
    return 1024


def detect_gpu_info() -> dict[str, Any]:
    """GPUの物理仕様および演算能力を動的に自動取得"""
    if not torch.cuda.is_available():
        return {
            "device_name": "CPU (CUDA Unavailable)",
            "total_vram_gb": 4.0,
            "sm_count": 0,
            "is_cuda": False,
        }

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    sm_count = getattr(props, "multi_processor_count", 0)

    return {
        "device_name": props.name,
        "total_vram_gb": round(vram_gb, 2),
        "sm_count": sm_count,
        "is_cuda": True,
    }


def extract_throughput_from_recent_logs() -> float | None:
    """最新のログファイルから直近の訓練スループット (tokens/sec) を自動抽出"""
    log_paths = list(Path(".").glob("**/logs/*.log")) + list(Path(".").glob("*.log"))
    if not log_paths:
        return None

    log_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    pattern = re.compile(r"(\d+\.\d+)s/it")
    for log_path in log_paths[:3]:
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in reversed(lines):
                match = pattern.search(line)
                if match:
                    sec_per_it = float(match.group(1))
                    if sec_per_it > 0:
                        tokens_per_it = 1024 * 32
                        tps = tokens_per_it / sec_per_it
                        return round(tps, 1)
        except Exception:
            continue
    return None


def run_quick_proxy_benchmark(seq_len: int = 1024, batch_size: int = 16) -> tuple[float, int]:
    """指定の seq_len および batch_size で GPU 上のプロキシモデルを実際に動的実行し、
    実効スループット (tokens/sec) および 実測パラメータ数を完全リアルタイム計測。
    base_config.yaml の gradient_checkpointing 設定を反映する。
    """
    if not torch.cuda.is_available():
        return 500.0, 27_450_000

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    base_cfg = load_base_config_yaml()
    use_grad_ckpt = base_cfg.get("gradient_checkpointing", True)
    optim_name = base_cfg.get("training", {}).get("optim", base_cfg.get("optim", "adamw_bnb_8bit"))

    target_bs = max(1, batch_size)

    while target_bs >= 1:
        try:
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaModel

            cfg = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                intermediate_size=1376,
                use_cache=False,
            )
            device = torch.device("cuda:0")
            model = LlamaModel(cfg).to(device=device, dtype=torch.bfloat16)

            if use_grad_ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            ref_n = sum(p.numel() for p in model.parameters())

            if "8bit" in optim_name:
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
                except Exception:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            dummy_input = torch.randint(0, 32000, (target_bs, seq_len), device=device)
            out = model(dummy_input)
            loss = out.last_hidden_state.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()

            start_t = time.time()
            n_steps = 3
            for _ in range(n_steps):
                out = model(dummy_input)
                loss = out.last_hidden_state.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize()
            elapsed = time.time() - start_t
            sec_per_step = elapsed / n_steps
            tps = (target_bs * seq_len) / sec_per_step

            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            return round(tps, 1), ref_n
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            target_bs = target_bs // 2

    return 500.0, 27_450_000


def detect_real_available_vram_gb(device_id: int = 0) -> float:
    """CUDAコンテキストを強制初期化し、現在のシステムにおける物理的な真の空きVRAM容量 (GB) を動的取得"""
    if not torch.cuda.is_available():
        return 4.0

    try:
        dummy = torch.zeros(1, device=f"cuda:{device_id}")
        del dummy
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_id)
        free_vram_gb = free_bytes / (1024**3)
        return round(free_vram_gb, 2)
    except Exception as e:
        logger.warning(f"Failed to query real available VRAM via mem_get_info: {e}")
        return 4.0

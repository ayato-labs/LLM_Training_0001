"""Benchmark script for comparing AdamW 8bit vs SplitOptimizer (Muon + AdamW 8bit)

Usage:
    python -m src.training.optimizers.benchmark_muon
"""

import gc
import time
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from src.training.trainer.dual_optimizer_trainer import SplitOptimizer
from src.common.logger import logger


def run_optimizer_benchmark(optimizer_type: str = "muon", num_steps: int = 20, seq_len: int = 1024, batch_size: int = 4):
    if not torch.cuda.is_available():
        logger.error("CUDA is required for GPU optimizer benchmark.")
        return

    device = torch.device("cuda:0")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # シンプルかつ計算密度の高い 2D 重み行列を持つプロキシモデル
    class SimpleProxyLLM(torch.nn.Module):
        def __init__(self, hidden_size=768, num_layers=12):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
            ])
            self.embed_tokens = torch.nn.Embedding(32000, hidden_size)
            self.lm_head = torch.nn.Linear(hidden_size, 32000, bias=False)

        def forward(self, x):
            h = self.embed_tokens(x)
            for layer in self.layers:
                h = torch.nn.functional.relu(layer(h))
            return self.lm_head(h)

    model = SimpleProxyLLM().to(device=device, dtype=torch.bfloat16)
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    opt_config = {
        "hidden_size": 768,
        "n_params": 150_000_000,
        "max_lr_2d": 3e-4,
        "max_lr_1d": 3e-3,
        "weight_decay": 0.1,
    }

    if optimizer_type == "muon":
        optimizer = SplitOptimizer(model, opt_config)
    else:
        # Standard AdamW 8bit for comparison
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4)
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for _ in range(3):
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Benchmark loop
    start_time = time.perf_counter()
    for _ in range(num_steps):
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    avg_step_ms = (total_time / num_steps) * 1000.0
    total_tokens = num_steps * batch_size * seq_len
    tokens_per_sec = total_tokens / total_time
    max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    logger.info(
        f"[{optimizer_type.upper()}] Steps: {num_steps} | "
        f"Step Time: {avg_step_ms:.2f} ms | "
        f"Throughput: {tokens_per_sec:.1f} tokens/sec | "
        f"Peak VRAM: {max_vram_gb:.2f} GB"
    )

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "step_ms": avg_step_ms,
        "tps": tokens_per_sec,
        "vram_gb": max_vram_gb,
    }


if __name__ == "__main__":
    logger.info("=== Starting Optimizer Performance Benchmark ===")
    res_adamw = run_optimizer_benchmark("adamw_8bit", num_steps=25)
    res_muon = run_optimizer_benchmark("muon", num_steps=25)

    if res_adamw and res_muon:
        speed_ratio = (res_muon["tps"] / res_adamw["tps"]) * 100
        logger.info(f"=== Benchmark Results Summary ===")
        logger.info(f"AdamW 8bit Throughput: {res_adamw['tps']:.1f} tokens/sec ({res_adamw['step_ms']:.2f} ms/step)")
        logger.info(f"Muon (SplitOpt) Throughput: {res_muon['tps']:.1f} tokens/sec ({res_muon['step_ms']:.2f} ms/step)")
        logger.info(f"Muon Throughput relative to AdamW 8bit: {speed_ratio:.1f}%")

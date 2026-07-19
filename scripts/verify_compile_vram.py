"""
torch.compile VRAM 検証スクリプト (Windows対応版)
==================================================
- Windows: aot_eager (Triton不要) + none で比較
- Linux: 全モード (none/default/reduce-overhead/max-autotune/max-autotune+cached)
- 縮小モデル (~30M) で 5ステップ実行
- JSON結果 + コンソールサマリ出力
"""

import gc
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

torch.set_float32_matmul_precision("high")

# -- 定数 --
STEPS = 5
SEQ_LEN = 512
BATCH_SIZE = 2
HIDDEN = 512
LAYERS = 6
HEADS = 8
KV_HEADS = 2
INTERMEDIATE = 2048
VOCAB = 8000

CACHE_DIR = Path("models/output/inductor_cache_verify")
RESULTS_DIR = Path("results")


def _check_triton_available() -> bool:
    try:
        import triton
        ver = getattr(triton, "__version__", "0.0")
        return ver != "0.0"
    except ImportError:
        return False


def _get_compile_modes():
    is_linux = sys.platform == "linux"
    has_triton = _check_triton_available()

    if is_linux and has_triton:
        return [
            ("none", None, False),
            ("default", "default", False),
            ("reduce-overhead", "reduce-overhead", False),
            ("max-autotune", "max-autotune", False),
            ("max-autotune+cached", "max-autotune", True),
        ]
    elif is_linux and not has_triton:
        print("[INFO] Linux but no Triton. Testing aot_eager + none.")
        return [
            ("none", None, False),
            ("aot_eager", "default", False),
        ]
    else:
        print("[INFO] Windows detected. Testing aot_eager (Triton-free compile) vs none.")
        print("       aot_eager uses AOT autograd but eager kernels (no speedup, compile overhead only).")
        print("       For full max-autotune test, run under WSL/Linux with Triton.\n")
        return [
            ("none", None, False),
            ("aot_eager", "default", False),
        ]


@dataclass
class StepMetrics:
    step: int
    allocated_gb: float
    reserved_gb: float
    fragmentation_gb: float
    step_time_s: float


@dataclass
class ModeResult:
    mode: str
    steps: list[StepMetrics] = field(default_factory=list)
    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    peak_fragmentation_gb: float = 0.0
    avg_step_time_s: float = 0.0
    compile_time_s: float = 0.0
    total_time_s: float = 0.0
    error: str | None = None


def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_mem_gb():
    alloc = torch.cuda.max_memory_allocated() / (1024**3)
    res = torch.cuda.max_memory_reserved() / (1024**3)
    return alloc, res


def create_small_model():
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        rope_theta=10000.0,
        max_position_embeddings=SEQ_LEN,
        use_cache=False,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")
    return model


def make_dummy_batch():
    ids = torch.randint(0, VOCAB, (BATCH_SIZE, SEQ_LEN), device="cuda")
    labels = ids.clone()
    return {"input_ids": ids, "labels": labels}


def run_one_mode(mode_name, compile_mode, use_cache=False):
    result = ModeResult(mode=mode_name)

    try:
        reset_cuda()

        print(f"  Building model...")
        model = create_small_model()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        model.train()

        if compile_mode is not None:
            print(f"  Compiling (mode={compile_mode}, cache={use_cache})...")
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR.resolve())

            t0 = time.perf_counter()

            if mode_name == "aot_eager":
                compiled_model = torch.compile(model, backend="aot_eager")
            else:
                compiled_model = torch.compile(model, mode=compile_mode)

            result.compile_time_s = time.perf_counter() - t0
            print(f"  Compile done in {result.compile_time_s:.2f}s")
        else:
            compiled_model = model
            print(f"  No compile (eager mode)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        print(f"  Running {STEPS} steps...")
        reset_cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        step_times = []
        for step in range(STEPS):
            batch = make_dummy_batch()

            t1 = time.perf_counter()
            out = compiled_model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            alloc, res = get_peak_mem_gb()
            frag = max(0.0, res - alloc)
            elapsed = t2 - t1
            step_times.append(elapsed)

            sm = StepMetrics(
                step=step,
                allocated_gb=round(alloc, 4),
                reserved_gb=round(res, 4),
                fragmentation_gb=round(frag, 4),
                step_time_s=round(elapsed, 4),
            )
            result.steps.append(sm)
            print(f"    Step {step}: alloc={alloc:.3f}GB  res={res:.3f}GB  frag={frag:.3f}GB  time={elapsed:.3f}s")

        result.peak_allocated_gb = round(max(s.allocated_gb for s in result.steps), 4)
        result.peak_reserved_gb = round(max(s.reserved_gb for s in result.steps), 4)
        result.peak_fragmentation_gb = round(max(s.fragmentation_gb for s in result.steps), 4)
        result.avg_step_time_s = round(sum(step_times) / len(step_times), 4)
        result.total_time_s = round(sum(step_times), 4)

        del compiled_model, model, optimizer
        reset_cuda()

    except Exception as e:
        result.error = str(e)
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        reset_cuda()

    return result


def print_summary(results):
    sep = "=" * 90
    thin = "-" * 90

    print(f"\n{sep}")
    print("  VRAM Verification Results")
    print(sep)

    header = f"{'Mode':<22} {'Peak Alloc GB':>14} {'Peak Res GB':>14} {'Frag GB':>10} {'Avg Step':>10} {'Compile':>10}"
    print(header)
    print(thin)

    none_result = next((r for r in results if r.mode == "none"), None)

    for r in results:
        if r.error:
            print(f"{r.mode:<22} {'ERROR':>14} {'':>14} {'':>10} {'':>10} {'':>10}")
            continue

        delta = ""
        if r.mode != "none" and none_result and not none_result.error:
            d = r.peak_allocated_gb - none_result.peak_allocated_gb
            delta = f" ({d:+.3f})"

        print(
            f"{r.mode:<22} {r.peak_allocated_gb:>14.4f}{delta:>10} "
            f"{r.peak_reserved_gb:>14.4f} {r.peak_fragmentation_gb:>10.4f} "
            f"{r.avg_step_time_s:>10.4f} {r.compile_time_s:>10.2f}"
        )

    print(f"\n{thin}")
    print("  Verdict:")

    if none_result and not none_result.error:
        for r in results:
            if r.mode == "none" or r.error:
                continue
            diff_alloc = r.peak_allocated_gb - none_result.peak_allocated_gb
            diff_time = r.avg_step_time_s - none_result.avg_step_time_s

            if diff_alloc > 0.5:
                print(f"  [!] {r.mode}: VRAM +{diff_alloc:.3f}GB vs baseline (compile overhead significant)")
            elif diff_alloc > 0.1:
                print(f"  [~] {r.mode}: VRAM +{diff_alloc:.3f}GB vs baseline (compile overhead small)")
            else:
                print(f"  [OK] {r.mode}: VRAM overhead negligible ({diff_alloc:+.3f}GB)")

            if diff_time > 0.5:
                print(f"  [!] {r.mode}: step time +{diff_time:.3f}s vs baseline (slower)")
            elif diff_time < -0.1:
                print(f"  [OK] {r.mode}: step time {diff_time:+.3f}s vs baseline (faster)")

    for r in results:
        if r.error:
            continue
        if r.peak_fragmentation_gb > 0.5:
            print(f"  [!] [{r.mode}] Fragmentation {r.peak_fragmentation_gb:.3f}GB detected")

    for r in results:
        if r.error or len(r.steps) < 2:
            continue
        times = [s.step_time_s for s in r.steps]
        mean = sum(times) / len(times)
        if mean == 0:
            continue
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std = variance**0.5
        cv = std / mean
        if cv > 0.3:
            print(f"  [!] [{r.mode}] Step time variance high (CV={cv:.2f}) - possible recompilation")

    print(sep)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if torch.cuda.is_available()
        else 0
    )

    print("torch.compile VRAM Verification")
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  GPU      : {gpu_name}")
    print(f"  VRAM     : {vram_gb} GB")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Triton   : {'available' if _check_triton_available() else 'NOT available'}")
    print(f"  Config   : SEQ_LEN={SEQ_LEN}, BATCH={BATCH_SIZE}, HIDDEN={HIDDEN}, LAYERS={LAYERS}, STEPS={STEPS}")
    print()

    modes = _get_compile_modes()
    all_results = []

    for mode_name, compile_mode, use_cache in modes:
        print(f"--- {mode_name} ---")
        r = run_one_mode(mode_name, compile_mode, use_cache=use_cache)
        all_results.append(r)
        print()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"vram_verify_{ts}.json"

    output = {
        "timestamp": ts,
        "platform": platform.system(),
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "pytorch_version": torch.__version__,
        "triton_available": _check_triton_available(),
        "config": {
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN,
            "num_layers": LAYERS,
            "num_steps": STEPS,
        },
        "results": [
            {
                "mode": r.mode,
                "peak_allocated_gb": r.peak_allocated_gb,
                "peak_reserved_gb": r.peak_reserved_gb,
                "peak_fragmentation_gb": r.peak_fragmentation_gb,
                "avg_step_time_s": r.avg_step_time_s,
                "compile_time_s": r.compile_time_s,
                "total_time_s": r.total_time_s,
                "error": r.error,
                "steps": [
                    {
                        "step": s.step,
                        "allocated_gb": s.allocated_gb,
                        "reserved_gb": s.reserved_gb,
                        "fragmentation_gb": s.fragmentation_gb,
                        "step_time_s": s.step_time_s,
                    }
                    for s in r.steps
                ],
            }
            for r in all_results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_summary(all_results)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

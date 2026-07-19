# ADR-0028-advanced-pretraining-optimizations: Advanced System Optimizations (Muon, Liger Kernel, torch.compile, use_reentrant)

- **Status:** Accepted
- **Date:** 2026-07-16 (Updated: 2026-07-20)
- **Deciders:** Solo Developer

## Context

To further improve GPU computation throughput and VRAM efficiency in scratch pre-training without leaving the standard library paradigm:
1. **Compilation overhead**: PyTorch eager mode issues many small GPU kernel launches. We need PyTorch Inductor compiler to fuse operations.
2. **Logit Bottleneck**: Projection layers and CrossEntropy calculations consume immense VRAM. Fused Triton kernels are needed.
3. **Compiler Deadlocks**: Legacy reentrant gradient checkpointing hook structure conflicts with `torch.compile` and FSDP, leading to potential hangs or compiler breaks.
4. **Data loading bottlenecks**: Stalling GPU due to CPU-to-GPU memory transfer latency.
5. **Scaling Law Compliance**: Standard AdamW applies single LR to all parameters, violating Step Law scaling exponent (-0.713) which requires separate LR for 2D (weight matrices) vs 1D (embeddings, biases, LayerNorm).

## Decision

We will adopt the following advanced system optimizations natively through Hugging Face `Trainer` hooks:

1. **Muon Optimizer for 2D Parameters** (NEW, ADR-0043):
   Replace AdamW for 2D weight matrices with Muon (Momentum Orthogonalization by Newton-Schulz). 1D parameters (embeddings, biases, LayerNorm) remain on AdamW 8bit. This enables correct Step Law scaling: `η_2D ∝ N^(-0.713)`, `η_1D ≈ 0.3 * η_2D`.

2. **`torch.compile` Integration**:
   Configure `torch_compile=True`, using the `inductor` backend and `reduce-overhead` mode to fuse operations and compile model graphs.

3. **Liger Kernel Integration**:
   Configure `use_liger_kernel=True` to replace standard layers (RMSNorm, SwiGLU, CrossEntropy) with optimized Triton kernels. Since Triton does not natively support Windows, this dependency is restricted to non-Windows platforms using a marker in `pyproject.toml` (`"liger-kernel>=0.3.0; sys_platform != 'win32'"`), and defaults to `False` in the Windows training configuration.

4. **Non-Reentrant Gradient Checkpointing**:
   Explicitly configure `gradient_checkpointing_kwargs={"use_reentrant": False}` to prevent compiler deadlocks and future-proof the pipeline.

5. **DataLoader Direct Memory Access (DMA)**:
   Enable `dataloader_pin_memory=True` and configure configurable worker counts (`dataloader_num_workers`).

6. **Llama 3 Base Frequency**:
   Configure default `rope_theta: 500000.0` in the default configurations.

## Consequences

### Pros
- **Scaling Law Compliance**: Muon + AdamW 1D split enables correct `-0.713` exponent scaling from 150M → 7B.
- **Throughput**: up to 1.5x-2x speedup via Inductor graph compilation.
- **Memory Efficiency**: 40-60% VRAM saving from Liger Kernel on Linux deployments; AdamW 8bit on 1D params saves additional optimizer VRAM.
- **Platform Portability**: Marker-based dependency management permits local Windows development and debugging without installation failures.

### Cons
- Initial compile step introduces a compilation warm-up delay (typically 1-3 minutes) on the first step.
- Custom Trainer + SplitOptimizer adds ~200 lines maintenance burden.
- Existing AdamW-only checkpoints incompatible (require fresh training).

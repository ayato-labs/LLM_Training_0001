# ADR-0029-tf32-paged-optim-default-collator: Hardware & Dataloader Optimizations (TF32, Paged Optimizers, default_data_collator)

- **Status:** Accepted
- **Date:** 2026-07-16
- **Deciders:** Solo Developer

## Context

Following the system-level compile and kernel fusions in `ADR-0028`, further optimizations are required to address:
1. **NVIDIA GPU Matrix Core Underutilization**: Matrix multiplications default to slower FP32 on Ampere+ architectures unless TensorFloat-32 (TF32) is explicitly allowed.
2. **Out of Memory (OOM) Vulnerabilities**: Peak memory allocation spikes during gradient step updates can cause catastrophic training failures on lower VRAM GPUs.
3. **Dataloader CPU Overhead**: The standard `DataCollatorForLanguageModeling` dynamically copies batch lists to construct `labels` on every batch load, adding redundant CPU instructions.

## Decision

We will adopt three additional hardware and dataloader optimization techniques:

1. **TF32 Matrix Multiplication Enablement**:
   Set `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` if `allow_tf32: true` is configured. This accelerates execution on Tensor Cores (RTX 30/40 series, etc.) by 1.2x–1.5x.

2. **Paged Optimizers Support**:
   Enable configuring `optim` to `"paged_adamw_8bit"`. Paged optimizers page state variables to CPU system memory when VRAM limits are exceeded, preventing process termination.

3. **`default_data_collator` and Pre-cloned Labels**:
   Modify [model_utils.py](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/src/training/model_utils.py)'s `PackedDatasetWrapper` to clone `input_ids` directly into a `"labels"` key during sequence packing. This allows the trainer to use the lightweight `default_data_collator` instead of executing dynamic, step-level CPU collation cloning.

## Consequences

### Pros
- **Throughput**: Accelerates float32 matmul computations without precision loss.
- **CPU Offloading**: Streamlines the DataLoader phase, avoiding CPU execution bottlenecks.
- **Fail-safety**: Paged AdamW acts as a safety valve against VRAM spikes.

### Cons
- None.

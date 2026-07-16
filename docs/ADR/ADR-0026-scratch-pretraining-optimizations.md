# ADR-0026-scratch-pretraining-optimizations: Standard Library Optimizations for Scratch Pre-training Efficiency

- **Status:** Accepted
- **Date:** 2026-07-16
- **Deciders:** Solo Developer

## Context

Training a Large Language Model (LLM) from scratch (pre-training) is extremely compute-intensive. To make optimal use of a single GPU local environment:
1. **Compute Waste**: Padding tokens (`<pad>`) in standard training waste massive amounts of matrix multiplication time.
2. **Optimizer Memory**: Standard AdamW optimizer states take up $8$ bytes of FP32 memory per model parameter (e.g., $1.2$ GB just for optimizer states in a 150M parameter model).
3. **Attention Scaling**: Standard attention has $O(N^2)$ memory growth with context size.
4. **Context Capability**: Expanding context length after pre-training is harder than natively supporting a wider RoPE frequency from step 0.

To maintain our rule **"第1段階：既存ライブラリを組み合わせたプロトタイプ構築" (Phase 1: Build a prototype by combining existing libraries)**, we must achieve these optimizations using native configurations from `transformers` and `datasets` rather than writing custom CUDA kernels.

## Decision

We will adopt four core native optimization policies for all scratch pre-training:

1. **FlashAttention-2 Integration**:
   Configure models to run with `attn_implementation="flash_attention_2"`. This reduces attention memory usage to $O(N)$ and speeds up computation.

2. **Fused & 8-bit Optimizers**:
   Set `optim="adamw_torch_fused"` (combining GPU kernel launches for optimization steps) or `optim="adamw_bnb_8bit"` (quantizing optimizer states to save 75% of optimizer VRAM) in `TrainingArguments`.

3. **Sequence Packing (Zero Padding)**:
   Group training samples together (concatenated with `<eos>` tokens) and divide them into exact chunks of `seq_len`. This ensures that every token inside the context window represents active text data, completely eliminating padding computation.

4. **Optimized RoPE Base Theta**:
   Set `rope_theta=500000` (or matching Llama 3 specifications) directly in the base `LlamaConfig` during initialization. This allows the model to naturally handle longer contexts without requiring post-hoc RoPE scaling after scratch pre-training.

## Consequences

### Pros
- **Efficiency**: Sequence packing increases data throughput by up to 2-3x depending on average text lengths.
- **Lower VRAM**: 8-bit AdamW frees up VRAM to allow larger per-device batches, stabilizing pre-training gradients.
- **Speed**: FlashAttention-2 and fused optimizers maximize GPU tensor core utilization.

### Cons
- FlashAttention-2 requires compatible GPUs (Ampere/Ada Lovelace/Hopper architectures) and CUDA environments.
- Packing removes natural boundaries between files, which requires attention mask adjustments if strict file separation is needed (though not typically required for pre-training).

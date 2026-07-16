# ADR-0030-hpo-scaling-transfer-and-fixes: HPO Proxy Scaling Transfer and Configuration Fixes

- **Status:** Accepted
- **Date:** 2026-07-16
- **Deciders:** Solo Developer

## Context

Several architectural improvements and bug fixes were identified in the offline HPO system and standard configuration stack:
1. **Search Range Scaling Bug**: In `hpo_manager.py`, the search range for `max_lr_1d` was set to `lr_center * 5` to `lr_center * 20`. This resulted in extremely large learning rates (e.g. 0.05 to 0.20) for embedding, bias, and layernorm weights, leading to immediate NaN values and gradient explosions.
2. **Missing Proxy model-to-target model Scaling**: Offline HPO executed trials directly on the target model architecture (`args.model_size`). For larger targets like 3B or 7B, this causes local GPU Out-of-Memory (OOM) failures. We need a way to run HPO on a lightweight proxy size (e.g. 50M) and scale parameters up to the target using Scaling Laws.
3. **Configuration Drift (`rope_theta`)**: The HPO manager had `rope_theta: 10000.0` hardcoded, conflicting with `config.yaml` setting `500000.0`.
4. **Missing Standard Optimizations**: Standard settings such as `max_position_embeddings`, `initializer_range` (weights initialization standard deviation), `torch_empty_cache_steps` (preventing CUDA fragmentation), and `dataloader_prefetch_factor` (pre-loading batches) were missing.

## Decision

We will implement the following changes:

1. **Proxy Scaling Transfer in HPO Launcher**:
   Introduce `--proxy-model-size` and `--target-model-size` parameters in `scripts/find_hparams.py`. Run HPO trials on the proxy model size to avoid local OOMs. After the study completes, scale `max_lr_2d` and `max_lr_1d` to the target model parameters using the Step Law scaling relationship:
   $$\eta_{target} = \eta_{proxy\_best} \times \left(\frac{N_{target}}{N_{proxy}}\right)^{-0.713}$$

2. **Correct `max_lr_1d` Search Space**:
   Update `create_search_space` in `hpo_manager.py` to center `max_lr_1d` around `step_law_hpo["max_lr_1d"]` with a standard $\times [0.5, 2.0]$ factor range.

3. **Dynamic Configuration Load**:
   Update `get_base_config` to only load from `configs/config.yaml` if no model size argument is explicitly provided, preventing local parameter overrides during HPO scaling. Dynamically load `rope_theta` inside `objective()`.

4. **Add Remaining Optimizations**:
   - Set `max_position_embeddings` and `initializer_range` inside `LlamaConfig`.
   - Forward `torch_empty_cache_steps` and `dataloader_prefetch_factor` inside `TrainingArguments`.

## Consequences

### Pros
- **Resilience**: Enables running HPO for 3B/7B models on single local GPUs by offloading the workload to a 50M/150M proxy model.
- **Accuracy**: Prevents trial divergence and NaN failures by correcting the `max_lr_1d` search range scale.
- **Consistency**: Eliminates configuration drift between the main training configuration and HPO architectures.

### Cons
- None.

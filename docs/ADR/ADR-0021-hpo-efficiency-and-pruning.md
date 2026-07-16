# ADR-0021-hpo-efficiency-and-pruning: HPO Efficiency and Pruning Optimization

- **Status:** Accepted (Updated by ADR-0033, ADR-0034, ADR-0035)
- **Date:** 2026-07-13
- **Deciders:** Ayato-labs (ayato-labs)

## Context

The current offline Hyperparameter Optimization (HPO) setup is highly inefficient:
1. It takes about 6.8 minutes per trial on RTX 3050 because the training dataset is tokenized from scratch in every single trial.
2. The search space is too wide (7 dimensions), requiring at least 70-140 trials to yield reliable results, but the search terminates after 9 trials due to a 1-hour timeout constraint.
3. Lack of early pruning leads to wasting computation resources on poor hyperparameters (e.g., exploding loss).

## Decision

We will implement the following optimizations:

1. **Tokenize Dataset Once**: Perform dataset tokenization outside the Optuna `objective` function, and pass the pre-tokenized dataset to each trial.
2. **Reduce Search Space to 5D**: Fix low-sensitivity hyperparameters to reduce optimization dimensionality (updated by ADR-0033, ADR-0034):
   - **Dynamic range based on model size** (ADR-0034):
     - `max_lr_2d`: `center × [0.6-0.8, 1.3-1.7]` (log) - 小モデルほど広く
     - `max_lr_1d`: `center × [0.6-0.8, 1.3-1.7]` (log) - 小モデルほど広く
     - `batch_size_seqs`: [8, 16, 32]
     - `weight_decay`: [0.03-0.06, 0.18-0.25] (linear) - 小モデルほど広く
     - `warmup_ratio`: (0.01, 0.1, "") ← ADR-0033 で追加
   - Fixed: `beta2` = 0.95, `grad_clip` = 1.0
3. **Introduce MedianPruner**: Enable early pruning of trials using Optuna's `MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5)`.
4. **Increase max_steps to 50**: Allow sufficient training steps to establish realistic training dynamics and enable reliable evaluation.
5. **Set 24-Hour Timeout**: Set `timeout = 86400` (24 hours) instead of disabling it entirely to prevent infinite loops/hangs while ensuring enough time for 150 trials.

### HPO結果確実反映パイプライン (ADR-0035)
6. **warmup_ratio 尊重**: HPO探索結果（5次元目）を `hparams.yaml` に反映（固定値上書き廃止）
7. **ターゲットVRAM基準バッチ計算**: `--target-vram-gb` 指定で本番環境VRAM基準の `per_device_batch_size` / `grad_accum_steps` 算出
8. **config.yaml 自動同期**: `--sync-config` で `model.target_params`, `model.llama.*`, `defaults[hparams_XXX]` 一括更新
9. **起動時整合性検証**: `config.py:_validate_config_consistency()` で `config.yaml` と `hparams_XXX.yaml` のモデルサイズ一致を自動検証

## Consequences

### Pros
- Reduces trial time by ~50% by caching tokenization.
- TPE optimizer focuses on the most critical parameters.
- Pruning saves ~40% execution time by discarding hopeless trials.
- warmup_ratio is now searched (1%-10%) for optimal warmup strategy.
- Dynamic ranges reduce search volume by ~60% for large models (ADR-0034).
- **Full HPO→Training traceability**: All 5 dimensions propagate to full training (ADR-0035).
- **Zero-manual-sync**: `--sync-config` eliminates human error in scale changes (ADR-0035).
- **Runtime safety net**: Config consistency check catches mismatches early (ADR-0035).

### Cons
- Requires implementing a custom `TrainerCallback` to report intermediate loss values to Optuna.
- `beta2` and `grad_clip` are no longer searched (fixed to theoretical/empirical defaults).
- `find_hparams.py` argument count increases (backward compatible via defaults).

## Related ADRs
- ADR-0033: warmup_ratio → warmup_steps migration
- ADR-0034: Dynamic Search Space based on Model Size
- ADR-0035: HPO Result Propagation Pipeline Fixes

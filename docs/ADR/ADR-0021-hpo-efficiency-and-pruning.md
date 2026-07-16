# ADR-0021-hpo-efficiency-and-pruning: HPO Efficiency and Pruning Optimization

- **Status:** Accepted (Updated by ADR-0033, ADR-0034)
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
2. **Reduce Search Space to 5D**: Fix low-sensitivity hyperparameters to reduce optimization dimensionality (updated by ADR-0033):
   - **Dynamic range based on model size** (ADR-0034):
     - `max_lr_2d`: `center × [0.6-0.8, 1.3-1.7]` (log) - 小モデルほど広く
     - `max_lr_1d`: `center × [0.6-0.8, 1.3-1.7]` (log) - 小モデルほど広く
     - `batch_size_seqs`: [8, 16, 32]
     - `weight_decay`: [0.03-0.06, 0.18-0.25] (linear) - 小モデルほど広く
     - `warmup_ratio`: (0.01, 0.1, "") ← ADR-0033 で追加
   - Fixed: `beta2` = 0.95, `grad_clip` = 1.0
3. **Introduce MedianPruner**: Enable early pruning of trials using Optuna's `MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5)`.
4. **Increase max_steps to 50**: Allow sufficient training steps to establish realistic training dynamics and enable reliable evaluation.
5. **Set 8-Hour Timeout**: Set `timeout = 28800` (8 hours) instead of disabling it entirely to prevent infinite loops/hangs while ensuring enough time for 100 trials.

## Consequences

### Pros
- Reduces trial time by ~50% by caching tokenization.
- TPE optimizer focuses on the most critical parameters.
- Pruning saves ~40% execution time by discarding hopeless trials.
- warmup_ratio is now searched (1%-10%) for optimal warmup strategy.
- Dynamic ranges reduce search volume by ~60% for large models (ADR-0034).

### Cons
- Requires implementing a custom `TrainerCallback` to report intermediate loss values to Optuna.
- `beta2` and `grad_clip` are no longer searched (fixed to theoretical/empirical defaults).

## Related ADRs
- ADR-0033: warmup_ratio → warmup_steps migration
- ADR-0034: Dynamic Search Space based on Model Size

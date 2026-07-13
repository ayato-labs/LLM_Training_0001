# ADR-0021-hpo-efficiency-and-pruning: HPO Efficiency and Pruning Optimization

- **Status:** Proposed
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
2. **Reduce Search Space to 4D**: Fix low-sensitivity hyperparameters to reduce optimization dimensionality:
   - `warmup_ratio` = 0.03
   - `beta2` = 0.95
   - `grad_clip` = 1.0
3. **Introduce MedianPruner**: Enable early pruning of trials using Optuna's `MedianPruner(n_startup_trials=10, n_warmup_steps=15, interval_steps=5)`.
4. **Increase max_steps to 50**: Allow sufficient training steps to establish realistic training dynamics and enable reliable evaluation.
5. **Set 8-Hour Timeout**: Set `timeout = 28800` (8 hours) instead of disabling it entirely to prevent infinite loops/hangs while ensuring enough time for 100 trials.

## Consequences

### Pros
- Reduces trial time by ~50% by caching tokenization.
- TPE optimizer focuses on the most critical parameters (`max_lr_2d`, `max_lr_1d`, `batch_size_seqs`, `weight_decay`).
- Pruning saves ~40% execution time by discarding hopeless trials.

### Cons
- Requires implementing a custom `TrainerCallback` to report intermediate loss values to Optuna.
- `warmup_ratio`, `beta2`, and `grad_clip` are no longer searched (fixed to theoretical/empirical defaults).

# ADR-0024-training-logic-unification: Consolidating Duplicate Training Pipelines into a Unified Training Engine

- **Status:** Accepted
- **Date:** 2026-07-15
- **Deciders:** Ayato-labs (ayato-labs)

## Context

Prior to this decision, the training pipeline implementation was split between two primary scripts:
1. `src/training/main.py`: The entrypoint for standalone final/production training runs, integrated with Hydra (`DictConfig`) and utilizing the standard Hugging Face `Trainer`.
2. `src/training/train_model.py`: An alternative script used mainly by the HPO (Hyperparameter Optimization) search suite (`scripts/find_hparams.py` and `src/hpo/hpo_manager.py`). It defined raw dictionary configuration parsing, a custom `DetailedLoggingCallback`, and a custom `CustomTrainer` implementing the Muon/AdamW split optimizer.

This duplication introduced several issues:
- **Violation of DRY**: Utility code (such as `TokenizerWrapper`, `get_optimal_num_proc`, `compute_file_hash`, and `detect_vram`) was copy-pasted and declared identically in multiple files.
- **Inconsistent Optimization**: Final training runs in `main.py` lacked the `CustomTrainer` and Muon optimizer split used during HPO, making HPO-discovered parameters less relevant and causing training performance mismatch.
- **Maintenance Overhead**: Any changes to model loading, data tokenization, or callback triggers had to be manually ported to both files, creating a significant risk of bugs.

## Decision

We will merge the two pipelines into a single, unified training structure:

1. **Extract Shared Utilities**: Move `TokenizerWrapper`, `get_optimal_num_proc()`, `compute_file_hash()`, and `detect_vram()` to a shared utilities module `src/training/model_utils.py`.
2. **Implement Unified `train_engine.py`**: Design a core `src/training/train_engine.py` which contains:
   - All standard callbacks (`ProgressBarFormatCallback`, `HashSaveCallback`, `DetailedLoggingCallback`).
   - `CustomTrainer` with automatic split optimizer configuration supporting both Muon and AdamW.
   - A unified `train(config, tokenized_datasets=None, extra_callbacks=None)` function that accepts normalized configurations (supporting both Hydra-resolved and HPO raw dictionaries) and executes the training loop.
3. **Refactor Entrypoints**:
   - Simplify `src/training/main.py` to only handle Hydra config loading and immediately delegate execution to `train_engine.train()`.
   - Update `src/hpo/hpo_manager.py` and `scripts/find_hparams.py` to import utilities from `model_utils.py` and invoke `train_engine.train()`.
4. **Decommission `train_model.py`**: Delete `src/training/train_model.py` entirely from the codebase to avoid legacy confusion.

## Consequences

### Pros
- **Consistency**: Guarantees that HPO runs and final training runs share the exact same training loop, custom callbacks, optimization stack (Muon/AdamW), and execution behavior.
- **Maintainability**: Centralizes the core training pipeline in a single module, simplifying updates, testing, and debugging.
- **Code Cleanliness**: Eliminates duplicate code and redundant entrypoints.

### Cons
- None identified.

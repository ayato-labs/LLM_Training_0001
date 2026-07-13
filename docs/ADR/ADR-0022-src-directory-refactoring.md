# ADR-0022-src-directory-refactoring: Src Directory Refactoring

- **Status:** Proposed
- **Date:** 2026-07-14
- **Deciders:** Ayato-labs (ayato-labs)

## Context

The flat structure of the `src/` directory has led to a lack of logical separation between:
1. Generic utility scripts and loggers.
2. Hyperparameter Optimization (HPO) logic (Optuna, Step Law prior calculations).
3. The main scratch training pipeline (Trainer, dataset preparation, drive uploader, main entrypoint).

This makes the code harder to navigate, increases cognitive load for onboarding, and risks tight coupling between HPO-specific and本番 training-specific logic.

## Decision

We will refactor the directory structure of the `src/` directory into three logical subdirectories:
- `src/common/`: Domain-agnostic utilities (logging, environment snapshots, random seed seeding, GPU verification).
- `src/hpo/`: Hyperparameter Optimization scripts and libraries (`hpo_manager.py`, `step_law.py`).
- `src/training/`: Core LLM scratch training pipeline, data preparation, tokenizer training, models/optimizers, and the main CLI entry point.

We will enforce a strict coding rule: **Always use absolute paths (`from src.common.logger import logger` instead of relative imports like `from ..common.logger import logger`)** to keep the import statements highly readable and clean.

## Directory Structure

```text
src/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── check_gpu.py
│   ├── env_snapshot.py
│   ├── logger.py
│   └── set_seed.py
├── hpo/
│   ├── __init__.py
│   ├── hpo_manager.py
│   └── step_law.py
└── training/
    ├── __init__.py
    ├── config.py
    ├── drive_uploader.py
    ├── extract_data.py
    ├── main.py
    ├── model_utils.py
    ├── normuon.py
    ├── prepare_dataset.py
    ├── split_dataset.py
    ├── train.py
    ├── train_model.py
    ├── train_tokenizer.py
    └── wsd.py
```

## Consequences

### Pros
- Clean logical boundaries: separates HPO, common utils, and core training pipeline.
- Reduces cognitive load for developers navigating the repository.
- Avoids circular dependency issues and messy relative imports.

### Cons
- Requires renaming and modifying import paths across almost all files in `src/`, `tests/`, and helper scripts (e.g., `scripts/find_hparams.py`).

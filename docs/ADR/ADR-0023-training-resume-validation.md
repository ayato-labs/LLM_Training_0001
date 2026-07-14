# ADR-0023-training-resume-validation: Config and Dataset Verification on Training Resume

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Ayato-labs (ayato-labs)

## Context

When resuming training from a saved checkpoint (using the `--resume` option in `run_train.bat`), it is critical to ensure that the training configuration (`configs/config.yaml`) and the training dataset (e.g. `data/dataset.jsonl`) remain identical to the state when the checkpoint was saved. 

If training is resumed with a modified configuration or a different dataset:
1. Training dynamics could be severely disrupted (e.g., mismatched vocabulary, sequence lengths, learning rate schedule, etc.).
2. Data leakage or inconsistent validation states could occur.
3. The reproducibility of the training run is compromised.

Currently, there is no verification mechanism to validate that the configuration or the training data has not changed since the checkpoint was created.

## Decision

We will implement a verification mechanism based on cryptographic hashing (SHA-256) of the configuration and dataset files:

1. **Calculate Hashes on Startup**: At the beginning of the training run, compute the SHA-256 hashes of `configs/config.yaml` and the dataset file specified in `config.data_path`.
2. **Save Hashes with Checkpoint**: Implement a custom Hugging Face `TrainerCallback` (named `HashSaveCallback`) that runs `on_save` to write a `hashes.json` metadata file inside the new checkpoint directory. This file contains the configuration hash, dataset hash, and timestamp.
3. **Verify on Resume**: When `resume_from_checkpoint` is requested:
   - Resolve `checkpoint-latest` to the latest local checkpoint directory using directory structure analysis.
   - Read the corresponding `hashes.json` in the target checkpoint folder.
   - Compare the saved hashes against current calculated hashes of `config.yaml` and the dataset.
   - If they match, proceed with resuming training. If they differ, print a detailed error log and abort execution immediately.
   - If `hashes.json` does not exist in the checkpoint, log a warning and proceed (for backward compatibility).

## Consequences

### Pros
- **Consistency and Safety**: Guarantees that resumed training continues under identical parameters and dataset conditions, preventing silent training corruption.
- **Fail-Fast**: Instantly aborts execution if configuration or data is altered, giving developers clear debugging information (hashes comparison).
- **Auto-Resolution of latest checkpoint**: Standardizes the `--resume` option in `run_train.bat` to find and resume from the latest checkpoint directory automatically.

### Cons
- Slightly increases disk I/O at the start of training to hash the dataset file. However, since the dataset is typically a few gigabytes at most and hashed once per run (not per epoch), this performance overhead is negligible.
- Legacy checkpoints created before this ADR will not have `hashes.json` and will only emit a warning, which skips validation for those specific checkpoints.

# ADR-0025-remove-google-drive-integration: Removing Google Drive Integration from the Training Pipeline

- **Status:** Accepted (Implementation Complete)
- **Date:** 2026-07-16
- **Deciders:** Solo Developer

## Context

In `ADR-032` (documented as [018-drive-upload-trainer-callback.md](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/docs/ADR/018-drive-upload-trainer-callback.md)), we integrated Google Drive upload into the training process as a `TrainerCallback`. 

However, this introduced several drawbacks:
1. **Runtime Instability**: Transient network drops, API rate limits, or token expiration could cause errors or slow down the training loop.
2. **Heavy Dependencies**: Requires Google API client libraries (`google-api-python-client`, `google-auth-oauthlib`, etc.) loaded into the main training environment.
3. **Complexity**: Checking `.uploaded` files and manually deleting old local checkpoints added custom state management that could fail.

According to our core principle **"引き算のエンジニアリング" (Subtraction Engineering)** in [概念的要件定義書.md](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/docs/概念的要件定義書.md), we should avoid custom code when standard tools suffice.

## Decision

We will remove the Google Drive upload callback entirely from the training loop, delegating checkpoint pruning to Hugging Face standard logic:

1. **Delete Drive Callback**: Remove `DriveUploadCallback` from [callbacks.py](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/src/training/callbacks.py).
2. **Configure Native Pruning**: Set `save_total_limit` in [train_engine.py](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/src/training/train_engine.py) to automatically restrict local checkpoints.
3. **Relocate Checkpoint Helpers**: Move local checkpoint resolution helpers (`get_checkpoints`, `cleanup_old_checkpoints`) to [model_utils.py](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/src/training/model_utils.py).
4. **Remove config references**: Remove `drive_upload_interval` from [config.yaml](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/configs/config.yaml) and [config.py](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/src/training/config.py).
5. **Clean up documentation**: Update all documentation files to remove Google Drive references.

## Consequences

### Pros
- **Robustness**: Core training is entirely offline and decoupled from external API statuses.
- **Simplicity**: No custom Google client dependencies inside training.
- **Standardization**: Uses Hugging Face `Trainer` native `save_total_limit`.

### Cons
- None. The Google Drive daemon script (`scripts/drive_uploader.py`) remains available for manual, asynchronous backups if needed.

@echo off
set "VENV_PYTHON=.venv\Scripts\python.exe"
cd /d "%~dp0"

echo [1/3] Splitting dataset...
"%VENV_PYTHON%" src/split_dataset.py --input "../DataPreprocessing/data/dataset.jsonl" --train-output "data/train_dataset.jsonl" --val-output "data/val_dataset.jsonl" --val-ratio 0.01 --seed 42
if %errorlevel% neq 0 (
    echo Dataset split failed
    pause
    exit /b 1
)

echo [2/3] Starting execution...
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
"%VENV_PYTHON%" -m src.main %*

pause

@echo off
REM ============================================================
REM Novel LLM Scratch Training Runner
REM ============================================================
REM Usage: run_training.bat [config_file] [--resume]
REM Default config: configs\experiment_config.json
REM ============================================================

setlocal enabledelayedexpansion

REM ------------------------------------------------------------
REM Configuration
REM ------------------------------------------------------------
set PROJECT_ROOT=%~dp0
set CONFIG_FILE=%1
set RESUME_FLAG=

if "%CONFIG_FILE%"=="" (
    set CONFIG_FILE=configs\experiment_config.json
) else if "%CONFIG_FILE%"=="--resume" (
    set CONFIG_FILE=configs\experiment_config.json
    set RESUME_FLAG=--resume
)

if "%2"=="--resume" set RESUME_FLAG=--resume

cd /d %PROJECT_ROOT%

REM ------------------------------------------------------------
REM Environment Setup
REM ------------------------------------------------------------
echo ============================================================
echo Novel LLM Scratch Training
echo Project: %PROJECT_ROOT%
echo Config:  %CONFIG_FILE%
echo Resume:  %RESUME_FLAG%
echo ============================================================

REM ------------------------------------------------------------
REM Step 1: Dataset Split
REM ------------------------------------------------------------
if not exist "data\train_dataset.jsonl" (
    echo.
    echo [Step 1/3] Splitting dataset (novel-unit split)...
    python src\scripts\split_dataset.py ^
        --input ..\DataPreprocessing\data\dataset.jsonl ^
        --train-output data\train_dataset.jsonl ^
        --val-output data\val_dataset.jsonl ^
        --val-ratio 0.01 ^
        --seed 42
    if errorlevel 1 (
        echo ERROR: Dataset split failed
        pause
        exit /b 1
    )
) else (
    echo.
    echo [Step 1/3] Dataset already split (skipping)
)

REM ------------------------------------------------------------
REM Step 2: Run Training
REM ------------------------------------------------------------
echo.
echo [Step 2/3] Starting training...
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

if "%RESUME_FLAG%"=="--resume" (
    echo Resuming from latest checkpoint...
    python -m src.training.train_model %CONFIG_FILE% --resume
) else (
    python -m src.training.train_model %CONFIG_FILE%
)

if errorlevel 1 (
    echo.
    echo ERROR: Training failed (exit code %errorlevel%)
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo Training completed successfully!
echo ============================================================
pause
@echo off
REM ================================================================
REM Novel LLM Training System - Windows Setup Script
REM ================================================================
REM This script automates the initial setup process.
REM Run from the LLM_Training directory.
REM ================================================================

setlocal enabledelayedexpansion
echo ==========================================
echo  Novel LLM Training System - Setup
echo ==========================================
echo.

REM Step 1: Python version check
echo [1/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.12+
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Python %PYVER% found.

REM Step 2: Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo   Virtual environment created.
) else (
    echo   Virtual environment already exists. Skipping.
)

REM Step 3: Activate and install dependencies
echo.
echo [3/6] Installing dependencies...
call .venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -e . --quiet
echo   Dependencies installed.

REM Step 4: Install DVC
echo.
echo [4/6] Installing DVC...
pip install dvc --quiet
echo   DVC installed.

REM Step 5: Initialize DVC and pull data
echo.
echo [5/6] Initializing DVC...
if not exist ".dvc" (
    dvc init
    echo   DVC initialized.
) else (
    echo   DVC already initialized. Skipping.
)

echo.
echo [6/6] Pulling data from DVC remote (if configured)...
dvc pull 2>nul
if errorlevel 1 (
    echo   WARNING: DVC pull failed. Data may need to be configured manually.
    echo   Run: dvc remote add -d myremote ^<path-to-remote^>
) else (
    echo   Data pulled successfully.
)

echo.
echo ==========================================
echo  Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Activate the virtual environment:
echo      .venv\Scripts\activate
echo.
echo   2. Run training:
echo      python main.py --config-name=config
echo.
echo   3. Run evaluation:
echo      python src/eval_inference/evaluate_model.py
echo.
echo   4. Compare experiments:
echo      python -m src.evaluation.compare_runs --top 10
echo.

endlocal

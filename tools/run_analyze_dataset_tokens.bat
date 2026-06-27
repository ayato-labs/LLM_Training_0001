@echo off
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
    echo Error: .venv\Scripts\python.exe not found.
    pause
    exit /b 1
)
.venv\Scripts\python.exe tools\analyze_dataset_tokens.py
pause

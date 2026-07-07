@echo off
cd /d "%~dp0.."
echo Current directory: %CD%
echo Python path: %CD%\.venv\Scripts\python.exe
if not exist .venv\Scripts\python.exe (
    echo Error: .venv\Scripts\python.exe not found.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -c "import transformers; print('transformers OK')"
.venv\Scripts\python.exe tools\analyze_dataset_tokens.py %*
pause

@echo off
echo Starting LLM Training...
echo Using virtual environment at .venv

:: Ensure the environment is present
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Please run setup first.
    pause
    exit /b
)

:: Run the training script
.venv\Scripts\python.exe src\train_model.py

echo Training finished.
pause

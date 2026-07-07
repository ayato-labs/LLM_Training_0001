@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo Starting Novel LLM 50M Model Test (30 min)...
echo ==========================================

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Set PYTHONPATH to project root
set PYTHONPATH=%~dp0

:: Ensure the package is installed in editable mode
python -m pip install -e .

echo Running 50M Model Training (30-minute test)...
python main.py --config-name=scaling_50m

echo.
echo ==========================================
echo 50M Model Test Finished.
echo ==========================================
pause
endlocal

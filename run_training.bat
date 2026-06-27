@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo Setting up environment and training...
echo ==========================================

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Ensure the package is installed in editable mode to resolve module imports
python -m pip install -e .

echo Running Training Pipeline...
python tools\orchestrator.py

echo.
echo ==========================================
echo Training Pipeline Finished.
echo ==========================================
pause
endlocal

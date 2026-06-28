@echo off
setlocal
cd /d %~dp0

echo ==========================================
echo Resuming Novel LLM Training Pipeline...
echo ==========================================

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Set PYTHONPATH to project root so 'import project_config' works
set PYTHONPATH=%~dp0

:: Ensure the package is installed in editable mode
python -m pip install -e .

echo Running Training Pipeline (RESUME MODE)...
python main.py --resume

echo.
echo ==========================================
echo Training Pipeline Finished.
echo ==========================================
pause
endlocal

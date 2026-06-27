@echo off
setlocal

cd /d "%~dp0"

echo Starting Training Pipeline...

.\.venv\Scripts\python.exe tools\orchestrator.py

echo Finished.
pause
endlocal

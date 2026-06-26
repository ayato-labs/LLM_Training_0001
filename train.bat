@echo off
setlocal

:: Use argument if provided, otherwise default to config/config.yaml
set CONFIG_FILE=%1
if "%~1"=="" set CONFIG_FILE=config\config.yaml

echo Running training with: %CONFIG_FILE%

:: Run the script
.venv\Scripts\python.exe -u src\train_model.py %CONFIG_FILE%

pause
endlocal

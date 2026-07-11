@echo off
REM ============================================================
REM TensorBoard Launcher
REM ============================================================
REM Usage:
REM   launch_tensorboard.bat              # defaults: --logdir=models/output
REM   launch_tensorboard.bat <logdir>     # custom log directory
REM ============================================================

setlocal
cd /d "%~dp0"

set LOGDIR=%1
if "%LOGDIR%"=="" set LOGDIR=models\output

echo ============================================================
echo TensorBoard
echo Log dir: %LOGDIR%
echo URL:     http://localhost:6006
echo ============================================================

.venv\Scripts\tensorboard.exe --logdir=%LOGDIR% --port=6006

endlocal

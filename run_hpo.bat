@echo off
REM ============================================================
REM Offline HPO Launcher for LLM Training
REM Search phase only - outputs best config to configs/hparams_XXX.yaml
REM ============================================================

REM Change to script directory (critical for -m module resolution)
cd /d "%~dp0"

REM Enable UTF-8 code page
chcp 65001 >nul

set MODEL_SIZE=150M
set DATA_PATH=C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\DataPreprocessing\data\dataset.jsonl
set OUTPUT=configs/hparams_150M.yaml
set N_TRIALS=20
set VRAM_GB=
set SEQ_LEN=1024

REM Argument parsing (simple)
if not "%1"=="" set MODEL_SIZE=%1
if not "%2"=="" set DATA_PATH=%2
if not "%3"=="" set OUTPUT=%3
if not "%4"=="" set N_TRIALS=%4
if not "%5"=="" set VRAM_GB=%5
if not "%6"=="" set SEQ_LEN=%6

REM Build VRAM flag if provided
set VRAM_GB_FLAG=
if not "%VRAM_GB%"=="" set VRAM_GB_FLAG=--vram-gb %VRAM_GB%

echo.
echo ============================================================
echo  Offline HPO Search
echo ============================================================
echo  Model Size    : %MODEL_SIZE%
echo  Data Path     : %DATA_PATH%
echo  Output YAML   : %OUTPUT%
echo  Trials        : %N_TRIALS%
echo  VRAM (GB)     : %VRAM_GB%
echo  Seq Length    : %SEQ_LEN%
echo ============================================================
echo.

REM Check data file exists
if not exist "%DATA_PATH%" (
    echo [ERROR] Data file not found: %DATA_PATH%
    exit /b 1
)

REM Run HPO search
"%~dp0.venv\Scripts\python.exe" -m scripts.find_hparams ^
    --model-size %MODEL_SIZE% ^
    --data-path "%DATA_PATH%" ^
    --output "%OUTPUT%" ^
    --n-trials %N_TRIALS% ^
    %VRAM_GB_FLAG% ^
    --seq-len %SEQ_LEN%

if errorlevel 1 (
    echo.
    echo [FAILED] HPO search failed.
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] HPO completed. Artifact saved to: %OUTPUT%
echo ============================================================
echo.
echo Next step: Run training with the optimized hyperparameters:
echo   .\run_train.bat
echo.
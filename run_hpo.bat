@echo off
REM ============================================================
REM Offline HPO Launcher
REM ============================================================

REM Change to script directory
cd /d "%~dp0"

set MODEL_SIZE=150M
set TARGET_MODEL_SIZE=
set DATA_PATH=C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\DataPreprocessing\data\dataset.jsonl
set OUTPUT=configs/hparams_150M.yaml
set N_TRIALS=150
set VRAM_GB=
set TARGET_VRAM_GB=
set SEQ_LEN=1024
set SYNC_CONFIG=

REM Argument parsing (positional for backward compatibility)
if not "%1"=="" set MODEL_SIZE=%1
if not "%2"=="" set DATA_PATH=%2
if not "%3"=="" set OUTPUT=%3
if not "%4"=="" set N_TRIALS=%4
if not "%5"=="" set VRAM_GB=%5
if not "%6"=="" set SEQ_LEN=%6

REM Named arguments (new features)
for %%a in (%*) do (
    if "%%a"=="--target-size" (
        shift
        set TARGET_MODEL_SIZE=%1
        shift
    )
    if "%%a"=="--target-vram" (
        shift
        set TARGET_VRAM_GB=%1
        shift
    )
    if "%%a"=="--sync-config" (
        set SYNC_CONFIG=--sync-config
    )
)

REM Backward compatibility: if TARGET_MODEL_SIZE not set, use MODEL_SIZE
if "%TARGET_MODEL_SIZE%"=="" set TARGET_MODEL_SIZE=%MODEL_SIZE%

set VRAM_GB_FLAG=
if not "%VRAM_GB%"=="" set VRAM_GB_FLAG=--vram-gb %VRAM_GB%

set TARGET_VRAM_FLAG=
if not "%TARGET_VRAM_GB%"=="" set TARGET_VRAM_FLAG=--target-vram-gb %TARGET_VRAM_GB%

echo ============================================================
echo  Offline HPO Search
echo ============================================================
echo.

if not exist "%DATA_PATH%" (
    echo [ERROR] Data file not found: %DATA_PATH%
    pause
    exit /b 1
)

REM Run HPO search
"%~dp0.venv\Scripts\python.exe" -m scripts.find_hparams ^
  --proxy-model-size %MODEL_SIZE% ^
  --target-model-size %TARGET_MODEL_SIZE% ^
  --data-path "%DATA_PATH%" ^
  --output "%OUTPUT%" ^
  --n-trials %N_TRIALS% ^
  --seq-len %SEQ_LEN% ^
  %VRAM_GB_FLAG% ^
  %TARGET_VRAM_FLAG% ^
  %SYNC_CONFIG%

if errorlevel 1 (
    echo.
    echo [FAILED] HPO search failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] HPO completed.

echo.
echo Usage examples:
echo   run_hpo.bat 150M data/dataset.jsonl configs/hparams_150M.yaml 150 8 1024
echo   run_hpo.bat 150M data/dataset.jsonl configs/hparams_3B.yaml 150 8 1024 --target-size 3B --target-vram 24 --sync-config
echo   run_hpo.bat --target-size 7B --target-vram 48 --sync-config

@echo off
REM ============================================================
REM Training Launcher for LLM Training
REM Run only training phase (refer to HPO config hparams_150M.yaml)
REM ============================================================

REM Change to script directory
cd /d "%~dp0"

set MAX_STEPS=
set DATA_FRACTION=
set RESUME=

REM Parse arguments
:parse
if "%1"=="" goto :run
if "%1"=="--max-steps" (
    shift
    set MAX_STEPS=%1
    shift
    goto :parse
)
if "%1"=="--data-fraction" (
    shift
    set DATA_FRACTION=%1
    shift
    goto :parse
)
if "%1"=="--resume" (
    set RESUME=resume_from_checkpoint=models/output/checkpoint-latest
    shift
    goto :parse
)
if "%1"=="-h" goto :help
shift
goto :parse

:run
echo.
echo ============================================================
echo  LLM Training (Inference: HF Trainer + LlamaForCausalLM)
echo ============================================================
echo  Max Steps      : %MAX_STEPS%
echo  Data Fraction  : %DATA_FRACTION%
echo  Resume         : %RESUME%
echo  Config         : configs/config.yaml + configs/hparams_150M.yaml
echo ============================================================
echo.

REM Execute training
set OVERRIDES=
if not "%MAX_STEPS%"=="" set OVERRIDES=%OVERRIDES% max_steps=%MAX_STEPS%
if not "%DATA_FRACTION%"=="" set OVERRIDES=%OVERRIDES% data_fraction=%DATA_FRACTION%
if not "%RESUME%"=="" set OVERRIDES=%OVERRIDES% %RESUME%

uv run --active python -m src.training.main %OVERRIDES%

if errorlevel 1 (
    echo.
    echo [FAILED] Training failed.
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] Training completed.
echo Model saved to: models/output/
echo Checkpoints auto-uploaded to Google Drive (if configured)
echo ============================================================
goto :eof

:help
echo Usage:
echo   run_train.bat                    # Full training (config.yaml + hparams_150M.yaml)
echo   run_train.bat --max-steps 100    # Debug run (100 steps)
echo   run_train.bat --data-fraction 0.01 --max-steps 10  # Tiny debug
echo   run_train.bat --resume           # Resume from latest checkpoint
echo.
echo HPO (re-run search):
echo   run_hpo.bat [MODEL_SIZE] [DATA_PATH] [OUTPUT] [TRIALS] [VRAM_GB] [SEQ_LEN]
echo.
echo Examples:
echo   run_hpo.bat 150M data/dataset.jsonl configs/hparams_150M.yaml 20 4 1024
echo   run_hpo.bat 3B data/dataset.jsonl configs/hparams_3B.yaml 30 24 2048
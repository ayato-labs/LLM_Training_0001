@echo off
REM ============================================================
REM setup.bat - Virtual Environment Setup (GPU / CUDA Support)
REM
REM Changes:
REM   - python -m venv -> uv venv (Unified under uv management)
REM   - uv sync installs pyproject.toml dependencies
REM   - torch is retrieved from cu124 index via [tool.uv.sources]
REM   - Verifies CUDA availability automatically after install
REM ============================================================

cd /d "%~dp0"

REM ---- Check if uv is installed --------------------------------
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv not found. Please install uv using:
    echo   winget install astral-sh.uv
    echo   Or refer to: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo ============================================================
echo  Step 1: Remove existing virtual environment
echo ============================================================
if exist .venv (
    echo [INFO] Removing .venv...
    rmdir /s /q .venv
)

echo.
echo ============================================================
echo  Step 2: Create virtual environment (uv venv)
echo ============================================================
uv venv .venv --python 3.12
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Step 3: Install dependency packages (uv sync)
echo  torch will be installed with CUDA 12.4 support (cu124)
echo ============================================================
uv sync
if %errorlevel% neq 0 (
    echo [ERROR] uv sync failed. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Step 4: Verify CUDA / torch installation
echo ============================================================
.venv\Scripts\python.exe -c "import torch; cuda_ok = torch.cuda.is_available(); print(f'torch       : {torch.__version__}'); print(f'CUDA build  : {torch.version.cuda}'); print(f'CUDA avail  : {cuda_ok}'); print(f'GPU name    : {torch.cuda.get_device_name(0) if cuda_ok else \"N/A\"}')"
if %errorlevel% neq 0 (
    echo [WARN] Failed to verify torch. Please check the installation status.
)

echo.
echo ============================================================
echo  Setup Completed Successfully
echo ============================================================
pause

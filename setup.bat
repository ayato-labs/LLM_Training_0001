@echo off
REM ============================================================
REM setup.bat - 仮想環境セットアップ (GPU / CUDA 対応)
REM
REM 変更点:
REM   - python -m venv -> uv venv (uv 管理に統一)
REM   - uv sync で pyproject.toml の依存を一括インストール
REM   - torch は [tool.uv.sources] で cu124 インデックスから取得
REM   - インストール後に CUDA 利用可否を自動検証
REM ============================================================

cd /d "%~dp0"

REM ---- uv の存在確認 ----------------------------------------
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv が見つかりません。以下を実行してインストールしてください:
    echo   winget install astral-sh.uv
    echo   または: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo ============================================================
echo  Step 1: 既存の仮想環境を削除
echo ============================================================
if exist .venv (
    echo [INFO] .venv を削除中...
    rmdir /s /q .venv
)

echo.
echo ============================================================
echo  Step 2: 仮想環境を作成 (uv venv)
echo ============================================================
uv venv .venv --python 3.12
if %errorlevel% neq 0 (
    echo [ERROR] 仮想環境の作成に失敗しました。 Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Step 3: 依存パッケージをインストール (uv sync)
echo  torch は CUDA 12.4 対応版 (cu124) をインストールします
echo ============================================================
uv sync
if %errorlevel% neq 0 (
    echo [ERROR] uv sync に失敗しました。 Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Step 4: プロジェクト自体をエディタブルインストール
echo ============================================================
uv pip install -e .
if %errorlevel% neq 0 (
    echo [ERROR] editable install に失敗しました。 Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Step 5: CUDA / torch バージョン確認
echo ============================================================
.venv\Scripts\python.exe -c "import torch; cuda_ok = torch.cuda.is_available(); print(f'torch       : {torch.__version__}'); print(f'CUDA build  : {torch.version.cuda}'); print(f'CUDA avail  : {cuda_ok}'); print(f'GPU name    : {torch.cuda.get_device_name(0) if cuda_ok else \"N/A\"}')"
if %errorlevel% neq 0 (
    echo [WARN] torch の検証中にエラーが発生しました。インストール結果を確認してください。
)

echo.
echo ============================================================
echo  Setup 完了
echo ============================================================
pause

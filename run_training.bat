@echo off
setlocal

:: 現在のバッチファイルの場所をベースにパスを構築
cd /d "%~dp0"

echo ==========================================
echo Starting Novel LLM Training Pipeline...
echo ==========================================

:: 仮想環境の Python を絶対パスで確実に指定
:: 現在のディレクトリの下に .venv がある前提
call .venv\Scripts\activate.bat
python tools\orchestrator.py

echo.
echo ==========================================
echo Training Pipeline Finished.
echo ==========================================
pause
endlocal

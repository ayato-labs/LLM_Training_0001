@echo off
setlocal

echo ==========================================
echo Starting Novel LLM Training Pipeline...
echo ==========================================

:: Step Lawに基づくパラメータ最適化と学習の実行
.venv\Scripts\python.exe tools\orchestrator.py

echo.
echo ==========================================
echo Training Pipeline Finished.
echo ==========================================
pause
endlocal

@echo off
set /p PROMPT="Enter prompt (e.g. input: text or [src] text): "
.venv\Scripts\python.exe src\eval_inference\generate.py --prompt "%PROMPT%"
pause

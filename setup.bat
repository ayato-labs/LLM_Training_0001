@echo off
echo Removing existing virtual environment if it exists...
if exist .venv rmdir /s /q .venv

echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. Error code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo Activating virtual environment and installing packages...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. Error code: %errorlevel%
    pause
    exit /b %errorlevel%
)

python -m pip install --upgrade pip
pip install .
if %errorlevel% neq 0 (
    echo Failed to install packages. Error code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo Setup completed successfully.
pause

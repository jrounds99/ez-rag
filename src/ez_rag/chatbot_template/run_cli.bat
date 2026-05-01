@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    python -m venv .venv
)
set PY=.venv\Scripts\python.exe
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r requirements.txt
"%PY%" chatbot_cli.py
endlocal

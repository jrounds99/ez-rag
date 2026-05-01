@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python is not installed or not on PATH.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and check the box that says "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    echo creating virtualenv .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtualenv.
        pause
        exit /b 1
    )
)

set PY=.venv\Scripts\python.exe
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

echo.
echo --- starting ez-rag chatbot ---
echo.
"%PY%" server.py
endlocal

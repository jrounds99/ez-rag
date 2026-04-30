@echo off
setlocal enableextensions

REM === ez-rag GUI launcher ==================================================
title ez-rag

call :find_gui

if defined EZRAG_GUI goto :have_gui
if not exist "%~dp0pyproject.toml" goto :no_install_path

REM ---- First-run install --------------------------------------------------
where python.exe >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo ez-rag isn't installed yet. Installing now (one-time, ~1 min)...
echo.
pushd "%~dp0"
python -m pip install --user -e ".[ocr,gui]"
set "INSTALL_RC=%ERRORLEVEL%"
popd
if not "%INSTALL_RC%"=="0" (
    echo.
    echo Install failed with exit code %INSTALL_RC%. See messages above.
    pause
    exit /b 1
)

echo.
echo Install complete.
echo.
call :find_gui

:have_gui
if not defined EZRAG_GUI goto :still_missing

echo Using: %EZRAG_GUI%

REM ---- Auto-start Ollama (best-effort) ------------------------------------
where ollama.exe >nul 2>&1
if errorlevel 1 goto :skip_ollama
tasklist /FI "IMAGENAME eq ollama.exe" /NH 2>nul | findstr /I "ollama.exe" >nul
if not errorlevel 1 goto :skip_ollama
echo Starting Ollama in background...
start "ollama" /B ollama serve >nul 2>&1
powershell -nop -c "Start-Sleep -Seconds 2" >nul 2>&1
:skip_ollama

REM ---- If a folder was dragged onto this script, open it as workspace -----
if not "%~1"=="" (
    if exist "%~1\" cd /d "%~1"
)

echo Launching GUI from: %CD%
echo.

%EZRAG_GUI%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo ez-rag-gui exited with code %RC%
    pause
)

endlocal
exit /b 0


:no_install_path
echo.
echo Could not find ez-rag-gui.exe and there's no pyproject.toml here to install from.
echo Install ez-rag manually:    pip install --user ez-rag[ocr,gui]
pause
exit /b 1


:still_missing
echo.
echo Install completed but ez-rag-gui.exe still wasn't found.
echo Looked in:
echo     %APPDATA%\Python\Python314\Scripts\
echo     %APPDATA%\Python\Python3*\Scripts\
echo     %USERPROFILE%\.local\bin\
echo     ^<PATH^>
pause
exit /b 1


REM ===== :find_gui ==========================================================
:find_gui
set "EZRAG_GUI="
if exist "%APPDATA%\Python\Python314\Scripts\ez-rag-gui.exe" (
    set "EZRAG_GUI=%APPDATA%\Python\Python314\Scripts\ez-rag-gui.exe"
    goto :eof
)
for /d %%D in ("%APPDATA%\Python\Python3*") do (
    if exist "%%~D\Scripts\ez-rag-gui.exe" set "EZRAG_GUI=%%~D\Scripts\ez-rag-gui.exe"
)
if defined EZRAG_GUI goto :eof
for /f "delims=" %%I in ('where ez-rag-gui.exe 2^>nul') do (
    if not defined EZRAG_GUI set "EZRAG_GUI=%%I"
)
if defined EZRAG_GUI goto :eof
if exist "%USERPROFILE%\.local\bin\ez-rag-gui.exe" set "EZRAG_GUI=%USERPROFILE%\.local\bin\ez-rag-gui.exe"
goto :eof

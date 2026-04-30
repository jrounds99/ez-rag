@echo off
setlocal enableextensions

REM === ez-rag CLI shortcut ==================================================
title ez-rag CLI

call :find_cli

if defined EZRAG goto :have_cli
if not exist "%~dp0pyproject.toml" goto :no_install_path

where python.exe >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from python.org.
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
    echo Install failed with exit code %INSTALL_RC%.
    pause
    exit /b 1
)

call :find_cli

:have_cli
if not defined EZRAG goto :still_missing

REM Add the discovered location to PATH for any sub-shell we open
for %%F in ("%EZRAG%") do set "EZRAG_DIR=%%~dpF"
set "PATH=%EZRAG_DIR%;%PATH%"

REM ---- Auto-start Ollama --------------------------------------------------
where ollama.exe >nul 2>&1
if errorlevel 1 goto :skip_ollama
tasklist /FI "IMAGENAME eq ollama.exe" /NH 2>nul | findstr /I "ollama.exe" >nul
if not errorlevel 1 goto :skip_ollama
start "ollama" /B ollama serve >nul 2>&1
:skip_ollama

REM ---- Run ----------------------------------------------------------------
if "%~1"=="" (
    echo ez-rag is on PATH in this shell. Try:
    echo     ez-rag init .
    echo     ez-rag ingest
    echo     ez-rag ask "your question"
    echo     ez-rag chat
    echo     ez-rag-gui
    echo.
    cmd /K
) else (
    %EZRAG% %*
    if errorlevel 1 pause
)

endlocal
exit /b 0


:no_install_path
echo Could not find ez-rag.exe and no pyproject.toml here to install from.
echo Install ez-rag manually:    pip install --user ez-rag[ocr,gui]
pause
exit /b 1

:still_missing
echo Install completed but ez-rag.exe still wasn't found.
pause
exit /b 1


REM ===== :find_cli ==========================================================
:find_cli
set "EZRAG="
if exist "%APPDATA%\Python\Python314\Scripts\ez-rag.exe" (
    set "EZRAG=%APPDATA%\Python\Python314\Scripts\ez-rag.exe"
    goto :eof
)
for /d %%D in ("%APPDATA%\Python\Python3*") do (
    if exist "%%~D\Scripts\ez-rag.exe" set "EZRAG=%%~D\Scripts\ez-rag.exe"
)
if defined EZRAG goto :eof
for /f "delims=" %%I in ('where ez-rag.exe 2^>nul') do (
    if not defined EZRAG set "EZRAG=%%I"
)
if defined EZRAG goto :eof
if exist "%USERPROFILE%\.local\bin\ez-rag.exe" set "EZRAG=%USERPROFILE%\.local\bin\ez-rag.exe"
goto :eof

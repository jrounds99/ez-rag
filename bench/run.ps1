# ez-rag-bench bootstrap (Windows PowerShell).
#
# Creates a self-contained venv under bench\.venv, installs the
# bench's tiny dep set, verifies Ollama is reachable, runs the bench
# with whatever args you pass.
#
# Idempotent — re-running reuses the venv. Safe to interrupt.
#
# Usage:
#     .\bench\run.ps1                          # `full` mode with defaults
#     .\bench\run.ps1 probe                    # just print system probe
#     .\bench\run.ps1 quick                    # 5-min sanity run
#     .\bench\run.ps1 search --workspace C:\dnd --questions 10
#     .\bench\run.ps1 report path\to\judged.json

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSCommandPath)
Set-Location -LiteralPath ".."

$REPO_ROOT = (Get-Location).Path
$VENV = "bench\.venv"
$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# --- Verify Python ---
$pyExe = (Get-Command $PY -ErrorAction SilentlyContinue)
if (-not $pyExe) {
    Write-Host "[!] '$PY' not found on PATH." -ForegroundColor Red
    Write-Host "    Install Python 3.11+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

$pyVer = & $PY -c "import sys; print('{0}.{1}'.format(*sys.version_info))"
$verParts = $pyVer.Split(".")
$major = [int]$verParts[0]
$minor = [int]$verParts[1]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
    Write-Host "[!] Python 3.11+ required (you have $pyVer)." -ForegroundColor Red
    exit 1
}

# --- Venv ---
if (-not (Test-Path $VENV)) {
    Write-Host ">>> creating bench venv ($VENV) ..." -ForegroundColor Cyan
    & $PY -m venv $VENV
    & "$VENV\Scripts\pip.exe" install --upgrade --quiet pip
    & "$VENV\Scripts\pip.exe" install --quiet -r bench\requirements.txt
}

# --- Run ---
$benchArgs = if ($args.Count -eq 0) { @("full") } else { $args }

Write-Host ">>> running ez-rag-bench $($benchArgs -join ' ')" -ForegroundColor Cyan
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
& "$VENV\Scripts\python.exe" -X utf8 -m bench.cli @benchArgs
exit $LASTEXITCODE

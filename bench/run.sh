#!/usr/bin/env bash
# ez-rag-bench bootstrap (Linux / macOS).
#
# Creates a self-contained venv under bench/.venv, installs the
# bench's tiny dep set, verifies Ollama is reachable, runs the bench
# with whatever args you pass.
#
# Idempotent — re-running reuses the venv. Safe to interrupt.
#
# Usage:
#     ./bench/run.sh                       # `full` mode with defaults
#     ./bench/run.sh probe                 # just print system probe
#     ./bench/run.sh quick                 # 5-min sanity run
#     ./bench/run.sh search --workspace ~/dnd --questions 10
#     ./bench/run.sh report path/to/judged.json
#
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
VENV="bench/.venv"
PY="${PYTHON:-python3}"

# --- Verify Python ---
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[!] '$PY' not found on PATH."
    echo "    Install Python 3.11+ from https://python.org"
    exit 1
fi

PY_VER="$($PY -c 'import sys; print("{0}.{1}".format(*sys.version_info))')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER##*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    echo "[!] Python 3.11+ required (you have $PY_VER)."
    exit 1
fi

# --- Venv ---
if [ ! -d "$VENV" ]; then
    echo ">>> creating bench venv ($VENV) ..."
    $PY -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade --quiet pip
    "$VENV/bin/pip" install --quiet -r bench/requirements.txt
fi

# --- Run ---
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=(full)
fi

echo ">>> running ez-rag-bench ${ARGS[*]}"
exec "$VENV/bin/python" -X utf8 -m bench.cli "${ARGS[@]}"

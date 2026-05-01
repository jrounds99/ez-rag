#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo
    echo "Python 3 is not installed."
    echo "On macOS:  brew install python"
    echo "On Linux:  sudo apt install python3 python3-venv  (or your distro's equivalent)"
    echo
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "creating virtualenv .venv ..."
    python3 -m venv .venv
fi

PY=".venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

echo
echo "--- starting ez-rag chatbot ---"
echo
exec "$PY" server.py

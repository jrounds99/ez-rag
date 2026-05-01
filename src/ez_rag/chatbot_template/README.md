# ez-rag chatbot

A self-contained chatbot bundled from a particular ez-rag workspace.
The retrieval and LLM settings are baked in — change them by re-exporting
from ez-rag (or by editing `data/config.toml` directly).

## What's in the box

```
chat.html, chat.css, chat.js   editable web UI
server.py                      tiny stdlib HTTP server, /api/ask streams replies
chatbot_cli.py                 terminal-only alternative
run.bat / run.sh               GUI launcher (Windows / Mac+Linux)
run_cli.bat / run_cli.sh       CLI launcher
requirements.txt               Python deps (numpy, httpx, fastembed)
data/meta.sqlite               the RAG index (already embedded chunks)
data/config.toml               frozen ez-rag settings
ezrag_lib/                     vendored retrieval/generation modules
```

## Prerequisites

1. **Python 3.10+** — https://www.python.org/downloads/ (Windows: tick
   "Add Python to PATH" during install).
2. **Ollama** — https://ollama.com/download
3. The model this RAG was built with. The launcher prints the required tag
   on startup; pull it with:
       ollama pull <tag>

If your config used `embedder_provider = "ollama"`, also pull the embedder
model (printed at startup).

## Quick start

### GUI

- Windows: double-click `run.bat`
- macOS / Linux: `chmod +x run.sh` then `./run.sh`

The launcher creates a `.venv/`, installs deps, starts the server, and
opens http://127.0.0.1:8765 in your browser.

### CLI

Same as above but `run_cli.bat` / `run_cli.sh`.

## Customizing the page

`chat.html`, `chat.css`, `chat.js` are plain static files — edit, save,
reload the browser. Theme colors are CSS variables at the top of `chat.css`.

## Updating the index

Re-export from ez-rag and replace `data/meta.sqlite` and `data/config.toml`.
Everything else stays the same.

## Troubleshooting

- "server unreachable" badge — Ollama isn't running, or the model named in
  `data/config.toml` isn't pulled.
- "0 files / 0 chunks" — `data/meta.sqlite` is missing or corrupt.
- Permission denied on `run.sh` — `chmod +x run.sh`.
- Antivirus blocks the `.bat` — right-click → Properties → Unblock.

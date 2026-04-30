# GUI

```bash
ez-rag-gui                       # cross-platform desktop app
```

Or, on Windows, double-click `ez-rag-gui.bat` from the repo. First run installs missing deps; later runs go straight to the window.

## Layout

| Tab | Purpose |
|---|---|
| **Chat** | Conversational Q&A grounded in your documents. Streaming + citations. |
| **Files** | Add/import documents, run ingest, see per-file chunk counts. |
| **Settings** | Chunk size, top-K, hybrid toggle, OCR toggle, models, Ollama URL. |
| **Doctor** | Environment diagnostics — what's installed, what's reachable. |

The header bar always shows your workspace name, file/chunk counts, and the active backend (`Ollama · qwen2.5:3b` etc.).

## Chat

- **Multi-turn memory.** Each new message is sent with the full prior conversation. Follow-ups like *"explain that more simply"* or *"what about the second one?"* work because the model sees all earlier turns.
- **Streaming answers.** Tokens appear as the model generates them. The send button turns into a red ⊘ Stop button while streaming.
- **Citations.** Every grounded answer ends with numbered chips showing the source files and pages. Click a chip to open a dialog with the exact passage that was retrieved.
- **General chat works too.** When retrieval finds nothing relevant ("hi", "what model are you?", small talk), ez-rag still calls the LLM and returns a normal conversational reply — just without citations.
- **Clear** in the chat header starts a fresh conversation.

### Keyboard shortcuts

| Key | Action |
|---|---|
| **Enter** | Send the message |
| **Shift + Enter** | Newline inside the message |
| **Ctrl + N** | Switch to Chat tab |
| **Ctrl + I** | Switch to Files tab |
| **Ctrl + ,** | Switch to Settings tab |

## Files

- **Add files…** copies one or more files into `<workspace>/docs/`.
- **Add folder…** recursively imports every supported file from a folder.
- **Open in Explorer** opens `docs/` in the OS file manager — drop files there directly if you prefer.
- **Ingest** runs the parse → chunk → embed pipeline. **Re-ingest (force)** rebuilds even unchanged files.
- Each file row shows extension icon, size, and how many chunks it produced (or "not indexed" if it hasn't been ingested yet).

The ingest log streams live: each file flips through `parsing` → `embedding N chunks` → `ok (N chunks)` so you can see what's happening.

## Settings

Four cards:

1. **Ingest** — chunk size (tokens), chunk overlap, OCR toggle, Contextual Retrieval toggle (slower ingest, better recall when enabled).
2. **Retrieval** — top-K, hybrid (BM25 + dense), rerank.
3. **LLM** — model dropdown, refresh ⟳, **Browse Ollama library**, **Use local GGUF…**, Ollama URL, temperature, max tokens.
4. **Embedder** — provider dropdown (auto / ollama / fastembed), Ollama embed model dropdown + Browse, fastembed model dropdown.

See [models](models) for everything model-related (browse dialog, VRAM estimates, GGUF support).

Hit **Save settings** to persist to `.ezrag/config.toml`. **Reset** reloads the on-disk values into the form.

## Doctor

Live status of the runtime environment:

- **LLM backend** — `ollama` / `llama-cpp` / `none`
- **fastembed** — installed?
- **llama-cpp-python** — installed? (optional)
- **RapidOCR** — installed? (optional, primary OCR engine)
- **Tesseract** — found on PATH? (optional, OCR fallback)
- **Flet (GUI)** — version
- **Python** — version + platform

The refresh icon re-runs every check.

## Empty state / first run

If no workspace is open the GUI shows a welcome screen with **Open workspace** and a list of recent workspaces (saved in `~/.ezrag_recents.txt`).

If a workspace is open but `docs/` is empty, the chat tab tells you to add files. The chat input is disabled until you've ingested something.

## Where things are saved

```
<workspace>/
├── docs/                 # your files
├── conversations/        # (reserved; future feature)
└── .ezrag/
    ├── config.toml       # what Settings writes
    ├── meta.sqlite       # files, chunks, FTS index, embeddings
    └── ingest.log
```

# Install

ez-rag is a Python package. Pick whichever path is easiest for you.

## Recommended (Windows / macOS / Linux)

```bash
# 1) ez-rag itself, with OCR + GUI extras
pipx install "ez-rag[ocr,gui]"

# 2) Ollama for the LLM and (optional) embeddings — https://ollama.com/download
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

## Without pipx

```bash
python -m pip install --user "ez-rag[ocr,gui]"
```

## Bare-minimum install (retrieval-only)

```bash
pipx install ez-rag
# no Ollama, no OCR — you get full-text retrieval and ranked passages
# instead of LLM-written answers.
```

## All optional extras

| Extra | Adds | Why |
|---|---|---|
| `ocr` | RapidOCR + Pillow + pytesseract | OCR for screenshots and scanned PDFs |
| `gui` | Flet | the desktop GUI (`ez-rag-gui`) |
| `llm` | llama-cpp-python | run a local GGUF without Ollama |

You can combine them: `pipx install "ez-rag[ocr,gui,llm]"`.

## Unsigned binaries

When we publish prebuilt installers (post-v1), they will not be code-signed
(see project goals — $0). On first run:

* **Windows SmartScreen**: click "More info" → "Run anyway".
* **macOS Gatekeeper**: right-click the app, choose Open, then Open in the
  dialog. Or run `xattr -d com.apple.quarantine /Applications/ez-rag.app`.
* **Linux**: `chmod +x ez-rag.AppImage` and run.

The pipx install path avoids this entirely.

## Verifying

```bash
ez-rag doctor
```

Should show:

* Ollama reachable: yes (or "no" — install it for LLM answers)
* fastembed: installed
* RapidOCR: ok (if you installed `[ocr]`)
* Tesseract: ok (if you have it on PATH)
* flet (GUI): installed (if you installed `[gui]`)

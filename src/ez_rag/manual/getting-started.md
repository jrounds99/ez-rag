# Getting started

```bash
# 1) make a workspace
mkdir my-rag && cd my-rag
ez-rag init .

# 2) drop files into docs/
cp ~/Downloads/*.pdf docs/
cp ~/Pictures/screenshot.png docs/

# 3) build the index
ez-rag ingest

# 4) ask
ez-rag ask "summarize the key findings"
ez-rag chat
```

## What it does

* walks `docs/` recursively
* parses PDFs, DOCX, XLSX, CSV, HTML, MD, TXT, EPUB, EML and screenshots
* chunks and embeds the text into a local SQLite index
* searches with hybrid BM25 + dense vectors
* asks the LLM (Ollama if installed, llama-cpp if a GGUF is provided, otherwise prints the matching passages)

## Where things live

```
my-rag/
  docs/                   # your files
  .ezrag/
    config.toml           # tweak chunk size, top-k, model names, …
    meta.sqlite           # the index
  conversations/          # saved chats
```

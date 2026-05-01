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

## Terminology

These three words are *not* synonyms — they're layers of the same system.

| Term | Refers to |
|---|---|
| **Corpus** | Your collection of source documents. The `docs/` folder is the corpus. |
| **Index** | The searchable structure ez-rag builds from the corpus: chunks, embeddings, BM25. Lives in `.ezrag/meta.sqlite`. |
| **RAG** | Retrieval-Augmented Generation — the whole technique. Corpus + index + retrieval + LLM, glued together. |

Quick litmus test: if you can re-do the operation by re-running `ez-rag ingest`, you only changed the **index**. If your original files survive, the **corpus** is intact. If you only changed Settings or the model, you only changed the **RAG**'s configuration. See `ez-rag help retrieval` for the embedder / reranker / chunk vocabulary.

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

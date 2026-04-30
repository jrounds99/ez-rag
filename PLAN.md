# Architecture

```
                  ┌──────────────────────────────────┐
                  │            ez-rag CLI            │
                  │  (Typer + Rich → src/ez_rag/cli) │
                  └────────┬───────────────┬─────────┘
                           │               │
          ┌────────────────┘               └────────────────┐
          ▼                                                 ▼
   ┌──────────────┐                                ┌────────────────┐
   │ Ingest       │                                │ Inference      │
   │  parsers.py  │                                │  generate.py   │
   │  ocr.py      │ ── embeds (embed.py) ────────▶ │   Ollama (auto)│
   │  chunker.py  │                                │   llama-cpp    │
   │  ingest.py   │                                │   none → top-k │
   └──────┬───────┘                                └────────▲───────┘
          ▼                                                 │
   ┌──────────────┐    hybrid (BM25 + cosine)               │
   │   index.py   │ ◀───────────────────────────────────────┤
   │  SQLite +    │                                         │
   │  FTS5 + BLOB │                                         │
   └──────────────┘                          ┌──────────────┴────────┐
                                             │ ez-rag-gui (Flet)     │
                                             │ gui/ez_rag_gui/main.py│
                                             └───────────────────────┘
```

## Workspace layout

```
my-rag/
├── docs/                         # user files
└── .ezrag/
    ├── config.toml               # chunk size, top-k, model, …
    ├── meta.sqlite               # files, chunks, FTS index, embeddings
    └── ingest.log
```

## Module map

| File | Responsibility |
|---|---|
| `cli.py` | Typer app: `init`, `ingest`, `ask`, `chat`, `status`, `models`, `serve`, `doctor`, `reindex`, `help` |
| `config.py` | TOML config dataclass |
| `workspace.py` | Find / create the `.ezrag/` workspace |
| `parsers.py` | PDF · DOCX · XLSX/CSV · HTML · MD/TXT · EPUB · EML · images |
| `ocr.py` | RapidOCR primary, Tesseract fallback |
| `chunker.py` | Recursive split with token-target + overlap |
| `embed.py` | Ollama embed if reachable, else fastembed |
| `index.py` | SQLite + FTS5; embeddings as BLOBs of float32 |
| `retrieve.py` | Hybrid BM25 + cosine fused with RRF |
| `generate.py` | Ollama → llama-cpp → retrieval-only fallback |
| `ingest.py` | Orchestrates the whole ingest pipeline |
| `server.py` | OpenAI-compatible `/v1/chat/completions` (stdlib HTTP) |
| `manual/*.md` | In-tool offline manual pages |
| `gui/ez_rag_gui/main.py` | Flet GUI (Workspace · Ingest · Chat · Settings · Models tabs) |

## Choices

- Ollama is the default LLM backend because it's free, cross-platform, and hands-off. llama-cpp-python is optional. If neither is present, `ez-rag ask` returns ranked passages instead of an LLM-written answer (still useful).
- fastembed (ONNX-based) is the default embedder so the package works offline immediately without a separate Ollama install. If Ollama is available, its `nomic-embed-text` model is used instead.
- SQLite + FTS5 + numpy is the index backend — single file, no server, fine to ~100k chunks. LanceDB is the obvious upgrade path when corpora outgrow this.
- Flet for the GUI: one Python codebase ships to Windows / macOS / Linux. The GUI shells through to the same library, so it can't drift from the CLI.

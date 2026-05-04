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
│   ├── handbook.pdf
│   └── handbook.pdf.ezrag-meta.toml   # optional per-file metadata
└── .ezrag/
    ├── config.toml               # chunk size, top-k, model, …
    ├── routing.toml              # multi-GPU daemon assignments (optional)
    ├── meta.sqlite               # files, chunks, FTS index, embeddings
    ├── preview_cache/            # PDF page-image cache (3-day rolling)
    └── ingest.log
```

## Module map

| File | Responsibility |
|---|---|
| `cli.py` | Typer app: `init`, `ingest`, `ask`, `chat`, `status`, `models`, `serve`, `doctor`, `reindex`, `export`, `import`, `scan`, `help` |
| `config.py` | TOML config dataclass |
| `workspace.py` | Find / create the `.ezrag/` workspace |
| `parsers.py` | PDF · DOCX · XLSX/CSV · HTML · MD/TXT · EPUB · EML · images |
| `ocr.py` | RapidOCR primary, Tesseract fallback |
| `embed.py` | Ollama embed if reachable, else fastembed |
| `index.py` | SQLite + FTS5; embeddings as BLOBs of float32 |
| `retrieve.py` | Hybrid BM25 + cosine fused with RRF, optional rerank/MMR/HyDE/multi-query/neighbor expansion |
| `generate.py` | Ollama → llama-cpp → retrieval-only fallback; reasoning-model `thinking` field surfaced separately |
| `ingest.py` | Orchestrates the whole ingest pipeline |
| `ingest_meta.py` | Per-file `.ezrag-meta.toml` sidecars: prefix / suffix / negatives + scope (global/topic/file-only) |
| `ingest_scan.py` | LLM discovery scan that auto-populates sidecars from corpus content |
| `models.py` | Ollama listing/pull, library scraper, VRAM estimator, running-model unload |
| `multi_gpu.py` | Routing table (TOML), per-model GPU pinning, free-VRAM auto-picker |
| `daemon_supervisor.py` | Spawn / detect / adopt / shutdown per-GPU Ollama daemons |
| `gpu_detect.py` | nvidia-smi · rocm-smi · xpu-smi · WMI fallback |
| `gpu_catalog.py` | Static GPU spec catalog (VRAM, bandwidth, FP16 TFLOPS) |
| `gpu_recommend.py` | Match detected hardware → tier → recommended models + estimated tps |
| `gpu_runtime.py` | Runtime detection (CUDA / ROCm / Metal / CPU) for binary preflight |
| `server.py` | OpenAI-compatible `/v1/chat/completions` (stdlib HTTP) |
| `export.py` | Workspace → portable `.zip` chatbot bundle (vendored ezrag_lib + HTML/CSS/JS chat UI) |
| `preview.py` | On-demand PDF page-image rendering (pypdfium2) with 3-day disk cache |
| `sysmon.py` | 1 Hz CPU/RAM/GPU telemetry sampler for the GUI footer |
| `manual/*.md` | In-tool offline manual pages |
| `gui/ez_rag_gui/main.py` | Flet GUI (Workspace · Files · Chat · Settings · Doctor tabs) |

## Choices

- Ollama is the default LLM backend because it's free, cross-platform, and hands-off. llama-cpp-python is optional. If neither is present, `ez-rag ask` returns ranked passages instead of an LLM-written answer (still useful).
- fastembed (ONNX-based) is the default embedder so the package works offline immediately without a separate Ollama install. If Ollama is available, its `nomic-embed-text` model is used instead.
- SQLite + FTS5 + numpy is the index backend — single file, no server, fine to ~100k chunks. LanceDB is the obvious upgrade path when corpora outgrow this.
- Flet for the GUI: one Python codebase ships to Windows / macOS / Linux. The GUI shells through to the same library, so it can't drift from the CLI.

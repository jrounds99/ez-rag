# Changelog

All notable changes to ez-rag.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/);
versions follow nothing yet because this is alpha.

## [0.1.0] — Initial experimental release

### Core

- CLI scaffold (`ez-rag init / ingest / ask / chat / status / models / serve / doctor / reindex / help`)
- Workspace layout: `docs/`, `.ezrag/{config.toml,meta.sqlite,index/,cache/,models/}`
- Document parsers: PDF (text + scanned via OCR fallback), DOCX, XLSX, CSV, HTML, MD/TXT/RST/LOG, EPUB, EML, PNG/JPG/WEBP/TIFF/BMP (OCR'd)
- OCR pipeline: RapidOCR primary, Tesseract fallback
- Chunker: recursive token-target with overlap, structure-aware where possible
- SQLite + FTS5 index; embeddings stored as float32 BLOBs

### Retrieval

- Hybrid search: BM25 + dense cosine, fused with reciprocal rank fusion
- **Cross-encoder reranking** (default ON) — single biggest accuracy lift
- HyDE — hypothetical-document-embedding query expansion (opt-in)
- Multi-query — LLM-generated paraphrases fanned out + RRF (opt-in)
- MMR diversity rebalancing (opt-in)
- Context-window expansion — neighbor chunks per hit (opt-in)
- Contextual Retrieval (Anthropic-style chunk-context summaries) at ingest (opt-in)
- `Use corpus` toggle for live A/B test of model-only vs RAG-augmented

### Generation

- Backend auto-detect: Ollama → llama-cpp-python → retrieval-only fallback
- Reasoning-model support — `deepseek-r1`, `qwen3-reasoner`, etc.
  - `thinking` tokens streamed separately from `content` so the chat bubble
    is never silent
  - Dedicated dim "Reasoning…" panel in the GUI, live char count
  - Separate system prompts for RAG-on vs RAG-off (the latter forbids
    citation markers — fixes a real hallucination bug found during testing)
- Streaming via Ollama and llama.cpp with stop button
- Multi-turn chat with full conversation history sent each turn

### Models

- **Ollama library browser** — full searchable catalog of every public model
  on `ollama.com/library` (~230 models), capability filters
  (LLMs / Vision / Reasoning / Embedding), one-click pull with streaming
  progress, ETA, MB/s, total bytes
- VRAM estimates per size variant, color-coded against the user's GPU
  (green / amber / red)
- Local GGUF picker — switches the backend to llama-cpp-python
- Default-model adaptive picker by VRAM

### GUI (Flet)

- Tabs: Chat, Files, Settings, Doctor
- Drag-and-drop files into the workspace
- Streaming markdown chat, citation chips that open the source passage
- Settings card exposes every retrieval option with a tooltip
- Help (?) and About (ⓘ) overlays render the offline manuals
- Header status pills (workspace, file/chunk count, backend + model)

### Benchmarks

- `benchmark/run_benchmark.py` — RAG end-to-end on a public-document corpus
- `benchmark/rag_compare.py` — same Q-set with RAG on vs off, multi-model
- `benchmark/bench_configs.py` — retrieval-config matrix (every option × on/off)
- Reports as both Markdown and JSON in `benchmark/reports/`

### Documentation

- Offline manual pages bundled with the package (Rich rendering)
  — `getting-started`, `workflow`, `ingestion`, `retrieval`, `models`,
  `chat`, `gui`, `ocr`, `cli`
- README with install, quickstart, architecture, retrieval pipeline, model
  picks by VRAM
- `STATUS.md` — what works, what's flaky, what's not implemented

### Known issues

- Reranker model (`Xenova/ms-marco-MiniLM-L-6-v2`) downloads on first use; if
  rate-limited by HuggingFace it'll just fail back to no-rerank.
- File pickers in Flet 0.84 are async; we wrap with `page.run_task` — works
  but if you click another picker while one is open it's undefined.
- Pure-text PDFs with very small fonts can fool the "should we OCR" heuristic
  (chars-per-page threshold). Set `enable_ocr=false` if you trust the text
  layer.
- deepseek-r1's reasoning output uses ~600–2000 chars. Default `max_tokens`
  bumped from 1024 → 4096 to compensate; lower it if you want short answers.

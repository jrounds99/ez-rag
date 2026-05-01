# Changelog

All notable changes to ez-rag.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/);
versions follow nothing yet because this is alpha.

## [Unreleased]

### Added

- **Themes** — 10 preset palettes (`dark`, `light`, `midnight`, `forest`, `solarized_dark`, `solarized_light`, `nord`, `dracula`, `rose_pine`, `rainbow`) loaded from `gui/ez_rag_gui/themes.toml`. Switchable from Settings → Appearance with a live swatch preview. Custom palettes by editing the TOML.
- **Live system telemetry footer** — pinned bar showing CPU %, CPU temp, RAM, GPU compute %, VRAM, GPU temp, power draw. Updates at 1 Hz. nvidia-smi for GPU, psutil for CPU/RAM. Hides per-field when unavailable (e.g. CPU temp on Windows).
- **Chapter-aware retrieval** — opt-in `expand_to_chapter` mode. PDF bookmarks (via `pypdf.outline`) and DOCX/MD/HTML headings produce per-file chapter metadata persisted as JSON on the `files` row. Retrieval can replace each hit with its full chapter (capped at `chapter_max_chars`) — best for "summarize the rules around X" questions. Auto-dedupes when two hits land in the same chapter.
- **Citation page-image preview** — clicking a PDF citation chip renders that page on demand via `pypdfium2`, cached to `~/.ezrag/preview_cache/`. 3-day rolling lease (mtime touched on access). Side-by-side image + chunk text in the source dialog. Pan/zoom via `ft.InteractiveViewer` + Download button (saves PNG).
- **Export Chatbot** — bundles workspace into a runnable zip. Includes the index, frozen config, vendored `ezrag_lib/` (just retrieve+generate+index+embed+config), editable `chat.html/css/js`, stdlib `server.py` with streaming `/api/ask`, terminal `chatbot_cli.py`, and cross-platform launchers (`run.bat` / `run.sh`). Theme colors baked into the page CSS.
  - **`include_sources`** option — bundles `docs/` under `data/sources/` so citation chips in the chatbot show real PDF page renders and original screenshots, not just chunk text. Adds `/api/source` and `/api/page-image` endpoints to the chatbot server. GUI export overlay shows file count + size estimate before commit.
  - Chatbot's `chat.js` has clickable citation chips opening a modal viewer with pan/zoom/download for PDF pages and images, source-text fallback for HTML/MD/TXT, graceful note when sources aren't bundled.
- **Test corpus fetcher** — `bench/fetch_test_corpus.py` pulls curated Wikipedia article sets (dinosaurs, Greek mythology, US presidents, space exploration) under CC BY-SA 4.0 with attribution preserved.
- **Benchmarks** — `bench/bench_ollama.py` (5-model × 10-run baseline) and `bench/bench_optimizations.py` (flash attention / KV cache quant / num_ctx / num_batch sweep). On RTX 5090 + Ollama 0.12, the combination of `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` + `num_batch=1024` gave +7.9% throughput / -23% TTFT on deepseek-r1:32b.
- **Suggested questions on chat welcome** — replaced the hardcoded "Summarize the corpus" placeholders with an opt-in **Suggest questions** button that asks the LLM for 3 corpus-grounded questions (sampled from 12 random chunks). Cached per-workspace; rejects generic templates.
- **Recovery actions in chat error bubbles** — `OllamaChatError` now carries a `kind` (`load_failure` / `oom` / `context_overflow` / `model_not_found` / `server_down`). The chat bubble surfaces inline buttons appropriate to the error: `Reload model`, `Free all VRAM`, `Update Ollama`, `Retry question`, etc. Same buttons available pre-emptively in the chat header (`Test model`, `Reload model`, `Free VRAM`).
- **`unload_running_models()` + `list_running_models()`** in `models.py`. Settings save flow auto-evicts old VRAM contents when the user changes LLM or embedder. Header download badge shows model pulls regardless of dialog state, ticked by the sysmon ticker.
- **Per-page parser progress** — `parse_pdf` and `_ocr_pdf_pages` accept an `on_progress(page, total, ocr=False)` callback. Ingest emits `parsing page 47/231` with rate + ETA every 200 ms.
- **Per-chunk contextualization progress** — `enable_contextual` now emits `contextualizing chunk 52/598 (ETA 27m12s)` per chunk; in-flight chunk + bytes counters mean the headline meta line ticks smoothly mid-file rather than waiting for file commit.
- **Clearer ingest narration** — preflight plan summary, step counter (`[1/4] loading embedder…  ↪ ready in 4.2s`), per-step timings, "still loading… Ns elapsed" heartbeat thread for slow phases (fastembed first-run download).
- **Finished-state ingest panel** — completion shows a green success card (chunks added, files new/changed, elapsed) and bookends the streaming log.
- **Doctor view embedder-match check** — flags index-vs-current-embedder dimension mismatch before the next chat fails with a numpy matmul error.
- **Manage RAGs overlay storage** — multi-folder named RAGs, default RAGs folder global config, list/open/export/delete, import workspace zip.
- **Query modifiers** — `apply_query_modifiers` (per-chat checkbox + Settings card) prepends/appends and adds `Avoid:` directives.
- **Agentic retrieval** — opt-in `agentic` flag. LLM evaluates initial hits and (if needed) generates 1–2 follow-up queries; results fused with RRF + reranked once. Provider dispatcher supports `same` / `openai` / `anthropic`.

### Changed

- **Config**: new fields `expand_to_chapter`, `chapter_max_chars`, `num_batch` (default 1024 — measured fastest), `num_ctx`, `agentic`, `agent_*`, `query_prefix/suffix/negatives`, `enable_contextual`, `unload_llm_during_ingest`, `embed_batch_size`, `apply_query_modifiers`, `serve_host/port`. Per-request `_ollama_options` helper centralizes the per-call options dict for `/api/chat` and `/api/generate`.
- **Index schema**: idempotent migration adds `files.chapters_json` for chapter-aware retrieval.
- **Workspace**: TOML writer now uses single-quoted literal strings so Windows paths (with backslashes) and any user-typed strings round-trip cleanly. `find_workspace()` requires `<dir>/.ezrag/config.toml` not just `<dir>/.ezrag/` (so `~/.ezrag/` global-config dir isn't misidentified as a workspace).
- **Chat tab toggle** renamed `Use corpus` → **`Use RAG`** for clarity.
- **GUI threading**: long-lived tickers (sysmon sampler, ingest watchdog, ingest worker) converted from `page.run_thread` (sync, broken on Windows for periodic updates) → `page.run_task` (async, works). Fixes a Flet/Flutter Desktop frame-loop issue where `page.update()` from a sync thread queued values without painting until the user clicked / alt-tabbed / Win+S.
- **Win32 paint kicker** — `force_window_redraw()` finds the Flutter view via `FLUTTER_RUNNER_WIN32_WINDOW` class and pumps `RedrawWindow` + `InvalidateRect` + `UpdateWindow` once per second from the always-on sysmon ticker. Belt-and-suspenders for the same frame-loop issue. References [flutter#75319](https://github.com/flutter/flutter/issues/75319), [flutter#102030](https://github.com/flutter/flutter/issues/102030), [flet#4829](https://github.com/flet-dev/flet/issues/4829).

### Fixed

- Ollama error translator detects + reports actionable fixes for: corrupt model blob (`unable to load model`), OOM, context overflow, model-not-found, server down. Prompt size is included so users can see when they're near the context limit.
- Embedder-mismatch dimension error replaces the numpy stack trace with a clear "switch embedder back or re-ingest" message and inline action buttons.
- TOML round-trip bug — Windows paths in `~/.ezrag/global.toml` were silently failing to parse because backslashes were being interpreted as escape sequences.
- `tomllib` import was missing in `workspace._read_global` (silent NameError swallowed by bare except — global config dir was effectively read-only).
- Flet 0.84 `Dropdown(on_change=...)` keyword raises; assigned via attribute instead.
- `ft.ImageFit` doesn't exist in Flet 0.84 — pass `fit="contain"` as a string.
- Source dialog wraps in try/except so a render failure doesn't brick the chat.

### Tests

- 12 test suites, 292 assertions: round1–5 (core retrieval + modifiers + agent + ingest + storage), export-chatbot, export-sources, preview, chapters, suggestions, sysmon, ollama-errors, unload-on-switch.

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
- `Use RAG` toggle for live A/B test of model-only vs RAG-augmented

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

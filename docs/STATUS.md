# Status — what works, what's flaky, what's planned

ez-rag is **experimental / alpha**. It works end-to-end and we've tested it
on real documents and real models on a real RTX 5090, but expect rough edges.

## ✅ Solid

- **Ingest** of PDF / DOCX / XLSX / CSV / HTML / MD / TXT / RST / LOG / EPUB /
  EML / PNG / JPG / WEBP / TIFF / BMP. Idempotent re-runs (sha256-based
  skip), force-reingest, watch-mode polling.
- **OCR** — RapidOCR primary, Tesseract fallback. Auto-trigger on
  low-text-density PDFs.
- **Hybrid retrieval** (BM25 + dense + RRF) — measured, gold-doc-in-top-5 on
  every corpus question we tried.
- **Cross-encoder reranking** — measured 11/12 → 12/12 on substring grounding
  in the retrieval matrix; ~1 s per query.
- **Streaming chat** with reasoning-model support. `thinking` tokens render in
  a dedicated panel; `content` tokens stream into the bubble.
- **Multi-turn conversation memory** — full history sent each turn.
- **OpenAI-compatible HTTP** at `ez-rag serve`. Compatible with OpenAI SDK
  clients pointing at `localhost:11533`.
- **Ollama integration** — listing, pulling, deleting, library browsing,
  VRAM estimates per variant. Tested with deepseek-r1:32b (32 GB VRAM, 58
  tok/s).
- **fastembed fallback** — if Ollama isn't reachable, ez-rag still does
  retrieval with a local CPU embedder (`BAAI/bge-small-en-v1.5`).
- **OpenAI-compat server** — survives concurrent requests after the SQLite
  threading fix.
- **Citation handling** — `[1]`, `[2]` markers correctly inserted in RAG mode,
  correctly NOT inserted in no-RAG mode (separate system prompts).
- **CLI manual pages** — every topic (`ez-rag help <topic>`) renders.
- **GUI overlays** — Help, About/Credits, model browser, source-citation
  popup all open / close cleanly via `page.overlay` + visibility toggle
  (Flet 0.84 dialog stack threading bug avoided).
- **Cross-platform installs** — pipx / pip-user / Windows .bat self-install
  on first run.

## ⚠️ Works but with caveats

- **Cross-encoder reranker** (`Xenova/ms-marco-MiniLM-L-6-v2`, ~23 MB) is
  downloaded from HuggingFace on first use. If unauthenticated rate limits
  bite, `rerank_hits()` falls back to the input order silently. Fine, but
  invisible to the user.
- **VRAM color-coding** uses `nvidia-smi`. On Apple Silicon / AMD / no driver
  there's no fit decision; estimates are still shown.
- **Default reranker is English-focused.** Multilingual corpora should swap
  to `BAAI/bge-reranker-base` (~280 MB, set `rerank_model` in `config.toml`).
- **Ollama library scrape** parses HTML directly. If `ollama.com/library`
  changes its DOM, `fetch_ollama_library()` will degrade — the dialog still
  works (you can type a tag manually) but capability filters and size chips
  go blank until we update the parser.
- **Contextual Retrieval** (chunk-context summaries at ingest) is opt-in
  because it's slow: one LLM call per chunk. ~10 minutes for a 100-chunk
  corpus on deepseek-r1:32b. Use a small model for it (qwen2.5:3b takes
  ~30 s for the same corpus).
- **HyDE / Multi-query** add ~3–5 s per query (one extra LLM call). They
  did NOT improve accuracy on our small-and-clean test corpus — they help
  on harder corpora. Off by default; documented when to flip on.
- **MMR diversity** had no measurable effect on our test corpus either; off
  by default. Worth turning on for redundant / FAQ-style corpora.
- **Reasoning models eat `max_tokens`.** Default bumped 1024 → 4096; lower
  it if you want short answers, raise it for very long outputs.
- **Auto-scroll during streaming** is "always to bottom." If you scroll up
  to read something while a long answer streams in, it'll yank you back. We
  may add a "stick / unstick" affordance later.
- **Parsers don't share a unified table extractor.** XLSX and CSV are joined
  with ` | `; PDFs use whatever pypdf returns; DOCX tables are linearized.
  Good enough for retrieval but not faithful for analysis.

## ⛔ Known issues

- **Flet 0.84 file picker is async** — we wrap with `page.run_task`. If you
  spam the picker while one is already open, behavior is undefined; may need
  to close the orphan dialog manually.
- **`page.update()` from a worker thread** doesn't reliably propagate
  `dialog.open=False` mutations on `DialogControl`s. We worked around this
  by replacing AlertDialog with a plain `ft.Container` overlay everywhere.
  If you add a new modal, follow that pattern.
- **`ez-rag-gui.bat` first-run UX on Windows** — installs deps via
  `python -m pip install --user -e .`. If the pip install fails (e.g. behind
  a corporate proxy with no PyPI access) the batch pauses with the error so
  you can read it. There's no offline-installer fallback yet.
- **GGUF picker switches backend but doesn't auto-install `llama-cpp-python`.**
  You get a toast asking you to install it manually.
- **No telemetry** — that's by design, but it also means we don't know what
  breaks in the wild. Open an issue if you hit something.
- **No model auto-update.** If Ollama updates a tag (e.g. `qwen2.5:7b-instruct`
  gets re-pushed with new weights), ez-rag's index won't notice. Re-ingest if
  you change embedding models.

## 🚧 Not yet implemented

- **Watch mode that uses filesystem events** — current `--watch` polls
  `docs/` every ~2 s.
- **Workspace import/export** — copy your index between machines.
- **Conversation persistence** — chats live only in memory.
- **Multi-workspace switching in the running GUI** — works but each session
  is a single workspace.
- **Native installers** — no `.msi` / `.dmg` / `.AppImage` yet. Use pipx.
- **Code signing** — explicitly out of scope; documented unsigned-binary UX
  in [docs/INSTALL.md](INSTALL.md).
- **GraphRAG / Self-RAG / agentic retrieval** — possible future work, not
  on the immediate roadmap.

## How we actually test this

Three reproducible benchmarks live under [`benchmark/`](../benchmark/):

```bash
python benchmark/run_benchmark.py            # public-doc corpus, RAG end-to-end
python benchmark/rag_compare.py              # RAG on vs off, multiple models
python benchmark/bench_configs.py            # retrieval-option matrix
```

Reports go to `benchmark/reports/`. Read them before believing anything we
say about quality.

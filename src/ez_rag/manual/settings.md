# Settings — the expanded guide

Every knob, what it actually does, and when to touch it. Settings live
in `<workspace>/.ezrag/config.toml`; the GUI Settings tab edits the
same file. **If you don't want to read any of this: pick a preset**
(`ez-rag preset`, or Settings → Presets) — they bundle everything below
into benchmark-backed choices.

## Models

- **llm_model** — the chat model (Ollama tag). Default `granite3.3:2b`:
  the winner of our 23-model benchmark (10.95/12) and small enough for
  ~4 GB cards. Bigger is NOT automatically better — this 2.5B model
  beat every model up to 32B on answer quality. If exact facts matter
  more than fluency, `deepseek-r1:32b` measured the best factual
  accuracy (78% vs granite's 67%) at 10× the size and a tenth the speed.
- **llm_url** — where Ollama listens. Change only for a second daemon
  or another machine on your LAN. With `proprietary_data` on, non-local
  URLs are refused.
- **llm_provider** — `auto` tries Ollama, then llama.cpp, then
  retrieval-only. You almost never set this by hand; picking a local
  GGUF file in the GUI sets it to `llama-cpp` for you.
- **embedder_provider / ollama_embed_model / embedder_model** — who
  turns text into vectors. `auto` = Ollama if reachable (using
  `ollama_embed_model`), else fastembed on CPU (using
  `embedder_model`). Our bench found the top three embedders within
  0.1/12 of each other, so pick by size: `bge-m3:567m` (recommended,
  multilingual, 1.2 GB) or `nomic-embed-text` (0.4 GB). **Changing the
  embedder forces a full re-ingest** — vectors from different embedders
  don't mix (ez-rag detects this and re-ingests automatically).

## Ingest

- **chunk_size / chunk_overlap** — how documents are split (~tokens
  per chunk / overlap between neighbors). 512/64 won a 7-strategy
  chunking benchmark; leave alone unless you have a reason.
- **chunk_headers** *(default on)* — prepends a `[document › section]`
  breadcrumb (+ sidecar title/description) to every chunk before
  embedding. The cheap variant of contextual retrieval — measurable
  retrieval gains, zero LLM cost. Toggling re-ingests affected files.
- **dedup_chunks** *(default on)* — drops exact-duplicate chunks
  within a file (page headers/footers, boilerplate) so your top-k
  isn't three copies of the same disclaimer.
- **enable_ocr / ocr_provider** — OCR for scanned pages and images.
  `auto` tries RapidOCR (bundled, fast) then Tesseract (if installed);
  or force one engine, or `none` to skip OCR entirely (fastest ingest
  for born-digital corpora).
- **pdf_backend** — `auto` (built-in pypdf+OCR) or the experimental ML
  parsers `marker` / `docling` for gnarly scanned/tabular PDFs.
  Requires installing the library yourself; falls back per-file if it
  fails. See docs/INGEST_RESEARCH.md (including marker's license note).
- **enable_contextual** — classic per-chunk LLM contextualization.
  Strong but SLOW (one LLM call per chunk — hours on big corpora).
  `chunk_headers` gets most of the benefit for free; only turn this on
  for small, high-value technical corpora.
- **llm_inspect_pages / llm_correct_garbled /
  preview_garbled_recoveries** — LLM-assisted cleanup for garbled PDF
  extractions (broken font maps, bad OCR). Costs one LLM call per
  flagged section. Turn on when a specific corpus has mangled pages.
- **unload_llm_during_ingest** — evicts the chat model from VRAM while
  embedding, so the embedder gets the GPU. Leave on.
- **embed_batch_size** — texts per embedding call. Raise to 32–64 on a
  strong GPU for faster ingest.

## Retrieval

- **top_k** — how many chunks reach the LLM. 8 is the sweet spot for
  small/medium models; more context can *distract* small models. Go
  12–16 only with capable models or "walk me through" questions.
- **hybrid** — BM25 keyword search fused with vector search (RRF).
  Nearly free, covers each method's blind spots. Leave on.
- **rerank / rerank_model** — cross-encoder re-scoring of candidates.
  The single biggest quality lift in our retrieval matrix (~1s/query).
  Turn off only when chasing minimum latency (the 'speed' preset does).
- **use_hyde / multi_query** — query-expansion via the LLM (hypothetical
  answer / paraphrases). One extra LLM call each; helps when your
  question vocabulary doesn't match the corpus vocabulary. Off by
  default because they didn't move the needle on clean corpora.
- **auto_list_mode** — detects "list X / name some X" questions and
  switches to an extraction prompt + entity-rich retrieval.
  Dramatically better exploratory answers; leave on.
- **use_mmr / mmr_lambda** — diversity re-selection when your corpus
  has many near-duplicate passages. Now uses stored vectors (fast).
- **diversify_per_source** — cap chunks per file in results (default 3)
  so one big PDF can't monopolize the context.
- **context_window** — include ±N neighboring chunks per hit. Good for
  narrative/long-form docs; costs context budget.
- **expand_to_chapter / chapter_max_chars** — replace a hit with its
  whole chapter (from PDF bookmarks/headings). For "summarize the
  rules around X" questions.
- **crag_filter** — one batched LLM call that drops retrieved chunks
  irrelevant to the question. For noisy mixed corpora.
- **reorder_for_attention** — puts best hits at start AND end of the
  prompt (anti "lost in the middle"). Measured slightly WORSE on
  normal contexts; consider only for >32 KB prompts.

## Agentic retrieval

- **agentic** — the LLM inspects initial results and issues follow-up
  searches when they look thin. Adds latency; good for hard questions.
- **agent_provider / agent_model / agent_api_key / agent_base_url** —
  which model does the reflecting. `same` = your local chat model.
  Cloud options exist but are refused under `proprietary_data`.
- **agent_max_iterations** — reflect→search cycles (2 is plenty).

## Generation

- **max_tokens** — reply length cap (4096 default; reasoning models
  eat budget on thinking).
- **temperature** — 0.2 default: factual, low-drift answers.
- **num_ctx** — context window Ollama allocates. 0 = auto-size to the
  prompt (recommended). **num_ctx_cap** bounds the auto-sizing when
  VRAM-tight.
- **num_batch** — prompt-processing batch. 1024 measured +8%
  throughput / −23% time-to-first-token vs the 512 default.

## Compression (optional)

- **compress_context / compress_context_min_tokens /
  compress_context_target_ratio** — shrink retrieved context ~49%
  before the LLM sees it (headroom-ai; `pip install "ez-rag[compress]"`).
  Measured quality delta −0.40/12 at 16k ctx. Worth it for long
  contexts and VRAM-tight machines; skip for short chats. Fails open
  if not installed.

## Proprietary data

- **proprietary_data** — the "nothing leaves this machine" flip:
  non-local endpoints raise, cloud agent providers refused. Pair with
  `ez-rag lock` / `unlock` (AES-256-GCM index encryption). Full
  explanation: docs/PROPRIETARY_DATA.md.

- **redact_terms / redact_replacement / redact_smart** —
  context-aware removal at ingest: listed terms (names, emails, IDs)
  never reach the index, FTS, or vectors. Two-word names match their
  variants; ambiguous single words use smart casing ("Stone" the
  name redacted, "crushed stone" kept). Exports verify the index is
  clean and refuse include_sources while set. `ez-rag redact-check`
  audits anytime.

## Query modifiers & metadata

- **apply_query_modifiers / query_prefix / query_suffix /
  query_negatives** — persistent wrappers around every question
  (persona, formatting, "avoid X").
- **use_file_metadata** — reads `<file>.ezrag-meta.toml` sidecars
  (title/topics/entities/modifiers) at query time. `ez-rag scan`
  writes them for you.

## Server

- **serve_host / serve_port** — the OpenAI-compatible endpoint for
  `ez-rag serve` (default 127.0.0.1:11533).

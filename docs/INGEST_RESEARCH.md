# Ingest techniques — research survey (2026-08)

A structured survey of current RAG ingestion/indexing techniques,
evaluated against ez-rag's hard constraints: **100% free/local, consumer
hardware (Ollama / fastembed / ONNX), permissive licenses, Windows,
no new heavyweight services.** Each verdict is ADOPT / TRIAL / SKIP
*for ez-rag specifically* — a SKIP here is not a judgment that the
technique is bad in general.

The headline: ez-rag's existing base (recursive 512-token chunking +
hybrid BM25/dense + cross-encoder rerank) is already the
evidence-backed winning baseline. The gains left on the table are in
**what goes into each chunk** (headers, structure) and **what stays out
of the index** (near-duplicates) — not in exotic chunkers or new
retrieval stacks.

## Chunking

| Technique | Verdict | Why |
|---|---|---|
| **Recursive token-target (current)** | KEEP | Won a 7-strategy Feb-2026 benchmark (69% vs semantic's 54%). [firecrawl.dev/blog/best-chunking-strategies-rag](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) |
| **Late chunking** (Jina) | SKIP | ~3.6% relative gain ([arXiv 2409.04701](https://arxiv.org/pdf/2409.04701)); needs token-level embedder outputs Ollama's API can't expose; best model is CC-BY-NC. |
| **Semantic chunking** | SKIP | Benchmarks *worse* than the recursive baseline (see above) at much higher ingest cost. |
| **Proposition chunking** | SKIP | LLM call per chunk, index bloat, marginal over a rerank pipeline. |
| **Structure-aware chunking** | **ADOPT** | Never split tables mid-row; carry `title › section` breadcrumbs into chunks. Consistent wins in layout-aware studies ([ACM](https://dl.acm.org/doi/10.1007/978-981-95-4969-6_3)). ez-rag already tracks `ParsedSection.section`, so this is plumbing, not research. |

## Document parsing

> **License policy note (project owner decision, 2026-08):** ez-rag is a
> personal research tool, not a commercial product. License conditions
> that only bite on commercial use or redistribution (revenue-capped
> weights, copyleft) are **acceptable for optional, user-installed
> extras** and are documented here rather than treated as
> disqualifying. Two boundaries still hold: everything must be free to
> use at this project's scale, and nothing non-permissive ships as a
> hard dependency of the Apache-2.0 core — these tools are only ever
> installed explicitly by the user, keeping the repo itself cleanly
> licensed.

| Tool | Verdict | Why |
|---|---|---|
| **marker** | **TRIAL** (available: `pdf_backend = "marker"`) | Fastest/most accurate in [2026 comparisons](https://themenonlab.blog/blog/best-open-source-pdf-to-markdown-tools-2026). Weights are OpenRAIL-M with a commercial revenue cap — free for personal research per the policy note; review only if output ever feeds a commercial product. Heavy (PyTorch). `pip install marker-pdf`, then set `pdf_backend = "marker"`. |
| **Docling** (IBM) | **TRIAL** (available: `pdf_backend = "docling"`) | MIT code, permissive models, CPU-capable layout + TableFormer, RapidOCR backend ([arXiv 2501.17887](https://arxiv.org/pdf/2501.17887)). Clearly stronger than pypdf on tabular/scanned PDFs; the cleanest-licensed of the ML parsers. `pip install docling`, then set `pdf_backend = "docling"`. |
| surya | possible | GPL-3 — fine for personal use as a user-installed extra. It's the engine underneath marker, so trial marker instead of using it directly. |
| MinerU | SKIP | Custom license + heavy, with no evidence it beats marker/Docling on our document types. |

Both alternative backends are wired into ez-rag as **experimental
options** (config `pdf_backend = "auto" | "marker" | "docling"`,
default `"auto"` = the built-in pypdf + OCR pipeline). They fail open:
if the library isn't installed or a conversion errors, the file falls
back to the built-in parser and a status line says so. Switching
backends re-ingests PDFs automatically (backend is part of parser
provenance).

## Embedding & indexing

| Technique | Verdict | Why |
|---|---|---|
| **Binary/int8 quantization + rescore** | TRIAL | `np.packbits` + XOR/popcount, rescore top-N in float32 — near-float32 accuracy ([Vespa](https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/)). But brute-force float32 is already fast at consumer corpus sizes (<100k chunks ≈ 300 MB). A *scale* feature — revisit if corpora grow past that. |
| **Matryoshka two-stage** | TRIAL | nomic-embed-text v1.5 (our default) is MRL-trained; truncate-to-256 search + full-dim rescore. Pair with quantization if/when scaling. |
| ColBERT late interaction | SKIP | fastembed ships ONNX ColBERT, but token matrices inflate the index ~50× and our MiniLM cross-encoder already covers the rerank stage. |
| SPLADE | SKIP | +0.002–0.105 nDCG@10 over BM25 ([comparison](https://suhasbhairav.com/blog/splade-vs-bm25-learned-sparse-retrieval-vs-traditional-keyword-scoring)), ~160 ms/doc CPU encode; BM25+dense hybrid already captures most of the gap. Apache-2.0 model exists in fastembed if ever wanted. |

## Ingest-time enrichment

| Technique | Verdict | Why |
|---|---|---|
| **Contextual chunk headers (cheap variant)** | **ADOPT** | Prepending one per-**document** summary + section breadcrumb to every chunk matches per-chunk LLM contextualization in effectiveness at ~1/50th the LLM cost ([Snowflake](https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/), [contextual chunk headers](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb)). ez-rag's `scan` already writes per-file summaries into sidecars — reuse them for free. |
| Synthetic questions (HyPE / doc2query--) | TRIAL | Works ([HyPE](https://arxiv.org/html/2607.29402)) but costs an LLM call per chunk — same pain as classic contextual retrieval. Candidate alternative mode for the existing `enable_contextual` flag. |

## Incremental / robust ingest

| Technique | Verdict | Why |
|---|---|---|
| **Chunk-level near-dup detection** | **ADOPT** | File-level sha256 delta already exists; chunk-level doesn't. Versioned docs and boilerplate headers/footers create near-dup chunks that actively poison RRF/top-k diversity ([Milvus on MinHash-LSH](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md), [dedup analysis](https://arxiv.org/pdf/2605.09611)). Implementable with stdlib hashing — no new dependency. |

## Top 3 for the next ingest

1. **Contextual chunk headers** — `Document › Section` breadcrumb (+
   sidecar summary when present) prepended to each chunk before
   embedding. Small effort; captures most of contextual retrieval's
   measured ~35–49% failure reduction at a fraction of the cost.
2. **Structure-aware chunking** — tables stay atomic; recursive-512
   base unchanged.
3. **Chunk-level dedup at ingest** — skip embedding duplicate chunks;
   keeps top-k diverse.

Optional 4th: bench **Docling** against pypdf+RapidOCR on the worst
scanned/tabular PDFs in the new corpus; adopt as an optional extra only
if it clearly wins.

---
*Survey conducted 2026-08-12 with web cross-checking; links inline.
See CHANGELOG for which recommendations have since been implemented.*

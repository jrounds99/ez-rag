# Retrieval-config measurement matrix

A retrieval-only benchmark (no LLM generation) to measure each option's
effect on retrieval quality and latency. Question set is intentionally a mix
of easy / paraphrased / vague / multi-hop / lexical-mismatch.

## Setup

- Workspace: 8 documents, 43 chunks (apollo.txt, attention.pdf,
  biology.docx, dogs.md, ocean.html, screenshot.png, store.xlsx,
  wikipedia_rag.html)
- Queries: 12 questions
- Embedder: `nomic-embed-text` via Ollama (RTX 5090, 32 GB VRAM)
- Reproduce: `python benchmark/bench_configs.py`

## Results

| Config | Top-1 | Gold@5 | Sub-hit | Avg latency | Verdict |
|---|:---:|:---:|:---:|---:|---|
| `01` Hybrid only | 11/12 | 11/12 | 10/12 | 12 ms | baseline |
| `02` + Rerank (MiniLM-L-6) | 11/12 | **12/12** | **12/12** | 1056 ms | **the only option that lifted accuracy** |
| `03` + Rerank (BGE-base) | 11/12 | 12/12 | 12/12 | 888 ms | same accuracy, +280 MB download |
| `04` + Rerank + MMR(λ=0.5) | 11/12 | 12/12 | 12/12 | 1126 ms | no measurable lift on this corpus |
| `05` + Rerank + MMR(λ=0.7) | 11/12 | 12/12 | 12/12 | 1167 ms | same |
| `06` + Rerank + Window(±1) | 11/12 | 12/12 | 12/12 | 887 ms | qualitative win, no top-K accuracy lift |
| `07` + Rerank + Window(±2) | 11/12 | 12/12 | 12/12 | 861 ms | same |
| `08` + Rerank + HyDE | 11/12 | 12/12 | 12/12 | 4708 ms | +3.5 s for no measurable lift |
| `09` + Rerank + Multi-query | 11/12 | 12/12 | 12/12 | 4625 ms | same |
| `10` Rerank + Window + MMR + HyDE | 11/12 | 12/12 | 12/12 | 4510 ms | kitchen sink, no marginal gain |

## Analysis

### What rerank fixed

The two questions where hybrid-only failed substring-grounding:

- **Q11** *"Which spreadsheet sheet contains pricing?"* — pure hybrid surfaced
  `biology.docx` first (false). With rerank, the right xlsx chunk ended up in
  the top-5 (gold@5 ✓). Rerank still missed top-1 because the xlsx tabular
  text reads poorly for cross-encoders, but the LLM can pick from the top-5.
- **Q12** *"What architecture does the Attention paper introduce?"* — pure
  hybrid returned the correct PDF as #1 but didn't include the chunk that
  literally says "Transformer." Rerank promoted that chunk into the visible
  top-K. **This is the failure mode rerank was designed to catch.**

### Why advanced techniques didn't move the needle

The corpus is small (8 docs, 43 chunks) and topically diverse. Rerank already
gets us to the ceiling — there's nothing left for HyDE / Multi-query / MMR
to fix in this dataset.

These techniques are designed for harder conditions:

- **HyDE** helps when corpus and questions use different vocabularies (e.g.
  customer-support tickets vs. internal jargon).
- **Multi-query** helps when the same concept can be asked many ways
  (typical of natural conversations on big knowledge bases).
- **MMR** helps when the top-K is full of near-duplicate paragraphs (long
  PDFs, FAQs that repeat themselves).
- **Window/parent expansion** helps when a chunk is meaningful only in
  context of its neighbors (legal text, narrative, code).

A bigger / messier corpus would expose differences. We can revisit the
benchmark with a 100+ doc corpus later.

## Recommendations & default policy

| Option | Default | Notes |
|---|:---:|---|
| Hybrid (BM25 + dense + RRF) | **ON** | always-win, near-zero cost |
| Cross-encoder rerank (MiniLM) | **ON** | clear quality lift, ~1 s overhead |
| HyDE | OFF | enable for vague queries on hard corpora; +LLM call |
| Multi-query | OFF | enable for paraphrase-heavy use cases; +LLM call |
| Context window expansion | OFF | enable for narrative/long-form docs; multiplies LLM tokens |
| MMR diversity | OFF | enable for redundant corpora |
| Contextual Retrieval (ingest-time) | OFF | enable for technical docs; slow ingest |
| Larger reranker (`BAAI/bge-reranker-base`) | OFF | offered as a swap; +280 MB |

## What's worth shipping

All eight options ship as configurable. **Defaults stay at hybrid + rerank**
because the data shows everything else is corpus-dependent: useful when needed,
neutral or negative when not. Each is documented with its specific cost/benefit
in the GUI tooltip and the manual page `ez-rag help retrieval`.

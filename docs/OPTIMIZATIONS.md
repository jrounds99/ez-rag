# RAG Optimization Workflow

Living technical reference for every optimization in ez-rag's
retrieval + generation pipeline. Updated as new techniques land.

> **How to read this doc:**
> - Section 1 is the big-picture pipeline diagram.
> - Section 2 covers ingest-side optimizations (one-time at index build).
> - Section 3 covers retrieval-side optimizations (per-query).
> - Section 4 covers generation-side optimizations (per-answer).
> - Section 5 is the empirical evidence — what we tested, what the data
>   actually says.
> - Section 6 is research-backed but-not-implemented ideas, ranked by
>   expected ROI.

---

## 1. Full pipeline (one diagram, all stages)

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INGEST  (one-time)                          │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  parse   │ →  │ heuristic│ →  │   OCR    │ →  │  chunk   │
  │  pypdf   │    │ garbled  │    │ fallback │    │  (512t   │
  │          │    │ detector │    │          │    │  + 64    │
  │          │    │          │    │          │    │ overlap) │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                        │
       ┌────────────────────────────────────────────────┘
       ▼
  ┌──────────┐  opt   ┌──────────┐  opt   ┌──────────┐
  │   LLM    │ ────→  │   LLM    │ ────→  │contextual│ ────→  ┌────────────┐
  │ inspect  │        │ correct  │        │  embed   │        │   Index    │
  │ (reject  │        │ (cleanup │        │ prefix   │        │ FTS5+vec   │
  │ garbled) │        │  OCR)    │        │ (recall+)│        │ + chapters │
  └──────────┘        └──────────┘        └──────────┘        └────────────┘


┌──────────────────────────────────────────────────────────────────────┐
│                         RETRIEVE  (per query)                        │
└──────────────────────────────────────────────────────────────────────┘
   user question
       │
       ▼
  ┌──────────────────────────────────────────────────┐
  │    auto-detect "list X / give examples of X"    │
  │  triggers? "tell me about", "the most ...",     │
  │  "interesting/memorable/notable", etc.          │
  └──────────┬───────────────────────────────────────┘
             │
       ┌─────┴──────┐
       │            │
   list query    other query
       │            │
       ▼            ▼
  ┌─────────┐   ┌─────────┐
  │ entity- │   │ generic │
  │  rich   │   │  HyDE   │  (only when cfg.use_hyde)
  │  HyDE   │   │ (opt)   │
  │ topic-  │   └─────────┘
  │ anchored│
  └────┬────┘
       │
       │   embed expanded query
       ▼
  ┌────────────────────────────────────────────────┐
  │            HYBRID SEARCH (always)              │
  │   ┌──────────┐         ┌──────────┐            │
  │   │   BM25   │   +     │  Dense   │            │
  │   │  (FTS5)  │         │ vectors  │            │
  │   └────┬─────┘         └────┬─────┘            │
  │        └──── RRF fusion ────┘                  │
  └───────────────────┬────────────────────────────┘
                      │  fetch_k = max(top_k×2+6, 20)
                      ▼
              ┌────────────────┐
              │  CROSS-ENCODER │   opt
              │     RERANK     │
              │ MiniLM L-6-v2  │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │  DIVERSIFY by  │   default: cap=3
              │    source PDF  │
              │  (cap-per-src) │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │ MMR (opt) or   │
              │ rank order     │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │  CRAG filter   │   opt — 1 extra LLM call
              │  (drop irrel.) │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │  EXPAND        │
              │  (chapter      │   capped at chapter_max_chars
              │   ± neighbors) │   (auto-tightened to 4 KB
              └────────┬───────┘    in list mode)
                       ▼
              ┌────────────────┐
              │  REORDER       │   opt; 'lost in the middle'
              │  (rank-1 first │   default OFF on small contexts
              │   rank-2 last) │
              └────────┬───────┘
                       ▼
                  retrieved chunks
                       │
                       ▼

┌──────────────────────────────────────────────────────────────────────┐
│                        GENERATE  (per answer)                        │
└──────────────────────────────────────────────────────────────────────┘
   ┌──────────────────────────────┐
   │   build chat messages:       │
   │   [system, history…, user]   │
   │   user = question + context  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   AUTO-SIZE num_ctx          │   queries Ollama for the
   │   prompt tokens →            │   model's max_ctx, picks
   │   {4k, 8k, 16k, 32k, 64k}    │   smallest bucket that fits
   │   bucket fitting under       │   prompt + reply
   │   model max                  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   list query?                │
   │   yes → SYSTEM_PROMPT_       │
   │         LIST_EXTRACTION      │
   │   no  → SYSTEM_PROMPT_RAG    │
   └──────────────┬───────────────┘
                  ▼
            Ollama / llama.cpp
                  │
                  ▼
              answer + citations
```

---

## 2. Ingest-side optimizations

| ID | Stage | What it does | Status | Cost | Win |
|----|-------|--------------|--------|------|-----|
| I-1 | pypdf parse | Extract text from PDFs via cmap | Always | Free | Baseline |
| I-2 | Heuristic garbled detector | Per-page check: replacement-char ratio, backslash escape ratio, vowel ratio, non-alnum ratio | Always | Free | Critical for broken-cmap PDFs |
| I-3 | OCR fallback | Re-extract garbled pages via pypdfium2+RapidOCR | Default ON | ~500 ms-2 s/bad page | Recovers ~95% of broken pages |
| I-4 | TOC-fragment OCR drop | Reject "Preface / 4 / Fighter / 59" salad even when not garbled | Always | Free | Prevents search-poison chunks |
| I-5 | LLM inspect pass | Per-section LLM call: `clean / partial / garbled` → drop garbled | Opt-in | 1 LLM call per section (slow) | Catches ~5% the heuristic missed |
| I-6 | LLM correct pass | LLM cleanup of OCR-recovered or partial-flagged sections; UNRECOVERABLE → drop | Opt-in | 1 LLM call per questionable section | Repairs OCR misreads in-place |
| I-7 | Anthropic contextual retrieval | Prepend a 1-sentence situational context to each chunk before embedding | Opt-in (`enable_contextual`) | 1 LLM call per chunk (very slow) | Up to **−67% retrieval errors** per Anthropic |
| I-8 | Chunking 512/64 | Split into ~512-token chunks with 64-token overlap | Always | Free | Good general default |
| I-9 | Chapter detection | Build chapter index from PDF outline + heading detection | Always | Free | Enables expand-to-chapter |
| I-10 | Per-page recovery preview | Save 2× page render + before/after text for live GUI preview during ingest | Opt-in | ~50 ms + ~200 KB / recovered page | Pure UX |

### I-7 deep dive: Contextual Retrieval

When ON, every chunk gets a 1-sentence prefix from the LLM:

```
Original chunk:        "...the spell does 8d6 fire damage..."
Contextualized chunk:  "From PHB chapter 11 'Spells', describing
                        Fireball at higher levels: ...the spell does
                        8d6 fire damage..."
```

The contextualized text is what gets embedded. The chunk's display text
stays original. Result: a query like *"how much damage does Fireball
do at 4th level"* matches the prefix's "Fireball at higher levels"
content even though the original chunk only had a stat block.

**Cost**: One LLM call per chunk × ~24,000 chunks = ~24,000 calls. With
qwen2.5:7b at ~5 s/call that's 33 hours. Anthropic's prompt caching
makes this much cheaper on Claude API; on local Ollama it's a slog.

Recommendation: enable for high-stakes corpora. Skip for casual use.

---

## 3. Retrieval-side optimizations

### 3.1 Auto-detect list queries

```
def _is_list_query(text):
    triggers = [
        "list", "name some", "give examples", "what are some",
        "suggest some", "tell me about the most", "interesting characters",
        "memorable", "notable", "give me some", "show me some",
        "examples of", "names of", "the most ...", "any examples", ...
    ]
    return any(t in text.lower() for t in triggers)
```

When a list query is detected, the pipeline switches to a different
mode:

- **Retrieval**: entity-rich HyDE instead of bare query
- **top_k**: bumped to ≥16
- **chapter_max_chars**: capped at 4000 (vs user's setting)
- **context_window**: forced to 0
- **System prompt**: LIST_EXTRACTION (forces named-item extraction)

### 3.2 Entity-rich HyDE (vs standard HyDE)

```
Standard HyDE prompt:                 Entity-rich (list) HyDE prompt:
─────────────────────                 ──────────────────────────────
Write a confident                      User wants a LIST of named items
single-paragraph answer.               for: "{query}"
2-3 sentences.                         Pack 4-6 specific named examples
                                        (proper nouns, capitalized) into
                                        2-3 sentences. Stay TIGHTLY on
                                        the user's literal topic.
```

Why it matters: standard HyDE produces summary-style answers that embed
well against explanatory prose, but POORLY against stat-block / sidebar
/ table chunks where named entities live. The entity-rich variant
matches reference-book content far better.

### 3.3 Hybrid search (BM25 + dense + RRF)

```
                 ┌──────────────┐
   query ────→   │    BM25      │ ──→ ranked by lexical match
                 │   (FTS5)     │
                 └──────────────┘
                                       Reciprocal Rank Fusion:
                                       score(d) = Σ 1/(60 + rank_i(d))
                 ┌──────────────┐
   query ────→   │    DENSE     │ ──→ ranked by cosine similarity
   (embedded)    │ (numpy cosine)│
                 └──────────────┘
```

Always on. Single biggest quality lift over either method alone.

### 3.4 Cross-encoder rerank

After hybrid search returns top-K candidates, a cross-encoder model
(`Xenova/ms-marco-MiniLM-L-6-v2`) re-scores each `(query, passage)`
pair with full attention. Catches relevance that bi-encoder cosine
similarity misses.

Cost: ~50–200 ms. ROI: very high. Keep on.

### 3.5 Source diversification

```
Without diversification:                 With cap-per-source=3:
────────────────────────                 ───────────────────────
Retrieved top-8:                          Retrieved top-8:
  PHB.pdf                                   PHB.pdf
  PHB.pdf                                   PHB.pdf
  PHB.pdf                                   PHB.pdf
  PHB.pdf                                   DMG.pdf       ← skipped 4 PHB
  PHB.pdf                                   DMG.pdf
  PHB.pdf                                   Tasha.pdf
  DMG.pdf                                   Tasha.pdf
  DMG.pdf                                   Volo.pdf
```

Forces the LLM to ground answers across multiple sources instead of
letting one PDF dominate. Implemented as a post-rank pass after
fetching `cfg.top_k * 2 + 6` candidates.

### 3.6 CRAG-style chunk relevance filter (opt-in)

After retrieval, ONE batched LLM call evaluates which chunks are
actually relevant:

```
LLM input:                              LLM output:
──────────                              ───────────
QUESTION: how does grappling work       1, 3, 5
PASSAGES:
  [1] Grappling rule: PHB p.195...
  [2] Charm Person duration: PHB p.221  (filtered out)
  [3] Athletics check rules...
  [4] Polymorph spell description...    (filtered out)
  [5] Conditions: prone, grappled...
  ...
```

Single LLM call instead of N (one per chunk). Costs ~1 small inference
per query. ROI is corpus-dependent — high for noisy mixed corpora,
neutral for clean focused ones.

### 3.7 Lost-in-the-middle reorder (opt-in)

```
Original rank order:        Reordered for attention:
[r1, r2, r3, r4, r5, r6]    [r1, r3, r5, r6, r4, r2]
                             ──top── ────── ──end──
                             attended      attended
```

LLMs attend most to the start and end of long contexts; middle gets
ignored ([Stanford 2023, arXiv:2307.03172](https://arxiv.org/abs/2307.03172)).
Interleave the rank order so the highest-ranked chunks land at both
attention bookends.

**Empirical finding**: helps on long contexts (>32 KB). On our bench
(~16 KB total context after expansion) the LLM-as-judge slightly
*preferred* original RRF order. **Default OFF** in ez-rag — turn on
for very long context windows.

### 3.8 Chapter / neighbor expansion

After top-K is selected, optionally expand each hit:

- **Chapter expansion** (`expand_to_chapter`): replace hit text with
  the full chapter text, capped at `chapter_max_chars`. Falls back to
  original chunk if chapter exceeds the cap.
- **Neighbor expansion** (`context_window=N`): include ±N chunks
  around each hit, joined with blank lines.

**Empirical finding**: smaller `chapter_max_chars` (2 KB-8 KB) beat the
default 16 KB on judge scores. The 16 KB setting causes the LLM to
*summarize* chapters instead of *extract* from them. List mode forces
the cap to 4 KB regardless of user setting.

---

## 4. Generation-side optimizations

### 4.1 Auto-sized `num_ctx` (CRITICAL)

This was the **biggest single bug** in ez-rag. Ollama defaults
`num_ctx` to 4096 tokens. With chapter expansion to 16 KB chunks at
top_k=8, prompts hit ~30,000 tokens — and 80% silently got dropped
before the LLM ever saw them.

Fix:

```python
def _auto_num_ctx(cfg, messages):
    # Estimate prompt tokens (~3.5 chars/token English)
    chars = sum(len(m['content']) for m in messages)
    needed = chars / 3.5 + cfg.max_tokens + 256

    # Probe model's native max via /api/show
    max_ctx = model_max_ctx(cfg)   # e.g. 32768 for qwen2.5:7b

    # Pick smallest fitting bucket so Ollama doesn't reload often
    for bucket in (4096, 8192, 16384, 32768, 65536, 131072):
        if needed <= bucket and bucket <= max_ctx:
            return bucket
    return max_ctx
```

**Result on bench**: keyword scores **66.9% → 86.8%** (+20 pp).

### 4.2 List-extraction system prompt

When `_is_list_query(question)` returns True, swap the default RAG
system prompt for one that forces named-item extraction:

```
You are an information-extraction assistant. The user is asking
for a LIST of specific named items.

1. SCAN every excerpt for proper nouns / capitalized names / table
   entries.
2. Output a BULLETED LIST. Each bullet:
     - <name> — <one sentence of context> [filename, page N]
3. Do NOT explain general concepts. Do NOT define the term. Do NOT
   pivot to "here's how X works".
4. EXTRACT names buried in unrelated paragraphs.
5. < 3 examples found → say "Only N specific examples found:" and
   list what you have. No padding.
6. Zero examples → "Not found in the indexed documents."
```

### 4.3 Strict prompts (tested, REJECTED)

We tested aggressive constraint prompts (`cite-required`, `extract-
verbatim`). Result: judge scores DROPPED:

```
ch-2k:            10.35 / 12   (control)
cite-required:     8.60 / 12   ↓
extract-verbatim:  6.95 / 12   ↓↓
```

Lesson: small models like qwen2.5:7b can't handle aggressive
constraints — they refuse or extract too tersely. Use the moderate
list-extraction prompt only when the auto-list heuristic fires.

### 4.4 Streaming with stage tracking

Both retrieval and generation emit stage events that the GUI's
workflow chip strip pulses in real time. See `gui/ez_rag_gui/main.py`
`chat_set_active_stage` + `smart_retrieve(status_cb=...)`.

---

## 5. Empirical evidence (loops + judge data)

All numbers from `bench/bench_dnd5e_quality.py` against
`C:\Users\jroun\Desktop\dnd books\2014 (5e)`, qwen2.5:7b, 55-question
mixed set across exploratory / rule / comparison / multi-step
categories. Judge: same model with strict 0-12 rubric.

### Loop 1: pre-fix baseline (broken)

| | keyword % |
|---|---|
| baseline (no fixes) | 33.3% (exploratory) |
| hyde+strict | 62.5% (exploratory) |

Key finding: **keyword heuristic is unreliable**. Real wins / regressions
got hidden by it. Switched to LLM-as-judge.

### Loop 2: chapter expansion sweep

| Strategy | judge total /12 |
|---|---|
| ch-off (no expand) | 10.30 |
| ch-2k | **10.45** |
| ch-4k | 10.15 |
| ch-8k | **10.45** |
| ch-neighbor (no chap, ±1) | 10.20 |

Tied at 10.45. Smaller chapters beat user's default 16 KB by lifting
specificity 2.20 → 2.45.

### Loop 3: grounding interventions (ALL HURT)

| Strategy | judge total /12 | Δ |
|---|---|---|
| ch-2k | 10.35 | (control) |
| cite-required | 8.60 | −1.75 ❌ |
| extract-verbatim | 6.95 | −3.40 ❌❌ |
| cite-+-list | 9.20 | −1.15 ❌ |

Strict grounding prompts caused refusals and over-terse answers on
qwen2.5:7b. Don't ship.

### Loop 4: lost-in-middle reorder + top_k

| Strategy | judge /12 | Δ vs control |
|---|---|---|
| ch-2k-no-reorder | **10.70** | (control) |
| ch-2k-reorder | 10.30 | −0.40 |
| ch-2k-top4 | 10.25 | −0.45 |
| ch-2k-top12 | 10.50 | −0.20 |

Surprise: reorder slightly *hurt* on this corpus. Default OFF; revisit
if context grows past 32 KB.

### Loop 5: CRAG relevance filter

| Strategy | judge /12 | Δ vs control | Latency |
|---|---|---|---|
| ch-2k-no-reorder | **10.35** | (control) | 8.4 s |
| ch-2k-crag (top_k=12) | 10.05 | −0.30 | 8.3 s |
| ch-2k-crag-top16 | 10.25 | −0.10 | 8.1 s |

CRAG modestly hurt + adds an extra LLM call. The retrieval is already
clean enough on this corpus that the filter mostly drops borderline-
relevant chunks that were actually contributing. Keep the flag for
users with noisier corpora; **default OFF** in ez-rag.

### Auto-`num_ctx` fix (separate measurement)

| | keyword % | judge /12 |
|---|---|---|
| Before fix (4 KB ctx truncation) | 66.9% | 10.24 |
| After fix (auto-bucketed ctx) | **86.8%** | (full bench TBD) |

**The single highest-impact change.** Was masking everything else.

---

## 6. Research-backed but not yet implemented

Ranked by expected ROI / effort.

### 6.1 Semantic chunking (high ROI)

Replace fixed 512-token chunks with semantic boundaries: split on
embedding-similarity drops between consecutive sentences. Reported to
lift accuracy from baseline 60% → 71% in vendor benchmarks.

**Effort**: Medium. Requires re-ingest. New chunker module.

### 6.2 Late chunking (medium ROI)

Embed the entire document at the token level FIRST, then chunk after.
Chunks inherit document-level context implicitly. Avoids the per-chunk
LLM call that contextual retrieval needs. Less accuracy lift than I-7
but free at inference time.

**Effort**: Large. New embed pipeline. Requires re-ingest.

### 6.3 Self-RAG / reflection on the answer (medium ROI)

After generation, ask the LLM: *"is your answer grounded in the
context? cite specific lines."* If not, regenerate. Adds one round-trip
but catches drift.

**Effort**: Small. Loop in `answer()` with a max-2 retry budget.

### 6.4 Larger reranker (small ROI)

Swap MiniLM-L-6-v2 (23 MB) for `BAAI/bge-reranker-large` (560 MB).
Better relevance scoring at ~5× the latency.

**Effort**: Tiny. Just change the default.

### 6.5 Adaptive top_k (small ROI)

Look at the score distribution after rerank. If there's a sharp
drop-off at rank 3, only keep top-3. If scores are flat, keep more.

**Effort**: Small. ~30 lines in retrieve.py.

### 6.6 Query routing by category (uncertain ROI)

Detect whether a query is:
- definitional ("what is X")
- comparative ("compare X and Y")
- exploratory / list ("name some X")
- procedural ("how do I X")
- multi-hop ("explain why X causes Y")

Each gets a different retrieval strategy. We already do this for list
queries; extending to comparative would mean retrieving twice (once
per side) and merging.

**Effort**: Medium. Could regress on edge cases.

---

## 7. Current default config (recommended)

After all loops, these are ez-rag's recommended defaults:

```python
# Retrieval
hybrid              = True       # always on
rerank              = True       # always on (cheap, big win)
top_k               = 8          # bumped to 16 in list mode
diversify_per_source = 3          # cap-per-source for variety
auto_list_mode      = True       # auto-route list queries
use_hyde            = False      # auto_list_mode supersedes for list Qs
multi_query         = False      # expensive, niche win
use_mmr             = False      # diversify_per_source covers most cases
expand_to_chapter   = True       # ON, but cap chapter_max_chars
chapter_max_chars   = 8000       # 16 KB was too much; 8 KB tested well
context_window      = 0          # off; expand-to-chapter covers it
reorder_for_attention = False    # tested negative on qwen2.5:7b
crag_filter         = False      # opt-in, costs +1 LLM call

# Generation
num_ctx             = 0          # 0 = auto-size (DO NOT manually set 4096!)
num_batch           = 1024       # tuned for throughput
temperature         = 0.2        # crisp factual answers
max_tokens          = 4096       # plenty of headroom

# Ingest
enable_ocr          = True       # critical for scans/broken cmaps
enable_contextual   = False      # opt-in, very expensive
llm_inspect_pages   = False      # opt-in
llm_correct_garbled = False      # opt-in
chunk_size          = 512        # default
chunk_overlap       = 64         # default
```

---

## 8. Sources

Research informing this doc:

- [Lost in the Middle (Liu et al. 2023, arXiv:2307.03172)](https://arxiv.org/abs/2307.03172)
- [Contextual Retrieval (Anthropic 2024)](https://www.anthropic.com/news/contextual-retrieval)
- [Corrective RAG / CRAG (arXiv:2401.15884)](https://arxiv.org/abs/2401.15884)
- [ChunkRAG: LLM-Chunk Filtering (arXiv:2410.19572)](https://arxiv.org/abs/2410.19572)
- [Late Chunking (arXiv:2409.04701)](https://arxiv.org/abs/2409.04701)
- [Self-RAG via LangGraph (LangChain blog 2024)](https://blog.langchain.com/agentic-rag-with-langgraph/)
- [Hybrid Search & Reranking (Superlinked VectorHub)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- Ollama `/api/show` documentation for `model_info.<arch>.context_length`

---

*Last updated: 2026-05-02. Update this doc whenever a new optimization
ships or test results change.*

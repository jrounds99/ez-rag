# Plan: Self-Tuning Ingestion (Part B)

**Status:** PLANNING — not yet implemented.

> **Part A — Per-file ingestion metadata — SHIPPED in 2026-05.**
> See `src/ez_rag/ingest_meta.py`, `src/ez_rag/ingest_scan.py`,
> the `ez-rag scan` CLI subcommand, and the
> `Per-file metadata sidecars` entry in `CHANGELOG.md`. The historical
> Part A design notes are preserved in git history for reference.

**Scope (this doc, Part B only).** Self-tuning ingestion — profile
system resources, pick batch sizes / parallelism intelligently, and
(opt-in) adjust on the fly during a run.

---

## Why this exists

Two problems showed up during the D&D 5e bench work:

1. **Domain-specific entities slip through the chunker.** The corpus has
   a Way of the Drunken Master subclass, named NPCs like Durnan and
   Vajra Safahr, custom magic items, faction names. Generic chunking +
   generic embedding doesn't *know* these are important, so they sit in
   chunks that score the same as random rule prose, and queries about
   them fail to retrieve cleanly.

2. **Ingest takes 12 hours on a corpus that fits comfortably in VRAM.**
   The current pipeline uses fixed `embed_batch_size=64` and one
   chunker worker. The user's RTX 5090 has 32 GB VRAM and 16 cores;
   neither resource is anywhere near saturated during ingest. We're
   leaving 5×+ throughput on the table.

The plan below addresses both, and keeps a clean separation: **Part A is
about retrieval quality**, **Part B is about ingest speed**. They share
infrastructure but ship independently.

---

## Part A — Per-file ingestion metadata

### A1. Goal

Every ingested file should carry a metadata sidecar. The sidecar is
plain TOML, lives alongside the source document, can be hand-edited,
and is consulted at three points in the RAG pipeline:

- **Ingest time** — entities discovered up front are indexed as
  first-class searchable terms (FTS5), and chunks containing those
  entities get a "high-importance" boost in the dense vector.
- **Retrieval time** — per-file `query_prefix` / `query_suffix` /
  `query_negatives` modify the embedded query when retrieving against
  that file's chunks (or globally, depending on scope).
- **Generation time** — per-file context (e.g. "this file is the
  Player's Handbook; expect rules and tables") goes into the system
  prompt when chunks from this file dominate the retrieval.

### A2. Data model

```toml
# <docs_dir>/<filename>.ezrag-meta.toml
# OR: <workspace>/.ezrag/file_meta/<sha256-prefix>.toml

schema_version = 1
file_path = "Core Rules/D&D Basic Rules (2014).pdf"
file_sha256 = "a3f7c1..."
last_scanned_at = "2026-05-03T12:30:00Z"
last_scanned_by = "qwen2.5:7b"

[summary]
title = "D&D 5e Player's Handbook (Basic Rules subset)"
description = "Core fantasy roleplaying rules: character creation, classes, combat, spells, magic items."
detected_topics = [
  "character creation",
  "ability scores",
  "races",
  "classes",
  "combat",
  "spellcasting",
  "magic items",
  "conditions",
]

[entities]
# Named things the LLM found that are likely to be queried for.
# Stored as a flat list with a "kind" per entry. Kinds are
# free-form so the same schema works on non-D&D corpora too.
characters = []   # for fiction; empty here
npcs = ["Durnan", "Vajra Safahr", "Mordenkainen", "Elminster"]
classes = [
  "Fighter", "Wizard", "Rogue", "Cleric", "Ranger",
  "Way of the Drunken Master",   # Monk subclass
  "Way of the Open Hand",
  "Way of Mercy",
]
items = ["Bag of Holding", "Sword of Sharpness", "Cape of the Mountebank"]
locations = ["Waterdeep", "the Yawning Portal", "Castle Ward"]
factions = ["Lords' Alliance", "Harpers"]
spells = ["Fireball", "Magic Missile", "Counterspell"]
monsters = ["Marilith", "Beholder", "Mind Flayer"]

# Free-form catch-all for things that don't fit the predefined kinds.
custom_terms = ["Ki points", "Sneak Attack", "Bardic Inspiration"]

[modifiers]
# These are the user's hand-curated query modifiers. Inherited from
# the workspace defaults but per-file overrides are allowed.
query_prefix = ""
query_suffix = "Use D&D 5e (2014) terminology."
query_negatives = ["Pathfinder", "3.5e", "5.5e"]   # avoid mixing editions

[scope]
# When does this metadata apply at retrieval time?
# "global"        — modifiers apply to every query
# "file-only"     — modifiers apply ONLY when this file is in the
#                   top-K hits
# "topic-aware"   — modifiers apply when query embeds near any of
#                   the entries in [summary].detected_topics
applies = "topic-aware"

[boost]
# Optional retrieval-time score boosts. Chunks tagged with any of
# these terms in their FTS5 text get a multiplicative bump on the
# rerank score. Keep small (1.05-1.20) — large boosts distort the
# ranking too aggressively.
entity_match_boost = 1.10     # any entity from [entities] matches
priority_term_match_boost = 1.20   # term from [boost.priority_terms] matches
priority_terms = ["Way of the Drunken Master", "Vajra Safahr"]
```

This file is human-readable, version-controlled-friendly, and the
admin can edit any of it directly.

### A3. The LLM discovery scan

A new pre-ingest step that takes a file and produces a draft
`ezrag-meta.toml`. The user reviews/edits/accepts before the actual
ingest runs.

#### Pipeline

```
                ┌──────────────────────────────────────┐
                │  source PDF (or DOCX, etc.)          │
                └────────────────┬─────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  Step 1 — Stratified sampling          │
              │  Pick ~12-20 chunks across the doc:    │
              │  - first 2 chunks (TOC, intro)         │
              │  - 1 chunk every N pages (uniform)     │
              │  - last 2 chunks (index, glossary)     │
              │  - 4-6 chunks chosen by k-means on     │
              │    embeddings (topical diversity)      │
              └────────────────┬───────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  Step 2 — Topic & summary pass         │
              │  Single LLM call:                      │
              │  "Read these excerpts. What is this    │
              │   document? List the topics. Output    │
              │   JSON: {title, description, topics}." │
              └────────────────┬───────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  Step 3 — Entity extraction (batched)  │
              │  Per ~5 chunks, one LLM call:          │
              │  "Extract proper nouns (people, places,│
              │   items, classes, factions, spells,    │
              │   monsters). Reply JSON-only."         │
              │  Dedupe across batches.                │
              └────────────────┬───────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  Step 4 — Sanity / consolidation       │
              │  - Merge near-duplicates (Mordenkainen │
              │    vs Mordenkainen's vs The Mage)      │
              │  - Drop entries that look noisy         │
              │  - Cap each kind at top N by frequency │
              │  - Suggest negative terms (e.g. "5.5e")│
              └────────────────┬───────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  Step 5 — Write draft TOML              │
              │  alongside source as <file>.ezrag-     │
              │  meta.toml.draft                       │
              └────────────────┬───────────────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────────┐
              │  Step 6 — Admin review UI              │
              │  See A4 below.                         │
              └────────────────────────────────────────┘
```

#### Cost / time estimates

For a 320-page PDF:
- Stratified sampling → no LLM, ~1 s
- Topic pass → 1 LLM call (medium prompt) → 5–10 s
- Entity extraction → ~3–5 batched LLM calls → 30–60 s
- Consolidation → no LLM, <1 s

**Total: ~1 minute per typical book** with `qwen2.5:7b`.
For a 25-book corpus: ~25 minutes of scanning before the user hits
"Ingest" — far cheaper than re-ingesting after discovering retrieval
is bad.

#### Why batched / sampled instead of "send the whole book"

A 320-page PDF is ~150 K tokens. Even with 32 K context that's 5+
calls just to read it once. Stratified sampling captures the
*flavor* of the document — topics, named entities, terminology — at
~5% of the token cost with empirically-similar entity recall. This
is the same approach Anthropic uses for their contextual retrieval
sampling.

### A4. Admin review UI

New tab or modal: **"Discovered metadata"**.

```
┌─────────────────────────────────────────────────────────────┐
│  Discovered metadata for: D&D Basic Rules (2014).pdf        │
│  Scanned by qwen2.5:7b · 1 min ago · 18 chunks sampled      │
├─────────────────────────────────────────────────────────────┤
│  TITLE        [D&D 5e Player's Handbook (Basic Rules)]     │
│  DESCRIPTION  [Core fantasy roleplaying rules…………………]      │
│                                                             │
│  TOPICS                                          [+ add]    │
│  • character creation             [×]                       │
│  • combat                         [×]                       │
│  • spellcasting                   [×]                       │
│  …                                                          │
│                                                             │
│  ENTITIES (LLM thinks these are queryable)                  │
│  ┌─ NPCs ────────────────────────────────────────────────┐ │
│  │ Durnan [×]  Vajra Safahr [×]  Mordenkainen [×]  …    │ │
│  │ [+ add]                                              │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌─ Classes ─────────────────────────────────────────────┐ │
│  │ Fighter [×]  Wizard [×]  Way of the Drunken Master[×]│ │
│  │ [+ add]                                              │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌─ Items, Locations, Factions, Spells, Monsters …      │ │
│                                                             │
│  QUERY MODIFIERS                                            │
│   prefix     [                          ]                   │
│   suffix     [Use D&D 5e (2014) terminology.]              │
│   negatives  [Pathfinder] [3.5e] [5.5e]    [+ add]         │
│   scope      ( ) global  (•) topic-aware  ( ) file-only    │
│                                                             │
│  [Re-scan]   [Accept & ingest]   [Save as draft]   [Cancel] │
└─────────────────────────────────────────────────────────────┘
```

- Each entity / topic / negative is removable; new ones can be added.
- "Re-scan" re-runs the LLM with admin-provided hints (e.g. "focus on
  named NPCs, not generic monster types").
- "Accept & ingest" writes the TOML and triggers the actual ingest
  for this file.
- Bulk mode: a "Scan all unindexed files" button kicks off the
  scanning queue for everything in `docs/`. The admin can come back
  and review each file as the scans complete.

### A5. How the metadata feeds into RAG

Three integration points. Each is a small, isolated change.

#### 1. Index-time entity boosting

When chunking a file, every chunk's text is checked against the
file's `[entities]` lists. If the chunk contains an entity:

- **FTS5 side**: prepend a synthetic header line to the indexed
  tokens — `[ENTITIES: Durnan Mordenkainen]` — so BM25 lights up on
  these terms even when the user's query phrases them differently.
- **Vector side**: no change to the embedding (we're not modifying
  source text), but the chunk gets a `priority` field in the index
  that the reranker can read.

#### 2. Retrieval-time query expansion

Reads the `[modifiers]` block from each file whose chunks are in the
top-K candidate pool. For files with `applies = "global"`, modifiers
apply to every query. For `topic-aware`, only when the query embeds
near any topic in `[summary].detected_topics`. For `file-only`,
modifiers apply only when this file is the top-1 source.

The query string sent to the embedder becomes:
```
{prefix} {original query} {suffix}
```
Negative terms get appended as `Avoid: {negatives}`.

#### 3. Generation-time prompt enrichment

When 50%+ of top-K hits come from one file with a known summary, the
system prompt prepends:

```
The user's question primarily concerns content from
"{title}" — {description}.
Use the terminology and proper nouns from that source when
giving your answer.
```

This gives the LLM a frame for ambiguous questions ("how does the
class work" → it knows "class" means D&D class, not Python class).

### A6. New config fields

Adds to `Config`:

```python
# Per-file metadata behavior
file_metadata_enabled: bool = True
file_metadata_default_scope: str = "topic-aware"   # "global" | "topic-aware" | "file-only"

# LLM discovery scan
llm_scan_enabled: bool = False     # opt-in (extra time/calls per file)
llm_scan_model: str = ""           # blank → use cfg.llm_model
llm_scan_sample_chunks: int = 18
llm_scan_topic_pass_max_tokens: int = 800
llm_scan_entity_pass_max_tokens: int = 600
```

---

## Part B — Self-tuning ingestion

### B1. Goal

ez-rag should figure out the best per-system ingest settings on its
own, instead of asking the user to tune `embed_batch_size`,
`parallel_workers`, etc. Two modes:

- **Profile-then-run** (default ON): profile the system at the start
  of each ingest, lock in a good config, run with it.
- **Adaptive on-the-fly** (opt-in): continuously monitor throughput
  during the run and adjust if the system is being under- or
  over-utilized.

### B2. The resource profiler

Runs in the first 30–60 seconds of an ingest. Probes:

1. **GPU & VRAM** — via the `gpu_detect` module already shipped.
   Knows the user's selected GPU and its total/free VRAM.

2. **CPU** — `os.cpu_count()` for cores, `psutil.cpu_percent()` for
   current load. Distinguishes physical cores from SMT threads.

3. **Embedder warm-up & throughput probe** — runs the embedder on
   batch sizes [4, 16, 64, 128, 256] using a fixed sample of the
   user's actual chunks and measures:
   - Embed wall time per batch
   - VRAM headroom remaining
   - Throughput in chunks/sec
   - Throughput in tokens/sec

   Picks the **largest batch size where (a) VRAM stays >2 GB free
   and (b) throughput is still climbing**. Standard Pareto knee.
   See [CocoIndex's adaptive batching](https://cocoindex.io/blogs/batching/)
   reporting ~5× throughput from this single technique.

4. **Disk I/O** — read 100 MB from the docs directory, time it.
   Distinguishes SSD from HDD; throttles parser-pool size on slow
   disks where ~6 parallel parsers thrash.

5. **Ollama embedder vs fastembed** — if Ollama is the configured
   embedder, the profiler runs a quick A/B against fastembed's
   `bge-small-en` (already a dep). If fastembed is faster on this
   machine for the chunk size in use, log a recommendation; never
   silently swap.

Output: a `system_profile` dict written to `<workspace>/.ezrag/
profile.json`. Cached for 24 h; force-refresh on hardware change.

### B3. Profile-driven defaults

Once the profile is in hand, the ingest engine derives:

```python
def derive_settings(profile: SystemProfile, user_cfg: Config) -> Config:
    """Take the user's configured cfg and adjust the *performance-only*
    knobs to what the profile says is sustainable. Quality knobs are
    never touched."""
    cfg = copy(user_cfg)

    # 1. embed_batch_size — set to the throughput knee
    if user_cfg.embed_batch_size_auto:
        cfg.embed_batch_size = profile.recommended_embed_batch

    # 2. parallel_workers — equal to physical cores, capped at 8 to
    #    avoid Ollama HTTP overload
    if user_cfg.parallel_workers_auto:
        cfg.parallel_workers = min(8, profile.physical_cores)

    # 3. on slow disks, parser pool drops to 2
    if profile.disk_kind == "hdd":
        cfg.parallel_workers = min(cfg.parallel_workers, 2)

    return cfg
```

Knobs the profiler will NEVER override:
- `chunk_size`, `chunk_overlap` (changes the index)
- `enable_contextual` (quality flag, not perf)
- `llm_inspect_pages`, `llm_correct_garbled` (quality flags)
- Any retrieval setting

### B4. Adaptive on-the-fly mode

Off by default. When `cfg.adaptive_ingest = True`:

```
                        ┌────────────────────────────┐
                        │  Every 30 seconds OR every │
                        │  1 GB of source processed: │
                        └──────────────┬─────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────┐
                        │  Snapshot:                  │
                        │  - chunks/sec (rolling)     │
                        │  - GPU util %               │
                        │  - VRAM headroom MB         │
                        │  - parser worker stalls     │
                        └──────────────┬─────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  │                    │                    │
                  ▼                    ▼                    ▼
         ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
         │ throughput stable│  │ GPU < 50% util   │  │ VRAM < 1 GB     │
         │ — leave alone    │  │ + VRAM > 4 GB    │  │ free OR OOM     │
         │                  │  │ — bump batch ×2  │  │ — halve batch   │
         └─────────────────┘  └─────────────────┘  └─────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────┐
                        │  Log the change, continue  │
                        │  with new settings.        │
                        │  Emit a UI event so the    │
                        │  workflow chip strip can   │
                        │  show "auto-tuning…"       │
                        └────────────────────────────┘
```

Constraints:
- Adjustments are throttled — at most one change per 30 s.
- Every change is logged with before/after metrics so the user can
  see what happened in the ingest report.
- Hard floor: `embed_batch_size` never drops below 4.
- Hard ceiling: `embed_batch_size` never exceeds 512.

### B5. Recommendations engine (post-run)

After every ingest, regardless of mode, write a short report:

```
=== Ingest performance summary ===
Files:        115        (115 OK · 0 errored · 0 skipped)
Chunks:       24,503
Total bytes:  2.1 GB
Wall time:    14m 22s
Avg throughput: 28.4 chunks/sec  ·  2.5 MB/sec

Resource usage:
  GPU avg:     67%  (peak 91%)
  VRAM peak:   18.4 GB / 32 GB
  CPU avg:     34%  (8 of 16 cores)
  Disk read:   2.1 GB total, peak 180 MB/s

Settings used:
  embed_batch_size: 64 (auto-tuned from 16 at 02:14)
  parallel_workers: 4
  embedder: qwen3-embedding:8b on GPU 0

Recommendations for next time:
  ✓ Your batch size auto-tuned correctly.
  ! Workers were under-used (CPU 34%). Try parallel_workers = 8.
  ! Disk reads peaked at 180 MB/s — well under your SSD's
    ~3 GB/s rated. Bottleneck is downstream of disk.
  ✓ VRAM has 14 GB headroom. Could fit a larger embedder
    (e.g. bge-large-en-v1.5) for better recall.
```

These recommendations are pure suggestions. The user can accept,
ignore, or click "apply for next ingest" to pin a setting.

### B6. New config fields

```python
# Self-tuning ingest
embed_batch_size_auto: bool = True
parallel_workers_auto: bool = True
adaptive_ingest: bool = False         # opt-in on-the-fly mode
profile_cache_hours: int = 24
ingest_recommendations: bool = True   # write the summary report
```

---

## Part C — Integration points

### C1. New modules

```
src/ez_rag/
  ingest_meta.py      # Read/write the *.ezrag-meta.toml files
  ingest_scan.py      # The LLM-driven discovery scan (Part A)
  ingest_profile.py   # The resource profiler (Part B)
  ingest_adaptive.py  # The on-the-fly tuner (Part B)
```

### C2. Touch points in existing code

- `src/ez_rag/parsers.py` — when chunking, query the file's metadata
  for entities and prepend the synthetic FTS5 header line.
- `src/ez_rag/retrieve.py` — query expansion reads per-file modifiers
  from metadata when building the embedder query.
- `src/ez_rag/generate.py` — system prompt enrichment when one file
  dominates top-K.
- `src/ez_rag/ingest.py` — calls `ingest_profile.profile_system()`
  before the run; calls `ingest_adaptive.maybe_adjust()` between
  files; emits the recommendations summary at the end.
- `gui/ez_rag_gui/main.py` — new "Scan & review metadata" button
  on the Files tab; new "Discovered metadata" modal; new "Auto-tune
  ingest performance" toggle in Settings.

### C3. Backwards compatibility

Workspaces that don't have `*.ezrag-meta.toml` files behave exactly
as today. Adding the metadata is purely additive. The LLM discovery
scan is opt-in. The profiler runs automatically but only changes
performance knobs, which are graceful regardless of value.

---

## Part D — Phasing & risks

### D1. Suggested phases

| Phase | Deliverable | Effort | Status |
|---|---|---|---|
| 1 | `ingest_meta.py` (read/write TOML) + per-file modifiers consulted at retrieve/generate time | small | not started |
| 2 | `ingest_scan.py` — LLM discovery + JSON normalization + dedup | medium | not started |
| 3 | GUI "Discovered metadata" review modal | medium | not started |
| 4 | Index-time entity boosting (FTS5 synthetic header) | small | not started |
| 5 | `ingest_profile.py` — resource profiler + recommended settings | medium | not started |
| 6 | Adaptive batch-size auto-tune (B3) | small | not started |
| 7 | Adaptive on-the-fly tuner (B4) — opt-in | medium | not started |
| 8 | Recommendations summary report (B5) | small | not started |
| 9 | Settings UI for the new toggles | small | not started |

### D2. Open design questions

1. **Scope of negatives.** Should `query_negatives` be additive
   across files when multiple files match a query, or last-write-
   wins, or merged with dedup? **Tentative answer:** merged with
   dedup, capped at 5 negatives total to avoid bloating the prompt.

2. **Re-scan triggers.** When a source PDF's SHA changes, is the
   metadata invalidated? **Tentative answer:** yes — but the user
   gets a "metadata is stale, re-scan?" notification, not an
   automatic delete.

3. **LLM scan model choice.** Should the scan use the chat LLM, or
   a dedicated smaller model? **Tentative answer:** new config
   `llm_scan_model` defaulting to `cfg.llm_model`. Users can pick
   a faster model (e.g. `qwen2.5:1.5b`) for scanning if they want.

4. **Adaptive tuning safety.** What if the on-the-fly tuner causes
   an OOM mid-ingest? **Mitigation:** every ingest writes per-file
   commits, so a crash drops only the current file. The tuner also
   caps batch sizes at 75% of measured headroom.

5. **Privacy.** The LLM scan reads source documents. For workspaces
   with sensitive content, the scan needs to be obviously opt-in
   per-file (which it already is, since the user clicks the button).
   No content leaves the local Ollama instance.

### D3. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| LLM hallucinates entities that don't exist in the doc | medium | Step 4 consolidation drops entries that don't appear verbatim in any sampled chunk |
| Per-file metadata files get out of sync with source PDFs | low | SHA tracking + "re-scan?" notification |
| Profiler's batch-size probe takes too long on huge corpora | low | Run probe on a fixed-size synthetic batch, not user data |
| Adaptive tuner oscillates (bumps then drops then bumps) | medium | 30-second throttle + hysteresis: must see same signal twice in a row before adjusting |
| Synthetic FTS5 entity header inflates index size | low | Dedup terms across chunks of same file; cap total at 200 terms/file |

---

## Sources / research

- [CocoIndex — adaptive batching: 5× throughput on data pipelines](https://cocoindex.io/blogs/batching/) — primary reference for B2 / B4
- [Snowflake — Scaling vLLM for Embeddings: 16× throughput](https://medium.com/snowflake/scaling-vllm-for-embeddings-16x-throughput-and-cost-reduction-f2b4d4c8e1bf) — confirms profile-then-tune wins on embeddings
- [Anyscale — Scalable RAG ingest with Ray](https://www.anyscale.com/blog/rag-pipelines-how-to) — distributed ingest pattern (out of scope for ez-rag, but informs B3 / B4 design)
- [vici0549 — sentence-transformers batch size pitfalls](https://medium.com/@vici0549/it-is-crucial-to-properly-set-the-batch-size-when-using-sentence-transformers-for-embedding-models-3d41a3f8b649) — VRAM/throughput inflection points
- [Zilliz — optimal embedding batch size FAQ](https://zilliz.com/ai-faq/what-is-the-optimal-batch-size-for-generating-embeddings) — sanity check on the bucket sizes [4, 16, 64, 128, 256]
- [PingCAP — using LLMs to extract knowledge graph entities/relations](https://www.pingcap.com/article/using-llm-extract-knowledge-graph-entities-and-relationships/) — the exact pattern used in A3 (sampled extraction + JSON consolidation)
- [Robert McDermott — unstructured text → interactive KG via LLMs](https://robert-mcdermott.medium.com/from-unstructured-text-to-interactive-knowledge-graphs-using-llms-dd02a1f71cd6) — informs A3 step ordering
- [Neo4j — knowledge-graph extraction challenges](https://neo4j.com/blog/developer/knowledge-graph-extraction-challenges/) — entity-resolution / dedup considerations for A3 step 4

---

## Glossary (terms used in this plan)

- **Sidecar metadata** — A small file that lives alongside a source
  document, carrying ez-rag-specific fields. Format: TOML.
- **Stratified sampling** — Picking chunks across the document
  intentionally (start, end, evenly-spaced, topic-diverse) instead
  of random or sequential.
- **Knee** — The point on a throughput-vs-batch-size curve where
  adding more batch size stops improving throughput. Standard
  Pareto-frontier inflection.
- **Adaptive batching** — Dynamically resizing batch size during
  a run based on observed throughput / VRAM pressure.
- **Synthetic FTS5 header** — A line we inject into the indexed
  tokens for a chunk that wasn't in the original text, so that
  BM25 can find chunks by entity name.
- **Topic-aware scope** — Per-file modifiers that activate only
  when the user's query embeds close to one of the file's known
  topics.

# Plan: EZ-MODE — guided ingest-to-chat for new users

**Status:** PLANNING — not implemented.

**Scope:** A new top-level mode for users who get lost in the
existing settings sprawl. They pick a folder, pick one of three
preset profiles, the app picks models / settings / GPU placements
for them, ingests, and lets them chat. **Every step explains what
it's doing in plain English.**

The point isn't dumbed-down ez-rag. The point is **teaching**:
EZ-MODE is verbose by design. New users come out of one ingest run
understanding the pipeline.

---

## Why this exists

The existing settings UI has 60+ knobs across Hardware, Ingest,
Retrieval, LLM, Embedder, Agent, Performance, Query Modifiers, and
Multi-GPU. Each is correct individually. Together they're
impenetrable to a first-time user. Three things go wrong:

1. **Decision paralysis.** First-run user opens Settings, sees 8
   cards, doesn't know which to touch first. Most likely outcome:
   close it, ingest with defaults that may not match their hardware.
2. **No mental model.** Even when they pick the right setting,
   they don't know *why*. They learn nothing about RAG.
3. **No model recommendations matched to their box.** The user has
   to know what `qwen2.5:7b` is, that it fits in 6 GB, that
   `qwen3-embedding:8b` exists and is the right embedder pair, etc.
   None of that is obvious.

EZ-MODE solves all three by removing the decisions, doing the work,
and narrating every step.

---

## The user story

```
1. User clicks "EZ-MODE" in the rail (or it auto-engages on a
   workspace with no config + no index).
2. App says "Pick a folder of documents."  → user picks.
3. App says "What kind of run?"
   ┌─────────────────────────────────────────────┐
   │  ⚡  FAST       chat going in 2-3 min         │
   │  ⚖  BALANCED   sweet spot — 10-15 min        │
   │  💎  BEST       high quality — 30-90 min     │
   └─────────────────────────────────────────────┘
4. App detects hardware, picks the right models, explains its
   choices, downloads any missing models, and starts ingest.
5. While ingesting, every stage of the pipeline shows up in
   plain English with a "what / why / how long" line.
6. When done: chat opens. Same verbose narration on the
   retrieval side too.
```

The user doesn't see Settings at all. The ONLY toggle EZ-MODE
shows them is **"Export this chatbot."** That's it. No top_k, no
chunk_size, no temperature, no anything else.

---

## The three profiles

### ⚡ FAST — for trying ez-rag, testing prompts, light corpora

**Goal:** chat as fast as possible. Ingest in minutes. Quality
acceptable for casual use.

| Setting | Value | Why |
|---|---|---|
| LLM | smallest model that fits comfortably | speed + low VRAM |
| Embedder | `nomic-embed-text` | tiny, fast, GPU-resident |
| Chunk size | 384 tokens | smaller chunks ingest faster, retrieval slightly noisier (acceptable in fast mode) |
| Chunk overlap | 32 tokens | half the default — saves time |
| OCR | OFF | scanned PDFs skipped — fast mode is for clean text |
| Garbled detection | ON | always on, free |
| LLM inspect | OFF | adds 1 LLM call/section, kills speed |
| LLM correct | OFF | same |
| Contextual retrieval | OFF | adds 1 LLM call/chunk, kills speed |
| top_k | 4 | fewer chunks → faster generation |
| Hybrid | ON | free quality lift |
| Rerank | OFF | saves 50–200 ms / query |
| HyDE / multi-query | OFF | extra LLM calls |
| Chapter expansion | OFF | smaller, faster context |
| Source diversification | OFF | not needed at top_k=4 |

Hardware-conditional model picks (auto-selected from the
multi-model bench data):

| GPU VRAM | Chat model | Reason |
|---|---|---|
| < 4 GB / no GPU | `phi4-mini` (~2.5 GB) | smallest viable on tiny VRAM |
| 4 – 8 GB | `llama3.2:3b` (~2 GB) | fast 3B with ~86% of best quality |
| 8 – 16 GB | `qwen2.5:7b` (~5 GB) | the practical optimum from our bench |
| 16+ GB | `qwen2.5:7b` | bench shows 14B is *worse*; don't upsize |

### ⚖ BALANCED — recommended default

**Goal:** the right answer most of the time without surprising the
user with an hours-long ingest.

| Setting | Value | Δ from FAST |
|---|---|---|
| OCR | ON | catches scans / broken PDFs |
| LLM correct | OFF | still off — quality vs cost trade isn't worth it |
| Garbled detection + OCR validation | ON | always |
| Chunk size | 512 | proven default |
| Chunk overlap | 64 | proven default |
| top_k | 8 | proven default |
| Rerank | ON | biggest single quality win |
| Source diversification | ON (cap=3) | forces variety |
| Chapter expansion | ON, 4000 char cap | tested sweet spot |
| Auto-list mode | ON | catches "list X" queries |
| Lost-in-middle reorder | OFF | tested negative for typical contexts |
| Auto num_ctx | ON | always — critical |

Same hardware-conditional model picks as FAST. BALANCED differs
mainly in retrieval quality flags (rerank, expansion, list mode),
not chat model size.

### 💎 BEST — high-stakes corpora, you have time

**Goal:** maximum quality. Slow ingest is acceptable. Every
opt-in quality knob is on.

| Setting | Value | Δ from BALANCED |
|---|---|---|
| LLM inspect | ON | catches garbled sections the heuristic missed |
| LLM correct | ON | repairs OCR damage with LLM cleanup |
| Contextual retrieval | ON if VRAM ≥ 16 GB AND corpus < 2 GB | Anthropic-style chunk context. Skip on huge corpora — it's prohibitively slow. |
| top_k | 12 | wider candidate pool |
| Embedder | `qwen3-embedding:8b` if VRAM ≥ 16 GB | bigger embedder, better recall |
| Larger chat model when VRAM allows | `qwen2.5:14b` if 24 GB+, `deepseek-r1:32b` if 32 GB+ | only at the VRAM tier where the bench shows actual gain |

BEST is the only profile that conditionally swaps the chat model.
FAST and BALANCED stick with `qwen2.5:7b` even on a 5090, because
the bench shows 7B is the practical optimum on our test corpus
and we don't want to lie to the user about "bigger is better."

---

## Hardware-aware model selection

Reuses what's already shipped in `gpu_detect.py` + `gpu_recommend.py`
+ the multi-model bench results in
`bench/reports/multimodel-*-judged.json`. EZ-MODE makes its picks
from those (cached as a small lookup baked into the source —
not re-scanned at runtime).

```python
# src/ez_rag/ez_mode.py

@dataclass
class EzPreset:
    profile: str             # "fast" | "balanced" | "best"
    chat_model: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    flags: dict[str, bool]   # rerank, contextual, llm_inspect, …
    estimated_ingest_minutes_per_gb: float
    rationale: str           # human-readable "why we picked this"


def pick_preset(profile: str, gpu: DetectedGpu | None,
                 corpus_bytes: int) -> EzPreset:
    """Single deterministic function from (profile + hardware +
    corpus size) to a fully-specified ingest config + chat model."""
    ...
```

### "Why we picked this" surface

Every choice EZ-MODE makes is shown to the user with a one-line
explanation. Example pre-ingest screen:

```
We're going to:

  📦  Use qwen2.5:7b for chat
      You have 32 GB VRAM. Our benchmark shows 7B is the
      practical optimum on D&D-style corpora — 14B was actually
      *worse* in tests.

  🔍  Use qwen3-embedding:8b for embeddings
      Bigger embedder = better recall. You have the VRAM.

  ✂  Chunk into 512-token slices with 64 overlap
      Standard. Smaller chunks miss context, bigger chunks
      retrieve too much.

  ⚙  Turn on hybrid search + reranking
      Hybrid = BM25 + vectors. Reranking re-scores the top
      candidates. These two combined add the biggest quality
      lift over plain vector search.

  🤖  Turn ON: LLM inspect + LLM correct + contextual retrieval
      You picked BEST. These add LLM calls per section/chunk
      but materially improve quality on noisy or domain-heavy
      corpora.

  ⏱  Estimated ingest: 23 minutes for your 1.2 GB corpus.
      ────────────────────────────────────────
      [Cancel]   [Looks good — start]
```

Every line is the kind of thing a new user wants to know but
doesn't ask. The "Looks good" button is the only thing the
mainline user clicks.

---

## The verbose narration

EZ-MODE owns its own ingest-progress UI, separate from the existing
ingest panel. It's chunkier (bigger fonts, more vertical space)
and explains every stage as it happens.

### Ingest narration example

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1 of 9 — Reading your folder                          │
│                                                             │
│  Looking at every file in C:/dnd-books/.                   │
│  Counting bytes, deciding which can be parsed.             │
│                                                             │
│  Result: 23 files, 1.2 GB total                            │
│           ▸ 21 PDFs · 1 EPUB · 1 DOCX                      │
│                                                             │
│  ⏱  Took 0.3 s.                                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Step 2 of 9 — Parsing each PDF                             │
│                                                             │
│  We extract text from every page. PDFs are tricky:         │
│  some are clean, some have broken fonts where the          │
│  letters look fine on screen but come out as gibberish     │
│  when we read them. We detect that automatically and       │
│  re-extract via OCR on the broken pages.                   │
│                                                             │
│  Currently parsing: D&D Basic Rules.pdf (page 47/318)       │
│  ████████░░░░░░░░░░░░░░░░░░░░░░  15%                       │
│                                                             │
│  So far: 4 garbled pages auto-recovered via OCR ✓          │
│  Bonus: 0 pages dropped (table-of-contents fragments       │
│         that we'd never want to retrieve anyway)           │
│                                                             │
│  ⏱  Started 0:42 ago. ETA ~3 min.                          │
└─────────────────────────────────────────────────────────────┘

… (Step 3 chunking, Step 4 embedding, Step 5 indexing, etc.)
```

Each step has the same shape:
- **What we're doing** (1-2 plain-English sentences)
- **Why** (when not obvious — embedded in the "what")
- **Live progress** (current file / page / chunk)
- **Counters** for interesting events ("4 garbled pages recovered")
- **Time** (started X ago, ETA Y)

The 9 steps mirror the existing pipeline + a few EZ-MODE-only
extras:

| # | Step | What's on screen |
|---|---|---|
| 1 | Reading folder | file count, byte total, extension breakdown |
| 2 | Parsing files | per-file progress, garbled-recovery counter |
| 3 | Detecting chapters | chapter-list summary |
| 4 | Chunking | chunks-so-far, average chunk size |
| 5 | Loading embedder | model load time, VRAM use |
| 6 | Embedding chunks | chunks/sec, ETA, GPU utilization |
| 7 | Indexing | DB-size growth, FTS5 build progress |
| 8 | (BEST only) Contextual retrieval | per-chunk LLM-call rate, ETA |
| 9 | Ready to chat | summary + "ask your first question" |

### Chat narration

When EZ-MODE is active, every chat answer also shows a "what
happened" panel below the answer. Collapsible, but expanded by
default for the first 5 chats:

```
┌─────────────────────────────────────────────────────────┐
│  ⓘ  How we got that answer                              │
├─────────────────────────────────────────────────────────┤
│  1.  Detected this as a "list X" query — switched to    │
│      entity-rich mode.                                  │
│  2.  Hybrid search returned 16 candidate passages       │
│      across 8 files.                                    │
│  3.  Cross-encoder reranker promoted the top 8.         │
│  4.  qwen2.5:7b read 12 KB of context (auto-sized for   │
│      its 32K window) and answered in 3.7 s.             │
│                                                         │
│  ⓘ  Sources: 8 chunks from 5 files (see chips below)   │
└─────────────────────────────────────────────────────────┘
```

This is what teaches the user the pipeline. After 5–10 questions
they can collapse the panel.

---

## What the user CAN do in EZ-MODE

Three buttons, that's it:

1. **Switch profile** — drops down to the same FAST / BALANCED /
   BEST picker. Triggers a "this requires re-ingest" prompt; user
   confirms; new ingest starts.
2. **Re-scan corpus** — re-runs ingest on the same folder
   (e.g. they added new files).
3. **Export chatbot as standalone** — the existing chatbot-export
   feature, surfaced as a primary action in this mode.

That's the entire surface. No top_k slider, no temperature knob.
If the user wants those, they exit EZ-MODE → goes to full Settings.

---

## What the user SEES (always)

A dashboard at the top of every screen in EZ-MODE shows the live
state — never hidden:

```
┌─────────────────────────────────────────────────────────────┐
│  EZ-MODE · BALANCED                       [Switch profile] │
│                                                             │
│  📁 corpus      C:/my-stuff/      1.2 GB · 23 files         │
│  📚 index       Ready · 24,503 chunks · 412 MB              │
│  🤖 chat model  qwen2.5:7b on GPU 0 (RTX 5090)             │
│  🔍 embedder    qwen3-embedding:8b on GPU 0                 │
│  ⏱  last ingest 14m 22s · 2026-05-03                        │
│                                                             │
│  📊 stats                                                   │
│      total questions: 47                                    │
│      avg answer time: 4.2 s                                 │
│      GPU peak VRAM:   18.4 GB / 32 GB (57%)                │
│      cost so far:     $0.00 (all-local)                    │
└─────────────────────────────────────────────────────────────┘
```

This is the "lots of stats and info" the user asked for. Always
visible, always live.

---

## Architecture / new modules

```
src/ez_rag/
  ez_mode.py          # presets + pick_preset() + auto-detect logic
  ez_narrator.py      # converts pipeline events → user-facing text
                       # (plain-English templates per stage)
gui/ez_rag_gui/
  ez_mode_view.py     # the entire EZ-MODE screen (replaces the
                       # tab rail when active)
docs/
  EZ_MODE.md          # user-facing manual; written for first-time
                       # users who haven't read anything else
```

### `ez_mode.py` shape

```python
PROFILES = ["fast", "balanced", "best"]

@dataclass
class EzModeState:
    workspace: Path
    profile: str
    chat_model: str
    embed_model: str
    cfg: Config            # the materialized ingest/chat config
    rationale: list[tuple[str, str]]  # [(field, why) for explanation UI]
    estimated_minutes: int


def auto_select(workspace: Path, profile: str,
                detected_gpus: list[DetectedGpu],
                corpus_bytes: int) -> EzModeState:
    """Single function that takes everything, returns a fully-specified
    EZ-MODE state. The narrator + the actual ingest read from this."""
    ...


def apply_to_workspace(state: EzModeState) -> None:
    """Materialize cfg.toml + ensure required models pulled +
    install routing-table assignments + set EZ-MODE flag in
    workspace.toml."""
    ...
```

### `ez_narrator.py` shape

Just templates + state-driven copy generation. No LLM.

```python
def narrate(stage: str, event: dict) -> str:
    """Take an ingest pipeline event and return user-facing text.
    Stage IDs match the 9-step list above. Events come from the
    existing IngestProgress objects."""
    ...

# Roughly 9 per-stage template functions, each handling a few
# event sub-types (start, progress, done, recovery, error).
```

---

## What's reused / not duplicated

EZ-MODE wires existing infrastructure rather than rebuilding it:

| Existing | EZ-MODE consumes |
|---|---|
| `gpu_detect.py` + `gpu_recommend.py` | hardware → preset |
| `multi_gpu.py` + `daemon_supervisor.py` | model placement + GPU assignment |
| `ingest.py` | the actual ingest engine, unchanged |
| `bench/reports/multimodel-*-judged.json` | model-quality data informing picks |
| `chat_answer()` / `smart_retrieve()` | chat path, unchanged |
| `extract_pdf_window()` | citation-page exports |
| Existing chatbot-export feature | the "export as standalone" button |

EZ-MODE adds NO new pipeline logic. It's a thin orchestration layer
that builds a Config + RoutingTable + narration overlay.

---

## Phasing

| Phase | Deliverable | Effort |
|---|---|---|
| 1 | `ez_mode.py` — profiles + pick_preset deterministic function | small |
| 2 | `ez_narrator.py` — templates for the 9 stages | small |
| 3 | Profile-picker screen (FAST / BALANCED / BEST) | small |
| 4 | Hardware-aware model selector + "why we picked this" screen | medium |
| 5 | Auto-pull missing models with progress bar | small |
| 6 | Verbose ingest narration UI (the 9-step screens) | medium |
| 7 | Verbose chat answer panel ("how we got this answer") | small |
| 8 | Always-on stats dashboard (corpus / model / VRAM / cost) | small |
| 9 | Switch-profile / re-scan / export-only button surface | small |
| 10 | Auto-engage on workspaces with no config + no index | small |
| 11 | `docs/EZ_MODE.md` user manual | small |
| 12 | Tests for pick_preset deterministic outputs | small |

Total: ~10–14 hours of focused work. Phases 1–4 are the foundation
(no UI yet); phases 6–8 are where the user-visible polish lives.

---

## Risks

| Risk | Mitigation |
|---|---|
| User picks BEST on a 200 GB corpus and ingest takes a week | Estimate up-front; if estimate > 6 hours, downgrade to BALANCED + warn. User can override. |
| User's GPU can't run any of our recommended models | `pick_preset` falls back to `phi4-mini` Q4 + retrieval-only no-LLM mode + a clear banner explaining |
| User's local ollama is on a non-default port | EZ-MODE reuses `detect_external` from the multi-GPU plan — picks up whatever's there |
| User updates corpus, forgets to re-scan | Stat dashboard shows "corpus modified after last ingest" badge in red |
| User wants more control after seeing what EZ-MODE picked | `[Switch to full settings]` button on the dashboard — preserves the exact same Config + just lifts the EZ-MODE flag, so the user keeps the picks they were comfortable with |
| Bench data informing picks goes stale | `bench/reports/` files are in-repo + regeneration script ships in `bench/` — re-run after a model release |

---

## Open design questions

1. **Auto-engage trigger.** Should EZ-MODE be the default for new
   workspaces? *Tentative yes* — show a "this is EZ-MODE; switch to
   full settings any time" banner. Veterans can dismiss
   permanently.

2. **What about agentic mode in EZ-MODE?** *Tentative no* —
   agentic adds latency + complexity without huge wins on this
   bench. BEST profile could optionally enable it; FAST and
   BALANCED definitely don't.

3. **Multiple corpora / workspaces in EZ-MODE.** Out of scope for
   v1. EZ-MODE is one workspace at a time. The full-settings UI
   already handles workspace switching.

4. **Do we surface query-time cost in EZ-MODE?** Yes for cloud
   providers (when the cloud-provider plan ships). For all-local,
   show "$0.00" but include token counts so users learn to think
   about prompt size.

---

## Sources / inspiration

- [Notion AI's "Set up in 60 seconds" onboarding flow](https://www.notion.so/help/notion-ai)
  — three-button preset model
- [Obsidian's first-run vault setup](https://help.obsidian.md/Getting+started/Set+up+a+vault)
  — pick-a-folder + show-what-it's-doing pattern
- [GitHub Copilot Chat's explain-mode](https://docs.github.com/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide)
  — verbose narration as teaching
- The user's own bench data
  (`bench/reports/multimodel-20260503-023744-judged.json`) — drives
  the model-tier picks. The plan is built on what we measured, not
  on guesswork.

---

*Last updated: 2026-05-03. Update when phases ship.*

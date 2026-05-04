# Roadmap

Planned work, not yet implemented. Each item has enough detail that a
future session can pick it up cold.

---

## ✅ Shipped: hardware-aware setup

Catalog, detection, tier-based recommendations, runtime pinning,
multi-GPU routing, daemon supervisor, Settings UI, chatbot export
VRAM check, Doctor tab integration, CPU fallback banner. 117
tests across `tests/test_gpu_*.py`, `test_multi_gpu.py`,
`test_daemon_supervisor.py`, `test_auto_placement.py`,
`test_health_check.py`. User-facing writeup in
[`docs/HARDWARE.md`](HARDWARE.md).

---

## ✅ Shipped: per-file metadata + LLM ingest scan

`<file>.ezrag-meta.toml` sidecars with `prefix` / `suffix` /
`negatives` / `topic` / `entities` and a three-tier scope (global /
topic-aware / file-only). Read at retrieval time — no re-ingest
needed to update modifiers. `ez-rag scan` runs an LLM discovery
pass and writes drafts. See `src/ez_rag/ingest_meta.py` and
`src/ez_rag/ingest_scan.py`. (Was Part A of
`PLAN_INGEST_INTELLIGENCE.md`. Part B — self-tuning chunker —
remains pending.)

---

## ✅ Shipped: bench suite + Ohio quality benchmark

System bench (`bench/cli.py probe/quick/full`), Ohio quality bench
(`bench/bench_ohio.py`), HTML reporter with recommendation card
and gold-truth check, public-domain corpus fetcher
(`sample_data/fetch.py`). See [`docs/BENCH.md`](BENCH.md).

---

## RAG metadata: name + description

**Goal.** Every workspace should carry a human-readable name and a
short description. Today the only identifier is the docs folder path.

**Target UX.**
- New "ABOUT" card at the top of the Files tab (or in Settings).
- `Name` text field — defaults to the workspace folder name; user can
  edit.
- `Description` text area — multiline. Empty by default.
- "Auto-describe" button: runs a one-shot LLM pass over a sample of
  the indexed corpus (~2–3 representative chunks per top-level topic
  cluster, or random sample if topic clustering isn't worth the
  complexity) and writes a 2–3 sentence description into the field.
  User can edit afterward.
- Persist as new fields on Config: `rag_name: str`, `rag_description: str`.
- Show the name in the window title and the chat tab header.

**Auto-describe prompt sketch.**
```
You are summarizing a document collection for a search index.
Below are 6 random excerpts. Write 2–3 sentences describing what
this collection is about — topic, scope, intended audience. No
preamble, just the description.
---
{excerpts}
```

**Why now (when this lands).** The chatbot export feature already
exists; without a name + description users can't tell their exports
apart. Also pairs naturally with versioning (below) where each
version carries its own description.

---

## RAG versioning

**Goal.** Every ingest creates a versioned snapshot of the index.
Previous versions stay loadable. Bad / incomplete ingests can be
abandoned without polluting the workspace.

### Layout

Today: `<workspace>/.ezrag/meta.sqlite` (single index file, mutated
in place by every ingest run).

Proposed:
```
<workspace>/.ezrag/
  versions/
    2026-05-01T143022_a3f7c1.sqlite       ← finished
    2026-04-28T091015_b2e8d0.sqlite       ← finished
    2026-05-02T112233_partial.sqlite      ← in-flight or abandoned
  current -> versions/2026-05-01T143022_a3f7c1.sqlite   (symlink/junction or pointer file)
  versions.toml                            ← metadata for each version
```

`versions.toml` — captures everything you'd want to compare across
runs (size, time, models, settings) so a user can answer "which
version was faster / used which embedder / how big is each one"
without launching the app:

```toml
[[version]]
id = "2026-05-01T143022_a3f7c1"
description = "May 1 ingest — added 12 PDFs from new chapters"

# Lifecycle
created = "2026-05-01T14:30:22"
finished = "2026-05-01T15:18:09"
duration_seconds = 2867         # 47 min 47 s
incomplete = false
abandoned = false

# Corpus shape
files = 115
chunks = 24503
total_corpus_bytes = 2147000000   # source docs/ size
index_bytes = 412000000           # the .sqlite + WAL + SHM size

# Models / pipeline used
llm_model = "qwen2.5:7b-instruct"
llm_provider = "ollama"
embedder = "nomic-embed-text"
embedder_provider = "ollama"
reranker = "Xenova/ms-marco-MiniLM-L-6-v2"
parser_version = "1"
chunker_version = "2"

# Hardware
gpu_name = "NVIDIA GeForce RTX 4070 Ti Super"
gpu_vram_gb = 16
runtime = "cuda"

# Knobs that materially affected the index
[version.settings]
chunk_size = 512
chunk_overlap = 64
enable_ocr = true
enable_contextual = false
llm_inspect_pages = false
llm_correct_garbled = false

[[version]]
id = "2026-05-02T112233_partial"
created = "2026-05-02T11:22:33"
finished = ""
duration_seconds = 0
incomplete = true
abandoned = true
files = 47
chunks = 8910
total_corpus_bytes = 800000000
index_bytes = 137000000
llm_model = "qwen2.5:7b-instruct"
embedder = "nomic-embed-text"
description = ""
```

These fields drive the UI:
- **Size column** in the version dropdown / Manage Versions dialog
  shows `index_bytes` formatted as MB/GB.
- **Duration column** shows `duration_seconds` formatted as `47 min 47 s`.
- **Hover / details panel** shows the full settings + models block so
  users can see *why* this version differs from the next one (e.g.
  "this one used contextual retrieval, that's why it's bigger but
  slower to build").
- **Compare button** on two selected versions diffs the settings
  blocks side-by-side — useful for "did turning OCR on actually
  change the index?"

### Behavior

- **Before ingest starts.** If `current` exists, copy/move it under
  `versions/` with a timestamp ID. The new run writes to a fresh
  file. Mark new file `incomplete=true` until ingest finishes
  cleanly, then flip to `false` and update `current` to point to it.
- **Crash / cancel.** The partial file stays in `versions/` flagged
  `incomplete=true`. On next launch the GUI shows a banner: *"An
  ingest from <date> didn't finish. Abandon and delete? Resume?
  Keep as-is?"*. "Resume" only works if the run was paused, not if
  the process died — so v1 just offers Abandon + Keep.
- **Switching versions.** Settings → Index card grows a "Version"
  dropdown listing all completed versions, plus a "browse abandoned"
  toggle. Each row shows: timestamp · size · chunks · duration ·
  embedder. Selecting a version updates the `current` pointer and
  reloads the index. Chat / Files / Doctor all reflect the
  newly-loaded version.
- **Disk pressure.** A "Manage versions" dialog lets the user delete
  individual versions. Show on-disk size per version. Maybe a
  default cap of "keep last 5 completed + 1 abandoned" with a UI
  toggle to override.

### What this DOESN'T do (v1)

- No "merge versions" — adding new docs always means a fresh ingest.
  Diff-incremental ingest is a separate hard problem.
- No automatic compaction or vacuum across versions.
- No cross-version search ("which version had this doc first?").

### Data model touch points

- `index.py` — open-by-path instead of "open the canonical
  meta.sqlite". Add `Index.from_version(workspace, version_id)`.
- `ingest.py` — accept `target_version_id` param; create + finalize
  the version in `versions.toml`; update `current` pointer atomically
  on success.
- `workspace.py` — load `versions.toml`, expose
  `Workspace.list_versions() / current_version() / set_current(id)
  / abandon(id) / delete(id)`.
- `gui/ez_rag_gui/main.py` — version dropdown + abandoned-version
  banner + "Manage versions" dialog.

### Migration

Existing workspaces have a single `meta.sqlite` and no `versions/`
dir. On first launch with the new code: move `meta.sqlite` into
`versions/<bootstrap-id>.sqlite`, write a `versions.toml` with that
single entry marked `description = "Migrated from pre-versioning ez-rag"`,
update `current` to point at it. Idempotent.

---

## When to pull these in

After hardware setup ships and is stable on the user's three test
cards. Versioning is the bigger of the two; metadata is half a day
of work and should ride along with the auto-describe feature so the
LLM hookup is fresh in mind.

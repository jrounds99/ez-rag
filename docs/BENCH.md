# `ez-rag-bench` — cross-platform benchmark suite

Reproducible RAG benchmarks across Linux / macOS / Windows. Captures
system info, samples GPU/CPU power, runs the multi-model RAG sweep,
LLM-judges the answers, and emits a self-contained bundle you can
tar up and send.

> Designed to be runnable from a fresh `git clone` with one command.

---

## Two flavors

ez-rag ships two complementary benches:

1. **System bench** (`bench/cli.py`) — hardware probe + throughput sweep
   across configs. Power sampling, diagnostic bundles. Detailed below.
2. **Quality bench** (`bench/bench_ohio.py`) — multi-model × multi-embedder
   × judged-answer sweep against the public-domain Ohio + Appalachian
   geology corpus. Produces an interactive HTML report with a
   recommendation card, gold-truth check, latency p50/p95, tokens/sec,
   and per-family rollups. See [Ohio quality bench](#ohio-quality-bench)
   below.

## Quick start (system bench)

```bash
git clone https://github.com/<user>/ez-rag.git
cd ez-rag
./bench/run.sh probe          # confirm we can see your hardware
./bench/run.sh quick          # 5-min sanity run
./bench/run.sh search --workspace /path/to/ingested-workspace
```

Windows:

```powershell
.\bench\run.ps1 probe
.\bench\run.ps1 quick
.\bench\run.ps1 search --workspace C:\path\to\workspace
```

What `./bench/run.sh` does:

1. Verify Python ≥ 3.11.
2. Create `bench/.venv/` (only on first run; subsequent runs reuse it).
3. `pip install -r bench/requirements.txt` — three packages:
   `httpx`, `psutil`, `py-cpuinfo`. **Total install: ~30 MB.**
4. Verify Ollama is reachable on `127.0.0.1:11434`.
5. Run the bench.

You don't need to `pip install ez-rag` itself — the bench imports
from `src/ez_rag/` directly. So the same clone you use for development
runs the bench.

---

## Subcommands

| Command | What it does |
|---|---|
| `probe` | Print system info + which models will be tested. Free, instant. |
| `quick` | 5-minute sanity bench (single tiny model, 5 throwaway prompts). Verifies plumbing without committing to a long run. |
| `search` | Multi-model sweep + LLM judge + HTML report. Requires `--workspace` pointing at an already-ingested ez-rag workspace. |
| `full` | Currently aliased to `search`. Future: chains ingest + search. |
| `report` | Re-render the HTML from a saved `*-judged.json`. |

Common flags:

```
--ollama-url URL      Ollama API endpoint (default 127.0.0.1:11434)
--workspace PATH      ez-rag workspace (with .ezrag/ + indexed docs)
--out DIR             output bundle dir (default: bench-results/<host>-<id>-<ts>)
--judge MODEL         LLM judge model (default qwen2.5:7b)
--questions N         cap question count (0 = all)
--limit-models N      cap chat-model count (0 = all)
--skip-pull           don't auto-pull missing models
--no-power            disable power sampling
```

---

## Output bundle

Every run produces a directory under `bench-results/`:

```
bench-results/<hostname>-<systemid8>-<UTC-timestamp>/
  manifest.json        # config + sysinfo + stages + errors
  system_info.json     # standalone system probe
  power_samples.csv    # time-series GPU/CPU wattage
  power_summary.json   # per-segment energy totals
  search/
    multimodel-*.json         # raw answers from the sweep
    multimodel-*-judged.json  # judged scores
  report.html          # self-contained interactive HTML
  diagnostic_bundle.zip      # only if a stage crashed
```

To send it: `tar czf my-bench.tar.gz bench-results/<dir>/`. Hand it
to me and I'll consolidate with the others.

### `manifest.json` shape

```json
{
  "bench_version": "0.1.0",
  "cli_invocation": "...search --workspace ...",
  "config": {...},
  "started_at": 1714770661.4,
  "duration_s": 873.2,
  "system_info": {...},
  "stages": {
    "search.sweep": {
      "status": "ok",
      "started_at": 1714770661.4,
      "duration_s": 412.3,
      "json": "search/multimodel-2026....json"
    },
    "search.judge": {...},
    "search.report": {...}
  },
  "errors": [],
  "power_capabilities": {"nvidia": true, "cpu": false}
}
```

---

## Power sampling

The sampler runs in a background thread, polling every 500 ms (tunable).

| Source | Linux | macOS | Windows |
|---|---|---|---|
| GPU power, util, VRAM | `nvidia-smi` | `nvidia-smi` | `nvidia-smi` |
| CPU power | Intel RAPL via `/sys/class/powercap/` | (skipped — needs root) | (skipped — vendor tooling required) |
| AMD GPU power | `rocm-smi` (planned) | n/a | n/a |
| Per-stage attribution | yes (`set_segment` / `measure()`) | yes | yes |
| Energy totals | trapezoidal integration | same | same |

`power_capabilities` in `manifest.json` tells you which signals
are real:

```json
"power_capabilities": {"nvidia": true, "cpu": false}
```

---

## What gets benchmarked (search mode)

1. **Pre-compute retrieval** once per question using the workspace's
   embedder + HyDE model. Saves dozens of seconds per chat model.
2. **For each chat model** in
   `bench/multi_model_sweep.py`'s `MODELS` list:
   - Pull the model if missing.
   - Generate an answer per question with the saved retrieved chunks.
   - Capture wall-time, model identity, source citations.
3. **Judge every answer** with `qwen2.5:7b` (or `--judge MODEL`)
   using the strict 4-rubric 0–12 scale: addresses / specificity /
   grounded / on_topic.
4. **Render the HTML report**.

The full default sweep is **12 models × 30 questions = 360 answers**,
plus 360 judge calls. On the user's RTX 5090 box this takes about
40–50 minutes. Caps:

- `--questions 5` → cuts to ~5 min/model
- `--limit-models 3` → only the first 3 from the list

---

## When it fails

The bench never silently retries. Any stage that raises an
exception:

1. Writes the full Python trace into `manifest.json` under `errors`.
2. Writes `diagnostic_bundle.zip` containing `manifest.json` +
   `system_info.json` + the partial JSONs from any stages that
   completed + a `diagnostic_README.md`.
3. Prints a one-line guidance message:

```
[!] Bench failed in segment: search.sweep
[!] Diagnostic bundle: ./bench-results/.../diagnostic_bundle.zip
[!] To get help: open the bundle in your AI coding agent
    and ask it to diagnose. Bundle is self-contained.
```

The bundle is **AI-agent-friendly**: `diagnostic_README.md` includes
the exact prompt to paste into Claude Code / Cursor / etc.:

> Read `diagnostic_README.md`, `manifest.json`, and the errors.
> Diagnose why the bench failed and propose a fix.

The bundle does **not** contain corpus content (privacy) — only file
metadata.

---

## Ohio quality bench

A higher-density, opinion-forming benchmark. Sweeps every chat model
across every embedder against a curated public-domain corpus and
produces a recommendation report.

### One-command run

```bash
# 1) Fetch ~76 MB of public-domain government PDFs
python sample_data/fetch.py

# 2) Run the bench (3–6 hours depending on model count)
python -X utf8 bench/bench_ohio.py
```

Or smaller scopes:

```bash
python -X utf8 bench/bench_ohio.py --limit-models 6 --limit-questions 5
python -X utf8 bench/bench_ohio.py --skip-pull --reuse-workspaces
```

### What it measures (per cell)

Each `(embedder × chat_model × question)` cell captures:

- **Judge rubric** (qwen2.5:7b @ T=0): `addresses` / `specificity` /
  `grounded` / `on_topic`, each 0–3, summed to a 0–12 score.
- **Gold-truth check**: rule-based must-contain phrase match across
  8 factual questions. Surfaces hallucination that the rubric judge
  rewards as "well-grounded-looking."
- **Latency**: wall-clock seconds, p50, p95.
- **Token economics**: `prompt_tokens`, `eval_tokens`,
  `tokens_per_sec` (from Ollama `eval_count` / `eval_duration`).
- **Sources**: top-5 file:page citations.

### Output

```
bench/reports/ohio-<UTC-timestamp>/
  manifest.json        # sysinfo + bench args + per-embedder ingest stats
  sweep.json           # raw answers + sources + token counts (per cell)
  judged.json          # sweep + judge scores + gold-truth + judge_err
  report.html          # interactive HTML — open in a browser
```

### The HTML report

- **Recommendation card** at the top: 4 picks (default · highest
  quality · tightest VRAM · highest factual accuracy) + the
  recommended embedder, with rationale.
- **Caveat block** when the rubric judge and the gold-truth check
  disagree (e.g. a model writes confident-sounding answers but
  hallucinates more facts than its size-class peers).
- **Headline table** with per-model: avg /12, gold%, per-rubric
  averages, mean / p50 / p95 latency, tokens/sec, errors.
- **Knee callout**: smallest model within 10% of the best score.
- **Embedder leaderboard**, **family rollup**, **per-category
  averages**.
- **Quality vs params scatter** (with knee circled), **cost-efficiency
  scatter**, **energy-efficiency scatter** (when power data present).
- **Per-rubric grouped bars**, **per-category bars**, **per-model
  factual cards**, **model × question heatmap**, **model × embedder
  heatmap**, **glossary expander**.

A showcase run is checked in at
[`bench/reports/ohio-20260503-211733/`](../bench/reports/ohio-20260503-211733/)
(23 models × 3 embedders × 20 questions = 1,380 cells, 0 errors,
~9.6 MB bundle including `report.html`, `judged.json`, `sweep.json`,
`manifest.json`).

### Methodology notes (read before quoting numbers)

- **`think=false`** for fair apples-to-apples on reasoning models
  (qwen3, deepseek-r1). Disables CoT so token budgets don't
  unbalance the comparison. Reasoning models score lower than they
  would with thinking on — this is intentional.
- **Single judge** (qwen2.5:7b @ T=0). The judge has its own biases
  — notably it rewards concise, citation-heavy answers and
  underweights verbose-but-correct ones. The gold-truth check is
  the cross-validation.
- **Identical retrieval** per `(embedder × question)`. We pre-compute
  retrieval once per embedder and reuse the chunks across every
  chat model. So this measures generation quality only — not the
  chat model's own retrieval logic.
- **One run per cell.** Temperature 0.2, no resampling. Score
  differences smaller than ~0.4/12 are within noise.

### Question set

20 questions spanning `factual` / `comparison` / `exploratory` /
`multi-step`. Defined in
[`bench/ohio_questions.json`](../bench/ohio_questions.json). The
8 most factual questions also have hand-authored gold-snippet sets
(must-contain phrases) in `bench/bench_ohio.py:GOLD_SNIPPETS`.

---

## Cross-system consolidation

After running on multiple machines, you can compare them. Currently
the consolidator is planned (Part E of `PLAN_BENCH_MODE.md`); for
now, the report.html files are self-contained and easy to view
side-by-side.

Future:

```bash
ez-rag-bench consolidate \
  ./bench-results/from-5090/ \
  ./bench-results/from-3090/ \
  ./bench-results/from-mobile/ \
  --out ./consolidated/
```

---

## Tests

The bench has its own unit tests:

```bash
python -X utf8 tests/test_bench_orchestrator.py
```

34 tests cover:

- `sysinfo.gather_system_info` shape + degradation
- `PowerSampler` lifecycle, segment marking, CSV output
- `BenchRun` directory creation, manifest persistence
- `stage()` context manager pre-populates dict so handlers can write into it
- `stage()` captures errors + re-raises
- `diagnostic_bundle.zip` contents

Combined with the multi-GPU suites
(`test_multi_gpu`, `test_daemon_supervisor`, `test_auto_placement`,
`test_health_check`), there are **151 tests** covering the
hardware/bench infrastructure.

---

## File map

| Path | Purpose |
|---|---|
| `bench/cli.py` | Argparse entry point + subcommand dispatch |
| `bench/run.py` | `BenchRun` orchestrator + `stage()` context manager |
| `bench/sysinfo.py` | Cross-platform system probe |
| `bench/power.py` | Background-thread power sampler |
| `bench/multi_model_sweep.py` | (existing) per-model answer sweep |
| `bench/judge_eval.py` | (existing) LLM-as-judge scoring |
| `bench/multi_model_html.py` | (existing) HTML report generator |
| `bench/run.sh` | Linux / macOS bootstrap |
| `bench/run.ps1` | Windows bootstrap |
| `bench/requirements.txt` | Bench-only deps (3 packages, ~30 MB) |
| `bench/test-corpus/` | Bundled test corpus (~5 KB) for fresh-clone runs |
| `bench/test-corpus/held_out_questions.jsonl` | Retrieval-recall ground truth |

---

## What's not yet implemented

These are documented in `docs/PLAN_BENCH_MODE.md` and roadmapped:

- Ingest matrix sweep (Phase 5 of the plan): chunk size × overlap ×
  batch × workers, with held-out recall@k as the quality metric.
- Retrieval-only benchmarks (BM25 / dense / hybrid / hybrid+rerank
  recall@k) without LLM calls.
- Cross-system consolidation tool.
- AMD CPU power on Linux (rapl falls back to TDP estimate currently).
- macOS CPU power via `powermetrics`.
- Windows CPU power via vendor tooling.

The infrastructure is in place; these are additive.

---

*Last updated: 2026-05-03.*

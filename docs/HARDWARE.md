# Hardware: GPUs, daemons, and per-model routing

Reference for the GPU-aware features in ez-rag. Living doc — updated
when phases ship or behavior changes.

> **TL;DR for single-GPU users:** ignore this. ez-rag detects your
> GPU, talks to your existing `ollama serve`, and routes everything
> through it. Nothing new to configure.
>
> **TL;DR for multi-GPU users:** open Settings → HARDWARE / GPU
> ROUTING. Tick the GPUs you want ez-rag to use. ez-rag will spawn
> one Ollama daemon per ticked GPU on a free port (sharing your
> existing model store, no duplicated weights). Pick which models
> live on which GPU in the assignment table. Done.

---

## What ez-rag does on your hardware

### Detection

On every Settings open + every "Re-scan hardware" click, ez-rag
runs a multi-vendor probe:

| Vendor | Probe |
|---|---|
| NVIDIA | `nvidia-smi` (preferred) → `pynvml` fallback |
| AMD | `rocm-smi` |
| Intel | `xpu-smi` (Level Zero) |
| Windows fallback | WMI `Win32_VideoController` |

The probe never raises — failures degrade silently to "no GPU
detected, run in CPU mode."

For each GPU it gets: vendor, model name, VRAM total, free VRAM,
driver version, runtime support tier (full / legacy / unsupported),
plus a cross-reference into `gpu_catalog.py`'s 124-entry static
catalog for stuff the probe doesn't tell us (memory bandwidth,
FP16 TFLOPS, recommended-model tier).

### External-daemon adoption

ez-rag does **not** start its own Ollama on its first run.
Instead, it probes your existing daemon (default
`http://127.0.0.1:11434`, configurable via `OLLAMA_HOST`) and treats
that as the **GPU-0 slot** in the routing table. We never touch its
lifecycle — kill ez-rag and your `ollama serve` keeps running.

### Managed daemons (only if you ask)

When you tick the "Spawn managed daemons for additional GPUs"
checkbox AND tick a GPU other than 0, ez-rag spawns an extra
`ollama serve` process pinned to that GPU. The spawn:

- Uses `CUDA_VISIBLE_DEVICES=<index>` (and `HIP_VISIBLE_DEVICES`
  for AMD)
- Uses an auto-picked free port starting at 11435
- Inherits your `OLLAMA_MODELS` directory — no duplicate model
  blobs on disk
- Writes a PID file at `~/.ezrag/daemons/daemon.<port>.pid` so a
  fresh ez-rag launch can adopt it instead of respawning

ez-rag SIGTERMs every managed daemon at exit (5 s grace, then
SIGKILL). Your external daemon is never touched.

---

## Routing: which GPU runs which model

Three resolution modes, in order of precedence:

### 1. Explicit per-model assignment

In Settings → HARDWARE / GPU ROUTING → Per-model GPU assignment, add
rows like:

| Model | Role | GPU |
|---|---|---|
| `qwen2.5:14b` | chat | GPU 1 — RTX 3090 |
| `qwen2.5:7b` | chat | GPU 0 — RTX 5090 |
| `qwen3-embedding:8b` | embed | GPU 0 — RTX 5090 |
| `nomic-embed-text` | embed | GPU 0 — RTX 5090 |

Persisted at `<workspace>/.ezrag/gpu_routing.toml`. Hand-editable.

### 2. Default GPU (table-level fallback)

Set `default_gpu_index = 0` in the routing TOML. Any model without
an explicit assignment routes to that GPU's daemon.

### 3. AUTO placement (the picker)

When an assignment is `gpu_index = -1` OR no default is set, the
**auto-placer** kicks in:

1. **Sticky:** if the model is already loaded on a daemon (checked
   via `/api/ps`), use that daemon. Avoids the 5–30 s reload cost
   of swapping.
2. **Most free VRAM:** among daemons with headroom, pick the one
   with the most free VRAM. Spreads load across cards.
3. **Fallback:** first registered daemon.

`/api/ps` results are cached for ~4 s, so the picker doesn't add
network round-trips to every chat call.

### 4. Hard fallback

When no daemons are registered at all (empty routing table, single-
GPU user with nothing configured), every call falls through to
`cfg.llm_url`. **Identical to today's pre-multi-GPU behavior.**

---

## OLLAMA_SCHED_SPREAD mode (single-daemon multi-GPU)

If you'd rather have one Ollama daemon spread layers across all
your GPUs (built-in Ollama behavior, no per-model pinning), tick
"Use OLLAMA_SCHED_SPREAD across all GPUs (single-daemon mode)" in
the Hardware card. This:

- Disables managed-daemon spawning (mutually exclusive)
- ez-rag emits a hint that you should set `OLLAMA_SCHED_SPREAD=1`
  on your external daemon (we can't set env vars on a daemon we
  didn't start)
- Per-model GPU assignment is a no-op in this mode

Use this when you want layer-splitting for a single huge model
that can't fit on one card. Use multi-daemon when you want
specific models on specific cards.

---

## Live placement panel

Under "Live placement" in the Hardware card, ez-rag polls
`/api/ps` on every registered daemon every 5 s and shows what's
currently loaded:

```
LIVE PLACEMENT (refreshed every 5s)

GPU 0 · RTX 5090 · http://127.0.0.1:11434           1 model(s) loaded
  qwen2.5:7b              [GPU]  5.2 GB VRAM            29m left

GPU 1 · RTX 3090 · http://127.0.0.1:11435           idle
  no models resident on this daemon
```

`[GPU]` = model is on the GPU. `[CPU]` = Ollama failed to fit it
on the GPU and is running it on system RAM (10–50× slower).

The "Xm left" timer is Ollama's keep-alive expiry — when it hits
zero, the daemon evicts the model. Default 30 minutes; raise
via `OLLAMA_KEEP_ALIVE` on your external daemon, or via the
`keep_alive_s` field in the routing TOML for managed daemons.

---

## Health-check + stranded-assignment recovery

The Hardware card runs a small watchdog that pings every daemon's
`/api/version` every 8 s. When a daemon stops responding for **2
consecutive checks**, ez-rag:

1. Removes it from the routing table
2. Demotes any assignment pinned to that GPU to AUTO so the picker
   reroutes the next call
3. Toasts the user once: *"⚠ Daemon for GPU 1 stopped responding —
   assignments demoted to auto."*
4. Saves the original `gpu_index` per assignment in
   `hw_health_state["stranded"]`

When the daemon comes back (PID alive, URL answers), the next
sweep automatically:

1. Restores the original pin on every stranded assignment
2. Toasts the user: *"✓ Daemon recovered: GPU 1"*

So a transient driver hiccup or eGPU unplug doesn't permanently
break your routing setup — recovery is automatic.

---

## File layout

| Path | What |
|---|---|
| `<workspace>/.ezrag/gpu_routing.toml` | per-workspace routing table. Hand-editable. |
| `~/.ezrag/daemons/daemon.<port>.pid` | PID file per managed daemon. Used to adopt across ez-rag restarts. |
| `~/.ollama/models/` | shared Ollama model store. All daemons share this — no duplicate weights. |

### Example `gpu_routing.toml`

```toml
default_gpu_index = 0
spawn_managed_daemons = true
use_sched_spread = false

[[daemon]]
gpu_index = 0
gpu_name = "RTX 5090"
vram_total_mb = 32768
url = 'http://127.0.0.1:11434'
managed = false
keep_alive_s = 1800
notes = 'external daemon (auto-detected)'

[[daemon]]
gpu_index = 1
gpu_name = "RTX 3090"
vram_total_mb = 24576
url = 'http://127.0.0.1:11435'
managed = true
keep_alive_s = 1800
pid = 9234
notes = 'spawned 2026-05-03T14:22:01 on port 11435'

[[assignment]]
model = 'qwen2.5:7b'
gpu_index = 0
role = 'chat'

[[assignment]]
model = 'qwen2.5:14b'
gpu_index = 1
role = 'chat'

[[assignment]]
model = 'qwen3-embedding:8b'
gpu_index = 0
role = 'embed'
```

---

## Code map

| Module | What |
|---|---|
| `src/ez_rag/gpu_detect.py` | NVIDIA / AMD / Intel / WMI hardware probes |
| `src/ez_rag/gpu_catalog.py` | 124-entry static catalog (consumer + workstation + datacenter) |
| `src/ez_rag/gpu_recommend.py` | tier classification + model recommendations from `bench/reports/` |
| `src/ez_rag/gpu_runtime.py` | `CUDA_VISIBLE_DEVICES` env var construction |
| `src/ez_rag/multi_gpu.py` | `RoutingTable`, `GpuDaemon`, `ModelAssignment`, `resolve_url` resolver, AUTO picker, TOML round-trip |
| `src/ez_rag/daemon_supervisor.py` | `DaemonSupervisor` (spawn / adopt / shutdown), `health_check_once`, `query_loaded_models` |
| `gui/ez_rag_gui/main.py` | Hardware card UI (detected GPUs, daemon list, assignment table, live placement, health watchdog) |

---

## Code call-site map

Every place ez-rag dispatches an Ollama call now goes through
`multi_gpu.resolve_url()`:

| File | Function | Role |
|---|---|---|
| `generate.py` | `_ollama_chat` | `chat` |
| `generate.py` | `_ollama_chat_stream` | `chat` |
| `generate.py` | `model_max_ctx` (via `/api/show`) | `chat` |
| `embed.py` | `make_embedder` | `embed` |
| `ingest.py` | unload-before-ingest | `chat` (the chat model getting unloaded) |

When the routing table is empty, `resolve_url` returns
`cfg.llm_url` — so single-GPU users see no behavior change.

---

## Test coverage (Phase 8)

| File | Tests | What it covers |
|---|---|---|
| `tests/test_multi_gpu.py` | 53 | data model, lookup precedence, TOML round-trip, malformed-input tolerance, atomic save, derive-default-table |
| `tests/test_daemon_supervisor.py` | 32 | external detection, OLLAMA_HOST env, PID-file lifecycle, adopt-from-previous-run, health probes, supervisor.is_alive |
| `tests/test_auto_placement.py` | 13 | sticky placement, free-VRAM placement, cache TTL, cache invalidation on table change, AUTO assignment routing, orphan-assignment fallback |
| `tests/test_health_check.py` | 19 | debounce on transient blips, daemon-down event, assignment stranding, recovery, flap-and-recover |
| **Total** | **117** | |

Run them all:

```bash
cd ez-rag
for f in tests/test_multi_gpu.py tests/test_daemon_supervisor.py \
         tests/test_auto_placement.py tests/test_health_check.py; do
    python -X utf8 "$f"
done
```

---

## Known limitations / out of scope

| Limitation | Status |
|---|---|
| Multi-daemon model layer-splitting (one model spread across 2 GPUs) | Use OLLAMA_SCHED_SPREAD mode, not multi-daemon |
| Live VRAM probe per GPU (currently uses recorded `vram_total_mb` minus loaded model size) | v2 — would require `nvidia-smi` poll on every health check |
| Cloud providers (Ollama Cloud, OpenAI, Anthropic) as fallback chain entries | See `PLAN_CLOUD_PROVIDERS.md` |
| Embedder per-GPU pinning at the request level | Resolved at session start; doesn't change mid-run |
| Pre-flight VRAM check before spawning a daemon | v2 — currently we let Ollama refuse the load if VRAM is tight |
| Cross-machine routing (one daemon on this box, one on the LAN) | Untested — works in theory if `OLLAMA_HOST` points at a remote |

---

## Sources / inspiration

- [Ollama Hardware Support](https://docs.ollama.com/gpu) — `CUDA_VISIBLE_DEVICES`, `OLLAMA_SCHED_SPREAD`, multi-GPU behavior
- [Ollama issue #8430 — per-model GPU pinning](https://github.com/ollama/ollama/issues/8430) — confirms upstream API doesn't support it; multi-daemon is the supported workaround
- [Ollama issue #1813 — single dedicated GPU](https://github.com/ollama/ollama/issues/1813)
- [knightli — Ollama Multi-GPU Notes (April 2026)](https://www.knightli.com/en/2026/04/19/ollama-multiple-gpu-notes/) — VRAM pooling, GPU selection, common misunderstandings
- [Local AI Master — Multi-GPU Ollama setup](https://localaimaster.com/blog/ollama-multi-gpu-setup)
- [`docs/PLAN_MULTI_GPU.md`](PLAN_MULTI_GPU.md) — original design doc

---

*Last updated: 2026-05-03. Update when phases change.*

"""Multi-GPU routing — per-model GPU pinning across multiple Ollama daemons.

The blocker that makes this module exist: Ollama has no per-model GPU
pin in its runtime API. The scheduler picks placement when the daemon
starts, based on whatever `CUDA_VISIBLE_DEVICES` exposed. So if a user
wants "model A on GPU 0, model B on GPU 1," they need TWO daemons —
each with its own CUDA_VISIBLE_DEVICES — and a router that sends each
call to the right one.

This module owns the data model + persistence:

    GpuDaemon         — one running ollama instance pinned to one GPU
    ModelAssignment   — user's stated preference: "this model on GPU N"
    RoutingTable      — the lookup that resolves (model, role) -> URL

Everything is persisted at <workspace>/.ezrag/gpu_routing.toml.

Phase 2 (DaemonSupervisor) will spawn / detect / shut down the
processes referenced by GpuDaemon. Phase 3 (call-site integration)
will replace the cfg.llm_url reads in generate.py / embed.py with
RoutingTable.url_for() lookups. This module's only job for now is
the data layer.

Single-GPU users see zero change after this phase ships — there's no
behavior wired up yet, just the data structures and file format.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


# ============================================================================
# Constants / sentinels
# ============================================================================

# `gpu_index = -1` in an assignment means "let the auto-placer pick at
# runtime." Phase 6 implements the actual picker; until then this is a
# documented config knob the user can already write into the TOML.
GPU_INDEX_AUTO = -1

# Default base port for managed daemons. The user's pre-existing
# `ollama serve` stays on whatever it was on (typically 11434).
# Managed daemons start at 11435 and walk up.
MANAGED_DAEMON_BASE_PORT = 11435


# ============================================================================
# Data model
# ============================================================================

@dataclass
class GpuDaemon:
    """One running Ollama daemon, pinned to one physical GPU.

    `managed=True` means ez-rag spawned this daemon and owns its
    lifecycle (will SIGTERM on exit). `managed=False` means an
    external daemon — typically the user's `ollama serve` on 11434
    that was running before ez-rag started. We never touch external
    daemons, only route to them.
    """
    gpu_index: int                  # the physical GPU index this daemon serves
    gpu_name: str = ""              # human label, e.g. "RTX 5090"
    vram_total_mb: int = 0          # nominal VRAM at the time of registration
    url: str = "http://127.0.0.1:11434"
    pid: Optional[int] = None       # None for external daemons
    managed: bool = False           # True = ez-rag spawned this one
    keep_alive_s: int = 1800        # OLLAMA_KEEP_ALIVE in seconds
    # Free-form notes set by the supervisor (e.g. "spawned 2026-05-03").
    notes: str = ""

    def is_external(self) -> bool:
        return not self.managed


@dataclass
class ModelAssignment:
    """User-stated preference: this model goes on this GPU.

    `gpu_index = GPU_INDEX_AUTO (-1)` defers to the runtime auto-placer.
    `role` is a hint for the UI grouping (`chat` / `embed` / `any`)
    and lets the same model be pinned differently when used as
    embedder vs chat — rare, but the data model supports it.
    """
    model_tag: str                  # exact tag, e.g. "qwen2.5:7b"
    gpu_index: int = GPU_INDEX_AUTO
    role: str = "any"               # "chat" | "embed" | "any"


@dataclass
class RoutingTable:
    """The lookup table resolved on every Ollama call.

    Persisted as <workspace>/.ezrag/gpu_routing.toml.
    """
    daemons: list[GpuDaemon] = field(default_factory=list)
    assignments: list[ModelAssignment] = field(default_factory=list)
    # Default GPU when no assignment matches. -1 means "first available
    # daemon" (typically GPU 0).
    default_gpu_index: int = -1
    # User toggle for the supervisor: when False, ez-rag won't spawn
    # any managed daemons even if multiple GPUs are present. Single-
    # daemon mode (the user's external ollama serve) handles everything.
    spawn_managed_daemons: bool = True
    # When True, ez-rag instead asks the single external daemon to
    # spread layers across all visible GPUs (sets OLLAMA_SCHED_SPREAD=1
    # via env when it spawns its own, or just documents this for the
    # user to set on their external daemon). Mutually exclusive with
    # spawn_managed_daemons in the UI.
    use_sched_spread: bool = False

    # ---- Lookup ----

    def daemon_for_gpu(self, gpu_index: int) -> Optional[GpuDaemon]:
        """Return the daemon that serves `gpu_index`, or None if no
        daemon is registered for that GPU yet."""
        for d in self.daemons:
            if d.gpu_index == gpu_index:
                return d
        return None

    def first_daemon(self) -> Optional[GpuDaemon]:
        """First registered daemon, in insertion order. Used as the
        ultimate fallback when nothing else matches."""
        return self.daemons[0] if self.daemons else None

    def assignment_for(self, model_tag: str,
                       role: str = "any") -> Optional[ModelAssignment]:
        """Find the most-specific matching assignment.

        Match precedence:
          1. exact model + exact role
          2. exact model + role="any"
          3. exact model + any other role (only if exactly one exists)
        Otherwise None.
        """
        exact = [a for a in self.assignments if a.model_tag == model_tag]
        if not exact:
            return None
        # Prefer exact role match
        for a in exact:
            if a.role == role:
                return a
        for a in exact:
            if a.role == "any":
                return a
        # Last resort: if there's a single assignment for this model,
        # use it regardless of role mismatch.
        if len(exact) == 1:
            return exact[0]
        return None

    def url_for(self, model_tag: str, role: str = "any") -> str:
        """Return the daemon URL to dispatch this call to.

        Resolution order:
          1. Explicit ModelAssignment for (model_tag, role) → that GPU's daemon
          2. default_gpu_index (if set) → that GPU's daemon
          3. First registered daemon (insertion order)
          4. Fallback to localhost:11434 (today's hardcoded behavior)

        Step 4 is the safety net — when the routing table is empty
        (single-GPU users who haven't configured anything), we behave
        identically to the pre-multi-GPU code.
        """
        a = self.assignment_for(model_tag, role)
        if a is not None and a.gpu_index != GPU_INDEX_AUTO:
            d = self.daemon_for_gpu(a.gpu_index)
            if d is not None:
                return d.url

        if self.default_gpu_index >= 0:
            d = self.daemon_for_gpu(self.default_gpu_index)
            if d is not None:
                return d.url

        d = self.first_daemon()
        if d is not None:
            return d.url

        return "http://127.0.0.1:11434"

    # ---- Mutators (used by the UI / supervisor) ----

    def upsert_daemon(self, daemon: GpuDaemon) -> None:
        for i, d in enumerate(self.daemons):
            if d.gpu_index == daemon.gpu_index:
                self.daemons[i] = daemon
                return
        self.daemons.append(daemon)

    def remove_daemon(self, gpu_index: int) -> None:
        self.daemons = [d for d in self.daemons
                         if d.gpu_index != gpu_index]

    def upsert_assignment(self, model_tag: str, gpu_index: int,
                           role: str = "any") -> None:
        for a in self.assignments:
            if a.model_tag == model_tag and a.role == role:
                a.gpu_index = gpu_index
                return
        self.assignments.append(ModelAssignment(
            model_tag=model_tag, gpu_index=gpu_index, role=role,
        ))

    def remove_assignment(self, model_tag: str,
                           role: str = "any") -> None:
        self.assignments = [
            a for a in self.assignments
            if not (a.model_tag == model_tag and a.role == role)
        ]


# ============================================================================
# TOML round-trip
# ============================================================================

# Match a fragment that needs escaping when written as a TOML basic string.
_BASIC_NEEDS_ESCAPE = re.compile(r'[\x00-\x1f"\\]')


def _toml_str(value: str) -> str:
    """Render a string for the TOML output.

    Prefers single-quoted *literal* strings (Windows paths and URLs
    with backslashes round-trip cleanly). Falls back to double-quoted
    when the value contains a literal single quote.
    """
    if "'" in value:
        # Escape only what TOML basic-string syntax requires: \\, \"
        # plus control chars. We keep this intentionally minimal so
        # human-edited files stay readable.
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return f"'{value}'"


def _render_daemon(d: GpuDaemon) -> str:
    lines = [
        "[[daemon]]",
        f"gpu_index = {d.gpu_index}",
        f"gpu_name = {_toml_str(d.gpu_name)}",
        f"vram_total_mb = {int(d.vram_total_mb)}",
        f"url = {_toml_str(d.url)}",
        f"managed = {'true' if d.managed else 'false'}",
        f"keep_alive_s = {int(d.keep_alive_s)}",
    ]
    if d.pid is not None:
        lines.append(f"pid = {int(d.pid)}")
    if d.notes:
        lines.append(f"notes = {_toml_str(d.notes)}")
    return "\n".join(lines)


def _render_assignment(a: ModelAssignment) -> str:
    return (
        "[[assignment]]\n"
        f"model = {_toml_str(a.model_tag)}\n"
        f"gpu_index = {a.gpu_index}\n"
        f"role = {_toml_str(a.role)}"
    )


def render_toml(table: RoutingTable) -> str:
    """Emit a routing table as TOML text. Hand-editable."""
    parts: list[str] = [
        "# ez-rag GPU routing — per-model GPU pinning across daemons.",
        "# This file is hand-editable. ez-rag re-reads on settings save.",
        "",
        f"default_gpu_index = {table.default_gpu_index}",
        f"spawn_managed_daemons = {'true' if table.spawn_managed_daemons else 'false'}",
        f"use_sched_spread = {'true' if table.use_sched_spread else 'false'}",
        "",
    ]
    for d in table.daemons:
        parts.append(_render_daemon(d))
        parts.append("")
    for a in table.assignments:
        parts.append(_render_assignment(a))
        parts.append("")
    # Trim trailing blank lines, end with single newline
    text = "\n".join(parts).rstrip() + "\n"
    return text


def parse_toml(text: str) -> RoutingTable:
    """Parse routing-table TOML. Tolerant of missing fields (uses
    sensible defaults so partial / hand-rolled files work)."""
    if not text or not text.strip():
        return RoutingTable()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # Don't lose user data on a corrupt file — just return empty
        # and let the caller decide whether to overwrite.
        return RoutingTable()

    # Tolerant int coercion — a hand-edited file with `default_gpu_index =
    # "junk"` shouldn't crash ez-rag's startup. Fall back to the safe
    # default and continue parsing the rest.
    def _safe_int(v, default: int) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    table = RoutingTable(
        default_gpu_index=_safe_int(data.get("default_gpu_index", -1), -1),
        spawn_managed_daemons=bool(data.get("spawn_managed_daemons", True)),
        use_sched_spread=bool(data.get("use_sched_spread", False)),
    )

    for d in data.get("daemon", []) or []:
        if not isinstance(d, dict):
            continue
        try:
            gpu_index = int(d.get("gpu_index", 0))
        except (TypeError, ValueError):
            continue
        table.daemons.append(GpuDaemon(
            gpu_index=gpu_index,
            gpu_name=str(d.get("gpu_name", "") or ""),
            vram_total_mb=int(d.get("vram_total_mb", 0) or 0),
            url=str(d.get("url", "http://127.0.0.1:11434")),
            pid=(int(d["pid"]) if d.get("pid") is not None else None),
            managed=bool(d.get("managed", False)),
            keep_alive_s=int(d.get("keep_alive_s", 1800) or 1800),
            notes=str(d.get("notes", "") or ""),
        ))

    for a in data.get("assignment", []) or []:
        if not isinstance(a, dict):
            continue
        model = str(a.get("model", "") or "")
        if not model:
            continue
        try:
            gpu_idx = int(a.get("gpu_index", GPU_INDEX_AUTO))
        except (TypeError, ValueError):
            gpu_idx = GPU_INDEX_AUTO
        table.assignments.append(ModelAssignment(
            model_tag=model,
            gpu_index=gpu_idx,
            role=str(a.get("role", "any") or "any"),
        ))

    return table


# ============================================================================
# Workspace integration
# ============================================================================

def routing_path(workspace_root: Path) -> Path:
    """Where the routing TOML lives within a workspace."""
    return Path(workspace_root) / ".ezrag" / "gpu_routing.toml"


def load_routing_table(workspace_root: Path) -> RoutingTable:
    """Load the routing table for a workspace. Returns an empty
    (single-default-daemon-implied) table if the file doesn't exist."""
    path = routing_path(workspace_root)
    if not path.is_file():
        return RoutingTable()
    try:
        return parse_toml(path.read_text(encoding="utf-8"))
    except OSError:
        return RoutingTable()


def save_routing_table(workspace_root: Path, table: RoutingTable) -> None:
    """Write the routing table to disk. Creates the .ezrag dir if needed."""
    path = routing_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically — temp file + rename — so a crash mid-write
    # doesn't leave an empty/half-baked TOML behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_toml(table), encoding="utf-8")
    os.replace(tmp, path)


# ============================================================================
# Helpers consumed by Phase 2 (supervisor)
# ============================================================================

def find_free_port(start: int = MANAGED_DAEMON_BASE_PORT,
                    *, host: str = "127.0.0.1",
                    max_tries: int = 50) -> int:
    """Find a free TCP port to bind a managed daemon to.

    Walks up from `start`. Used by the supervisor in Phase 2 — kept
    here in the data-model module because the chosen port lands on
    a `GpuDaemon.url` and we want the picker logic close to the
    record it populates.
    """
    import socket
    for offset in range(max_tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                # Bind succeeded — port is free. Close the socket so
                # the supervisor can reuse it for ollama serve.
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in range {start}-{start + max_tries}"
    )


def url_for_port(port: int, *, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}"


# ============================================================================
# Active-table singleton + resolver
# ============================================================================
# Phase 3 wires every Ollama call through resolve_url(). The resolver
# consults the *active* routing table, which the GUI populates after
# the user opens a workspace. When no table is active (CLI usage,
# tests, fresh install), every call falls through to `cfg.llm_url` —
# i.e. today's single-daemon behavior.

_ACTIVE_TABLE: Optional["RoutingTable"] = None


def get_active_table() -> Optional["RoutingTable"]:
    return _ACTIVE_TABLE


# ----- Phase 6: auto-placement picker --------------------------------
# When an assignment is GPU_INDEX_AUTO (or no explicit assignment
# exists) AND there are 2+ daemons, the picker decides which daemon to
# route the call to using a small live-state probe.
#
# Picker rules, in order:
#   1. Sticky: if any daemon already has this model resident, use it.
#      Avoids the 5-30 s reload cost of swapping daemons.
#   2. Most free VRAM: among daemons that have headroom, pick the one
#      with the most free RAM. Spreads load across cards.
#   3. Default: first daemon in the table.
#
# /api/ps probes are CACHED for AUTO_PROBE_TTL_S so we don't hit
# every daemon on every chat call. Cache is invalidated when the
# table mutates (set_active_table clears it).

import time as _time

AUTO_PROBE_TTL_S = 4.0   # cache /api/ps for ~4 s; live panel polls at 5 s
_PROBE_CACHE: dict[str, tuple[float, list]] = {}


def _invalidate_auto_cache() -> None:
    _PROBE_CACHE.clear()


def set_active_table(table: Optional["RoutingTable"]) -> None:
    """Install (or clear) the routing table the resolver consults.
    Called by the GUI after workspace open. Also invalidates the
    auto-pick probe cache so the next dispatch sees fresh state."""
    global _ACTIVE_TABLE
    _ACTIVE_TABLE = table
    _invalidate_auto_cache()


def _probe_loaded(url: str) -> list:
    """Return cached /api/ps results for a daemon URL. Lazy: only
    actually probes when the cache entry is older than AUTO_PROBE_TTL_S
    or missing.

    Imports daemon_supervisor lazily so this module doesn't pull
    httpx into the pure-data unit tests.
    """
    now = _time.monotonic()
    cached = _PROBE_CACHE.get(url)
    if cached is not None and (now - cached[0]) < AUTO_PROBE_TTL_S:
        return cached[1]
    try:
        from .daemon_supervisor import query_loaded_models
        models = query_loaded_models(url, timeout=1.5)
    except Exception:
        models = []
    _PROBE_CACHE[url] = (now, models)
    return models


def _used_vram_mb(loaded: list) -> int:
    """Sum size_vram_bytes across the LoadedModel records, in MB."""
    return sum(int(getattr(m, "size_vram_bytes", 0) or 0)
                 for m in loaded) // (1024 * 1024)


def auto_pick_url(table: "RoutingTable", model_tag: str) -> Optional[str]:
    """Pick the best daemon URL for a model under AUTO placement.

    Returns None if no daemon at all is registered. Caller falls
    back to its own default in that case.
    """
    if not table.daemons:
        return None

    # 1. STICKY — model already loaded somewhere?
    sticky_target: Optional[GpuDaemon] = None
    for d in table.daemons:
        try:
            loaded = _probe_loaded(d.url)
        except Exception:
            continue
        for m in loaded:
            name = getattr(m, "name", "") or ""
            # Ollama's /api/ps reports "qwen2.5:7b" — exact match
            if name == model_tag:
                sticky_target = d
                break
        if sticky_target is not None:
            break
    if sticky_target is not None:
        return sticky_target.url

    # 2. MOST FREE VRAM — pick the daemon with the most headroom.
    #    Per-daemon VRAM is `gpu_total - currently_resident_vram`.
    #    We use the daemon's recorded vram_total_mb (set at registration)
    #    as the ceiling, since we can't query the GPU directly from
    #    here without pulling in gpu_detect.
    best: Optional[GpuDaemon] = None
    best_free = -1
    for d in table.daemons:
        if d.vram_total_mb <= 0:
            # Unknown total — can't reason about free space; skip
            # for the comparison but keep as a fallback candidate.
            continue
        try:
            loaded = _probe_loaded(d.url)
        except Exception:
            loaded = []
        free_mb = d.vram_total_mb - _used_vram_mb(loaded)
        if free_mb > best_free:
            best = d
            best_free = free_mb
    if best is not None:
        return best.url

    # 3. FALLBACK — first registered daemon.
    return table.first_daemon().url if table.first_daemon() else None


def resolve_url(cfg, model_tag: str, role: str = "any") -> str:
    """Return the daemon URL to dispatch a call for (model_tag, role).

    Behavior matrix:
      - No active table     → cfg.llm_url (today's behavior)
      - Active but empty    → cfg.llm_url (no daemons registered yet)
      - Explicit assignment → that GPU's daemon (Phase 1+3)
      - AUTO assignment     → auto_pick_url (Phase 6: sticky → free-VRAM
                              → first daemon)
      - No assignment + default_gpu_index set → that GPU's daemon
      - No assignment + no default → auto_pick_url (degrades to first)

    The result is the URL ALONE — no path. Callers append /api/chat,
    /api/show, etc. as before.
    """
    fallback = (getattr(cfg, "llm_url", "http://127.0.0.1:11434")
                or "http://127.0.0.1:11434")
    table = _ACTIVE_TABLE
    if table is None or not table.daemons:
        return fallback

    # 1. Explicit assignment?
    a = table.assignment_for(model_tag, role)
    if a is not None:
        if a.gpu_index == GPU_INDEX_AUTO:
            picked = auto_pick_url(table, model_tag)
            return picked or fallback
        d = table.daemon_for_gpu(a.gpu_index)
        if d is not None:
            return d.url
        # Assigned to a GPU that doesn't have a daemon registered —
        # fall through to AUTO instead of returning the wrong daemon.
        picked = auto_pick_url(table, model_tag)
        return picked or fallback

    # 2. No explicit assignment — use the table-level default if set,
    #    otherwise fall through to AUTO so multi-daemon setups don't
    #    pile every unassigned model onto GPU 0 unnecessarily.
    if table.default_gpu_index >= 0:
        d = table.daemon_for_gpu(table.default_gpu_index)
        if d is not None:
            return d.url

    picked = auto_pick_url(table, model_tag)
    return picked or fallback


def derive_default_table(detected_gpus: list, *,
                          external_url: str = "http://127.0.0.1:11434"
                          ) -> RoutingTable:
    """Build a starter routing table from the user's detected hardware.

    Used on first run — gives the user a reasonable default that
    works without them touching the TOML:
      - Daemon entry for GPU 0 pointed at their external ollama
      - default_gpu_index = 0
      - No model assignments (everything routes to GPU 0 by default)
      - spawn_managed_daemons = True so phase 2 can register additional
        daemons for the other GPUs when the user enables them

    `detected_gpus` is a list of `gpu_detect.DetectedGpu` objects.
    Imported here lazily so this module doesn't pull the gpu_detect
    chain into pure-data-model unit tests.
    """
    table = RoutingTable(
        default_gpu_index=0 if detected_gpus else -1,
        spawn_managed_daemons=True,
        use_sched_spread=False,
    )
    if detected_gpus:
        # GPU 0 = the external user daemon (we'll detect it for real
        # in Phase 2; this just registers the slot).
        g0 = detected_gpus[0]
        table.daemons.append(GpuDaemon(
            gpu_index=getattr(g0, "index", 0),
            gpu_name=getattr(g0, "name", ""),
            vram_total_mb=int(getattr(g0, "vram_total_mb", 0) or 0),
            url=external_url,
            pid=None,
            managed=False,
            notes="external daemon (auto-detected default slot)",
        ))
    return table

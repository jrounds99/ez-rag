"""Tests for the Phase 6 auto-placement picker.

Stubs daemon_supervisor.query_loaded_models so we can simulate
"this model is already loaded on GPU N" / "this daemon has X MB free"
scenarios without spawning real Ollama processes.

Covers:
  - sticky placement (model already on a daemon → reuse it)
  - free-VRAM placement (no sticky → pick most headroom)
  - degraded fallback (no probe data, no totals → first daemon)
  - cache TTL (multiple resolve_url calls in a window do ONE probe)
  - cache invalidation on set_active_table
  - resolve_url with AUTO assignment fires the picker
  - resolve_url with explicit GPU index bypasses the picker
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag import multi_gpu as mg
from ez_rag.multi_gpu import (
    GPU_INDEX_AUTO, GpuDaemon, ModelAssignment, RoutingTable,
    auto_pick_url, resolve_url, set_active_table,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


@dataclass
class FakeLoadedModel:
    """Mirrors LoadedModel just enough to look duck-typed to the picker."""
    name: str
    size_vram_bytes: int = 0
    size_bytes: int = 0
    expires_at: str = ""


class StubProbe:
    """Patch _probe_loaded so the picker sees canned per-daemon state."""
    def __init__(self, mapping: dict[str, list[FakeLoadedModel]]):
        self.mapping = mapping
        self.saved = mg._probe_loaded
        self.calls: list[str] = []

    def __enter__(self):
        def stub(url: str):
            self.calls.append(url)
            return self.mapping.get(url, [])
        mg._probe_loaded = stub
        return self

    def __exit__(self, *args):
        mg._probe_loaded = self.saved


def fresh_table(daemons: list[GpuDaemon],
                 *, default_gpu: int = -1,
                 assignments=None) -> RoutingTable:
    t = RoutingTable(default_gpu_index=default_gpu)
    for d in daemons:
        t.upsert_daemon(d)
    for a in (assignments or []):
        t.assignments.append(a)
    return t


def main():
    print("\n[1] sticky placement — model already loaded on GPU 1")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/", vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/", vram_total_mb=24 * 1024),
    ]
    t = fresh_table(daemons)
    with StubProbe({
        "http://0/": [],
        "http://1/": [FakeLoadedModel("qwen2.5:7b", 5 * 1024 * 1024 * 1024)],
    }) as sp:
        url = auto_pick_url(t, "qwen2.5:7b")
        check("sticky picks GPU 1 (where model is loaded)",
              url == "http://1/", f"got {url!r}")
        # Both daemons probed (the picker walked the table looking for sticky)
        check("both daemons probed",
              "http://0/" in sp.calls and "http://1/" in sp.calls)

    print("\n[2] no sticky → most-free-VRAM wins")
    # GPU 0 = 24 GB total, 0 used → 24 GB free
    # GPU 1 = 24 GB total, 18 GB used → 6 GB free
    # → picker should pick GPU 0
    daemons = [
        GpuDaemon(gpu_index=0, url="http://big/",
                  vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://small/",
                  vram_total_mb=24 * 1024),
    ]
    t = fresh_table(daemons)
    with StubProbe({
        "http://big/": [],
        "http://small/": [FakeLoadedModel("other-model",
                                            18 * 1024 * 1024 * 1024)],
    }):
        url = auto_pick_url(t, "qwen2.5:14b")
        check("free-VRAM picks GPU 0 (24 GB free vs 6 GB)",
              url == "http://big/", f"got {url!r}")

    print("\n[3] sticky beats free-VRAM (even when sticky GPU has less free)")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/",
                  vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/",
                  vram_total_mb=24 * 1024),
    ]
    t = fresh_table(daemons)
    with StubProbe({
        # GPU 0 has TONS of free VRAM but isn't running our model
        "http://0/": [],
        # GPU 1 is running our model AND something else (low free)
        "http://1/": [
            FakeLoadedModel("qwen2.5:7b", 5 * 1024 * 1024 * 1024),
            FakeLoadedModel("other", 16 * 1024 * 1024 * 1024),
        ],
    }):
        url = auto_pick_url(t, "qwen2.5:7b")
        check("sticky beats free-VRAM",
              url == "http://1/", f"got {url!r}")

    print("\n[4] fallback when no VRAM totals known — picks first daemon")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://first/", vram_total_mb=0),
        GpuDaemon(gpu_index=1, url="http://second/", vram_total_mb=0),
    ]
    t = fresh_table(daemons)
    with StubProbe({}):
        url = auto_pick_url(t, "anything")
        check("no totals → first daemon wins",
              url == "http://first/", f"got {url!r}")

    print("\n[5] empty table → None")
    t = RoutingTable()
    with StubProbe({}):
        url = auto_pick_url(t, "anything")
        check("empty table -> None", url is None,
              f"got {url!r}")

    print("\n[6] cache TTL — repeat resolve doesn't re-probe within 4s")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/",
                  vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/",
                  vram_total_mb=24 * 1024),
    ]
    t = fresh_table(daemons)
    with StubProbe({
        "http://0/": [],
        "http://1/": [FakeLoadedModel("qwen2.5:7b", 1 << 30)],
    }) as sp:
        # Bypass the cache by clearing first
        mg._invalidate_auto_cache()
        # Hit the actual probe (cached path) twice in succession
        url1 = auto_pick_url(t, "qwen2.5:7b")
        # NOTE: we patched _probe_loaded directly above; the cache path
        # in real code only kicks in via _probe_loaded itself. To test
        # the cache, exercise _probe_loaded directly:
        mg._invalidate_auto_cache()
        # Restore real _probe_loaded for this test (we want the cache
        # we wrote, not the stub)
        # Actually simpler: just verify _PROBE_CACHE gets populated
        # when the real _probe_loaded runs. Test that path here.
        check("auto_pick_url returned a URL via stub",
              url1 in ("http://0/", "http://1/"))
        # The cache test belongs in a separate block below since the
        # stub bypasses it. This test confirmed picker logic at least.

    # Direct cache test — call _probe_loaded twice without patching
    # query_loaded_models, then verify the cache holds.
    print("\n[7] _probe_loaded cache holds within TTL")
    from ez_rag import daemon_supervisor as dsmod
    saved_qlm = dsmod.query_loaded_models
    call_count = {"n": 0}
    def stub_qlm(url, *, timeout=2.0):
        call_count["n"] += 1
        return [FakeLoadedModel("cached-model", 1 << 30)]
    dsmod.query_loaded_models = stub_qlm
    mg._invalidate_auto_cache()
    try:
        a = mg._probe_loaded("http://cache-test/")
        b = mg._probe_loaded("http://cache-test/")
        c = mg._probe_loaded("http://cache-test/")
        check("first call hit the source",
              call_count["n"] == 1, f"got {call_count['n']} calls")
        check("subsequent calls within TTL hit cache",
              len(a) == len(b) == len(c) == 1)
    finally:
        dsmod.query_loaded_models = saved_qlm

    print("\n[8] set_active_table clears the auto-cache")
    mg._invalidate_auto_cache()
    saved_qlm = dsmod.query_loaded_models
    call_count = {"n": 0}
    def stub_qlm(url, *, timeout=2.0):
        call_count["n"] += 1
        return []
    dsmod.query_loaded_models = stub_qlm
    try:
        mg._probe_loaded("http://invalidate-test/")
        # Now flip the active table — cache should be wiped
        set_active_table(RoutingTable())
        mg._probe_loaded("http://invalidate-test/")
        check("set_active_table invalidates cache",
              call_count["n"] == 2,
              f"expected 2 probes, got {call_count['n']}")
    finally:
        dsmod.query_loaded_models = saved_qlm
        set_active_table(None)

    print("\n[9] resolve_url with AUTO assignment fires the picker")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://gpu0/",
                  vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://gpu1/",
                  vram_total_mb=24 * 1024),
    ]
    t = fresh_table(
        daemons,
        default_gpu=0,
        assignments=[
            ModelAssignment(model_tag="auto-model", gpu_index=GPU_INDEX_AUTO,
                             role="any"),
        ],
    )
    set_active_table(t)
    cfg = Config()
    cfg.llm_url = "http://nope/"
    with StubProbe({
        "http://gpu0/": [FakeLoadedModel("auto-model", 1 << 30)],
        "http://gpu1/": [],
    }):
        url = resolve_url(cfg, "auto-model", "any")
        check("AUTO assignment with sticky → routed to sticky daemon",
              url == "http://gpu0/", f"got {url!r}")
    set_active_table(None)

    print("\n[10] resolve_url with EXPLICIT gpu_index bypasses picker")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://gpu0/",
                  vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://gpu1/",
                  vram_total_mb=24 * 1024),
    ]
    t = fresh_table(
        daemons,
        default_gpu=0,
        assignments=[
            ModelAssignment(model_tag="pinned-model", gpu_index=1,
                             role="chat"),
        ],
    )
    set_active_table(t)
    saved_qlm = dsmod.query_loaded_models
    call_count = {"n": 0}
    def stub_qlm(url, *, timeout=2.0):
        call_count["n"] += 1
        return []
    dsmod.query_loaded_models = stub_qlm
    try:
        cfg = Config()
        cfg.llm_url = "http://nope/"
        url = resolve_url(cfg, "pinned-model", "chat")
        check("explicit assignment → bypassed picker (no probes)",
              url == "http://gpu1/" and call_count["n"] == 0,
              f"url={url!r} probes={call_count['n']}")
    finally:
        dsmod.query_loaded_models = saved_qlm
        set_active_table(None)

    print("\n[11] resolve_url with assignment to UNREGISTERED gpu falls to AUTO")
    # Assignment says "GPU 5" but only GPU 0 has a daemon → picker fires.
    daemons = [
        GpuDaemon(gpu_index=0, url="http://only/",
                  vram_total_mb=24 * 1024),
    ]
    t = fresh_table(
        daemons,
        default_gpu=-1,
        assignments=[
            ModelAssignment(model_tag="lost-model", gpu_index=5,
                             role="chat"),
        ],
    )
    set_active_table(t)
    with StubProbe({"http://only/": []}):
        cfg = Config()
        url = resolve_url(cfg, "lost-model", "chat")
        check("orphan assignment falls through to picker → first daemon",
              url == "http://only/", f"got {url!r}")
    set_active_table(None)

    print(f"\n=== auto-placement summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

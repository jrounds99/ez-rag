"""Tests for Phase 7 — health-check sweep + stranded-assignment recovery.

Stubs daemon_supervisor._probe_url so we can simulate "this daemon is up"
or "this daemon stopped answering" without real subprocesses.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import daemon_supervisor as ds
from ez_rag.daemon_supervisor import (
    ExternalDetection, HEALTH_FAIL_THRESHOLD, HealthEvent,
    health_check_once,
)
from ez_rag.multi_gpu import (
    GPU_INDEX_AUTO, GpuDaemon, ModelAssignment, RoutingTable,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


class StubProbe:
    """Patch _probe_url so we can flip daemons up/down per test."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.saved = ds._probe_url

    def __enter__(self):
        def stub(url, *, timeout=2.0):
            return self.mapping.get(url, ExternalDetection(
                reachable=False, url=url, error="stub: not in mapping",
            ))
        ds._probe_url = stub
        return self

    def __exit__(self, *args):
        ds._probe_url = self.saved


def make_table(daemons, assignments=None) -> RoutingTable:
    t = RoutingTable()
    for d in daemons:
        t.upsert_daemon(d)
    for a in (assignments or []):
        t.assignments.append(a)
    return t


def main():
    print("\n[1] all daemons healthy → no events")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/", vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/", vram_total_mb=24 * 1024),
    ]
    table = make_table(daemons)
    fail_counts = {}
    stranded = {}
    with StubProbe({
        "http://0/": ExternalDetection(reachable=True, url="http://0/"),
        "http://1/": ExternalDetection(reachable=True, url="http://1/"),
    }):
        events = health_check_once(
            table, fail_counts=fail_counts, stranded_backup=stranded,
        )
        check("no events when all healthy", events == [])
        check("fail_counts cleared", fail_counts == {0: 0, 1: 0})

    print("\n[2] one transient miss → no event yet (debounce)")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/", vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/", vram_total_mb=24 * 1024),
    ]
    table = make_table(daemons)
    fail_counts = {}
    stranded = {}
    with StubProbe({
        "http://0/": ExternalDetection(reachable=True, url="http://0/"),
        # GPU 1 missing from mapping → stub returns reachable=False
    }):
        events = health_check_once(
            table, fail_counts=fail_counts, stranded_backup=stranded,
        )
        check("one miss, no down event yet",
              events == [], f"got {events}")
        check("fail_count incremented",
              fail_counts.get(1) == 1)
        check("daemon still in table",
              table.daemon_for_gpu(1) is not None)

    print("\n[3] persistent miss → daemon removed + assignments stranded")
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/", vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/", vram_total_mb=24 * 1024),
    ]
    table = make_table(daemons, [
        ModelAssignment(model_tag="qwen2.5:14b", gpu_index=1, role="chat"),
        ModelAssignment(model_tag="bigmodel", gpu_index=1, role="chat"),
    ])
    fail_counts = {1: HEALTH_FAIL_THRESHOLD - 1}   # already at threshold-1
    stranded = {}
    with StubProbe({
        "http://0/": ExternalDetection(reachable=True, url="http://0/"),
    }):
        events = health_check_once(
            table, fail_counts=fail_counts, stranded_backup=stranded,
        )
        check("got one down event",
              len(events) == 1
              and events[0].kind == "down"
              and events[0].gpu_index == 1)
        check("daemon removed from table",
              table.daemon_for_gpu(1) is None)
        check("assignment 1 demoted to AUTO",
              all(a.gpu_index == GPU_INDEX_AUTO for a in table.assignments))
        check("stranded backup recorded",
              stranded.get(("qwen2.5:14b", "chat")) == 1
              and stranded.get(("bigmodel", "chat")) == 1)

    print("\n[4] daemon recovers → 'back' event + assignments restored")
    # Caller reinstalls the daemon (their UI / supervisor would do this
    # when the user re-spawns or detects external is back). Then the
    # next health_check_once should restore the stranded assignments.
    daemons = [
        GpuDaemon(gpu_index=0, url="http://0/", vram_total_mb=24 * 1024),
        GpuDaemon(gpu_index=1, url="http://1/", vram_total_mb=24 * 1024),
    ]
    # Pre-populate as if we'd been through a down cycle.
    fail_counts = {0: 0, 1: 0}
    stranded = {("model-a", "chat"): 1, ("model-b", "any"): 1}
    table = make_table(daemons, [
        ModelAssignment(model_tag="model-a", gpu_index=GPU_INDEX_AUTO,
                         role="chat"),
        ModelAssignment(model_tag="model-b", gpu_index=GPU_INDEX_AUTO,
                         role="any"),
    ])
    with StubProbe({
        "http://0/": ExternalDetection(reachable=True, url="http://0/"),
        "http://1/": ExternalDetection(reachable=True, url="http://1/"),
    }):
        events = health_check_once(
            table, fail_counts=fail_counts, stranded_backup=stranded,
        )
        kinds = sorted(e.kind for e in events)
        check("two 'back' events emitted",
              kinds == ["back", "back"], f"got {kinds}")
        # Both stranded entries removed
        check("stranded backup cleared",
              stranded == {})
        # Both assignments restored to GPU 1
        gpu_indices = sorted(a.gpu_index for a in table.assignments)
        check("assignments restored to original GPU 1",
              gpu_indices == [1, 1], f"got {gpu_indices}")

    print("\n[5] empty table → returns []")
    check("empty table sweep is no-op",
          health_check_once(RoutingTable()) == [])
    check("None table → []",
          health_check_once(None) == [])

    print("\n[6] flap recovery — fails twice, recovers, fails again, recovers")
    daemons = [GpuDaemon(gpu_index=0, url="http://0/",
                          vram_total_mb=24 * 1024)]
    table = make_table(daemons, [
        ModelAssignment(model_tag="m", gpu_index=0, role="chat"),
    ])
    fail_counts = {}
    stranded = {}
    # Fail once
    with StubProbe({}):
        health_check_once(table, fail_counts=fail_counts,
                            stranded_backup=stranded)
    # Fail twice — daemon removed, assignment stranded
    with StubProbe({}):
        events = health_check_once(table, fail_counts=fail_counts,
                                     stranded_backup=stranded)
        check("flap: down on second consecutive miss",
              any(e.kind == "down" for e in events))
        check("flap: assignment demoted",
              table.assignments[0].gpu_index == GPU_INDEX_AUTO)
    # Caller reinstalls the daemon
    table.upsert_daemon(GpuDaemon(gpu_index=0, url="http://0/",
                                    vram_total_mb=24 * 1024))
    with StubProbe({
        "http://0/": ExternalDetection(reachable=True, url="http://0/"),
    }):
        events = health_check_once(table, fail_counts=fail_counts,
                                     stranded_backup=stranded)
        check("flap: 'back' event after restoration",
              any(e.kind == "back" for e in events))
        check("flap: assignment pinned back to GPU 0",
              table.assignments[0].gpu_index == 0)
        check("flap: stranded cleared",
              stranded == {})

    print(f"\n=== health-check summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

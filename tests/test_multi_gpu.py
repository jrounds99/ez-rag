"""Tests for multi_gpu — the data layer for per-model GPU routing.

Pure unit tests, no subprocess / network. Cover:
  - GpuDaemon / ModelAssignment / RoutingTable construction
  - Lookup precedence: assignment → default → first → fallback
  - Role matching (chat / embed / any)
  - TOML round-trip (rendered → parsed → equal)
  - Tolerant parsing of malformed / partial TOML
  - Workspace load/save with atomic write
  - find_free_port, derive_default_table helpers
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.multi_gpu import (
    GPU_INDEX_AUTO, GpuDaemon, ModelAssignment, RoutingTable,
    derive_default_table, find_free_port, load_routing_table,
    parse_toml, render_toml, routing_path, save_routing_table,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    print("\n[1] empty table falls back to localhost:11434")
    t = RoutingTable()
    check("empty table url_for any model = localhost:11434",
          t.url_for("qwen2.5:7b") == "http://127.0.0.1:11434",
          f"got {t.url_for('qwen2.5:7b')!r}")
    check("empty table first_daemon = None",
          t.first_daemon() is None)
    check("empty table assignment_for = None",
          t.assignment_for("qwen2.5:7b") is None)

    print("\n[2] single daemon on GPU 0")
    t = RoutingTable(default_gpu_index=0)
    t.upsert_daemon(GpuDaemon(
        gpu_index=0, gpu_name="RTX 5090", vram_total_mb=32 * 1024,
        url="http://127.0.0.1:11434", managed=False,
    ))
    check("single daemon resolves",
          t.url_for("any-model") == "http://127.0.0.1:11434")
    check("first_daemon returns it",
          t.first_daemon() is not None
           and t.first_daemon().gpu_index == 0)
    check("daemon_for_gpu(0) hits",
          t.daemon_for_gpu(0) is not None)
    check("daemon_for_gpu(1) misses",
          t.daemon_for_gpu(1) is None)

    print("\n[3] two daemons + per-model assignment")
    t = RoutingTable(default_gpu_index=0)
    t.upsert_daemon(GpuDaemon(
        gpu_index=0, gpu_name="RTX 5090", vram_total_mb=32 * 1024,
        url="http://127.0.0.1:11434", managed=False,
    ))
    t.upsert_daemon(GpuDaemon(
        gpu_index=1, gpu_name="RTX 3090", vram_total_mb=24 * 1024,
        url="http://127.0.0.1:11435", managed=True, pid=12345,
    ))
    t.upsert_assignment("qwen2.5:14b", 1, role="chat")
    t.upsert_assignment("qwen2.5:7b", 0, role="chat")
    t.upsert_assignment("qwen3-embedding:8b", 0, role="embed")

    check("qwen2.5:14b -> GPU 1 daemon",
          t.url_for("qwen2.5:14b", "chat") == "http://127.0.0.1:11435")
    check("qwen2.5:7b -> GPU 0 daemon",
          t.url_for("qwen2.5:7b", "chat") == "http://127.0.0.1:11434")
    check("qwen3-embedding:8b embed -> GPU 0",
          t.url_for("qwen3-embedding:8b", "embed") == "http://127.0.0.1:11434")
    check("unassigned model falls to default GPU 0",
          t.url_for("llama3.1:8b", "chat") == "http://127.0.0.1:11434")

    print("\n[4] role precedence: exact role beats 'any'")
    t = RoutingTable(default_gpu_index=0)
    t.upsert_daemon(GpuDaemon(gpu_index=0, url="http://0/"))
    t.upsert_daemon(GpuDaemon(gpu_index=1, url="http://1/"))
    t.upsert_daemon(GpuDaemon(gpu_index=2, url="http://2/"))
    t.upsert_assignment("foo", 1, role="chat")
    t.upsert_assignment("foo", 2, role="any")
    check("foo+chat -> GPU 1 (exact role)",
          t.url_for("foo", "chat") == "http://1/")
    check("foo+embed -> GPU 2 (any wins)",
          t.url_for("foo", "embed") == "http://2/")

    print("\n[5] GPU_INDEX_AUTO defers to default")
    t = RoutingTable(default_gpu_index=1)
    t.upsert_daemon(GpuDaemon(gpu_index=0, url="http://0/"))
    t.upsert_daemon(GpuDaemon(gpu_index=1, url="http://1/"))
    t.upsert_assignment("automodel", GPU_INDEX_AUTO, role="any")
    check("auto assignment falls through to default GPU 1",
          t.url_for("automodel", "any") == "http://1/")

    print("\n[6] mutators")
    t = RoutingTable()
    t.upsert_daemon(GpuDaemon(gpu_index=0, url="http://a/"))
    t.upsert_daemon(GpuDaemon(gpu_index=0, url="http://b/"))   # update
    check("upsert replaces existing daemon by gpu_index",
          len(t.daemons) == 1 and t.daemons[0].url == "http://b/")
    t.upsert_daemon(GpuDaemon(gpu_index=1, url="http://c/"))
    t.remove_daemon(0)
    check("remove_daemon(0) drops it",
          len(t.daemons) == 1 and t.daemons[0].gpu_index == 1)

    t.upsert_assignment("m", 1, role="chat")
    t.upsert_assignment("m", 0, role="chat")
    check("upsert_assignment replaces same (model, role)",
          len([a for a in t.assignments
               if a.model_tag == "m" and a.role == "chat"]) == 1
           and t.assignments[0].gpu_index == 0)
    t.remove_assignment("m", role="chat")
    check("remove_assignment drops it",
          len(t.assignments) == 0)

    print("\n[7] TOML round-trip preserves data")
    t = RoutingTable(default_gpu_index=0,
                      spawn_managed_daemons=True,
                      use_sched_spread=False)
    t.upsert_daemon(GpuDaemon(
        gpu_index=0, gpu_name="RTX 5090", vram_total_mb=32768,
        url="http://127.0.0.1:11434", managed=False,
        keep_alive_s=1800, notes="external",
    ))
    t.upsert_daemon(GpuDaemon(
        gpu_index=1, gpu_name="RTX 3090", vram_total_mb=24576,
        url="http://127.0.0.1:11435", managed=True, pid=9999,
        keep_alive_s=600, notes="spawned 2026-05-03",
    ))
    t.upsert_assignment("qwen2.5:14b", 1, role="chat")
    t.upsert_assignment("qwen3-embedding:8b", 0, role="embed")

    rendered = render_toml(t)
    parsed = parse_toml(rendered)

    check("parsed default_gpu_index matches",
          parsed.default_gpu_index == 0)
    check("parsed spawn_managed_daemons matches",
          parsed.spawn_managed_daemons is True)
    check("parsed use_sched_spread matches",
          parsed.use_sched_spread is False)
    check("parsed daemon count matches", len(parsed.daemons) == 2)
    check("parsed assignment count matches", len(parsed.assignments) == 2)
    check("daemon 0 url roundtrip",
          parsed.daemons[0].url == "http://127.0.0.1:11434")
    check("daemon 1 pid roundtrip",
          parsed.daemons[1].pid == 9999)
    check("daemon 1 managed=True roundtrip",
          parsed.daemons[1].managed is True)
    check("daemon 0 managed=False roundtrip",
          parsed.daemons[0].managed is False)
    check("daemon notes roundtrip",
          parsed.daemons[1].notes == "spawned 2026-05-03")
    check("assignment qwen2.5:14b -> 1 (chat) roundtrip",
          any(a.model_tag == "qwen2.5:14b" and a.gpu_index == 1
               and a.role == "chat" for a in parsed.assignments))
    check("assignment embedder -> 0 (embed) roundtrip",
          any(a.model_tag == "qwen3-embedding:8b"
               and a.gpu_index == 0 and a.role == "embed"
               for a in parsed.assignments))

    print("\n[8] tolerant parsing of malformed / partial TOML")
    check("empty string -> empty table",
          len(parse_toml("").daemons) == 0)
    check("garbage -> empty table (no crash)",
          isinstance(parse_toml("not toml at all {{{"), RoutingTable))
    check("missing [[daemon]] section -> empty daemons",
          len(parse_toml("default_gpu_index = 0\n").daemons) == 0)
    partial = """
default_gpu_index = 0
[[daemon]]
gpu_index = 0
url = 'http://127.0.0.1:11434'
[[assignment]]
model = ''
gpu_index = 1
"""
    p = parse_toml(partial)
    check("partial daemon entry parses (defaults fill in)",
          len(p.daemons) == 1 and p.daemons[0].gpu_name == "")
    check("assignment with empty model is skipped",
          len(p.assignments) == 0,
          f"got {p.assignments!r}")

    bad_types = """
default_gpu_index = "junk"
[[daemon]]
gpu_index = "not a number"
"""
    p = parse_toml(bad_types)
    # default_gpu_index parse error → falls through, table is empty-ish
    check("bad types don't crash the parser",
          isinstance(p, RoutingTable))

    print("\n[9] strings with special characters round-trip")
    t = RoutingTable(default_gpu_index=0)
    t.upsert_daemon(GpuDaemon(
        gpu_index=0,
        gpu_name="GPU with 'apostrophe' in name",
        url="http://127.0.0.1:11434",
        notes="Has \"double\" and 'single' quotes",
    ))
    rendered = render_toml(t)
    parsed = parse_toml(rendered)
    check("apostrophe in gpu_name survives",
          parsed.daemons[0].gpu_name
          == "GPU with 'apostrophe' in name")
    check("mixed quotes in notes survive",
          parsed.daemons[0].notes
          == "Has \"double\" and 'single' quotes")

    print("\n[10] workspace load/save (atomic, creates dir)")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ws_root = Path(tmp)
        # File doesn't exist → empty table
        t0 = load_routing_table(ws_root)
        check("missing file -> empty table",
              len(t0.daemons) == 0)
        # Save populates the file
        t = RoutingTable(default_gpu_index=0)
        t.upsert_daemon(GpuDaemon(
            gpu_index=0, gpu_name="X", url="http://x/",
        ))
        t.upsert_assignment("m", 0, role="chat")
        save_routing_table(ws_root, t)
        check("save creates the file",
              routing_path(ws_root).is_file())
        # Reload returns equivalent data
        t2 = load_routing_table(ws_root)
        check("reloaded daemon count matches",
              len(t2.daemons) == 1)
        check("reloaded assignment count matches",
              len(t2.assignments) == 1)
        check("reloaded gpu_name matches",
              t2.daemons[0].gpu_name == "X")
        # Save again — overwrite works without leaving .tmp behind
        save_routing_table(ws_root, RoutingTable())
        leftover = list(routing_path(ws_root).parent.glob("*.tmp"))
        check("no .tmp leftover after second save",
              leftover == [])

    print("\n[11] find_free_port returns a port that's actually bindable")
    p = find_free_port(start=15000)
    check("find_free_port returns int",
          isinstance(p, int) and p >= 15000)
    # Bind it ourselves to confirm
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", p))
    check("returned port was actually bindable",
          True)   # if bind didn't raise, we're good

    print("\n[12] derive_default_table from synthetic detected GPUs")
    class FakeGpu:
        def __init__(self, idx, name, vram):
            self.index = idx
            self.name = name
            self.vram_total_mb = vram
    detected = [
        FakeGpu(0, "RTX 5090", 32 * 1024),
        FakeGpu(1, "RTX 3090", 24 * 1024),
    ]
    t = derive_default_table(detected)
    check("derive: default GPU is 0",
          t.default_gpu_index == 0)
    check("derive: only ONE daemon registered (the external one)",
          len(t.daemons) == 1,
          f"got {len(t.daemons)} daemons")
    check("derive: spawn_managed_daemons defaults True",
          t.spawn_managed_daemons is True)
    check("derive: external daemon URL = localhost:11434",
          t.daemons[0].url == "http://127.0.0.1:11434")
    check("derive: external daemon is unmanaged",
          t.daemons[0].managed is False)

    t_empty = derive_default_table([])
    check("derive: no GPUs -> default_gpu_index = -1",
          t_empty.default_gpu_index == -1)
    check("derive: no GPUs -> no daemons",
          len(t_empty.daemons) == 0)

    print(f"\n=== multi_gpu summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

"""Tests for list_running_models / unload_running_models.

These stub the httpx layer so no Ollama server is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import models as m


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


# Stub helpers --------------------------------------------------------------

class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self):
        return self._payload


def main():
    saved_get = m.httpx.get
    saved_post = m.httpx.post
    saved_request = m.httpx.request

    print("\n[1] list_running_models parses /api/ps response")
    m.httpx.get = lambda url, timeout=2.0: FakeResp(200, {
        "models": [
            {"name": "qwen3:35b", "size_vram": 22_000_000_000},
            {"name": "bge-m3:latest"},
            {"model": "phi4"},          # alt key
        ]
    })
    out = m.list_running_models("http://x")
    check("returned 3 tags",
          len(out) == 3, f"got {out}")
    check("includes qwen3:35b",
          "qwen3:35b" in out, f"got {out}")
    check("falls back to 'model' key when 'name' missing",
          "phi4" in out, f"got {out}")

    print("\n[2] list_running_models is empty on network error")
    def boom(url, timeout=2.0): raise RuntimeError("offline")
    m.httpx.get = boom
    out = m.list_running_models("http://x")
    check("empty list on error", out == [], f"got {out}")

    print("\n[3] unload_running_models evicts tags not in except_")
    # Configure /api/ps to return 3 models.
    m.httpx.get = lambda url, timeout=2.0: FakeResp(200, {
        "models": [{"name": "qwen3:35b"}, {"name": "phi4"}, {"name": "bge-m3"}]
    })
    posted: list[dict] = []
    def fake_post(url, json=None, timeout=10.0):
        posted.append({"url": url, "body": json})
        return FakeResp(200, {})
    m.httpx.post = fake_post

    unloaded = m.unload_running_models("http://x", except_={"bge-m3"})
    check("kept 'bge-m3', unloaded the rest",
          set(unloaded) == {"qwen3:35b", "phi4"}, f"got {unloaded}")
    # Each unload posts to /api/generate with keep_alive=0
    eviction_targets = [p["body"]["model"] for p in posted]
    check("posted unload for qwen3:35b",
          "qwen3:35b" in eviction_targets, f"posted={eviction_targets}")
    check("posted unload for phi4",
          "phi4" in eviction_targets, f"posted={eviction_targets}")
    check("did NOT post unload for bge-m3",
          "bge-m3" not in eviction_targets, f"posted={eviction_targets}")
    check("each unload uses keep_alive=0",
          all(p["body"].get("keep_alive") == 0 for p in posted),
          f"posted={posted}")

    print("\n[4] unload_running_models is a no-op when nothing is loaded")
    m.httpx.get = lambda url, timeout=2.0: FakeResp(200, {"models": []})
    posted.clear()
    out = m.unload_running_models("http://x")
    check("empty result", out == [], f"got {out}")
    check("no posts made", posted == [], f"posted={posted}")

    print("\n[5] except_=None evicts everything")
    m.httpx.get = lambda url, timeout=2.0: FakeResp(200, {
        "models": [{"name": "a"}, {"name": "b"}]
    })
    posted.clear()
    out = m.unload_running_models("http://x")
    check("evicted all listed", set(out) == {"a", "b"}, f"got {out}")

    print("\n[6] unload failure (post returns non-200) is reflected in result")
    m.httpx.get = lambda url, timeout=2.0: FakeResp(200, {
        "models": [{"name": "a"}, {"name": "b"}]
    })
    def half_fail(url, json=None, timeout=10.0):
        return FakeResp(500 if json["model"] == "b" else 200, {})
    m.httpx.post = half_fail
    out = m.unload_running_models("http://x")
    check("only successful unloads are returned",
          out == ["a"], f"got {out}")

    # Restore
    m.httpx.get = saved_get
    m.httpx.post = saved_post
    m.httpx.request = saved_request

    print(f"\n=== Unload-on-switch summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

"""Tests for include_sources export option + server endpoints.

Builds a workspace with a mix of file types (md, html, txt), exports
TWICE (once lean, once with sources), and verifies:
  - the lean export does NOT contain data/sources/
  - the with-sources export contains every file under data/sources/
  - manifest.json correctly flags include_sources
  - the server's /api/status reports has_sources accordingly
  - /api/source returns the file bytes
  - /api/source rejects path traversal
  - /api/source rejects 'docs/' paths and rebases under data/sources/
  - /api/source on a lean bundle returns 404
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.export import estimate_sources_size, export_chatbot
from ez_rag.ingest import ingest
from ez_rag.workspace import Workspace


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def make_ws(tmp: Path) -> Workspace:
    ws = Workspace(tmp / "src")
    ws.initialize()
    # 3 small files: md, html, txt — covers the most common ingest types.
    (ws.docs_dir / "alpha.md").write_text(
        "# Alpha\nBorder Collies are smart herding dogs.\n",
        encoding="utf-8")
    (ws.docs_dir / "beta.html").write_text(
        "<html><body><h1>Beta</h1><p>Some HTML content.</p>"
        "</body></html>",
        encoding="utf-8")
    (ws.docs_dir / "gamma.txt").write_text(
        "Plain text file with some content.\n",
        encoding="utf-8")
    cfg = Config(
        embedder_provider="fastembed",
        embedder_model="BAAI/bge-small-en-v1.5",
        unload_llm_during_ingest=False,
    )
    cfg.save(ws.config_path)
    ingest(ws, cfg=cfg)
    return ws


def http_get(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url)
    return urllib.request.urlopen(req, timeout=timeout)


def spawn_server_for(extracted: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=extracted,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def wait_for_server(url: str, timeout_s: int = 60) -> dict | None:
    """Poll /api/status until it responds. Returns the parsed JSON or None
    on timeout."""
    for _ in range(timeout_s):
        try:
            with http_get(url, timeout=1.0) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(1)
    return None


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_export_src_"))
    print(f"[setup] tmp: {tmp}")
    procs: list[subprocess.Popen] = []
    try:
        ws = make_ws(tmp)

        # ----- estimate_sources_size sanity --------------------------------
        n, b = estimate_sources_size(ws)
        check("estimate_sources_size finds 3 files",
              n == 3, f"got {n}")
        check("estimate_sources_size byte total > 0",
              b > 0, f"got {b}")

        # ----- export A: lean (default include_sources=False) --------------
        lean = export_chatbot(ws, tmp / "lean.zip",
                              title="LeanBot")
        with zipfile.ZipFile(lean) as zf:
            names = set(zf.namelist())
            mf = json.loads(zf.read("manifest.json").decode("utf-8"))
            sources_in_zip = [n for n in names if n.startswith("data/sources/")]
        check("lean export has manifest.json",
              "manifest.json" in names, "")
        check("lean manifest.include_sources is False",
              mf.get("include_sources") is False,
              f"got {mf.get('include_sources')!r}")
        check("lean export has NO data/sources/ entries",
              len(sources_in_zip) == 0,
              f"unexpected source files: {sources_in_zip[:5]}")

        # ----- export B: with sources --------------------------------------
        full = export_chatbot(ws, tmp / "full.zip",
                              title="FullBot",
                              include_sources=True)
        with zipfile.ZipFile(full) as zf:
            names = set(zf.namelist())
            mf = json.loads(zf.read("manifest.json").decode("utf-8"))
            source_entries = sorted(n for n in names
                                    if n.startswith("data/sources/"))
        check("full manifest.include_sources is True",
              mf.get("include_sources") is True,
              f"got {mf.get('include_sources')!r}")
        check("full manifest sources_count = 3",
              mf.get("sources_count") == 3,
              f"got {mf.get('sources_count')}")
        for expected in ("data/sources/alpha.md",
                         "data/sources/beta.html",
                         "data/sources/gamma.txt"):
            check(f"full zip contains {expected}",
                  expected in source_entries,
                  f"missing — got {source_entries}")

        # ----- spin up server.py from each bundle and probe ---------------
        # Lean bundle first
        lean_dir = tmp / "lean_extracted"
        lean_dir.mkdir()
        with zipfile.ZipFile(lean) as zf:
            zf.extractall(lean_dir)
        p = spawn_server_for(lean_dir); procs.append(p)
        status = wait_for_server("http://127.0.0.1:8765/api/status")
        if not status:
            err = (p.stderr.read() if p.stderr else b"").decode(
                "utf-8", errors="replace")[:300]
            check("lean server came up", False, f"stderr: {err}")
        else:
            check("lean server reports has_sources=False",
                  status.get("has_sources") is False,
                  f"got {status}")

            # /api/source must 404 on lean bundle
            try:
                http_get("http://127.0.0.1:8765/api/source?path=alpha.md",
                         timeout=3.0)
                check("lean /api/source -> 404", False,
                      "expected 404, got 200")
            except urllib.error.HTTPError as e:
                check("lean /api/source -> 404", e.code == 404,
                      f"got {e.code}")

        # Stop lean server, start full one (same port — they don't overlap).
        p.terminate(); p.wait(timeout=5)
        procs.remove(p)

        full_dir = tmp / "full_extracted"
        full_dir.mkdir()
        with zipfile.ZipFile(full) as zf:
            zf.extractall(full_dir)
        p = spawn_server_for(full_dir); procs.append(p)
        status = wait_for_server("http://127.0.0.1:8765/api/status")
        if not status:
            err = (p.stderr.read() if p.stderr else b"").decode(
                "utf-8", errors="replace")[:300]
            check("full server came up", False, f"stderr: {err}")
        else:
            check("full server reports has_sources=True",
                  status.get("has_sources") is True,
                  f"got {status}")

            # Plain path resolution
            path_q = urllib.parse.quote("alpha.md")
            with http_get(f"http://127.0.0.1:8765/api/source?path={path_q}") as r:
                body = r.read().decode("utf-8")
            check("full /api/source returns alpha.md content",
                  "Border Collies" in body, f"body[:120]={body[:120]!r}")

            # Citations stored as 'docs/<rel>' should also work
            path_q = urllib.parse.quote("docs/beta.html")
            with http_get(f"http://127.0.0.1:8765/api/source?path={path_q}") as r:
                body = r.read().decode("utf-8")
            check("docs/-prefixed citation path resolves",
                  "<h1>Beta</h1>" in body, f"body[:120]={body[:120]!r}")

            # Path traversal must be rejected
            evil = urllib.parse.quote("../../../etc/passwd")
            try:
                http_get(f"http://127.0.0.1:8765/api/source?path={evil}",
                         timeout=3.0)
                check("path-traversal rejected", False,
                      "got 200 — should have been 404/403")
            except urllib.error.HTTPError as e:
                check("path-traversal rejected",
                      e.code in (403, 404),
                      f"got {e.code}")

            # Missing file → 404
            try:
                http_get("http://127.0.0.1:8765/api/source?path=nope.md",
                         timeout=3.0)
                check("missing source -> 404", False, "got 200")
            except urllib.error.HTTPError as e:
                check("missing source -> 404", e.code == 404,
                      f"got {e.code}")

            # /api/page-image on a non-PDF returns 404 (not 500)
            try:
                http_get(
                    "http://127.0.0.1:8765/api/page-image?path=alpha.md&page=1",
                    timeout=3.0)
                check("page-image on non-PDF -> 404", False, "got 200")
            except urllib.error.HTTPError as e:
                check("page-image on non-PDF -> 404", e.code == 404,
                      f"got {e.code}")

    finally:
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try: p.kill()
                except Exception: pass
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== Export-sources summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

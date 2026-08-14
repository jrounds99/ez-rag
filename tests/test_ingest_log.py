"""Ingest-log manifest + wipe tests.

Covers:
  1. Manifest collection: per-file embed timestamp, chunk counts,
     pipeline decoding (embedder / parser backend / headers / dedup /
     redaction) from stored provenance strings.
  2. Mixed-embedder warning surfaces in the HTML.
  3. Auto-refresh: ingest leaves .ezrag/reports/ingest-log.html behind.
  4. Wipe: index + derived artifacts deleted, documents + sidecars
     untouched, refused while locked, rebuildable after.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.ingest_log import (
    _decode_pipeline, collect_manifest, render_html, wipe_index,
    write_ingest_log,
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
    print("\n[1] pipeline decoding")
    d = _decode_pipeline("1", "2+hdr+dedup", "ollama:bge-m3:567m")
    check("embedder shown", "bge-m3:567m embeddings" in d, d)
    check("headers + dedup decoded",
          "chunk headers" in d and "dedup" in d, d)
    check("built-in parser default", "built-in parser" in d, d)
    d2 = _decode_pipeline("1+marker", "2+hdr+dedup+redactabcd1234",
                          "ollama:nomic-embed-text")
    check("ML pdf backend decoded", "marker PDF parser" in d2, d2)
    check("redaction decoded", "redaction" in d2, d2)

    print("\n[2] end-to-end manifest on a real workspace")
    from ez_rag.workspace import Workspace
    from ez_rag.ingest import ingest

    tmp = Path(tempfile.mkdtemp(prefix="ezrag_ilog_"))
    ws = Workspace(tmp)
    ws.initialize()
    (ws.docs_dir / "a.md").write_text("Limestone quarry production data.",
                                       encoding="utf-8")
    (ws.docs_dir / "b.md").write_text("Salt mining outputs by county.",
                                       encoding="utf-8")
    cfg = ws.load_config()
    cfg.embedder_provider = "fastembed"
    ingest(ws, cfg=cfg)

    m = collect_manifest(tmp)
    check("two documents listed", m["n_files"] == 2, f"{m['n_files']}")
    check("chunks counted", m["n_chunks"] >= 2)
    check("embed timestamps present",
          all(f[3] > 0 for f in m["files"]))
    check("not flagged mixed", not m["mixed_embedders"])

    html_text = render_html(m, cfg=cfg)
    check("documents in HTML", "a.md" in html_text and "b.md" in html_text)
    check("pipeline column rendered", "chunk headers" in html_text)
    check("settings snapshot rendered", "Chunk size" in html_text)
    check("self-contained", "cdn." not in html_text
          and "<script src" not in html_text)

    print("\n[3] auto-refresh after ingest")
    stable = tmp / ".ezrag" / "reports" / "ingest-log.html"
    check("ingest wrote the stable report", stable.is_file())
    (ws.docs_dir / "c.md").write_text("Gravel aggregate notes.",
                                       encoding="utf-8")
    before = stable.stat().st_mtime_ns
    ingest(ws, cfg=cfg)
    check("report refreshed by next ingest",
          stable.stat().st_mtime_ns > before
          and "c.md" in stable.read_text(encoding="utf-8"))

    print("\n[4] mixed-embedder warning")
    import sqlite3
    conn = sqlite3.connect(str(ws.meta_db_path))
    conn.execute("UPDATE files SET embedder='ollama:other' "
                 "WHERE path LIKE '%a.md'")
    conn.commit()
    conn.close()
    m2 = collect_manifest(tmp)
    check("mixed flag set", m2["mixed_embedders"])
    check("warning in HTML",
          "MORE THAN ONE embedder" in render_html(m2))

    print("\n[5] wipe")
    sidecar = ws.docs_dir / "a.md.ezrag-meta.toml"
    sidecar.write_text("# sidecar placeholder", encoding="utf-8")
    deleted = wipe_index(tmp)
    check("index deleted", not ws.meta_db_path.exists(),
          f"deleted={deleted}")
    check("documents untouched",
          (ws.docs_dir / "a.md").exists()
          and (ws.docs_dir / "c.md").exists())
    check("sidecars untouched", sidecar.exists())
    check("wipe idempotent", wipe_index(tmp) == [])
    st = ingest(ws, cfg=cfg)
    check("rebuild works after wipe",
          st.files_new == 3 and st.files_errored == 0, f"{vars(st)}")

    print("\n[6] wipe refused while locked")
    import gc
    gc.collect()
    from ez_rag.security import WorkspaceLockedError, lock_workspace, \
        unlock_workspace
    lock_workspace(tmp, "passphrase-123")
    try:
        wipe_index(tmp)
        check("locked wipe refused", False)
    except WorkspaceLockedError:
        check("locked wipe refused", True)
    unlock_workspace(tmp, "passphrase-123")
    check("unlock after refusal works", ws.meta_db_path.exists())

    print(f"\n=== ingest-log summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

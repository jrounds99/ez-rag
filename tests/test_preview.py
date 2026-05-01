"""Tests for the citation page-image preview cache."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import preview as prev


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    tmp_cache = Path(tempfile.mkdtemp(prefix="ezrag_preview_"))
    saved_cache = prev.PREVIEW_CACHE_DIR
    prev.PREVIEW_CACHE_DIR = tmp_cache
    print(f"[setup] cache: {tmp_cache}")

    try:
        # Use any PDF the test harness has on hand.
        candidates = [
            ROOT / "tests" / "fixtures" / "sample.pdf",
            Path(r"C:\Users\jroun\workstuff\ez-rag-test\docs\attention.pdf"),
        ]
        pdf = next((c for c in candidates if c.exists()), None)

        # ----- key + cache path -----
        ck = prev.cache_path_for(Path("/some/file.pdf"), 5)
        check("cache_path_for produces a path under cache dir",
              ck.parent == tmp_cache, f"got {ck}")
        check("cache key includes _p<page>",
              "_p5.png" in ck.name, f"name={ck.name}")

        # ----- bad inputs -----
        check("page=0 returns None",
              prev.render_pdf_page(Path("x.pdf"), 0) is None, "")
        check("missing file returns None",
              prev.render_pdf_page(tmp_cache / "nope.pdf", 1) is None, "")

        # ----- happy path (only if we have a PDF on hand) -----
        if pdf is not None:
            out = prev.render_pdf_page(pdf, 1)
            check("render produced an image",
                  out is not None and out.exists() and out.stat().st_size > 0,
                  f"out={out}")
            if out is not None:
                # PNG magic bytes
                with out.open("rb") as f:
                    head = f.read(8)
                check("output is a PNG",
                      head[:8] == b"\x89PNG\r\n\x1a\n", f"head={head!r}")

                # Re-render is idempotent + bumps mtime
                first_mtime = out.stat().st_mtime
                time.sleep(1.05)
                out2 = prev.render_pdf_page(pdf, 1)
                check("idempotent re-render returns same path",
                      out2 == out, f"out={out} out2={out2}")
                check("re-render touches mtime (cache hit lease)",
                      out2.stat().st_mtime > first_mtime,
                      f"first={first_mtime} second={out2.stat().st_mtime}")

                # Out-of-range page returns None and doesn't crash
                check("out-of-range page returns None",
                      prev.render_pdf_page(pdf, 99999) is None, "")
        else:
            print("  SKIP  PDF rendering tests — no sample PDF available")

        # ----- sweep -----
        # Manufacture an old + a fresh image
        old = tmp_cache / "OLD_p1.png"
        old.write_bytes(b"\x89PNG\r\n\x1a\n" + b"old")
        old_t = time.time() - (5 * 86400)
        os.utime(old, (old_t, old_t))
        fresh = tmp_cache / "FRESH_p1.png"
        fresh.write_bytes(b"\x89PNG\r\n\x1a\n" + b"new")

        removed = prev.sweep_old_previews(days=3)
        check("sweep removed old image",
              not old.exists(), "old image still present")
        check("sweep kept fresh image",
              fresh.exists(), "fresh image was deleted")
        check("sweep returned correct count",
              removed == 1, f"got {removed}")

        # Sweep on missing dir is safe
        bogus_cache = Path(tempfile.mkdtemp())
        bogus_cache.rmdir()
        prev.PREVIEW_CACHE_DIR = bogus_cache
        try:
            n = prev.sweep_old_previews()
            check("sweep on missing dir returns 0",
                  n == 0, f"got {n}")
        finally:
            prev.PREVIEW_CACHE_DIR = tmp_cache

    finally:
        prev.PREVIEW_CACHE_DIR = saved_cache
        shutil.rmtree(tmp_cache, ignore_errors=True)

    print(f"\n=== Preview summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

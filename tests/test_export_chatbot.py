"""End-to-end test for the chatbot export.

Steps:
  1. Build a tiny workspace and ingest a doc
  2. export_chatbot() → archive
  3. Unzip into a temp dir
  4. Verify all expected files present, theme placeholders substituted
  5. Verify ezrag_lib package imports cleanly from the unzipped tree
  6. Smoke-test the server: spawn server.py in a subprocess, GET /api/status,
     ensure it returns valid JSON identifying the bundled index

Does NOT:
  - exercise /api/ask (would require pulling an Ollama model)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.export import export_chatbot
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
    (ws.docs_dir / "alpha.md").write_text(
        "Border Collies are intelligent herding dogs.\n", encoding="utf-8")
    (ws.docs_dir / "beta.md").write_text(
        "Newton's second law: F equals m times a.\n", encoding="utf-8")
    cfg = Config(
        embedder_provider="fastembed",
        embedder_model="BAAI/bge-small-en-v1.5",
        unload_llm_during_ingest=False,
    )
    cfg.save(ws.config_path)
    ingest(ws, cfg=cfg)
    return ws


CUSTOM_PALETTE = {
    "accent": "#FF00AA", "accent_soft": "#AA00FF",
    "bg": "#001020", "surface": "#002030", "surface_hi": "#003040",
    "on_surface": "#FFFFFF", "on_surface_dim": "#888888",
    "success": "#11FF11", "warning": "#FFAA00", "danger": "#FF1111",
    "user_bubble": "#112233", "assist_bubble": "#001122",
    "chip_bg": "#223344",
}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_export_"))
    print(f"[setup] tmp: {tmp}")
    try:
        ws = make_ws(tmp)
        check("workspace ingested", ws.meta_db_path.exists(), "")

        out = export_chatbot(ws, tmp / "bundle.zip",
                             palette=CUSTOM_PALETTE, title="MyBot")
        check("zip created", out.exists() and out.stat().st_size > 0,
              f"size={out.stat().st_size if out.exists() else 0}")

        # ----- structural checks -----
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            for required in [
                "data/meta.sqlite", "data/config.toml",
                "ezrag_lib/__init__.py",
                "ezrag_lib/config.py", "ezrag_lib/embed.py",
                "ezrag_lib/generate.py", "ezrag_lib/index.py",
                "ezrag_lib/retrieve.py",
                "chat.html", "chat.css", "chat.js",
                "server.py", "chatbot_cli.py",
                "run.bat", "run.sh", "run_cli.bat", "run_cli.sh",
                "requirements.txt", "README.md",
            ]:
                check(f"zip contains {required}", required in names,
                      f"missing — entries: {sorted(names)[:5]}…")

            # placeholder substitution in chat.css
            css = zf.read("chat.css").decode("utf-8")
            check("chat.css ACCENT placeholder filled",
                  CUSTOM_PALETTE["accent"] in css, "")
            check("chat.css BG placeholder filled",
                  CUSTOM_PALETTE["bg"] in css, "")
            check("chat.css has no unfilled placeholders",
                  "{{" not in css, "found '{{' in css")

            html = zf.read("chat.html").decode("utf-8")
            check("chat.html title substituted",
                  "MyBot" in html, "title not in html")
            check("chat.html no unfilled placeholders",
                  "{{" not in html, "found '{{' in html")

            # shell script LF endings
            sh = zf.read("run.sh")
            check("run.sh has no CRLF",
                  b"\r\n" not in sh, "CRLF found in run.sh")

            # data/config.toml is a clean TOML write of cfg
            cfg_text = zf.read("data/config.toml").decode("utf-8")
            check("config.toml has llm_model",
                  "llm_model" in cfg_text, "")

        # ----- extract & import test -----
        extracted = tmp / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(out) as zf:
            zf.extractall(extracted)
        # ezrag_lib should be importable from the extracted root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(extracted)
        rc = subprocess.run(
            [sys.executable, "-c",
             "import ezrag_lib.config, ezrag_lib.embed, ezrag_lib.generate, "
             "ezrag_lib.index, ezrag_lib.retrieve; "
             "print('ok')"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        check("vendored ezrag_lib imports cleanly",
              rc.returncode == 0 and "ok" in rc.stdout,
              f"stderr={rc.stderr!r}")

        # ----- /api/status smoke test -----
        # Spawn server.py and GET /api/status. Skip if fastembed isn't
        # installed in the test environment (it is, since ez-rag uses it).
        port = "8766"
        server_env = os.environ.copy()
        # Server reads PORT-fixed (8765) — patch by injecting env. The script
        # hard-codes 8765, so we'll just use that and accept the conflict
        # risk.
        proc = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=extracted, env=server_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            # Wait up to 60s for server to listen (fastembed init can be slow)
            url = "http://127.0.0.1:8765/api/status"
            ok = False
            err = ""
            for _ in range(60):
                if proc.poll() is not None:
                    out, perr = proc.communicate(timeout=2)
                    err = (perr or out).decode("utf-8", errors="replace")[:500]
                    break
                try:
                    with urllib.request.urlopen(url, timeout=1) as r:
                        body = r.read().decode("utf-8")
                        data = json.loads(body)
                        ok = True
                        break
                except Exception:
                    time.sleep(1)
            check("server /api/status returns 200 + JSON", ok,
                  f"server didn't come up. stderr/stdout snippet: {err}")
            if ok:
                check("status reports files > 0",
                      data.get("files", 0) > 0, f"data={data}")
                check("status reports chunks > 0",
                      data.get("chunks", 0) > 0, f"data={data}")
                check("status reports embedder",
                      bool(data.get("embedder")), f"data={data}")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== Export-chatbot summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

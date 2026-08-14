"""Ingest manifest: what's in this RAG, when, and how it was built.

Answers "what did I actually ingest?" without spelunking SQLite:
every document with its embed timestamp, chunk count, size, and the
exact pipeline that processed it — embedder, parser (including any ML
PDF backend), and chunker features (headers / dedup / redaction),
decoded from the provenance strings the index already stores per file.

Artifacts:
  - `.ezrag/reports/ingest-log.html`  — stable name, refreshed after
    every ingest, so "open the ingest log" always works. A dated copy
    is NOT kept (the manifest describes current state; history lives
    in git-style provenance per file).
  - `ez-rag ingest-log [--open]` renders on demand; the GUI Files tab
    has an Open button.

Self-contained HTML — no CDNs, nothing external.
"""
from __future__ import annotations

import html
import sqlite3
import time
from pathlib import Path


def _decode_pipeline(parser_version: str, chunker_version: str,
                      embedder: str) -> str:
    """Turn stored provenance strings into a human-readable tool list."""
    bits = []
    if embedder:
        bits.append(embedder.replace("ollama:", "") + " embeddings")
    pv = parser_version or ""
    if "+" in pv:
        bits.append(f"{pv.split('+', 1)[1]} PDF parser")
    else:
        bits.append("built-in parser")
    cv = chunker_version or ""
    if "+hdr" in cv:
        bits.append("chunk headers")
    if "+dedup" in cv:
        bits.append("dedup")
    if "+redact" in cv:
        bits.append("redaction")
    return " · ".join(bits)


def collect_manifest(ws_root: Path) -> dict:
    """Gather the manifest data (no rendering)."""
    ws_root = Path(ws_root)
    db = ws_root / ".ezrag" / "meta.sqlite"
    if not db.is_file():
        raise FileNotFoundError(f"No index at {db} — nothing ingested yet.")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True,
                            timeout=3)
    try:
        files = conn.execute(
            "SELECT path, bytes, n_chunks, created_at, embedder, "
            "parser_version, chunker_version FROM files "
            "ORDER BY created_at DESC"
        ).fetchall()
        n_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks").fetchone()[0]
        dims = conn.execute(
            "SELECT length(embedding)/4 AS d, COUNT(*) FROM chunks "
            "GROUP BY d").fetchall()
    finally:
        conn.close()

    embedders = sorted({f[4] for f in files})
    return {
        "workspace": str(ws_root),
        "files": files,
        "n_files": len(files),
        "n_chunks": n_chunks,
        "total_bytes": sum(f[1] for f in files),
        "embedders": embedders,
        "dims": dims,
        "mixed_embedders": len(embedders) > 1 or len(dims) > 1,
    }


def render_html(m: dict, cfg=None) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M")

    warn = ""
    if m["mixed_embedders"]:
        warn = ("<div class='warn'>⚠ This index contains vectors from "
                "MORE THAN ONE embedder — retrieval will refuse until a "
                "re-ingest finishes converging on one. Embedders: "
                + html.escape(", ".join(m["embedders"]))
                + " · dimensions: "
                + ", ".join(f"{d}-d ×{n}" for d, n in m["dims"])
                + "</div>")

    cfg_rows = ""
    if cfg is not None:
        pairs = [
            ("Chat model", getattr(cfg, "llm_model", "")),
            ("Embedder", getattr(cfg, "ollama_embed_model", "")),
            ("Chunk size / overlap",
             f"{getattr(cfg, 'chunk_size', '')} / "
             f"{getattr(cfg, 'chunk_overlap', '')}"),
            ("Chunk headers", getattr(cfg, "chunk_headers", True)),
            ("Dedup", getattr(cfg, "dedup_chunks", True)),
            ("Redaction",
             f"{len(getattr(cfg, 'redact_terms', []) or [])} term(s)"),
            ("PDF backend", getattr(cfg, "pdf_backend", "auto")),
            ("OCR", f"{getattr(cfg, 'enable_ocr', True)} "
                    f"({getattr(cfg, 'ocr_provider', 'auto')})"),
            ("Contextual (per-chunk LLM)",
             getattr(cfg, "enable_contextual", False)),
        ]
        cfg_rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td>"
            f"<td>{html.escape(str(v))}</td></tr>" for k, v in pairs)
        cfg_rows = (f"<h2>Current workspace settings</h2>"
                    f"<table class='kv'><tbody>{cfg_rows}</tbody></table>"
                    f"<p class='dim'>Settings shown are the CURRENT config; "
                    f"the per-file rows below show what each file was "
                    f"actually built with.</p>")

    rows = []
    for path, size, chunks, created, embedder, pv, cv in m["files"]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(created))
        tools = _decode_pipeline(pv, cv, embedder)
        size_s = (f"{size/1024/1024:.1f} MB" if size > 1024 * 1024
                  else f"{size/1024:.0f} KB")
        rows.append(
            f"<tr><td class='p'>{html.escape(path)}</td>"
            f"<td>{when}</td><td class='n'>{chunks}</td>"
            f"<td class='n'>{size_s}</td>"
            f"<td class='tools'>{html.escape(tools)}</td></tr>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Ingest log — {html.escape(Path(m['workspace']).name)}</title>
<style>
 body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 2rem auto;
        max-width: 1150px; padding: 0 1rem; background:#12141c;
        color:#e6e7eb; }}
 h1 {{ margin-bottom: 0; }} .dim {{ color:#9097a6; }}
 h2 {{ margin-top: 1.6rem; border-bottom: 1px solid #2a2e3f;
      padding-bottom: 4px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
 th, td {{ text-align: left; padding: 6px 10px;
          border-bottom: 1px solid #23263a; vertical-align: top; }}
 th {{ color:#9097a6; font-weight:600; position: sticky; top: 0;
      background:#12141c; }}
 .n {{ text-align: right; white-space: nowrap; }}
 .p {{ font-family: Consolas, monospace; font-size: 12.5px; }}
 .tools {{ color:#9097a6; font-size: 12px; }}
 .kv td:first-child {{ color:#9097a6; width: 260px; }}
 .warn {{ background:#3a2530; border:1px solid #f75a68; color:#ffd7db;
         padding:10px 14px; border-radius:8px; margin:12px 0; }}
 .stats {{ color:#9097a6; }}
</style></head><body>
<h1>Ingest log</h1>
<p class="stats">{html.escape(m['workspace'])} · generated {ts} ·
{m['n_files']} documents · {m['n_chunks']} chunks ·
{m['total_bytes']/1024/1024:.1f} MB of source material</p>
{warn}
{cfg_rows}
<h2>Documents ({m['n_files']})</h2>
<table><thead><tr><th>Document</th><th>Embedded</th><th>Chunks</th>
<th>Size</th><th>Pipeline used</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="dim">Regenerated automatically after every ingest ·
<code>ez-rag ingest-log</code></p>
</body></html>"""


def write_ingest_log(ws_root: Path, cfg=None) -> Path:
    """Render the manifest to the stable report path. Returns it."""
    ws_root = Path(ws_root)
    m = collect_manifest(ws_root)
    out_dir = ws_root / ".ezrag" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ingest-log.html"
    out.write_text(render_html(m, cfg=cfg), encoding="utf-8")
    return out


# ============================================================================
# Wipe — delete the index (vector DB) while leaving documents intact
# ============================================================================

def wipe_index(ws_root: Path) -> list[str]:
    """Delete the vector index + derived artifacts. Documents in docs/
    and sidecar metadata are untouched. Returns the deleted paths.

    Refuses while the workspace is locked (unlock first — proves you
    hold the passphrase before destroying the encrypted copy)."""
    from .security import require_unlocked
    ws_root = Path(ws_root)
    require_unlocked(ws_root)
    ez = ws_root / ".ezrag"
    targets = [
        ez / "meta.sqlite",
        ez / "meta.sqlite-wal",
        ez / "meta.sqlite-shm",
        ez / "glossary.json",
    ]
    deleted = []
    from .index import invalidate_matrix_cache
    invalidate_matrix_cache(ez / "meta.sqlite")
    # Windows refuses to delete files with open handles. Drop dangling
    # Index connections (GC) and retry briefly; callers that hold a
    # live Index (the GUI) close it before calling us.
    import gc
    import time as _time
    gc.collect()
    for t in targets:
        if not t.exists():
            continue
        last_err = None
        for attempt in range(4):
            try:
                t.unlink()
                deleted.append(str(t))
                last_err = None
                break
            except PermissionError as ex:
                last_err = ex
                gc.collect()
                _time.sleep(0.3)
        if last_err is not None:
            raise PermissionError(
                f"{t.name} is open in another process (chat window, "
                f"server, or a second ez-rag). Close it and retry."
            ) from last_err
    return deleted

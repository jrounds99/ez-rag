"""Export a workspace as a runnable chatbot zip.

The bundle contains:
  - the workspace's index (data/meta.sqlite + WAL/SHM if present)
  - the workspace's config.toml frozen as `data/config.toml`
  - chat.html / chat.css / chat.js — editable static UI
  - server.py — tiny stdlib HTTP server with streaming /api/ask
  - chatbot_cli.py — terminal alternative
  - run.bat / run.sh / run_cli.bat / run_cli.sh — cross-platform launchers
  - requirements.txt — pip deps
  - ezrag_lib/ — vendored retrieve / generate / index / embed / config
  - README.md

Theme colors are baked into chat.css at export time. The destination machine
needs Python 3.10+, Ollama with the configured model pulled, and a network
connection to install fastembed/numpy/httpx on first run.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from .workspace import Workspace


# Modules vendored into the chatbot bundle. These are exactly the import-graph
# leaves needed for retrieve+generate at chat time (no parsers, chunker,
# ingest, models, workspace, etc).
_VENDORED_MODULES = [
    "config.py",
    "embed.py",
    "generate.py",
    "index.py",
    "multi_gpu.py",  # resolve_url() — chat-time URL routing
    "retrieve.py",
]


def _palette_to_placeholders(palette: dict) -> dict[str, str]:
    """Map a theme palette to chat.css placeholder names."""
    return {
        "{{ACCENT}}":         palette.get("accent", "#7C7BFF"),
        "{{ACCENT_SOFT}}":    palette.get("accent_soft", "#5856D6"),
        "{{BG}}":             palette.get("bg", "#0F1115"),
        "{{SURFACE}}":        palette.get("surface", "#171922"),
        "{{SURFACE_HI}}":     palette.get("surface_hi", "#1F2230"),
        "{{ON_SURFACE}}":     palette.get("on_surface", "#E6E7EB"),
        "{{ON_SURFACE_DIM}}": palette.get("on_surface_dim", "#9097A6"),
        "{{SUCCESS}}":        palette.get("success", "#3DDC84"),
        "{{WARNING}}":        palette.get("warning", "#F6B042"),
        "{{DANGER}}":         palette.get("danger", "#F75A68"),
        "{{USER_BUBBLE}}":    palette.get("user_bubble", "#272A39"),
        "{{ASSIST_BUBBLE}}":  palette.get("assist_bubble", "#1A1D29"),
        "{{CHIP_BG}}":        palette.get("chip_bg", "#23263A"),
    }


def _substitute(text: str, repl: dict[str, str]) -> str:
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def estimate_sources_size(ws: Workspace) -> tuple[int, int]:
    """Return (file_count, byte_size) for the workspace's docs/ tree.

    Used by the GUI to show the user how big the export will be when they
    tick the 'include source files' checkbox.
    """
    if not ws.docs_dir.is_dir():
        return (0, 0)
    n, b = 0, 0
    try:
        for f in ws.docs_dir.rglob("*"):
            if f.is_file():
                n += 1
                try:
                    b += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return (n, b)


def export_chatbot(
    ws: Workspace,
    dest: Path,
    *,
    palette: dict | None = None,
    title: str | None = None,
    include_sources: bool = False,
    progress=None,
) -> Path:
    """Zip up a runnable chatbot rooted in this workspace's index.

    `include_sources` (default False) bundles every file in the workspace's
    `docs/` tree under `data/sources/<rel-path>` so the chatbot can render
    PDF page previews and serve original screenshots when the user clicks
    a citation chip. Off by default because PDF-heavy workspaces can be
    multi-GB; turn on for portable / "evidence-trail-included" exports.

    `progress(label, done, total)` is an optional callback fired during
    source copying so the GUI can show real progress for big workspaces.

    Returns the written archive path. `dest` is created or overwritten.
    `palette` should be one of the dicts loaded from `themes.toml`; if None
    the dark default is used. `title` is shown in the page <title> and H1
    (defaults to the workspace folder name).
    """
    dest = Path(dest)
    if not dest.suffix:
        dest = dest.with_suffix(".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)

    template_dir = Path(__file__).parent / "chatbot_template"
    if not template_dir.is_dir():
        raise RuntimeError(f"chatbot template missing at {template_dir}")

    # Effective theme + title for placeholder substitution
    palette = palette or {
        "accent": "#7C7BFF", "accent_soft": "#5856D6",
        "bg": "#0F1115", "surface": "#171922", "surface_hi": "#1F2230",
        "on_surface": "#E6E7EB", "on_surface_dim": "#9097A6",
        "success": "#3DDC84", "warning": "#F6B042", "danger": "#F75A68",
        "user_bubble": "#272A39", "assist_bubble": "#1A1D29",
        "chip_bg": "#23263A",
    }
    repl = _palette_to_placeholders(palette)
    bot_title = title or ws.root.name or "ez-rag chatbot"
    repl["{{TITLE}}"] = bot_title

    src_dir = Path(__file__).parent

    # Always-on flag so the chatbot's server knows whether to expose
    # /api/source / /api/page-image. Written into bundle/manifest.json.
    bundle_manifest = {
        "title": bot_title,
        "include_sources": bool(include_sources),
    }

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # ----- Index (data/) ----------------------------------------------
        if not ws.meta_db_path.exists():
            raise RuntimeError(
                f"No index at {ws.meta_db_path}. Run an ingest first."
            )
        zf.write(ws.meta_db_path, "data/meta.sqlite")
        # WAL/SHM if present (active write lock on the source DB)
        for sib in (ws.meta_db_path.with_suffix(ws.meta_db_path.suffix + "-wal"),
                    ws.meta_db_path.with_suffix(ws.meta_db_path.suffix + "-shm")):
            if sib.exists():
                zf.write(sib, f"data/{sib.name}")

        # Frozen config — write a clean copy from the live Config so we
        # normalize formatting / quoting on the way out.
        cfg = ws.load_config()
        tmp_cfg = dest.parent / f".{dest.stem}.config.toml.tmp"
        try:
            cfg.save(tmp_cfg)
            zf.write(tmp_cfg, "data/config.toml")
        finally:
            try:
                tmp_cfg.unlink()
            except OSError:
                pass

        # ----- Vendored ezrag_lib/ ----------------------------------------
        # Empty __init__.py so it's a proper package; relative imports work.
        zf.writestr("ezrag_lib/__init__.py", "")
        for mod in _VENDORED_MODULES:
            src = src_dir / mod
            if not src.is_file():
                raise RuntimeError(f"missing source module: {src}")
            zf.write(src, f"ezrag_lib/{mod}")

        # ----- Source files (data/sources/) -------------------------------
        # Walk ws.docs_dir and replicate it under data/sources/ in the bundle.
        # We preserve relative paths so citations like 'docs\\foo.pdf' resolve
        # cleanly on the destination (server.py strips the 'docs\\' prefix).
        source_count = 0
        source_bytes = 0
        if include_sources and ws.docs_dir.is_dir():
            files_to_copy = [f for f in ws.docs_dir.rglob("*") if f.is_file()]
            total = len(files_to_copy)
            for i, f in enumerate(files_to_copy, 1):
                try:
                    rel = f.relative_to(ws.docs_dir)
                except ValueError:
                    continue
                # Write under data/sources/<rel> using forward slashes so
                # the zip layout is portable to non-Windows extractors.
                arcname = "data/sources/" + str(rel).replace("\\", "/")
                try:
                    zf.write(f, arcname)
                    source_count += 1
                    source_bytes += f.stat().st_size
                except OSError as e:
                    # Don't let one bad file abort the whole export.
                    print(f"[export] skipped {rel}: {e}")
                if progress:
                    try:
                        progress("bundling sources", i, total)
                    except Exception:
                        pass
            bundle_manifest["sources_count"] = source_count
            bundle_manifest["sources_bytes"] = source_bytes

        # Manifest sits at the bundle root so server.py can read it cheaply
        # to know whether sources are available without scanning the zip.
        import json as _json
        zf.writestr("manifest.json",
                    _json.dumps(bundle_manifest, indent=2))

        # ----- Templates with placeholder substitution --------------------
        # Text files we substitute placeholders in before writing.
        for name in ("chat.html", "chat.css", "chat.js",
                     "server.py", "chatbot_cli.py", "README.md",
                     "run.bat", "run.sh", "run_cli.bat", "run_cli.sh",
                     "requirements.txt"):
            src = template_dir / name
            if not src.is_file():
                continue
            text = src.read_text(encoding="utf-8")
            text = _substitute(text, repl)
            # Keep shell scripts LF-only so they're executable on Mac/Linux.
            if name.endswith(".sh"):
                text = text.replace("\r\n", "\n")
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                # Mark executable (rwxr-xr-x) for Unix
                info.external_attr = (0o755 & 0xFFFF) << 16
                zf.writestr(info, text)
            else:
                zf.writestr(name, text)

    return dest

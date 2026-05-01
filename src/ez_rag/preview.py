"""PDF page → PNG preview cache.

Renders the cited page of a PDF to an image so the chat UI can show users
exactly what they're looking at when they click a citation chip.

Cache layout (cross-workspace, single shared folder):
    ~/.ezrag/preview_cache/{sha256(abs_path)[:16]}_p{page}.png

`sweep_old_previews(days=3)` runs on app startup and deletes any image
older than the cutoff. We do NOT delete on access — `mtime` is touched
when an image is re-served so frequently-viewed previews effectively get a
3-day rolling lease.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

PREVIEW_CACHE_DIR = Path.home() / ".ezrag" / "preview_cache"
# Chapter PDFs extracted via "Chapter (preview)" land here. Browser opens
# them inline; user keeps via the browser's built-in Download/Save.
# Same 3-day sweep as the page-image cache.
CHAPTER_CACHE_DIR = Path.home() / ".ezrag" / "chapter_cache"
DEFAULT_TTL_DAYS = 3
# 2.5x gives ~180 DPI when source is 72 DPI — readable on hi-DPI displays
# and still zoomable. The cost is ~4x the bytes vs 1.5x; preview cache
# self-evicts after 3 days so it doesn't grow forever.
DEFAULT_RENDER_SCALE = 2.5


def _key(pdf_path: Path, page: int) -> str:
    abs_path = str(pdf_path.resolve())
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_p{int(page)}.png"


def cache_path_for(pdf_path: Path, page: int) -> Path:
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREVIEW_CACHE_DIR / _key(pdf_path, page)


def render_pdf_page(
    pdf_path: Path,
    page: int,
    *,
    scale: float = DEFAULT_RENDER_SCALE,
    force: bool = False,
) -> Path | None:
    """Render `pdf_path` page `page` (1-indexed) to PNG and return the path.

    Returns None on any failure (PDF unreadable, page out of range, renderer
    not installed). Idempotent — re-renders only when the cached file is
    missing or `force=True`. Touches mtime on cache hits so the 3-day sweep
    effectively becomes a 3-day rolling lease per image.
    """
    if page is None or page <= 0:
        return None
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        return None
    out = cache_path_for(pdf_path, page)
    if out.exists() and not force:
        try:
            os.utime(out, None)  # refresh mtime so the sweep treats this as fresh
        except OSError:
            pass
        return out

    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        return None

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        idx = page - 1
        if idx < 0 or idx >= len(pdf):
            return None
        bitmap = pdf[idx].render(scale=scale)
        pil = bitmap.to_pil()
        pil.save(out, format="PNG", optimize=True)
        return out
    except Exception:
        return None


def extract_pdf_pages(
    pdf_path: Path,
    start_page: int,
    end_page: int,
    dest: Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Extract `start_page`..`end_page` (1-indexed, inclusive) from
    `pdf_path` and write them as a new PDF at `dest`.

    Used by the citation modal's "Download chapter (experimental)" button
    to save just the chapter that contains a cited passage. Returns the
    written path on success, None on any failure (no pypdf, page out of
    range, write error, etc.).
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return None
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        return None
    if start_page <= 0 or end_page < start_page:
        return None
    try:
        reader = PdfReader(str(pdf_path))
        n = len(reader.pages)
        # Clamp to actual page count — ingest sometimes records page
        # numbers that don't match pypdf's page index after a reparse.
        s = max(1, min(start_page, n))
        e = max(s, min(end_page, n))

        writer = PdfWriter()
        for i in range(s - 1, e):
            writer.add_page(reader.pages[i])

        # Best-effort metadata so the saved chapter is identifiable.
        try:
            writer.add_metadata({
                "/Title": (title or f"{pdf_path.stem} (pp. {s}-{e})"),
                "/Producer": "ez-rag chapter extract",
                "/Subject": f"Pages {s}-{e} extracted from {pdf_path.name}",
            })
        except Exception:
            pass

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            writer.write(f)
        return dest
    except Exception:
        return None


def sweep_old_previews(*, days: int = DEFAULT_TTL_DAYS) -> int:
    """Delete cached previews older than `days`. Returns number removed.

    Runs at app startup. Sweeps both the page-image cache (PNG) and the
    chapter-PDF cache (PDF). Failures are silent — a corrupt cache
    shouldn't prevent the GUI from starting.
    """
    cutoff = time.time() - (days * 86400)
    removed = 0
    for cache_dir, ext in (
        (PREVIEW_CACHE_DIR, ".png"),
        (CHAPTER_CACHE_DIR, ".pdf"),
    ):
        if not cache_dir.is_dir():
            continue
        try:
            for f in cache_dir.iterdir():
                if not f.is_file() or f.suffix.lower() != ext:
                    continue
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
        except OSError:
            pass
    return removed


def chapter_cache_path_for(pdf_path: Path, start_page: int, end_page: int) -> Path:
    """Stable cache path for the chapter PDF extract — same hash scheme
    as cache_path_for() so identical chapter requests reuse the file."""
    abs_path = str(pdf_path.resolve())
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]
    CHAPTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CHAPTER_CACHE_DIR / f"{digest}_p{int(start_page)}-{int(end_page)}.pdf"

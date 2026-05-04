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


def extract_pdf_window(
    pdf_path: Path,
    page: int,
    dest: Path,
    *,
    window: int = 7,
    title: str | None = None,
) -> Path | None:
    """Extract a windowed PDF centered on `page`.

    Layout of the produced PDF:
        Page 1                — the page in question (cover; verbatim
                                copy of source page `page`)
        Page 2                — auto-generated notes page explaining
                                what's in this extract
        Pages 3..(2+window)   — the `window` pages before `page`
        Page (3+window)       — the page in question (in its proper
                                sequence)
        Pages (4+window)..end — the `window` pages after `page`

    With the default window=7, the result is 17 total pages. The page
    in question is duplicated at position 1 (cover) and position 10
    (in-sequence), so a reader sees the relevant page immediately and
    can scan the surrounding context.

    Falls back to a smaller window when `page` is near the start or
    end of the source PDF (clamps to available pages).

    Returns the written path on success, None on any failure.

    Replaces the older chapter-based extract for the
    "Chapter (experimental)" download — chapter metadata was too
    unreliable on PDFs without bookmarks (e.g. the user's D&D Basic
    Rules PDF was indexed as one giant 320-page chapter, so the
    chapter export dumped the whole book).
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return None
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        return None
    if page <= 0 or window < 0:
        return None
    try:
        reader = PdfReader(str(pdf_path))
        n = len(reader.pages)
        if n == 0:
            return None
        # Clamp the page in question to the document.
        target = max(1, min(page, n))
        # Window edges, clamped to source bounds.
        start = max(1, target - window)
        end = min(n, target + window)

        writer = PdfWriter()

        # Page 1 — cover (the page in question, verbatim).
        writer.add_page(reader.pages[target - 1])

        # Page 2 — notes page generated with reportlab.
        try:
            notes_pdf_bytes = _build_notes_page(
                pdf_name=pdf_path.name,
                page=target,
                window=window,
                start=start,
                end=end,
                title=title,
            )
            if notes_pdf_bytes:
                from io import BytesIO
                notes_reader = PdfReader(BytesIO(notes_pdf_bytes))
                if notes_reader.pages:
                    writer.add_page(notes_reader.pages[0])
        except Exception:
            # If reportlab isn't available or the notes page fails to
            # build, skip it silently — user still gets the windowed
            # extract, just without the explanatory page.
            pass

        # Pages 3.. — the windowed range from the source.
        for i in range(start - 1, end):
            writer.add_page(reader.pages[i])

        # Best-effort metadata.
        try:
            writer.add_metadata({
                "/Title": (title
                            or f"{pdf_path.stem} (page {target} ±{window})"),
                "/Producer": "ez-rag windowed extract",
                "/Subject": (
                    f"Page {target} of {pdf_path.name}, with "
                    f"±{window} pages of context"
                ),
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


def _build_notes_page(
    *,
    pdf_name: str,
    page: int,
    window: int,
    start: int,
    end: int,
    title: str | None = None,
) -> bytes | None:
    """Build a single-page PDF (in-memory bytes) describing the layout
    of the windowed extract. Used as page 2 of the output PDF.

    Returns None if reportlab isn't installed (caller skips the page).
    """
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
    except ImportError:
        return None

    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=LETTER)
    width, height = LETTER

    accent = colors.HexColor("#5856D6")
    subtle = colors.HexColor("#5F6779")

    # Header
    c.setFillColor(accent)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(0.85 * inch, height - 1.0 * inch,
                  "Cited-page extract")

    if title:
        c.setFillColor(subtle)
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(0.85 * inch, height - 1.28 * inch,
                      _truncate_for_pdf(title, 90))

    # Body
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    y = height - 1.7 * inch
    line_h = 16

    cover_pos = 1
    notes_pos = 2
    seq_pos = 3 + (page - start)

    body_lines = [
        f"Source PDF:        {pdf_name}",
        f"Page of interest:  {page}",
        f"Context window:    {window} pages on each side  "
        f"(actual range: {start}–{end})",
        "",
        "Layout of this extract:",
        f"   Page {cover_pos} — the page in question (cover; same as "
        f"source page {page})",
        f"   Page {notes_pos} — these notes",
    ]
    if start <= page - 1:
        body_lines.append(
            f"   Pages 3–{2 + (page - start)}  — source pages "
            f"{start}–{page - 1} (preceding context)"
        )
    body_lines.append(
        f"   Page {seq_pos} — source page {page} (in proper sequence)"
    )
    if page + 1 <= end:
        body_lines.append(
            f"   Pages {seq_pos + 1}–{seq_pos + (end - page)} — source "
            f"pages {page + 1}–{end} (following context)"
        )
    body_lines.extend([
        "",
        "Why this layout:",
        "   When ez-rag retrieves a passage, this extract gives you "
        "the cited page",
        "   plus the surrounding context — usually enough to verify "
        "the citation",
        "   without opening the full source PDF (which can be "
        "hundreds of pages).",
        "",
        "   The cited page appears twice — once as the cover (page 1) "
        "for quick",
        "   reference, and once in proper sequence so you can read it "
        "with its",
        "   surrounding context.",
    ])

    for line in body_lines:
        c.drawString(0.85 * inch, y, line)
        y -= line_h

    # Footer
    c.setFillColor(subtle)
    c.setFont("Helvetica", 8)
    c.drawString(0.85 * inch, 0.5 * inch,
                  "ez-rag · windowed citation extract")
    c.drawRightString(width - 0.85 * inch, 0.5 * inch,
                       f"Page 2 · cover at page {cover_pos}, full "
                       f"context starts page 3")
    c.save()
    return bio.getvalue()


def _truncate_for_pdf(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


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


def window_cache_path_for(pdf_path: Path, page: int, window: int) -> Path:
    """Stable cache path for the windowed extract — keyed by source PDF
    + the centered page + window radius, so identical requests reuse
    the file."""
    abs_path = str(pdf_path.resolve())
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]
    CHAPTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CHAPTER_CACHE_DIR / f"{digest}_window_p{int(page)}_w{int(window)}.pdf"

"""Chapter boundary detection.

Given a list of chunks for a single file, return a list of chapter dicts:
    [{"title": str, "start_ord": int, "end_ord": int,
      "start_page": int|None, "end_page": int|None}, ...]

Strategy by source format:

- PDFs with bookmarks: extract via `pypdf.outline` and map page ranges to
  chunk-ord ranges using the chunks' `page` field.
- Anything with `section` set on its chunks (DOCX heading style, future MD
  H1 splitting, HTML <h1>, etc.): group consecutive chunks sharing the same
  non-empty `section` value.
- Otherwise: a single "Document" chapter spanning the whole file.

Chapters never overlap. Chunks not covered by any chapter (e.g., front
matter before the first bookmark) get rolled into the prologue chapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .chunker import Chunk


def _from_sections(chunks: Sequence[Chunk]) -> list[dict]:
    """Group consecutive chunks with the same non-empty `section` value."""
    out: list[dict] = []
    current_title: str | None = None
    start_ord: int | None = None
    start_page: int | None = None
    last_page: int | None = None
    for c in chunks:
        title = (c.section or "").strip() or None
        if title != current_title:
            # flush previous
            if current_title is not None and start_ord is not None:
                out.append({
                    "title": current_title,
                    "start_ord": start_ord,
                    "end_ord": c.ord - 1,
                    "start_page": start_page,
                    "end_page": last_page,
                })
            current_title = title
            start_ord = c.ord
            start_page = c.page
        last_page = c.page if c.page is not None else last_page
    if current_title is not None and start_ord is not None:
        out.append({
            "title": current_title,
            "start_ord": start_ord,
            "end_ord": chunks[-1].ord,
            "start_page": start_page,
            "end_page": last_page,
        })
    return out


def _from_pdf_outline(pdf_path: Path, chunks: Sequence[Chunk]) -> list[dict]:
    """Extract chapter boundaries from a PDF's bookmark tree.

    Maps each bookmark to its destination page, then converts that to a
    chunk-ord range using the chunks' `page` fields. Returns [] if the PDF
    has no usable outline.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return []

    pages_total = len(reader.pages)

    # Walk the outline tree, flatten to (title, page_1indexed)
    flat: list[tuple[str, int]] = []

    def walk(items):
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                title = (item.title or "").strip()
            except Exception:
                title = ""
            if not title:
                continue
            try:
                p0 = reader.get_destination_page_number(item)
            except Exception:
                continue
            if p0 is None:
                continue
            flat.append((title, p0 + 1))

    try:
        walk(reader.outline)
    except Exception:
        return []

    if not flat:
        return []

    # De-dupe + sort by page; only keep the first bookmark per page so
    # nested outlines (which often repeat the parent's page) collapse.
    seen_pages: set[int] = set()
    flat.sort(key=lambda x: x[1])
    deduped = []
    for title, page in flat:
        if page in seen_pages:
            continue
        seen_pages.add(page)
        deduped.append((title, page))

    if not deduped:
        return []

    # Build chapter ranges keyed by page.
    chapters_by_page = []
    for i, (title, start_page) in enumerate(deduped):
        end_page = (deduped[i + 1][1] - 1
                    if i + 1 < len(deduped) else pages_total)
        chapters_by_page.append({
            "title": title,
            "start_page": start_page,
            "end_page": end_page,
        })

    # Now translate page ranges into chunk-ord ranges.
    # Index chunks by page for fast lookup. A chunk with page=None gets
    # rolled into the previous chapter (it's typically front matter or a
    # full-file post-chunk).
    out: list[dict] = []
    chunk_orders = [c.ord for c in chunks]
    if not chunk_orders:
        return []

    for ch in chapters_by_page:
        sp, ep = ch["start_page"], ch["end_page"]
        in_range = [c for c in chunks
                    if c.page is not None and sp <= c.page <= ep]
        if not in_range:
            continue
        out.append({
            "title": ch["title"],
            "start_ord": min(c.ord for c in in_range),
            "end_ord": max(c.ord for c in in_range),
            "start_page": sp,
            "end_page": ep,
        })

    if not out:
        return []

    # Stretch the first chapter back to chunk 0 to absorb any front matter
    # whose page wasn't covered by a bookmark.
    out.sort(key=lambda c: c["start_ord"])
    if out[0]["start_ord"] > chunk_orders[0]:
        out[0] = {**out[0], "start_ord": chunk_orders[0]}

    # Ensure non-overlapping, contiguous ranges by clamping each end to
    # the next start - 1, and stretching the last one to the final chunk.
    for i in range(len(out) - 1):
        out[i]["end_ord"] = out[i + 1]["start_ord"] - 1
    out[-1]["end_ord"] = max(out[-1]["end_ord"], chunk_orders[-1])
    return out


def detect_chapters(file_path: Path, chunks: Sequence[Chunk]) -> list[dict]:
    """Top-level dispatch. Returns [] if no useful boundaries can be found —
    callers should treat that as 'one big chapter spanning the whole file'.
    """
    if not chunks:
        return []

    suffix = file_path.suffix.lower()

    # PDFs: try outline first, fall back to section-based (rare for PDFs but
    # possible) and finally to a single "Document" chapter.
    if suffix == ".pdf":
        from_outline = _from_pdf_outline(file_path, chunks)
        if from_outline:
            return from_outline

    # Section-based for everything else (DOCX, MD, HTML, EPUB, …)
    from_sections = _from_sections(chunks)
    if from_sections and len(from_sections) > 1:
        return from_sections

    # Single-chapter fallback so callers can still expand-to-chapter (it's
    # just a no-op if there's only one chapter).
    pages = [c.page for c in chunks if c.page is not None]
    return [{
        "title": "Document",
        "start_ord": chunks[0].ord,
        "end_ord": chunks[-1].ord,
        "start_page": min(pages) if pages else None,
        "end_page": max(pages) if pages else None,
    }]


def find_chapter(chapters: list[dict], chunk_ord: int) -> dict | None:
    """Linear scan to find the chapter containing `chunk_ord`."""
    for ch in chapters:
        if ch["start_ord"] <= chunk_ord <= ch["end_ord"]:
            return ch
    return None

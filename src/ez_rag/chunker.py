"""Recursive text chunker. Word-token approximation: 1 token ≈ 0.75 words."""
from __future__ import annotations

from dataclasses import dataclass
from .parsers import ParsedSection


@dataclass
class Chunk:
    text: str
    page: int | None = None
    section: str = ""
    ord: int = 0


_SPLITTERS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]


def _approx_word_target(token_target: int) -> int:
    return max(64, int(token_target * 0.75))


def _split_recursive(text: str, max_words: int, splitters: list[str]) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    if not splitters:
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    sep, rest = splitters[0], splitters[1:]
    parts = text.split(sep)
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in parts:
        n = len(p.split())
        if buf_len + n > max_words and buf:
            joined = sep.join(buf)
            if len(joined.split()) > max_words:
                out.extend(_split_recursive(joined, max_words, rest))
            else:
                out.append(joined)
            buf = [p]
            buf_len = n
        else:
            buf.append(p)
            buf_len += n
    if buf:
        joined = sep.join(buf)
        if len(joined.split()) > max_words:
            out.extend(_split_recursive(joined, max_words, rest))
        else:
            out.append(joined)
    return [s.strip() for s in out if s.strip()]


def _split_table(text: str, max_words: int) -> list[str]:
    """Structure-aware split for table sections (XLSX/CSV): never break
    mid-row. Rows are packed up to the word target, and the first row
    (usually the column header) is repeated at the top of every
    continuation piece so each chunk stays self-describing."""
    rows = [r for r in text.split("\n") if r.strip()]
    if not rows:
        return []
    header = rows[0]
    header_words = len(header.split())
    pieces: list[str] = []
    buf: list[str] = [header]
    buf_words = header_words
    for row in rows[1:]:
        n = len(row.split())
        if buf_words + n > max_words and len(buf) > 1:
            pieces.append("\n".join(buf))
            buf = [header, row]          # carry header into next piece
            buf_words = header_words + n
        else:
            buf.append(row)
            buf_words += n
    if len(buf) > 1 or not pieces:
        pieces.append("\n".join(buf))
    return pieces


def chunk_sections(
    sections: list[ParsedSection],
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    max_words = _approx_word_target(chunk_tokens)
    # NOTE: no minimum floor on overlap — overlap_tokens=0 must mean zero.
    # (_approx_word_target's max(64, …) floor previously forced 64 words
    # of overlap even when the user asked for none.)
    overlap_words = max(0, int(overlap_tokens * 0.75))
    chunks: list[Chunk] = []
    ord_ = 0
    for sec in sections:
        sec_text = (sec.text or "").strip()
        if not sec_text:
            continue
        is_table = (sec.meta or {}).get("kind") == "table"
        if is_table:
            # Row-atomic packing with header carry; overlap would just
            # duplicate rows, so tables get none.
            pieces = _split_table(sec_text, max_words)
        else:
            pieces = _split_recursive(sec_text, max_words, _SPLITTERS)
        for i, piece in enumerate(pieces):
            if not piece.strip():
                continue
            text = piece
            if not is_table and i > 0 and overlap_words > 0:
                prev_tail = " ".join(pieces[i - 1].split()[-overlap_words:])
                text = (prev_tail + " " + piece).strip()
            chunks.append(Chunk(text=text, page=sec.page, section=sec.section, ord=ord_))
            ord_ += 1
    return chunks

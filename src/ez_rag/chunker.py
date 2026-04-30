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


def chunk_sections(
    sections: list[ParsedSection],
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    max_words = _approx_word_target(chunk_tokens)
    overlap_words = _approx_word_target(overlap_tokens)
    chunks: list[Chunk] = []
    ord_ = 0
    for sec in sections:
        pieces = _split_recursive(sec.text, max_words, _SPLITTERS)
        # Re-add overlap by prepending the tail of the previous piece in the same section.
        for i, piece in enumerate(pieces):
            text = piece
            if i > 0 and overlap_words > 0:
                prev_tail = " ".join(pieces[i - 1].split()[-overlap_words:])
                text = (prev_tail + " " + piece).strip()
            chunks.append(Chunk(text=text, page=sec.page, section=sec.section, ord=ord_))
            ord_ += 1
    return chunks

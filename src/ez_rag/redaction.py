"""Context-aware redaction at ingest time.

Removes configured terms (names, emails, IDs) from chunk text BEFORE
anything derived from it exists — so the redacted data never reaches
the index text, the FTS tokens, or the embedding vectors, and
therefore never reaches a distributed export.

THE AMBIGUITY PROBLEM (why "context-aware"):

A surname like "Stone" is also an ordinary word. Blind
case-insensitive replacement would mangle "the crew hauled crushed
stone" into "the miners drilled blast [REDACTED]". The rules:

  - **Multi-word phrases** ("Casey Stone") and terms containing
    @ / digits (emails, phone numbers, IDs) are unambiguous →
    every occurrence is redacted, case-insensitive.
  - **Name variants are generated automatically** from multi-word
    person names: "Casey Stone" also matches "Stone, Casey",
    "C. Stone", and "Casey S." — the forms documents actually use.
  - **Single tokens** ("Stone" alone in the term list) use SMART
    CASING: capitalized occurrences are redacted (that's how names
    appear, including at sentence start — we bias toward
    over-redaction there because a false redaction is annoying but a
    leak is unacceptable), while lowercase occurrences are kept
    (that's the common-noun usage). Set `redact_smart = false` to
    redact every occurrence regardless of case.

Redaction happens in `ingest` after chunk headers are applied (headers
can contain a document title carrying the name) and is folded into the
effective chunker version, so editing the term list re-ingests
affected files automatically.

WHAT THIS CANNOT COVER (surfaced, not hidden):
  - File NAMES: if a term appears in a filename, the path is stored in
    the index and shown in citations. Ingest emits a warning telling
    you to rename the file — we won't rename your documents.
  - Original documents in docs/: they keep the term by definition.
    `export_chatbot` therefore refuses `include_sources` when redact
    terms are configured, and verifies the INDEX is clean before
    bundling it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RedactionResult:
    text: str
    n_redacted: int


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Case-insensitive, boundary-guarded, whitespace-tolerant phrase.

    Uses lookarounds instead of \b because \b misfires next to
    punctuation: "Casey S." + trailing \b can never match
    "Casey S., et al." (no word character follows the dot)."""
    parts = [re.escape(p) for p in phrase.split()]
    return re.compile(
        r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)",
        re.IGNORECASE)


def _variants_for_name(term: str) -> list[str]:
    """Generate the forms documents actually use for a person name."""
    words = term.split()
    if len(words) != 2:
        return []
    first, last = words
    out = [
        f"{last}, {first}",                 # Stone, Casey
        f"{first[0]}. {last}",              # C. Stone
        f"{first} {last[0]}.",              # Casey S.
    ]
    return out


def compile_matchers(terms: list[str], smart: bool = True):
    """Build [(pattern, kind)] for a term list.

    kind: "always" (case-insensitive full redaction) or "cap-only"
    (smart casing for ambiguous single tokens)."""
    matchers: list[tuple[re.Pattern, str]] = []
    for raw in terms:
        term = (raw or "").strip()
        if not term:
            continue
        is_single = " " not in term
        unambiguous = ("@" in term) or any(ch.isdigit() for ch in term)
        if not is_single or unambiguous or not smart:
            matchers.append((_phrase_pattern(term), "always"))
            for var in _variants_for_name(term):
                matchers.append((_phrase_pattern(var), "always"))
        else:
            # Ambiguous lone token: redact Capitalized / ALLCAPS forms,
            # keep lowercase (common-noun usage).
            cap = term[0].upper() + term[1:].lower()
            allcaps = term.upper()
            matchers.append((
                re.compile(
                    r"\b(?:" + re.escape(cap) + r"|"
                    + re.escape(allcaps) + r")\b"
                ),
                "cap-only",
            ))
    return matchers


def redact_text(text: str, matchers, replacement: str = "[REDACTED]"
                ) -> RedactionResult:
    if not text or not matchers:
        return RedactionResult(text, 0)
    n = 0
    for pattern, _kind in matchers:
        text, k = pattern.subn(replacement, text)
        n += k
    return RedactionResult(text, n)


def terms_fingerprint(terms: list[str], smart: bool = True) -> str:
    """Short stable hash of the redaction config — folded into the
    effective chunker version so term changes re-ingest files."""
    import hashlib
    norm = "\x00".join(sorted(t.strip().lower() for t in terms if t.strip()))
    norm += f"\x01smart={bool(smart)}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:8]


def scan_index_for_terms(db_path, terms: list[str],
                          smart: bool = True) -> dict[str, int]:
    """Count occurrences of each term in the index's chunk text.

    `smart=True` applies the SAME matching rules as redaction itself
    (ambiguous lone tokens count only in capitalized form) — this is
    what the export gate uses, so a corpus where "crushed stone" was
    deliberately preserved still verifies clean.

    `smart=False` is the aggressive advisory mode used by
    `ez-rag redact-check`: case-insensitive everywhere, so a human can
    review possible common-noun hits. Returns {term: hits}."""
    import sqlite3
    hits: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT text FROM chunks").fetchall()
    finally:
        conn.close()
    for raw in terms:
        term = (raw or "").strip()
        if not term:
            continue
        pats = [p for p, _kind in compile_matchers([term], smart=smart)]
        count = 0
        for (text,) in rows:
            for p in pats:
                count += len(p.findall(text or ""))
        if count:
            hits[term] = count
    return hits


def filename_warnings(paths: list[str], terms: list[str]) -> list[str]:
    """Filenames containing a redact term — the index stores paths and
    citations display them, so these leak regardless of text redaction."""
    out = []
    for raw in terms:
        term = (raw or "").strip().lower()
        if not term:
            continue
        for p in paths:
            hay = str(p).lower()
            if all(w in hay for w in term.split()):
                out.append(
                    f"'{term}' appears in filename '{p}' — rename the "
                    f"file; paths are stored in the index and shown in "
                    f"citations."
                )
    return out

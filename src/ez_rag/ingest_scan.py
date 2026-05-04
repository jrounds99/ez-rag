"""LLM-driven discovery scan that auto-populates ingest metadata.

Pipeline (per file):
  1. Parse the document via the existing parser registry.
  2. Stratified sampling — first 2 chunks (TOC/intro), uniform middle,
     last 2 chunks (index/glossary). Default ~12 chunks.
  3. Topic & summary pass: 1 LLM call. Asks the model to produce a
     {title, description, topics} JSON.
  4. Entity extraction: batched LLM calls over the chunks asking for
     {npcs, classes, items, locations, factions, spells, monsters,
      custom_terms} per batch. Deduplicated case-insensitively.
  5. Consolidation: drop entries that don't appear verbatim in any
     sampled chunk (cheap hallucination filter), cap by frequency.
  6. Build a FileMetadata with the user's choice of scope and any
     pre-existing modifiers preserved.
  7. Save as `<file>.ezrag-meta.toml.draft` so the admin reviews
     before it goes live. The user renames `.draft` → final to
     activate.

The LLM-call helpers all have small max_tokens budgets so a 200-page
PDF scan doesn't run for hours. Total LLM cost per file:
  - Sampling pass: 0 calls
  - Summary pass: 1 short call (~120 tokens)
  - Entity passes: ceil(samples / batch_size) calls (~250 tokens each)
At default settings (12 sample chunks, batch_size=4) that's 4 calls
per file. With qwen2.5:7b → ~25 s per file. A 25-book corpus → ~10 min.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .ingest_meta import (
    SCOPE_TOPIC_AWARE, FileMetadata, FileMetadataEntities,
    primary_sidecar_path, render_toml,
)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_SAMPLE_CHUNKS = 12
DEFAULT_ENTITY_BATCH_SIZE = 4

# Per-pass token budgets — kept small so total scan cost stays bounded.
TOPIC_PASS_MAX_TOKENS = 300
ENTITY_PASS_MAX_TOKENS = 400


# ============================================================================
# Prompts
# ============================================================================

_TOPIC_PROMPT = """You are reading SAMPLED EXCERPTS from a document.

Your job: produce a SHORT factual summary as JSON with three fields:
  - "title":       short title (≤80 chars)
  - "description": 1–2 sentence description (≤200 chars)
  - "topics":      list of 4–8 short topic strings the document covers

Output ONLY valid JSON. No prose, no preamble, no code fences.

EXCERPTS:
{excerpts}

JSON:"""


_ENTITY_PROMPT = """You are extracting NAMED ENTITIES from text excerpts.

Find the proper nouns and capitalized named items present in these
EXCERPTS. Sort each into the right bucket. If a bucket has no entries,
output an empty list.

Buckets:
  - npcs:         named characters / people (excluding the reader)
  - classes:      character classes / archetypes / professions
  - items:        named magic items, artifacts, equipment
  - locations:    cities, regions, dungeons, named places
  - factions:     organizations, guilds, governments, cults
  - spells:       named spells / abilities / techniques
  - monsters:     named creature types
  - custom_terms: any other domain-specific terminology that
                  doesn't fit the above buckets

Rules:
  - Use the EXACT name as it appears in the text.
  - Do NOT invent entities. If the text doesn't mention any, return
    empty lists.
  - Capitalized words at the start of a sentence are NOT entities
    unless they're proper nouns.

Output ONLY valid JSON with all eight keys. No prose, no preamble.

EXCERPTS:
{excerpts}

JSON:"""


# ============================================================================
# Helpers
# ============================================================================

@dataclass
class ChunkSample:
    text: str
    page: Optional[int] = None
    section: str = ""


def _stratified_sample(chunks: list, count: int) -> list:
    """Pick `count` chunks across the document — start, end, evenly
    distributed in the middle. `chunks` is whatever the parser returns
    (objects with a `text` attribute)."""
    n = len(chunks)
    if n <= count:
        return list(chunks)
    if count < 4:
        # Trivial uniform sampling for very small counts
        step = n / count
        return [chunks[int(i * step)] for i in range(count)]

    head = list(chunks[:2])
    tail = list(chunks[-2:])
    middle_count = count - 4
    if middle_count <= 0:
        return head + tail
    middle_pool = chunks[2:-2]
    if not middle_pool:
        return head + tail
    step = max(1, len(middle_pool) // middle_count)
    middle = [middle_pool[i] for i in range(0, len(middle_pool), step)]
    middle = middle[:middle_count]
    return head + middle + tail


def _extract_text(chunk) -> str:
    """ParsedSection / Chunk both have .text. Use that."""
    return (getattr(chunk, "text", "") or "").strip()


def _format_excerpts(samples: list[ChunkSample], max_chars_per: int = 600
                     ) -> str:
    """Build the EXCERPTS block fed to the LLM."""
    out: list[str] = []
    for i, s in enumerate(samples, start=1):
        text = s.text.strip()
        if len(text) > max_chars_per:
            text = text[:max_chars_per] + "…"
        page_tag = f" (p.{s.page})" if s.page else ""
        out.append(f"[{i}{page_tag}] {text}")
    return "\n\n".join(out)


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_safely(raw: str) -> dict | None:
    """Tolerant JSON parse. Strips fences, finds the first {...} block,
    returns None on any failure."""
    if not raw:
        return None
    raw = _strip_code_fences(raw)
    # Find the first balanced {...} block — LLMs occasionally append
    # explanatory prose after the JSON.
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(raw)):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    blob = raw[start:end]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _consolidate_entities(buckets_per_batch: list[dict],
                            sampled_text: str,
                            ) -> FileMetadataEntities:
    """Merge entities from N batched LLM calls + drop hallucinations.

    A "hallucination" here = an entity name that doesn't appear
    verbatim (case-insensitive) in any sampled chunk. Keeps the
    sidecar honest.
    """
    sampled_lower = sampled_text.lower()
    seen_per_bucket: dict[str, dict[str, int]] = {}
    valid_buckets = (
        "npcs", "classes", "items", "locations", "factions",
        "spells", "monsters", "custom_terms",
    )
    for batch in buckets_per_batch:
        if not isinstance(batch, dict):
            continue
        for key in valid_buckets:
            entries = batch.get(key) or []
            if not isinstance(entries, list):
                continue
            counter = seen_per_bucket.setdefault(key, {})
            for raw_entry in entries:
                if not isinstance(raw_entry, str):
                    continue
                term = raw_entry.strip()
                if not term or len(term) < 2:
                    continue
                # Hallucination filter: must appear in the sampled text
                if term.lower() not in sampled_lower:
                    continue
                # Frequency-aware dedup (case-insensitive)
                key_low = term.lower()
                if key_low in counter:
                    counter[key_low] = (counter[key_low][0] + 1,
                                          counter[key_low][1])
                else:
                    counter[key_low] = (1, term)

    def _by_freq(d: dict) -> list[str]:
        # Sort by frequency desc, original-form alphabetically as tiebreak
        items = sorted(
            d.items(),
            key=lambda kv: (-kv[1][0], kv[1][1].lower()),
        )
        return [v[1] for _, v in items][:30]   # cap per-bucket at 30

    return FileMetadataEntities(
        npcs=_by_freq(seen_per_bucket.get("npcs", {})),
        classes=_by_freq(seen_per_bucket.get("classes", {})),
        items=_by_freq(seen_per_bucket.get("items", {})),
        locations=_by_freq(seen_per_bucket.get("locations", {})),
        factions=_by_freq(seen_per_bucket.get("factions", {})),
        spells=_by_freq(seen_per_bucket.get("spells", {})),
        monsters=_by_freq(seen_per_bucket.get("monsters", {})),
        custom_terms=_by_freq(seen_per_bucket.get("custom_terms", {})),
    )


# ============================================================================
# Public API
# ============================================================================

def scan_file(file_path: Path, cfg: Config,
              *, sample_chunks: int = DEFAULT_SAMPLE_CHUNKS,
              entity_batch_size: int = DEFAULT_ENTITY_BATCH_SIZE,
              progress_cb=None,
              ) -> FileMetadata:
    """Parse, sample, and run the LLM passes. Returns a populated
    FileMetadata. Caller decides whether to save it.

    `progress_cb(stage_id, payload)` is fired at each step so callers
    can show progress. Stages: "parse" / "sample" / "topic_pass" /
    "entity_pass" / "consolidate".

    Never raises — failures fall through with empty fields and a
    warning printed to stderr.
    """
    from .parsers import get_parser
    from .generate import _llm_complete, detect_backend

    file_path = Path(file_path)
    if progress_cb:
        progress_cb("parse", {"path": str(file_path)})
    parser = get_parser(file_path)
    if parser is None:
        return FileMetadata(file_path=file_path.name)
    try:
        sections = parser(file_path)
    except Exception as ex:
        sys.stderr.write(f"[scan] parse failed for {file_path}: {ex}\n")
        return FileMetadata(file_path=file_path.name)

    # Filter to non-empty text sections
    sections = [s for s in sections if _extract_text(s)]
    if not sections:
        return FileMetadata(file_path=file_path.name)

    # Stratified sampling
    sampled_raw = _stratified_sample(sections, sample_chunks)
    samples = [
        ChunkSample(
            text=_extract_text(s),
            page=getattr(s, "page", None),
            section=getattr(s, "section", "") or "",
        )
        for s in sampled_raw
    ]
    if progress_cb:
        progress_cb("sample", {"chunks": len(samples)})

    backend = detect_backend(cfg)
    md = FileMetadata(
        file_path=file_path.name,
        last_scanned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        last_scanned_by=cfg.llm_model,
        scope=SCOPE_TOPIC_AWARE,
    )
    if backend == "none":
        sys.stderr.write(
            "[scan] no LLM backend reachable — sidecar will have "
            "samples but no LLM-derived fields.\n"
        )
        return md

    excerpts = _format_excerpts(samples)

    # Topic & summary pass
    if progress_cb:
        progress_cb("topic_pass", {})
    topic_raw = _llm_complete(
        cfg, _TOPIC_PROMPT.format(excerpts=excerpts),
        max_tokens=TOPIC_PASS_MAX_TOKENS,
    ) or ""
    topic_obj = _parse_json_safely(topic_raw) or {}
    md.title = str(topic_obj.get("title", "") or "")[:200]
    md.description = str(topic_obj.get("description", "") or "")[:500]
    raw_topics = topic_obj.get("topics") or []
    if isinstance(raw_topics, list):
        md.detected_topics = [
            str(t).strip() for t in raw_topics
            if isinstance(t, str) and t.strip()
        ][:10]

    # Entity extraction passes (batched)
    batch_results: list[dict] = []
    sampled_text = "\n".join(s.text for s in samples)
    for batch_start in range(0, len(samples), entity_batch_size):
        batch = samples[batch_start: batch_start + entity_batch_size]
        if progress_cb:
            progress_cb("entity_pass", {
                "batch_start": batch_start,
                "batch_size": len(batch),
            })
        ex = _format_excerpts(batch, max_chars_per=400)
        raw = _llm_complete(
            cfg, _ENTITY_PROMPT.format(excerpts=ex),
            max_tokens=ENTITY_PASS_MAX_TOKENS,
        ) or ""
        obj = _parse_json_safely(raw)
        if obj:
            batch_results.append(obj)

    if progress_cb:
        progress_cb("consolidate", {"batches": len(batch_results)})
    md.entities = _consolidate_entities(batch_results, sampled_text)
    return md


def scan_and_save(file_path: Path, cfg: Config,
                    *, workspace_root: Optional[Path] = None,
                    suffix: str = ".draft",
                    prefer_workspace: bool = False,
                    progress_cb=None,
                    ) -> Path:
    """Scan + write a `.draft` sidecar that the admin reviews. Returns
    the saved path."""
    md = scan_file(file_path, cfg, progress_cb=progress_cb)
    target = primary_sidecar_path(
        file_path, workspace_root, prefer_workspace=prefer_workspace,
    )
    if suffix:
        target = target.with_suffix(target.suffix + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(render_toml(md), encoding="utf-8")
    import os
    os.replace(tmp, target)
    return target

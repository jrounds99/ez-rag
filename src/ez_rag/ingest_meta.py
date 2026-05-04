"""Per-file ingest metadata.

A sidecar TOML file alongside every document carrying:
  - LLM-discovered (or hand-curated) entities, topics, summary
  - Per-file query_prefix / query_suffix / query_negatives that
    override / augment the workspace-level ones at retrieval time
  - A `scope` flag that decides WHEN those modifiers fire

The sidecar lives at one of:
  <docs_dir>/<filename>.ezrag-meta.toml          # alongside the source
  <workspace>/.ezrag/file_meta/<sha-prefix>.toml # in the workspace

Lookup tries both, alongside-source first. Read-only at retrieval
time; write happens at ingest-scan time (`ingest_scan.py`).

Tolerant parsing — a malformed file is treated as "no metadata"
rather than crashing the retriever. The sidecar is purely additive;
ez-rag works fine on workspaces without any metadata files.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


# ============================================================================
# Constants
# ============================================================================

SIDECAR_SUFFIX = ".ezrag-meta.toml"
DRAFT_SUFFIX = ".ezrag-meta.toml.draft"

# Where workspace-scoped metadata lives (when the user can't or
# doesn't want to write next to the source PDF).
WORKSPACE_META_SUBDIR = ".ezrag/file_meta"

# Scope values for per-file modifiers.
SCOPE_GLOBAL = "global"          # apply to every query
SCOPE_TOPIC_AWARE = "topic-aware"  # apply when query embeds near a topic
SCOPE_FILE_ONLY = "file-only"    # apply only when this file is top-1


# ============================================================================
# Data model
# ============================================================================

@dataclass
class FileMetadataEntities:
    """Buckets of named things the user (or LLM) thinks are queryable."""
    characters: list[str] = field(default_factory=list)
    npcs: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    factions: list[str] = field(default_factory=list)
    spells: list[str] = field(default_factory=list)
    monsters: list[str] = field(default_factory=list)
    custom_terms: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        """Flat unique list of every entity, deduplicated case-insensitively."""
        seen: set[str] = set()
        out: list[str] = []
        for bucket in (self.characters, self.npcs, self.classes, self.items,
                       self.locations, self.factions, self.spells,
                       self.monsters, self.custom_terms):
            for term in bucket:
                t = term.strip()
                if not t:
                    continue
                key = t.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
        return out


@dataclass
class FileMetadata:
    """One sidecar's worth of metadata."""
    schema_version: int = 1
    file_path: str = ""             # rel-to-workspace, populated by writer
    file_sha256: str = ""           # of the source bytes; "" if unknown
    last_scanned_at: str = ""
    last_scanned_by: str = ""

    title: str = ""
    description: str = ""
    detected_topics: list[str] = field(default_factory=list)

    entities: FileMetadataEntities = field(default_factory=FileMetadataEntities)

    query_prefix: str = ""
    query_suffix: str = ""
    query_negatives: list[str] = field(default_factory=list)

    scope: str = SCOPE_TOPIC_AWARE     # global | topic-aware | file-only

    # Optional retrieval-time score boost applied when entities match.
    entity_match_boost: float = 1.10
    priority_terms: list[str] = field(default_factory=list)
    priority_term_match_boost: float = 1.20

    # Free-form notes (kept by the user; ez-rag doesn't read).
    notes: str = ""

    def has_modifiers(self) -> bool:
        return bool(self.query_prefix
                    or self.query_suffix
                    or self.query_negatives)


# ============================================================================
# Sidecar path resolution
# ============================================================================

def sidecar_paths_for(file_path: Path,
                      workspace_root: Optional[Path] = None,
                      ) -> list[Path]:
    """Return candidate sidecar paths in lookup order.

    1. `<file_path>.ezrag-meta.toml`  — alongside the source
    2. `<workspace>/.ezrag/file_meta/<sha-prefix>.toml`  — workspace-scoped
    """
    file_path = Path(file_path)
    candidates: list[Path] = [Path(str(file_path) + SIDECAR_SUFFIX)]
    if workspace_root is not None:
        ws_meta_dir = Path(workspace_root) / WORKSPACE_META_SUBDIR
        # Stable hash from the absolute path so workspace-relative moves
        # don't lose the metadata.
        digest = hashlib.sha256(
            str(file_path.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        candidates.append(ws_meta_dir / f"{digest}.toml")
    return candidates


def find_sidecar(file_path: Path,
                  workspace_root: Optional[Path] = None,
                  ) -> Optional[Path]:
    """Return the first sidecar path that exists, or None."""
    for c in sidecar_paths_for(file_path, workspace_root):
        if c.is_file():
            return c
    return None


def primary_sidecar_path(file_path: Path,
                          workspace_root: Optional[Path] = None,
                          *, prefer_workspace: bool = False,
                          ) -> Path:
    """Where a NEW sidecar should be written. Caller picks whether
    they want it next to the source or in the workspace tree."""
    candidates = sidecar_paths_for(file_path, workspace_root)
    if prefer_workspace and len(candidates) > 1:
        return candidates[1]
    return candidates[0]


# ============================================================================
# Read
# ============================================================================

def parse_toml(text: str) -> FileMetadata:
    """Parse one sidecar TOML. Tolerant — missing fields use defaults,
    bad types fall back, never raises."""
    if not text or not text.strip():
        return FileMetadata()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return FileMetadata()

    def _str(v, default=""):
        return str(v) if isinstance(v, str) else default

    def _int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _list_str(v):
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    md = FileMetadata(
        schema_version=_int(data.get("schema_version", 1), 1),
        file_path=_str(data.get("file_path")),
        file_sha256=_str(data.get("file_sha256")),
        last_scanned_at=_str(data.get("last_scanned_at")),
        last_scanned_by=_str(data.get("last_scanned_by")),
        notes=_str(data.get("notes")),
    )

    summary = data.get("summary") or {}
    if isinstance(summary, dict):
        md.title = _str(summary.get("title"))
        md.description = _str(summary.get("description"))
        md.detected_topics = _list_str(summary.get("detected_topics"))

    ents = data.get("entities") or {}
    if isinstance(ents, dict):
        md.entities = FileMetadataEntities(
            characters=_list_str(ents.get("characters")),
            npcs=_list_str(ents.get("npcs")),
            classes=_list_str(ents.get("classes")),
            items=_list_str(ents.get("items")),
            locations=_list_str(ents.get("locations")),
            factions=_list_str(ents.get("factions")),
            spells=_list_str(ents.get("spells")),
            monsters=_list_str(ents.get("monsters")),
            custom_terms=_list_str(ents.get("custom_terms")),
        )

    mods = data.get("modifiers") or {}
    if isinstance(mods, dict):
        md.query_prefix = _str(mods.get("query_prefix"))
        md.query_suffix = _str(mods.get("query_suffix"))
        # Negatives accepted as either a list OR a single comma-separated string
        raw_neg = mods.get("query_negatives")
        if isinstance(raw_neg, str):
            md.query_negatives = [t.strip() for t in raw_neg.split(",")
                                   if t.strip()]
        else:
            md.query_negatives = _list_str(raw_neg)

    scope_section = data.get("scope") or {}
    if isinstance(scope_section, dict):
        applies = _str(scope_section.get("applies"), SCOPE_TOPIC_AWARE)
    else:
        applies = SCOPE_TOPIC_AWARE
    if applies not in (SCOPE_GLOBAL, SCOPE_TOPIC_AWARE, SCOPE_FILE_ONLY):
        applies = SCOPE_TOPIC_AWARE
    md.scope = applies

    boost = data.get("boost") or {}
    if isinstance(boost, dict):
        md.entity_match_boost = _float(
            boost.get("entity_match_boost", 1.10), 1.10,
        )
        md.priority_term_match_boost = _float(
            boost.get("priority_term_match_boost", 1.20), 1.20,
        )
        md.priority_terms = _list_str(boost.get("priority_terms"))

    return md


def load(file_path: Path,
         workspace_root: Optional[Path] = None,
         ) -> Optional[FileMetadata]:
    """Load the sidecar for a file, or return None if it doesn't exist."""
    side = find_sidecar(file_path, workspace_root)
    if side is None:
        return None
    try:
        return parse_toml(side.read_text(encoding="utf-8"))
    except OSError:
        return None


# ============================================================================
# Write
# ============================================================================

def _toml_str(v: str) -> str:
    """Render a string for TOML. Single-quoted literal preserves
    backslashes; double-quoted with escapes when value contains a
    single quote."""
    if "'" in v:
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return f"'{v}'"


def render_toml(md: FileMetadata) -> str:
    """Hand-editable, comment-rich TOML output."""
    lines: list[str] = [
        "# ez-rag per-file metadata sidecar.",
        "# Hand-editable. Re-read at every retrieval — no need to re-ingest "
        "after editing.",
        "",
        f"schema_version = {md.schema_version}",
        f"file_path = {_toml_str(md.file_path)}",
        f"file_sha256 = {_toml_str(md.file_sha256)}",
        f"last_scanned_at = {_toml_str(md.last_scanned_at)}",
        f"last_scanned_by = {_toml_str(md.last_scanned_by)}",
    ]
    if md.notes:
        lines.append(f"notes = {_toml_str(md.notes)}")
    lines.append("")

    # [summary]
    lines.append("[summary]")
    lines.append(f"title = {_toml_str(md.title)}")
    lines.append(f"description = {_toml_str(md.description)}")
    lines.append("detected_topics = [")
    for t in md.detected_topics:
        lines.append(f"  {_toml_str(t)},")
    lines.append("]")
    lines.append("")

    # [entities]
    lines.append("[entities]")
    for bucket_name in ("characters", "npcs", "classes", "items", "locations",
                        "factions", "spells", "monsters", "custom_terms"):
        bucket = getattr(md.entities, bucket_name)
        if not bucket:
            lines.append(f"{bucket_name} = []")
        else:
            lines.append(f"{bucket_name} = [")
            for t in bucket:
                lines.append(f"  {_toml_str(t)},")
            lines.append("]")
    lines.append("")

    # [modifiers]
    lines.append("[modifiers]")
    lines.append(f"query_prefix = {_toml_str(md.query_prefix)}")
    lines.append(f"query_suffix = {_toml_str(md.query_suffix)}")
    lines.append("query_negatives = [")
    for n in md.query_negatives:
        lines.append(f"  {_toml_str(n)},")
    lines.append("]")
    lines.append("")

    # [scope]
    lines.append("[scope]")
    lines.append("# global: modifiers apply to every query")
    lines.append("# topic-aware: only when query embeds near a topic")
    lines.append("# file-only: only when this file is the top-1 source")
    lines.append(f"applies = {_toml_str(md.scope)}")
    lines.append("")

    # [boost]
    lines.append("[boost]")
    lines.append(f"entity_match_boost = {md.entity_match_boost}")
    lines.append(f"priority_term_match_boost = {md.priority_term_match_boost}")
    lines.append("priority_terms = [")
    for t in md.priority_terms:
        lines.append(f"  {_toml_str(t)},")
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def save(md: FileMetadata,
         file_path: Path,
         *, workspace_root: Optional[Path] = None,
         prefer_workspace: bool = False,
         ) -> Path:
    """Write the sidecar. Returns the path written."""
    out = primary_sidecar_path(
        file_path, workspace_root, prefer_workspace=prefer_workspace,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(render_toml(md), encoding="utf-8")
    import os
    os.replace(tmp, out)
    return out


# ============================================================================
# Aggregation helper used by the retriever
# ============================================================================

@dataclass
class MergedModifiers:
    """The product of merging workspace-level + per-file modifiers
    for ONE query. Returned by `merged_modifiers_for_hits`."""
    prefix: str = ""
    suffix: str = ""
    negatives: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # files that contributed


def merged_modifiers_for_hits(
    *,
    workspace_prefix: str,
    workspace_suffix: str,
    workspace_negatives: str,
    hits: list,
    workspace_root: Optional[Path] = None,
    query: str = "",
) -> MergedModifiers:
    """Build the effective query modifiers for one retrieval result.

    Merge order:
      1. Workspace-wide modifiers (always)
      2. Per-file modifiers, filtered by scope:
         - global: always
         - topic-aware: at least one of the file's detected_topics
                        appears literally in the query (cheap heuristic)
         - file-only: only when this hit's file is the top-1 source

    Negatives are concatenated and deduplicated (case-insensitive).
    Prefixes/suffixes from multiple files are concatenated with " "
    separator, capped at ~200 chars total to keep the embedded query
    sane.

    `hits` is whatever smart_retrieve returns (Hit-like objects with
    `.path` and `.file_id`). Caller guarantees they're in rank order.
    """
    merged = MergedModifiers(
        prefix=(workspace_prefix or "").strip(),
        suffix=(workspace_suffix or "").strip(),
    )
    if workspace_negatives:
        merged.negatives.extend(
            t.strip() for t in str(workspace_negatives).split(",")
            if t.strip()
        )

    if not hits:
        return merged

    seen_paths: set[str] = set()
    for i, h in enumerate(hits):
        path = getattr(h, "path", None)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        # Resolve to filesystem path. Hit.path is workspace-relative.
        if workspace_root is not None:
            file_path = (Path(workspace_root) / path).resolve()
        else:
            file_path = Path(path)
        md = load(file_path, workspace_root=workspace_root)
        if md is None:
            continue

        # Scope check
        applies = False
        if md.scope == SCOPE_GLOBAL:
            applies = True
        elif md.scope == SCOPE_FILE_ONLY:
            applies = (i == 0)
        elif md.scope == SCOPE_TOPIC_AWARE:
            # Cheap check: does any topic appear literally in the query?
            ql = (query or "").lower()
            for t in md.detected_topics:
                if t and t.lower() in ql:
                    applies = True
                    break
        if not applies:
            continue

        if md.query_prefix and md.query_prefix not in merged.prefix:
            merged.prefix = (merged.prefix + " " + md.query_prefix).strip()
        if md.query_suffix and md.query_suffix not in merged.suffix:
            merged.suffix = (merged.suffix + " " + md.query_suffix).strip()
        for n in md.query_negatives:
            n = n.strip()
            if not n:
                continue
            if not any(n.lower() == x.lower() for x in merged.negatives):
                merged.negatives.append(n)
        merged.sources.append(path)

    # Cap the prefix / suffix so we don't bloat the embedded query.
    if len(merged.prefix) > 200:
        merged.prefix = merged.prefix[:200]
    if len(merged.suffix) > 200:
        merged.suffix = merged.suffix[:200]
    # Negatives capped at 5 to avoid prompt bloat
    merged.negatives = merged.negatives[:5]

    return merged


def apply_modifiers_to_query(query: str,
                              merged: MergedModifiers,
                              ) -> str:
    """Build the actual string that gets handed to the embedder.

    Format:
        <prefix> <query> <suffix>   Avoid: <neg1>, <neg2>, ...
    """
    parts: list[str] = []
    if merged.prefix:
        parts.append(merged.prefix)
    parts.append(query)
    if merged.suffix:
        parts.append(merged.suffix)
    out = " ".join(p for p in parts if p)
    if merged.negatives:
        out = out + "  Avoid: " + ", ".join(merged.negatives)
    return out

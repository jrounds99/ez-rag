"""Tests for ingest_meta — per-file metadata sidecars + retrieval injection.

Pure unit tests, no LLM calls. Cover:
  - FileMetadata + FileMetadataEntities construction + entity dedup
  - TOML round-trip (render → parse → equal)
  - Tolerant parsing of malformed / partial TOML
  - Sidecar path resolution (alongside-source first, workspace fallback)
  - merged_modifiers_for_hits applies scope rules correctly
  - apply_query_modifiers picks up global-scope sidecars
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.ingest_meta import (
    SCOPE_FILE_ONLY, SCOPE_GLOBAL, SCOPE_TOPIC_AWARE,
    SIDECAR_SUFFIX, WORKSPACE_META_SUBDIR, FileMetadata,
    FileMetadataEntities, MergedModifiers, apply_modifiers_to_query,
    find_sidecar, load, merged_modifiers_for_hits, parse_toml,
    primary_sidecar_path, render_toml, save, sidecar_paths_for,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


@dataclass
class FakeHit:
    """Mirrors the bits of retrieve.Hit that ingest_meta cares about."""
    path: str
    file_id: int = 0


def main():
    print("\n[1] FileMetadataEntities — flat all() with dedup")
    e = FileMetadataEntities(
        npcs=["Durnan", "DURNAN", "Vajra"],
        items=["Bag of Holding"],
        custom_terms=["bag of holding"],   # dup vs items, different case
    )
    flat = e.all()
    # Should preserve order, dedupe case-insensitively
    check("dedup order preserved",
          flat[0] == "Durnan" and flat[1] == "Vajra"
          and flat[2] == "Bag of Holding",
          f"got {flat}")
    check("case-insensitive dedup drops duplicates",
          len(flat) == 3, f"got {len(flat)}: {flat}")

    print("\n[2] empty TOML -> empty FileMetadata")
    md = parse_toml("")
    check("empty -> defaults",
          isinstance(md, FileMetadata)
          and md.title == ""
          and md.scope == SCOPE_TOPIC_AWARE)
    md_garbage = parse_toml("not toml at all {{{")
    check("garbage -> defaults (no crash)",
          isinstance(md_garbage, FileMetadata))

    print("\n[3] TOML round-trip preserves data")
    md = FileMetadata(
        file_path="docs/Player's Handbook.pdf",
        file_sha256="abcdef" * 4,
        last_scanned_at="2026-05-03T12:00:00Z",
        last_scanned_by="qwen2.5:7b",
        title="D&D 5e Player's Handbook",
        description="Core fantasy roleplaying rules.",
        detected_topics=["combat", "spells", "character creation"],
        entities=FileMetadataEntities(
            npcs=["Durnan", "Vajra Safahr"],
            classes=["Fighter", "Wizard", "Way of the Drunken Master"],
            items=["Bag of Holding"],
        ),
        query_prefix="In D&D 5e (2014):",
        query_suffix="Use 5e terminology.",
        query_negatives=["Pathfinder", "3.5e"],
        scope=SCOPE_TOPIC_AWARE,
        priority_terms=["Way of the Drunken Master"],
    )
    rendered = render_toml(md)
    parsed = parse_toml(rendered)
    check("title roundtrip", parsed.title == md.title)
    check("description roundtrip",
          parsed.description == md.description)
    check("detected_topics roundtrip",
          parsed.detected_topics == md.detected_topics)
    check("npcs roundtrip",
          parsed.entities.npcs == md.entities.npcs)
    check("classes roundtrip (with apostrophes / spaces)",
          parsed.entities.classes == md.entities.classes)
    check("query_prefix roundtrip",
          parsed.query_prefix == md.query_prefix)
    check("query_negatives roundtrip",
          parsed.query_negatives == md.query_negatives)
    check("scope roundtrip", parsed.scope == md.scope)
    check("priority_terms roundtrip",
          parsed.priority_terms == md.priority_terms)
    check("file_sha256 roundtrip",
          parsed.file_sha256 == md.file_sha256)

    print("\n[4] negatives accept comma-separated string OR list")
    csv_form = """
[modifiers]
query_negatives = "Pathfinder, 3.5e, 5.5e"
"""
    md = parse_toml(csv_form)
    check("comma-separated string parses to list",
          md.query_negatives == ["Pathfinder", "3.5e", "5.5e"])

    print("\n[5] tolerant parsing — bad fields don't crash")
    weird = """
schema_version = "junk"

[summary]
title = 42
detected_topics = "not a list"

[scope]
applies = "made-up-value"

[boost]
entity_match_boost = "ten"
"""
    md = parse_toml(weird)
    check("bad ints fall back to default",
          md.schema_version == 1)
    check("non-string title falls back to ''",
          md.title == "")
    check("non-list detected_topics falls back to []",
          md.detected_topics == [])
    check("invalid scope falls back to topic-aware",
          md.scope == SCOPE_TOPIC_AWARE)
    check("non-float boost falls back to 1.10",
          md.entity_match_boost == 1.10)

    print("\n[6] sidecar path resolution")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs_dir = ws / "docs"
        docs_dir.mkdir()
        src = docs_dir / "Player's Handbook.pdf"
        src.write_bytes(b"fake pdf")
        candidates = sidecar_paths_for(src, ws)
        check("alongside-source path is first candidate",
              candidates[0] == Path(str(src) + SIDECAR_SUFFIX))
        check("workspace path is second candidate",
              len(candidates) == 2
              and candidates[1].parent.name == "file_meta")

        # Nothing exists → find_sidecar returns None
        check("no sidecar -> None",
              find_sidecar(src, ws) is None)

        # Write alongside, find resolves to it
        alongside = candidates[0]
        alongside.write_text(render_toml(FileMetadata(title="X")),
                              encoding="utf-8")
        check("alongside sidecar found",
              find_sidecar(src, ws) == alongside)

        # Workspace path also works as a fallback when alongside
        # doesn't exist.
        alongside.unlink()
        ws_path = candidates[1]
        ws_path.parent.mkdir(parents=True, exist_ok=True)
        ws_path.write_text(render_toml(FileMetadata(title="W")),
                            encoding="utf-8")
        check("workspace sidecar found when no alongside",
              find_sidecar(src, ws) == ws_path)

        # Alongside takes priority over workspace
        alongside.write_text(render_toml(FileMetadata(title="A")),
                              encoding="utf-8")
        check("alongside takes priority over workspace",
              find_sidecar(src, ws) == alongside)

    print("\n[7] save / load round-trip (atomic write, no leftover .tmp)")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs_dir = ws / "docs"
        docs_dir.mkdir()
        src = docs_dir / "x.pdf"
        src.write_bytes(b"fake")
        md = FileMetadata(
            title="X",
            entities=FileMetadataEntities(npcs=["Alpha"]),
        )
        out = save(md, src, workspace_root=ws)
        check("save returned the alongside path",
              out == Path(str(src) + SIDECAR_SUFFIX))
        check("file written", out.is_file())
        check("no .tmp left",
              not list(out.parent.glob("*.tmp")))
        loaded = load(src, workspace_root=ws)
        check("loaded matches saved title",
              loaded is not None and loaded.title == "X")
        check("loaded matches saved entities",
              loaded.entities.npcs == ["Alpha"])

    print("\n[8] merged_modifiers_for_hits — global scope always applies")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs_dir = ws / "docs"
        docs_dir.mkdir()
        f = docs_dir / "phb.pdf"
        f.write_bytes(b"fake")
        save(
            FileMetadata(
                title="PHB",
                query_prefix="In D&D 5e:",
                query_negatives=["Pathfinder"],
                scope=SCOPE_GLOBAL,
            ),
            f, workspace_root=ws,
        )
        merged = merged_modifiers_for_hits(
            workspace_prefix="",
            workspace_suffix="",
            workspace_negatives="",
            hits=[FakeHit(path="docs/phb.pdf")],
            workspace_root=ws,
            query="anything at all",
        )
        check("global-scope prefix applied",
              "In D&D 5e:" in merged.prefix)
        check("global-scope negative applied",
              "Pathfinder" in merged.negatives)

    print("\n[9] merged_modifiers_for_hits — topic-aware fires only when topic in query")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs_dir = ws / "docs"
        docs_dir.mkdir()
        f = docs_dir / "combat.pdf"
        f.write_bytes(b"fake")
        save(
            FileMetadata(
                title="Combat",
                detected_topics=["grappling", "shoving"],
                query_suffix="Use combat-rules wording.",
                scope=SCOPE_TOPIC_AWARE,
            ),
            f, workspace_root=ws,
        )
        # Query mentions a topic — modifier fires
        merged = merged_modifiers_for_hits(
            workspace_prefix="", workspace_suffix="",
            workspace_negatives="",
            hits=[FakeHit(path="docs/combat.pdf")],
            workspace_root=ws,
            query="how does grappling work?",
        )
        check("topic-aware fires when topic in query",
              "Use combat-rules wording." in merged.suffix)
        # Query without the topic — modifier silent
        merged_silent = merged_modifiers_for_hits(
            workspace_prefix="", workspace_suffix="",
            workspace_negatives="",
            hits=[FakeHit(path="docs/combat.pdf")],
            workspace_root=ws,
            query="describe a wizard's spellbook",
        )
        check("topic-aware skipped when topic NOT in query",
              "Use combat-rules wording." not in merged_silent.suffix)

    print("\n[10] merged_modifiers_for_hits — file-only fires only at top-1")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs_dir = ws / "docs"
        docs_dir.mkdir()
        f = docs_dir / "exclusive.pdf"
        f.write_bytes(b"fake")
        save(
            FileMetadata(
                title="Excl",
                query_prefix="EXCLUSIVE_PREFIX",
                scope=SCOPE_FILE_ONLY,
            ),
            f, workspace_root=ws,
        )
        # File at top-1 — fires
        merged = merged_modifiers_for_hits(
            workspace_prefix="", workspace_suffix="",
            workspace_negatives="",
            hits=[
                FakeHit(path="docs/exclusive.pdf"),
                FakeHit(path="docs/other.pdf"),
            ],
            workspace_root=ws,
            query="anything",
        )
        check("file-only fires when file is top-1",
              "EXCLUSIVE_PREFIX" in merged.prefix)
        # File at rank 2 — silent
        merged_silent = merged_modifiers_for_hits(
            workspace_prefix="", workspace_suffix="",
            workspace_negatives="",
            hits=[
                FakeHit(path="docs/other.pdf"),
                FakeHit(path="docs/exclusive.pdf"),
            ],
            workspace_root=ws,
            query="anything",
        )
        check("file-only skipped when file isn't top-1",
              "EXCLUSIVE_PREFIX" not in merged_silent.prefix)

    print("\n[11] apply_modifiers_to_query format")
    merged = MergedModifiers(
        prefix="In D&D 5e:",
        suffix="Use 5e terminology.",
        negatives=["Pathfinder", "3.5e"],
    )
    out = apply_modifiers_to_query("how does grappling work?", merged)
    check("output contains prefix",
          "In D&D 5e:" in out)
    check("output contains query",
          "how does grappling work?" in out)
    check("output contains suffix",
          "Use 5e terminology." in out)
    check("output contains 'Avoid:' and negatives",
          "Avoid: Pathfinder, 3.5e" in out)

    print("\n[12] apply_query_modifiers integrates global-scope sidecars")
    from ez_rag.config import Config
    from ez_rag.generate import apply_query_modifiers
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs_dir = ws / "docs"
        docs_dir.mkdir()
        f = docs_dir / "any.pdf"
        f.write_bytes(b"fake")
        save(
            FileMetadata(
                title="Any",
                query_suffix="Use 5e terminology.",
                query_negatives=["Pathfinder"],
                scope=SCOPE_GLOBAL,
            ),
            f, workspace_root=ws,
        )
        cfg = Config()
        cfg.use_file_metadata = True
        cfg.apply_query_modifiers = True
        # Bypass the cache — the test creates a fresh dir per call,
        # so cache won't have a hit anyway, but be defensive.
        from ez_rag import generate
        generate._GLOBAL_SCOPE_CACHE.clear()

        out = apply_query_modifiers(
            "how does grappling work?", cfg, workspace_root=ws,
        )
        check("global-scope suffix injected into the query",
              "Use 5e terminology." in out, f"got {out!r}")
        check("global-scope negative injected",
              "Pathfinder" in out, f"got {out!r}")

    print("\n[13] _build_per_file_brief — global scope + topic-aware + file-only")
    from ez_rag.generate import _build_per_file_brief
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs = ws / "docs"
        docs.mkdir()
        # File A: global scope; should always appear
        a = docs / "phb.pdf"
        a.write_bytes(b"a")
        save(FileMetadata(
            file_path="docs/phb.pdf",
            title="Player's Handbook",
            description="Core rules.",
            detected_topics=["combat", "spells"],
            entities=FileMetadataEntities(classes=["Wizard", "Fighter"]),
            scope=SCOPE_GLOBAL,
        ), a, workspace_root=ws)

        # File B: topic-aware; needs the topic mentioned in question
        b = docs / "monsters.pdf"
        b.write_bytes(b"b")
        save(FileMetadata(
            file_path="docs/monsters.pdf",
            title="Monster Manual",
            detected_topics=["monsters", "lore"],
            entities=FileMetadataEntities(monsters=["Dragon", "Orc"]),
            scope=SCOPE_TOPIC_AWARE,
        ), b, workspace_root=ws)

        # File C: file-only; only when top-1
        c = docs / "exclusive.pdf"
        c.write_bytes(b"c")
        save(FileMetadata(
            file_path="docs/exclusive.pdf",
            title="Exclusive",
            entities=FileMetadataEntities(items=["SpecialThing"]),
            scope=SCOPE_FILE_ONLY,
        ), c, workspace_root=ws)

        # Hits ordered: monsters first, phb second, exclusive third
        hits = [
            FakeHit(path="docs/monsters.pdf"),
            FakeHit(path="docs/phb.pdf"),
            FakeHit(path="docs/exclusive.pdf"),
        ]
        brief = _build_per_file_brief(
            hits, workspace_root=ws,
            question="tell me about famous monsters",
        )
        check("brief contains global-scope phb",
              "Player's Handbook" in brief, f"got: {brief}")
        check("brief contains topic-aware monsters (matching topic)",
              "Monster Manual" in brief, f"got: {brief}")
        check("brief does NOT contain file-only exclusive (rank 2)",
              "Exclusive" not in brief, f"got: {brief}")

        # Different question — no monster topic match
        brief2 = _build_per_file_brief(
            hits, workspace_root=ws,
            question="describe a wizard's spellbook",
        )
        check("topic-aware silent when topic not in question",
              "Monster Manual" not in brief2, f"got: {brief2}")
        check("global still present (always)",
              "Player's Handbook" in brief2)

        # Reorder: exclusive at rank 0 → file-only fires
        hits_excl = [
            FakeHit(path="docs/exclusive.pdf"),
            FakeHit(path="docs/phb.pdf"),
        ]
        brief3 = _build_per_file_brief(
            hits_excl, workspace_root=ws,
            question="anything",
        )
        check("file-only fires when file is top-1",
              "Exclusive" in brief3, f"got: {brief3}")

    print("\n[14] _build_per_file_brief — empty hits / no metadata")
    check("empty hits -> empty string",
          _build_per_file_brief([], workspace_root=Path("/tmp")) == "")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        docs = ws / "docs"
        docs.mkdir()
        f = docs / "no-meta.pdf"
        f.write_bytes(b"no")
        # No sidecar saved
        brief = _build_per_file_brief(
            [FakeHit(path="docs/no-meta.pdf")],
            workspace_root=ws, question="any",
        )
        check("file with no sidecar contributes nothing",
              brief == "")

    print(f"\n=== ingest_meta summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

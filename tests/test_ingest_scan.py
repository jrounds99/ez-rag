"""Tests for ingest_scan — LLM-driven discovery scan that auto-populates
sidecars.

Stubs `_llm_complete` so no Ollama is required. Covers:
  - Stratified sampling (head + middle + tail)
  - JSON extraction (fence stripping, partial output, balanced braces)
  - Entity consolidation: hallucination filter + dedup + freq sort
  - End-to-end scan_and_save round-trip into a draft sidecar
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


@dataclass
class FakeSection:
    text: str
    page: int = 0
    section: str = ""


def main():
    from ez_rag import ingest_scan as scan
    from ez_rag.ingest_meta import FileMetadata, FileMetadataEntities

    print("\n[1] _stratified_sample picks head + middle + tail")
    sections = [FakeSection(text=f"chunk {i}") for i in range(20)]
    out = scan._stratified_sample(sections, count=8)
    check("returned 8 chunks",
          len(out) == 8, f"got {len(out)}")
    check("first 2 are heads",
          out[0].text == "chunk 0" and out[1].text == "chunk 1")
    check("last 2 are tails",
          out[-2].text == "chunk 18" and out[-1].text == "chunk 19")
    # And the small-input case: count >= len → return all
    out2 = scan._stratified_sample(sections[:5], count=8)
    check("count > len -> returns all",
          len(out2) == 5)

    print("\n[2] _strip_code_fences handles both ``` and ```json")
    check("no fence -> unchanged",
          scan._strip_code_fences("hello") == "hello")
    check("```json fence stripped",
          scan._strip_code_fences("```json\n{\"x\":1}\n```") == '{"x":1}')
    check("plain ``` fence stripped",
          scan._strip_code_fences("```\n{\"x\":1}\n```") == '{"x":1}')

    print("\n[3] _parse_json_safely tolerates LLM noise")
    cases = [
        ('{"title":"X","topics":["a","b"]}',
         {"title": "X", "topics": ["a", "b"]}),
        ('Sure! Here is your JSON:\n{"x": 1}\n\nLet me know.',
         {"x": 1}),
        ('```\n{"x": "with \\"quotes\\""}\n```',
         {"x": 'with "quotes"'}),
        ('not even close', None),
        ('', None),
    ]
    for raw, expected in cases:
        got = scan._parse_json_safely(raw)
        check(f"parse {raw[:30]!r} → {expected!r}",
              got == expected, f"got {got!r}")

    print("\n[4] _consolidate_entities — happy path")
    sampled = ("Durnan owns the Yawning Portal in Waterdeep. "
               "Vajra Safahr leads the Lords' Alliance. "
               "Volo wrote a guide.")
    batches = [
        {"npcs": ["Durnan", "Vajra Safahr", "Volo"],
         "factions": ["Lords' Alliance"],
         "locations": ["Waterdeep", "Yawning Portal"]},
        {"npcs": ["Durnan", "Volo"],   # repeats — should bump frequency
         "items": []},
    ]
    e = scan._consolidate_entities(batches, sampled)
    # All three NPCs survived (verbatim in the sampled text)
    check("npcs deduped",
          set(e.npcs) == {"Durnan", "Vajra Safahr", "Volo"},
          f"got {e.npcs}")
    # Durnan appeared in both batches → first in sort
    check("highest-freq NPC sorts first",
          e.npcs[0] == "Durnan" or e.npcs[0] == "Volo",
          f"got {e.npcs}")
    check("locations carried over",
          set(e.locations) == {"Waterdeep", "Yawning Portal"})
    check("factions carried over",
          e.factions == ["Lords' Alliance"])

    print("\n[5] hallucination filter drops entities not in sampled text")
    sampled = "The Player's Handbook describes basic combat."
    batches = [{
        "npcs": ["Mordenkainen", "Elminster"],   # NOT in sampled
        "items": ["Bag of Holding"],              # NOT in sampled
        "classes": ["Fighter"],                    # NOT in sampled
    }]
    e = scan._consolidate_entities(batches, sampled)
    check("hallucinated NPCs dropped",
          e.npcs == [], f"got {e.npcs}")
    check("hallucinated items dropped",
          e.items == [], f"got {e.items}")
    check("hallucinated classes dropped",
          e.classes == [], f"got {e.classes}")

    print("\n[6] very short / nonsense entries are dropped")
    sampled = "X. Y. Lorem ipsum dolor sit amet."
    batches = [{"npcs": ["X", "Y", "Lorem", ""]}]
    e = scan._consolidate_entities(batches, sampled)
    check("single-char names dropped",
          all(len(n) >= 2 for n in e.npcs))
    check("empty entries dropped",
          "" not in e.npcs)
    check("Lorem kept",
          "Lorem" in e.npcs)

    print("\n[7] case-insensitive dedup, original casing preserved")
    sampled = "Volo Volo VOLO."
    batches = [{"npcs": ["Volo", "VOLO", "volo"]}]
    e = scan._consolidate_entities(batches, sampled)
    check("case dedup keeps one",
          len(e.npcs) == 1, f"got {e.npcs}")
    # Whichever case the LLM emitted first wins for the display form
    check("first-form preserved",
          e.npcs[0] in ("Volo", "VOLO", "volo"))

    print("\n[8] scan_file end-to-end with stubbed LLM")
    # Stub `_llm_complete` and `detect_backend` so we don't need Ollama.
    from ez_rag import generate as gen
    saved_complete = gen._llm_complete
    saved_detect = gen.detect_backend

    # Two-call pattern: first call = topic pass, subsequent = entity passes
    call_index = {"n": 0}
    expected_topic_json = json.dumps({
        "title": "Bench Test Doc",
        "description": "Synthetic test content.",
        "topics": ["topic-a", "topic-b"],
    })
    expected_entity_json = json.dumps({
        "npcs": ["Alpha", "Beta"],
        "classes": ["Wizard"],
        "items": [],
        "locations": [],
        "factions": [],
        "spells": [],
        "monsters": [],
        "custom_terms": [],
    })

    def stub_llm(cfg, prompt, max_tokens=200):
        call_index["n"] += 1
        if "topic" in prompt.lower() or "title" in prompt.lower():
            # First call = topic
            return expected_topic_json
        return expected_entity_json

    gen._llm_complete = stub_llm
    gen.detect_backend = lambda c: "ollama"
    # Also stub it inside ingest_scan's namespace since it does
    # `from .generate import _llm_complete` — we need to patch the
    # canonical one before that import resolves. The dynamic import
    # in scan_file gets a fresh reference, so patching gen is enough.

    # Build a tiny fake parser registration so scan_file can read
    # our test file.
    from ez_rag import parsers
    saved_get_parser = parsers.get_parser

    def fake_parser(path):
        return [
            FakeSection(text="Alpha is a Wizard. Beta is brave."),
            FakeSection(text="The story continues."),
            FakeSection(text="Alpha returns to the tower."),
            FakeSection(text="Beta defeats the dragon."),
            FakeSection(text="Final chapter: epilogue."),
        ]
    parsers.get_parser = lambda path: fake_parser

    try:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            docs = ws / "docs"
            docs.mkdir()
            f = docs / "story.txt"
            f.write_text("dummy", encoding="utf-8")
            from ez_rag.config import Config
            cfg = Config()
            cfg.llm_model = "qwen2.5:7b"
            md = scan.scan_file(f, cfg)
            check("scan returned FileMetadata",
                  isinstance(md, FileMetadata))
            check("title populated from stub",
                  md.title == "Bench Test Doc")
            check("description populated from stub",
                  md.description == "Synthetic test content.")
            check("topics populated from stub",
                  set(md.detected_topics) == {"topic-a", "topic-b"})
            # Entities: Alpha + Beta + Wizard all appear in the stubbed
            # sections, so consolidation should keep them.
            check("Alpha kept (verbatim in samples)",
                  "Alpha" in md.entities.npcs,
                  f"npcs: {md.entities.npcs}")
            check("Beta kept",
                  "Beta" in md.entities.npcs)
            check("Wizard class kept",
                  "Wizard" in md.entities.classes)
            check("at least one LLM call fired",
                  call_index["n"] >= 1)

            # scan_and_save writes a .draft sidecar
            from ez_rag.ingest_scan import scan_and_save
            saved_path = scan_and_save(f, cfg, workspace_root=ws)
            check(".draft sidecar created",
                  saved_path.is_file()
                   and saved_path.suffix == ".draft")
            text = saved_path.read_text(encoding="utf-8")
            check("draft contains the title",
                  "Bench Test Doc" in text)
            check("draft contains entities",
                  "Alpha" in text and "Beta" in text)
    finally:
        gen._llm_complete = saved_complete
        gen.detect_backend = saved_detect
        parsers.get_parser = saved_get_parser

    print(f"\n=== ingest_scan summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

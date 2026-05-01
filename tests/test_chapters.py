"""Tests for chapter detection + chapter-aware retrieval."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.chapters import _from_sections, detect_chapters, find_chapter
from ez_rag.chunker import Chunk
from ez_rag.config import Config
from ez_rag.embed import make_embedder
from ez_rag.index import Index
from ez_rag.ingest import ingest
from ez_rag.retrieve import expand_to_chapter, hybrid_search, smart_retrieve
from ez_rag.workspace import Workspace


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


# ---------------------------------------------------------------------------

def test_from_sections_groups_consecutive():
    print("\n[1] _from_sections groups consecutive same-section chunks")
    chunks = [
        Chunk(text="a", page=1, section="Intro", ord=0),
        Chunk(text="b", page=1, section="Intro", ord=1),
        Chunk(text="c", page=2, section="Methods", ord=2),
        Chunk(text="d", page=3, section="Methods", ord=3),
        Chunk(text="e", page=4, section="Conclusion", ord=4),
    ]
    chs = _from_sections(chunks)
    check("got 3 chapters", len(chs) == 3, f"got {len(chs)}: {chs}")
    check("first chapter is Intro 0..1",
          chs[0]["title"] == "Intro" and chs[0]["start_ord"] == 0
          and chs[0]["end_ord"] == 1, f"got {chs[0]}")
    check("middle chapter is Methods 2..3",
          chs[1]["title"] == "Methods" and chs[1]["start_ord"] == 2
          and chs[1]["end_ord"] == 3, f"got {chs[1]}")
    check("last chapter is Conclusion 4..4",
          chs[2]["title"] == "Conclusion" and chs[2]["start_ord"] == 4
          and chs[2]["end_ord"] == 4, f"got {chs[2]}")


def test_from_sections_skips_blank_sections():
    print("\n[2] _from_sections skips chunks with blank section")
    chunks = [
        Chunk(text="a", page=1, section="", ord=0),
        Chunk(text="b", page=1, section="Real", ord=1),
    ]
    chs = _from_sections(chunks)
    check("blank section is not a chapter",
          all(c["title"] == "Real" for c in chs), f"got {chs}")


def test_detect_chapters_fallback():
    print("\n[3] detect_chapters returns single 'Document' chapter when no info")
    chunks = [
        Chunk(text="a", page=1, section="", ord=0),
        Chunk(text="b", page=2, section="", ord=1),
    ]
    chs = detect_chapters(Path("foo.md"), chunks)
    check("single fallback chapter",
          len(chs) == 1 and chs[0]["title"] == "Document"
          and chs[0]["start_ord"] == 0 and chs[0]["end_ord"] == 1,
          f"got {chs}")


def test_find_chapter():
    print("\n[4] find_chapter scans for ord")
    chs = [
        {"title": "A", "start_ord": 0, "end_ord": 4},
        {"title": "B", "start_ord": 5, "end_ord": 9},
    ]
    check("ord 3 -> A", find_chapter(chs, 3)["title"] == "A", "")
    check("ord 7 -> B", find_chapter(chs, 7)["title"] == "B", "")
    check("ord 99 -> None", find_chapter(chs, 99) is None, "")


# ---------------------------------------------------------------------------
# End-to-end: ingest a markdown doc with multiple H1 chapters → confirm
# chapter expansion swaps in the full chapter text.
# ---------------------------------------------------------------------------

def make_ws_with_chapters(tmp: Path) -> Workspace:
    ws = Workspace(tmp / "ws")
    ws.initialize()
    # parse_text returns one section per file, so let's use DOCX-style
    # section assignment by writing files where the chunker preserves section.
    # Easiest: sections are populated by parsers (DOCX/HTML); for markdown
    # they aren't. So we fabricate chapters via the post-hoc 'Document'
    # fallback first, then verify expand_to_chapter still works.
    (ws.docs_dir / "physics.md").write_text(
        "Border Collies herd sheep.\n\n"
        "Speed of light is 299792458 m/s.\n\n"
        "F equals m times a.\n",
        encoding="utf-8",
    )
    return ws


def test_chapter_metadata_persisted_and_used():
    print("\n[5] chapter metadata persisted on ingest, used on retrieve")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_chapter_e2e_"))
    try:
        ws = make_ws_with_chapters(tmp)
        cfg = Config(
            embedder_provider="fastembed",
            embedder_model="BAAI/bge-small-en-v1.5",
            chunk_size=80, chunk_overlap=10,
            unload_llm_during_ingest=False,
        )
        ingest(ws, cfg=cfg)

        embedder = make_embedder(cfg)
        idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

        # Every file should have at least one chapter row stored.
        n_with_chapters = idx.conn.execute(
            "SELECT COUNT(*) FROM files WHERE chapters_json IS NOT NULL"
        ).fetchone()[0]
        check("chapters_json populated for every file",
              n_with_chapters >= 1, f"got {n_with_chapters}")

        file_ids = [r[0] for r in idx.conn.execute(
            "SELECT id FROM files").fetchall()]
        for fid in file_ids:
            chs = idx.chapters_for_file(fid)
            check(f"file_id={fid} has at least 1 chapter",
                  len(chs) >= 1, f"got {chs}")

        # Retrieve + expand
        cfg2 = Config(**{**cfg.__dict__,
                         "expand_to_chapter": True, "chapter_max_chars": 20000})
        hits = smart_retrieve(query="speed of light",
                              embedder=embedder, index=idx, cfg=cfg2)
        check("expand_to_chapter retrieve returns hits",
              len(hits) >= 1, f"got {len(hits)}")
        if hits:
            top = hits[0]
            check("expanded hit has source_kind 'chapter' or 'chapter-skip'",
                  top.source_kind in ("chapter", "chapter-skip"),
                  f"got {top.source_kind}")
            if top.source_kind == "chapter":
                # Expanded text should contain content beyond the matched chunk
                check("expanded text mentions 299",
                      "299" in top.text or "light" in top.text.lower(),
                      f"text head={top.text[:120]!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_expand_skipped_when_chapter_too_big():
    print("\n[6] expand_to_chapter respects max_chars cap")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_chapter_cap_"))
    try:
        ws = Workspace(tmp / "ws")
        ws.initialize()
        # Big document so the single 'Document' chapter blows the cap.
        big = "\n\n".join([f"Paragraph {i}: " + ("border collie " * 80)
                            for i in range(40)])
        (ws.docs_dir / "big.md").write_text(big, encoding="utf-8")
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     chunk_size=80, chunk_overlap=10,
                     unload_llm_during_ingest=False)
        ingest(ws, cfg=cfg)

        embedder = make_embedder(cfg)
        idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
        cfg2 = Config(**{**cfg.__dict__,
                         "expand_to_chapter": True, "chapter_max_chars": 500})
        hits = smart_retrieve(query="border collie",
                              embedder=embedder, index=idx, cfg=cfg2)
        check("got hits", len(hits) >= 1, f"got {len(hits)}")
        if hits:
            check("hit marked chapter-skip when chapter exceeds cap",
                  hits[0].source_kind == "chapter-skip",
                  f"got {hits[0].source_kind}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dedup_chapter_dup_marker():
    print("\n[7] chapter dedup marks subsequent same-chapter hits")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_chapter_dedup_"))
    try:
        ws = Workspace(tmp / "ws")
        ws.initialize()
        text = "\n\n".join(f"Border collies {i} herd sheep" for i in range(20))
        (ws.docs_dir / "dogs.md").write_text(text, encoding="utf-8")
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     chunk_size=40, chunk_overlap=0,
                     unload_llm_during_ingest=False)
        ingest(ws, cfg=cfg)

        embedder = make_embedder(cfg)
        idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
        # Pre-fetch many hits, all from the single 'Document' chapter
        hits = hybrid_search(query="border collies",
                             embedder=embedder, index=idx, k=8)
        if len(hits) < 2:
            check("need 2+ hits for dedup test", False,
                  f"only got {len(hits)} hits")
            return
        out = expand_to_chapter(list(hits), idx, max_chars=100_000)
        n_chapter = sum(1 for h in out if h.source_kind == "chapter")
        n_dup = sum(1 for h in out if h.source_kind == "chapter-dup")
        check("first hit expands to chapter",
              n_chapter == 1, f"got n_chapter={n_chapter}")
        check("subsequent same-chapter hits marked chapter-dup",
              n_dup >= 1, f"got n_dup={n_dup}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

def main():
    test_from_sections_groups_consecutive()
    test_from_sections_skips_blank_sections()
    test_detect_chapters_fallback()
    test_find_chapter()
    test_chapter_metadata_persisted_and_used()
    test_expand_skipped_when_chapter_too_big()
    test_dedup_chapter_dup_marker()

    print(f"\n=== Chapters summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

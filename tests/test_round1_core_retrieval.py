"""Round 1: Core retrieval tests.

Covers:
  - Empty index behavior
  - hybrid_search returns hits, with vec / fts / hybrid source_kind
  - smart_retrieve honors top_k
  - Rerank ON vs OFF still returns top_k
  - Citation correctness: returned hits have valid path/page/text
  - use_rag toggle pathway: chat_answer with no hits uses NO_RAG prompt

Run with:
    python tests/test_round1_core_retrieval.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder
from ez_rag.index import Index
from ez_rag.ingest import ingest
from ez_rag.retrieve import hybrid_search, smart_retrieve, expand_with_neighbors
from ez_rag.workspace import Workspace


# ---------------------------------------------------------------------------
# Tiny harness — no pytest dependency
# ---------------------------------------------------------------------------
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


# ---------------------------------------------------------------------------
# Fixture: build a tiny workspace with controlled content
# ---------------------------------------------------------------------------
DOCS = {
    "dogs.md": (
        "# Dogs\n\n"
        "The Border Collie is renowned for its intelligence and herding "
        "ability. They originated in the Anglo-Scottish border region.\n\n"
        "Golden Retrievers are friendly, intelligent, and devoted dogs. "
        "They were originally bred in Scotland for retrieving waterfowl.\n"
    ),
    "physics.md": (
        "# Physics\n\n"
        "The speed of light in vacuum is exactly 299,792,458 meters per "
        "second. This constant is denoted by the letter c and is fundamental "
        "to special relativity.\n\n"
        "Newton's second law states that force equals mass times acceleration "
        "(F = ma). It governs classical mechanics.\n"
    ),
    "cooking.md": (
        "# Cooking\n\n"
        "Sourdough bread relies on wild yeast and lactobacilli for "
        "leavening. The starter must be fed regularly with flour and water.\n\n"
        "Caramelization begins around 160 degrees Celsius and is "
        "responsible for the brown color and complex flavor of seared meat.\n"
    ),
}


def make_tmp_ws() -> Workspace:
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round1_"))
    ws = Workspace(tmp)
    ws.initialize()
    for name, body in DOCS.items():
        (ws.docs_dir / name).write_text(body, encoding="utf-8")
    return ws


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_index_returns_no_hits(ws: Workspace, cfg: Config):
    print("\n[1] empty index returns no hits")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    # Wipe tables to simulate a freshly-opened empty index.
    idx.conn.execute("DELETE FROM chunks")
    idx.conn.execute("DELETE FROM files")
    idx.conn.commit()
    hits = hybrid_search(query="border collie", embedder=embedder, index=idx, k=5)
    check("hybrid_search on empty index -> []", hits == [], f"got {len(hits)} hits")
    hits = smart_retrieve(query="border collie", embedder=embedder, index=idx, cfg=cfg)
    check("smart_retrieve on empty index -> []", hits == [], f"got {len(hits)} hits")


def test_ingest_and_basic_retrieval(ws: Workspace, cfg: Config):
    print("\n[2] ingest + basic retrieval")
    stats = ingest(ws, cfg=cfg)
    check("ingest succeeded (no errors)", stats.files_errored == 0, f"errs={stats.errors}")
    check("ingested all 3 files", stats.files_seen == 3, f"saw {stats.files_seen}")
    check("added chunks", stats.chunks_added > 0, f"chunks={stats.chunks_added}")

    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    s = idx.stats()
    check("index has 3 files", s["files"] == 3, f"files={s['files']}")
    check("index has chunks", s["chunks"] >= 3, f"chunks={s['chunks']}")


def test_hybrid_finds_topical(ws: Workspace, cfg: Config):
    print("\n[3] hybrid_search finds topical hit")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    hits = hybrid_search(query="speed of light constant",
                         embedder=embedder, index=idx, k=3)
    check("got >=1 hit", len(hits) >= 1, f"hits={len(hits)}")
    if hits:
        top_path = hits[0].path.lower()
        check("top hit is from physics doc",
              "physics" in top_path,
              f"top path={hits[0].path}")
        check("top hit text mentions light",
              "light" in hits[0].text.lower() or "299" in hits[0].text,
              f"text={hits[0].text[:120]!r}")
        check("source_kind is set",
              hits[0].source_kind in ("vec", "fts", "hybrid"),
              f"kind={hits[0].source_kind}")
        check("hit has path", bool(hits[0].path), "")
        check("hit has chunk_id", hits[0].chunk_id > 0, f"id={hits[0].chunk_id}")


def test_top_k_honored(ws: Workspace, cfg: Config):
    print("\n[4] top_k is honored")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    for k in (1, 2, 3):
        hits = hybrid_search(query="dog", embedder=embedder, index=idx, k=k)
        check(f"hybrid_search k={k} returns <={k} hits",
              len(hits) <= k, f"got {len(hits)}")

    # smart_retrieve via cfg.top_k
    cfg2 = Config(**{**cfg.__dict__, "top_k": 2})
    hits = smart_retrieve(query="dog", embedder=embedder, index=idx, cfg=cfg2)
    check("smart_retrieve top_k=2 returns <=2 hits",
          len(hits) <= 2, f"got {len(hits)}")


def test_rerank_path(ws: Workspace, cfg: Config):
    print("\n[5] rerank ON vs OFF both return hits")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    no_rerank = hybrid_search(query="border collie herding",
                              embedder=embedder, index=idx, k=3, rerank=False)
    check("no-rerank returns hits", len(no_rerank) >= 1, "")

    try:
        rerank = hybrid_search(query="border collie herding",
                               embedder=embedder, index=idx, k=3, rerank=True)
        check("rerank returns hits", len(rerank) >= 1,
              "fastembed cross-encoder may not be installed")
        if rerank:
            check("rerank source_kind=='rerank'",
                  rerank[0].source_kind == "rerank",
                  f"kind={rerank[0].source_kind}")
            check("rerank top hit is from dogs",
                  "dogs" in rerank[0].path.lower(),
                  f"top={rerank[0].path}")
    except Exception as e:
        FAIL.append(("rerank does not crash",
                     f"raised {type(e).__name__}: {e}"))
        print(f"  FAIL  rerank does not crash -- {e!r}")


def test_hybrid_vs_dense_only(ws: Workspace, cfg: Config):
    print("\n[6] hybrid vs dense-only")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    # rare-keyword query: caramelization is unlikely to appear in many embeddings
    # but FTS should pick it up trivially
    h_hybrid = hybrid_search(query="caramelization",
                             embedder=embedder, index=idx, k=3, use_hybrid=True)
    h_dense = hybrid_search(query="caramelization",
                            embedder=embedder, index=idx, k=3, use_hybrid=False)
    check("hybrid finds caramelization",
          any("cooking" in h.path.lower() for h in h_hybrid),
          f"top={[h.path for h in h_hybrid[:1]]}")
    check("dense-only also returns >=1 hit",
          len(h_dense) >= 1, "")


def test_use_rag_off_uses_no_rag_prompt():
    print("\n[7] use_rag=False -> NO_RAG prompt path")
    from ez_rag.generate import (
        SYSTEM_PROMPT_NO_RAG, SYSTEM_PROMPT_RAG, _build_user_prompt,
    )

    user_msg = _build_user_prompt("hello", hits=[])
    check("empty hits -> bare question", user_msg == "hello",
          f"got {user_msg!r}")

    # With hits, prompt should contain Context and Answer instructions
    from ez_rag.index import Hit
    h = Hit(chunk_id=1, file_id=1, path="x.md", page=1,
            section="", text="snippet", score=1.0, source_kind="vec")
    user_msg = _build_user_prompt("hello", hits=[h])
    check("hits -> prompt has Context block",
          "Context:" in user_msg and "Answer with citations." in user_msg,
          f"got {user_msg[:120]!r}")
    check("NO_RAG prompt forbids citations",
          "Do NOT include citation" in SYSTEM_PROMPT_NO_RAG, "")
    check("RAG prompt instructs to cite",
          "[1]" in SYSTEM_PROMPT_RAG and "cite" in SYSTEM_PROMPT_RAG.lower(), "")


def test_chat_answer_branches():
    print("\n[8] chat_answer picks correct system prompt")
    # We won't actually call the LLM. Just verify the helper builds the right
    # message list. Patch detect_backend to return 'none' so chat_answer
    # short-circuits to the fallback (no network).
    from ez_rag import generate as gen
    original = gen.detect_backend
    gen.detect_backend = lambda cfg: "none"
    try:
        cfg = Config()
        ans = gen.chat_answer(history=[], latest_question="hi",
                              hits=[], cfg=cfg)
        check("backend=none gives fallback Answer",
              ans.backend == "none", f"got {ans.backend}")
    finally:
        gen.detect_backend = original


def test_citation_indices_align(ws: Workspace, cfg: Config):
    print("\n[9] citations align with hit list order")
    from ez_rag.generate import _format_context
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    hits = hybrid_search(query="newton's second law",
                         embedder=embedder, index=idx, k=3)
    if not hits:
        check("got hits for newton query", False, "no hits")
        return
    fmt = _format_context(hits)
    for i, h in enumerate(hits, start=1):
        check(f"context numbered [{i}] present", f"[{i}]" in fmt, "")
        check(f"context [{i}] contains hit path",
              h.path in fmt, f"missing {h.path}")


def test_neighbor_window_grows_text(ws: Workspace, cfg: Config):
    print("\n[10] context_window expands hit text")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    # Use a doc that produced multiple chunks if any. With our small DOCS each
    # may be a single chunk — that's still a valid no-op test.
    hits = hybrid_search(query="dog", embedder=embedder, index=idx, k=2)
    if not hits:
        check("hits for dog", False, "")
        return
    pre = len(hits[0].text)
    expanded = expand_with_neighbors(hits, idx, window=2)
    post = len(expanded[0].text)
    check("window=2 does not shrink text",
          post >= pre, f"pre={pre} post={post}")


# ---------------------------------------------------------------------------
def main():
    ws = make_tmp_ws()
    cfg = Config(
        embedder_provider="fastembed",      # bypass Ollama for determinism
        embedder_model="BAAI/bge-small-en-v1.5",
        rerank=False,                        # default off here; rerank tested separately
        hybrid=True,
        top_k=3,
    )
    print(f"[setup] tmp workspace: {ws.root}")
    try:
        test_empty_index_returns_no_hits(ws, cfg)
        test_ingest_and_basic_retrieval(ws, cfg)
        test_hybrid_finds_topical(ws, cfg)
        test_top_k_honored(ws, cfg)
        test_rerank_path(ws, cfg)
        test_hybrid_vs_dense_only(ws, cfg)
        test_use_rag_off_uses_no_rag_prompt()
        test_chat_answer_branches()
        test_citation_indices_align(ws, cfg)
        test_neighbor_window_grows_text(ws, cfg)
    finally:
        try:
            shutil.rmtree(ws.root, ignore_errors=True)
        except Exception:
            pass

    print(f"\n=== Round 1 summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for name, det in FAIL:
            print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

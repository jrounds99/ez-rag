"""Round 2: Advanced retrieval modifiers.

Covers HyDE, multi-query, MMR, and context_window — using a slightly larger
fixture corpus so MMR has near-duplicates to diversify across.

LLM-dependent paths (HyDE, multi-query) are tested with a stubbed
`_llm_complete` so they don't need Ollama.
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
from ez_rag import generate as gen
from ez_rag.retrieve import (
    expand_with_neighbors, hybrid_search, mmr_select, smart_retrieve,
)
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
# Fixture — multiple paragraphs about the same topics so MMR has duplicates
# and context_window has neighbors to expand.
# ---------------------------------------------------------------------------
DOCS = {
    "dogs.md": "\n\n".join([
        "# Dogs",
        "The Border Collie is the most intelligent breed of dog. They were "
        "bred for herding sheep in the Anglo-Scottish border.",
        "Border Collies excel at obedience trials and agility competitions. "
        "Their work ethic is legendary among shepherds.",
        "The Border Collie's stare, called the eye, is a key herding tool. "
        "They use it to control sheep without barking.",
        "Border Collies require significant mental stimulation. Without it "
        "they may develop neurotic behaviors.",
        "Golden Retrievers are gentle family dogs originally bred to retrieve "
        "waterfowl in 19th-century Scotland.",
        "Labrador Retrievers come in three colors: black, yellow, and "
        "chocolate. They are the most popular breed in North America.",
    ]),
    "physics.md": "\n\n".join([
        "# Physics",
        "Einstein's special relativity, published in 1905, links space and "
        "time into a single four-dimensional manifold.",
        "The speed of light in vacuum is exactly 299,792,458 meters per "
        "second, denoted c.",
        "Newton's laws of motion describe classical mechanics. The second "
        "law: force equals mass times acceleration, F=ma.",
        "Quantum mechanics replaces Newtonian determinism with probabilistic "
        "wave functions for very small particles.",
    ]),
    "cooking.md": "\n\n".join([
        "# Cooking",
        "Sourdough is leavened with wild yeast and lactobacilli — no "
        "commercial yeast required.",
        "The Maillard reaction occurs around 140 °C and creates the brown "
        "crust on baked bread and seared meat.",
        "Caramelization, distinct from Maillard, is the thermal breakdown "
        "of sugars beginning around 160 °C.",
    ]),
}


def make_tmp_ws() -> Workspace:
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round2_"))
    ws = Workspace(tmp)
    ws.initialize()
    for name, body in DOCS.items():
        (ws.docs_dir / name).write_text(body, encoding="utf-8")
    return ws


def test_hyde_uses_llm_output(ws, cfg):
    print("\n[1] HyDE replaces query with LLM output for embedding")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    captured = {}

    def stub_complete(_cfg, prompt, max_tokens=200):
        captured["prompt"] = prompt
        return "Border collies are renowned herding dogs from Britain."

    saved = gen._llm_complete
    gen._llm_complete = stub_complete
    try:
        cfg2 = Config(**{**cfg.__dict__, "use_hyde": True})
        hits = smart_retrieve(query="smart sheep dogs",
                              embedder=embedder, index=idx, cfg=cfg2)
        check("HyDE called LLM",
              "Question: smart sheep dogs" in captured.get("prompt", ""),
              f"prompt={captured.get('prompt', '')[:120]!r}")
        check("HyDE retrieval still produces hits",
              len(hits) >= 1, f"got {len(hits)}")
        if hits:
            check("HyDE top hit is dog-related",
                  "dogs" in hits[0].path.lower(),
                  f"top={hits[0].path}")
    finally:
        gen._llm_complete = saved


def test_hyde_falls_back_to_query_on_empty(ws, cfg):
    print("\n[2] HyDE returns bare query when LLM returns empty")
    out = gen.generate_hyde("test query",
                            Config(llm_provider="none"))
    # detect_backend returns 'none' so _llm_complete returns "" — thus the
    # raw query should pass through.
    check("HyDE falls back when backend=none",
          out == "test query", f"got {out!r}")


def test_multi_query_dedupes_and_falls_back(ws, cfg):
    print("\n[3] multi-query produces variations + always includes original")
    saved = gen._llm_complete
    gen._llm_complete = lambda _c, _p, max_tokens=200: (
        "intelligent sheep herding dog\n"
        "what is the smartest dog\n"
    )
    try:
        out = gen.generate_query_variations("smart sheep dogs",
                                            Config(), n=2)
        check("includes original query first",
              out[0] == "smart sheep dogs", f"got {out!r}")
        check("includes variations",
              len(out) >= 2, f"len={len(out)}")
    finally:
        gen._llm_complete = saved


def test_multi_query_path_in_smart_retrieve(ws, cfg):
    print("\n[4] multi_query=True triggers RRF-fusion path in smart_retrieve")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    saved = gen._llm_complete
    gen._llm_complete = lambda _c, _p, max_tokens=200: (
        "border collie herding\n"
        "intelligent shepherd dog\n"
    )
    try:
        cfg2 = Config(**{**cfg.__dict__, "multi_query": True, "rerank": False})
        hits = smart_retrieve(query="smart sheep dogs",
                              embedder=embedder, index=idx, cfg=cfg2)
        check("multi-query returns hits",
              len(hits) >= 1, f"got {len(hits)}")
        if hits:
            check("multi-query top hit is dogs",
                  "dogs" in hits[0].path.lower(),
                  f"top={hits[0].path}")
    finally:
        gen._llm_complete = saved


def test_mmr_picks_diverse(ws, cfg):
    print("\n[5] MMR selects fewer near-duplicates than relevance-only")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    # Big fetch to have similar dog chunks
    pool = hybrid_search(query="border collie",
                         embedder=embedder, index=idx, k=20)
    n_dog = sum(1 for h in pool if "dogs" in h.path.lower())
    check("pool has at least one dog hit", n_dog >= 1,
          f"dog hits in pool: {n_dog}")

    # MMR with strong diversity weight
    selected = mmr_select(list(pool), embedder, top_k=4, lambda_=0.0)
    check("MMR returns top_k items",
          len(selected) == min(4, len(pool)), f"len={len(selected)}")
    # With lambda=0 (pure diversity), MMR should NOT collapse to all-dogs.
    # Dense pool may still be dog-heavy, but diversity should reduce it.
    relevance_only = pool[:4]
    rel_dog_share = sum(1 for h in relevance_only if "dogs" in h.path.lower())
    mmr_dog_share = sum(1 for h in selected if "dogs" in h.path.lower())
    check("MMR (lambda=0) reduces or matches dog share vs relevance-only",
          mmr_dog_share <= rel_dog_share,
          f"mmr={mmr_dog_share} rel={rel_dog_share}")


def test_mmr_handles_small_pool(ws, cfg):
    print("\n[6] MMR with pool <= top_k returns input unchanged")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    pool = hybrid_search(query="caramelization",
                         embedder=embedder, index=idx, k=2)
    out = mmr_select(list(pool), embedder, top_k=10)
    check("MMR returns input when pool <= top_k",
          out == pool, "expected identity")


def test_context_window_grows_text(ws, cfg):
    print("\n[7] context_window expands hits with neighbors")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    base = hybrid_search(query="border collie",
                         embedder=embedder, index=idx, k=2)
    if not base:
        check("hits for border collie", False, "")
        return
    # snapshot original lengths
    orig_lens = [len(h.text) for h in base]
    expanded = expand_with_neighbors([type(h)(**h.__dict__) for h in base],
                                     idx, window=2)
    new_lens = [len(h.text) for h in expanded]
    check("window=2 grows or preserves text length",
          all(n >= o for n, o in zip(new_lens, orig_lens)),
          f"orig={orig_lens} new={new_lens}")
    # at least one hit should grow if there are neighboring chunks in same file
    grew = any(n > o for n, o in zip(new_lens, orig_lens))
    check("window expanded at least one hit (multi-chunk file)",
          grew, f"orig={orig_lens} new={new_lens}")


def test_context_window_zero_is_noop(ws, cfg):
    print("\n[8] context_window=0 is a no-op")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    base = hybrid_search(query="caramelization",
                         embedder=embedder, index=idx, k=2)
    out = expand_with_neighbors(base, idx, window=0)
    check("window=0 returns input unchanged", out is base, "")


def test_smart_retrieve_combines_modifiers(ws, cfg):
    print("\n[9] smart_retrieve with rerank+MMR+window all together")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    cfg2 = Config(**{**cfg.__dict__,
                     "rerank": True, "use_mmr": True,
                     "mmr_lambda": 0.5, "context_window": 1, "top_k": 4})
    hits = smart_retrieve(query="border collie herding",
                          embedder=embedder, index=idx, cfg=cfg2)
    check("combined modifiers return hits",
          len(hits) >= 1, f"got {len(hits)}")
    check("combined hits respect top_k",
          len(hits) <= 4, f"got {len(hits)}")
    if hits:
        check("combined top hit is dog-related",
              "dogs" in hits[0].path.lower(),
              f"top={hits[0].path}")


def main():
    ws = make_tmp_ws()
    cfg = Config(
        embedder_provider="fastembed",
        embedder_model="BAAI/bge-small-en-v1.5",
        rerank=False,
        hybrid=True,
        top_k=4,
        chunk_size=80,        # small so docs split into many chunks
        chunk_overlap=10,
    )
    print(f"[setup] tmp workspace: {ws.root}")
    try:
        stats = ingest(ws, cfg=cfg)
        print(f"[setup] ingested: files={stats.files_seen} "
              f"chunks={stats.chunks_added}")
        test_hyde_uses_llm_output(ws, cfg)
        test_hyde_falls_back_to_query_on_empty(ws, cfg)
        test_multi_query_dedupes_and_falls_back(ws, cfg)
        test_multi_query_path_in_smart_retrieve(ws, cfg)
        test_mmr_picks_diverse(ws, cfg)
        test_mmr_handles_small_pool(ws, cfg)
        test_context_window_grows_text(ws, cfg)
        test_context_window_zero_is_noop(ws, cfg)
        test_smart_retrieve_combines_modifiers(ws, cfg)
    finally:
        try:
            shutil.rmtree(ws.root, ignore_errors=True)
        except Exception:
            pass

    print(f"\n=== Round 2 summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for name, det in FAIL:
            print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

"""Hybrid retrieval: dense (cosine) + sparse (FTS5 BM25), fused with RRF.

Optional pipeline stages on top:
- HyDE             : embed a hypothetical answer instead of the bare question
- Multi-query      : run N paraphrases of the question and fuse
- Reranking        : cross-encoder reranks the top-K candidates for relevance
- MMR              : diversify the final selection (Maximum Marginal Relevance)
- Context window   : include neighbor chunks around each hit for more context
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .embed import DEFAULT_RERANKER, Embedder, cosine_top_k, rerank_hits
from .generate import generate_hyde, generate_query_variations
from .index import Hit, Index


def hybrid_search(
    *, query: str, embedder: Embedder, index: Index, k: int = 8, fetch: int = 30,
    use_hybrid: bool = True,
    rerank: bool = False,
    rerank_model: str | None = None,
) -> list[Hit]:
    """Retrieve top-K hits for a single query string.

    With `rerank=True` we fetch a larger candidate pool, then narrow with a
    cross-encoder. The reranker is the highest-impact addition you can make.
    """
    fetch_n = max(fetch, 30) if rerank else fetch
    qvec = embedder.embed([query])[0]
    mat, ids = index.all_embeddings()
    if mat.shape[0] == 0:
        return []

    vec_idx, vec_scores = cosine_top_k(qvec, mat, fetch_n)
    vec_chunk_ids = [ids[i] for i in vec_idx.tolist()]
    vec_scores_l = vec_scores.tolist()

    fts = index.fts_search(query, fetch_n) if use_hybrid else []

    fused = _rrf({cid: rank + 1 for rank, cid in enumerate(vec_chunk_ids)},
                 {cid: rank + 1 for rank, (cid, _) in enumerate(fts)})

    candidate_n = fetch_n if rerank else k
    top_ids = [cid for cid, _ in
               sorted(fused.items(), key=lambda kv: -kv[1])[:candidate_n]]

    score_by_id = {cid: float(s) for cid, s in zip(vec_chunk_ids, vec_scores_l)}
    fts_by_id = dict(fts)

    hits = index.get_chunks(top_ids)
    for h in hits:
        v = score_by_id.get(h.chunk_id, 0.0)
        f = fts_by_id.get(h.chunk_id, 0.0)
        h.score = max(v, f)
        h.source_kind = "hybrid" if (v > 0 and f > 0) else ("vec" if v > 0 else "fts")

    if rerank and len(hits) > 1:
        return rerank_hits(query, hits, top_k=k,
                           model_name=rerank_model or DEFAULT_RERANKER)
    return hits[:k]


def expand_with_neighbors(hits: list[Hit], index: Index, window: int) -> list[Hit]:
    """For each hit, expand its `text` to include ±`window` neighbor chunks
    from the same source file (joined by blank lines). The `chunk_id`,
    `path`, `page`, `section`, `score`, `source_kind` of the original hit
    are preserved — only the text grows.

    Equivalent to a "sentence-window" or "parent-document" retrieval: we
    match on small chunks but pass larger context to the LLM.
    """
    if window <= 0 or not hits:
        return hits
    chunk_ids = [h.chunk_id for h in hits]
    placeholders = ",".join("?" * len(chunk_ids))
    rows = index.conn.execute(
        f"SELECT id, file_id, ord FROM chunks WHERE id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    info = {r[0]: (r[1], r[2]) for r in rows}
    for h in hits:
        if h.chunk_id not in info:
            continue
        fid, ord_ = info[h.chunk_id]
        rows = index.conn.execute(
            "SELECT ord, text FROM chunks "
            "WHERE file_id = ? AND ord BETWEEN ? AND ? ORDER BY ord",
            (fid, ord_ - window, ord_ + window),
        ).fetchall()
        if not rows:
            continue
        # Replace hit.text with the joined neighborhood
        h.text = "\n\n".join(t for _, t in rows)
    return hits


def mmr_select(
    hits: list[Hit],
    embedder: Embedder | None,
    top_k: int,
    lambda_: float = 0.5,
) -> list[Hit]:
    """Maximum Marginal Relevance — pick `top_k` hits that balance relevance
    (existing scores) with diversity (low pairwise similarity).

    Falls back to relevance-only ordering if `embedder` is None or any
    embedding step fails.
    """
    if len(hits) <= top_k:
        return hits
    if embedder is None:
        return hits[:top_k]
    try:
        embs = embedder.embed([h.text for h in hits])
    except Exception:
        return hits[:top_k]
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs_n = embs / norms
    relevance = np.array([float(h.score) for h in hits])

    selected: list[int] = []
    remaining: set[int] = set(range(len(hits)))
    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_emb = embs_n[selected]
            best, best_score = -1, -1e18
            for i in remaining:
                sims = sel_emb @ embs_n[i]
                max_sim = float(np.max(sims)) if sims.size else 0.0
                score = lambda_ * float(relevance[i]) - (1 - lambda_) * max_sim
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.discard(best)
    return [hits[i] for i in selected]


def smart_retrieve(
    *,
    query: str,
    embedder: Embedder,
    index: Index,
    cfg: Config,
) -> list[Hit]:
    """End-to-end retrieval honoring all `cfg` retrieval flags.

    Pipeline:
        if hyde:           replace query with LLM-generated hypothetical answer
        if multi_query:    fan out to N paraphrases
        for each query:    hybrid search → top-K candidates
        merge + rerank (cross-encoder) if enabled
    """
    queries: list[str] = [query]

    if getattr(cfg, "use_hyde", False):
        queries = [generate_hyde(query, cfg)]

    if getattr(cfg, "multi_query", False):
        # Generate variations of the (post-HyDE) query
        variations = generate_query_variations(queries[0], cfg, n=2)
        # always include the original raw question too, so we never lose it
        queries = list(dict.fromkeys(variations + [query]))

    if len(queries) == 1:
        # When MMR is enabled, fetch a wider candidate pool so it has
        # something to diversify across.
        fetch_k = max(cfg.top_k * 3, 20) if getattr(cfg, "use_mmr", False) else cfg.top_k
        hits = hybrid_search(
            query=queries[0], embedder=embedder, index=index,
            k=fetch_k, use_hybrid=cfg.hybrid,
            rerank=cfg.rerank,
        )
        if getattr(cfg, "use_mmr", False) and len(hits) > cfg.top_k:
            hits = mmr_select(
                hits, embedder, top_k=cfg.top_k,
                lambda_=getattr(cfg, "mmr_lambda", 0.5),
            )
        else:
            hits = hits[:cfg.top_k]
        if getattr(cfg, "context_window", 0) > 0:
            hits = expand_with_neighbors(hits, index, cfg.context_window)
        return hits

    # Multi-query: search each, fuse with RRF.
    rank_maps: list[dict[int, int]] = []
    by_id: dict[int, Hit] = {}
    for q in queries:
        hs = hybrid_search(
            query=q, embedder=embedder, index=index,
            k=max(cfg.top_k, 12), use_hybrid=cfg.hybrid,
            rerank=False,           # rerank ONCE at the end with the original query
        )
        rank_maps.append({h.chunk_id: rank + 1 for rank, h in enumerate(hs)})
        for h in hs:
            by_id.setdefault(h.chunk_id, h)
    fused = _rrf(*rank_maps)
    fused_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    candidates = [by_id[cid] for cid in fused_ids if cid in by_id]

    if cfg.rerank and len(candidates) > 1:
        candidates = rerank_hits(
            query, candidates[:max(30, cfg.top_k * 4)],
            top_k=max(cfg.top_k * 3, 20) if getattr(cfg, "use_mmr", False) else cfg.top_k,
        )
    if getattr(cfg, "use_mmr", False) and len(candidates) > cfg.top_k:
        candidates = mmr_select(
            candidates, embedder, top_k=cfg.top_k,
            lambda_=getattr(cfg, "mmr_lambda", 0.5),
        )
    else:
        candidates = candidates[:cfg.top_k]
    if getattr(cfg, "context_window", 0) > 0:
        candidates = expand_with_neighbors(candidates, index, cfg.context_window)
    return candidates


def _rrf(*ranks: dict[int, int], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in ranks:
        for cid, rank in r.items():
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + rank)
    return out

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
from .generate import (
    _is_list_query, agent_complete, detect_backend, generate_hyde,
    generate_list_hyde, generate_query_variations,
)
from .index import Hit, Index


class EmbedderMismatchError(RuntimeError):
    """Current embedder's vector dimension doesn't match what the index
    was built with. Re-ingest required (or switch embedder back)."""


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
    # Embedder swap detection — produces a much more useful error than
    # numpy's `matmul: Input operand 1 has a mismatch in its core
    # dimension 0...` when the current embedder produces vectors of a
    # different dimension than what the index was built with.
    if qvec.shape[0] != mat.shape[1]:
        # Read which embedder built the index for a more precise message.
        prev = ""
        try:
            row = index.conn.execute(
                "SELECT embedder FROM files ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                prev = row[0]
        except Exception:
            pass
        prev_str = f" (built with `{prev}`)" if prev else ""
        raise EmbedderMismatchError(
            f"Embedder dimension mismatch: query vector is {qvec.shape[0]}-d "
            f"but the index has {mat.shape[1]}-d vectors{prev_str}.\n\n"
            f"This means the embedder changed since ingest. Current "
            f"embedder: `{embedder.name}` ({embedder.dim}-d).\n\n"
            f"Two ways to fix:\n"
            f"  1. Re-ingest with the current embedder — Files tab → "
            f"Re-ingest (force). Will rebuild every chunk vector.\n"
            f"  2. Switch the embedder back in Settings to whatever built "
            f"the index{prev_str}, then retry the question."
        )

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


def expand_to_chapter(
    hits: list[Hit], index: Index, *, max_chars: int = 16000,
) -> list[Hit]:
    """For each hit, replace `text` with the full text of its chapter.

    Chapter boundaries come from `files.chapters_json` (built at ingest
    time from PDF bookmarks or section headings). When two hits land in
    the same chapter, only the first keeps the expanded text — the others
    are tagged with `source_kind='chapter-dup'` so the UI/LLM can dedupe
    visually.

    Falls back to the original hit text when no chapter metadata exists or
    the chapter would exceed `max_chars` (in which case the original chunk
    is kept to preserve the precise hit).
    """
    if not hits:
        return hits
    from .chapters import find_chapter

    seen_ids: dict[int, str] = {}     # file_id → chapter title already used
    chapters_by_file: dict[int, list[dict]] = {}
    for h in hits:
        if h.file_id not in chapters_by_file:
            chapters_by_file[h.file_id] = index.chapters_for_file(h.file_id)
        chapters = chapters_by_file[h.file_id]
        if not chapters:
            continue

        # Get the hit's ord from the chunks table.
        row = index.conn.execute(
            "SELECT ord FROM chunks WHERE id = ?", (h.chunk_id,),
        ).fetchone()
        if not row:
            continue
        ord_ = row[0]
        ch = find_chapter(chapters, ord_)
        if ch is None:
            continue

        # Dedupe: only first hit in a chapter expands; later hits get
        # marked but keep their own text (so the LLM still sees them as
        # distinct citations and the UI can warn about overlap).
        key = (h.file_id, ch["title"])
        if key in seen_ids:
            h.source_kind = "chapter-dup"
            continue
        seen_ids[key] = ch["title"]

        rows = index.conn.execute(
            "SELECT text FROM chunks "
            "WHERE file_id = ? AND ord BETWEEN ? AND ? ORDER BY ord",
            (h.file_id, ch["start_ord"], ch["end_ord"]),
        ).fetchall()
        if not rows:
            continue
        full = "\n\n".join(r[0] for r in rows if r[0])
        if not full:
            continue
        if len(full) > max_chars:
            # Chapter is too big — keep the original hit text but flag it
            # so callers know we tried.
            h.source_kind = "chapter-skip"
            continue
        h.text = full
        h.section = ch["title"] or h.section
        h.source_kind = "chapter"
    return hits


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


def crag_filter_chunks(query: str, hits: list[Hit], cfg: Config,
                        *, max_keep: int | None = None) -> list[Hit]:
    """CRAG-style chunk relevance filter (Corrective Retrieval AG).

    After retrieval, ask the LLM in ONE batched call which of the
    retrieved chunks are actually relevant to the query. Drop the
    irrelevant ones before they reach the answer prompt.

    Single call, not N — chunks are presented as a numbered list and
    the LLM returns the kept IDs as a comma-separated list. Costs ~1
    extra small LLM call per query.

    Returns the filtered hits in their original rank order. If parsing
    fails or the LLM is unavailable, returns the input unchanged
    (degrades gracefully).
    """
    from .generate import _llm_complete, detect_backend
    if not hits or detect_backend(cfg) == "none":
        return hits
    # Cap each chunk at ~400 chars in the filter prompt so it stays
    # cheap and the LLM can scan all of them in one pass.
    snippets = []
    for i, h in enumerate(hits, start=1):
        snippet = (h.text or "").strip().replace("\n", " ")[:400]
        snippets.append(f"[{i}] {snippet}")
    chunks_block = "\n\n".join(snippets)
    prompt = (
        "You are filtering search results for relevance. The user has a "
        "question and we retrieved several candidate passages. Mark "
        "which ones are actually relevant to answering the question.\n\n"
        f"QUESTION: {query}\n\n"
        f"PASSAGES:\n{chunks_block}\n\n"
        "Reply with ONLY a comma-separated list of the passage numbers "
        "that are RELEVANT — no commentary, no preamble. If none are "
        "relevant reply with 'none'. If you're unsure about a passage, "
        "include it (we'd rather over-include than under-include)."
    )
    raw = _llm_complete(cfg, prompt, max_tokens=80) or ""
    raw = raw.strip().lower()
    if "none" in raw and not any(c.isdigit() for c in raw):
        return []   # explicit refusal — nothing relevant
    # Parse digits out
    keep_ids = set()
    cur = ""
    for ch in raw + ",":
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                try:
                    n = int(cur)
                    if 1 <= n <= len(hits):
                        keep_ids.add(n - 1)
                except ValueError:
                    pass
                cur = ""
    if not keep_ids:
        return hits   # parse failure → keep everything (don't drop the answer)
    filtered = [h for i, h in enumerate(hits) if i in keep_ids]
    if max_keep:
        filtered = filtered[:max_keep]
    return filtered


def reorder_for_attention(hits: list[Hit]) -> list[Hit]:
    """Reorder a ranked hit list to combat the 'lost in the middle' effect.

    Stanford 2023 (arXiv:2307.03172) showed LLMs attend strongly to
    content at the START and END of the prompt, less to the middle. So
    we interleave the ranking: most-relevant FIRST, second-most-relevant
    LAST, third in second position, fourth in second-to-last, etc.

    For ranks [0, 1, 2, 3, 4, 5, 6, 7] this yields:
        [0, 2, 4, 6, 7, 5, 3, 1]
        ─────high-attention bookends────

    Free win: zero LLM calls, zero retrieval changes. Documented to lift
    quality on long-context RAG by 5-15% in the original paper.
    """
    if len(hits) <= 2:
        return hits
    front: list[Hit] = []
    back: list[Hit] = []
    for i, h in enumerate(hits):
        if i % 2 == 0:
            front.append(h)
        else:
            back.append(h)
    return front + list(reversed(back))


def diversify_by_source(hits: list[Hit], *, cap_per_source: int = 3,
                          target_k: int = 0) -> list[Hit]:
    """Rebalance a ranked hit list so no single source file dominates.

    Walks the input in rank order. Each path gets at most `cap_per_source`
    chunks in the output before we start skipping. Once the diversified
    list is shorter than `target_k` (because too many candidates came
    from few sources), it back-fills by relaxing the cap one slot at a
    time on the highest-ranked sources.

    Why this matters: on a corpus with 100+ PDFs, an unfiltered top-K
    can come back with all 8 chunks from the same source, which gives
    the LLM no diversity to extract from. A cap of 3 per source forces
    the answer to be grounded across multiple books.

    Pass `cap_per_source=0` to disable (returns hits unchanged).
    """
    if cap_per_source <= 0 or not hits:
        return hits
    counts: dict[str, int] = {}
    kept: list[Hit] = []
    skipped: list[Hit] = []
    for h in hits:
        path = h.path or ""
        if counts.get(path, 0) < cap_per_source:
            kept.append(h)
            counts[path] = counts.get(path, 0) + 1
        else:
            skipped.append(h)
    # If diversification dropped us below the requested top_k, back-fill
    # from the skipped pile in original rank order — better to slightly
    # over-represent a strong source than to return too few hits.
    if target_k and len(kept) < target_k and skipped:
        backfill = skipped[: target_k - len(kept)]
        kept.extend(backfill)
    return kept


def copy_cfg_for_list(cfg: Config) -> Config:
    """Return a shallow copy of cfg adjusted for list queries.

    Three changes from the user's normal cfg:

    1. **top_k bumped to ≥16** — list answers extract named entities
       from across many chunks, so they benefit from a wider candidate
       pool.

    2. **chapter_max_chars capped at 4000** — without this cap, a
       chapter-expanded retrieval at top_k=8 gives ~128 KB of context
       and the LLM summarizes the chapters instead of extracting
       specific named items (this is the failure mode the user saw —
       getting "summaries of whole manuals"). Capping each chapter
       expansion to ~4 KB keeps the per-chunk-anchor benefit but
       prevents any single source from drowning the others. At
       top_k=16 the total context stays around 64 KB.

    3. **context_window forced to 0** — neighbor expansion compounds
       with chapter expansion to inflate context further. Off for
       list queries.

    Non-list queries keep the user's normal cfg (full chapter expansion
    is great for "explain this rule" type questions).
    """
    import copy as _copy
    bumped = _copy.copy(cfg)
    bumped.top_k = max(16, getattr(cfg, "top_k", 8))
    bumped.chapter_max_chars = min(4000,
                                    getattr(cfg, "chapter_max_chars", 16000))
    bumped.context_window = 0
    return bumped


def smart_retrieve(
    *,
    query: str,
    embedder: Embedder,
    index: Index,
    cfg: Config,
    status_cb=None,
) -> list[Hit]:
    """End-to-end retrieval honoring all `cfg` retrieval flags.

    Pipeline:
        if hyde:           replace query with LLM-generated hypothetical answer
        if multi_query:    fan out to N paraphrases
        for each query:    hybrid search → top-K candidates
        merge + rerank (cross-encoder) if enabled
        diversify (cap chunks per source)
        expand (chapter / neighbors)
        reorder (lost-in-middle)

    `status_cb(stage_id)` — optional. Fires as the pipeline enters each
    stage so a UI can pulse the active step. Stage ids:
      "query_expand" | "hybrid_search" | "rerank" | "diversify" |
      "expand" | "reorder" | "done"
    """
    def _emit(stage):
        if status_cb is None:
            return
        try:
            status_cb(stage)
        except Exception:
            pass

    queries: list[str] = [query]

    # Auto-detect list-style queries — they retrieve dramatically better
    # with the entity-rich HyDE variant than with the bare question.
    # Falls back to the generic HyDE when the user has explicitly opted
    # in via cfg.use_hyde.
    auto_list = (getattr(cfg, "auto_list_mode", True)
                  and _is_list_query(query))
    if auto_list:
        _emit("query_expand")
        queries = [generate_list_hyde(query, cfg)]
        # List answers benefit from more context — every chunk is a
        # potential source of additional named items to extract.
        # Bump top_k for THIS retrieval only, never below the user's
        # configured value.
        cfg = copy_cfg_for_list(cfg)
    elif getattr(cfg, "use_hyde", False):
        _emit("query_expand")
        queries = [generate_hyde(query, cfg)]

    if getattr(cfg, "multi_query", False):
        _emit("query_expand")
        # Generate variations of the (post-HyDE) query
        variations = generate_query_variations(queries[0], cfg, n=2)
        # always include the original raw question too, so we never lose it
        queries = list(dict.fromkeys(variations + [query]))

    if len(queries) == 1:
        # When MMR or diversification is enabled, fetch a wider candidate
        # pool so we have something to diversify / re-select across.
        diversify_n = int(getattr(cfg, "diversify_per_source", 3) or 0)
        fetch_k = cfg.top_k
        if getattr(cfg, "use_mmr", False):
            fetch_k = max(fetch_k, cfg.top_k * 3, 20)
        if diversify_n > 0:
            fetch_k = max(fetch_k, cfg.top_k * 2 + 6)
        _emit("hybrid_search")
        hits = hybrid_search(
            query=queries[0], embedder=embedder, index=index,
            k=fetch_k, use_hybrid=cfg.hybrid,
            rerank=cfg.rerank,
        )
        if cfg.rerank:
            _emit("rerank")
        if getattr(cfg, "use_mmr", False) and len(hits) > cfg.top_k:
            _emit("diversify")
            hits = mmr_select(
                hits, embedder, top_k=cfg.top_k,
                lambda_=getattr(cfg, "mmr_lambda", 0.5),
            )
        elif diversify_n > 0:
            _emit("diversify")
            hits = diversify_by_source(
                hits, cap_per_source=diversify_n, target_k=cfg.top_k,
            )[:cfg.top_k]
        else:
            hits = hits[:cfg.top_k]
        if getattr(cfg, "crag_filter", False):
            _emit("crag")
            hits = crag_filter_chunks(query, hits, cfg) or hits
        if getattr(cfg, "context_window", 0) > 0:
            _emit("expand")
            hits = expand_with_neighbors(hits, index, cfg.context_window)
        if getattr(cfg, "expand_to_chapter", False):
            _emit("expand")
            hits = expand_to_chapter(
                hits, index,
                max_chars=int(getattr(cfg, "chapter_max_chars", 16000)),
            )
        if getattr(cfg, "reorder_for_attention", False):
            _emit("reorder")
            hits = reorder_for_attention(hits)
        _emit("done")
        return hits

    # Multi-query: search each, fuse with RRF.
    _emit("hybrid_search")
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
        _emit("rerank")
        candidates = rerank_hits(
            query, candidates[:max(30, cfg.top_k * 4)],
            top_k=max(cfg.top_k * 3, 20) if getattr(cfg, "use_mmr", False) else cfg.top_k,
        )
    diversify_n = int(getattr(cfg, "diversify_per_source", 3) or 0)
    if getattr(cfg, "use_mmr", False) and len(candidates) > cfg.top_k:
        _emit("diversify")
        candidates = mmr_select(
            candidates, embedder, top_k=cfg.top_k,
            lambda_=getattr(cfg, "mmr_lambda", 0.5),
        )
    elif diversify_n > 0:
        _emit("diversify")
        candidates = diversify_by_source(
            candidates, cap_per_source=diversify_n, target_k=cfg.top_k,
        )[:cfg.top_k]
    else:
        candidates = candidates[:cfg.top_k]
    if getattr(cfg, "context_window", 0) > 0:
        _emit("expand")
        candidates = expand_with_neighbors(candidates, index, cfg.context_window)
    if getattr(cfg, "expand_to_chapter", False):
        _emit("expand")
        candidates = expand_to_chapter(
            candidates, index,
            max_chars=int(getattr(cfg, "chapter_max_chars", 16000)),
        )
    if getattr(cfg, "reorder_for_attention", True):
        _emit("reorder")
        candidates = reorder_for_attention(candidates)
    _emit("done")
    return candidates


# ============================================================================
# Agentic retrieval — iterate retrieve → reflect → maybe re-search
# ============================================================================

def agentic_retrieve(
    *,
    query: str,
    embedder: Embedder,
    index: Index,
    cfg: Config,
    status_cb=None,
) -> list[Hit]:
    """Brute-force agentic retrieval. The LLM looks at the initial hits and,
    if they're insufficient, generates 1-2 follow-up queries; their results
    are fused with RRF; the final list is reranked once.

    `status_cb(message)` is invoked at each step so UIs can show progress.
    """

    def emit(msg: str):
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    emit("agent · initial retrieval")
    initial = smart_retrieve(query=query, embedder=embedder,
                             index=index, cfg=cfg)
    backend = detect_backend(cfg)

    iterations = max(0, getattr(cfg, "agent_max_iterations", 2))
    if iterations == 0 or backend == "none":
        return initial

    accumulated: dict[int, Hit] = {h.chunk_id: h for h in initial}
    rank_maps: list[dict[int, int]] = [
        {h.chunk_id: i + 1 for i, h in enumerate(initial)}
    ]

    for step in range(iterations):
        ctx_summary = "\n".join(
            f"[{i+1}] ({h.path}) {h.text[:200].strip()}…"
            for i, h in enumerate(list(accumulated.values())[:6])
        )
        prompt = (
            "You are helping a search engine find passages that answer a "
            "user's question.\n\n"
            f"User question: {query}\n\n"
            f"Already retrieved (top so far):\n{ctx_summary or '(none)'}\n\n"
            "If the passages clearly contain the answer, output exactly:\n"
            "  SUFFICIENT\n\n"
            "Otherwise, output 1 or 2 short alternative search queries that "
            "would surface missing information. ONE per line. No numbering, "
            "no explanation, no quotes.\n"
        )
        emit(f"agent · reflecting (step {step + 1}/{iterations})")
        response = agent_complete(cfg, [{"role": "user", "content": prompt}],
                                  max_tokens=200)
        if not response or response.strip().split()[:1] == ["SUFFICIENT"] \
                or "SUFFICIENT" in response.upper().split()[:3]:
            break
        new_qs = []
        for line in response.split("\n"):
            s = line.strip().strip("-•*0123456789. ").strip('"').strip("'")
            if s and s.lower() != query.lower() and "SUFFICIENT" not in s.upper():
                new_qs.append(s)
            if len(new_qs) >= 2:
                break
        if not new_qs:
            break
        emit(f"agent · refining: {', '.join(q[:30] + '…' for q in new_qs)}")
        for q in new_qs:
            sub = smart_retrieve(query=q, embedder=embedder,
                                 index=index, cfg=cfg)
            for h in sub:
                accumulated.setdefault(h.chunk_id, h)
            rank_maps.append({h.chunk_id: i + 1 for i, h in enumerate(sub)})

    # Fuse all rankings, then rerank once with the original query.
    fused = _rrf(*rank_maps)
    sorted_ids = [cid for cid, _ in
                  sorted(fused.items(), key=lambda kv: -kv[1])]
    candidates = [accumulated[cid] for cid in sorted_ids
                  if cid in accumulated]

    if cfg.rerank and len(candidates) > 1:
        emit("agent · final rerank")
        candidates = rerank_hits(query, candidates[:max(30, cfg.top_k * 4)],
                                 top_k=cfg.top_k)
    return candidates[:cfg.top_k]


def _rrf(*ranks: dict[int, int], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in ranks:
        for cid, rank in r.items():
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + rank)
    return out

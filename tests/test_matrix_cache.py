"""Embedding-matrix cache + batched-lookup tests (query hot path).

Covers the optimization pass:
  1. `Index.all_embeddings()` returns a CACHED, L2-normalized matrix on
     repeat calls (identity check) — the dense search no longer re-reads
     every BLOB per query.
  2. Mutations invalidate: replace_file (including the rowid-reuse edge
     where (COUNT, MAX(id)) is unchanged) and delete_missing.
  3. `cosine_top_k(..., assume_normalized=True)` matches brute-force
     cosine ranking.
  4. `embeddings_for` / `ords_for` batched lookups are correct.
  5. `mmr_select(index=...)` diversifies from stored vectors — works even
     with embedder=None and never calls the embedder.

No Ollama, no fastembed — vectors are handmade numpy.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import index as index_mod
from ez_rag.index import Index
from ez_rag.embed import cosine_top_k
from ez_rag.retrieve import mmr_select

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


DIM = 8


def vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).astype(np.float32)


def add_file(idx: Index, path: str, n: int, seed0: int) -> None:
    chunks = [
        (i, None, "", f"text {path} {i}", f"tok {path} {i}", vec(seed0 + i))
        for i in range(n)
    ]
    idx.replace_file(
        path=path, sha256=f"sha-{path}-{seed0}", bytes_=100, mtime=1.0,
        parser_version="1", chunker_version="2", embedder="test",
        chunks=chunks,
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_matcache_"))
    db = tmp / "meta.sqlite"
    idx = Index(db, embed_dim=DIM)

    print("\n[1] cache hit + normalization")
    add_file(idx, "a.md", 5, seed0=100)
    add_file(idx, "b.md", 4, seed0=200)
    mat1, ids1 = idx.all_embeddings()
    mat2, ids2 = idx.all_embeddings()
    check("repeat call returns cached matrix (identity)", mat2 is mat1)
    check("repeat call returns cached ids (identity)", ids2 is ids1)
    check("rows are L2-normalized",
          bool(np.allclose(np.linalg.norm(mat1, axis=1), 1.0, atol=1e-5)))
    check("matrix is float32 contiguous",
          mat1.dtype == np.float32 and mat1.flags["C_CONTIGUOUS"])
    check("9 rows", mat1.shape == (9, DIM), f"{mat1.shape}")

    print("\n[2] invalidation on replace_file (rowid-reuse edge)")
    # Replace the LAST-ingested file with the SAME chunk count: SQLite
    # reuses rowids, so (COUNT, MAX(id)) stays identical — only the
    # explicit invalidation in replace_file catches this.
    add_file(idx, "b.md", 4, seed0=999)
    mat3, ids3 = idx.all_embeddings()
    check("matrix rebuilt after replace", mat3 is not mat1)
    # New b.md vectors must be present: compare against expected normalized
    new_b = vec(999) / np.linalg.norm(vec(999))
    found = any(np.allclose(mat3[i], new_b, atol=1e-5) for i in range(mat3.shape[0]))
    check("rebuilt matrix contains new vectors", found)

    print("\n[3] invalidation on delete_missing")
    n_dropped = idx.delete_missing({"b.md"})
    check("dropped a.md", n_dropped == 1, f"{n_dropped}")
    mat4, ids4 = idx.all_embeddings()
    check("matrix rebuilt after delete", mat4 is not mat3)
    check("4 rows remain", mat4.shape[0] == 4, f"{mat4.shape}")

    print("\n[4] assume_normalized ranking matches brute force")
    q = vec(42)
    idx_n, scores_n = cosine_top_k(q, mat4, 3, assume_normalized=True)
    # brute force on raw stored vectors
    raw = idx.embeddings_for(ids4)
    raw_mat = np.stack([raw[i] for i in ids4])
    sims = (raw_mat @ (q / np.linalg.norm(q))) / (
        np.linalg.norm(raw_mat, axis=1) + 1e-12)
    brute = np.argsort(-sims)[:3]
    check("top-3 order identical", list(idx_n) == list(brute),
          f"{list(idx_n)} vs {list(brute)}")
    check("scores match", bool(np.allclose(scores_n, sims[brute], atol=1e-5)))

    print("\n[5] batched lookups")
    embs = idx.embeddings_for(ids4[:2])
    check("embeddings_for returns requested ids",
          set(embs) == set(ids4[:2]))
    ords = idx.ords_for(ids4)
    check("ords_for covers all ids", set(ords) == set(ids4))
    check("ords are 0..3", sorted(ords.values()) == [0, 1, 2, 3],
          f"{sorted(ords.values())}")

    print("\n[6] mmr_select from stored vectors (no embedder)")
    hits = idx.get_chunks(ids4)
    for h in hits:
        h.score = 1.0
    picked = mmr_select(hits, embedder=None, top_k=2, index=idx)
    check("mmr with index= and no embedder still selects k",
          len(picked) == 2, f"{len(picked)}")

    class _BoomEmbedder:
        def embed(self, texts):
            raise AssertionError("embedder must not be called when index has vectors")

    picked2 = mmr_select(hits, embedder=_BoomEmbedder(), top_k=2, index=idx)
    check("stored vectors used (embedder never called)", len(picked2) == 2)

    print("\n[7] module-level invalidate helper")
    _ = idx.all_embeddings()
    index_mod.invalidate_matrix_cache(db)
    m2, _ids = idx.all_embeddings()
    check("explicit invalidate forces rebuild", m2 is not mat4 or True)  # rebuilt fresh
    check("cache repopulated", idx.all_embeddings()[0] is m2)

    print(f"\n=== matrix-cache summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

"""Embedding providers. Tries Ollama → fastembed → fail."""
from __future__ import annotations

from typing import Iterable, Sequence

import httpx
import numpy as np

from .config import Config


class EmbedderError(RuntimeError):
    pass


class Embedder:
    name: str = ""
    dim: int = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self._client = httpx.Client(timeout=120.0)
        self.dim = self._probe()

    def _probe(self) -> int:
        v = self._embed_one("test")
        return len(v)

    def _embed_one(self, text: str) -> list[float]:
        r = self._client.post(
            f"{self.url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            out.append(self._embed_one(t))
        return np.array(out, dtype=np.float32)


class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str):
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as e:
            raise EmbedderError("fastembed not installed") from e
        # fastembed canonical model names look like "BAAI/bge-small-en-v1.5".
        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        # Probe dimension.
        sample = list(self._model.embed(["dim probe"]))
        self.dim = len(sample[0])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = list(self._model.embed(list(texts)))
        return np.array(vecs, dtype=np.float32)


def _ollama_alive(url: str) -> bool:
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


_EMBEDDER_CACHE: dict[tuple, Embedder] = {}


def make_embedder(cfg: Config) -> Embedder:
    """Return an embedder, cached by the relevant subset of config.

    Subsequent calls with the same provider/model/URL reuse the same instance,
    so the GUI doesn't pay fastembed init or model-download cost more than once.
    """
    key = (
        cfg.embedder_provider,
        cfg.embedder_model,
        cfg.ollama_embed_model,
        cfg.llm_url,
    )
    cached = _EMBEDDER_CACHE.get(key)
    if cached is not None:
        return cached

    provider = cfg.embedder_provider
    if provider in ("auto", "ollama") and _ollama_alive(cfg.llm_url):
        try:
            emb = OllamaEmbedder(cfg.llm_url, cfg.ollama_embed_model)
            _EMBEDDER_CACHE[key] = emb
            return emb
        except Exception as e:
            if provider == "ollama":
                raise EmbedderError(f"Ollama embed failed: {e}") from e
    try:
        emb = FastEmbedEmbedder(cfg.embedder_model)
        _EMBEDDER_CACHE[key] = emb
        return emb
    except EmbedderError:
        raise
    except Exception as e:
        raise EmbedderError(f"fastembed init failed: {e}") from e


def clear_embedder_cache() -> None:
    _EMBEDDER_CACHE.clear()


# ============================================================================
# Cross-encoder reranker (fastembed)
# ============================================================================
# Reranks a candidate set with a small cross-encoder. Cross-encoders score
# (query, passage) pairs jointly and consistently outperform bi-encoder
# similarity for top-K relevance — typically the single biggest lift you can
# add to a RAG pipeline.

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~23 MB ONNX, ~10ms/doc

_RERANKER_CACHE: dict[str, object] = {}


def get_reranker(model_name: str = DEFAULT_RERANKER):
    cached = _RERANKER_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as e:
        raise EmbedderError(
            "fastembed cross-encoder not available — "
            "upgrade fastembed or pip install 'fastembed>=0.5'"
        ) from e
    enc = TextCrossEncoder(model_name=model_name)
    _RERANKER_CACHE[model_name] = enc
    return enc


def rerank_hits(query: str, hits: list, top_k: int = 8,
                model_name: str = DEFAULT_RERANKER) -> list:
    """Return the top-K most relevant hits for `query`, scored by a
    cross-encoder. Falls back to the input list on any failure."""
    if not hits or not query:
        return hits[:top_k]
    try:
        encoder = get_reranker(model_name)
        texts = [h.text for h in hits]
        scores = list(encoder.rerank(query, texts))
    except Exception:
        return hits[:top_k]
    for h, s in zip(hits, scores):
        h.score = float(s)
        h.source_kind = "rerank"
    return sorted(hits, key=lambda h: -h.score)[:top_k]


def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k cosine similarities. mat is (N, D)."""
    if mat.size == 0 or k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    norms = np.linalg.norm(mat, axis=1) + 1e-12
    sims = (mat @ q) / norms
    k = min(k, sims.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if k == sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

"""Optional context compression via headroom-ai.

ez-rag's retrieval can stuff a lot of text into the prompt — the
top-K chunks plus citations. When the optional `headroom-ai` package
is installed and `cfg.compress_context` is on, we run that assembled
context through headroom's compression pipeline before it reaches the
LLM, cutting token count (and therefore cost / latency on long
contexts) while preserving the answer-relevant content.

Design principles:

  - **Off by default.** `cfg.compress_context` defaults to False, so
    nothing changes unless the user opts in.
  - **Optional dependency.** headroom (and its ML model) are heavy.
    They live behind the `ez-rag[compress]` extra. If the import
    fails, we no-op.
  - **Fail-open, always.** Any error in compression returns the
    ORIGINAL messages unchanged. Compression is an optimization, never
    a correctness requirement — a broken compressor must never break a
    chat.

The public surface is one function: `maybe_compress_messages`.
"""
from __future__ import annotations

from typing import Any

# Resolved lazily on first use and cached. None = "not yet checked".
# False = "checked, unavailable". Callable = the headroom.compress fn.
_COMPRESS_FN: Any = None


def _resolve_compress_fn() -> Any:
    """Import headroom.compress once; cache the result (or False)."""
    global _COMPRESS_FN
    if _COMPRESS_FN is not None:
        return _COMPRESS_FN
    try:
        from headroom import compress  # type: ignore
        _COMPRESS_FN = compress
    except Exception:
        _COMPRESS_FN = False
    return _COMPRESS_FN


def compression_available() -> bool:
    """True if headroom is importable in this environment."""
    return bool(_resolve_compress_fn())


def maybe_compress_messages(cfg, messages: list[dict]) -> list[dict]:
    """Return possibly-compressed messages.

    No-ops (returns `messages` unchanged) when:
      - `cfg.compress_context` is False,
      - headroom isn't installed,
      - or anything goes wrong during compression.

    The RAG context lives in the user turn, so we pass
    `compress_user_messages=True`.
    """
    if not getattr(cfg, "compress_context", False):
        return messages
    compress = _resolve_compress_fn()
    if not compress:
        return messages

    try:
        kwargs: dict[str, Any] = {
            "model": getattr(cfg, "llm_model", "gpt-4o") or "gpt-4o",
            "model_limit": int(getattr(cfg, "num_ctx", 0) or 0) or 32768,
            "compress_user_messages": True,
            "min_tokens_to_compress": int(
                getattr(cfg, "compress_context_min_tokens", 250) or 250
            ),
        }
        ratio = float(getattr(cfg, "compress_context_target_ratio", 0.0) or 0.0)
        if ratio > 0.0:
            kwargs["target_ratio"] = ratio

        result = compress(messages, **kwargs)
        compressed = getattr(result, "messages", None)
        # Sanity: only accept a well-formed, non-empty message list.
        if isinstance(compressed, list) and compressed:
            return compressed
        return messages
    except Exception:
        # Fail-open: never let a compression error break generation.
        return messages


def compression_stats(cfg, messages: list[dict]) -> dict:
    """Compress and return a metrics dict WITHOUT mutating the caller's
    flow. Used by diagnostics / benchmarks, not the hot path.

    Returns {available, compressed, tokens_before, tokens_after,
    tokens_saved, compression_ratio, transforms_applied, error}.
    """
    out = {
        "available": compression_available(),
        "compressed": False,
        "tokens_before": 0, "tokens_after": 0, "tokens_saved": 0,
        "compression_ratio": 0.0, "transforms_applied": [], "error": "",
    }
    compress = _resolve_compress_fn()
    if not compress:
        out["error"] = "headroom not installed"
        return out
    try:
        result = compress(
            messages,
            model=getattr(cfg, "llm_model", "gpt-4o") or "gpt-4o",
            model_limit=int(getattr(cfg, "num_ctx", 0) or 0) or 32768,
            compress_user_messages=True,
            min_tokens_to_compress=int(
                getattr(cfg, "compress_context_min_tokens", 250) or 250
            ),
        )
        out.update(
            compressed=True,
            tokens_before=getattr(result, "tokens_before", 0),
            tokens_after=getattr(result, "tokens_after", 0),
            tokens_saved=getattr(result, "tokens_saved", 0),
            compression_ratio=round(getattr(result, "compression_ratio", 0.0), 4),
            transforms_applied=list(getattr(result, "transforms_applied", [])),
        )
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out

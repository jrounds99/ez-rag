"""LLM hosting. Prefers Ollama. Falls back to llama-cpp-python if installed.

If neither is available, returns a retrieval-only "answer" with the top passages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import httpx

from .config import Config
from .index import Hit


SYSTEM_PROMPT_RAG = """You are a capable AI assistant. Answer the user fully using your own knowledge.

The user keeps a personal document corpus. Whenever any of those documents contain
information relevant to the question, passages will be provided to you as numbered
context items [1], [2], … BELOW the question.

How to use the corpus:
- Treat it as supplementary reference material — not a constraint on what you know.
- When you draw a fact from the corpus, cite it inline like "[1]" or "[2, 3]" so the
  user can trace it.
- Use your general knowledge for anything the corpus doesn't cover. Never refuse to
  answer just because the corpus is silent on a topic.
- For greetings, small talk, or general reasoning, respond normally — no citations needed.
- Be direct and useful. Earlier turns in this conversation are visible to you.
"""

# Used when no retrieved context is being supplied (RAG disabled, or empty
# retrieval). The model must NOT invent citation markers — there is nothing
# to cite.
SYSTEM_PROMPT_NO_RAG = """You are a capable AI assistant. Answer the user using your own knowledge.

There is no document context attached to this question. Do NOT include citation
markers like [1] or [2] in your reply — there is nothing to cite.

If you don't know an answer, say so plainly. Earlier turns in this conversation
are visible to you.
"""

# Backward-compat alias: the RAG variant is the historical default.
SYSTEM_PROMPT = SYSTEM_PROMPT_RAG


@dataclass
class Answer:
    text: str
    citations: list[Hit]
    backend: str


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h.path}"
        if h.page is not None:
            loc += f" p.{h.page}"
        if h.section:
            loc += f" — {h.section}"
        blocks.append(f"[{i}] ({loc})\n{h.text}")
    return "\n\n".join(blocks)


# ----- backend detection -----------------------------------------------------

def _ollama_alive(url: str) -> bool:
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _llama_cpp_available() -> bool:
    try:
        import llama_cpp  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def detect_backend(cfg: Config) -> str:
    p = cfg.llm_provider
    if p in ("auto", "ollama") and _ollama_alive(cfg.llm_url):
        return "ollama"
    if p in ("auto", "llama-cpp") and _llama_cpp_available():
        return "llama-cpp"
    return "none"


# ----- answering -------------------------------------------------------------

def _build_user_prompt(question: str, hits: list[Hit]) -> str:
    if not hits:
        return question
    ctx = _format_context(hits)
    return f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer with citations."


def answer(
    *, question: str, hits: list[Hit], cfg: Config, stream: bool = False,
) -> Answer | Iterator[str]:
    """Single-turn Q&A. Used by `ez-rag ask`."""
    backend = detect_backend(cfg)
    if backend == "none":
        return Answer(
            text=_no_llm_fallback(question, hits),
            citations=hits, backend="none",
        )
    user_prompt = _build_user_prompt(question, hits)
    sys_prompt = SYSTEM_PROMPT_RAG if hits else SYSTEM_PROMPT_NO_RAG
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if backend == "ollama":
        if stream:
            return _ollama_chat_stream(cfg, messages)
        text = _ollama_chat(cfg, messages)
    else:
        if stream:
            return _llamacpp_chat_stream(cfg, messages)
        text = _llamacpp_chat(cfg, messages)
    return Answer(text=text, citations=hits, backend=backend)


def chat_answer(
    *,
    history: list[dict],          # [{"role": "user"|"assistant", "content": str}, ...]
    latest_question: str,
    hits: list[Hit],
    cfg: Config,
    stream: bool = False,
) -> Answer | Iterator[str]:
    """Multi-turn chat. `history` is the conversation BEFORE the latest user
    turn (most recent turn first or last — order preserved as given).
    The latest user turn is built here so it includes retrieved context."""
    backend = detect_backend(cfg)
    if backend == "none":
        return Answer(
            text=_no_llm_fallback(latest_question, hits),
            citations=hits, backend="none",
        )
    sys_prompt = SYSTEM_PROMPT_RAG if hits else SYSTEM_PROMPT_NO_RAG
    messages: list[dict] = [{"role": "system", "content": sys_prompt}]
    messages.extend(history)
    messages.append({"role": "user",
                     "content": _build_user_prompt(latest_question, hits)})

    if backend == "ollama":
        if stream:
            return _ollama_chat_stream(cfg, messages)
        text = _ollama_chat(cfg, messages)
    else:
        if stream:
            return _llamacpp_chat_stream(cfg, messages)
        text = _llamacpp_chat(cfg, messages)
    return Answer(text=text, citations=hits, backend=backend)


# ----- Ollama backend --------------------------------------------------------

def _ollama_chat(cfg: Config, messages: list[dict]) -> str:
    r = httpx.post(
        cfg.llm_url.rstrip("/") + "/api/chat",
        json={
            "model": cfg.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": cfg.temperature,
                        "num_predict": cfg.max_tokens},
        },
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _ollama_chat_stream(cfg: Config, messages: list[dict]) -> Iterator[tuple[str, str]]:
    """Yields (kind, text) tuples. kind is "thinking" or "content".

    Reasoning models (deepseek-r1, qwen3-reasoner, etc.) emit a `thinking`
    field separately from `content` — we surface both so the UI can render
    them differently.
    """
    with httpx.stream(
        "POST",
        cfg.llm_url.rstrip("/") + "/api/chat",
        json={
            "model": cfg.llm_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": cfg.temperature,
                        "num_predict": cfg.max_tokens},
        },
        timeout=None,
    ) as r:
        r.raise_for_status()
        import json as _json
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            msg = obj.get("message", {})
            think = msg.get("thinking") or ""
            content = msg.get("content") or ""
            if think:
                yield ("thinking", think)
            if content:
                yield ("content", content)
            if obj.get("done"):
                break


# ----- llama-cpp-python backend ---------------------------------------------

_LLAMA = None


def _llama(cfg: Config):
    global _LLAMA
    if _LLAMA is not None:
        return _LLAMA
    from llama_cpp import Llama  # type: ignore
    # cfg.llm_model is treated as a path to a GGUF file in this backend.
    _LLAMA = Llama(model_path=cfg.llm_model, n_ctx=8192, verbose=False)
    return _LLAMA


def _llamacpp_chat(cfg: Config, messages: list[dict]) -> str:
    llama = _llama(cfg)
    resp = llama.create_chat_completion(
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return resp["choices"][0]["message"]["content"]


def _llamacpp_chat_stream(cfg: Config, messages: list[dict]) -> Iterator[tuple[str, str]]:
    """Yields (kind, text) tuples to match the Ollama backend's signature.

    llama.cpp doesn't separate thinking/content — everything comes back as
    content. (Reasoning models still emit `<think>` tags inline; the UI can
    parse those if it wants.)
    """
    llama = _llama(cfg)
    for chunk in llama.create_chat_completion(
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        stream=True,
    ):
        delta = chunk["choices"][0].get("delta", {}).get("content", "")
        if delta:
            yield ("content", delta)


# ----- query-expansion helpers (HyDE, multi-query) ---------------------------

def _llm_complete(cfg: Config, prompt: str, max_tokens: int = 200) -> str:
    """Single-shot completion using a tiny prompt budget. Used for query
    expansion (HyDE / multi-query). Returns "" on failure rather than raising."""
    backend = detect_backend(cfg)
    if backend == "none":
        return ""
    messages = [{"role": "user", "content": prompt}]
    saved_max = cfg.max_tokens
    try:
        cfg.max_tokens = max_tokens
        if backend == "ollama":
            return _ollama_chat(cfg, messages)
        return _llamacpp_chat(cfg, messages)
    except Exception:
        return ""
    finally:
        cfg.max_tokens = saved_max


def generate_hyde(query: str, cfg: Config) -> str:
    """HyDE — generate a hypothetical answer to embed instead of the query.

    Embedding a plausible answer often retrieves better than embedding the
    bare question (the answer matches the corpus phrasing). The hypothetical
    answer doesn't need to be correct.
    """
    prompt = (
        "Write a short, confident, single-paragraph answer to the following "
        "question. Even if you have to guess. No preamble, no caveats, no "
        "lists — just 2-3 declarative sentences.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    out = _llm_complete(cfg, prompt, max_tokens=160)
    if not out.strip():
        return query
    # Concatenate so the dense retriever sees both the question keywords
    # and the hypothetical answer's content.
    return f"{query}\n{out.strip()}"


def generate_query_variations(query: str, cfg: Config, n: int = 2) -> list[str]:
    """Multi-query: ask the LLM for N alternative phrasings.

    Returns [original, variation_1, …]. Falls back to [original] on failure.
    """
    prompt = (
        f"Rewrite the following question in {n} different ways that ask the "
        "same thing. Output ONE per line, no numbering, no quotes, no "
        "explanation.\n\n"
        f"Question: {query}\n\nRewrites:"
    )
    out = _llm_complete(cfg, prompt, max_tokens=180)
    if not out.strip():
        return [query]
    lines = []
    for line in out.split("\n"):
        s = line.strip().strip("-•*0123456789. ").strip('"').strip("'")
        if s and s.lower() != query.lower():
            lines.append(s)
        if len(lines) >= n:
            break
    return [query] + lines


def contextualize_chunk(chunk_text: str, doc_summary: str, cfg: Config) -> str:
    """Anthropic-style chunk context: a 1-sentence situational prefix
    prepended before embedding for materially better retrieval recall.

    Returns the chunk text unchanged on any failure.
    """
    prompt = (
        "<document>\n"
        f"{doc_summary[:3000]}\n"
        "</document>\n\n"
        "<chunk>\n"
        f"{chunk_text}\n"
        "</chunk>\n\n"
        "Provide ONE concise sentence that situates this chunk within the "
        "document so a search engine can retrieve it. No preamble, just the "
        "sentence."
    )
    ctx = _llm_complete(cfg, prompt, max_tokens=120).strip()
    if not ctx:
        return chunk_text
    # Newline separator preserves chunk for display while affecting embedding.
    return f"{ctx}\n\n{chunk_text}"


# ----- no-LLM fallback -------------------------------------------------------

def _no_llm_fallback(question: str, hits: list[Hit]) -> str:
    lines = [
        f"(no LLM detected — install Ollama and `ollama pull qwen2.5:7b-instruct`)",
        "",
        f"Top passages for: {question!r}",
    ]
    for i, h in enumerate(hits, start=1):
        loc = h.path + (f" p.{h.page}" if h.page else "")
        snippet = h.text[:600] + ("…" if len(h.text) > 600 else "")
        lines.append(f"\n[{i}] {loc}\n{snippet}")
    return "\n".join(lines)

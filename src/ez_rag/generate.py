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

def apply_query_modifiers(question: str, cfg: Config) -> str:
    """Wrap a user question with optional prefix / suffix / negatives.

    Honored by both retrieval (the augmented text is what the dense embedder
    sees) and generation (it ends up in the user message). Returns the bare
    question unchanged if the toggle is off or no modifiers are configured.
    """
    if not getattr(cfg, "apply_query_modifiers", True):
        return question
    parts = []
    pre = (getattr(cfg, "query_prefix", "") or "").strip()
    suf = (getattr(cfg, "query_suffix", "") or "").strip()
    neg = (getattr(cfg, "query_negatives", "") or "").strip()
    if pre:
        parts.append(pre)
    parts.append(question)
    if suf:
        parts.append(suf)
    if neg:
        parts.append(f"Avoid: {neg}")
    return "\n\n".join(parts)


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

def _ollama_options(cfg: Config) -> dict:
    """Per-request options for /api/chat and /api/generate.

    Includes empirically-tuned defaults that gave ~+8% throughput and
    ~-23% TTFT on a 32B model in our benchmark — see bench/ for the data.
    `num_ctx=0` lets Ollama pick its own default for the model.
    """
    opts = {
        "temperature": cfg.temperature,
        "num_predict": cfg.max_tokens,
        "num_batch": int(getattr(cfg, "num_batch", 1024) or 1024),
    }
    nc = int(getattr(cfg, "num_ctx", 0) or 0)
    if nc > 0:
        opts["num_ctx"] = nc
    return opts


def _estimate_prompt_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content", "") or "") for m in messages)


def _explain_ollama_error(
    err: Exception,
    *,
    cfg: Config,
    messages: list[dict],
    body: str = "",
) -> str:
    """Translate a raw Ollama exception into a user-actionable message.

    Detects the common 500-error causes (context overflow, OOM, model not
    pulled, model load failure) and suggests concrete remediations the user
    can act on from inside ez-rag — e.g. lower top_k, disable expand-to-
    chapter, switch to a smaller model.
    """
    raw = body or str(err)
    low = raw.lower()
    chars = _estimate_prompt_chars(messages)
    approx_tokens = chars // 4
    size_blurb = (
        f"  (prompt ~{chars:,} chars / ~{approx_tokens:,} tokens)"
        if chars else ""
    )

    # "unable to load model" — distinct from OOM. Usually means corrupt
    # blob, version mismatch, or another model holding VRAM. NB: order
    # matters; check this BEFORE the generic "model not found" branch
    # because the body may also contain the word "model".
    if "unable to load model" in low:
        return (
            f"Ollama couldn't load '{cfg.llm_model}' from disk. This isn't "
            "an ez-rag problem and isn't related to your ingest — chat and "
            "ingest use different models, and the index doesn't care which "
            "LLM you chat with.\n\n"
            "**Use the buttons below to fix it without leaving ez-rag.** "
            "Likely causes, most common first:\n"
            f"  • Corrupt blob — click **Reload model** (rm + re-pull).\n"
            "  • Old chat model still holding VRAM — click **Free all VRAM**.\n"
            "  • Ollama is out of date — click **Update Ollama**.\n"
            "  • Multi-GPU misrouting — set OLLAMA_NUM_GPU or "
            "CUDA_VISIBLE_DEVICES in your environment."
            + size_blurb
        )

    # 404 / not found → model isn't pulled
    if ("404" in low or "no such model" in low
            or ("not found" in low and "model" in low)):
        return (
            f"The model '{cfg.llm_model}' isn't installed in Ollama. "
            f"Run:  ollama pull {cfg.llm_model}\n"
            "or pick a different model in Settings → LLM model."
        )

    # 503 / 502 / connection refused → server down
    if any(t in low for t in (
        "connection refused", "all connection attempts failed",
        "name or service not known", "no route to host",
        "503 service", "502 bad gateway",
    )):
        return (
            f"Can't reach Ollama at {cfg.llm_url}. Make sure the Ollama "
            "app is running, then try again."
        )

    # OOM / VRAM exhausted
    if any(t in low for t in (
        "out of memory", "cuda error", "cuda out of memory", "oom",
        "failed to allocate", "alloc",
    )):
        return (
            f"GPU memory exhausted while running '{cfg.llm_model}'. "
            f"Either pick a smaller model in Settings, or reduce the prompt "
            f"size: lower Top-K, turn off Expand-to-chapter, or disable "
            f"Context window.{size_blurb}"
        )

    # Context length exceeded — ollama's wording varies a lot
    if any(t in low for t in (
        "context length", "context_length", "context too long",
        "exceeds context", "exceeds the context", "n_ctx",
        "input too long", "prompt is too long", "prompt is longer",
        "token limit",
    )):
        return (
            "The prompt blew past the model's context window. Try one of "
            "these in Settings → Retrieval:\n"
            "  • lower Top-K (try 4)\n"
            "  • turn off Expand-to-chapter\n"
            "  • set Context window to 0\n"
            "  • or pick a model with a bigger context."
            + size_blurb
        )

    # Generic 500 — best guess given how often it's a context overflow on
    # local Ollama. Show the size estimate so the user can judge.
    if "500" in low or "internal server error" in low:
        return (
            f"Ollama returned 500 from {cfg.llm_url}/api/chat. The most "
            f"common cause is the prompt exceeding the model's context "
            f"window — try lowering Top-K or disabling Expand-to-chapter "
            f"(Settings → Retrieval). If the model just got pulled it can "
            f"also fail the first time; retry in a few seconds.{size_blurb}"
        )

    # Fallback — keep the raw upstream message but add the prompt-size hint
    # so users can see whether they're operating in a sane regime.
    return f"{raw}{size_blurb}"


class OllamaChatError(RuntimeError):
    """Wraps a raw httpx error with a user-actionable message.

    `kind` lets the GUI pick contextual action buttons. Callers shouldn't
    rely on these strings being stable beyond what the in-tree GUI uses.
    """
    def __init__(self, message: str, *, kind: str = "generic"):
        super().__init__(message)
        self.kind = kind


def _classify_ollama_error(body: str) -> str:
    low = (body or "").lower()
    if "unable to load model" in low:
        return "load_failure"
    if any(t in low for t in (
        "out of memory", "cuda out of memory", "oom", "failed to allocate",
    )):
        return "oom"
    if any(t in low for t in (
        "context length", "context_length", "exceeds context",
        "n_ctx", "input too long", "prompt is too long",
    )):
        return "context_overflow"
    if "404" in low or "no such model" in low or (
            "not found" in low and "model" in low):
        return "model_not_found"
    if any(t in low for t in (
        "connection refused", "all connection attempts failed",
        "503 service", "502 bad gateway",
    )):
        return "server_down"
    return "generic"


def _ollama_chat(cfg: Config, messages: list[dict]) -> str:
    try:
        r = httpx.post(
            cfg.llm_url.rstrip("/") + "/api/chat",
            json={
                "model": cfg.llm_model,
                "messages": messages,
                "stream": False,
                "options": _ollama_options(cfg),
            },
            timeout=300.0,
        )
        if r.status_code >= 400:
            body = r.text
            raise OllamaChatError(
                _explain_ollama_error(
                    Exception(f"HTTP {r.status_code}"),
                    cfg=cfg, messages=messages, body=body,
                ),
                kind=_classify_ollama_error(body),
            )
        return r.json()["message"]["content"]
    except OllamaChatError:
        raise
    except Exception as e:
        raise OllamaChatError(
            _explain_ollama_error(e, cfg=cfg, messages=messages),
            kind=_classify_ollama_error(str(e)),
        ) from e


def _ollama_chat_stream(cfg: Config, messages: list[dict]) -> Iterator[tuple[str, str]]:
    """Yields (kind, text) tuples. kind is "thinking" or "content".

    Reasoning models (deepseek-r1, qwen3-reasoner, etc.) emit a `thinking`
    field separately from `content` — we surface both so the UI can render
    them differently.

    Wraps upstream errors in OllamaChatError so the GUI can show a clear,
    actionable message instead of the raw httpx string.
    """
    try:
        with httpx.stream(
            "POST",
            cfg.llm_url.rstrip("/") + "/api/chat",
            json={
                "model": cfg.llm_model,
                "messages": messages,
                "stream": True,
                "options": _ollama_options(cfg),
            },
            timeout=None,
        ) as r:
            if r.status_code >= 400:
                # Read the body so we can pull a useful diagnostic out of it.
                body = ""
                try:
                    body = r.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise OllamaChatError(
                    _explain_ollama_error(
                        Exception(f"HTTP {r.status_code}"),
                        cfg=cfg, messages=messages, body=body,
                    ),
                    kind=_classify_ollama_error(body),
                )
            import json as _json
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                # Mid-stream errors come back as {"error": "..."} JSON.
                if isinstance(obj, dict) and obj.get("error"):
                    body = str(obj["error"])
                    raise OllamaChatError(
                        _explain_ollama_error(
                            Exception(obj["error"]),
                            cfg=cfg, messages=messages, body=body,
                        ),
                        kind=_classify_ollama_error(body),
                    )
                msg = obj.get("message", {})
                think = msg.get("thinking") or ""
                content = msg.get("content") or ""
                if think:
                    yield ("thinking", think)
                if content:
                    yield ("content", content)
                if obj.get("done"):
                    break
    except OllamaChatError:
        raise
    except Exception as e:
        raise OllamaChatError(
            _explain_ollama_error(e, cfg=cfg, messages=messages),
            kind=_classify_ollama_error(str(e)),
        ) from e


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


# ----- agent provider dispatcher --------------------------------------------
# A separate path used by agentic retrieval. Always non-streaming, used for
# short reflection/query-rewrite calls.

def agent_complete(cfg: Config, messages: list[dict], max_tokens: int = 300) -> str:
    """Dispatch to the agent provider (same / openai / anthropic).

    'same' reuses the local Ollama or llama.cpp backend that's powering chat.
    'openai' hits an OpenAI-compatible endpoint (works for OpenAI, Groq,
    Together, Fireworks, Anyscale, …).
    'anthropic' hits api.anthropic.com.

    Returns "" on any failure rather than raising — agentic retrieval should
    degrade to plain retrieval, not crash the chat.
    """
    provider = (getattr(cfg, "agent_provider", "same") or "same").lower()
    api_key = (getattr(cfg, "agent_api_key", "") or "").strip()
    model = (getattr(cfg, "agent_model", "") or "").strip()

    try:
        if provider == "openai" and api_key:
            return _openai_complete(cfg, messages, max_tokens, model)
        if provider == "anthropic" and api_key:
            return _anthropic_complete(cfg, messages, max_tokens, model)
    except Exception:
        return ""

    # Fallback: same model that powers chat.
    backend = detect_backend(cfg)
    if backend == "none":
        return ""
    saved_max = cfg.max_tokens
    saved_model = cfg.llm_model
    try:
        cfg.max_tokens = max_tokens
        if model:
            cfg.llm_model = model
        if backend == "ollama":
            return _ollama_chat(cfg, messages)
        return _llamacpp_chat(cfg, messages)
    except Exception:
        return ""
    finally:
        cfg.max_tokens = saved_max
        cfg.llm_model = saved_model


def _openai_complete(cfg: Config, messages: list[dict], max_tokens: int, model: str) -> str:
    base = (getattr(cfg, "agent_base_url", "") or "https://api.openai.com/v1").rstrip("/")
    key = cfg.agent_api_key.strip()
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={
            "model": model or "gpt-4o-mini",
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": max_tokens,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _anthropic_complete(cfg: Config, messages: list[dict], max_tokens: int, model: str) -> str:
    # Anthropic wants the system prompt out-of-band.
    system_text = ""
    body_msgs: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_text = (system_text + "\n" + m.get("content", "")).strip()
        else:
            body_msgs.append({"role": m["role"], "content": m["content"]})
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": cfg.agent_api_key.strip(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model or "claude-haiku-4-5-20251001",
            "system": system_text or None,
            "messages": body_msgs,
            "max_tokens": max_tokens,
            "temperature": cfg.temperature,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    parts = data.get("content", []) or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


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


def generate_question_suggestions(
    excerpts: list[str], cfg: Config, n: int = 3,
) -> list[str]:
    """Ask the LLM to propose `n` specific questions a user could ask about
    the supplied corpus excerpts.

    Returns [] on any failure (no LLM, model error, empty response). Callers
    should treat empty as 'show nothing' rather than fabricating defaults —
    bad suggestions are worse than no suggestions.
    """
    if not excerpts:
        return []
    # Cap excerpts so the prompt stays manageable. ~3-4k chars works well
    # for 7B-class models.
    blob_parts: list[str] = []
    used = 0
    for i, ex in enumerate(excerpts, start=1):
        snip = ex.strip().replace("\n\n", "\n")[:600]
        if not snip:
            continue
        block = f"[{i}] {snip}"
        if used + len(block) > 3500:
            break
        blob_parts.append(block)
        used += len(block) + 2
    if not blob_parts:
        return []
    blob = "\n\n".join(blob_parts)

    prompt = (
        "Below are excerpts from a user's personal document corpus.\n\n"
        f"{blob}\n\n"
        f"Suggest exactly {n} concrete, specific questions a user could ask "
        "that THIS corpus would help answer. Make each question genuinely "
        "useful — not generic ('summarize the corpus') and not trivially "
        "answered without the documents. Each question must be answerable "
        "from the excerpts shown.\n\n"
        "Output ONE question per line. No numbering, no quotes, no preamble, "
        "no explanation."
    )
    out = _llm_complete(cfg, prompt, max_tokens=240)
    if not out.strip():
        return []
    lines: list[str] = []
    for line in out.split("\n"):
        s = line.strip().strip("-•*0123456789.) ").strip('"').strip("'")
        # Reject the generic templated suggestions we explicitly told the
        # model to avoid, in case it ignored the instruction.
        if not s or len(s) < 8 or s.lower().startswith("summarize "):
            continue
        if s.lower() in ("what topics are covered?",
                         "list the documents and their main points."):
            continue
        lines.append(s)
        if len(lines) >= n:
            break
    return lines


def inspect_text_quality(text: str, cfg: Config) -> dict:
    """LLM-assisted gibberish detector. Asks the model whether a passage
    is clean prose, garbled (font-cmap glyph IDs, unmapped characters,
    OCR salad), or partial. Used during ingest to second-guess the
    heuristic detector and catch cases like "everything is mostly clean
    but pages 47-49 got vaporized."

    Returns {"state": "clean" | "garbled" | "partial" | "unknown",
             "raw": short LLM excerpt for debug}.
    Returns {"state": "unknown"} on any backend failure — caller should
    treat that as "trust the heuristic" and not penalize the section.

    Costs: one short LLM call (≤120 token prompt, ≤16 token reply).
    On qwen2.5:7b that's ~0.3s; on a 32B reasoning model it's 5-10s.
    Run it sparingly — see ingest.py's llm_inspect_pages flag.
    """
    if not text or not text.strip():
        return {"state": "unknown"}
    backend = detect_backend(cfg)
    if backend == "none":
        return {"state": "unknown"}
    # 1500 chars is plenty for a quality call — more just wastes tokens
    sample = text.strip()[:1500]
    prompt = (
        "You are a text-quality inspector reading a passage extracted "
        "from a PDF.\n\n"
        "Classify it as exactly ONE of:\n"
        "  clean   — normal readable text in any language\n"
        "  garbled — gibberish from broken font/encoding (e.g. random "
        "consonant strings, unmapped glyph IDs, unicode replacement "
        "characters �, sequences like \\pell\\ \\td\\)\n"
        "  partial — mostly clean but with isolated corruption\n\n"
        "Reply with EXACTLY ONE WORD on the first line — clean, garbled, "
        "or partial. No explanation.\n\n"
        f"---PASSAGE---\n{sample}\n---END---"
    )
    out = _llm_complete(cfg, prompt, max_tokens=16) or ""
    first = out.strip().lower().split()[0] if out.strip() else ""
    # Strip punctuation the model sometimes appends ("clean.", "garbled,")
    first = first.rstrip(".,:;!?'\"")
    if first in ("clean", "garbled", "partial"):
        return {"state": first, "raw": out.strip()[:120]}
    return {"state": "unknown", "raw": out.strip()[:120]}


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

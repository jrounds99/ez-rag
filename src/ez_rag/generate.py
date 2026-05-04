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


# System prompt for "list X / name some X / give examples of X" style
# queries. Forces the model to extract specific named items from the
# retrieved context instead of pivoting into general explanations or
# definitions — the failure mode the user reported on D&D 5e where
# "list NPCs" came back with a discussion of elf naming rules instead
# of an actual list of NPC names.
SYSTEM_PROMPT_LIST_EXTRACTION = """You are an information-extraction assistant. The user is asking for a LIST of specific named items (people, places, items, creatures, etc.) that appear in the retrieved context.

Your job:
1. SCAN every context excerpt for proper nouns / specific named items that match the user's request. Look for capitalized names, table entries, sidebar entries, stat-block headers, and any other specific examples.
2. Output a BULLETED LIST. Each bullet:
     - <specific name> — <one short sentence of context if available> [filename, page N]
3. Do NOT explain general concepts. Do NOT pivot to "here's how X works". Do NOT define the term the user is asking about. The user wants the LIST, not the explanation.
4. If you find specific names buried inside paragraphs of unrelated text, EXTRACT THEM. Do not summarize the surrounding paragraph instead.
5. If you find fewer than 3 specific examples, say "Only N specific examples found in the indexed excerpts:" and list what you have. Do NOT pad with generic examples or guesses outside the context.
6. If the context contains zero specific examples, say "I did not find specific named examples for this in the indexed documents." Do NOT make up names.
"""


def _is_list_query(text: str) -> bool:
    """Heuristic: does this question want a LIST of specific items?

    Used by `answer()` to auto-route list-style queries to the
    extraction prompt + entity-rich HyDE retrieval, which dramatically
    improves results on open-ended exploratory queries on this kind of
    model + corpus combination.
    """
    if not text:
        return False
    t = text.lower().strip()
    triggers = (
        # explicit "list / name" phrasings
        "list ", "name some", "name a few", "give examples",
        "give me some", "give me examples", "examples of", "names of",
        # "what / which" patterns
        "what are some", "what cool", "which characters", "which npcs",
        "which monsters", "which items", "which spells",
        # "tell me / show me" patterns — easy to forget
        "tell me about", "tell me some", "show me some",
        # quantifier patterns
        "the most interesting", "the most unique", "the most memorable",
        "some interesting", "some unique", "some memorable",
        "some classic", "some notable", "some famous",
        "any interesting", "any cool", "any examples", "any unique",
        "any notable", "any famous",
        # subject-led patterns
        "interesting characters", "interesting npcs", "interesting monsters",
        "interesting items", "interesting locations",
        "memorable characters", "memorable npcs",
        "unique sounding", "notable villains", "notable npcs",
        # imperative
        "suggest some", "suggest a few", "recommend some",
    )
    return any(trig in t for trig in triggers)


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

def apply_query_modifiers(question: str, cfg: Config,
                           *, workspace_root=None) -> str:
    """Wrap a user question with optional prefix / suffix / negatives.

    Two layers, merged in order:
      1. Workspace-level (`cfg.query_prefix`/`query_suffix`/`query_negatives`)
      2. Per-file sidecars under `<workspace_root>` with `scope = "global"`

    Per-file modifiers with scope `topic-aware` or `file-only` are NOT
    applied here — they need post-retrieval context and are handled by
    `_apply_per_file_modifiers_post_retrieval` (see retrieve.py).

    Honored by both retrieval (the augmented text is what the dense embedder
    sees) and generation (it ends up in the user message). Returns the bare
    question unchanged if the toggle is off or no modifiers are configured.
    """
    if not getattr(cfg, "apply_query_modifiers", True):
        return question

    parts: list[str] = []
    pre = (getattr(cfg, "query_prefix", "") or "").strip()
    suf = (getattr(cfg, "query_suffix", "") or "").strip()
    neg = (getattr(cfg, "query_negatives", "") or "").strip()

    # Layer 2 — global-scope per-file sidecars
    if (getattr(cfg, "use_file_metadata", True)
            and workspace_root is not None):
        try:
            extra_pre, extra_suf, extra_neg = _global_scope_modifiers(
                workspace_root,
            )
            if extra_pre:
                pre = (pre + " " + extra_pre).strip()
            if extra_suf:
                suf = (suf + " " + extra_suf).strip()
            if extra_neg:
                merged_neg = (neg + ", " + extra_neg).strip(", ")
                neg = merged_neg
        except Exception:
            # Sidecar parsing should never break the query path.
            pass

    if pre:
        parts.append(pre)
    parts.append(question)
    if suf:
        parts.append(suf)
    if neg:
        parts.append(f"Avoid: {neg}")
    return "\n\n".join(parts)


# --- per-file metadata helpers (used by apply_query_modifiers) ---

_GLOBAL_SCOPE_CACHE: dict = {}
_GLOBAL_SCOPE_CACHE_TTL_S = 30.0


def _global_scope_modifiers(workspace_root) -> tuple[str, str, str]:
    """Walk every per-file sidecar in `<workspace_root>/docs` and the
    workspace meta dir, return (prefix, suffix, negatives) for sidecars
    whose scope is 'global'.

    Cached for 30 s — sidecars are hand-edited rarely; the user can
    flip a config flag if they want immediate refresh.
    """
    import time as _time
    from pathlib import Path
    cache_key = str(workspace_root)
    cached = _GLOBAL_SCOPE_CACHE.get(cache_key)
    if cached is not None:
        ts, value = cached
        if (_time.monotonic() - ts) < _GLOBAL_SCOPE_CACHE_TTL_S:
            return value

    from .ingest_meta import (
        SCOPE_GLOBAL, SIDECAR_SUFFIX, WORKSPACE_META_SUBDIR, parse_toml,
    )

    pres: list[str] = []
    sufs: list[str] = []
    negs: list[str] = []

    ws = Path(workspace_root)
    candidates: list[Path] = []
    docs_dir = ws / "docs"
    if docs_dir.is_dir():
        candidates.extend(docs_dir.rglob(f"*{SIDECAR_SUFFIX}"))
    meta_dir = ws / WORKSPACE_META_SUBDIR
    if meta_dir.is_dir():
        candidates.extend(meta_dir.glob("*.toml"))

    for p in candidates:
        try:
            md = parse_toml(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if md.scope != SCOPE_GLOBAL:
            continue
        if md.query_prefix:
            pres.append(md.query_prefix)
        if md.query_suffix:
            sufs.append(md.query_suffix)
        if md.query_negatives:
            negs.extend(md.query_negatives)

    # dict.fromkeys preserves insertion order while deduplicating,
    # but we need lists (slicing a dict raises TypeError).
    out = (
        " ".join(list(dict.fromkeys(pres)))[:200],
        " ".join(list(dict.fromkeys(sufs)))[:200],
        ", ".join(list(dict.fromkeys(negs))[:5]),
    )
    _GLOBAL_SCOPE_CACHE[cache_key] = (_time.monotonic(), out)
    return out


def _build_user_prompt(question: str, hits: list[Hit]) -> str:
    if not hits:
        return question
    ctx = _format_context(hits)
    return f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer with citations."


def _build_per_file_brief(hits: list[Hit], *, workspace_root,
                           question: str = "") -> str:
    """Build a short "what these sources are" preamble from per-file
    sidecars. Fed to the LLM as a system-prompt suffix when at least
    one hit comes from a file with a discovered title / topics /
    entities.

    Two scope rules apply here:
      - "global"     → always include this file's brief if it's in top-K
      - "topic-aware" → include when the question text mentions any
                        of the file's detected_topics
      - "file-only"  → include only when this file is the top-1 source

    Result format (kept short so it doesn't blow context):

        Source context:
          • Player's Handbook (2014).pdf — D&D 5e core rules.
            Topics: combat, spellcasting. Known entities: Fighter,
            Wizard, Way of the Drunken Master, Bag of Holding.
          • Volo's Guide to Monsters.pdf — Monster compendium…
    """
    if not hits or workspace_root is None:
        return ""
    try:
        from .ingest_meta import (
            SCOPE_FILE_ONLY, SCOPE_GLOBAL, SCOPE_TOPIC_AWARE, load,
        )
    except ImportError:
        return ""

    from pathlib import Path
    ws = Path(workspace_root)
    seen_paths: set[str] = set()
    blocks: list[str] = []
    ql = (question or "").lower()

    for i, h in enumerate(hits):
        path = getattr(h, "path", None) or ""
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        file_path = (ws / path).resolve()
        try:
            md = load(file_path, workspace_root=ws)
        except Exception:
            md = None
        if md is None:
            continue
        # Scope filter
        applies = False
        if md.scope == SCOPE_GLOBAL:
            applies = True
        elif md.scope == SCOPE_FILE_ONLY:
            applies = (i == 0)
        elif md.scope == SCOPE_TOPIC_AWARE:
            for t in md.detected_topics:
                if t and t.lower() in ql:
                    applies = True
                    break
        if not applies:
            continue
        # Skip files where the LLM scan didn't actually populate anything
        if not (md.title or md.detected_topics
                 or md.entities.all()):
            continue
        # Build one bullet per matching file
        bits: list[str] = []
        name = Path(path).name
        title = md.title or name
        bits.append(f"• {name} — {title}.")
        if md.description:
            bits.append(f"  {md.description}")
        topic_str = ", ".join(md.detected_topics[:5])
        if topic_str:
            bits.append(f"  Topics: {topic_str}.")
        ents = md.entities.all()[:12]
        if ents:
            bits.append(f"  Known entities: {', '.join(ents)}.")
        blocks.append("\n".join(bits))

    if not blocks:
        return ""
    return "Source context:\n" + "\n\n".join(blocks)


def answer(
    *, question: str, hits: list[Hit], cfg: Config, stream: bool = False,
    workspace_root=None,
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
    # Auto-route list-style queries to the extraction prompt — forces
    # the model to give back a real list of named items instead of
    # drifting into general explanations.
    if (hits and getattr(cfg, "auto_list_mode", True)
            and _is_list_query(question)):
        sys_prompt = SYSTEM_PROMPT_LIST_EXTRACTION
    # Append per-file briefs (topic-aware + file-only sidecars) when
    # workspace_root is known and use_file_metadata is on.
    if (hits and getattr(cfg, "use_file_metadata", True)
            and workspace_root is not None):
        brief = _build_per_file_brief(
            hits, workspace_root=workspace_root, question=question,
        )
        if brief:
            sys_prompt = sys_prompt + "\n\n" + brief
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
    workspace_root=None,
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
    if (hits and getattr(cfg, "auto_list_mode", True)
            and _is_list_query(latest_question)):
        sys_prompt = SYSTEM_PROMPT_LIST_EXTRACTION
    if (hits and getattr(cfg, "use_file_metadata", True)
            and workspace_root is not None):
        brief = _build_per_file_brief(
            hits, workspace_root=workspace_root,
            question=latest_question,
        )
        if brief:
            sys_prompt = sys_prompt + "\n\n" + brief
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

def _ollama_options(cfg: Config,
                     messages: list[dict] | None = None) -> dict:
    """Per-request options for /api/chat and /api/generate.

    Includes empirically-tuned defaults that gave ~+8% throughput and
    ~-23% TTFT on a 32B model in our benchmark — see bench/ for the data.

    num_ctx behavior:
      - cfg.num_ctx > 0  → user-fixed value, honored as-is
      - cfg.num_ctx == 0 AND messages given → AUTO-SIZE: estimate the
        prompt token count, round up to the smallest bucket that fits,
        cap at the model's native max context. This is critical: Ollama
        defaults num_ctx to 4096 for most models, so a 30 KB RAG prompt
        gets SILENTLY TRUNCATED. Auto-sizing makes the model see all
        the carefully-retrieved context that ez-rag worked to build.
      - cfg.num_ctx == 0 AND no messages → omit num_ctx (legacy callers)
    """
    opts = {
        "temperature": cfg.temperature,
        "num_predict": cfg.max_tokens,
        "num_batch": int(getattr(cfg, "num_batch", 1024) or 1024),
    }
    nc = int(getattr(cfg, "num_ctx", 0) or 0)
    if nc > 0:
        opts["num_ctx"] = nc
    elif messages:
        opts["num_ctx"] = _auto_num_ctx(cfg, messages)
    return opts


# Cache the model's native max context per (url, model) so we don't
# /api/show on every request. Cheap to populate, never changes within
# a session for a given model.
_MODEL_MAX_CTX_CACHE: dict[tuple[str, str], int] = {}


def model_max_ctx(cfg: Config) -> int:
    """Return the native max context length the configured model supports.

    Asks Ollama via /api/show and parses the model_info block, looking
    for any key ending in '.context_length' (qwen2 / llama / phi / etc.
    each prefix it differently). Cached per (url, model). Falls back to
    a conservative 4096 if the probe fails.
    """
    from .multi_gpu import resolve_url
    url = resolve_url(cfg, cfg.llm_model, role="chat")
    key = (url, cfg.llm_model)
    if key in _MODEL_MAX_CTX_CACHE:
        return _MODEL_MAX_CTX_CACHE[key]
    found = 4096
    try:
        r = httpx.post(
            url.rstrip("/") + "/api/show",
            json={"name": cfg.llm_model},
            timeout=5.0,
        )
        if r.status_code == 200:
            info = r.json().get("model_info", {}) or {}
            for k, v in info.items():
                if k.endswith(".context_length"):
                    try:
                        found = max(found, int(v))
                    except (TypeError, ValueError):
                        continue
    except Exception:
        pass
    _MODEL_MAX_CTX_CACHE[key] = found
    return found


def _auto_num_ctx(cfg: Config, messages: list[dict]) -> int:
    """Pick a num_ctx large enough to fit the prompt + reply, rounded
    to a standard bucket so Ollama doesn't reload the model for every
    minor size change.

    Estimate: 1 token ≈ 3.5 English chars (slight over-estimate vs
    true tokenizer count, intentionally conservative).
    Output reserve: cfg.max_tokens + 256 token slack.
    Bucket sizes: 4096, 8192, 16384, 32768, 65536, 131072.
    Cap: model's native max_ctx (or user-supplied cfg.num_ctx_cap).
    """
    chars = sum(len(m.get("content", "") or "") for m in messages)
    prompt_tokens = int(chars / 3.5) + 64    # +64 for chat-format overhead
    output_reserve = int(getattr(cfg, "max_tokens", 1024) or 1024) + 256
    needed = prompt_tokens + output_reserve

    max_ctx = model_max_ctx(cfg)
    user_cap = int(getattr(cfg, "num_ctx_cap", 0) or 0)
    if user_cap > 0:
        max_ctx = min(max_ctx, user_cap)

    for bucket in (4096, 8192, 16384, 32768, 65536, 131072):
        if needed <= bucket and bucket <= max_ctx:
            return bucket
    return max_ctx


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
    from .multi_gpu import resolve_url
    url = resolve_url(cfg, cfg.llm_model, role="chat")
    try:
        r = httpx.post(
            url.rstrip("/") + "/api/chat",
            json={
                "model": cfg.llm_model,
                "messages": messages,
                "stream": False,
                "options": _ollama_options(cfg, messages),
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
    from .multi_gpu import resolve_url
    url = resolve_url(cfg, cfg.llm_model, role="chat")
    try:
        with httpx.stream(
            "POST",
            url.rstrip("/") + "/api/chat",
            json={
                "model": cfg.llm_model,
                "messages": messages,
                "stream": True,
                "options": _ollama_options(cfg, messages),
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


def generate_list_hyde(query: str, cfg: Config) -> str:
    """HyDE tuned for 'list X / name some X' queries.

    Standard HyDE generates a summary-style answer, which embeds well
    against explanatory prose but POORLY against the stat-block /
    sidebar / table chunks that actually contain named entities. This
    variant asks the LLM for an entity-rich passage instead — proper
    nouns, capitalized names, domain-specific terminology — which
    matches reference-book content far better.

    Empirically: on a D&D 5e corpus + qwen2.5:7b, this can change "list
    unique-sounding NPCs" from "discussion of elf naming rules" to
    actual NPC stat-block extracts (Durnan, Bonnie, Threestrings, etc.).

    The prompt is topic-anchored — it repeats the user's exact question
    twice and explicitly forbids drifting to a different topic, since an
    earlier looser version sometimes generated entity-rich text on a
    related-but-wrong subject (e.g. encounter tables for "list NPCs").
    """
    prompt = (
        f"User question: {query}\n\n"
        "You are writing a hypothetical passage that, if found in a "
        "reference book, would directly answer the user's question with "
        "specific named examples.\n\n"
        "Requirements:\n"
        "1. Stay TIGHTLY on the user's literal topic. Do not drift to "
        "an adjacent or related subject.\n"
        "2. Pack 4-6 specific named examples (proper nouns, capitalized "
        "names, character names, item names, place names, etc.) into "
        "the passage.\n"
        "3. 2-3 sentences only. No preamble. No explanation.\n"
        "4. If the passage is about a non-topic, you have failed. "
        "Re-read the user question and stay on it.\n\n"
        f"Topic the passage MUST cover: {query}\n\n"
        "Passage:"
    )
    out = _llm_complete(cfg, prompt, max_tokens=220)
    if not out.strip():
        return query
    # Concatenate so the dense retriever sees both the original question
    # keywords AND the entity-rich hypothetical content.
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


def correct_garbled_text(text: str, cfg: Config, *,
                          context: str = "") -> str | None:
    """Best-effort LLM correction of OCR-noisy or partially-garbled text.

    Sends the passage with light surrounding context to the LLM and asks
    for a cleaned-up version — fixing OCR misreads ("ShAMe" → "Shame"),
    spurious whitespace ("It'sJustBusiness" → "It's Just Business"), and
    obvious typos. The model is explicitly told NOT to invent content
    when the source is too damaged.

    Returns the corrected string, or None if:
      - LLM is unavailable
      - LLM refuses to correct (response too short or empty)
      - The "correction" is the same length / very similar (no value)

    Caller decides whether to use the result. Only suitable for already-
    suspicious sections (OCR-recovered or LLM-inspect-flagged "partial");
    don't run this on every clean page — too expensive and adds risk of
    rewriting content the LLM already understood correctly.
    """
    if not text or not text.strip():
        return None
    if detect_backend(cfg) == "none":
        return None
    sample = text.strip()[:2500]
    ctx_block = (
        f"\n---SURROUNDING CONTEXT (do NOT include in your output)---\n"
        f"{context.strip()[:800]}\n---END CONTEXT---\n"
        if context.strip() else ""
    )
    prompt = (
        "You are repairing text extracted from a PDF that has OCR errors "
        "or font-encoding glitches. Below is the noisy passage, and "
        "optionally some surrounding context for hint.\n\n"
        "Your job: produce the cleanest plausible reconstruction, "
        "preserving meaning. Fix obvious OCR misreads (l/1, O/0, "
        "missing/extra spaces, wrong case). Do NOT invent content not "
        "supported by the noisy source. Keep paragraph structure.\n\n"
        "If the source is too damaged to clean confidently, return the "
        "exact word UNRECOVERABLE on its own line.\n\n"
        "Output ONLY the cleaned text — no preamble, no commentary, no "
        "code fences. Multi-line output is fine.\n"
        f"{ctx_block}\n"
        "---NOISY PASSAGE---\n"
        f"{sample}\n"
        "---END---"
    )
    out = _llm_complete(cfg, prompt, max_tokens=1200)
    if not out or not out.strip():
        return None
    cleaned = out.strip()
    # Strip code fences if the model added them despite instructions.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop opening fence line and (if present) closing fence line.
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.upper().startswith("UNRECOVERABLE"):
        return None
    # Basic sanity: must have substantive content
    if len(cleaned) < 20:
        return None
    # Reject responses that look like the LLM ignored the instruction
    # and described the input ("This passage appears to be...")
    low = cleaned.lower()
    if low.startswith(("this passage", "the passage", "this text",
                        "the text", "here is", "i cannot")):
        return None
    return cleaned


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

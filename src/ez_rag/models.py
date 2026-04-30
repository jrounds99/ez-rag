"""Model management — list/pull/delete on the local Ollama, plus curated lists.

Used by the GUI's pull-a-model dialog and the LLM/embedder dropdowns.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import httpx


# Curated recommendations. (tag, approx GB on disk, blurb)
RECOMMENDED_LLM: list[tuple[str, float, str]] = [
    ("qwen2.5:3b",            1.9, "Qwen 2.5 3B — fast, low VRAM (default)"),
    ("qwen2.5:7b-instruct",   4.4, "Qwen 2.5 7B — recommended balance"),
    ("qwen2.5:14b-instruct",  8.7, "Qwen 2.5 14B — better quality, 16GB+ VRAM"),
    ("qwen3:8b",              5.2, "Qwen 3 8B"),
    ("qwen3:14b",             9.0, "Qwen 3 14B"),
    ("llama3.2:3b",           2.0, "Llama 3.2 3B"),
    ("llama3.1:8b",           4.7, "Llama 3.1 8B"),
    ("llama3.3:70b",         42.5, "Llama 3.3 70B — needs 48GB+ VRAM"),
    ("phi4-mini",             2.5, "Phi-4 mini"),
    ("phi4",                  9.1, "Phi-4 14B"),
    ("gemma3:4b",             3.3, "Gemma 3 4B"),
    ("gemma3:12b",            8.1, "Gemma 3 12B"),
    ("gemma3:27b",           17.4, "Gemma 3 27B"),
    ("mistral:7b",            4.1, "Mistral 7B"),
    ("mistral-nemo:12b",      7.1, "Mistral Nemo 12B"),
    ("deepseek-r1:8b",        4.9, "DeepSeek R1 8B (reasoning)"),
    ("gpt-oss:20b",          12.0, "GPT-OSS 20B"),
    ("llama3.2-vision:11b",   7.8, "Llama 3.2 Vision 11B (multimodal)"),
]

RECOMMENDED_EMBED: list[tuple[str, float, str]] = [
    ("nomic-embed-text",          0.27, "Nomic Embed Text (default)"),
    ("mxbai-embed-large",         0.67, "mxbai-embed-large"),
    ("bge-m3",                    1.2,  "BGE-M3 — multilingual, dense+sparse"),
    ("snowflake-arctic-embed:l",  0.67, "Snowflake Arctic Large"),
    ("snowflake-arctic-embed2",   1.2,  "Snowflake Arctic v2"),
    ("granite-embedding:278m",    0.55, "IBM Granite 278M"),
]

# Models that fastembed supports out of the box (downloaded on first use).
FASTEMBED_MODELS: list[str] = [
    "BAAI/bge-small-en-v1.5",                          # 130MB,  default
    "BAAI/bge-base-en-v1.5",                           # 440MB
    "BAAI/bge-large-en-v1.5",                          # 1.34GB, best quality
    "sentence-transformers/all-MiniLM-L6-v2",          # 90MB,   tiny
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "jinaai/jina-embeddings-v2-base-en",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "snowflake/snowflake-arctic-embed-s",
    "snowflake/snowflake-arctic-embed-m",
]


@dataclass
class OllamaModel:
    tag: str
    size: int                # bytes on disk
    digest: str = ""
    parameter_size: str = "" # "7B", "14B", etc.
    quant: str = ""          # "Q4_K_M", etc.
    family: str = ""

    @property
    def size_gb(self) -> float:
        return self.size / 1e9


def list_ollama_models(url: str, timeout: float = 2.0) -> list[OllamaModel]:
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
    except Exception:
        return []
    out: list[OllamaModel] = []
    for m in r.json().get("models", []) or []:
        details = m.get("details") or {}
        out.append(OllamaModel(
            tag=m.get("name", ""),
            size=int(m.get("size", 0)),
            digest=m.get("digest", ""),
            parameter_size=details.get("parameter_size", "") or "",
            quant=details.get("quantization_level", "") or "",
            family=details.get("family", "") or "",
        ))
    out.sort(key=lambda x: x.tag)
    return out


def pull_ollama_model(url: str, tag: str) -> Iterator[dict]:
    """Yield progress events as `ollama pull <tag>` runs.

    Each event is a dict like:
      {"status": "pulling manifest"}
      {"status": "downloading", "digest": "...", "total": 12345, "completed": 678}
      {"status": "success"}
    """
    with httpx.stream(
        "POST",
        url.rstrip("/") + "/api/pull",
        json={"name": tag, "stream": True},
        timeout=None,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def unload_ollama_model(url: str, tag: str, *, timeout: float = 10.0) -> bool:
    """Ask Ollama to evict `tag` from VRAM. We POST a no-op generate request
    with `keep_alive: 0`, which tells Ollama to unload as soon as the call
    returns. Returns True on a 2xx response, False on any error.

    Useful before a long ingest: the chat model isn't doing anything during
    embedding, and unloading frees VRAM the OS can use for parsing/OCR.
    """
    try:
        r = httpx.post(
            url.rstrip("/") + "/api/generate",
            json={"model": tag, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def delete_ollama_model(url: str, tag: str) -> bool:
    try:
        r = httpx.request(
            "DELETE",
            url.rstrip("/") + "/api/delete",
            json={"name": tag},
            timeout=30.0,
        )
        return r.status_code == 200
    except Exception:
        return False


def is_embed_capable(name: str) -> bool:
    """Best-effort guess: tags whose name looks like an embedder."""
    low = name.lower()
    return any(t in low for t in (
        "embed", "bge-", "minilm", "e5-", "arctic-embed", "granite-embedding",
    ))


def fmt_size(n_bytes: int) -> str:
    n = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            if unit in ("B", "KB"):
                return f"{n:.0f} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ============================================================================
# VRAM estimation
# ============================================================================
#
# Ollama defaults to Q4_K_M which is roughly 4.5 bits per weight on average.
# With a moderate (4–8K) context the runtime overhead (KV cache + activations
# + framework) lands around 0.5 GB on small models and grows ~5% of weights
# for larger ones. This is a heuristic, not exact.

_QUANT_BITS = {
    "Q2_K": 3.0, "Q3_K_S": 3.4, "Q3_K_M": 3.8, "Q3_K_L": 4.0,
    "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K_S": 4.3, "Q4_K_M": 4.5,
    "Q5_0": 5.5, "Q5_1": 6.0, "Q5_K_S": 5.4, "Q5_K_M": 5.5,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "F16": 16.0, "FP16": 16.0,
    "BF16": 16.0,
    "F32": 32.0, "FP32": 32.0,
}


def parse_param_count(size_str: str) -> float | None:
    """'7b' -> 7.0,  '70m' -> 0.07,  '1.5b' -> 1.5.  Returns billions of params."""
    if not size_str:
        return None
    s = size_str.strip().lower().replace(",", "")
    m = re.match(r"^([\d.]+)\s*([bm])?$", s)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "b")
    return n / 1000.0 if unit == "m" else n


def estimate_vram_gb(
    size_str: str,
    quant: str = "Q4_K_M",
    *,
    context_k: int = 4,
) -> float | None:
    """Rough VRAM estimate (GB) for `<size>` at `<quant>`.

    Returns None if the size string can't be parsed (e.g. tag without a size).
    """
    params_b = parse_param_count(size_str)
    if params_b is None:
        return None
    bits = _QUANT_BITS.get(quant.upper().replace(" ", ""), 4.5)
    weights_gb = params_b * bits / 8.0
    # KV cache scales with context; ballpark 0.05 GB per 1B params per 4K ctx.
    kv_gb = params_b * 0.05 * max(1, context_k / 4.0)
    overhead_gb = 0.5
    return weights_gb + kv_gb + overhead_gb


def fmt_vram_gb(gb: float | None) -> str:
    if gb is None:
        return ""
    if gb < 1.0:
        return f"~{gb*1024:.0f} MB"
    return f"~{gb:.1f} GB"


# ============================================================================
# Local GPU detection
# ============================================================================

def detect_total_vram_gb() -> float | None:
    """Return total VRAM available across all NVIDIA GPUs (GB).

    Falls back to None on machines without nvidia-smi (e.g. Apple Silicon /
    integrated GPUs / no driver installed).
    """
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        )
        total_mib = sum(int(x.strip()) for x in out.splitlines() if x.strip())
        return total_mib / 1024.0
    except Exception:
        return None


def vram_fit(needed_gb: float | None, available_gb: float | None) -> str:
    """Classify how a model fits the local GPU. Returns 'fits' | 'tight' | 'over' | 'unknown'."""
    if needed_gb is None or available_gb is None:
        return "unknown"
    if needed_gb <= available_gb * 0.85:
        return "fits"
    if needed_gb <= available_gb * 1.05:
        return "tight"
    return "over"


# ============================================================================
# Ollama library browser — list every model available on ollama.com
# ============================================================================

@dataclass
class LibraryModel:
    name: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)  # tools, vision, embedding…
    sizes: list[str] = field(default_factory=list)         # ["3b", "7b", "70b"]
    pulls: str = ""                                        # "113.8M"
    tag_count: str = ""                                    # "93"
    updated: str = ""                                      # "1 year ago"


_LIB_CACHE: dict[str, tuple[float, list[LibraryModel]]] = {}
_LIB_CACHE_TTL_S = 6 * 3600  # 6 hours


def fetch_ollama_library(
    url: str = "https://ollama.com/library",
    *,
    timeout: float = 12.0,
    use_cache: bool = True,
) -> list[LibraryModel]:
    """Fetch and parse the public Ollama library page.

    Returns one LibraryModel per public model. Cached in-process for 6h.
    Raises httpx errors on network failure.
    """
    if use_cache:
        cached = _LIB_CACHE.get(url)
        if cached and (time.time() - cached[0]) < _LIB_CACHE_TTL_S:
            return cached[1]

    r = httpx.get(url, timeout=timeout, headers={"User-Agent": "ez-rag/0.1"})
    r.raise_for_status()
    html = r.text

    # Slice into per-card chunks. Each card is an <li x-test-model …> wrapping
    # an <a href="/library/{name}"> with descriptive markup inside.
    cards: list[LibraryModel] = []
    # Anchors keyed by model name; capture the segment up to the next /library/ anchor.
    starts = list(re.finditer(
        r'<a [^>]*href="/library/([A-Za-z0-9._\-]+)"', html,
    ))
    for i, m in enumerate(starts):
        name = m.group(1)
        chunk_end = starts[i + 1].start() if i + 1 < len(starts) else (m.end() + 6000)
        chunk = html[m.start():chunk_end]
        cards.append(_parse_library_card(name, chunk))

    if use_cache:
        _LIB_CACHE[url] = (time.time(), cards)
    return cards


_RE_DESC = re.compile(
    r'<p class="max-w-lg [^"]*">([\s\S]*?)</p>', re.IGNORECASE,
)
_RE_CAP = re.compile(
    r'x-test-capability[^>]*>([\s\S]*?)</span>', re.IGNORECASE,
)
_RE_SIZE = re.compile(
    r'x-test-size[^>]*>([\s\S]*?)</span>', re.IGNORECASE,
)
_RE_PULLS = re.compile(
    r'x-test-pull-count[^>]*>([\s\S]*?)</span>', re.IGNORECASE,
)
_RE_TAGS = re.compile(
    r'x-test-tag-count[^>]*>([\s\S]*?)</span>', re.IGNORECASE,
)
_RE_UPDATED = re.compile(
    r'x-test-updated[^>]*>([\s\S]*?)</span>', re.IGNORECASE,
)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_library_card(name: str, chunk: str) -> LibraryModel:
    desc = _strip_tags(_RE_DESC.search(chunk).group(1)) if _RE_DESC.search(chunk) else ""
    caps = [_strip_tags(s) for s in _RE_CAP.findall(chunk)]
    sizes = [_strip_tags(s) for s in _RE_SIZE.findall(chunk)]
    pulls = _strip_tags(_RE_PULLS.search(chunk).group(1)) if _RE_PULLS.search(chunk) else ""
    tag_count = _strip_tags(_RE_TAGS.search(chunk).group(1)) if _RE_TAGS.search(chunk) else ""
    updated = _strip_tags(_RE_UPDATED.search(chunk).group(1)) if _RE_UPDATED.search(chunk) else ""
    return LibraryModel(
        name=name, description=desc,
        capabilities=caps, sizes=sizes,
        pulls=pulls, tag_count=tag_count, updated=updated,
    )


def search_library(
    models: Sequence[LibraryModel], query: str, *, capability: str | None = None,
) -> list[LibraryModel]:
    """Filter library by free-text search and optional capability filter."""
    q = (query or "").lower().strip()
    out = []
    for m in models:
        if capability and capability not in m.capabilities:
            continue
        if q and q not in m.name.lower() and q not in m.description.lower():
            continue
        out.append(m)
    return out

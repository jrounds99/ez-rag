"""Hardware-aware model recommendations.

Two responsibilities:

1. `assess(gpu)` — turn a DetectedGpu into a usability verdict (tier,
    headroom, warnings). Used by the Settings → Hardware card to give
    the user a clear "what can I run" answer.

2. `recommend_models(gpu)` — given a card, list LLMs / embedders /
    rerankers that will actually fit and run well. Drives the
    "Recommended models for this GPU" panel in Settings.

3. `estimate_required_vram(cfg)` — used at chatbot-export time to embed
    a minimum-VRAM hint in the bundle manifest.

Pure / stateless — no I/O, no GPU calls. All numbers come from a small
in-file table of model sizes at common quantizations. The recommender
deliberately leans conservative (over-budget rather than under).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .gpu_detect import DetectedGpu


# ============================================================================
# Per-model VRAM table (Q4_K_M weight + KV cache + activations + slack)
# ============================================================================
# Numbers are approximations gathered from Ollama's `ollama show <model>`
# output and llama.cpp memory-estimation issue threads. They include a
# typical 4096-token KV cache and ~500 MB of slack. Real usage varies
# with context window, batch size, and flash attention.

@dataclass(frozen=True)
class ModelEntry:
    tag: str
    pretty_name: str
    role: str                # "chat" | "embedder" | "reranker" | "ingest-fast"
    quant: str               # "Q4_K_M" | "Q5_K_M" | "Q8_0" | "F16" | "BGE"
    vram_mb: int             # estimated VRAM with default ctx
    notes: str = ""


_MODELS: list[ModelEntry] = [
    # ----- Chat models, smallest → largest -----
    ModelEntry("llama3.2:1b",            "Llama 3.2 1B",
               "chat", "Q4_K_M", 1300,
               "Tiny — useful for retrieval-only fallback or extreme low-VRAM."),
    ModelEntry("qwen2.5:0.5b",           "Qwen 2.5 0.5B",
               "chat", "Q4_K_M", 800,
               "Almost a toy — okay for testing pipelines."),
    ModelEntry("qwen2.5:1.5b",           "Qwen 2.5 1.5B",
               "chat", "Q4_K_M", 1700,
               "Tiny model — surprisingly coherent for retrieval-only chat."),
    ModelEntry("phi4-mini",              "Phi-4 Mini (3.8B)",
               "chat", "Q4_K_M", 2900,
               "Microsoft's reasoning-focused tiny model. Good on 6 GB cards."),
    ModelEntry("llama3.2:3b",            "Llama 3.2 3B",
               "chat", "Q4_K_M", 2700,
               "Solid balance of size and quality on small GPUs."),
    ModelEntry("qwen2.5:3b",             "Qwen 2.5 3B",
               "chat", "Q4_K_M", 2600),
    ModelEntry("gemma2:2b",              "Gemma 2 2B",
               "chat", "Q4_K_M", 2400),
    ModelEntry("qwen2.5:7b-instruct",    "Qwen 2.5 7B Instruct",
               "chat", "Q4_K_M", 5500,
               "Best balance of quality + speed for 8–12 GB cards."),
    ModelEntry("llama3.1:8b",            "Llama 3.1 8B",
               "chat", "Q4_K_M", 6200),
    ModelEntry("mistral:7b",             "Mistral 7B",
               "chat", "Q4_K_M", 5400),
    ModelEntry("qwen2.5:14b",            "Qwen 2.5 14B",
               "chat", "Q4_K_M", 9500,
               "Higher quality; needs 12 GB+ headroom."),
    ModelEntry("qwen2.5-coder:14b",      "Qwen 2.5 Coder 14B",
               "chat", "Q4_K_M", 9500,
               "Code-specialized; same VRAM as base 14B."),
    ModelEntry("qwen2.5:32b",            "Qwen 2.5 32B",
               "chat", "Q4_K_M", 20500,
               "Very capable; needs 24 GB+ for headroom."),
    ModelEntry("qwen2.5:32b",            "Qwen 2.5 32B",
               "chat", "Q5_K_M", 24000,
               "Slightly higher quality at the cost of more VRAM."),
    ModelEntry("llama3.1:70b",           "Llama 3.1 70B",
               "chat", "Q3_K_M", 36000,
               "Q3 fits on 48 GB pro cards."),
    ModelEntry("llama3.1:70b",           "Llama 3.1 70B",
               "chat", "Q4_K_M", 45000,
               "Needs 48 GB+ workstation card."),
    ModelEntry("qwen2.5:72b",            "Qwen 2.5 72B",
               "chat", "Q4_K_M", 46000,
               "Needs 48 GB+; great for professional cards."),

    # ----- Embedders -----
    ModelEntry("nomic-embed-text",       "Nomic Embed Text",
               "embedder", "Q4_K_M", 320,
               "Default Ollama embedder. Fast, GPU-accelerated."),
    ModelEntry("BAAI/bge-small-en-v1.5", "BGE Small EN v1.5",
               "embedder", "F16", 150,
               "CPU-friendly fastembed default. No GPU required."),
    ModelEntry("BAAI/bge-large-en-v1.5", "BGE Large EN v1.5",
               "embedder", "F16", 1300,
               "Higher recall; consider for technical corpora."),

    # ----- Reranker -----
    ModelEntry("Xenova/ms-marco-MiniLM-L-6-v2", "MiniLM Cross-Encoder",
               "reranker", "ONNX", 25,
               "Small but high-impact reranker. Always recommended."),
]


# Baseline overhead reserved before model VRAM is counted: embedder +
# reranker + OS / driver slack. Keeps the recommender from suggesting a
# model that exactly fills the card.
_BASELINE_OVERHEAD_MB = 1500


# ============================================================================
# Assessment
# ============================================================================

@dataclass
class Assessment:
    runnable: bool
    tier: str                    # "min" | "comfortable" | "ample" | "professional" | "extreme"
    headroom_mb: int             # vram_total - baseline overhead
    runtime_blurb: str
    warnings: list[str] = field(default_factory=list)


def _runtime_blurb(gpu: DetectedGpu) -> str:
    if gpu.vendor == "nvidia":
        if gpu.driver_version:
            return f"CUDA · driver {gpu.driver_version} · full performance expected"
        return "CUDA runtime"
    if gpu.vendor == "amd":
        spec = gpu.matched_spec
        if spec and spec.architecture in ("rdna3", "rdna4",
                                            "cdna2", "cdna3", "cdna4"):
            return "ROCm / HIP · supported on Linux + Windows"
        if spec and spec.architecture in ("rdna2",):
            return "ROCm Linux primary; HIP Windows preview (older arch)"
        return "ROCm runtime"
    if gpu.vendor == "intel":
        return "Intel oneAPI / Level Zero · experimental"
    return "Unknown runtime"


def _tier_from_headroom(headroom_mb: int) -> str:
    """Map free-after-overhead VRAM to a coarse tier label.

    Thresholds are slightly tighter than catalog tiers because we already
    subtracted the embedder + reranker overhead.
    """
    if headroom_mb >= 60 * 1024:
        return "extreme"
    if headroom_mb >= 24 * 1024:
        return "professional"
    if headroom_mb >= 11 * 1024:
        return "ample"
    if headroom_mb >= 5 * 1024:
        return "comfortable"
    if headroom_mb >= 2 * 1024:
        return "min"
    return "insufficient"


def assess(gpu: DetectedGpu) -> Assessment:
    """Produce a usability verdict for a single detected GPU."""
    headroom = max(0, gpu.vram_total_mb - _BASELINE_OVERHEAD_MB)
    tier = _tier_from_headroom(headroom)
    warnings: list[str] = list(gpu.health_notes)
    runnable = gpu.is_compatible and tier != "insufficient"

    if tier == "insufficient":
        warnings.append(
            "After embedder + reranker overhead, less than 2 GB remains "
            "for the chat model. Consider CPU-only retrieval mode."
        )
    if not gpu.is_compatible:
        warnings.append(
            "ez-rag does not consider this GPU compatible — it may still "
            "work, but no model recommendations will be shown."
        )
    spec = gpu.matched_spec
    if spec and spec.legacy:
        warnings.append(
            f"{spec.name} is a legacy architecture. Stick to Q4_K_M; "
            "newer quantizations may not load."
        )

    return Assessment(
        runnable=runnable,
        tier=tier,
        headroom_mb=headroom,
        runtime_blurb=_runtime_blurb(gpu),
        warnings=warnings,
    )


# ============================================================================
# Model recommendations
# ============================================================================

@dataclass
class ModelSuggestion:
    tag: str
    pretty_name: str
    role: str
    quant: str
    estimated_vram_mb: int
    fits: bool
    expected_tps: tuple[int, int]   # (low, high) tokens/sec range
    rationale: str
    notes: str = ""


def _expected_tps(model: ModelEntry, gpu: DetectedGpu) -> tuple[int, int]:
    """Tokens/sec band, derived from spec.bandwidth_gbps and model size.

    Rough rule: for memory-bound inference, throughput ≈ bandwidth / model_size.
    We give a 30% range around that to keep expectations honest.
    """
    spec = gpu.matched_spec
    bw = spec.bandwidth_gbps if spec else 400
    # Convert the model VRAM (mostly weights) to GB for the rough divide.
    weight_gb = max(0.5, model.vram_mb / 1024)
    midpoint = max(5, int(bw / weight_gb * 0.6))   # 0.6 = empirical scaler
    return (max(2, int(midpoint * 0.7)), int(midpoint * 1.3))


def _rationale(model: ModelEntry, headroom_mb: int) -> str:
    if model.role == "embedder":
        return "Always recommended — embeddings drive retrieval."
    if model.role == "reranker":
        return "Always recommended — biggest single retrieval-quality win."
    # Chat-model rationales by VRAM tier
    if headroom_mb < 5 * 1024:
        return "Tiny model that fits a low-VRAM card."
    if headroom_mb < 11 * 1024:
        if model.vram_mb < 4000:
            return "Solid choice for low-VRAM cards."
        return "Best balance of quality and speed for this VRAM tier."
    if headroom_mb < 24 * 1024:
        if model.vram_mb < 7000:
            return "Fast option — leaves room for big context windows."
        if model.vram_mb < 11000:
            return "Recommended — best balance of quality + speed."
        return "Higher quality at the cost of speed."
    # Professional / extreme
    if model.vram_mb < 11000:
        return "Light load — leaves headroom for contextual retrieval and big contexts."
    if model.vram_mb < 24000:
        return "Sweet spot for this card — high quality with comfortable headroom."
    return "Top-tier model that this card can host."


def recommend_models(gpu: DetectedGpu) -> list[ModelSuggestion]:
    """Return ordered list of models that fit this GPU.

    Order: embedders → reranker → chat models (smallest → largest fitting
    model). Caller can group by `role` to render in sections.
    """
    a = assess(gpu)
    out: list[ModelSuggestion] = []
    headroom = a.headroom_mb

    # Embedders & reranker — always include the small ones; only the
    # large embedder gets a fit check.
    for model in _MODELS:
        if model.role not in ("embedder", "reranker"):
            continue
        fits = model.vram_mb <= max(headroom, 1024)
        if model.role == "reranker" or model.vram_mb <= 500 or fits:
            out.append(ModelSuggestion(
                tag=model.tag,
                pretty_name=model.pretty_name,
                role=model.role,
                quant=model.quant,
                estimated_vram_mb=model.vram_mb,
                fits=True,
                expected_tps=(0, 0),   # n/a for embedder/reranker
                rationale=_rationale(model, headroom),
                notes=model.notes,
            ))

    # Chat models — every fitting one, plus the SMALLEST non-fitting one
    # marked fits=False so the user sees what would unlock at next tier.
    chat_candidates = [m for m in _MODELS if m.role == "chat"]
    chat_candidates.sort(key=lambda m: m.vram_mb)
    fitting = [m for m in chat_candidates if m.vram_mb <= headroom]
    not_fitting = [m for m in chat_candidates if m.vram_mb > headroom]

    for model in fitting:
        lo, hi = _expected_tps(model, gpu)
        out.append(ModelSuggestion(
            tag=model.tag,
            pretty_name=f"{model.pretty_name} ({model.quant})",
            role="chat",
            quant=model.quant,
            estimated_vram_mb=model.vram_mb,
            fits=True,
            expected_tps=(lo, hi),
            rationale=_rationale(model, headroom),
            notes=model.notes,
        ))

    if not_fitting:
        peek = not_fitting[0]
        lo, hi = _expected_tps(peek, gpu)
        out.append(ModelSuggestion(
            tag=peek.tag,
            pretty_name=f"{peek.pretty_name} ({peek.quant})",
            role="chat",
            quant=peek.quant,
            estimated_vram_mb=peek.vram_mb,
            fits=False,
            expected_tps=(lo, hi),
            rationale=(
                f"Out of reach — needs ~{peek.vram_mb // 1024} GB. "
                "Shown so you know what unlocks at the next tier."
            ),
            notes=peek.notes,
        ))

    return out


# ============================================================================
# Export-time VRAM estimation
# ============================================================================

@dataclass
class VramRequirement:
    min_vram_gb: int
    recommended_vram_gb: int
    llm_model: str
    llm_quant: str
    embedder: str
    estimated_baseline_mb: int


def estimate_required_vram(cfg) -> VramRequirement:
    """Calculate VRAM requirements to embed in a chatbot export manifest.

    Reads the active llm_model + ollama_embed_model from cfg, looks them
    up in our model table, and computes:
        min_vram_gb         = ceil((model_mb + baseline_mb) / 1024)
        recommended_vram_gb = ceil(min * 1.25)   # 25% headroom

    Falls back to conservative defaults when the configured model isn't
    in our table (better to over-warn than to silently ship a bundle that
    won't fit).
    """
    llm_tag = getattr(cfg, "llm_model", "qwen2.5:7b-instruct")
    embedder_tag = (
        getattr(cfg, "ollama_embed_model", None)
        or getattr(cfg, "embedder_model", None)
        or "nomic-embed-text"
    )

    # Match exact tag first; fall back to prefix-only match (so
    # ":latest" / unknown variants still resolve). Without the exact
    # pass, the first-prefix match could return the smallest variant
    # of a family ("qwen2.5:0.5b" instead of the requested
    # "qwen2.5:7b-instruct").
    def _lookup(tag: str, role: str) -> Optional[ModelEntry]:
        bare = tag.split(":")[0]
        # Pass 1: exact tag match
        for m in _MODELS:
            if m.role == role and m.tag == tag:
                return m
        # Pass 2: prefix match (legacy fallback)
        for m in _MODELS:
            if m.role == role and m.tag.split(":")[0] == bare:
                return m
        return None

    llm = _lookup(llm_tag, "chat")
    emb = _lookup(embedder_tag, "embedder")

    # Defaults if unknown — mid-pack 7B and small embedder is the most
    # common ez-rag profile, so use that as the over-warn fallback.
    llm_mb = llm.vram_mb if llm else 6000
    llm_quant = llm.quant if llm else "Q4_K_M"
    emb_mb = emb.vram_mb if emb else 320

    # Reranker + OS slack regardless.
    baseline = emb_mb + 25 + 500   # 25 MB MiniLM + 500 MB slack

    total_mb = llm_mb + baseline
    min_gb = max(2, _ceil_div(total_mb, 1024))
    rec_gb = max(min_gb + 1, _ceil_div(int(total_mb * 1.25), 1024))

    return VramRequirement(
        min_vram_gb=min_gb,
        recommended_vram_gb=rec_gb,
        llm_model=llm_tag,
        llm_quant=llm_quant,
        embedder=embedder_tag,
        estimated_baseline_mb=baseline,
    )


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)

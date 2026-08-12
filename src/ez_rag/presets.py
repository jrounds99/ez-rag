"""Configuration presets — benchmark-backed bundles of settings.

ez-rag has grown a lot of knobs (models, embedders, retrieval stages,
ingest quality features, compression, PDF backends). Presets collapse
the common intents into one choice, with every number traceable to a
benchmark this repo actually ran:

  - The Ohio corpus bench: 23 chat models × 3 embedders × 20 questions
    (1,380 judged cells) — bench/reports/ohio-20260503-211733/
  - The compression bench: 240 real retrieval prompts through
    headroom-ai — headroom_bench/run_summary.json
  - The retrieval-option matrix — docs/OPTIMIZATIONS.md

A preset only SETS config fields; nothing is installed or pulled. If a
preset's model isn't in Ollama yet, the CLI/GUI shows the pull command.
Fields the preset doesn't mention keep their current values, so users
can still hand-tune afterward.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    tagline: str                    # one line, shown in lists
    settings: dict = field(default_factory=dict)
    requires_vram_gb: int = 0       # rough total for chat model + embedder
    models_needed: tuple = ()       # ollama tags the preset expects
    details: str = ""               # the "more info" text — full rationale


PRESETS: list[Preset] = [
    Preset(
        id="balanced",
        name="Balanced (benchmark winner)",
        tagline="Best measured quality per GB — the recommended default.",
        settings={
            "llm_model": "granite3.3:2b",
            "ollama_embed_model": "bge-m3:567m",
            "rerank": True,
            "chunk_headers": True,
            "dedup_chunks": True,
            "compress_context": False,
            "top_k": 8,
        },
        requires_vram_gb=4,
        models_needed=("granite3.3:2b", "bge-m3:567m"),
        details="""\
WHY THIS PRESET
Across 1,380 judged answers (23 chat models × 3 embedders × 20
questions on a public-domain geology corpus), granite3.3:2b scored
10.95/12 — the highest of ANY model tested, beating models up to 13×
its size including deepseek-r1:32b (10.00) and qwen2.5:14b (9.75). At
2.5B parameters it also generated at ~299 tokens/sec, so it's the
best answer quality AND nearly the best speed simultaneously.

The bge-m3 embedder scored within 0.1/12 of the leader (9.56 vs 9.55
— a statistical tie) while being 14× smaller than qwen3-embedding:8b,
freeing ~7.5 GB of VRAM for nothing.

WHAT'S ON
- Cross-encoder reranking: the single biggest retrieval accuracy lift
  measured in the option matrix (~1s/query).
- Contextual chunk headers + within-file dedup (ingest defaults).
- Compression OFF: at this context size the token savings don't repay
  the ~4s/query compression latency on a local setup.

TRADE-OFF
granite3.3:2b's rule-based factual-accuracy score was 67% (middle of
the pack) — it writes confident, well-structured answers but drops a
specific fact more often than the larger models. If exact names/dates
matter more than fluency, see the 'factual' preset.""",
    ),
    Preset(
        id="factual",
        name="Factual precision",
        tagline="Highest measured factual accuracy — for names, dates, numbers.",
        settings={
            "llm_model": "deepseek-r1:32b",
            "ollama_embed_model": "bge-m3:567m",
            "rerank": True,
            "chunk_headers": True,
            "dedup_chunks": True,
            "compress_context": False,
            "top_k": 10,
        },
        requires_vram_gb=24,
        models_needed=("deepseek-r1:32b", "bge-m3:567m"),
        details="""\
WHY THIS PRESET
On the rule-based gold-truth check (must-contain factual phrases
across the 8 most factual benchmark questions), deepseek-r1:32b
scored 78% — the best of all 23 models — while granite3.3:2b (the
'balanced' pick) scored 67%. That 11-point gap is the difference
between "answers that read well" and "answers that contain the exact
founding date and the right person's name."

Its rubric score (10.00/12) ranks #9, not #1 — the LLM judge dings
its verbose, heavily-structured style. That's a style penalty, not a
correctness one. When you can't afford a hallucinated fact, this is
the measured best available.

MIDDLE GROUND
granite3.3:8b hit 73% gold at a quarter the VRAM (~8 GB) with a
10.33/12 rubric — edit the model field after applying if 24 GB is
too rich.

TRADE-OFFS
- ~24 GB VRAM for the 32B model at Q4 quantization.
- ~56 tokens/sec vs granite-2b's ~299 — answers take noticeably
  longer.
- top_k raised to 10: the bigger model handles more context well and
  factual questions benefit from wider evidence.""",
    ),
    Preset(
        id="light",
        name="Light (laptop / low VRAM)",
        tagline="Runs in ~4 GB — best small-model quality measured.",
        settings={
            "llm_model": "llama3.2:3b",
            "ollama_embed_model": "nomic-embed-text",
            "rerank": True,
            "chunk_headers": True,
            "dedup_chunks": True,
            "compress_context": True,
            "compress_context_min_tokens": 500,
            "top_k": 6,
        },
        requires_vram_gb=4,
        models_needed=("llama3.2:3b", "nomic-embed-text"),
        details="""\
WHY THIS PRESET
llama3.2:3b scored 10.03/12 in the bench — the best result under
4 GB and ahead of several 7–14B models — at ~326 tokens/sec.
nomic-embed-text (0.3 GB) landed within 0.1/12 of the best embedder;
at this budget the smallest of the statistically tied leaders wins.

WHY COMPRESSION IS ON HERE
The 240-prompt compression bench cut retrieval context ~49% with an
answer-quality delta of −0.40/12 (at the judge's noise floor). On a
VRAM-tight machine that halves the KV-cache footprint and prompt-eval
time exactly where it hurts. min_tokens=500 keeps small questions
away from the compressor entirely, so short chats stay instant. Note:
compression needs `pip install "ez-rag[compress]"` — without it, this
setting silently no-ops (fail-open) and everything still works.

TRADE-OFFS
- top_k lowered to 6: small models get distracted by wide context.
- Expect competent answers with occasional missed details — 10.03/12
  measured vs 10.95 for 'balanced'. The gap is real but modest.""",
    ),
    Preset(
        id="deep-context",
        name="Deep context (big documents)",
        tagline="Wide retrieval + compression — for long reports and manuals.",
        settings={
            "llm_model": "granite3.3:2b",
            "ollama_embed_model": "bge-m3:567m",
            "rerank": True,
            "top_k": 16,
            "context_window": 1,
            "chunk_headers": True,
            "dedup_chunks": True,
            "compress_context": True,
            "compress_context_min_tokens": 1000,
        },
        requires_vram_gb=6,
        models_needed=("granite3.3:2b", "bge-m3:567m"),
        details="""\
WHY THIS PRESET
For "walk me through this 300-page manual" corpora, retrieval depth
matters more than model size. This keeps the benchmark-winning
granite3.3:2b but doubles retrieval width (top_k 16) and expands each
hit with its neighboring chunk (context_window=1) so answers can
stitch across sections.

WHY COMPRESSION IS ON HERE
Wide retrieval is exactly where the compression bench showed its
best results — savings grow with context size (deeper top-k = more
redundancy to prune), averaging ~49% token reduction at a −0.40/12
quality delta. The min_tokens=1000 gate means only genuinely large
contexts pay the ~4s compression latency; at 16 chunks you're almost
always past the gate, and the compressed prompt often finishes
GENERATING sooner than the uncompressed one would have. Requires
`pip install "ez-rag[compress]"`; silently no-ops without it.

TRADE-OFFS
- More retrieval + compression latency per question (~5-8s before
  generation starts) in exchange for materially wider evidence.
- If your documents are short (notes, emails), this preset is
  overkill — 'balanced' will feel snappier with identical quality.""",
    ),
]


def get_preset(preset_id: str) -> Preset | None:
    pid = (preset_id or "").strip().lower()
    for p in PRESETS:
        if p.id == pid:
            return p
    return None


def apply_preset(cfg, preset_id: str) -> list[tuple[str, object, object]]:
    """Apply a preset's settings to a Config in place.

    Returns [(field, old_value, new_value)] for every field actually
    changed, so callers can show a diff. Unknown fields are skipped
    (forward/backward compatibility) rather than erroring.
    """
    p = get_preset(preset_id)
    if p is None:
        raise ValueError(
            f"Unknown preset '{preset_id}'. "
            f"Available: {', '.join(x.id for x in PRESETS)}"
        )
    changed: list[tuple[str, object, object]] = []
    for k, v in p.settings.items():
        if not hasattr(cfg, k):
            continue
        old = getattr(cfg, k)
        if old != v:
            setattr(cfg, k, v)
            changed.append((k, old, v))
    return changed

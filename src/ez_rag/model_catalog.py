"""Curated model catalog — the trusted, locally-bundled model list.

WHERE MODEL LISTS COME FROM (trust chain):

  1. **Live**: ollama.com/library (scraped by `models.fetch_ollama_library`)
     — the authoritative source for what's pullable, but it's a scrape of
     HTML and it needs the network.
  2. **This file**: a curated catalog MAINTAINED IN THIS REPO, refreshed
     whenever we run benchmarks. It is the offline/backup source for the
     model browser, and the only source that carries *measured* numbers:
     every model marked `benchmarked=True` has scores from the Ohio bench
     (23 chat models × 3 embedders × 20 questions, judged 0–12 +
     rule-based gold-truth — bench/reports/ohio-20260503-211733/).

There is no third-party "model list repo" we treat as authoritative —
lists like that go stale and unmaintained fast. ollama.com is the
distribution source of truth; this file is the quality source of truth,
versioned in git where you can audit every change.

VRAM estimates are for the default Ollama quant (Q4_K_M unless noted)
with a ~4k context; add headroom for bigger contexts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CATALOG_UPDATED = "2026-08-13"


@dataclass(frozen=True)
class CatalogModel:
    tag: str                     # exact ollama tag to pull
    name: str                    # human name
    kind: str                    # "chat" | "embedder"
    params_b: float
    est_vram_gb: float
    license: str
    benchmarked: bool = False
    bench_score: float | None = None      # judge avg /12 (Ohio bench)
    bench_gold_pct: int | None = None     # rule-based factual accuracy %
    bench_tok_s: int | None = None        # measured on RTX 5090
    note: str = ""


CATALOG: list[CatalogModel] = [
    # ----- Chat models (every one below `benchmarked=True` has measured
    # numbers from the Ohio sweep; sorted by bench score) -----
    CatalogModel("granite3.3:2b", "Granite 3.3 2B", "chat", 2.5, 2.0,
                 "Apache-2.0", True, 10.95, 67, 299,
                 "Bench winner — best quality per GB of anything tested."),
    CatalogModel("mistral-nemo:12b", "Mistral NeMo 12B", "chat", 12.0, 7.5,
                 "Apache-2.0", True, 10.62, 64, 140,
                 "#2 overall; strongest 12B-class."),
    CatalogModel("granite3.3:8b", "Granite 3.3 8B", "chat", 8.2, 5.5,
                 "Apache-2.0", True, 10.33, 73, 159,
                 "Best factual accuracy under 10 GB."),
    CatalogModel("qwen2.5:7b", "Qwen 2.5 7B", "chat", 7.6, 5.5,
                 "Apache-2.0", True, 10.30, 62, 209,
                 "Also ez-rag's default judge model."),
    CatalogModel("gemma3:4b", "Gemma 3 4B", "chat", 4.3, 3.3,
                 "Gemma license (free, usage policy)", True, 10.20, 64, 129),
    CatalogModel("llama3.1:8b", "Llama 3.1 8B", "chat", 8.0, 5.6,
                 "Llama community (free <700M MAU)", True, 10.12, 58, 192),
    CatalogModel("mistral:7b", "Mistral 7B", "chat", 7.0, 5.0,
                 "Apache-2.0", True, 10.10, 71, 200),
    CatalogModel("llama3.2:3b", "Llama 3.2 3B", "chat", 3.0, 2.5,
                 "Llama community (free <700M MAU)", True, 10.03, 67, 326,
                 "Best under 4 GB — the 'light' preset pick."),
    CatalogModel("deepseek-r1:32b", "DeepSeek-R1 32B", "chat", 32.8, 21.0,
                 "MIT", True, 10.00, 78, 56,
                 "Best factual accuracy measured (78%) — the 'factual' "
                 "preset pick. Reasoning model; slow but precise."),
    CatalogModel("qwen3:1.7b", "Qwen 3 1.7B", "chat", 1.7, 1.5,
                 "Apache-2.0", True, 9.90, 60, 174),
    CatalogModel("qwen3:8b", "Qwen 3 8B", "chat", 8.2, 5.6,
                 "Apache-2.0", True, 9.88, 71, 128),
    CatalogModel("qwen2.5:14b", "Qwen 2.5 14B", "chat", 14.0, 9.5,
                 "Apache-2.0", True, 9.75, 71, 105),
    CatalogModel("qwen3:14b", "Qwen 3 14B", "chat", 14.8, 9.8,
                 "Apache-2.0", True, 9.73, 64, 93),
    CatalogModel("qwen3:0.6b", "Qwen 3 0.6B", "chat", 0.6, 0.8,
                 "Apache-2.0", True, 9.52, 64, 180,
                 "Smallest model within 15% of the best score."),
    CatalogModel("qwen2.5:1.5b", "Qwen 2.5 1.5B", "chat", 1.5, 1.4,
                 "Apache-2.0", True, 9.47, 60, 379),
    CatalogModel("qwen2.5:3b", "Qwen 2.5 3B", "chat", 3.1, 2.5,
                 "Qwen research license", True, 9.37, 58, 274),
    CatalogModel("llama3.2:1b", "Llama 3.2 1B", "chat", 1.2, 1.1,
                 "Llama community (free <700M MAU)", True, 9.22, 60, 578,
                 "Fastest measured (578 tok/s) — the 'speed' preset pick."),
    CatalogModel("phi4-mini", "Phi-4 Mini", "chat", 3.8, 2.9,
                 "MIT", True, 9.20, 67, 282),
    CatalogModel("deepseek-r1:1.5b", "DeepSeek-R1 1.5B", "chat", 1.8, 1.6,
                 "MIT", True, 8.27, 44, 380,
                 "Weak at this size — prefer qwen3:1.7b or llama3.2:3b."),
    CatalogModel("gemma2:2b", "Gemma 2 2B", "chat", 2.6, 2.1,
                 "Gemma license", True, 7.88, 44, 315,
                 "Superseded by gemma3 — kept for reference."),
    CatalogModel("gemma2:9b", "Gemma 2 9B", "chat", 9.0, 6.5,
                 "Gemma license", True, 7.88, 60, 142,
                 "Underperformed its size class in our bench."),
    CatalogModel("qwen2.5:0.5b", "Qwen 2.5 0.5B", "chat", 0.5, 0.7,
                 "Apache-2.0", True, 7.35, 47, 604,
                 "Below the useful floor — bench shows sub-1B cliff."),
    # ----- Larger, NOT benchmarked (flagged as such in the UI) -----
    CatalogModel("qwen2.5:32b", "Qwen 2.5 32B", "chat", 32.8, 20.5,
                 "Apache-2.0", False,
                 note="Not in our bench; strong general model for 24 GB+."),
    CatalogModel("llama3.1:70b", "Llama 3.1 70B", "chat", 70.6, 43.0,
                 "Llama community (free <700M MAU)", False,
                 note="Needs 2×24 GB or a 48 GB card. Not benchmarked here."),
    CatalogModel("qwen2.5:72b", "Qwen 2.5 72B", "chat", 72.7, 46.0,
                 "Qwen license", False,
                 note="Not benchmarked here."),

    # ----- Embedders (bench: all three within 0.1/12 — pick by size) -----
    CatalogModel("bge-m3:567m", "BGE-M3", "embedder", 0.57, 1.2,
                 "MIT", True, 9.56, None, None,
                 "Recommended: tied for best retrieval, 14× smaller than "
                 "qwen3-embedding:8b. Multilingual."),
    CatalogModel("qwen3-embedding:8b", "Qwen 3 Embedding 8B", "embedder",
                 8.0, 8.5, "Apache-2.0", True, 9.55, None, None,
                 "Statistically tied with bge-m3 at 14× the size."),
    CatalogModel("nomic-embed-text", "Nomic Embed Text", "embedder",
                 0.14, 0.4, "Apache-2.0", True, 9.47, None, None,
                 "Smallest footprint; within 0.1/12 of the best."),
]


def catalog_chat_models() -> list[CatalogModel]:
    return [m for m in CATALOG if m.kind == "chat"]


def catalog_embedders() -> list[CatalogModel]:
    return [m for m in CATALOG if m.kind == "embedder"]


def catalog_lookup(tag: str) -> CatalogModel | None:
    t = (tag or "").strip().lower()
    for m in CATALOG:
        if m.tag.lower() == t or m.tag.split(":")[0].lower() == t.split(":")[0]:
            if m.tag.lower() == t:
                return m
    for m in CATALOG:
        if m.tag.split(":")[0].lower() == t.split(":")[0]:
            return m
    return None


def as_library_models():
    """Adapt the curated catalog to the model-browser's LibraryModel shape
    — used as the offline/backup source when the ollama.com scrape is
    unavailable."""
    from .models import LibraryModel
    out = []
    for m in CATALOG:
        bench_bits = []
        if m.benchmarked and m.bench_score is not None:
            bench_bits.append(f"benchmarked: {m.bench_score:.2f}/12")
            if m.bench_gold_pct is not None:
                bench_bits.append(f"factual {m.bench_gold_pct}%")
            if m.bench_tok_s:
                bench_bits.append(f"{m.bench_tok_s} tok/s")
        desc = " · ".join(
            x for x in [m.note or m.name, " ".join(bench_bits),
                        f"license: {m.license}",
                        f"~{m.est_vram_gb:g} GB VRAM"] if x
        )
        size = m.tag.split(":")[1] if ":" in m.tag else f"{m.params_b:g}b"
        out.append(LibraryModel(
            name=m.tag.split(":")[0],
            description=desc,
            capabilities=(["embedding"] if m.kind == "embedder" else []),
            sizes=[size],
            pulls="", tag_count="", updated=f"curated {CATALOG_UPDATED}",
        ))
    return out

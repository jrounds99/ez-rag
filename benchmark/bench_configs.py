"""Retrieval-quality matrix: every retrieval option ON/OFF, scored on a
harder Q-set. Runs RETRIEVAL ONLY (no LLM generation) so the matrix is fast.

Metrics per config:
    top1_doc      : did the gold doc come back as #1?
    gold_at_5     : did the gold doc appear in the top-5?
    sub_hit       : do the required substrings appear anywhere in the
                    top-K retrieved text?
    avg_latency_ms

Usage:
    python bench_configs.py
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder, clear_embedder_cache
from ez_rag.index import Index
from ez_rag.retrieve import smart_retrieve
from ez_rag.workspace import Workspace


# ---- Harder Q-set -----------------------------------------------------------
# Mix: easy / paraphrased / vague / multi-hop / lexical-mismatch / inferential.

QUESTIONS = [
    # Easy baseline
    ("Q01", "Who was the last person to walk on the moon?", "apollo.txt", ["cernan"]),
    ("Q02", "What is the price of SKU A101?",                  "store.xlsx",     ["24.95"]),
    # Paraphrased (corpus uses different exact words)
    ("Q03", "Which crewmember of Apollo 11 didn't set foot on the lunar surface?",
                                                                "apollo.txt",     ["collins"]),
    ("Q04", "Earth's biggest body of saltwater — how large is it?", "ocean.html", ["165"]),
    ("Q05", "Which dog breed in my notes is praised for shepherding ability?",
                                                                "dogs.md",        ["border collie"]),
    # Lexical mismatch
    ("Q06", "What green pigment in plant cells absorbs sunlight?",
                                                                "biology.docx",   ["chlorophyll"]),
    ("Q07", "What did the secret OCR test image actually say?",
                                                                "screenshot.png", ["phoenix falcon"]),
    # Vague / abstract
    ("Q08", "Tell me something I have on file about big oceans.", "ocean.html",   ["pacific"]),
    ("Q09", "Anything in my notes about herding animals?",       "dogs.md",       ["border collie"]),
    # Multi-hop / requires combining
    ("Q10", "What was the date of the final crewed lunar landing?",
                                                                "apollo.txt",     ["1972"]),
    ("Q11", "Which spreadsheet sheet contains pricing?",
                                                                "store.xlsx",     ["pric"]),
    # Tricky / out-of-scope (gold should still be best of weak choices)
    ("Q12", "What architecture does the Attention paper introduce?",
                                                                "attention.pdf",  ["transformer"]),
]


# ---- Configurations to test ------------------------------------------------

def with_overrides(base: Config, **kw) -> Config:
    c = deepcopy(base)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def build_configs(base: Config) -> dict[str, Config]:
    return {
        # Baseline
        "01 hybrid only":
            with_overrides(base, hybrid=True, rerank=False, use_hyde=False,
                           multi_query=False, use_mmr=False, context_window=0),
        # Rerank ladder
        "02 + rerank (MiniLM)":
            with_overrides(base, hybrid=True, rerank=True),
        "03 + rerank (BGE-base)":
            with_overrides(base, hybrid=True, rerank=True,
                           rerank_model="BAAI/bge-reranker-base"),
        # Diversity
        "04 rerank + MMR(0.5)":
            with_overrides(base, rerank=True, use_mmr=True, mmr_lambda=0.5),
        "05 rerank + MMR(0.7)":
            with_overrides(base, rerank=True, use_mmr=True, mmr_lambda=0.7),
        # Neighbor expansion
        "06 rerank + window(1)":
            with_overrides(base, rerank=True, context_window=1),
        "07 rerank + window(2)":
            with_overrides(base, rerank=True, context_window=2),
        # Query expansion (LLM call)
        "08 rerank + HyDE":
            with_overrides(base, rerank=True, use_hyde=True),
        "09 rerank + multi-query":
            with_overrides(base, rerank=True, multi_query=True),
        # Combined "kitchen sink"
        "10 rerank + window(1) + MMR + HyDE":
            with_overrides(base, rerank=True, context_window=1,
                           use_mmr=True, mmr_lambda=0.5, use_hyde=True),
    }


# ---- Runner ----------------------------------------------------------------

@dataclass
class Result:
    cfg_label: str
    qid: str
    elapsed_ms: float
    top1_doc: bool
    gold_at_5: bool
    sub_hit: bool
    top_path: str = ""
    n_chars_retrieved: int = 0


def evaluate(cfg: Config, ws: Workspace, *, label: str) -> list[Result]:
    """Run all questions through one configuration and return Result rows."""
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    out: list[Result] = []
    for qid, question, gold, must in QUESTIONS:
        t0 = time.perf_counter()
        try:
            hits = smart_retrieve(query=question, embedder=embedder,
                                  index=idx, cfg=cfg)
        except Exception as e:
            print(f"  [{label}] {qid} ERROR: {e}")
            hits = []
        ms = (time.perf_counter() - t0) * 1000
        top1 = bool(hits) and gold.lower() in hits[0].path.lower()
        gold5 = any(gold.lower() in h.path.lower() for h in hits[:5])
        joined = " ".join(h.text.lower() for h in hits[:5])
        sub = all(m.lower() in joined for m in must)
        n_chars = sum(len(h.text) for h in hits[:cfg.top_k])
        out.append(Result(
            cfg_label=label, qid=qid, elapsed_ms=ms,
            top1_doc=top1, gold_at_5=gold5, sub_hit=sub,
            top_path=hits[0].path if hits else "",
            n_chars_retrieved=n_chars,
        ))
    return out


def main() -> int:
    ws = Workspace(Path(r"C:/Users/jroun/workstuff/ez-rag-test"))
    base = ws.load_config()
    base.use_rag = True
    # We don't need to keep retest results between configs.
    clear_embedder_cache()

    configs = build_configs(base)

    print(f"Q-set: {len(QUESTIONS)} questions")
    print(f"Configs: {len(configs)}\n")
    print(f"{'config':<42} {'top1':<6} {'g@5':<5} {'sub':<5} {'avg_ms':<9}  notes")
    print("-" * 100)

    all_results: dict[str, list[Result]] = {}
    for label, cfg in configs.items():
        try:
            rs = evaluate(cfg, ws, label=label)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            continue
        all_results[label] = rs
        n = len(rs)
        top1 = sum(1 for r in rs if r.top1_doc)
        g5 = sum(1 for r in rs if r.gold_at_5)
        sub = sum(1 for r in rs if r.sub_hit)
        avg = sum(r.elapsed_ms for r in rs) / n
        notes = ""
        if "BGE" in label: notes = "downloads ~280MB on first use"
        if "HyDE" in label or "multi-query" in label:
            notes = "+1 LLM call per query"
        print(f"  {label:<40} {top1}/{n:<4} {g5}/{n:<3} {sub}/{n:<3} {avg:>7.1f}    {notes}")

    # Per-question failure surface
    print("\n--- per-question failures (sub_hit=False) ---")
    for label, rs in all_results.items():
        misses = [r.qid for r in rs if not r.sub_hit]
        if misses:
            print(f"  {label:<42} miss: {','.join(misses)}")

    # Save
    out_dir = THIS / "reports"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "bench_configs.json"
    json_path.write_text(json.dumps(
        {k: [asdict(r) for r in v] for k, v in all_results.items()},
        indent=2,
    ))
    print(f"\nReport: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

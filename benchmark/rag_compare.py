"""RAG-on / RAG-off comparison harness.

For each question, asks the same model twice:
    1. With retrieval (use_rag=True)
    2. Without retrieval (use_rag=False)

Reports per-question findings (latency, answer length, expected-substring hit,
citation count) and surfaces issues:
    - empty answers
    - missing required substrings (RAG-on only — corpus-grounded questions)
    - LLM errors / timeouts
    - extreme latency outliers
    - retrieval misses (gold doc not in top-K)

Usage:
    python rag_compare.py --model deepseek-r1:32b --workspace ../ez-rag-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder, clear_embedder_cache
from ez_rag.generate import chat_answer, detect_backend
from ez_rag.index import Index
from ez_rag.retrieve import hybrid_search
from ez_rag.workspace import Workspace


# ---- test set ---------------------------------------------------------------
# Each question declares:
#   question:    the prompt
#   category:    "corpus" (answer expected to be in the docs) or
#                "general" (model's own knowledge, RAG should be neutral) or
#                "tricky" (corpus has related but not exact info)
#   gold_doc:    expected source filename if category=="corpus"
#   must:        substrings that MUST appear in a correct answer
#   any_of:      one of these groups must appear (alt phrasings)
#   fail_if:     substrings that, if present, indicate a bad answer

QUESTIONS = [
    # ── Corpus-specific ─────────────────────────────────────────────
    {
        "id": "Q1", "category": "corpus", "gold_doc": "apollo.txt",
        "question": "Who was the very last person to walk on the Moon, and during which mission?",
        "must": ["cernan"],
        "any_of": [["apollo 17"], ["apollo-17"]],
    },
    {
        "id": "Q2", "category": "corpus", "gold_doc": "dogs.md",
        "question": "Which dog breed in my notes is described as the most intelligent?",
        "must": ["border collie"],
    },
    {
        "id": "Q3", "category": "corpus", "gold_doc": "store.xlsx",
        "question": "What is the price of SKU A101 in our store inventory?",
        "must": ["24.95"],
    },
    {
        "id": "Q4", "category": "corpus", "gold_doc": "screenshot.png",
        "question": "What is the magic phrase from the OCR test screenshot?",
        "must": ["phoenix falcon"],
    },
    {
        "id": "Q5", "category": "corpus", "gold_doc": "ocean.html",
        "question": "How many million square kilometers does the Pacific Ocean cover, according to my notes?",
        "must": ["165"],
    },
    {
        "id": "Q6", "category": "corpus", "gold_doc": "biology.docx",
        "question": "What pigment in chloroplasts captures light energy?",
        "must": ["chlorophyll"],
    },
    # ── General knowledge (RAG should not hurt) ─────────────────────
    {
        "id": "Q7", "category": "general",
        "question": "What is the capital of Japan?",
        "must": ["tokyo"],
    },
    {
        "id": "Q8", "category": "general",
        "question": "What does HTTP stand for?",
        "must": ["hypertext"],
        "any_of": [["transfer protocol"]],
    },
    {
        "id": "Q9", "category": "general",
        "question": "Roughly what is the speed of light in a vacuum, in km per second?",
        "any_of": [["299,792"], ["299792"], ["300,000"], ["300000"], ["3 × 10"]],
    },
    {
        "id": "Q10", "category": "general",
        "question": "In one sentence: explain photosynthesis.",
        "must": ["light"],
    },
]


# ---- runner -----------------------------------------------------------------

@dataclass
class Result:
    qid: str
    category: str
    use_rag: bool
    elapsed_s: float
    answer: str
    thinking_chars: int
    citations: list                # [(path, page, score), ...]
    expected_doc_in_topk: bool | None
    must_hit: bool
    any_of_hit: bool
    issues: list[str] = field(default_factory=list)
    error: str = ""


def _check_substring(text: str, must: list[str], any_of: list[list[str]]) -> tuple[bool, bool]:
    low = (text or "").lower()
    must_hit = all(s.lower() in low for s in must) if must else True
    any_of_hit = True
    if any_of:
        any_of_hit = any(all(s.lower() in low for s in group) for group in any_of)
    return must_hit, any_of_hit


def run_question(q: dict, *, cfg: Config, embedder, idx: Index, use_rag: bool) -> Result:
    issues: list[str] = []
    error = ""
    expected_doc_in_topk: bool | None = None

    # Retrieval
    if use_rag:
        try:
            hits = hybrid_search(query=q["question"], embedder=embedder, index=idx,
                                 k=cfg.top_k, use_hybrid=cfg.hybrid)
        except Exception as e:
            error = f"retrieval error: {e}"
            hits = []
            issues.append("retrieval-failed")
        if q["category"] == "corpus":
            gold = q.get("gold_doc")
            if gold:
                top_paths = [h.path.lower() for h in hits[:cfg.top_k]]
                expected_doc_in_topk = any(gold.lower() in p for p in top_paths)
                if not expected_doc_in_topk:
                    issues.append(f"retrieval-miss(gold={gold} not in top-{cfg.top_k})")
    else:
        hits = []

    # Generation
    text = ""
    thinking_chars = 0
    t0 = time.perf_counter()
    try:
        for kind, piece in chat_answer(history=[], latest_question=q["question"],
                                       hits=hits, cfg=cfg, stream=True):
            if kind == "thinking":
                thinking_chars += len(piece)
            else:
                text += piece
    except Exception as e:
        error = f"llm error: {e}"
        issues.append("llm-failed")
    elapsed = time.perf_counter() - t0

    # Quality checks
    must = q.get("must", [])
    any_of = q.get("any_of", [])
    must_hit, any_of_hit = _check_substring(text, must, any_of)

    if not text.strip():
        issues.append("empty-answer")
    elif not must_hit and (q["category"] != "general" or use_rag):
        # For general questions, the model should know — but RAG-off is the
        # cleaner test of model knowledge.
        issues.append(f"missing-required:{must}")
    elif not any_of_hit:
        issues.append(f"missing-any-of:{any_of}")
    if elapsed > 90:
        issues.append(f"slow:{elapsed:.1f}s")

    return Result(
        qid=q["id"], category=q["category"], use_rag=use_rag,
        elapsed_s=elapsed, answer=text, thinking_chars=thinking_chars,
        citations=[(h.path, h.page, round(h.score, 3)) for h in hits],
        expected_doc_in_topk=expected_doc_in_topk,
        must_hit=must_hit, any_of_hit=any_of_hit,
        issues=issues, error=error,
    )


def loop(model: str, workspace: Path, *, label: str, max_tokens: int = 4096) -> list[Result]:
    """Run the full Q-set under the given model, both with and without RAG."""
    print(f"\n{'='*78}")
    print(f"  LOOP: {label}   model={model}")
    print(f"{'='*78}")
    ws = Workspace(workspace)
    cfg = ws.load_config()
    cfg.llm_model = model
    cfg.max_tokens = max_tokens
    cfg.temperature = 0.2
    clear_embedder_cache()
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)

    backend = detect_backend(cfg)
    print(f"  backend={backend}  embedder={embedder.name}  embed_dim={embedder.dim}")

    results: list[Result] = []
    for q in QUESTIONS:
        for use_rag in [True, False]:
            mode = "RAG" if use_rag else "RAW"
            print(f"  [{q['id']:>3} {q['category']:<7} {mode:>3}] ", end="", flush=True)
            r = run_question(q, cfg=cfg, embedder=embedder, idx=idx, use_rag=use_rag)
            tag = "✓" if (r.must_hit and r.any_of_hit and not r.issues) else "✗"
            print(f"{tag}  {r.elapsed_s:>5.1f}s  ans={len(r.answer):>4}c"
                  f"  think={r.thinking_chars:>4}c"
                  f"  cites={len(r.citations):>2}"
                  f"  issues={','.join(r.issues) or '-'}")
            results.append(r)
    return results


def render_summary(loops: dict[str, list[Result]]) -> str:
    lines = []
    lines.append("# RAG comparison — summary\n")
    for label, results in loops.items():
        lines.append(f"## {label}\n")

        rag = [r for r in results if r.use_rag]
        raw = [r for r in results if not r.use_rag]

        rag_hit = sum(1 for r in rag if r.must_hit and r.any_of_hit and not r.error)
        raw_hit = sum(1 for r in raw if r.must_hit and r.any_of_hit and not r.error)
        rag_t = sum(r.elapsed_s for r in rag) / max(1, len(rag))
        raw_t = sum(r.elapsed_s for r in raw) / max(1, len(raw))
        rag_corpus_hit = sum(1 for r in rag if r.category == "corpus"
                              and r.must_hit and r.any_of_hit)
        raw_corpus_hit = sum(1 for r in raw if r.category == "corpus"
                              and r.must_hit and r.any_of_hit)
        n_corpus = sum(1 for r in rag if r.category == "corpus")

        lines.append(
            f"- Overall correct (RAG vs RAW): "
            f"**{rag_hit}/{len(rag)}**  vs  **{raw_hit}/{len(raw)}**"
        )
        lines.append(
            f"- Corpus-question correct: "
            f"**{rag_corpus_hit}/{n_corpus}**  vs  **{raw_corpus_hit}/{n_corpus}**"
        )
        lines.append(f"- Avg latency: RAG **{rag_t:.1f}s**, RAW **{raw_t:.1f}s**")
        lines.append("")

        # Per-question table
        lines.append("| Q | cat | mode | ok | latency | answer head | issues |")
        lines.append("|---|---|---|:---:|---:|---|---|")
        for r in results:
            ok = "✓" if (r.must_hit and r.any_of_hit and not r.error) else "✗"
            head = (r.answer or r.error or "(empty)").strip().replace("\n", " ")[:80]
            lines.append(
                f"| {r.qid} | {r.category} | "
                f"{'RAG' if r.use_rag else 'RAW'} | {ok} | "
                f"{r.elapsed_s:.1f}s | {head} | {', '.join(r.issues) or '-'} |"
            )
        lines.append("")

        # Issue list across the loop
        all_issues = []
        for r in results:
            for i in r.issues:
                all_issues.append((r.qid, "RAG" if r.use_rag else "RAW", i))
        if all_issues:
            lines.append("### Issues found")
            for qid, mode, i in all_issues:
                lines.append(f"- **{qid} {mode}**: {i}")
            lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path,
                    default=Path(r"C:/Users/jroun/workstuff/ez-rag-test"))
    ap.add_argument("--report", type=Path,
                    default=THIS / "reports" / "rag_compare.md")
    ap.add_argument("--quick", action="store_true",
                    help="Only run two loops (qwen2.5:3b + deepseek-r1:32b)")
    args = ap.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)

    loops = (
        [("L1 qwen2.5:3b",        "qwen2.5:3b"),
         ("L2 deepseek-r1:32b",   "deepseek-r1:32b")]
        if args.quick else
        [("L1 qwen2.5:3b",         "qwen2.5:3b"),
         ("L2 deepseek-r1:1.5b",   "deepseek-r1:1.5b"),
         ("L3 deepseek-r1:32b",    "deepseek-r1:32b"),
         ("L4 deepseek-r1:32b/2",  "deepseek-r1:32b")]
    )

    out: dict[str, list[Result]] = {}
    for label, model in loops:
        try:
            out[label] = loop(model, args.workspace, label=label)
        except Exception as e:
            traceback.print_exc()
            print(f"  LOOP FAILED: {e}")

    # Save
    md = render_summary(out)
    args.report.write_text(md, encoding="utf-8")
    print(f"\n\nReport written: {args.report}")

    # Also dump raw json for later inspection
    raw = {label: [asdict(r) for r in rs] for label, rs in out.items()}
    args.report.with_suffix(".json").write_text(
        json.dumps(raw, default=str, indent=2), encoding="utf-8",
    )

    # Console summary
    print("\n" + "="*78)
    print("  CONSOLIDATED")
    print("="*78)
    for label, rs in out.items():
        rag_correct = sum(1 for r in rs if r.use_rag and r.must_hit and r.any_of_hit and not r.error)
        raw_correct = sum(1 for r in rs if not r.use_rag and r.must_hit and r.any_of_hit and not r.error)
        n_each = len(rs) // 2
        rag_corpus = sum(1 for r in rs if r.use_rag and r.category=="corpus" and r.must_hit and r.any_of_hit)
        raw_corpus = sum(1 for r in rs if not r.use_rag and r.category=="corpus" and r.must_hit and r.any_of_hit)
        n_corpus = sum(1 for r in rs if r.use_rag and r.category=="corpus")
        print(f"  {label:<28}  overall RAG {rag_correct}/{n_each}  RAW {raw_correct}/{n_each}"
              f"   corpus RAG {rag_corpus}/{n_corpus}  RAW {raw_corpus}/{n_corpus}")


if __name__ == "__main__":
    main()

"""LLM-as-judge evaluation of bench results.

The keyword-presence heuristic in bench_dnd5e_quality.py is misleading —
it rewards generic terminology ("monster", "magic") and penalizes
answers that list specific named items instead. This module replaces it
with rubric-based LLM scoring on four dimensions:

  - addresses (0-3): does the answer answer the literal question?
  - specificity (0-3): are there concrete examples / specific names?
  - grounded   (0-3): is it actually drawn from the context (not made up)?
  - on_topic   (0-3): does it stay on topic, no drift?

Total = sum (0-12).

Each answer gets a single judge call. The judge runs against the same
local Ollama instance, but uses a strict, deterministic prompt and only
sees the QUESTION + ANSWER + retrieved SOURCES — never the corpus
itself, so it judges quality of the answer as written, not factual
correctness against ground truth.

Usage:
    python bench/judge_eval.py path/to/bench-results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.generate import _ollama_chat


JUDGE_PROMPT = """You are a strict evaluator scoring a RAG answer.

You will see:
  - QUESTION: what the user asked
  - SOURCES: citation markers from chunks the system retrieved
  - ANSWER: what the model produced

Score the ANSWER on four dimensions, each 0-3:

addresses  — does the answer DIRECTLY answer the literal question?
              0 = answers a different question entirely
              1 = tangentially related; vague
              2 = mostly answers but with drift
              3 = directly and clearly answers

specificity — does it contain SPECIFIC named items / proper nouns / concrete examples?
              0 = only generic terminology, no named items
              1 = one or two named items
              2 = several named items
              3 = many specific named items, well-organized

grounded    — does it appear to be drawn from real document content (vs hallucinated generic info)?
              0 = generic platitudes, no source-grounded detail
              1 = some grounded detail but mostly generic
              2 = clearly grounded in retrieved sources
              3 = densely grounded with quotes, page numbers, proper nouns

on_topic    — does the answer stay on topic without drifting?
              0 = drifts to a different subject mid-answer
              1 = drifts somewhat
              2 = mostly on topic
              3 = laser-focused on the question

Output ONE LINE in EXACTLY this format:
addresses=N specificity=N grounded=N on_topic=N

No commentary. No preamble. JUST the four scores.

QUESTION: {question}

SOURCES: {sources}

ANSWER:
{answer}
"""

SCORE_RE = re.compile(
    r"addresses\s*=\s*(\d).*?specificity\s*=\s*(\d).*?grounded\s*=\s*(\d).*?on_topic\s*=\s*(\d)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class JudgeScore:
    addresses: int = 0
    specificity: int = 0
    grounded: int = 0
    on_topic: int = 0
    raw: str = ""
    err: str = ""

    @property
    def total(self) -> int:
        return self.addresses + self.specificity + self.grounded + self.on_topic


def judge_one(question: str, answer: str, sources: list[str],
               cfg: Config) -> JudgeScore:
    if not answer or not answer.strip():
        return JudgeScore(err="empty answer")
    src = "; ".join(sources[:8]) if sources else "(none)"
    # Cap answer at 4 KB for the judge — we're scoring quality of the
    # opening response, not throughput. This keeps judge prompt small.
    capped = answer[:4000] + ("…[truncated]" if len(answer) > 4000 else "")
    prompt = JUDGE_PROMPT.format(question=question, sources=src,
                                  answer=capped)
    msgs = [
        {"role": "system",
         "content": "You are a strict, terse evaluator. Output exactly one line."},
        {"role": "user", "content": prompt},
    ]
    try:
        out = _ollama_chat(cfg, msgs)
    except Exception as ex:
        return JudgeScore(err=f"{type(ex).__name__}: {ex}")
    out = (out or "").strip()
    m = SCORE_RE.search(out)
    if not m:
        return JudgeScore(raw=out[:200], err="unparseable")
    try:
        a, s, g, t = (int(m.group(i)) for i in range(1, 5))
        return JudgeScore(
            addresses=min(3, max(0, a)),
            specificity=min(3, max(0, s)),
            grounded=min(3, max(0, g)),
            on_topic=min(3, max(0, t)),
            raw=out[:200],
        )
    except (TypeError, ValueError):
        return JudgeScore(raw=out[:200], err="bad ints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="path to bench results JSON")
    ap.add_argument("--judge-model", default="qwen2.5:7b",
                     help="model to use as judge")
    ap.add_argument("--llm-url", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0,
                     help="cap number of judgments")
    args = ap.parse_args()

    raw = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    # Accept either a flat list (from bench_dnd5e_quality.py) or
    # the wrapped form {"results": [...]} (from multi_model_sweep.py).
    if isinstance(raw, dict) and "results" in raw:
        results = raw["results"]
    else:
        results = raw
    if args.limit:
        results = results[: args.limit]

    cfg = Config()
    cfg.llm_url = args.llm_url
    cfg.llm_model = args.judge_model
    cfg.temperature = 0.0   # judge should be deterministic
    cfg.max_tokens = 50     # one line of scores

    print(f"Judging {len(results)} answers with {args.judge_model}…\n")
    out: list[dict] = []
    t0 = time.perf_counter()

    for i, r in enumerate(results, start=1):
        score = judge_one(
            question=r["question"],
            answer=r.get("answer", ""),
            sources=r.get("sources", []),
            cfg=cfg,
        )
        merged = dict(r)
        merged["judge_addresses"] = score.addresses
        merged["judge_specificity"] = score.specificity
        merged["judge_grounded"] = score.grounded
        merged["judge_on_topic"] = score.on_topic
        merged["judge_total"] = score.total
        merged["judge_err"] = score.err
        merged["judge_raw"] = score.raw
        out.append(merged)
        elapsed = time.perf_counter() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(results) - i) / rate if rate else 0
        flag = ""
        if score.err:
            flag = f"  ERR: {score.err}"
        # Identifier for log line: prefer strategy (single-cfg bench),
        # fall back to model (multi-model sweep).
        ident = r.get("strategy") or r.get("model") or "?"
        print(
            f"[{i}/{len(results)}] {ident[:18]:18s} "
            f"a={score.addresses} s={score.specificity} "
            f"g={score.grounded} t={score.on_topic} = {score.total}/12  "
            f"({elapsed:.0f}s, ~{int(eta)}s left){flag}"
        )

    out_path = Path(args.results_json).with_name(
        Path(args.results_json).stem + "-judged.json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Summary — group by strategy if present, otherwise by model
    by_strat: dict[str, dict] = {}
    for r in out:
        key = r.get("strategy") or r.get("model") or "?"
        s = by_strat.setdefault(key, {
            "n": 0, "addresses": 0, "specificity": 0,
            "grounded": 0, "on_topic": 0, "total": 0, "errs": 0,
        })
        s["n"] += 1
        if r.get("judge_err"):
            s["errs"] += 1
            continue
        s["addresses"] += r["judge_addresses"]
        s["specificity"] += r["judge_specificity"]
        s["grounded"] += r["judge_grounded"]
        s["on_topic"] += r["judge_on_topic"]
        s["total"] += r["judge_total"]

    print("\n=== Judge summary (avg per answer, 0-3 each, 0-12 total) ===")
    print(f"{'strategy':<12} {'addr':<5} {'spec':<5} {'grnd':<5} "
          f"{'topic':<5} {'TOTAL':<6} n  errs")
    for name, s in by_strat.items():
        ok = max(1, s["n"] - s["errs"])
        print(
            f"{name:<12} "
            f"{s['addresses']/ok:<5.2f} {s['specificity']/ok:<5.2f} "
            f"{s['grounded']/ok:<5.2f} {s['on_topic']/ok:<5.2f} "
            f"{s['total']/ok:<6.2f} {s['n']:<3d} {s['errs']:d}"
        )
    print(f"\nDetailed JSON: {out_path}")


if __name__ == "__main__":
    sys.exit(main())

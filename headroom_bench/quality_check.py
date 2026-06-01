"""Quality-preservation check for the headroom compression bench.

Reads compressed.json (real headroom output from WSL). For a sampled
subset of cases, generates an answer from BOTH the original and the
compressed context via Ollama, then LLM-judges both answers on the
same 0-12 rubric ez-rag's bench uses. The delta tells us whether
compression hurt answer quality.

This runs on Windows (native Ollama). Generation + judging only — the
compression already happened in WSL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import httpx

OLLAMA = "http://127.0.0.1:11434"


def ollama_chat(model: str, messages: list[dict], *, num_ctx: int = 16384,
                timeout: float = 180.0) -> str:
    r = httpx.post(
        OLLAMA + "/api/chat",
        json={"model": model, "messages": messages, "stream": False,
              "think": False,
              "options": {"temperature": 0.2, "num_ctx": num_ctx,
                          "num_predict": 1024}},
        timeout=timeout,
    )
    r.raise_for_status()
    return (r.json().get("message", {}) or {}).get("content", "") or ""


JUDGE_PROMPT = """You are a strict grader. Score the ANSWER to the QUESTION on four axes, each 0-3:
- addresses: does it actually answer the question asked?
- specificity: concrete, specific details vs vague generalities?
- grounded: supported by / consistent with the kind of source material cited?
- on_topic: stays on the question without drifting?

Return ONLY a JSON object: {"addresses":N,"specificity":N,"grounded":N,"on_topic":N}

QUESTION: {question}

ANSWER:
{answer}
"""


def judge(judge_model: str, question: str, answer: str) -> dict:
    if not (answer or "").strip():
        return {"addresses": 0, "specificity": 0, "grounded": 0,
                "on_topic": 0, "total": 0, "err": "empty answer"}
    prompt = JUDGE_PROMPT.replace("{question}", question).replace("{answer}", answer[:6000])
    try:
        raw = ollama_chat(judge_model,
                          [{"role": "user", "content": prompt}],
                          num_ctx=8192, timeout=120)
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        a = int(obj.get("addresses", 0)); s = int(obj.get("specificity", 0))
        g = int(obj.get("grounded", 0)); t = int(obj.get("on_topic", 0))
        a, s, g, t = (max(0, min(3, x)) for x in (a, s, g, t))
        return {"addresses": a, "specificity": s, "grounded": g,
                "on_topic": t, "total": a + s + g + t, "err": ""}
    except Exception as ex:
        return {"addresses": 0, "specificity": 0, "grounded": 0,
                "on_topic": 0, "total": 0, "err": f"{type(ex).__name__}: {ex}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(REPO / "headroom_bench" / "compressed.json"))
    ap.add_argument("--out", dest="out",
                    default=str(REPO / "headroom_bench" / "quality.json"))
    ap.add_argument("--gen-model", default="qwen2.5:7b")
    ap.add_argument("--judge-model", default="qwen2.5:7b")
    ap.add_argument("--sample", type=int, default=40,
                    help="number of cases to quality-check")
    ap.add_argument("--min-saved", type=int, default=1,
                    help="only check cases where compression saved >= this many tokens")
    ap.add_argument("--num-ctx", type=int, default=16384,
                    help="Ollama context window. Set large enough that the "
                         "ORIGINAL (uncompressed) context is never truncated, "
                         "so the quality delta isolates pruning effects from "
                         "context-fit effects. Max original here is ~9.9k tok.")
    args = ap.parse_args()

    results = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    # Prefer cases where compression actually did something, spread across
    # questions/embedders/depths. Deterministic stride sampling (no RNG).
    eligible = [r for r in results
                if not r["error"] and r["tokens_saved"] >= args.min_saved
                and r.get("compressed_messages")]
    if not eligible:
        print("[!] no eligible compressed cases to quality-check")
        eligible = [r for r in results if r.get("compressed_messages")]
    stride = max(1, len(eligible) // args.sample)
    sample = eligible[::stride][:args.sample]
    print(f"quality-checking {len(sample)} of {len(eligible)} eligible cases "
          f"(gen={args.gen_model}, judge={args.judge_model})")

    out = []
    t_all = time.perf_counter()
    for i, r in enumerate(sample, 1):
        case_id = r["case_id"]
        q = r["question"]
        # Reconstruct the ORIGINAL messages from the source case file.
        orig_msgs = _original_messages(case_id)
        comp_msgs = r["compressed_messages"]
        if orig_msgs is None:
            continue
        try:
            ans_orig = ollama_chat(args.gen_model, orig_msgs, num_ctx=args.num_ctx)
            ans_comp = ollama_chat(args.gen_model, comp_msgs, num_ctx=args.num_ctx)
        except Exception as ex:
            print(f"  [{i}] gen err: {ex}")
            continue
        j_orig = judge(args.judge_model, q, ans_orig)
        j_comp = judge(args.judge_model, q, ans_comp)
        row = {
            "case_id": case_id, "question": q,
            "category": r["category"], "embedder": r["embedder"],
            "depth": r["depth"],
            "tokens_before": r["tokens_before"],
            "tokens_after": r["tokens_after"],
            "tokens_saved": r["tokens_saved"],
            "compression_ratio": r["compression_ratio"],
            "score_orig": j_orig["total"], "score_comp": j_comp["total"],
            "score_delta": j_comp["total"] - j_orig["total"],
            "judge_orig": j_orig, "judge_comp": j_comp,
            "answer_orig": ans_orig[:1200],
            "answer_comp": ans_comp[:1200],
        }
        out.append(row)
        print(f"  [{i:2d}/{len(sample)}] case {case_id:3d} {r['embedder']:11s} "
              f"d{r['depth']:<2d} saved {r['tokens_saved']:4d}tok  "
              f"orig {j_orig['total']:2d}/12 -> comp {j_comp['total']:2d}/12 "
              f"(Δ{j_comp['total']-j_orig['total']:+d})")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Summary
    if out:
        n = len(out)
        avg_o = sum(x["score_orig"] for x in out) / n
        avg_c = sum(x["score_comp"] for x in out) / n
        avg_saved = sum(x["tokens_saved"] for x in out) / n
        print(f"\n[OK] wrote {n} quality rows -> {args.out}")
        print(f"     avg score: orig {avg_o:.2f}/12 -> comp {avg_c:.2f}/12 "
              f"(Δ {avg_c-avg_o:+.2f})")
        print(f"     avg tokens saved/case: {avg_saved:.0f}")
        print(f"     elapsed: {time.perf_counter()-t_all:.0f}s")
    return 0


_CASES_CACHE = None


def _original_messages(case_id: int):
    global _CASES_CACHE
    if _CASES_CACHE is None:
        cases = json.loads(
            (REPO / "headroom_bench" / "cases.json").read_text(encoding="utf-8")
        )
        _CASES_CACHE = {c["case_id"]: c["messages"] for c in cases}
    return _CASES_CACHE.get(case_id)


if __name__ == "__main__":
    raise SystemExit(main())

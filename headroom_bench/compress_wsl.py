"""WSL-side headroom compressor for the ez-rag compression bench.

Reads cases.json (built on the Windows side by build_cases.py),
compresses each case's messages with the real headroom pipeline
(Rust _core + ModernBERT relevance model), and writes compressed.json
with per-case token metrics + the compressed messages so the
Windows side can run generation/quality scoring.

Run inside WSL:
    python3 headroom_bench/compress_wsl.py \
        --in  /mnt/c/.../headroom_bench/cases.json \
        --out /mnt/c/.../headroom_bench/compressed.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import headroom


def compress_case(case: dict, *, model: str, model_limit: int) -> dict:
    messages = case["messages"]
    t0 = time.perf_counter()
    err = ""
    try:
        res = headroom.compress(
            messages,
            model=model,
            model_limit=model_limit,
            compress_user_messages=True,   # RAG context lives in the user turn
        )
        dt = time.perf_counter() - t0
        return {
            **{k: case[k] for k in
               ("case_id", "embedder", "question", "category",
                "depth", "n_hits", "sources")},
            "tokens_before": res.tokens_before,
            "tokens_after": res.tokens_after,
            "tokens_saved": res.tokens_saved,
            "compression_ratio": round(res.compression_ratio, 4),
            "transforms_applied": res.transforms_applied,
            "compress_seconds": round(dt, 3),
            "compressed_messages": res.messages,
            "error": "",
        }
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"
        traceback.print_exc()
        return {
            **{k: case[k] for k in
               ("case_id", "embedder", "question", "category",
                "depth", "n_hits", "sources")},
            "tokens_before": 0, "tokens_after": 0, "tokens_saved": 0,
            "compression_ratio": 0.0, "transforms_applied": [],
            "compress_seconds": round(time.perf_counter() - t0, 3),
            "compressed_messages": messages,
            "error": err,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="gpt-4o",
                    help="model name for token counting (cl100k_base)")
    ap.add_argument("--model-limit", type=int, default=32768)
    args = ap.parse_args()

    cases = json.loads(open(args.inp, encoding="utf-8").read())
    print(f"loaded {len(cases)} cases")

    # Warm up the model once (first call loads ModernBERT, ~15s).
    if cases:
        print("warming up compressor…")
        _ = compress_case(cases[0], model=args.model,
                          model_limit=args.model_limit)

    results = []
    t_all = time.perf_counter()
    for i, case in enumerate(cases, 1):
        r = compress_case(case, model=args.model, model_limit=args.model_limit)
        results.append(r)
        if i % 20 == 0 or i == len(cases):
            saved = sum(x["tokens_saved"] for x in results)
            before = sum(x["tokens_before"] for x in results)
            pct = (saved / before * 100) if before else 0.0
            print(f"[{i:3d}/{len(cases)}] cumulative saved "
                  f"{saved}/{before} tok ({pct:.1f}%) "
                  f"elapsed {time.perf_counter()-t_all:.0f}s")

    open(args.out, "w", encoding="utf-8").write(
        json.dumps(results, indent=2)
    )
    # Summary
    before = sum(r["tokens_before"] for r in results)
    after = sum(r["tokens_after"] for r in results)
    saved = before - after
    errs = sum(1 for r in results if r["error"])
    print(f"\n[OK] wrote {len(results)} results -> {args.out}")
    print(f"     tokens: before={before} after={after} "
          f"saved={saved} ({saved/before*100:.1f}%)" if before else "     no tokens")
    print(f"     errors: {errs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

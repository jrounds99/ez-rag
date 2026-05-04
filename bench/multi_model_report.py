"""Generate a Markdown report from a judged multi-model sweep.

Input: a `*-judged.json` produced by judge_eval.py over a multimodel-*.json.
Output: a Markdown report with:
  - Per-model average quality (addresses, specificity, grounded, on_topic)
  - Per-category breakdown
  - ASCII bar plot: quality vs parameter count
  - ASCII bar plot: quality per GB VRAM (efficiency)
  - The "knee" — smallest model still within 10% of the best
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def fmt_bar(value, max_value, width=30):
    if max_value <= 0:
        return ""
    filled = int(round((value / max_value) * width))
    return "█" * filled + "░" * (width - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged_json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.judged_json).read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    else:
        rows = data.get("results", [])

    # Aggregate per model
    by_model: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "errs": 0, "addresses": 0, "specificity": 0,
        "grounded": 0, "on_topic": 0, "total": 0, "seconds": 0.0,
        "params_b": 0.0, "family": "",
        "by_cat": defaultdict(lambda: {"n": 0, "total": 0}),
    })
    for r in rows:
        m = by_model[r["model"]]
        m["params_b"] = r.get("params_b", 0.0)
        m["family"] = r.get("family", "")
        m["n"] += 1
        m["seconds"] += r.get("seconds", 0)
        if r.get("judge_err") or r.get("err"):
            m["errs"] += 1
            continue
        for k in ("addresses", "specificity", "grounded", "on_topic"):
            m[k] += r.get(f"judge_{k}", 0)
        total = r.get("judge_total", 0)
        m["total"] += total
        cat = r.get("category", "?")
        c = m["by_cat"][cat]
        c["n"] += 1
        c["total"] += total

    # Sort by params for plotting
    ordered = sorted(by_model.items(), key=lambda kv: kv[1]["params_b"])
    if not ordered:
        print("No data")
        return 1

    # ===== Build report =====
    lines = ["# Multi-model RAG quality sweep\n"]
    lines.append(f"Models tested: **{len(ordered)}**  ·  "
                  f"questions per model: "
                  f"{ordered[0][1]['n']}\n")
    lines.append("> Higher = better. Maximum is 12.0\n")

    # ----- Headline table -----
    lines.append("## Headline scores\n")
    lines.append("| Model | Params | Avg /12 | Addr | Spec | Grnd | Topic | Avg s/q | Errs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    headline_rows = []
    for tag, m in ordered:
        ok = max(1, m["n"] - m["errs"])
        avg = m["total"] / ok
        addr = m["addresses"] / ok
        spec = m["specificity"] / ok
        grnd = m["grounded"] / ok
        topic = m["on_topic"] / ok
        secs = m["seconds"] / max(1, m["n"])
        headline_rows.append((tag, m["params_b"], avg, addr, spec,
                                grnd, topic, secs, m["errs"]))
        lines.append(
            f"| `{tag}` | {m['params_b']:.1f}B | **{avg:.2f}** | "
            f"{addr:.2f} | {spec:.2f} | {grnd:.2f} | {topic:.2f} | "
            f"{secs:.1f}s | {m['errs']} |"
        )
    lines.append("")

    # ----- Quality vs params (ASCII plot) -----
    lines.append("## Quality vs parameter count\n")
    lines.append("```")
    max_score = max(r[2] for r in headline_rows)
    for tag, params, avg, *_ in headline_rows:
        bar = fmt_bar(avg, 12.0)
        lines.append(
            f"  {tag:<22s} {params:>6.1f}B  {avg:5.2f}  {bar}"
        )
    lines.append("```\n")

    # ----- Find the knee (smallest model within 10% of best) -----
    best = max(headline_rows, key=lambda r: r[2])
    threshold = best[2] * 0.90
    knee = None
    for r in headline_rows:    # ordered by params asc
        if r[2] >= threshold:
            knee = r
            break
    if knee:
        lines.append(
            f"## The knee\n\n"
            f"Best score: **{best[0]}** at **{best[2]:.2f}/12** "
            f"({best[1]:.1f}B params).\n\n"
            f"Smallest model that scores within 10% of best "
            f"(≥ {threshold:.2f}/12): "
            f"**`{knee[0]}` at {knee[2]:.2f}/12 with only "
            f"{knee[1]:.1f}B parameters**.\n\n"
        )
        # Compute "size savings"
        if knee[0] != best[0]:
            ratio = best[1] / max(0.01, knee[1])
            lines.append(
                f"That's a **{ratio:.1f}× smaller** model giving "
                f"{(knee[2]/best[2]*100):.0f}% of the quality.\n"
            )

    # ----- Quality per GB (efficiency) -----
    lines.append("## Quality per billion parameters (efficiency)\n")
    lines.append("> Higher = more quality squeezed out of each B params.\n")
    lines.append("```")
    eff_rows = sorted(headline_rows, key=lambda r: r[2] / max(0.1, r[1]),
                       reverse=True)
    max_eff = eff_rows[0][2] / max(0.1, eff_rows[0][1])
    for tag, params, avg, *_ in eff_rows:
        eff = avg / max(0.1, params)
        bar = fmt_bar(eff, max_eff)
        lines.append(
            f"  {tag:<22s} {eff:5.2f}/B  {bar}"
        )
    lines.append("```\n")

    # ----- Per-category breakdown -----
    lines.append("## Per-category breakdown\n")
    cats = sorted({c for _, m in ordered for c in m["by_cat"]})
    header = "| Model | " + " | ".join(cats) + " |"
    sep = "|---|" + "|".join(["---"] * len(cats)) + "|"
    lines.append(header)
    lines.append(sep)
    for tag, m in ordered:
        row = [f"`{tag}`"]
        for c in cats:
            d = m["by_cat"].get(c, {"n": 0, "total": 0})
            if d["n"] == 0:
                row.append("—")
            else:
                row.append(f"{d['total']/d['n']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ----- VRAM cost reference (rough Q4_K_M sizes) -----
    lines.append("## Approximate VRAM cost (Q4_K_M)\n")
    lines.append("| Params | VRAM (weights) | + KV cache @ 8K ctx | Total |")
    lines.append("|---|---|---|---|")
    for params in (0.5, 1.5, 3, 7, 8, 14, 32):
        weights_gb = params * 0.55     # rough Q4 ≈ 0.55 GB per B params
        kv_gb = params * 0.06           # rough KV at 8K ctx
        total = weights_gb + kv_gb + 0.5  # OS slack
        lines.append(
            f"| {params}B | ~{weights_gb:.1f} GB | ~{kv_gb:.1f} GB | "
            f"~{total:.1f} GB |"
        )

    out = "\n".join(lines) + "\n"
    out_path = (Path(args.out) if args.out
                 else Path(args.judged_json).with_suffix(".report.md"))
    out_path.write_text(out, encoding="utf-8")
    print(f"[OK] Report: {out_path}")
    print()
    # Also dump the headline + knee to stdout for quick reading
    print("=== Headline ===")
    print(f"{'model':<22s} {'params':>7s}  {'score':>5s}")
    for r in headline_rows:
        print(f"{r[0]:<22s} {r[1]:>6.1f}B  {r[2]:5.2f}")
    if knee:
        print(f"\nKnee (within 10% of best): {knee[0]} @ "
               f"{knee[2]:.2f}/12 ({knee[1]:.1f}B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

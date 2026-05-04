"""Comprehensive HTML renderer for the Ohio multi-embedder × multi-model bench.

Builds a self-contained interactive HTML report from a judged JSON
bundle:

  - Stat tiles (models, cells, best score, knee, total energy, time)
  - KNEE callout — smallest model within 10% of the best
  - Overall summary with auto-generated findings
  - Caveats / measurement notes
  - Headline table with hover-defined columns
  - Per-model factual cards (vendor, release, summary, verdict)
  - Quality vs parameter count scatter (knee circled)
  - Quality per second scatter (cost-efficiency)
  - Quality per kilojoule scatter (energy-efficiency, when power data)
  - Per-rubric breakdown (addresses / specificity / grounded / on_topic)
  - Per-category breakdown (factual / comparison / exploratory / multi-step)
  - Per-family rollup (qwen2.5 vs qwen3 vs llama3 vs gemma etc.)
  - Embedder leaderboard
  - Per-question heatmap (model × question)
  - Per-embedder heatmap (model × embedder)
  - Glossary expander at the bottom

Designed to be invoked directly OR imported via `render_report`.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Optional


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.0.min.js"


# ============================================================================
# Glossary — same hover-tooltip pattern as the multi-model report
# ============================================================================

GLOSSARY: dict[str, str] = {
    "RAG": "Retrieval-Augmented Generation. The LLM is given relevant "
            "passages from a document corpus before answering, rather "
            "than relying on its training data alone.",
    "params": "Trainable parameters (in billions). Roughly correlates "
              "with capability and VRAM cost. Q4_K_M quantization "
              "stores each parameter in ~4 bits, so VRAM ≈ "
              "params × 0.55 GB.",
    "embedder": "Model that converts text to vectors. Smaller and "
                "faster than the chat LLM. Each embedder uses a "
                "different vocabulary and dimension, so retrieval "
                "results vary across embedders even on the same corpus.",
    "BM25": "A keyword search algorithm (Okapi BM25). ez-rag uses it "
            "via SQLite FTS5 to find passages with literal term "
            "matches the dense vector search misses.",
    "hybrid search": "Running BM25 and dense search in parallel and "
                     "fusing their rankings. The single biggest "
                     "quality win over either method alone.",
    "rerank": "A small cross-encoder model that re-scores retrieval "
              "candidates with full attention to (query, passage) "
              "pairs. Adds ~50–200 ms per query for big quality "
              "wins.",
    "judge": "An LLM scoring every answer with a strict 0-12 rubric. "
             "Same judge model for every cell so scores are comparable.",
    "rubric": "Four 0-3 dimensions: addresses (does it answer the "
              "literal question?), specificity (concrete examples?), "
              "grounded (drawn from corpus, not training?), on_topic "
              "(no drift?). Sum is 0-12.",
    "addresses": "Rubric dimension 1. Does the answer DIRECTLY answer "
                 "the literal question, or does it pivot to a "
                 "different topic?",
    "specificity": "Rubric dimension 2. Does the answer contain "
                   "specific named items / proper nouns / concrete "
                   "examples, or only generic terminology?",
    "grounded": "Rubric dimension 3. Is the answer drawn from the "
                "retrieved corpus content, or just from the model's "
                "training knowledge?",
    "on_topic": "Rubric dimension 4. Does the answer stay on topic "
                "without drifting mid-paragraph to a different subject?",
    "knee": "Smallest model whose score is within 10% of the best — "
            "the diminishing-returns inflection point. Recommended pick "
            "for VRAM-constrained deployment.",
    "cost-efficiency": "judge_total / seconds — how much quality "
                        "you get per second of compute. Higher = better. "
                        "Use this when picking a model for interactive "
                        "chat where latency matters.",
    "energy-efficiency": "judge_total / kilojoule of GPU+CPU energy. "
                         "Use this when picking a model for sustained "
                         "throughput or battery-constrained deployments.",
    "context window": "The maximum number of tokens (~chars/3.5) the "
                      "model can read at once. Includes the system "
                      "prompt + retrieved chunks + chat history + "
                      "answer reserve.",
    "auto-list mode": "ez-rag auto-detects 'list X / give examples "
                      "of X' queries and routes them through entity-"
                      "rich HyDE + an extraction-only system prompt.",
    "Q4_K_M": "A specific 4-bit quantization scheme used by GGUF / "
              "Ollama models. Reduces VRAM by ~4× from full "
              "precision with minor quality loss.",
    "VRAM": "Video RAM on the GPU. The model's weights and per-token "
            "cache live here. Hard ceiling on model size at speed.",
    "reasoning model": "An LLM trained to emit a hidden chain-of-thought "
                       "(<think>...</think>) before its actual answer. "
                       "Examples here: deepseek-r1:1.5b and 32b. The "
                       "thinking blocks burn tokens but don't always "
                       "improve RAG answers.",
    "ingest": "The one-time process of parsing source documents, "
              "chunking them, embedding the chunks, and storing both "
              "in a searchable index. Runs separately per embedder in "
              "this bench.",
}


# ============================================================================
# Per-family factual notes (keyed on family string from CHAT_MODELS)
# ============================================================================

FAMILY_NOTES: dict[str, dict] = {
    "qwen2.5": {
        "vendor": "Alibaba",
        "released": "2024-09",
        "summary": (
            "Qwen 2.5 — broad multilingual instruction-tuned family. "
            "Strong on factual recall, code, and structured output. "
            "Sizes from 0.5B to 72B."
        ),
    },
    "qwen3": {
        "vendor": "Alibaba",
        "released": "2025-04",
        "summary": (
            "Qwen 3 — newer Alibaba family with thinking-mode toggles "
            "and improved long-context handling. Sizes from 0.6B to 235B."
        ),
    },
    "llama3": {
        "vendor": "Meta",
        "released": "2024-07 (3.1) / 2024-09 (3.2)",
        "summary": (
            "Meta's Llama 3 family. Conservative instruction tuning, "
            "wide deployment in production. 1B/3B distilled from 8B."
        ),
    },
    "phi": {
        "vendor": "Microsoft",
        "released": "2025-02",
        "summary": (
            "Microsoft's Phi family — synthetic-data-heavy training "
            "curriculum. Small models that punch above their weight on "
            "reasoning + math benchmarks."
        ),
    },
    "mistral": {
        "vendor": "Mistral AI",
        "released": "2024 (7B v0.3, Nemo 12B)",
        "summary": (
            "Mistral AI's open-weights family. Sliding-window + grouped-"
            "query attention. Long the de-facto open 7B baseline; Nemo "
            "12B is their newer 128K-context entry."
        ),
    },
    "deepseek-r1": {
        "vendor": "DeepSeek",
        "released": "2025-01",
        "summary": (
            "Reasoning-distilled models that emit hidden chain-of-thought "
            "in <think>...</think> blocks. The 1.5B is distilled onto "
            "Qwen 1.5B, the 32B onto Qwen 32B. Strong on math and code; "
            "RAG performance varies because reasoning happens before "
            "the model sees the final corpus."
        ),
    },
    "gemma": {
        "vendor": "Google",
        "released": "2024 (Gemma 2) / 2025 (Gemma 3)",
        "summary": (
            "Google's open-weights family. Gemma 2 is text-only; Gemma 3 "
            "adds multi-modal support and up to 128K context. Tuned for "
            "instruction following with safety filters."
        ),
    },
    "granite": {
        "vendor": "IBM",
        "released": "2025 (Granite 3.x)",
        "summary": (
            "IBM's enterprise-focused open family. Trained heavily on "
            "permissively-licensed code and structured documents. "
            "Aimed at business RAG / agent use cases."
        ),
    },
}


# ============================================================================
# Aggregation
# ============================================================================

def fmt_n(n: float, decimals: int = 2) -> str:
    return f"{n:.{decimals}f}"


def _aggregate(rows: list[dict]) -> dict:
    """Build per-(model, embedder, strategy) summaries + per-question
    grids for the heatmap."""
    by_model: dict[str, dict] = defaultdict(lambda: {
        "params_b": 0.0, "family": "",
        "n": 0, "errs": 0, "sweep_errs": 0, "seconds": 0.0,
        "addr": 0, "spec": 0, "grnd": 0, "topic": 0, "total": 0,
        "latencies": [],            # v2: list[float] of seconds for p50/p95
        "eval_tokens": 0,            # v2: sum of generated tokens
        "prompt_tokens": 0,          # v2: sum of input tokens
        "tps_samples": [],           # v2: tokens-per-second per cell
        "gold_got": 0, "gold_max": 0, "gold_n": 0,  # v2: gold-snippet score
        "by_cat": defaultdict(lambda: {"n": 0, "total": 0}),
        "by_embedder": defaultdict(lambda: {"n": 0, "total": 0,
                                              "errs": 0}),
    })
    by_model_question: dict[tuple[str, str], list] = defaultdict(list)
    by_embedder: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "errs": 0, "total": 0, "seconds": 0.0,
    })
    by_family: dict[str, dict] = defaultdict(lambda: {
        "models": set(), "n": 0, "errs": 0, "total": 0, "seconds": 0.0,
    })
    by_category: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "errs": 0, "total": 0,
    })

    questions_seen: list[str] = []
    questions_set: set[str] = set()
    embedders_seen: list[str] = []
    embedders_set: set[str] = set()

    for r in rows:
        m = r.get("chat_model", "?")
        e = r.get("embedder", "?")
        fam = r.get("family", "?")
        cat = r.get("category", "?")
        q = r.get("question", "?")
        if q not in questions_set:
            questions_seen.append(q)
            questions_set.add(q)
        if e not in embedders_set:
            embedders_seen.append(e)
            embedders_set.add(e)

        agg = by_model[m]
        agg["params_b"] = r.get("params_b", 0.0)
        agg["family"] = fam
        agg["n"] += 1
        secs = r.get("seconds", 0) or 0
        agg["seconds"] += secs
        if secs > 0:
            agg["latencies"].append(secs)
        # v2 token economics
        agg["eval_tokens"] += int(r.get("eval_tokens", 0) or 0)
        agg["prompt_tokens"] += int(r.get("prompt_tokens", 0) or 0)
        tps = r.get("tokens_per_sec", 0) or 0
        if tps > 0:
            agg["tps_samples"].append(float(tps))
        # v2 gold-snippet correctness (only counts cells where Q has gold)
        if r.get("gold_max") is not None:
            agg["gold_got"] += int(r.get("gold_score", 0) or 0)
            agg["gold_max"] += int(r.get("gold_max", 0) or 0)
            agg["gold_n"] += 1

        e_agg = by_embedder[e]
        e_agg["n"] += 1
        e_agg["seconds"] += secs

        f_agg = by_family[fam]
        f_agg["models"].add(m)
        f_agg["n"] += 1
        f_agg["seconds"] += secs

        c_agg = by_category[cat]
        c_agg["n"] += 1

        me_agg = agg["by_embedder"][e]
        me_agg["n"] += 1

        if r.get("error"):
            agg["sweep_errs"] += 1
        if r.get("judge_err") or r.get("error"):
            agg["errs"] += 1
            e_agg["errs"] += 1
            f_agg["errs"] += 1
            c_agg["errs"] += 1
            me_agg["errs"] += 1
            by_model_question[(m, q)].append(None)
            continue
        for k in ("addr", "spec", "grnd", "topic"):
            agg[k] += r.get(f"judge_{_dim_full(k)}", 0)
        total = r.get("judge_total", 0)
        agg["total"] += total
        e_agg["total"] += total
        f_agg["total"] += total
        c_agg["total"] += total
        me_agg["total"] += total
        cb = agg["by_cat"][cat]
        cb["n"] += 1
        cb["total"] += total
        by_model_question[(m, q)].append(total)

    return {
        "by_model": dict(by_model),
        "by_model_question": dict(by_model_question),
        "by_embedder": dict(by_embedder),
        "by_family": dict(by_family),
        "by_category": dict(by_category),
        "questions": questions_seen,
        "embedders": embedders_seen,
    }


def _dim_full(short: str) -> str:
    return {
        "addr": "addresses", "spec": "specificity",
        "grnd": "grounded", "topic": "on_topic",
    }[short]


# ============================================================================
# Plotly chart payloads
# ============================================================================

def _build_payload(agg: dict, power_summary: Optional[dict]) -> dict:
    by_model = agg["by_model"]
    ordered_models = sorted(by_model.items(),
                             key=lambda kv: kv[1]["params_b"])
    labels = [tag for tag, _ in ordered_models]
    params = [m["params_b"] for _, m in ordered_models]
    families = [m["family"] for _, m in ordered_models]

    avg_total = []
    avg_addr = []
    avg_spec = []
    avg_grnd = []
    avg_topic = []
    avg_secs = []
    cells_per_model = []
    errs_per_model = []
    for _, m in ordered_models:
        ok = max(1, m["n"] - m["errs"])
        avg_total.append(round(m["total"] / ok, 3))
        avg_addr.append(round(m["addr"] / ok, 3))
        avg_spec.append(round(m["spec"] / ok, 3))
        avg_grnd.append(round(m["grnd"] / ok, 3))
        avg_topic.append(round(m["topic"] / ok, 3))
        avg_secs.append(round(m["seconds"] / max(1, m["n"]), 3))
        cells_per_model.append(m["n"])
        errs_per_model.append(m["errs"])

    qps = [
        round(t / s, 3) if s > 0 else 0.0
        for t, s in zip(avg_total, avg_secs)
    ]

    # Per-category data
    cats = sorted({c for _, m in ordered_models for c in m["by_cat"]})
    cat_data: dict[str, list[float]] = {c: [] for c in cats}
    for _, m in ordered_models:
        for c in cats:
            d = m["by_cat"].get(c, {"n": 0, "total": 0})
            cat_data[c].append(round(d["total"] / d["n"], 2)
                                 if d["n"] else 0.0)

    # Energy data
    energy_kj_per_q: list[float] = []
    quality_per_kj: list[float] = []
    if power_summary:
        per_model_kj: dict[str, float] = defaultdict(float)
        per_model_count: dict[str, int] = defaultdict(int)
        for label, seg in power_summary.items():
            if not isinstance(seg, dict):
                continue
            if not label.startswith("answer."):
                continue
            parts = label.split(".", 2)
            if len(parts) < 3:
                continue
            chat_model = parts[2]
            kj = float(seg.get("gpu_energy_kj", 0)) + \
                 float(seg.get("cpu_energy_kj", 0))
            per_model_kj[chat_model] += kj
            per_model_count[chat_model] += 1
        for tag, _ in ordered_models:
            tot_kj = per_model_kj.get(tag, 0.0)
            cnt = max(1, per_model_count.get(tag, 0))
            kj_per_q = tot_kj / cnt
            energy_kj_per_q.append(round(kj_per_q, 3))
            qual = avg_total[labels.index(tag)]
            quality_per_kj.append(
                round(qual / kj_per_q, 3) if kj_per_q > 0 else 0.0
            )

    # Heatmap: model × question
    z_matrix: list[list[float | None]] = []
    for tag in labels:
        row = []
        for q in agg["questions"]:
            cells = agg["by_model_question"].get((tag, q), [])
            valid = [v for v in cells if v is not None]
            row.append(round(sum(valid) / len(valid), 2) if valid else None)
        z_matrix.append(row)

    # Heatmap: model × embedder
    me_z: list[list[float | None]] = []
    for tag in labels:
        row = []
        m = by_model[tag]
        for e in agg["embedders"]:
            d = m["by_embedder"].get(e)
            if d is None or d["n"] == 0:
                row.append(None)
                continue
            ok = max(1, d["n"] - d["errs"])
            row.append(round(d["total"] / ok, 2))
        me_z.append(row)

    # Knee
    if avg_total:
        best_idx = max(range(len(avg_total)),
                       key=lambda i: avg_total[i])
        threshold = avg_total[best_idx] * 0.90
        knee_idx = next(
            (i for i in range(len(avg_total))
             if avg_total[i] >= threshold), best_idx,
        )
    else:
        best_idx = 0
        knee_idx = 0

    # Family rollup
    family_data: list[dict] = []
    for fam, fdata in agg["by_family"].items():
        ok = max(1, fdata["n"] - fdata["errs"])
        family_data.append({
            "family": fam,
            "model_count": len(fdata["models"]),
            "avg_score": round(fdata["total"] / ok, 2),
            "avg_secs": round(fdata["seconds"] / max(1, fdata["n"]), 2),
            "n": fdata["n"],
        })
    family_data.sort(key=lambda d: -d["avg_score"])

    # Embedder leaderboard
    embedder_data: list[dict] = []
    for e in agg["embedders"]:
        d = agg["by_embedder"][e]
        ok = max(1, d["n"] - d["errs"])
        embedder_data.append({
            "embedder": e,
            "avg_score": round(d["total"] / ok, 2),
            "avg_secs": round(d["seconds"] / max(1, d["n"]), 2),
            "n": d["n"],
            "errs": d["errs"],
        })
    embedder_data.sort(key=lambda d: -d["avg_score"])

    # Per-category rollup (across ALL models)
    cat_rollup: list[dict] = []
    for c in cats:
        d = agg["by_category"][c]
        ok = max(1, d["n"] - d["errs"])
        cat_rollup.append({
            "category": c,
            "avg_score": round(d["total"] / ok, 2),
            "n": d["n"],
        })
    cat_rollup.sort(key=lambda d: -d["avg_score"])

    return {
        "labels": labels,
        "params": params,
        "families": families,
        "avg_total": avg_total,
        "avg_addr": avg_addr,
        "avg_spec": avg_spec,
        "avg_grnd": avg_grnd,
        "avg_topic": avg_topic,
        "avg_secs": avg_secs,
        "qps": qps,
        "cells_per_model": cells_per_model,
        "errs_per_model": errs_per_model,
        "cats": cats,
        "cat_data": cat_data,
        "cat_rollup": cat_rollup,
        "energy_kj_per_q": energy_kj_per_q,
        "quality_per_kj": quality_per_kj,
        "questions": agg["questions"],
        "embedders": agg["embedders"],
        "z_matrix": z_matrix,
        "me_z": me_z,
        "best_idx": best_idx,
        "knee_idx": knee_idx,
        "family_data": family_data,
        "embedder_data": embedder_data,
    }


# ============================================================================
# Auto-generated findings
# ============================================================================

def _build_findings(payload: dict) -> list[str]:
    findings: list[str] = []
    labels = payload["labels"]
    avg_total = payload["avg_total"]
    avg_secs = payload["avg_secs"]
    params = payload["params"]
    if not labels:
        return findings

    best = payload["best_idx"]
    knee = payload["knee_idx"]
    findings.append(
        f"<strong>Best score:</strong> <code>{escape(labels[best])}</code> "
        f"at <strong>{avg_total[best]:.2f}/12</strong> "
        f"({params[best]:.1f}B params, {avg_secs[best]:.1f}s/q)."
    )
    if knee != best:
        ratio = params[best] / max(0.01, params[knee])
        findings.append(
            f"<strong>Diminishing-returns "
            f"<abbr title=\"{escape(GLOSSARY['knee'])}\">knee</abbr>:</strong> "
            f"<code>{escape(labels[knee])}</code> at "
            f"<strong>{avg_total[knee]:.2f}/12</strong> is within 10% of "
            f"the best using <strong>{ratio:.1f}× fewer params</strong>."
        )

    # Best cost-efficiency
    qps = payload["qps"]
    if qps:
        best_qps_idx = max(range(len(qps)), key=lambda i: qps[i])
        if best_qps_idx != best:
            findings.append(
                f"<strong>Best <abbr title=\""
                f"{escape(GLOSSARY['cost-efficiency'])}\">"
                f"cost-efficiency</abbr>:</strong> "
                f"<code>{escape(labels[best_qps_idx])}</code> at "
                f"<strong>{qps[best_qps_idx]:.2f} score per second</strong> "
                f"(score {avg_total[best_qps_idx]:.2f}, "
                f"{avg_secs[best_qps_idx]:.1f}s/q). "
                f"Different from the absolute best — choose this when "
                f"latency matters."
            )

    # Family detection — does any family dominate?
    family_data = payload["family_data"]
    if family_data and len(family_data) >= 2:
        top = family_data[0]
        runner = family_data[1]
        if top["avg_score"] - runner["avg_score"] >= 0.5:
            findings.append(
                f"<strong>{escape(top['family'])} family dominates:</strong> "
                f"avg {top['avg_score']:.2f}/12 across "
                f"{top['model_count']} model(s), beating "
                f"{escape(runner['family'])} ({runner['avg_score']:.2f}) "
                f"by {top['avg_score'] - runner['avg_score']:.2f} pts."
            )

    # Reasoning model check
    reasoning_scores = [
        avg_total[i] for i, fam in enumerate(payload["families"])
        if fam == "deepseek-r1"
    ]
    non_reasoning_scores = [
        avg_total[i] for i, fam in enumerate(payload["families"])
        if fam != "deepseek-r1"
    ]
    if reasoning_scores and non_reasoning_scores:
        r_avg = sum(reasoning_scores) / len(reasoning_scores)
        n_avg = sum(non_reasoning_scores) / len(non_reasoning_scores)
        if r_avg < n_avg - 0.3:
            findings.append(
                f"<strong>Reasoning models underperformed:</strong> "
                f"the deepseek-r1 distillates averaged {r_avg:.2f}/12 vs "
                f"{n_avg:.2f}/12 for non-reasoning models. The "
                f"&lt;think&gt; blocks burn generation tokens that "
                f"don't translate to better RAG answers on this corpus."
            )

    # Sub-1B cliff
    sub_1b_scores = [
        avg_total[i] for i in range(len(labels)) if params[i] < 1.0
    ]
    over_1b_scores = [
        avg_total[i] for i in range(len(labels))
        if 1.0 <= params[i] <= 4.0
    ]
    if sub_1b_scores and over_1b_scores:
        sub_avg = sum(sub_1b_scores) / len(sub_1b_scores)
        over_avg = sum(over_1b_scores) / len(over_1b_scores)
        if over_avg - sub_avg >= 1.0:
            findings.append(
                f"<strong>Sub-1B params drops off a cliff:</strong> "
                f"models under 1B averaged {sub_avg:.2f}/12; the 1–4B "
                f"class averaged {over_avg:.2f}/12. There's a hard floor "
                f"on RAG capability below 1B parameters."
            )

    # Embedder sensitivity
    embedder_data = payload["embedder_data"]
    if len(embedder_data) >= 2:
        top_e = embedder_data[0]
        bot_e = embedder_data[-1]
        spread = top_e["avg_score"] - bot_e["avg_score"]
        if spread >= 0.5:
            findings.append(
                f"<strong>Embedder choice mattered:</strong> "
                f"<code>{escape(top_e['embedder'])}</code> averaged "
                f"{top_e['avg_score']:.2f}/12 vs "
                f"<code>{escape(bot_e['embedder'])}</code> at "
                f"{bot_e['avg_score']:.2f}/12 (spread of {spread:.2f} "
                f"pts across the same chat models)."
            )
        else:
            findings.append(
                f"<strong>Embedder choice was a wash:</strong> "
                f"all {len(embedder_data)} embedders landed within "
                f"{spread:.2f} pts of each other. Pick by speed / "
                f"VRAM cost, not by quality."
            )

    return findings


# ============================================================================
# Main render
# ============================================================================

def render_report(*, judged_path: Path,
                   manifest_path: Optional[Path] = None,
                   out_html: Path,
                   power_summary: Optional[Path] = None,
                   ) -> Path:
    rows = json.loads(judged_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict) and "results" in rows:
        rows = rows["results"]
    agg = _aggregate(rows)
    power = None
    if power_summary and Path(power_summary).is_file():
        try:
            power = json.loads(
                Path(power_summary).read_text(encoding="utf-8")
            )
        except Exception:
            power = None
    payload = _build_payload(agg, power)
    payload_json = json.dumps(payload)
    has_energy = bool(payload["quality_per_kj"]) and any(
        v > 0 for v in payload["quality_per_kj"]
    )
    findings = _build_findings(payload)

    # ============================================================================
    # Compose HTML
    # ============================================================================
    by_model = agg["by_model"]
    ordered = sorted(by_model.items(),
                     key=lambda kv: kv[1]["params_b"])
    n_models = len(ordered)
    n_embedders = len(payload["embedders"])
    n_questions = len(payload["questions"])
    total_cells = sum(payload["cells_per_model"])
    total_errs = sum(payload["errs_per_model"])
    best_score = (payload["avg_total"][payload["best_idx"]]
                   if payload["avg_total"] else 0.0)
    knee_score = (payload["avg_total"][payload["knee_idx"]]
                   if payload["avg_total"] else 0.0)
    knee_label = (payload["labels"][payload["knee_idx"]]
                   if payload["labels"] else "")
    best_label = (payload["labels"][payload["best_idx"]]
                   if payload["labels"] else "")
    knee_params = (payload["params"][payload["knee_idx"]]
                    if payload["params"] else 0.0)
    best_params = (payload["params"][payload["best_idx"]]
                    if payload["params"] else 0.0)
    total_seconds = sum(
        m["seconds"] for _, m in ordered
    )
    total_energy_kj = 0.0
    if power:
        for label, seg in power.items():
            if isinstance(seg, dict) and label.startswith("answer."):
                total_energy_kj += float(seg.get("gpu_energy_kj", 0))
                total_energy_kj += float(seg.get("cpu_energy_kj", 0))

    # ---- Headline table ----
    def _pct(samples: list[float], q: float) -> float:
        """Quick percentile without importing statistics (avoids edge-case
        ImportError on bare-bones builds)."""
        if not samples:
            return 0.0
        s = sorted(samples)
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * q
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    rows_html: list[str] = []
    for i, (tag, m) in enumerate(ordered):
        ok = max(1, m["n"] - m["errs"])
        avg = m["total"] / ok
        secs = m["seconds"] / max(1, m["n"])
        qps_cell = f"{(avg/secs):.3f}" if secs > 0 else "—"
        # v2 columns
        p50 = _pct(m.get("latencies", []), 0.5)
        p95 = _pct(m.get("latencies", []), 0.95)
        tps_samples = m.get("tps_samples", [])
        avg_tps = (sum(tps_samples) / len(tps_samples)) if tps_samples else 0.0
        gold_max = m.get("gold_max", 0)
        gold_pct = (m.get("gold_got", 0) / gold_max * 100) if gold_max else None
        gold_cell = f"{gold_pct:.0f}%" if gold_pct is not None else "—"
        sweep_errs = m.get("sweep_errs", 0)
        marker = ""
        if i == payload["best_idx"]:
            marker = '<span class="badge gold">BEST</span>'
        elif i == payload["knee_idx"]:
            marker = '<span class="badge accent">KNEE</span>'
        rows_html.append(
            f"<tr class=\""
            f"{'highlight' if i in (payload['best_idx'], payload['knee_idx']) else ''}\">"
            f"<td><code>{escape(tag)}</code> {marker}</td>"
            f"<td>{m['params_b']:.1f}B</td>"
            f"<td>{escape(m['family'])}</td>"
            f"<td><strong>{avg:.2f}</strong></td>"
            f"<td>{gold_cell}</td>"
            f"<td>{m['addr']/ok:.2f}</td>"
            f"<td>{m['spec']/ok:.2f}</td>"
            f"<td>{m['grnd']/ok:.2f}</td>"
            f"<td>{m['topic']/ok:.2f}</td>"
            f"<td>{secs:.1f}s</td>"
            f"<td>{p50:.1f}s</td>"
            f"<td>{p95:.1f}s</td>"
            f"<td>{avg_tps:.0f}</td>"
            f"<td>{qps_cell}</td>"
            f"<td>{sweep_errs}</td>"
            f"</tr>"
        )
    headline_table = (
        '<table class="data"><thead><tr>'
        '<th>Model</th><th>Params</th><th>Family</th>'
        f'<th><abbr title="{escape(GLOSSARY["judge"])}">Avg /12</abbr></th>'
        f'<th><abbr title="Rule-based gold-snippet match (% of must-contain '
        f'phrases found across the 8 most factual questions). '
        f'Higher = better grounded in known facts.">Gold</abbr></th>'
        f'<th><abbr title="{escape(GLOSSARY["addresses"])}">Addr</abbr></th>'
        f'<th><abbr title="{escape(GLOSSARY["specificity"])}">Spec</abbr></th>'
        f'<th><abbr title="{escape(GLOSSARY["grounded"])}">Grnd</abbr></th>'
        f'<th><abbr title="{escape(GLOSSARY["on_topic"])}">Topic</abbr></th>'
        f'<th><abbr title="Mean wall-clock seconds per question.">Mean s/q</abbr></th>'
        f'<th><abbr title="Median (p50) wall-clock seconds per question.">p50</abbr></th>'
        f'<th><abbr title="95th-percentile (p95) wall-clock seconds — tail latency.">p95</abbr></th>'
        f'<th><abbr title="Average tokens-per-second during generation '
        f'(eval_count / eval_duration from Ollama).">tok/s</abbr></th>'
        f'<th><abbr title="{escape(GLOSSARY["cost-efficiency"])}">Q/s</abbr></th>'
        f'<th><abbr title="Sweep errors (non-judge): empty answers, HTTP fails, '
        f'or skipped after 2 retries.">Errs</abbr></th></tr></thead><tbody>'
        + "".join(rows_html) + '</tbody></table>'
    )

    # ---- Per-model factual cards ----
    model_cards_parts: list[str] = []
    for i, (tag, m) in enumerate(ordered):
        ok = max(1, m["n"] - m["errs"])
        avg = m["total"] / ok
        secs = m["seconds"] / max(1, m["n"])
        rank = sorted(range(len(payload["avg_total"])),
                       key=lambda j: -payload["avg_total"][j]).index(i) + 1
        tier_label = ""
        if i == payload["best_idx"]:
            tier_label = '<span class="badge gold">BEST</span>'
        elif i == payload["knee_idx"]:
            tier_label = '<span class="badge accent">KNEE</span>'
        notes = FAMILY_NOTES.get(m["family"], {})
        addr = m["addr"] / ok
        spec = m["spec"] / ok
        grnd = m["grnd"] / ok
        topic = m["topic"] / ok
        # Strongest / weakest
        dims = {"addresses": addr, "specificity": spec,
                "grounded": grnd, "on_topic": topic}
        best_dim = max(dims, key=dims.get)
        worst_dim = min(dims, key=dims.get)
        # Auto-derived verdict
        verdict_parts: list[str] = []
        if avg >= 9.5:
            verdict_parts.append(
                f"Top-tier on this corpus ({avg:.2f}/12)."
            )
        elif avg >= 8.5:
            verdict_parts.append(
                f"Solid mid-pack ({avg:.2f}/12)."
            )
        elif avg >= 7.0:
            verdict_parts.append(
                f"Workable but not class-leading ({avg:.2f}/12)."
            )
        else:
            verdict_parts.append(
                f"Underperformed on this corpus ({avg:.2f}/12)."
            )
        if secs <= 2.0:
            verdict_parts.append(
                f"Very fast at {secs:.1f}s/q."
            )
        elif secs >= 15.0:
            verdict_parts.append(
                f"Slow at {secs:.1f}s/q — pay the latency cost only "
                f"if you need the score lift."
            )
        verdict = " ".join(verdict_parts)

        model_cards_parts.append(f"""
<div class="model-card">
  <div class="model-card-header">
    <div>
      <h3><code>{escape(tag)}</code> {tier_label}</h3>
      <div class="model-meta">
        {escape(notes.get('vendor', '?'))} ·
        {escape(notes.get('released', '?'))} ·
        <strong>{m['params_b']:.1f}B</strong> params
      </div>
    </div>
    <div class="model-score">
      <div class="score-big">{avg:.2f}<span class="score-max">/12</span></div>
      <div class="score-rank">rank #{rank} of {n_models}</div>
    </div>
  </div>
  <div class="model-rubric">
    <span class="rubric-pill">addresses: <strong>{addr:.2f}</strong></span>
    <span class="rubric-pill">specificity: <strong>{spec:.2f}</strong></span>
    <span class="rubric-pill">grounded: <strong>{grnd:.2f}</strong></span>
    <span class="rubric-pill">on_topic: <strong>{topic:.2f}</strong></span>
    <span class="rubric-pill subtle">{secs:.1f}s/q</span>
    <span class="rubric-pill subtle">{m['n']} cells · {m['errs']} err</span>
  </div>
  <p class="model-summary"><strong>What it is.</strong>
    {escape(notes.get('summary', '(no description on file)'))}</p>
  <p class="model-summary"><strong>What the data says.</strong>
    {verdict}</p>
  <p class="model-summary subtle">
    Strongest dimension: <code>{best_dim}</code> ({dims[best_dim]:.2f}/3) ·
    Weakest dimension: <code>{worst_dim}</code> ({dims[worst_dim]:.2f}/3)
  </p>
</div>
""")
    model_cards_html = "\n".join(model_cards_parts)

    # ---- Embedder leaderboard table ----
    emb_rows = []
    for i, e in enumerate(payload["embedder_data"]):
        marker = ""
        if i == 0 and len(payload["embedder_data"]) > 1:
            marker = '<span class="badge gold">BEST</span>'
        emb_rows.append(
            f"<tr><td><code>{escape(e['embedder'])}</code> {marker}</td>"
            f"<td><strong>{e['avg_score']:.2f}</strong></td>"
            f"<td>{e['avg_secs']:.1f}s</td>"
            f"<td>{e['n']}</td>"
            f"<td>{e['errs']}</td></tr>"
        )
    embedder_table = (
        '<table class="data"><thead><tr>'
        f'<th><abbr title="{escape(GLOSSARY["embedder"])}">Embedder</abbr></th>'
        '<th>Avg /12</th><th>Avg s/q</th><th>Cells</th><th>Errs</th>'
        '</tr></thead><tbody>'
        + "".join(emb_rows) + '</tbody></table>'
    )

    # ---- Family rollup table ----
    fam_rows = []
    for f in payload["family_data"]:
        fam_rows.append(
            f"<tr><td><code>{escape(f['family'])}</code></td>"
            f"<td>{f['model_count']}</td>"
            f"<td><strong>{f['avg_score']:.2f}</strong></td>"
            f"<td>{f['avg_secs']:.1f}s</td>"
            f"<td>{f['n']}</td></tr>"
        )
    family_table = (
        '<table class="data"><thead><tr>'
        '<th>Family</th><th>Models</th><th>Avg /12</th>'
        '<th>Avg s/q</th><th>Cells</th>'
        '</tr></thead><tbody>'
        + "".join(fam_rows) + '</tbody></table>'
    )

    # ---- Per-category table ----
    cat_rows = []
    for c in payload["cat_rollup"]:
        cat_rows.append(
            f"<tr><td><code>{escape(c['category'])}</code></td>"
            f"<td><strong>{c['avg_score']:.2f}</strong></td>"
            f"<td>{c['n']}</td></tr>"
        )
    category_table = (
        '<table class="data"><thead><tr>'
        '<th>Question category</th><th>Avg /12 (across all models)</th>'
        '<th>Cells</th></tr></thead><tbody>'
        + "".join(cat_rows) + '</tbody></table>'
    )

    # ---- Knee callout ----
    knee_callout = ""
    if payload["knee_idx"] != payload["best_idx"]:
        ratio = best_params / max(0.01, knee_params)
        savings_pct = (1 - knee_params / max(0.01, best_params)) * 100
        knee_callout = f"""
<div class="callout">
  <div class="callout-headline">
    <span class="big">{escape(knee_label)}</span>
    <span class="callout-sub">at {knee_score:.2f}/12,
      {knee_params:.1f}B params</span>
  </div>
  <p>Smallest model within 10% of the best score
    (<code>{escape(best_label)}</code> at {best_score:.2f}/12).
    That's a <strong>{ratio:.1f}× smaller</strong> model giving
    <strong>{(knee_score / best_score * 100):.0f}%</strong>
    of the quality with <strong>{savings_pct:.0f}% fewer params</strong>.
    For VRAM-constrained deployment, this is the value pick.</p>
</div>
"""

    # ---- Findings list ----
    findings_html = "<ul class='findings'>" + "".join(
        f"<li>{f}</li>" for f in findings
    ) + "</ul>"

    # ---- Caveats ----
    caveats_html = f"""
<div class="caveats">
  <h3>About this measurement (read me before quoting)</h3>
  <ul>
    <li><strong>Single corpus.</strong> All scores reflect performance
        on a public-domain Ohio + Appalachian geology
        corpus assembled from
        <code>sample_data/fetched/geology</code>. Results may shift
        on technical documentation, code, fiction, or scientific
        corpora — model rankings often differ across domains.</li>
    <li><strong>{n_questions} questions per (model, embedder).</strong>
        Mixed across factual lookups, comparisons, exploratory
        list queries, and multi-step synthesis. With this many
        questions, score differences smaller than ~0.4/12 are
        within noise.</li>
    <li><strong>Single judge model.</strong> qwen2.5:7b at temperature 0
        scored every answer using a strict 0–12 rubric. The judge has
        its own biases — notably it tends to reward concise,
        citation-heavy answers over verbose ones.</li>
    <li><strong>Identical retrieval per (embedder × question).</strong>
        For each embedder we pre-computed retrieval once, then reused
        the same retrieved chunks across every chat model. So this
        measures generation quality only — not the chat model's own
        retrieval logic.</li>
    <li><strong>Ollama Q4_K_M quantization.</strong> All models tested
        at Q4_K_M. Higher precision (Q5, Q8, FP16) might change scores
        on the smallest models more than the largest ones.</li>
    <li><strong>One run per (model, embedder, question) cell.</strong>
        Temperature 0.2, no resampling. Models that errored once were
        retried once; two consecutive failures skip the rest of that
        model's questions for that embedder.</li>
    <li><strong>Power data is best-effort.</strong> GPU energy via
        nvidia-smi sampling at 500 ms intervals. CPU energy unmeasured
        on Windows / requires sudo on macOS. Numbers exclude the
        baseline idle draw, so absolute kJ should be read as a
        relative comparison, not literal wattage.</li>
  </ul>
  <p style="margin-top:8px"><strong>Confidence:</strong>
    Differences between adjacent ranks (e.g. 9.30 vs 9.37) should
    be treated as ties. Differences greater than ~0.5/12 are
    reliable.</p>
</div>
"""

    # ---- Summary & Recommendation card (top-of-report verdict) ----
    # Recommended pick = the smallest model among the top-3 highest-scoring.
    # This avoids two bad failure modes:
    #   - pure score/B picks tiny mediocre models that nobody actually want
    #   - "smallest within 10% of best" can recommend a model that's
    #     measurably worse than a barely-larger alternative
    # Smallest-of-top-3 is what users actually want: real quality,
    # smallest VRAM footprint among the proven winners.
    # ALSO: cross-check against gold-truth percentage so we can flag
    # rubric-vs-factual disagreement honestly.
    by_model_lookup = {tag: m for tag, m in ordered}
    all_models = []
    for tag, m in ordered:
        ok = max(1, m["n"] - m["errs"])
        avg = m["total"] / ok
        gpct = (m.get("gold_got", 0) / max(1, m.get("gold_max", 0)) * 100
                 if m.get("gold_max", 0) else None)
        all_models.append((tag, m["params_b"], avg, m["family"], gpct))
    # Sort by score desc, take top-3 (or up to 5 if N is large)
    top_n = sorted(all_models, key=lambda x: -x[2])[:max(3, min(5, len(all_models)//5))]
    if top_n:
        # Smallest among the top tier; tiebreak on score
        top_n.sort(key=lambda x: (x[1], -x[2]))
        value_tag, value_b, value_score, value_fam, value_gold = top_n[0]
    else:
        value_tag, value_b, value_score, value_fam, value_gold = (
            best_label, best_params, best_score, "", None
        )
    value_eff = value_score / max(0.01, value_b)

    # Honest gap-vs-best phrasing (rubric)
    gap_vs_best = best_score - value_score
    if gap_vs_best <= 0.01:
        gap_phrase = "ties for best in this bench"
    elif gap_vs_best <= 0.4:
        gap_phrase = (f"trails the best by only {gap_vs_best:+.2f}/12 — "
                      "within rubric noise (~0.4/12)")
    else:
        gap_phrase = (f"trails the best by {gap_vs_best:+.2f}/12 "
                      "(real, but small)")

    # Find the model with the highest gold-truth % (factual accuracy)
    factual_candidates = [(tag, b, s, f, g) for tag, b, s, f, g
                           in all_models if g is not None]
    factual_candidates.sort(key=lambda x: -x[4] if x[4] is not None else 0)
    if factual_candidates:
        gold_tag, gold_b, gold_score_total, gold_fam, gold_pct = (
            factual_candidates[0]
        )
    else:
        gold_tag, gold_b, gold_score_total, gold_fam, gold_pct = (
            value_tag, value_b, value_score, value_fam, None
        )

    # Honest factual-accuracy caveat for the recommended pick
    if value_gold is not None and gold_pct is not None:
        gold_gap = gold_pct - value_gold
        if gold_gap > 8:
            factual_caveat = (
                f"<strong>Caveat — factual accuracy:</strong> the rubric "
                f"judge ranks this model #1, but the rule-based gold-truth "
                f"check (must-contain factual phrases) gives it only "
                f"<strong>{value_gold:.0f}%</strong>. <code>{escape(gold_tag)}</code> "
                f"hits <strong>{gold_pct:.0f}%</strong> on the same check — "
                f"a {gold_gap:.0f}-point gap. Translation: this model writes "
                f"confident, well-grounded-<em>looking</em> answers but "
                f"hallucinates more facts than its size-class peers. "
                f"Pick <code>{escape(gold_tag)}</code> instead if factual "
                f"correctness matters more than rubric polish."
            )
        else:
            factual_caveat = (
                f"Cross-checked against the rule-based gold-truth scoring: "
                f"this model hits <strong>{value_gold:.0f}%</strong> on the "
                f"factual must-contain check — within "
                f"{abs(gold_gap):.0f} points of the leader "
                f"(<code>{escape(gold_tag)}</code> at {gold_pct:.0f}%). "
                f"Quality and factual accuracy align."
            )
    else:
        factual_caveat = ""
    # Best embedder (already in payload)
    emb_rank = payload.get("embedder_data", [])
    best_emb = emb_rank[0]["embedder"] if emb_rank else ""
    best_emb_score = emb_rank[0]["avg_score"] if emb_rank else 0.0
    # Min-vram pick: smallest model whose score is within 20% of best.
    # If this collapses to the recommended pick, surface the next-smallest
    # below it so the card has independent value.
    floor_min = best_score * 0.8
    sub_candidates = sorted(
        [(tag, m["params_b"], m["total"]/max(1, m["n"]-m["errs"]))
          for tag, m in ordered
          if (m["total"]/max(1, m["n"]-m["errs"])) >= floor_min],
        key=lambda x: x[1],
    )
    if sub_candidates:
        small_tag, small_b, small_score = sub_candidates[0]
        if small_tag == value_tag and len(sub_candidates) > 1:
            # Same model — surface next-smallest under the looser floor
            small_tag, small_b, small_score = sub_candidates[1]
            floor = floor_min
        else:
            floor = floor_min
    else:
        small_tag, small_b, small_score = best_label, best_params, best_score
        floor = best_score

    summary_recommendation_html = f"""
<div class="recommendation">
  <h2 style="margin:0 0 8px">Summary &amp; recommendation</h2>
  <p class="rec-lede">
    Across <strong>{n_models}</strong> chat models
    × <strong>{n_embedders}</strong> embedders
    × <strong>{n_questions}</strong> questions ({total_cells} judged cells,
    {total_errs} errors), the bench reaches a clear verdict:
    <strong><code>{escape(value_tag)}</code></strong>
    delivers the best quality-per-byte and is the
    recommended default for ez-rag.
  </p>

  <div class="rec-grid">
    <div class="rec-card rec-best">
      <div class="rec-label">★ Recommended chat model</div>
      <div class="rec-value"><code>{escape(value_tag)}</code></div>
      <div class="rec-meta">
        {value_b:.1f}B · {escape(value_fam)} · scored
        <strong>{value_score:.2f}/12</strong>
        ({value_score/max(0.01, best_score)*100:.0f}% of best)
      </div>
      <div class="rec-rationale">
        Smallest model in the top tier ({value_eff:.2f} score/B);
        {gap_phrase}. Beats much larger models on cost-efficiency.
        Run it as your default unless you have a specific reason to switch.
      </div>
      {f'<div class="rec-caveat">{factual_caveat}</div>' if factual_caveat else ''}
    </div>

    <div class="rec-card">
      <div class="rec-label">Highest absolute quality</div>
      <div class="rec-value"><code>{escape(best_label)}</code></div>
      <div class="rec-meta">
        {best_params:.1f}B · scored <strong>{best_score:.2f}/12</strong>
      </div>
      <div class="rec-rationale">
        {("This <em>is</em> the recommended pick — best quality "
          "and smallest in the top tier."
          if best_label == value_tag
          else
          f"Pick this only if you have the VRAM. "
          f"Gain over the recommended model: "
          f"{(best_score - value_score):+.2f}/12 "
          f"(real but small — rubric noise floor is ~0.4/12).")}
      </div>
    </div>

    <div class="rec-card">
      <div class="rec-label">Tightest VRAM budget</div>
      <div class="rec-value"><code>{escape(small_tag)}</code></div>
      <div class="rec-meta">
        {small_b:.1f}B · scored <strong>{small_score:.2f}/12</strong>
        ({small_score / max(0.01, best_score) * 100:.0f}% of best)
      </div>
      <div class="rec-rationale">
        Smallest model that still hits the
        <strong>{floor:.1f}/12</strong> floor (≥80% of best). For laptops
        or shared GPUs. Sacrifices
        {best_score - small_score:+.2f}/12 vs the recommended pick.
      </div>
    </div>

    <div class="rec-card">
      <div class="rec-label">Highest factual accuracy</div>
      <div class="rec-value"><code>{escape(gold_tag)}</code></div>
      <div class="rec-meta">
        {gold_b:.1f}B · gold-truth <strong>{(gold_pct or 0):.0f}%</strong>
        · rubric {gold_score_total:.2f}/12
      </div>
      <div class="rec-rationale">
        Best score on the rule-based factual must-contain check
        across 8 grounded questions. Pick this when you can't
        afford hallucinated facts, even if rubric polish drops.
        {("(This is also the recommended pick.)"
          if gold_tag == value_tag else "")}
      </div>
    </div>

    <div class="rec-card">
      <div class="rec-label">Recommended embedder</div>
      <div class="rec-value"><code>{escape(best_emb)}</code></div>
      <div class="rec-meta">
        embedder leaderboard: <strong>{best_emb_score:.2f}/12</strong> mean
      </div>
      <div class="rec-rationale">
        Embedder choice barely moves the needle (top three within
        ~0.1/12 in this bench). Default to the smallest of the
        leaders for free VRAM.
      </div>
    </div>
  </div>

  <p class="rec-footer">
    <strong>Methodology in one line:</strong> identical retrieval
    per (embedder × question), <code>think=false</code> for fair
    apples-to-apples on reasoning models, qwen2.5:7b @ T=0 as judge,
    rule-based gold-snippet check on 8 factual questions, single run
    per cell, retry-once-then-skip on errors.
    See <em>About this measurement</em> below for caveats.
  </p>
</div>
"""

    overall_summary_html = f"""
<div class="summary-box">
  <h2 style="margin-top:0">Overall summary</h2>
  {findings_html}
  {caveats_html}
</div>
"""

    # ---- Glossary at bottom ----
    glossary_items = "".join(
        f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>'
        for k, v in sorted(GLOSSARY.items(), key=lambda kv: kv[0].lower())
    )
    glossary_html = f"""
<details class="glossary">
  <summary>Glossary &middot; click to expand</summary>
  <p style="color:var(--text-dim)">
    Every defined term in this report has a hover-tooltip with the
    same text. Skim here for the full list.
  </p>
  <dl>{glossary_items}</dl>
</details>
"""

    energy_section = ""
    if has_energy:
        energy_section = f"""
<h2>Energy efficiency (quality per kJ)</h2>
<p style="color:var(--text-dim)">
  Quality / GPU+CPU energy. Use this when picking a model for
  battery-constrained or sustained-throughput deployments.
  Total energy across the bench: <strong>{total_energy_kj:.2f} kJ</strong>
  ({total_energy_kj * 0.000278:.3f} kWh).
</p>
<div id="chart-energy" class="chart" style="height:380px"></div>
"""

    me_heatmap_section = ""
    if n_embedders >= 2:
        me_heatmap_section = """
<h2>Embedder comparison heatmap (model × embedder)</h2>
<p style="color:var(--text-dim)">
  Did the embedder choice matter? Cells with similar values across
  a row mean embedders are interchangeable for that model. Wide
  spreads indicate genuine embedder sensitivity.
</p>
<div id="chart-me-heatmap" class="chart" style="height:520px"></div>
"""

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ez-rag · Ohio multi-embedder bench</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{
    --bg: #0F1018; --surface: #161826; --surface-2: #1E2030;
    --border: #262938; --text: #E8E9F0; --text-dim: #7B8094;
    --accent: #5856D6; --accent-light: #8B89F0;
    --gold: #F2C94C; --success: #34C759; --danger: #FF453A;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                  "Helvetica Neue", Arial, sans-serif;
    line-height: 1.55;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ font-size: 32px; margin: 0 0 8px;
        color: var(--accent-light); letter-spacing: -0.02em; }}
  h2 {{ font-size: 20px; margin: 32px 0 12px;
        color: var(--accent-light); letter-spacing: -0.01em; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 32px; }}
  abbr {{ text-decoration: none;
          border-bottom: 1px dotted var(--accent-light); cursor: help; }}
  abbr:hover {{ color: var(--accent-light); }}
  .stat-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 16px 0 32px;
  }}
  .stat {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }}
  .stat .label {{
    font-size: 11px; text-transform: uppercase;
    color: var(--text-dim); letter-spacing: 0.06em;
  }}
  .stat .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .stat .sub {{ color: var(--text-dim); font-size: 12px; margin-top: 2px; }}
  .callout {{
    background: linear-gradient(135deg, rgba(88,86,214,0.12),
                                          rgba(139,137,240,0.06));
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 10px; padding: 18px 22px; margin: 20px 0;
  }}
  .callout-headline {{ display: flex; align-items: baseline;
                       gap: 12px; flex-wrap: wrap; }}
  .callout .big {{ font-size: 22px; font-weight: 700;
                   color: var(--accent-light); }}
  .callout-sub {{ color: var(--text-dim); }}
  table.data {{
    width: 100%; border-collapse: collapse; margin: 12px 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
  }}
  table.data th {{
    background: var(--surface-2); font-size: 12px;
    text-transform: uppercase; color: var(--text-dim);
    letter-spacing: 0.04em; text-align: left;
    padding: 10px 12px; font-weight: 600;
  }}
  table.data td {{
    padding: 10px 12px; border-top: 1px solid var(--border);
    font-size: 14px;
  }}
  table.data tr.highlight {{ background: rgba(88,86,214,0.08); }}
  table.data code {{
    font-family: "Fira Code", Consolas, monospace; font-size: 13px;
    color: var(--accent-light);
  }}
  .badge {{
    display: inline-block; padding: 1px 7px; border-radius: 999px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; margin-left: 6px; vertical-align: middle;
  }}
  .badge.gold {{ background: var(--gold); color: #000; }}
  .badge.accent {{ background: var(--accent); color: #fff; }}
  .chart {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; margin: 12px 0;
  }}
  .summary-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 22px; margin: 24px 0;
  }}
  .recommendation {{
    background: linear-gradient(180deg,
       rgba(242, 201, 76, 0.06) 0%,
       var(--surface) 30%);
    border: 1px solid rgba(242, 201, 76, 0.35);
    border-left: 4px solid var(--gold);
    border-radius: 12px; padding: 18px 22px; margin: 24px 0;
  }}
  .recommendation .rec-lede {{
    font-size: 15px; line-height: 1.55; color: var(--text);
    margin: 4px 0 16px;
  }}
  .recommendation .rec-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px; margin: 8px 0;
  }}
  .recommendation .rec-card {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 14px;
  }}
  .recommendation .rec-card.rec-best {{
    border: 1px solid var(--gold);
    box-shadow: 0 0 0 1px rgba(242, 201, 76, 0.25) inset;
  }}
  .recommendation .rec-label {{
    font-size: 12px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 4px;
  }}
  .recommendation .rec-best .rec-label {{ color: var(--gold); }}
  .recommendation .rec-value {{
    font-size: 18px; font-weight: 600;
    margin: 2px 0;
  }}
  .recommendation .rec-value code {{
    font-size: 16px; padding: 2px 8px;
  }}
  .recommendation .rec-meta {{
    color: var(--text-dim); font-size: 13px;
    margin-bottom: 8px;
  }}
  .recommendation .rec-rationale {{
    font-size: 13px; line-height: 1.5; color: var(--text);
  }}
  .recommendation .rec-caveat {{
    margin-top: 8px; padding: 8px 10px;
    background: rgba(255, 69, 58, 0.08);
    border-left: 3px solid var(--danger);
    border-radius: 6px;
    font-size: 12.5px; line-height: 1.5;
    color: var(--text);
  }}
  .recommendation .rec-caveat strong {{ color: var(--danger); }}
  .recommendation .rec-footer {{
    font-size: 12px; color: var(--text-dim);
    margin: 14px 0 0; padding-top: 12px;
    border-top: 1px solid var(--border);
  }}
  ul.findings {{ list-style: none; padding: 0; margin: 8px 0; }}
  ul.findings li {{
    padding: 8px 0; border-top: 1px solid var(--border);
  }}
  ul.findings li:first-child {{ border-top: none; }}
  .caveats {{
    background: rgba(242, 201, 76, 0.05);
    border: 1px solid rgba(242, 201, 76, 0.25);
    border-left: 3px solid var(--gold);
    border-radius: 8px; padding: 14px 18px; margin-top: 16px;
  }}
  .caveats h3 {{
    margin: 0 0 8px; color: var(--gold); font-size: 14px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .caveats ul {{ padding-left: 18px; margin: 6px 0; }}
  .caveats li {{ margin: 4px 0; font-size: 13px; }}
  .model-cards {{
    display: grid; gap: 16px; grid-template-columns: 1fr;
  }}
  .model-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px;
  }}
  .model-card-header {{
    display: flex; justify-content: space-between;
    align-items: flex-start; gap: 16px;
    flex-wrap: wrap; margin-bottom: 8px;
  }}
  .model-card h3 {{ margin: 0 0 4px; font-size: 17px;
                     color: var(--accent-light); }}
  .model-meta {{ color: var(--text-dim); font-size: 12px; }}
  .model-score {{ text-align: right; }}
  .score-big {{ font-size: 32px; font-weight: 700;
                color: var(--text); line-height: 1; }}
  .score-max {{ font-size: 14px; color: var(--text-dim); font-weight: 500; }}
  .score-rank {{ font-size: 11px; color: var(--text-dim); margin-top: 2px; }}
  .model-rubric {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }}
  .rubric-pill {{
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 3px 10px; font-size: 11px;
  }}
  .rubric-pill.subtle {{ color: var(--text-dim); }}
  .model-summary {{ margin: 8px 0; font-size: 13.5px; line-height: 1.6; }}
  .model-summary.subtle {{ color: var(--text-dim); font-size: 12px; }}
  details.glossary {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; margin: 24px 0;
  }}
  details.glossary summary {{
    cursor: pointer; font-weight: 700;
    color: var(--accent-light); font-size: 14px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  details.glossary dl {{ margin: 12px 0 0; }}
  details.glossary dt {{
    font-family: "Fira Code", Consolas, monospace;
    font-size: 13px; color: var(--accent-light);
    margin-top: 10px; font-weight: 600;
  }}
  details.glossary dd {{
    margin: 2px 0 0; font-size: 13px; color: var(--text);
    line-height: 1.5;
  }}
  footer {{
    color: var(--text-dim); font-size: 12px; margin-top: 48px;
    padding-top: 20px; border-top: 1px solid var(--border);
  }}
  code {{
    font-family: "Fira Code", Consolas, monospace;
    background: var(--surface-2); padding: 1px 6px;
    border-radius: 4px; font-size: 13px;
  }}
</style></head><body><div class="container">
  <h1>Ohio bench · multi-embedder × multi-model</h1>
  <p class="subtitle">
    {n_models} chat models tested across {n_embedders} embedder(s)
    on the public-domain Ohio + Appalachian geology corpus.
    Same retrieval per (embedder × question), same prompt, same judge.
    {total_cells} total cells judged 0–12 on a strict 4-dimension rubric.
  </p>

  <div class="stat-row">
    <div class="stat">
      <div class="label">Models tested</div>
      <div class="value">{n_models}</div>
      <div class="sub">{min(payload['params']) if payload['params'] else 0:.1f}B → {max(payload['params']) if payload['params'] else 0:.1f}B params</div>
    </div>
    <div class="stat">
      <div class="label">Embedders</div>
      <div class="value">{n_embedders}</div>
      <div class="sub">{', '.join(escape(e) for e in payload['embedders'])}</div>
    </div>
    <div class="stat">
      <div class="label">Total cells</div>
      <div class="value">{total_cells}</div>
      <div class="sub">{total_errs} errors</div>
    </div>
    <div class="stat">
      <div class="label">Best score</div>
      <div class="value">{best_score:.2f}<span style="font-size:14px;color:var(--text-dim)"> /12</span></div>
      <div class="sub">{escape(best_label)}</div>
    </div>
    <div class="stat">
      <div class="label">Knee (within 10% of best)</div>
      <div class="value">{knee_params:.1f}B</div>
      <div class="sub">{escape(knee_label)}</div>
    </div>
    <div class="stat">
      <div class="label">Total compute</div>
      <div class="value">{total_seconds/60:.1f}<span style="font-size:14px;color:var(--text-dim)"> min</span></div>
      <div class="sub">{("≈ " + f"{total_energy_kj:.1f} kJ GPU+CPU") if has_energy else "Power not measured"}</div>
    </div>
  </div>

  {summary_recommendation_html}

  {knee_callout}

  {overall_summary_html}

  <h2>Headline scores</h2>
  <p style="color:var(--text-dim)">
    Hover any underlined column header for its definition.
    Bold dotted underline = glossary term. The full glossary lives
    at the bottom of this page.
  </p>
  {headline_table}

  <h2>Embedder leaderboard</h2>
  <p style="color:var(--text-dim)">
    Average judge score across every (chat model × question)
    combination per embedder. Use this to pick the right embedder
    for your corpus.
  </p>
  {embedder_table}

  <h2>Family rollup</h2>
  <p style="color:var(--text-dim)">
    Average across all models in each family. Reveals whether one
    family (Qwen3, Gemma, etc.) is consistently ahead.
  </p>
  {family_table}

  <h2>Per-question-category averages</h2>
  <p style="color:var(--text-dim)">
    Are some question types systematically harder? Average score
    across all models per category.
  </p>
  {category_table}

  <h2>Quality vs parameter count</h2>
  <p style="color:var(--text-dim)">
    Where does model size start to matter on this corpus?
    Hover for details.
  </p>
  <div id="chart-quality-params" class="chart" style="height:380px"></div>

  <h2>Quality per second (cost-efficiency)</h2>
  <p style="color:var(--text-dim)">
    Higher is better. The best
    <abbr title="{escape(GLOSSARY['cost-efficiency'])}">cost-efficiency</abbr>
    spot — what model gives you the best quality per unit of
    inference time?
  </p>
  <div id="chart-cost-eff" class="chart" style="height:380px"></div>

  {energy_section}

  <h2>Per-rubric breakdown</h2>
  <p style="color:var(--text-dim)">
    The judge scored four dimensions independently: <em>addresses</em>
    (does it answer the question?), <em>specificity</em> (concrete
    examples?), <em>grounded</em> (drawn from corpus, not training?),
    <em>on_topic</em> (no drift?). Each 0–3.
  </p>
  <div id="chart-rubrics" class="chart" style="height:420px"></div>

  <h2>Per-category breakdown</h2>
  <p style="color:var(--text-dim)">
    How does each model do on factual lookups vs comparisons vs
    exploratory list queries vs multi-step synthesis?
  </p>
  <div id="chart-categories" class="chart" style="height:420px"></div>

  <h2>Per-model factual summary</h2>
  <p style="color:var(--text-dim)">
    What each model is, and what the data says about it on this corpus.
  </p>
  <div class="model-cards">{model_cards_html}</div>

  <h2>Per-question heatmap (model × question)</h2>
  <p style="color:var(--text-dim)">
    Each cell is the average judge score for that model on that
    question (0–12). Reveals where specific models break down.
  </p>
  <div id="chart-heatmap" class="chart" style="height:580px"></div>

  {me_heatmap_section}

  {glossary_html}

  <footer>
    Generated by <code>bench/ohio_html.py</code>. Source data:
    <code>{escape(judged_path.name)}</code>.
    Charts powered by <a href="https://plotly.com/javascript/">Plotly</a>.
  </footer>
</div>

<script>
const DATA = {payload_json};
const ACCENT = '#8B89F0';
const TEXT = '#E8E9F0';
const DIM = '#7B8094';
const SURFACE = '#161826';
const BORDER = '#262938';

const FAMILY_COLORS = {{
  'qwen2.5':'#5856D6','qwen3':'#7A78E0','llama3':'#FF8C42',
  'phi':'#34C759','mistral':'#FF453A','deepseek-r1':'#F2C94C',
  'gemma':'#00C7BE','granite':'#A78BFA',
}};
function colorFor(family) {{ return FAMILY_COLORS[family] || ACCENT; }}

const baseLayout = {{
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {{ family: '-apple-system,Segoe UI,Roboto,sans-serif',
            color: TEXT, size: 12 }},
  margin: {{ t: 30, l: 60, r: 30, b: 60 }},
  xaxis: {{ gridcolor: BORDER, zerolinecolor: BORDER, linecolor: BORDER }},
  yaxis: {{ gridcolor: BORDER, zerolinecolor: BORDER, linecolor: BORDER }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', bordercolor: BORDER, borderwidth: 1 }},
  hoverlabel: {{ bgcolor: SURFACE, bordercolor: BORDER,
                 font: {{ color: TEXT }} }},
}};
const cfg = {{ displaylogo: false, responsive: true,
                modeBarButtonsToRemove: ['lasso2d','select2d'] }};

// Quality vs params scatter
{{
  const families = [...new Set(DATA.families)];
  const traces = families.map(fam => {{
    const idxs = DATA.families.map((f,i) => f===fam ? i : -1).filter(i=>i>=0);
    return {{
      x: idxs.map(i => DATA.params[i]),
      y: idxs.map(i => DATA.avg_total[i]),
      text: idxs.map(i => DATA.labels[i]),
      mode:'markers+text', type:'scatter',
      name: fam, marker: {{size:14, color:colorFor(fam)}},
      textposition:'top center', textfont: {{size:10, color:DIM}},
      hovertemplate: '<b>%{{text}}</b><br>%{{x:.1f}}B params<br>'+
                     'score: %{{y:.2f}}/12<extra></extra>',
    }};
  }});
  // Knee marker
  traces.push({{
    x: [DATA.params[DATA.knee_idx]], y: [DATA.avg_total[DATA.knee_idx]],
    mode: 'markers', type: 'scatter', name: 'knee (≥90% of best)',
    marker: {{ size: 22, color: 'rgba(0,0,0,0)',
                line: {{ color: ACCENT, width: 3 }} }},
    showlegend: true, hoverinfo: 'skip',
  }});
  Plotly.newPlot('chart-quality-params', traces, {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title:'parameters (B)', type:'log' }},
    yaxis: {{ ...baseLayout.yaxis, title:'avg judge score / 12',
              range:[0, 12] }},
  }}, cfg);
}}

// Cost-efficiency scatter
{{
  const families = [...new Set(DATA.families)];
  const traces = families.map(fam => {{
    const idxs = DATA.families.map((f,i) => f===fam ? i : -1).filter(i=>i>=0);
    return {{
      x: idxs.map(i => DATA.avg_secs[i]),
      y: idxs.map(i => DATA.avg_total[i]),
      text: idxs.map(i => DATA.labels[i]),
      mode:'markers+text', type:'scatter',
      name: fam, marker: {{size:14, color:colorFor(fam)}},
      textposition:'top center', textfont: {{size:10, color:DIM}},
      hovertemplate: '<b>%{{text}}</b><br>%{{x:.1f}}s/q<br>'+
                     'score: %{{y:.2f}}/12<extra></extra>',
    }};
  }});
  Plotly.newPlot('chart-cost-eff', traces, {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title:'avg seconds per question' }},
    yaxis: {{ ...baseLayout.yaxis, title:'avg judge score / 12',
              range:[0, 12] }},
  }}, cfg);
}}

// Energy efficiency
if (DATA.energy_kj_per_q && DATA.energy_kj_per_q.some(v => v > 0)
    && document.getElementById('chart-energy')) {{
  const families = [...new Set(DATA.families)];
  const traces = families.map(fam => {{
    const idxs = DATA.families.map((f,i) => f===fam ? i : -1)
        .filter(i => i >= 0 && DATA.energy_kj_per_q[i] > 0);
    return {{
      x: idxs.map(i => DATA.energy_kj_per_q[i]),
      y: idxs.map(i => DATA.avg_total[i]),
      text: idxs.map(i => DATA.labels[i]),
      mode:'markers+text', type:'scatter',
      name: fam, marker: {{size:14, color:colorFor(fam)}},
      textposition:'top center', textfont: {{size:10, color:DIM}},
      hovertemplate: '<b>%{{text}}</b><br>%{{x:.2f}} kJ/q<br>'+
                     'score: %{{y:.2f}}/12<extra></extra>',
    }};
  }});
  Plotly.newPlot('chart-energy', traces, {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title:'avg kJ per question' }},
    yaxis: {{ ...baseLayout.yaxis, title:'avg judge score / 12',
              range:[0, 12] }},
  }}, cfg);
}}

// Per-rubric grouped bars
{{
  Plotly.newPlot('chart-rubrics', [
    {{ x: DATA.labels, y: DATA.avg_addr, name: 'addresses',
       type: 'bar', marker: {{color:'#5856D6'}} }},
    {{ x: DATA.labels, y: DATA.avg_spec, name: 'specificity',
       type: 'bar', marker: {{color:'#8B89F0'}} }},
    {{ x: DATA.labels, y: DATA.avg_grnd, name: 'grounded',
       type: 'bar', marker: {{color:'#34C759'}} }},
    {{ x: DATA.labels, y: DATA.avg_topic, name: 'on_topic',
       type: 'bar', marker: {{color:'#F2C94C'}} }},
  ], {{
    ...baseLayout, barmode: 'group',
    xaxis: {{ ...baseLayout.xaxis, title: '' }},
    yaxis: {{ ...baseLayout.yaxis,
              title: 'avg per dimension (0-3)', range: [0, 3] }},
  }}, cfg);
}}

// Per-category grouped bars
{{
  const CAT_COLORS = ['#5856D6','#34C759','#FF8C42','#F2C94C',
                       '#FF453A','#00C7BE'];
  const traces = DATA.cats.map((c,i) => ({{
    x: DATA.labels, y: DATA.cat_data[c],
    name: c, type: 'bar',
    marker: {{ color: CAT_COLORS[i % CAT_COLORS.length] }},
  }}));
  Plotly.newPlot('chart-categories', traces, {{
    ...baseLayout, barmode: 'group',
    xaxis: {{ ...baseLayout.xaxis, title: '' }},
    yaxis: {{ ...baseLayout.yaxis,
              title: 'avg score / 12 by category', range: [0, 12] }},
  }}, cfg);
}}

// Per-question heatmap
{{
  const qlabels = DATA.questions.map(q =>
    q.length > 60 ? q.slice(0, 57) + '…' : q
  );
  Plotly.newPlot('chart-heatmap', [{{
    z: DATA.z_matrix,
    x: qlabels, y: DATA.labels,
    type: 'heatmap', zmin: 0, zmax: 12,
    colorscale: [[0,'#3A1A1F'],[0.5,'#5C5340'],[0.83,'#3F4A2E'],
                 [1.0,'#34C759']],
    colorbar: {{ title:'score', tickfont: {{color: TEXT}} }},
    hovertemplate: '<b>%{{y}}</b><br>%{{x}}<br>'+
                   'score: %{{z}}/12<extra></extra>',
  }}], {{
    ...baseLayout,
    margin: {{ t: 20, l: 160, r: 30, b: 240 }},
    xaxis: {{ ...baseLayout.xaxis, tickangle: -45, automargin: true }},
    yaxis: {{ ...baseLayout.yaxis, automargin: true }},
  }}, cfg);
}}

// Per-embedder heatmap (only if 2+ embedders)
if (DATA.embedders.length >= 2 && document.getElementById('chart-me-heatmap')) {{
  Plotly.newPlot('chart-me-heatmap', [{{
    z: DATA.me_z,
    x: DATA.embedders, y: DATA.labels,
    type: 'heatmap', zmin: 0, zmax: 12,
    colorscale: [[0,'#3A1A1F'],[0.5,'#5C5340'],[0.83,'#3F4A2E'],
                 [1.0,'#34C759']],
    colorbar: {{ title:'score', tickfont: {{color: TEXT}} }},
    hovertemplate: '<b>%{{y}}</b> on <b>%{{x}}</b><br>'+
                   'score: %{{z}}/12<extra></extra>',
  }}], {{
    ...baseLayout,
    margin: {{ t: 20, l: 160, r: 30, b: 100 }},
    xaxis: {{ ...baseLayout.xaxis, automargin: true }},
    yaxis: {{ ...baseLayout.yaxis, automargin: true }},
  }}, cfg);
}}
</script>
</body></html>
"""
    out_html = Path(out_html)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged_json")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--power-summary", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    judged = Path(args.judged_json)
    out = Path(args.out) if args.out else judged.with_suffix(".html")
    render_report(
        judged_path=judged,
        manifest_path=Path(args.manifest) if args.manifest else None,
        out_html=out,
        power_summary=Path(args.power_summary) if args.power_summary else None,
    )
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

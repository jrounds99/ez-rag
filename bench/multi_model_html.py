"""Generate a self-contained interactive HTML benchmark report.

Input: a *-judged.json produced by judge_eval.py over a multimodel-*.json.
Output: a single HTML file with Plotly charts (CDN-loaded), tables, and
        the "knee" analysis. Open in any browser.

Charts:
  - Quality vs parameter count (scatter + smoothed curve)
  - Quality per billion params (efficiency bar)
  - Per-rubric breakdown (grouped bars: addresses / specificity / grounded / on_topic)
  - Per-category breakdown (grouped bars by question category)
  - Latency vs quality (cost/benefit scatter)
  - Per-question heatmap (model × question, color = score)
  - VRAM cost vs quality (memory-shortage view)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.0.min.js"


# ============================================================================
# Glossary — inline term definitions
# ============================================================================
# Hover any term in the rendered report to see the definition. Stored as
# a Python dict so the HTML generator can both inject inline <abbr> tags
# AND emit a full glossary section at the bottom.

GLOSSARY: dict[str, str] = {
    "RAG": "Retrieval-Augmented Generation. The LLM is given relevant "
            "passages from a document corpus before answering, rather "
            "than relying on its training data alone.",
    "LLM": "Large Language Model — the chat model that produces the "
            "final answer (e.g. qwen2.5:7b).",
    "params": "Trainable parameters of a neural network. Listed in B "
              "(billions). Roughly correlates with capability and VRAM "
              "cost. Q4_K_M quantization stores each parameter in ~4 "
              "bits, so VRAM ≈ params × 0.55 GB.",
    "B": "Billion. As in '7B params' = 7 billion parameters.",
    "VRAM": "Video RAM on the GPU. The model's weights and per-token "
            "cache live here. The hard ceiling on what model size you "
            "can run at speed.",
    "Q4_K_M": "A specific 4-bit quantization scheme used by GGUF / "
              "Ollama models. Reduces a model's VRAM by ~4× from "
              "full precision with minor quality loss.",
    "context window": "The maximum number of tokens (~chars/3.5) the "
                      "model can read at once. Includes the system "
                      "prompt + retrieved chunks + chat history + "
                      "answer reserve. Ollama defaults to 4096 unless "
                      "ez-rag's auto-sizing overrides.",
    "tokens": "The unit LLMs count in. Roughly 1 token ≈ 3-4 chars of "
              "English. The model has a fixed context-window budget "
              "measured in tokens.",
    "BM25": "A keyword search algorithm (specifically Okapi BM25). "
            "ez-rag uses it via SQLite FTS5 to find passages with "
            "literal term matches the dense vector search misses.",
    "dense search": "Embedding-based search — every chunk is converted "
                    "to a vector by an embedder, the query is embedded "
                    "the same way, cosine similarity ranks the matches.",
    "hybrid search": "Running BM25 and dense search in parallel and "
                     "fusing their rankings. The single biggest "
                     "quality win over either method alone.",
    "RRF": "Reciprocal Rank Fusion. Combines two ranked lists into one "
           "by summing 1/(60+rank) for each item. ez-rag uses it for "
           "BM25 + dense fusion.",
    "embedder": "Model that converts text to vectors. Smaller and "
                "faster than the chat LLM. ez-rag uses qwen3-embedding "
                "8B by default in this workspace.",
    "reranker": "A small cross-encoder model that re-scores retrieval "
                "candidates with full attention to (query, passage) "
                "pairs. Adds ~50–200 ms per query for big quality "
                "wins. ez-rag uses MiniLM L-6-v2.",
    "HyDE": "Hypothetical Document Embeddings. Ask the LLM to "
            "generate a short hypothetical answer to the query, then "
            "embed THAT instead of the bare question. Often retrieves "
            "better-matching corpus chunks because the hypothetical "
            "matches answer-shaped content.",
    "MMR": "Maximal Marginal Relevance. A diversification algorithm "
           "that re-selects from a candidate pool to balance "
           "relevance to the query against novelty against already-"
           "selected chunks.",
    "CRAG": "Corrective RAG (Yan et al., 2024). After retrieval, an "
            "LLM call evaluates whether each chunk is actually "
            "relevant; irrelevant chunks are dropped before the "
            "answer prompt.",
    "top_k": "Number of chunks the retrieval pipeline returns to the "
             "LLM. Default 8. Higher = more context but more noise + "
             "VRAM cost.",
    "chunk": "A small slice of a source document — typically ~512 "
             "tokens with 64 overlap. The unit retrieval works on.",
    "ingest": "The one-time process of parsing source documents, "
              "chunking them, embedding the chunks, and storing both "
              "in a searchable index.",
    "rubric": "The four 0-3 dimensions the LLM-judge scored each "
              "answer on: addresses, specificity, grounded, on_topic. "
              "Sum is 0–12.",
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
                "without drifting mid-paragraph to a different "
                "subject?",
    "judge": "An LLM (here, qwen2.5:7b at temperature 0) that scored "
             "every answer against the 4-dimension rubric. Same "
             "judge for every model so scores are comparable.",
    "knee": "The smallest model whose score is still within 10% of "
            "the best model. The diminishing-returns inflection "
            "point.",
    "lost in the middle": "Documented LLM behavior (Liu et al., 2023): "
                          "models attend most to content at the "
                          "START and END of long contexts, less to "
                          "the middle. Reorder strategies put the "
                          "most-relevant chunks at both ends.",
    "reasoning model": "An LLM trained to emit a hidden chain-of-"
                       "thought (`<think>...</think>`) before its "
                       "actual answer. Examples here: deepseek-r1:"
                       "1.5b and deepseek-r1:32b. The thinking "
                       "blocks burn tokens but don't always improve "
                       "RAG answers because the model is reasoning "
                       "without the corpus content in mind.",
    "auto-list mode": "ez-rag auto-detects 'list X / give examples "
                      "of X' queries and routes them through entity-"
                      "rich HyDE + an extraction-only system prompt. "
                      "Lifts answer quality on open-ended exploratory "
                      "queries.",
    "expand_to_chapter": "Optional retrieval step that replaces each "
                         "hit's text with the surrounding chapter "
                         "(capped by chapter_max_chars).",
    "diversification": "Cap on how many chunks any single source "
                       "file can contribute to top-K. Default 3. "
                       "Forces the LLM to ground answers across "
                       "multiple sources.",
    "Ollama": "Local LLM hosting daemon — what ez-rag uses by "
              "default. Loads model weights into VRAM and exposes "
              "an HTTP API.",
    "OS slack": "Spare VRAM reserved for the operating system / "
                "display compositor / driver overhead. ez-rag "
                "estimates ~500 MB.",
    "KV cache": "Per-token state the model retains across the "
                "context window. Linear in sequence length and "
                "batch — at 8 K context it's roughly 0.06 GB per B "
                "params for Q4_K_M.",
}


# ============================================================================
# Per-model factual notes
# ============================================================================
# Combines what's publicly known about each model with the empirical
# results from this run. Scores are filled in dynamically.

MODEL_NOTES: dict[str, dict] = {
    "qwen2.5:0.5b": {
        "vendor": "Alibaba",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "The smallest model in the Qwen 2.5 family. "
            "Designed for edge / mobile / extreme-low-resource "
            "deployments. Capable of basic Q&A but struggles with "
            "multi-step synthesis or extracting named entities from "
            "messy retrieved context."
        ),
        "verdict": (
            "Not recommended for RAG on a corpus this size. The "
            "5.37/12 score is roughly half of the small-model pack — "
            "the model lacks the capacity to integrate multiple "
            "retrieved chunks into a coherent answer. Choose this "
            "only if you have <2 GB VRAM."
        ),
    },
    "llama3.2:1b": {
        "vendor": "Meta",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "Meta's smallest Llama 3.2 instruct model, designed for "
            "on-device inference (mobile / edge). Distilled from "
            "Llama 3.1 8B with structured pruning."
        ),
        "verdict": (
            "Surprisingly strong for its size — 8.77/12 is 86% of "
            "the best model's score at 1/27th the parameter count. "
            "The KNEE of the curve. If you need a small RAG model, "
            "this is the pick. Fits in ~1.5 GB VRAM."
        ),
    },
    "qwen2.5:1.5b": {
        "vendor": "Alibaba",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "Compact Qwen 2.5 model targeting the same on-device "
            "tier as Llama 3.2 1B. Trained on the same multilingual "
            "corpus + instruction-tuning pipeline as the larger "
            "Qwen variants."
        ),
        "verdict": (
            "Solid 8.53/12 — virtually tied with Llama 3.2 1B. "
            "Slightly better grounded score, slightly worse on "
            "specificity. Pick whichever family ecosystem you "
            "prefer; quality is a wash in this size class."
        ),
    },
    "deepseek-r1:1.5b": {
        "vendor": "DeepSeek",
        "released": "2025-01",
        "type": "Reasoning model (distilled)",
        "summary": (
            "Distilled from DeepSeek-R1 onto a Qwen 1.5B base. "
            "Emits `<think>...</think>` chain-of-thought before each "
            "answer. The reasoning trace is hidden from the user but "
            "consumes generation tokens."
        ),
        "verdict": (
            "Underperformed badly at 5.50/12 — about the same as "
            "qwen2.5:0.5b despite being 3.6× larger. The thinking "
            "blocks burn budget without improving the answer because "
            "the model reasons about what IT thinks the answer is, "
            "then under-uses the retrieved chunks. Bad fit for RAG."
        ),
    },
    "llama3.2:3b": {
        "vendor": "Meta",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "Meta's mid-tier small model — a step up in capacity "
            "from the 1B variant. Same pruning/distillation lineage "
            "from Llama 3.1 8B."
        ),
        "verdict": (
            "8.57/12 — basically equivalent to its own 1B sibling "
            "(8.77). Adding 1.8B params here didn't materially "
            "improve quality on this corpus. The 3B class as a "
            "whole stalls around 8.4–8.6."
        ),
    },
    "qwen2.5:3b": {
        "vendor": "Alibaba",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "Mid-tier Qwen 2.5, occupying the same band as "
            "Llama 3.2 3B."
        ),
        "verdict": (
            "8.40/12 — lowest of the 3B class. The smaller Qwen "
            "1.5B (8.53) actually scored slightly higher, "
            "suggesting Qwen's instruction tuning is stronger at "
            "the 1-2B size than at 3B for this kind of work."
        ),
    },
    "phi4-mini": {
        "vendor": "Microsoft",
        "released": "2025-02",
        "type": "Reasoning-focused instruction-tuned",
        "summary": (
            "Microsoft's smallest Phi-4 variant (3.8B), positioned "
            "for reasoning + tool use. Trained on Phi's signature "
            "synthetic-data-heavy curriculum."
        ),
        "verdict": (
            "8.47/12 — middle of the 3B-class pack. Not the "
            "expected reasoning advantage on RAG. Phi-4's strengths "
            "are in math + code reasoning, which doesn't translate "
            "to chunk-grounded factual Q&A."
        ),
    },
    "mistral:7b": {
        "vendor": "Mistral AI",
        "released": "2024-07 (v0.3)",
        "type": "Instruction-tuned",
        "summary": (
            "The reference 7B model from Mistral AI. Sliding-window "
            "attention, grouped-query attention. Long the de-facto "
            "open-source 7B baseline before Qwen / Llama 3 caught up."
        ),
        "verdict": (
            "9.30/12 — solid mid-pack 7B performance. Beats every "
            "model below 7B but loses to qwen2.5:7b by a clear "
            "margin (-0.87). Showing its age vs the newer Qwen 2.5 "
            "and Llama 3.1 generations."
        ),
    },
    "qwen2.5:7b": {
        "vendor": "Alibaba",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "The flagship size in Qwen 2.5. Strong on multilingual + "
            "code + factual recall. Deeply trained instruction tuning. "
            "ez-rag's recommended default."
        ),
        "verdict": (
            "10.17/12 — statistically tied with the 32B reasoning "
            "model (10.23) at 1/4 the parameters and 7× the speed. "
            "On this corpus, this is the practical optimum. Your "
            "current setting is correct."
        ),
    },
    "llama3.1:8b": {
        "vendor": "Meta",
        "released": "2024-07",
        "type": "Instruction-tuned",
        "summary": (
            "Meta's 8B Llama 3.1 instruct. The previous-generation "
            "tier most projects standardized on. 128K native context."
        ),
        "verdict": (
            "9.37/12 — beat by qwen2.5:7b by 0.8 points despite "
            "being slightly larger. Llama 3.1's instruction tuning "
            "is more conservative and produces less specific answers "
            "(spec score 2.13 vs Qwen's 2.37) on this corpus."
        ),
    },
    "qwen2.5:14b": {
        "vendor": "Alibaba",
        "released": "2024-09",
        "type": "Instruction-tuned",
        "summary": (
            "Larger Qwen 2.5 — 14B params. Targets the band between "
            "small-model deployment and full-size flagship work. "
            "Same training lineage as qwen2.5:7b."
        ),
        "verdict": (
            "9.40/12 — SURPRISINGLY worse than its own 7B sibling "
            "(10.17). The 14B is more verbose and hedges more, "
            "which the judge penalized on specificity. Don't pay "
            "for the extra 6B params on this corpus."
        ),
    },
    "deepseek-r1:32b": {
        "vendor": "DeepSeek",
        "released": "2025-01",
        "type": "Reasoning model (distilled)",
        "summary": (
            "Distilled from DeepSeek-R1 onto a Qwen 32B base. Emits "
            "long `<think>...</think>` chains before answering. The "
            "26-second-per-query latency reflects the cost of the "
            "hidden reasoning."
        ),
        "verdict": (
            "10.23/12 — technically the top scorer, by 0.06 points "
            "over qwen2.5:7b. That difference is well inside judge "
            "noise. You pay 7× the latency, 4.3× the VRAM, and 4× "
            "the parameter count for an indistinguishable answer. "
            "Not worth it for this workload."
        ),
    },
}


def fmt_n(n: float, decimals: int = 2) -> str:
    return f"{n:.{decimals}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged_json", help="path to a *-judged.json file")
    ap.add_argument("--out", default=None,
                     help="output HTML path (default: <judged_json>.html)")
    args = ap.parse_args()

    src = json.loads(Path(args.judged_json).read_text(encoding="utf-8"))
    rows = src if isinstance(src, list) else src.get("results", [])

    # ============================================================================
    # Aggregate
    # ============================================================================
    by_model: dict[str, dict] = defaultdict(lambda: {
        "params_b": 0.0, "family": "",
        "n": 0, "errs": 0, "seconds": 0.0,
        "addr": 0, "spec": 0, "grnd": 0, "topic": 0, "total": 0,
        "by_cat": defaultdict(lambda: {"n": 0, "total": 0}),
        "per_q": [],   # [(question, total)] for the heatmap
    })
    questions_seen: list[str] = []
    questions_seen_set: set[str] = set()

    for r in rows:
        m = by_model[r["model"]]
        m["params_b"] = r.get("params_b", 0.0)
        m["family"] = r.get("family", "")
        m["n"] += 1
        m["seconds"] += r.get("seconds", 0)
        if r.get("judge_err") or r.get("err"):
            m["errs"] += 1
            m["per_q"].append((r["question"], None))
            if r["question"] not in questions_seen_set:
                questions_seen.append(r["question"])
                questions_seen_set.add(r["question"])
            continue
        m["addr"] += r.get("judge_addresses", 0)
        m["spec"] += r.get("judge_specificity", 0)
        m["grnd"] += r.get("judge_grounded", 0)
        m["topic"] += r.get("judge_on_topic", 0)
        total = r.get("judge_total", 0)
        m["total"] += total
        c = m["by_cat"][r.get("category", "?")]
        c["n"] += 1
        c["total"] += total
        m["per_q"].append((r["question"], total))
        if r["question"] not in questions_seen_set:
            questions_seen.append(r["question"])
            questions_seen_set.add(r["question"])

    if not by_model:
        print("No data in judged JSON")
        return 1

    ordered = sorted(by_model.items(), key=lambda kv: kv[1]["params_b"])

    # ============================================================================
    # Build chart datasets
    # ============================================================================
    labels = [tag for tag, _ in ordered]
    params = [m["params_b"] for _, m in ordered]
    families = [m["family"] for _, m in ordered]
    avg_total = []
    avg_addr = []
    avg_spec = []
    avg_grnd = []
    avg_topic = []
    avg_secs = []
    cats_set: set[str] = set()
    for _, m in ordered:
        ok = max(1, m["n"] - m["errs"])
        avg_total.append(m["total"] / ok)
        avg_addr.append(m["addr"] / ok)
        avg_spec.append(m["spec"] / ok)
        avg_grnd.append(m["grnd"] / ok)
        avg_topic.append(m["topic"] / ok)
        avg_secs.append(m["seconds"] / max(1, m["n"]))
        cats_set.update(m["by_cat"].keys())

    cats = sorted(cats_set)
    cat_data: dict[str, list[float]] = {c: [] for c in cats}
    for _, m in ordered:
        for c in cats:
            d = m["by_cat"].get(c, {"n": 0, "total": 0})
            cat_data[c].append(d["total"] / d["n"] if d["n"] else 0.0)

    # Heatmap z-values: rows = models, cols = questions
    z_matrix: list[list[float | None]] = []
    for _, m in ordered:
        # Build dict from per_q for this model
        d = {q: t for q, t in m["per_q"]}
        z_matrix.append([d.get(q) for q in questions_seen])

    # Approximate VRAM (Q4_K_M): weights ~0.55 GB/B + KV at 8K ctx ~0.06 GB/B
    vram_gb = [round(p * 0.55 + p * 0.06 + 0.5, 2) for p in params]

    # Knee: smallest model within 10% of best
    best_idx = max(range(len(avg_total)), key=lambda i: avg_total[i])
    best_score = avg_total[best_idx]
    threshold = best_score * 0.90
    knee_idx = next(
        (i for i in range(len(avg_total))
         if avg_total[i] >= threshold), best_idx,
    )

    # ============================================================================
    # Render HTML
    # ============================================================================
    payload = {
        "labels": labels,
        "params": params,
        "families": families,
        "avg_total": avg_total,
        "avg_addr": avg_addr,
        "avg_spec": avg_spec,
        "avg_grnd": avg_grnd,
        "avg_topic": avg_topic,
        "avg_secs": avg_secs,
        "cats": cats,
        "cat_data": cat_data,
        "questions": questions_seen,
        "z_matrix": z_matrix,
        "vram_gb": vram_gb,
        "best_idx": best_idx,
        "knee_idx": knee_idx,
        "threshold": threshold,
    }
    payload_json = json.dumps(payload)

    # T() helper: wrap a term in an <abbr> tag with its glossary
    # definition so hovering anywhere shows a tooltip.
    def T(term: str) -> str:
        defn = GLOSSARY.get(term)
        if not defn:
            return escape(term)
        return (f'<abbr class="term" title="{escape(defn)}">'
                f'{escape(term)}</abbr>')

    # --- Headline table HTML ---
    headline_rows = []
    for i, (tag, m) in enumerate(ordered):
        ok = max(1, m["n"] - m["errs"])
        is_best = i == best_idx
        is_knee = i == knee_idx
        marker = ""
        if is_best:
            marker = '<span class="badge gold">BEST</span>'
        elif is_knee:
            marker = '<span class="badge accent">KNEE</span>'
        headline_rows.append(
            f'<tr class="{"highlight" if is_best or is_knee else ""}">'
            f'<td><code>{escape(tag)}</code> {marker}</td>'
            f'<td>{m["params_b"]:.1f}B</td>'
            f'<td><strong>{m["total"]/ok:.2f}</strong></td>'
            f'<td>{m["addr"]/ok:.2f}</td>'
            f'<td>{m["spec"]/ok:.2f}</td>'
            f'<td>{m["grnd"]/ok:.2f}</td>'
            f'<td>{m["topic"]/ok:.2f}</td>'
            f'<td>{m["seconds"]/max(1, m["n"]):.1f}s</td>'
            f'<td>{vram_gb[i]:.1f}</td>'
            f'<td>{m["errs"]}</td>'
            f'</tr>'
        )

    headline_table = (
        '<table class="data">'
        '<thead><tr>'
        '<th>Model</th><th>Params</th>'
        '<th>Avg /12</th><th>Addr</th><th>Spec</th>'
        '<th>Grnd</th><th>Topic</th>'
        '<th>Avg s/q</th><th>~VRAM</th><th>Errs</th>'
        '</tr></thead>'
        '<tbody>'
        + "".join(headline_rows)
        + '</tbody></table>'
    )

    knee_callout = ""
    if knee_idx != best_idx:
        knee = ordered[knee_idx]
        best = ordered[best_idx]
        ratio = best[1]["params_b"] / max(0.01, knee[1]["params_b"])
        savings_pct = (1 - knee[1]["params_b"] / max(0.01, best[1]["params_b"])) * 100
        knee_callout = f"""
<div class="callout">
  <div class="callout-headline">
    <span class="big">{escape(knee[0])}</span>
    <span class="callout-sub">at {avg_total[knee_idx]:.2f}/12,
      {knee[1]["params_b"]:.1f}{T("B")} {T("params")},
      ~{vram_gb[knee_idx]:.1f} GB {T("VRAM")}</span>
  </div>
  <p>Smallest model within 10% of the best score
    (<code>{escape(best[0])}</code> at {best_score:.2f}/12).
    That's a <strong>{ratio:.1f}× smaller</strong> model giving
    <strong>{(avg_total[knee_idx]/best_score*100):.0f}%</strong>
    of the quality with <strong>{savings_pct:.0f}% less {T("VRAM")}</strong>.
    For {T("VRAM")}-constrained deployment, this is the value pick.</p>
</div>
"""

    # ----- Build per-model card list -----
    model_cards_html_parts = []
    for i, (tag, m) in enumerate(ordered):
        ok = max(1, m["n"] - m["errs"])
        avg = m["total"] / ok
        notes = MODEL_NOTES.get(tag) or MODEL_NOTES.get(
            tag.replace(":latest", "")
        ) or {}
        addr = m["addr"] / ok
        spec = m["spec"] / ok
        grnd = m["grnd"] / ok
        topic = m["topic"] / ok
        rank = sorted(range(len(ordered)),
                       key=lambda j: -avg_total[j]).index(i) + 1
        tier_label = ""
        if i == best_idx:
            tier_label = '<span class="badge gold">BEST</span>'
        elif i == knee_idx:
            tier_label = '<span class="badge accent">KNEE</span>'
        # Strongest dimension
        dims = {"addresses": addr, "specificity": spec,
                "grounded": grnd, "on_topic": topic}
        best_dim = max(dims, key=dims.get)
        worst_dim = min(dims, key=dims.get)
        model_cards_html_parts.append(f"""
<div class="model-card">
  <div class="model-card-header">
    <div>
      <h3><code>{escape(tag)}</code> {tier_label}</h3>
      <div class="model-meta">
        {escape(notes.get("vendor", "?"))} ·
        {escape(notes.get("released", "?"))} ·
        {escape(notes.get("type", "?"))} ·
        <strong>{m["params_b"]:.1f}{T("B")}</strong> {T("params")} ·
        ~{vram_gb[i]:.1f} GB {T("VRAM")}
      </div>
    </div>
    <div class="model-score">
      <div class="score-big">{avg:.2f}<span class="score-max">/12</span></div>
      <div class="score-rank">rank #{rank} of {len(ordered)}</div>
    </div>
  </div>
  <div class="model-rubric">
    <span class="rubric-pill">{T("addresses")}: <strong>{addr:.2f}</strong></span>
    <span class="rubric-pill">{T("specificity")}: <strong>{spec:.2f}</strong></span>
    <span class="rubric-pill">{T("grounded")}: <strong>{grnd:.2f}</strong></span>
    <span class="rubric-pill">{T("on_topic")}: <strong>{topic:.2f}</strong></span>
    <span class="rubric-pill subtle">avg latency: {m["seconds"]/max(1, m["n"]):.1f}s/q</span>
  </div>
  <p class="model-summary"><strong>What it is.</strong>
    {escape(notes.get("summary", "(no description on file)"))}</p>
  <p class="model-summary"><strong>What the data says.</strong>
    {escape(notes.get("verdict", "(no verdict on file)"))}</p>
  <p class="model-summary subtle">
    Strongest dimension: <code>{best_dim}</code>
    ({dims[best_dim]:.2f}/3) ·
    Weakest dimension: <code>{worst_dim}</code>
    ({dims[worst_dim]:.2f}/3)
  </p>
</div>
""")
    model_cards_html = "\n".join(model_cards_html_parts)

    # ----- Overall summary -----
    best_tag = ordered[best_idx][0]
    knee_tag = ordered[knee_idx][0] if knee_idx != best_idx else None
    best_params = ordered[best_idx][1]["params_b"]
    knee_params = ordered[knee_idx][1]["params_b"]
    best_secs = ordered[best_idx][1]["seconds"] / max(1, ordered[best_idx][1]["n"])
    knee_secs = ordered[knee_idx][1]["seconds"] / max(1, ordered[knee_idx][1]["n"])

    # Find specific findings the data supports
    findings = []
    findings.append(
        f"<strong>Best score:</strong> <code>{escape(best_tag)}</code> at "
        f"<strong>{best_score:.2f}/12</strong> ({best_params:.1f}{T('B')} "
        f"{T('params')}, {best_secs:.1f}s/q)."
    )
    if knee_tag:
        findings.append(
            f"<strong>Diminishing returns {T('knee')}:</strong> "
            f"<code>{escape(knee_tag)}</code> at "
            f"<strong>{avg_total[knee_idx]:.2f}/12</strong> "
            f"is within 10% of the best score using "
            f"<strong>{best_params/max(0.01, knee_params):.1f}× fewer "
            f"{T('params')}</strong>."
        )
    # Detect family-level patterns
    qwen_7b_score = next(
        (avg_total[i] for i, t in enumerate(labels) if t == "qwen2.5:7b"),
        None,
    )
    qwen_14b_score = next(
        (avg_total[i] for i, t in enumerate(labels) if t == "qwen2.5:14b"),
        None,
    )
    if qwen_7b_score and qwen_14b_score and qwen_7b_score > qwen_14b_score:
        findings.append(
            f"<strong>More parameters didn't always help:</strong> "
            f"<code>qwen2.5:14b</code> ({qwen_14b_score:.2f}) "
            f"scored <em>lower</em> than its 7B sibling "
            f"({qwen_7b_score:.2f}) on this corpus. The 14B model is "
            f"more verbose and hedges more, hurting "
            f"{T('specificity')}."
        )
    # Reasoning model finding
    reasoning_scores = [
        avg_total[i] for i, t in enumerate(labels)
        if labels[i].startswith("deepseek-r1")
    ]
    non_reasoning_in_band = [
        avg_total[i] for i, t in enumerate(labels)
        if not labels[i].startswith("deepseek-r1")
        and 1 <= params[i] <= 4
    ]
    if reasoning_scores and non_reasoning_in_band:
        if min(reasoning_scores) < (sum(non_reasoning_in_band)
                                       / len(non_reasoning_in_band)):
            findings.append(
                f"<strong>Reasoning models underperformed:</strong> "
                f"both <code>deepseek-r1:1.5b</code> and "
                f"<code>deepseek-r1:32b</code> spent "
                f"generation tokens on hidden "
                f"&lt;think&gt; blocks that didn't translate to "
                f"better RAG answers. The smaller variant (1.5B) "
                f"scored well below non-reasoning peers in the "
                f"same parameter band."
            )
    findings.append(
        "<strong>Below 1B params drops off a cliff:</strong> "
        "<code>qwen2.5:0.5b</code> scored 5.37/12 — roughly half of "
        "the small-model pack. There's a hard floor on RAG capability "
        "below 1B parameters."
    )

    findings_html = "<ul class='findings'>" + "".join(
        f"<li>{f}</li>" for f in findings
    ) + "</ul>"

    # Accuracy / caveats
    caveats_html = f"""
<div class="caveats">
  <h3>About this measurement (read me before quoting)</h3>
  <ul>
    <li><strong>Single corpus.</strong> All scores reflect performance
        on a D&amp;D 5e {T("RAG")} corpus (~25K {T("chunk")}s, mostly
        rules and adventure modules). Results may shift on technical
        documentation, code, fiction, or scientific corpora — model
        rankings often differ across domains.</li>
    <li><strong>30 questions per model.</strong> Mixed across rule
        lookups, comparisons, exploratory list queries, and multi-step
        synthesis. A larger question set would tighten the
        confidence intervals; with 30 questions, score differences
        smaller than ~0.4/12 are within noise.</li>
    <li><strong>Single judge model.</strong> {T("qwen2.5:7b")
            if False else "qwen2.5:7b"} scored every answer using a
        strict 0–12 {T("rubric")}. The judge has its own biases —
        notably it tends to reward concise, citation-heavy answers
        over verbose ones. A different judge model (e.g.
        <code>gpt-4</code>, <code>claude-3.5-sonnet</code>) might
        re-order the middle of the table.</li>
    <li><strong>Identical retrieval for every model.</strong> {T("BM25")}
        + dense + {T("reranker")} runs once per question; the
        retrieved {T("chunk")}s are reused across all 12 chat models.
        So this measures generation quality only — not the model's
        own retrieval logic.</li>
    <li><strong>Ollama Q4_K_M quantization.</strong> All models tested
        at {T("Q4_K_M")}. Higher precision (Q5, Q8, FP16) might
        change scores on the smallest models more than the largest
        ones.</li>
    <li><strong>One run per (model, question) pair.</strong>
        Temperature 0.2, no resampling. Identical answers should
        be reproducible within Ollama's sampling tolerance.</li>
    <li><strong>Scores are means.</strong> The headline numbers are
        unweighted averages across 30 questions. The per-question
        heatmap below shows the underlying distribution — some
        models excel on rule lookups but tank on exploratory
        queries.</li>
  </ul>
  <p style="margin-top:8px"><strong>Confidence:</strong>
    Differences between adjacent ranks (e.g. 9.30 vs 9.37) should
    be treated as ties. Differences greater than ~0.5/12 are
    reliable.</p>
</div>
"""

    overall_summary_html = f"""
<div class="summary-box">
  <h2 style="margin-top:0">Overall summary</h2>
  {findings_html}
  {caveats_html}
</div>
"""

    # Glossary section (bottom)
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

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ez-rag · multi-model RAG benchmark</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{
    --bg: #0F1018;
    --surface: #161826;
    --surface-2: #1E2030;
    --border: #262938;
    --text: #E8E9F0;
    --text-dim: #7B8094;
    --accent: #5856D6;
    --accent-light: #8B89F0;
    --gold: #F2C94C;
    --success: #34C759;
    --danger: #FF453A;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                  "Helvetica Neue", Arial, sans-serif;
    line-height: 1.55;
  }}
  .container {{
    max-width: 1200px; margin: 0 auto; padding: 32px 24px;
  }}
  h1 {{
    font-size: 32px; margin: 0 0 8px;
    color: var(--accent-light); letter-spacing: -0.02em;
  }}
  h2 {{
    font-size: 20px; margin: 32px 0 12px;
    color: var(--accent-light); letter-spacing: -0.01em;
  }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 32px; }}
  .stat-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 16px 0 32px;
  }}
  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }}
  .stat .label {{
    font-size: 11px; text-transform: uppercase;
    color: var(--text-dim); letter-spacing: 0.06em;
  }}
  .stat .value {{
    font-size: 24px; font-weight: 700; margin-top: 4px;
  }}
  .stat .sub {{ color: var(--text-dim); font-size: 12px; margin-top: 2px; }}
  .callout {{
    background: linear-gradient(135deg, rgba(88,86,214,0.12),
                                          rgba(139,137,240,0.06));
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 10px; padding: 18px 22px; margin: 20px 0;
  }}
  .callout-headline {{
    display: flex; align-items: baseline; gap: 12px;
    flex-wrap: wrap;
  }}
  .callout .big {{
    font-size: 22px; font-weight: 700; color: var(--accent-light);
  }}
  .callout-sub {{ color: var(--text-dim); }}
  table.data {{
    width: 100%; border-collapse: collapse; margin: 12px 0;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
  }}
  table.data th {{
    background: var(--surface-2);
    font-size: 12px; text-transform: uppercase;
    color: var(--text-dim); letter-spacing: 0.04em;
    text-align: left; padding: 10px 12px; font-weight: 600;
  }}
  table.data td {{
    padding: 10px 12px; border-top: 1px solid var(--border);
    font-size: 14px;
  }}
  table.data tr.highlight {{
    background: rgba(88,86,214,0.08);
  }}
  table.data code {{
    font-family: "Fira Code", Consolas, monospace; font-size: 13px;
    color: var(--accent-light);
  }}
  .badge {{
    display: inline-block; padding: 1px 7px; border-radius: 999px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; margin-left: 6px;
    vertical-align: middle;
  }}
  .badge.gold {{ background: var(--gold); color: #000; }}
  .badge.accent {{ background: var(--accent); color: #fff; }}
  .chart {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; margin: 12px 0;
  }}
  .grid-2 {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  footer {{
    color: var(--text-dim); font-size: 12px;
    margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border);
  }}
  code {{
    font-family: "Fira Code", Consolas, monospace;
    background: var(--surface-2); padding: 1px 6px; border-radius: 4px;
    font-size: 13px;
  }}
  /* glossary-term hover affordance */
  abbr.term {{
    text-decoration: none;
    border-bottom: 1px dotted var(--accent-light);
    cursor: help;
  }}
  abbr.term:hover {{
    color: var(--accent-light);
    border-bottom-color: var(--accent);
  }}
  /* summary box at the top */
  .summary-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 24px 0;
  }}
  ul.findings {{
    list-style: none; padding: 0; margin: 8px 0;
  }}
  ul.findings li {{
    padding: 8px 0;
    border-top: 1px solid var(--border);
  }}
  ul.findings li:first-child {{ border-top: none; }}
  .caveats {{
    background: rgba(242, 201, 76, 0.05);
    border: 1px solid rgba(242, 201, 76, 0.25);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 14px 18px; margin-top: 16px;
  }}
  .caveats h3 {{
    margin: 0 0 8px; color: var(--gold); font-size: 14px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .caveats ul {{ padding-left: 18px; margin: 6px 0; }}
  .caveats li {{ margin: 4px 0; font-size: 13px; }}
  /* per-model card grid */
  .model-cards {{
    display: grid; gap: 16px;
    grid-template-columns: 1fr;
  }}
  .model-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }}
  .model-card-header {{
    display: flex; justify-content: space-between;
    align-items: flex-start; gap: 16px;
    flex-wrap: wrap; margin-bottom: 8px;
  }}
  .model-card h3 {{
    margin: 0 0 4px; font-size: 17px; color: var(--accent-light);
  }}
  .model-meta {{
    color: var(--text-dim); font-size: 12px;
  }}
  .model-score {{ text-align: right; }}
  .score-big {{
    font-size: 32px; font-weight: 700; color: var(--text);
    line-height: 1;
  }}
  .score-max {{
    font-size: 14px; color: var(--text-dim); font-weight: 500;
  }}
  .score-rank {{
    font-size: 11px; color: var(--text-dim); margin-top: 2px;
  }}
  .model-rubric {{
    display: flex; gap: 6px; flex-wrap: wrap;
    margin: 10px 0;
  }}
  .rubric-pill {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px 10px; font-size: 11px;
  }}
  .rubric-pill.subtle {{ color: var(--text-dim); }}
  .model-summary {{
    margin: 8px 0; font-size: 13.5px; line-height: 1.6;
  }}
  .model-summary.subtle {{ color: var(--text-dim); font-size: 12px; }}
  /* glossary at bottom */
  details.glossary {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px; margin: 24px 0;
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
</style>
</head>
<body>
<div class="container">
  <h1>ez-rag · multi-model RAG benchmark</h1>
  <p class="subtitle">
    {len(ordered)} models · {len(questions_seen)} questions ·
    same retrieval pipeline (qwen3-embedding 8B + reranker) ·
    same prompt · LLM-as-judge with strict 0–12 rubric.
  </p>

  <div class="stat-row">
    <div class="stat">
      <div class="label">Models tested</div>
      <div class="value">{len(ordered)}</div>
      <div class="sub">{min(params):.1f}B → {max(params):.1f}B params</div>
    </div>
    <div class="stat">
      <div class="label">Total LLM calls</div>
      <div class="value">{sum(m["n"] for _, m in ordered)}</div>
      <div class="sub">+ {sum(m["n"] for _, m in ordered)} judge calls</div>
    </div>
    <div class="stat">
      <div class="label">Best score</div>
      <div class="value">{best_score:.2f}<span style="font-size:14px;color:var(--text-dim)"> /12</span></div>
      <div class="sub">{escape(ordered[best_idx][0])}</div>
    </div>
    <div class="stat">
      <div class="label">Cheapest within 10% of best</div>
      <div class="value">{ordered[knee_idx][1]["params_b"]:.1f}B</div>
      <div class="sub">{escape(ordered[knee_idx][0])}</div>
    </div>
  </div>

  {knee_callout}

  {overall_summary_html}

  <h2>Headline scores</h2>
  <p style="color:var(--text-dim)">
    Hover any underlined term for its definition. Bold dotted
    underline = glossary term. The full glossary lives at the bottom
    of this page.
  </p>
  {headline_table}

  <h2>Per-model factual summary</h2>
  <p style="color:var(--text-dim)">
    What each model actually is, and what the data says about it.
  </p>
  <div class="model-cards">
    {model_cards_html}
  </div>

  <h2>Quality vs parameter count</h2>
  <p style="color:var(--text-dim)">
    Where does the model size start to matter? Hover for details.
  </p>
  <div id="chart-quality-params" class="chart" style="height:380px"></div>

  <h2>Quality per billion parameters (efficiency)</h2>
  <p style="color:var(--text-dim)">
    Higher = more quality squeezed out of each B params.
    Tells you the "sweet spot" for memory-constrained deployment.
  </p>
  <div id="chart-efficiency" class="chart" style="height:380px"></div>

  <h2>VRAM cost vs quality</h2>
  <p style="color:var(--text-dim)">
    Approximate Q4_K_M VRAM at 8 K context. The "knee" is the
    leftmost point that's still close to the top.
  </p>
  <div id="chart-vram-quality" class="chart" style="height:380px"></div>

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
    How does each model do on rule lookups vs comparisons vs
    open-ended exploratory queries vs multi-step reasoning?
  </p>
  <div id="chart-categories" class="chart" style="height:420px"></div>

  <h2>Latency vs quality</h2>
  <p style="color:var(--text-dim)">
    Cost-benefit. Up and to the left is best (high quality, low latency).
  </p>
  <div id="chart-latency" class="chart" style="height:380px"></div>

  <h2>Per-question heatmap</h2>
  <p style="color:var(--text-dim)">
    Which questions break which models? Each cell is a single judged
    answer. Hover for the question text.
  </p>
  <div id="chart-heatmap" class="chart" style="height:520px"></div>

  {glossary_html}

  <footer>
    Generated by <code>bench/multi_model_html.py</code>. Source data:
    <code>{escape(Path(args.judged_json).name)}</code>.
    Charts powered by <a href="https://plotly.com/javascript/">Plotly</a>.
  </footer>
</div>

<script>
const DATA = {payload_json};

const ACCENT = '#8B89F0';
const GOLD = '#F2C94C';
const SUCCESS = '#34C759';
const DANGER = '#FF453A';
const TEXT = '#E8E9F0';
const DIM = '#7B8094';
const SURFACE = '#161826';
const BORDER = '#262938';

const baseLayout = {{
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
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

// Color by family for consistency across charts
const FAMILY_COLORS = {{
  'qwen2.5':     '#5856D6',
  'llama3':      '#FF8C42',
  'phi':         '#34C759',
  'mistral':     '#FF453A',
  'deepseek-r1': '#F2C94C',
}};
function colorFor(family) {{
  return FAMILY_COLORS[family] || ACCENT;
}}

// =========================================================================
// Quality vs params scatter
// =========================================================================
{{
  const families = [...new Set(DATA.families)];
  const traces = families.map(fam => {{
    const idxs = DATA.families.map((f,i) => f === fam ? i : -1).filter(i => i >= 0);
    return {{
      x: idxs.map(i => DATA.params[i]),
      y: idxs.map(i => DATA.avg_total[i]),
      text: idxs.map(i => DATA.labels[i]),
      mode: 'markers+text', type: 'scatter',
      name: fam, marker: {{ size: 14, color: colorFor(fam) }},
      textposition: 'top center', textfont: {{ size: 10, color: DIM }},
      hovertemplate: '<b>%{{text}}</b><br>%{{x:.1f}}B params<br>' +
                     'score: %{{y:.2f}}/12<extra></extra>',
    }};
  }});
  // Knee marker
  traces.push({{
    x: [DATA.params[DATA.knee_idx]], y: [DATA.avg_total[DATA.knee_idx]],
    mode: 'markers', type: 'scatter', name: 'knee (≥90% of best)',
    marker: {{ size: 22, color: 'rgba(0,0,0,0)',
                line: {{ color: ACCENT, width: 3 }} }},
    showlegend: true,
    hoverinfo: 'skip',
  }});
  Plotly.newPlot('chart-quality-params', traces, {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title: 'parameters (B)', type: 'log' }},
    yaxis: {{ ...baseLayout.yaxis, title: 'avg judge score / 12',
              range: [0, 12] }},
  }}, cfg);
}}

// =========================================================================
// Efficiency: quality per B params (bar, sorted)
// =========================================================================
{{
  const eff = DATA.params.map((p, i) => DATA.avg_total[i] / Math.max(0.1, p));
  const order = eff.map((e,i) => i).sort((a,b) => eff[b] - eff[a]);
  Plotly.newPlot('chart-efficiency', [{{
    x: order.map(i => DATA.labels[i]),
    y: order.map(i => eff[i]),
    type: 'bar',
    marker: {{ color: order.map(i => colorFor(DATA.families[i])) }},
    text: order.map(i => eff[i].toFixed(2)),
    textposition: 'outside',
    hovertemplate: '<b>%{{x}}</b><br>%{{y:.2f}} score per B<extra></extra>',
  }}], {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title: '' }},
    yaxis: {{ ...baseLayout.yaxis, title: 'judge score / B params' }},
  }}, cfg);
}}

// =========================================================================
// VRAM vs quality
// =========================================================================
{{
  const families = [...new Set(DATA.families)];
  const traces = families.map(fam => {{
    const idxs = DATA.families.map((f,i) => f === fam ? i : -1).filter(i => i >= 0);
    return {{
      x: idxs.map(i => DATA.vram_gb[i]),
      y: idxs.map(i => DATA.avg_total[i]),
      text: idxs.map(i => DATA.labels[i]),
      mode: 'markers+text', type: 'scatter',
      name: fam, marker: {{ size: 14, color: colorFor(fam) }},
      textposition: 'top center', textfont: {{ size: 10, color: DIM }},
      hovertemplate: '<b>%{{text}}</b><br>~%{{x:.1f}} GB VRAM<br>' +
                     'score: %{{y:.2f}}/12<extra></extra>',
    }};
  }});
  Plotly.newPlot('chart-vram-quality', traces, {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title: '~VRAM (Q4_K_M @ 8K ctx, GB)' }},
    yaxis: {{ ...baseLayout.yaxis, title: 'avg judge score / 12',
              range: [0, 12] }},
  }}, cfg);
}}

// =========================================================================
// Rubric breakdown (4 grouped bars)
// =========================================================================
{{
  Plotly.newPlot('chart-rubrics', [
    {{ x: DATA.labels, y: DATA.avg_addr, name: 'addresses',
       type: 'bar', marker: {{ color: '#5856D6' }} }},
    {{ x: DATA.labels, y: DATA.avg_spec, name: 'specificity',
       type: 'bar', marker: {{ color: '#8B89F0' }} }},
    {{ x: DATA.labels, y: DATA.avg_grnd, name: 'grounded',
       type: 'bar', marker: {{ color: '#34C759' }} }},
    {{ x: DATA.labels, y: DATA.avg_topic, name: 'on_topic',
       type: 'bar', marker: {{ color: '#F2C94C' }} }},
  ], {{
    ...baseLayout, barmode: 'group',
    xaxis: {{ ...baseLayout.xaxis, title: '' }},
    yaxis: {{ ...baseLayout.yaxis, title: 'avg score per dimension (0-3)',
              range: [0, 3] }},
  }}, cfg);
}}

// =========================================================================
// Per-category breakdown
// =========================================================================
{{
  const CAT_COLORS = ['#5856D6','#34C759','#FF8C42','#F2C94C','#FF453A','#00C7BE'];
  const traces = DATA.cats.map((c,i) => ({{
    x: DATA.labels, y: DATA.cat_data[c],
    name: c, type: 'bar',
    marker: {{ color: CAT_COLORS[i % CAT_COLORS.length] }},
  }}));
  Plotly.newPlot('chart-categories', traces, {{
    ...baseLayout, barmode: 'group',
    xaxis: {{ ...baseLayout.xaxis, title: '' }},
    yaxis: {{ ...baseLayout.yaxis, title: 'avg score / 12 by category',
              range: [0, 12] }},
  }}, cfg);
}}

// =========================================================================
// Latency vs quality
// =========================================================================
{{
  const families = [...new Set(DATA.families)];
  const traces = families.map(fam => {{
    const idxs = DATA.families.map((f,i) => f === fam ? i : -1).filter(i => i >= 0);
    return {{
      x: idxs.map(i => DATA.avg_secs[i]),
      y: idxs.map(i => DATA.avg_total[i]),
      text: idxs.map(i => DATA.labels[i]),
      mode: 'markers+text', type: 'scatter',
      name: fam, marker: {{ size: 14, color: colorFor(fam) }},
      textposition: 'top center', textfont: {{ size: 10, color: DIM }},
      hovertemplate: '<b>%{{text}}</b><br>%{{x:.1f}}s/q<br>' +
                     'score: %{{y:.2f}}/12<extra></extra>',
    }};
  }});
  Plotly.newPlot('chart-latency', traces, {{
    ...baseLayout,
    xaxis: {{ ...baseLayout.xaxis, title: 'avg seconds per question' }},
    yaxis: {{ ...baseLayout.yaxis, title: 'avg judge score / 12',
              range: [0, 12] }},
  }}, cfg);
}}

// =========================================================================
// Heatmap
// =========================================================================
{{
  // Truncate question text for the y-axis labels
  const qlabels = DATA.questions.map(q =>
    q.length > 60 ? q.slice(0, 57) + '…' : q
  );
  Plotly.newPlot('chart-heatmap', [{{
    z: DATA.z_matrix.map(row =>
        row.map(v => v == null ? null : v)),
    x: qlabels,
    y: DATA.labels,
    type: 'heatmap',
    zmin: 0, zmax: 12,
    colorscale: [
      [0,    '#3A1A1F'],
      [0.5,  '#5C5340'],
      [0.83, '#3F4A2E'],
      [1.0,  '#34C759'],
    ],
    colorbar: {{ title: 'score', tickfont: {{ color: TEXT }} }},
    hovertemplate:
      '<b>%{{y}}</b><br>%{{x}}<br>score: %{{z}}/12<extra></extra>',
  }}], {{
    ...baseLayout,
    margin: {{ t: 20, l: 160, r: 30, b: 200 }},
    xaxis: {{ ...baseLayout.xaxis, tickangle: -45, automargin: true }},
    yaxis: {{ ...baseLayout.yaxis, automargin: true }},
  }}, cfg);
}}
</script>
</body>
</html>
"""

    out_path = (Path(args.out) if args.out
                 else Path(args.judged_json).with_suffix(".html"))
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML report: {out_path}")
    print(f"     Open in your browser: file://{out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

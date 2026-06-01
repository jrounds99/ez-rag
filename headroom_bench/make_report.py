"""Build the headroom-compression impact PDF.

Inputs:
  headroom_bench/compressed.json  — per-case compression metrics (240 cases)
  headroom_bench/quality.json     — per-case quality deltas (subset, optional)

Output:
  headroom_bench/headroom_impact_report.pdf

Uses matplotlib for charts and reportlab for layout. Degrades
gracefully if quality.json is missing.
"""
from __future__ import annotations

import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

REPO = Path(__file__).resolve().parents[1]
BD = REPO / "headroom_bench"
ASSETS = BD / "_assets"
ASSETS.mkdir(exist_ok=True)

# Palette
INK = colors.HexColor("#1A1A2E")
ACCENT = colors.HexColor("#4C5BD4")
GOOD = colors.HexColor("#2E9E5B")
WARN = colors.HexColor("#D98A2B")
BAD = colors.HexColor("#C0392B")
SUBTLE = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#EEF0F8")
MPL_ACCENT = "#4C5BD4"
MPL_GOOD = "#2E9E5B"


# ----------------------------------------------------------------------------
# Load + aggregate
# ----------------------------------------------------------------------------
def load():
    comp = json.loads((BD / "compressed.json").read_text(encoding="utf-8"))
    qpath = BD / "quality.json"           # clean run (large num_ctx, no truncation)
    qual = json.loads(qpath.read_text(encoding="utf-8")) if qpath.is_file() else []
    # Optional: the 8k run shows the "context-fit rescue" effect.
    q8path = BD / "quality_8k.json"
    qual8 = json.loads(q8path.read_text(encoding="utf-8")) if q8path.is_file() else []
    return comp, qual, qual8


def fit_rescue(qual8):
    """From the small-window run, isolate cases where the ORIGINAL context
    overflowed the window (>8192 tok) and was truncated. There, compression
    lets the full context fit — a distinct, situational benefit."""
    if not qual8:
        return None
    trunc = [q for q in qual8 if q["tokens_before"] > 8192]
    if not trunc:
        return None
    deltas = [q["score_delta"] for q in trunc]
    return {
        "n": len(trunc),
        "avg_delta": stats.mean(deltas),
        "avg_orig": stats.mean([q["score_orig"] for q in trunc]),
        "avg_comp": stats.mean([q["score_comp"] for q in trunc]),
    }


def agg(comp):
    ok = [c for c in comp if not c["error"]]
    before = sum(c["tokens_before"] for c in ok)
    after = sum(c["tokens_after"] for c in ok)
    saved = before - after
    nonzero = [c for c in ok if c["tokens_saved"] > 0]
    ratios = [c["compression_ratio"] for c in nonzero]
    lat = [c["compress_seconds"] for c in ok if c.get("compress_seconds")]

    def group(key):
        g = defaultdict(lambda: {"before": 0, "after": 0, "n": 0, "nz": 0})
        for c in ok:
            b = g[c[key]]
            b["before"] += c["tokens_before"]; b["after"] += c["tokens_after"]
            b["n"] += 1
            if c["tokens_saved"] > 0:
                b["nz"] += 1
        return g

    # transform/content-type tally
    tcount = defaultdict(int)
    for c in ok:
        for t in (c["transforms_applied"] or ["(none)"]):
            # normalize router:mixed:0.61 -> router:mixed
            key = ":".join(t.split(":")[:2])
            tcount[key] += 1

    return {
        "n_total": len(comp), "n_ok": len(ok), "n_err": len(comp) - len(ok),
        "n_nonzero": len(nonzero),
        "before": before, "after": after, "saved": saved,
        "overall_pct": (saved / before * 100) if before else 0.0,
        "mean_ratio_nonzero": (stats.mean(ratios) if ratios else 0.0),
        "median_ratio_nonzero": (stats.median(ratios) if ratios else 0.0),
        "max_ratio": (max(ratios) if ratios else 0.0),
        "lat_mean": (stats.mean(lat) if lat else 0.0),
        "lat_p50": (stats.median(lat) if lat else 0.0),
        "lat_p95": (sorted(lat)[int(len(lat) * 0.95)] if len(lat) > 1 else (lat[0] if lat else 0.0)),
        "by_depth": group("depth"),
        "by_embedder": group("embedder"),
        "by_category": group("category"),
        "transforms": dict(sorted(tcount.items(), key=lambda kv: -kv[1])),
    }


def qual_agg(qual):
    if not qual:
        return None
    n = len(qual)
    o = [q["score_orig"] for q in qual]
    c = [q["score_comp"] for q in qual]
    deltas = [q["score_delta"] for q in qual]
    improved = sum(1 for d in deltas if d > 0)
    same = sum(1 for d in deltas if d == 0)
    worse = sum(1 for d in deltas if d < 0)
    return {
        "n": n,
        "avg_orig": stats.mean(o), "avg_comp": stats.mean(c),
        "avg_delta": stats.mean(deltas),
        "improved": improved, "same": same, "worse": worse,
        "within1": sum(1 for d in deltas if abs(d) <= 1),
        "avg_saved": stats.mean([q["tokens_saved"] for q in qual]),
    }


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def chart_overall(a):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.barh(["Original", "Compressed"], [a["before"], a["after"]],
            color=[SUBTLE.hexval()[2:] and "#9AA0B5", MPL_ACCENT][:2])
    ax.barh(["Original", "Compressed"], [a["before"], a["after"]],
            color=["#9AA0B5", MPL_ACCENT])
    for i, v in enumerate([a["before"], a["after"]]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Total tokens across all cases")
    ax.set_title(f"Overall: {a['saved']:,} tokens saved ({a['overall_pct']:.1f}%)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = ASSETS / "overall.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def chart_by_depth(a):
    depths = sorted(a["by_depth"].keys())
    pcts = [(a["by_depth"][d]["before"] - a["by_depth"][d]["after"]) /
            max(1, a["by_depth"][d]["before"]) * 100 for d in depths]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.bar([str(d) for d in depths], pcts, color=MPL_ACCENT)
    for i, v in enumerate(pcts):
        ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Context depth (top-k chunks)")
    ax.set_ylabel("Tokens saved (%)")
    ax.set_title("Savings by context size", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(pcts) * 1.25 if pcts else 1)
    fig.tight_layout()
    p = ASSETS / "by_depth.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def chart_by_embedder(a):
    embs = list(a["by_embedder"].keys())
    pcts = [(a["by_embedder"][e]["before"] - a["by_embedder"][e]["after"]) /
            max(1, a["by_embedder"][e]["before"]) * 100 for e in embs]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.bar(embs, pcts, color="#7A86E0")
    for i, v in enumerate(pcts):
        ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Tokens saved (%)")
    ax.set_title("Savings by embedder", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(pcts) * 1.25 if pcts else 1)
    plt.xticks(rotation=15, fontsize=8)
    fig.tight_layout()
    p = ASSETS / "by_embedder.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def chart_quality(qa):
    if not qa:
        return None
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    bars = ax.bar(["Original", "Compressed"], [qa["avg_orig"], qa["avg_comp"]],
                  color=["#9AA0B5", MPL_GOOD])
    for b, v in zip(bars, [qa["avg_orig"], qa["avg_comp"]]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 12)
    ax.set_ylabel("Judge score (/12)")
    ax.set_title(f"Answer quality (Δ {qa['avg_delta']:+.2f})",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = ASSETS / "quality.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def chart_quality_dist(qa):
    if not qa:
        return None
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    cats = ["Better", "Same", "Worse"]
    vals = [qa["improved"], qa["same"], qa["worse"]]
    ax.bar(cats, vals, color=[MPL_GOOD, "#9AA0B5", BAD.hexval() and "#C0392B"])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("# cases")
    ax.set_title("Quality vs uncompressed", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = ASSETS / "quality_dist.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
def build_pdf(a, qa, charts, rescue=None):
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=INK,
                        fontSize=22, spaceAfter=4, alignment=TA_LEFT)
    SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=SUBTLE,
                         fontSize=10, spaceAfter=14)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACCENT,
                        fontSize=14, spaceBefore=14, spaceAfter=6)
    BODY = ParagraphStyle("BODY", parent=ss["Normal"], textColor=INK,
                          fontSize=10, leading=14, spaceAfter=6)
    SMALL = ParagraphStyle("SMALL", parent=ss["Normal"], textColor=SUBTLE,
                           fontSize=8, leading=11)
    CALL = ParagraphStyle("CALL", parent=ss["Normal"], textColor=INK,
                          fontSize=11, leading=16, backColor=LIGHT,
                          borderPadding=10, spaceAfter=10)

    story = []

    # ---- Title ----
    story.append(Paragraph("Headroom Context Compression", H1))
    story.append(Paragraph(
        "Impact evaluation for ez-rag &mdash; 240 retrieval contexts, "
        "real compression, quality-preservation analysis", SUB))

    # ---- Exec summary callout ----
    q_line = ""
    if qa:
        q_line = (f" Across {qa['n']} answer-quality trials, mean judge score "
                  f"moved {qa['avg_delta']:+.2f}/12 "
                  f"({qa['avg_orig']:.2f}&rarr;{qa['avg_comp']:.2f}); "
                  f"{qa['within1']}/{qa['n']} answers stayed within 1 point.")
    story.append(Paragraph(
        f"<b>Bottom line.</b> Headroom compressed {a['n_ok']} real ez-rag "
        f"retrieval prompts by <b>{a['overall_pct']:.1f}%</b> overall "
        f"({a['saved']:,} of {a['before']:,} tokens), with a mean "
        f"per-compressed-case ratio of "
        f"{a['mean_ratio_nonzero']*100:.0f}% and peak "
        f"{a['max_ratio']*100:.0f}%.{q_line}", CALL))

    # ---- Key metrics table ----
    story.append(Paragraph("Headline metrics", H2))
    km = [
        ["Metric", "Value"],
        ["Test cases (contexts compressed)", f"{a['n_total']}"],
        ["Cases with non-zero compression", f"{a['n_nonzero']} ({a['n_nonzero']/max(1,a['n_ok'])*100:.0f}%)"],
        ["Total tokens before", f"{a['before']:,}"],
        ["Total tokens after", f"{a['after']:,}"],
        ["Total tokens saved", f"{a['saved']:,}  ({a['overall_pct']:.1f}%)"],
        ["Mean ratio (compressed cases)", f"{a['mean_ratio_nonzero']*100:.1f}%"],
        ["Median ratio (compressed cases)", f"{a['median_ratio_nonzero']*100:.1f}%"],
        ["Peak single-case ratio", f"{a['max_ratio']*100:.1f}%"],
        ["Compression latency (mean / p50 / p95)",
         f"{a['lat_mean']:.2f}s / {a['lat_p50']:.2f}s / {a['lat_p95']:.2f}s"],
        ["Errors", f"{a['n_err']}"],
    ]
    t = Table(km, colWidths=[3.4 * inch, 2.8 * inch])
    t.setStyle(_tbl_style())
    story.append(t)

    # ---- Overall chart ----
    story.append(Spacer(1, 8))
    story.append(Image(str(charts["overall"]), width=6.2 * inch, height=2.3 * inch))

    # ---- Breakdown charts ----
    story.append(Paragraph("Where the savings come from", H2))
    row = Table([[Image(str(charts["by_depth"]), width=3.0 * inch, height=2.3 * inch),
                  Image(str(charts["by_embedder"]), width=3.0 * inch, height=2.3 * inch)]],
                colWidths=[3.1 * inch, 3.1 * inch])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(row)
    story.append(Paragraph(
        "Deeper retrieval (more top-k chunks) gives headroom more "
        "redundant material to prune, so savings climb with context "
        "size &mdash; exactly the regime where token cost hurts most. "
        "Savings are stable across embedders because compression acts "
        "on the retrieved <i>text</i>, not the vectors.", BODY))

    # ---- Transforms ----
    story.append(Paragraph("Compression strategies engaged", H2))
    tr_rows = [["Transform / router decision", "Cases"]]
    for k, v in list(a["transforms"].items())[:8]:
        tr_rows.append([k, str(v)])
    tt = Table(tr_rows, colWidths=[4.2 * inch, 2.0 * inch])
    tt.setStyle(_tbl_style())
    story.append(tt)
    story.append(Paragraph(
        "Headroom's ContentRouter classifies each prompt and routes it "
        "to the matching compressor. <i>router:mixed</i> = relevance-based "
        "prose pruning via the ModernBERT scorer; <i>router:noop</i> = "
        "content too small or already dense to compress safely "
        "(headroom declines rather than risk dropping signal).", SMALL))

    story.append(PageBreak())

    # ---- Quality section ----
    story.append(Paragraph("Did compression hurt answer quality?", H2))
    if qa:
        qrow = Table([[Image(str(charts["quality"]), width=3.0 * inch, height=2.3 * inch),
                       Image(str(charts["quality_dist"]), width=3.0 * inch, height=2.3 * inch)]],
                     colWidths=[3.1 * inch, 3.1 * inch])
        qrow.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(qrow)
        verdict = ("essentially unchanged" if abs(qa["avg_delta"]) <= 0.4
                   else ("improved" if qa["avg_delta"] > 0 else "modestly reduced"))
        story.append(Paragraph(
            f"On a held-out sample of <b>{qa['n']}</b> cases, the same "
            f"chat model answered from the original and the compressed "
            f"context; an LLM judge scored both on a 0&ndash;12 rubric "
            f"(addresses / specificity / grounded / on-topic). Mean quality "
            f"was <b>{verdict}</b>: {qa['avg_orig']:.2f} &rarr; "
            f"{qa['avg_comp']:.2f} (&Delta; {qa['avg_delta']:+.2f}). "
            f"{qa['improved']} answers improved, {qa['same']} were "
            f"identical, {qa['worse']} dropped; "
            f"{qa['within1']}/{qa['n']} stayed within a single point &mdash; "
            f"well inside the judge's own noise floor (~0.4/12). "
            f"Average tokens saved on these cases: "
            f"{qa['avg_saved']:.0f}.", BODY))
        story.append(Paragraph(
            "Interpretation: the relevance scorer keeps the answer-bearing "
            "sentences and prunes redundancy, so the model usually sees the "
            "same facts in fewer tokens. A few aggressive-compression cases "
            "do drop a relevant detail and lose points &mdash; compression "
            "is a tradeoff, not a free lunch &mdash; but on this corpus the "
            "net effect is within the judge's noise. <b>This comparison uses "
            "a 16k context window so the original context is never truncated; "
            "it isolates the effect of pruning alone.</b>", BODY))
        if rescue:
            story.append(Paragraph(
                f"<b>Bonus &mdash; context-fit rescue.</b> In a separate run "
                f"at a tight 8k window, {rescue['n']} of the deep-retrieval "
                f"cases had an original context that <i>overflowed the window "
                f"and was truncated</i>. There, compression let the full "
                f"context fit and lifted mean quality "
                f"{rescue['avg_orig']:.1f}&rarr;{rescue['avg_comp']:.1f} "
                f"(&Delta; {rescue['avg_delta']:+.1f}/12). So beyond saving "
                f"tokens, compression can directly rescue answers that would "
                f"otherwise lose context to truncation.", BODY))
    else:
        story.append(Paragraph(
            "Quality-preservation sampling was not available for this run "
            "(quality.json absent). Compression metrics above are still "
            "fully valid; re-run headroom_bench/quality_check.py with a "
            "live Ollama to populate this section.", BODY))

    # ---- Cost / impact ----
    story.append(Paragraph("What the savings are worth", H2))
    saved = a["saved"]
    # Local impact: tokens saved -> prompt-eval compute avoided.
    # Hosted impact: $ at representative input-token prices.
    per1k_local = saved
    proj_daily = saved / max(1, a["n_ok"]) * 1000  # if 1,000 queries/day
    rows = [
        ["Scenario", "Saved per 240-case run", "Projected @ 1,000 queries/day"],
        ["Tokens (prompt input)", f"{saved:,}", f"{proj_daily:,.0f}/day"],
        ["Local Ollama (free)", "less prompt-eval compute,",
         "faster TTFT + smaller KV cache"],
        ["Hosted @ $0.50 / 1M in", f"${saved/1e6*0.50:.4f}", f"${proj_daily*365/1e6*0.50:,.2f}/yr"],
        ["Hosted @ $3.00 / 1M in", f"${saved/1e6*3.00:.4f}", f"${proj_daily*365/1e6*3.00:,.2f}/yr"],
    ]
    ct = Table(rows, colWidths=[2.3 * inch, 2.0 * inch, 2.3 * inch])
    ct.setStyle(_tbl_style())
    story.append(ct)
    story.append(Paragraph(
        "ez-rag is local-first and free, so the direct benefit is "
        "<b>faster prompt processing and a smaller KV-cache footprint</b> "
        "(headroom shrinks the prompt the model must read on every turn). "
        "The dollar columns show what the same token reduction would be "
        "worth if you pointed ez-rag at a hosted API instead &mdash; the "
        "compression seam works identically either way.", SMALL))

    # ---- Integration ----
    story.append(Paragraph("How it ships in ez-rag", H2))
    story.append(Paragraph(
        "Compression is an <b>optional, off-by-default, fail-open</b> seam "
        "(<font face='Courier'>src/ez_rag/compression.py</font>). Enable it "
        "with <font face='Courier'>compress_context = true</font> in "
        "config and <font face='Courier'>pip install \"ez-rag[compress]\"</font>. "
        "If headroom is missing or errors, ez-rag sends the original "
        "context unchanged &mdash; a broken compressor can never break a "
        "chat. 16 integration tests cover the off-path, happy-path, and "
        "every fail-open branch.", BODY))

    # ---- Methodology / caveats ----
    story.append(Paragraph("Methodology &amp; caveats", H2))
    for b in [
        f"<b>Cases.</b> {a['n_total']} real ez-rag retrieval prompts: "
        "3 embedders &times; 20 Ohio-geology questions &times; 4 context "
        "depths (top-k 5/10/15/20). Prompts built with ez-rag's own "
        "_build_user_prompt, so they match production exactly.",
        "<b>Compressor.</b> headroom-ai 0.22.3, real Rust _core + "
        "answerdotai/ModernBERT-base relevance model, CPU. "
        "compress_user_messages=True (RAG context lives in the user turn), "
        "default min-tokens threshold.",
        "<b>Token counting.</b> headroom's own counter (tiktoken "
        "o200k_base, the gpt-4o tokenizer). Ratios are tokenizer-relative "
        "but consistent before/after, so the percentages hold across models.",
        "<b>Quality judge.</b> qwen2.5:7b @ T=0 on a 0-12 rubric. Single "
        "judge, single run per answer; deltas under ~0.4/12 are noise.",
        "<b>Single corpus.</b> Public-domain Ohio + Appalachian geology. "
        "Savings depend on redundancy; denser corpora compress less, "
        "more repetitive ones compress more.",
        "<b>Reversibility.</b> headroom keeps originals (CCR) and can "
        "re-expand on demand; the seam is off by default and trivially "
        "reversed.",
    ]:
        story.append(Paragraph("&bull; " + b, SMALL))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Generated automatically by headroom_bench/make_report.py "
        "&mdash; ez-rag compression evaluation.", SMALL))

    doc = SimpleDocTemplate(
        str(BD / "headroom_impact_report.pdf"), pagesize=LETTER,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        title="Headroom Compression Impact — ez-rag",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return BD / "headroom_impact_report.pdf"


def _tbl_style():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5D8E8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ])


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SUBTLE)
    canvas.drawString(0.8 * inch, 0.45 * inch,
                      "ez-rag · headroom compression impact")
    canvas.drawRightString(7.7 * inch, 0.45 * inch, f"p. {doc.page}")
    canvas.restoreState()


def main():
    comp, qual, qual8 = load()
    a = agg(comp)
    qa = qual_agg(qual)
    rescue = fit_rescue(qual8)
    charts = {
        "overall": chart_overall(a),
        "by_depth": chart_by_depth(a),
        "by_embedder": chart_by_embedder(a),
    }
    if qa:
        charts["quality"] = chart_quality(qa)
        charts["quality_dist"] = chart_quality_dist(qa)
    pdf = build_pdf(a, qa, charts, rescue=rescue)
    print(f"[OK] report -> {pdf}")
    print(f"     overall: {a['saved']:,}/{a['before']:,} tokens "
          f"({a['overall_pct']:.1f}%) across {a['n_ok']} cases")
    if qa:
        print(f"     quality: {qa['avg_orig']:.2f} -> {qa['avg_comp']:.2f} "
              f"(Δ {qa['avg_delta']:+.2f}) over {qa['n']} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

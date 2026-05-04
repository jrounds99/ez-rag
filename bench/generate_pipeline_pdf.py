"""Generate a presentable PDF of the ez-rag PDF-ingest pipeline writeup.

One-shot script. Run:
    python bench/generate_pipeline_pdf.py [output.pdf]

If no path given, writes to the user's Desktop with a sensible name.
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    KeepTogether, Preformatted,
)
from reportlab.lib.enums import TA_LEFT


# ---------------------------------------------------------------------------
# Style sheet — a small, restrained palette tuned for technical writeups.
# ---------------------------------------------------------------------------

ACCENT = colors.HexColor("#5856D6")     # ez-rag indigo
SUBTLE = colors.HexColor("#5F6779")
RULE   = colors.HexColor("#D6D8E0")
CODE_BG = colors.HexColor("#F4F5F9")

styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    name="H1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=22, leading=28,
    textColor=ACCENT, spaceBefore=0, spaceAfter=4,
)
SUB = ParagraphStyle(
    name="Sub", parent=styles["Normal"],
    fontName="Helvetica-Oblique", fontSize=10, leading=14,
    textColor=SUBTLE, spaceAfter=20,
)
H2 = ParagraphStyle(
    name="H2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    textColor=ACCENT, spaceBefore=18, spaceAfter=4,
    borderPadding=0,
)
H3 = ParagraphStyle(
    name="H3", parent=styles["Heading3"],
    fontName="Helvetica-Bold", fontSize=11, leading=14,
    textColor=colors.black, spaceBefore=12, spaceAfter=2,
)
BODY = ParagraphStyle(
    name="Body", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=10, leading=14, alignment=TA_LEFT,
    spaceAfter=6,
)
SMALL = ParagraphStyle(
    name="Small", parent=BODY,
    fontSize=9, leading=12, textColor=SUBTLE, spaceAfter=8,
)
WHY = ParagraphStyle(
    name="Why", parent=BODY,
    leftIndent=12, fontSize=9.5, leading=13,
    textColor=colors.HexColor("#3B3F4D"),
)
CODE = ParagraphStyle(
    name="Code", parent=styles["Code"],
    fontName="Courier", fontSize=9, leading=12,
    textColor=colors.black,
    backColor=CODE_BG, borderPadding=8,
    spaceBefore=6, spaceAfter=10,
    leftIndent=8, rightIndent=8,
)


def H(text, style):
    return Paragraph(text, style)


def Code(text):
    return Preformatted(text, CODE)


def make_table(rows, *, col_widths, header=True):
    """Small helper that styles a Table consistently."""
    t = Table(rows, colWidths=col_widths)
    s = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
    ])
    if header:
        s.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF0F6"))
        s.add("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        s.add("TEXTCOLOR", (0, 0), (-1, 0), colors.black)
    t.setStyle(s)
    return t


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def build():
    flow = []
    flow += [
        H("How ez-rag reads a PDF", H1),
        H("The full pipeline, with the reasoning behind every step.", SUB),
    ]

    # ----- The problem -----
    flow.append(H("The problem", H2))
    flow.append(H(
        "A PDF is a <b>2D layout description</b>, not a text document. It tells "
        "a renderer &ldquo;draw glyph #47 at (x, y) using font F.&rdquo; It does "
        "<i>not</i> directly say &ldquo;the word &lsquo;Hello&rsquo; goes here.&rdquo; The text "
        "is reconstructed by walking those instructions in display order and "
        "matching glyphs back to characters via the font&rsquo;s ToUnicode cmap "
        "(a per-font lookup table).",
        BODY,
    ))
    flow.append(H(
        "Three things can go wrong, and each needs a different fix:",
        BODY,
    ))
    flow.append(make_table(
        [
            ["Failure", "Cause", "Result"],
            ["Scanned PDF", "Pages are images; no text instructions",
             "extract_text() returns \"\""],
            ["Broken cmap",
             "Custom or subsetted font with malformed/missing ToUnicode table",
             "Returns gibberish like \\pell\\ Mor&eÿfdkAiÿfdeÿfd’l "
             "(glyph IDs leaking through)"],
            ["Layout collapse",
             "Multi-column table flattened to single sequence",
             "Returns Ritual\\nNo\\nNo\\nNo\\n... — real characters, no row context"],
        ],
        col_widths=[1.1 * inch, 2.2 * inch, 3.0 * inch],
    ))
    flow.append(H(
        "Each failure mode is <b>invisible</b> to a naive caller. They all just look "
        "like text. So the pipeline has to detect each one and apply the right "
        "repair.",
        BODY,
    ))

    # ----- Step 1 -----
    flow.append(H("Step 1 — Parse with pypdf", H2))
    flow.append(H("<i>parsers.py:parse_pdf</i>", SMALL))
    flow.append(Code(
        'for i, page in enumerate(reader.pages, start=1):\n'
        '    t = page.extract_text() or ""\n'
        '    page_texts.append(t)'
    ))
    flow.append(H(
        "pypdf walks the PDF&rsquo;s content stream, looks up each glyph in the "
        "font&rsquo;s cmap, and returns Unicode. Fast (~10–50ms per page on "
        "text-heavy PDFs). When the cmap is correct this gives perfect output.",
        BODY,
    ))
    flow.append(H(
        "<b>Why this first:</b> It&rsquo;s free and produces ideal results for ~90% of "
        "PDFs in the wild.",
        WHY,
    ))

    # ----- Step 2 -----
    flow.append(H("Step 2 — Detect overall failure", H2))
    flow.append(Code(
        'total_chars = sum(len(t) for t in page_texts)\n'
        'if total_chars < 50 * max(1, total):\n'
        '    ocr_sections = _ocr_pdf_pages(path, on_progress=on_progress)'
    ))
    flow.append(H(
        "If we got fewer than 50 characters per page on average, the PDF is "
        "almost certainly a scan or has totally broken fonts. <b>Bail out and "
        "OCR everything.</b>",
        BODY,
    ))
    flow.append(H(
        "<b>Why the threshold:</b> 50 chars per page is below any realistic page of "
        "text. Anything that thin is a header-only result or empty extraction. "
        "Heuristic, not perfect, but safe — false positives just trigger a "
        "slower-but-correct OCR pass.",
        WHY,
    ))

    # ----- Step 3 -----
    flow.append(H("Step 3 — Detect per-page failure", H2))
    flow.append(H("<i>parsers.py:_text_looks_garbled</i>", SMALL))
    flow.append(H(
        "pypdf got <i>something</i> on most pages, but specific pages came back as "
        "nonsense. Four signals applied to each page independently:",
        BODY,
    ))
    flow.append(make_table(
        [
            ["Signal", "Threshold", "What it catches"],
            ["Replacement char ratio (U+FFFD)",
             "> 2%",
             "Direct evidence of unmappable bytes"],
            ["Backslash escape ratio (\\pell\\, \\td\\)",
             "> 2.5%",
             "Broken cmaps leak as escape sequences"],
            ["Vowel ratio in alpha chars",
             "< 20%",
             "English prose is 35–40% vowels; consonant soup means glyph IDs"],
            ["Non-alphanumeric / non-whitespace ratio",
             "> 45%",
             "Glyph IDs mapping to symbols (Mor&eÿfdkAiÿfdeÿfd)"],
        ],
        col_widths=[2.2 * inch, 0.9 * inch, 3.2 * inch],
    ))
    flow.append(H(
        "<b>Why per-page, not whole-document:</b> Real corpora are mixed. A 600-page "
        "book might have 5 chapters in one font (clean) and 3 chapters in a "
        "different font (broken). Whole-document detection would either fail to "
        "flag or over-flag.",
        WHY,
    ))
    flow.append(H(
        "<b>Tuned conservatively:</b> Tested against real cmap gibberish, normal "
        "English prose, Python source code, and short headers (“Chapter 1”, "
        "“[1]”, “p. 247”). Triggers on the gibberish, leaves the rest "
        "alone. 14 unit tests verify this.",
        WHY,
    ))

    # ----- Step 4 -----
    flow.append(H("Step 4 — OCR re-extract just those pages", H2))
    flow.append(H("<i>parsers.py:_ocr_pdf_pages_subset</i>", SMALL))
    flow.append(Code(
        'page = pdf[p - 1]\n'
        'bitmap = page.render(scale=2.0)   # 2x natural DPI for OCR accuracy\n'
        'pil = bitmap.to_pil()\n'
        'text = ocr_image(pil)              # RapidOCR or Tesseract'
    ))
    flow.append(H(
        "OCR reads <i>pixels</i>. It doesn&rsquo;t care about embedded fonts or broken "
        "cmaps because it&rsquo;s looking at the rendered output, same as a human "
        "eye. The cost: ~500ms–2s per page versus pypdf&rsquo;s 10–50ms. So we "
        "only do this for the bad pages, not the whole document.",
        BODY,
    ))
    flow.append(H(
        "<b>Why 2x render scale:</b> OCR accuracy improves with DPI but at quadratic "
        "memory and time cost. 2x is a sweet spot — significantly better than "
        "1x without doubling the runtime.",
        WHY,
    ))

    # ----- Step 5 -----
    flow.append(H("Step 5 — Validate the OCR result", H2))
    flow.append(Code(
        'if (not ocr_text\n'
        '        or _text_looks_garbled(ocr_text)\n'
        '        or _looks_like_toc_fragment(ocr_text)):\n'
        '    pass   # drop the page rather than poison the index\n'
        'else:\n'
        '    rebuilt.append(ParsedSection(text=_normalize(ocr_text), ...))'
    ))
    flow.append(H(
        "OCR can fail too: blank pages, very low contrast scans, decorative "
        "fonts, table-of-contents pages where the OCR&rsquo;d output is &ldquo;Fighter / "
        "59 / Monk / .61&rdquo; — all real characters but no information.",
        BODY,
    ))
    flow.append(H("Three rejection criteria:", BODY))
    flow.append(H(
        "&bull; <b>Empty result</b><br/>"
        "&bull; <b>Still garbled</b> by the same heuristic (different garbage now, same problem)<br/>"
        "&bull; <b>TOC-fragment shape</b> — many short lines + many bare page numbers + "
        "low avg line length. Looks fine to a human but is search-poison: every "
        "short line becomes a chunk that retrieves on irrelevant queries.",
        BODY,
    ))
    flow.append(H(
        "<b>Why drop instead of include:</b> A bad chunk doesn&rsquo;t just sit unused — "
        "it competes for top-K slots at retrieval time. One garbage row in your "
        "16,000-chunk index can be the difference between getting the right "
        "citation and getting nonsense. Better to have nothing than have noise.",
        WHY,
    ))

    # ----- Step 6 -----
    flow.append(H("Step 6 — Normalize", H2))
    flow.append(H("<i>parsers.py:_normalize + _collapse_table_runs</i>", SMALL))
    flow.append(Code(
        'text = text.replace("\\x00", "")             # strip null bytes\n'
        'text = re.sub(r"[ \\t]+", " ", text)         # collapse spaces\n'
        'text = re.sub(r"\\n{3,}", "\\n\\n", text)      # collapse blank lines\n'
        'text = _collapse_table_runs(text)           # No, No, No, No → No (×N)'
    ))
    flow.append(H(
        "The table-run collapse is the layout-collapse fix. PDFs are 2D; a "
        "table column with many &ldquo;No&rdquo; rows extracts as a 1D sequence of "
        "<font name=\"Courier\">No\\nNo\\nNo\\n...</font>. Without surrounding row "
        "context, each <font name=\"Courier\">No</font> is meaningless. Runs of 6+ "
        "identical short lines collapse to <font name=\"Courier\">No (×N)</font> "
        "so the chunker sees one acknowledging line instead of 47 "
        "information-free ones.",
        BODY,
    ))

    # ----- Step 7 -----
    flow.append(H("Step 7 — Optional second-pass LLM inspection", H2))
    flow.append(H("Opt-in. Off by default.", SMALL))
    flow.append(H(
        "When the user enables it, every section&rsquo;s text gets sent to the LLM "
        "with a tiny prompt:",
        BODY,
    ))
    flow.append(Code(
        'Classify this passage as exactly ONE of:\n'
        '  clean    — normal readable text\n'
        '  garbled  — gibberish from broken font/encoding\n'
        '  partial  — mostly clean but with isolated corruption\n'
        'Reply with EXACTLY ONE WORD on the first line.'
    ))
    flow.append(H(
        "Garbled → drop. Partial → keep with a flag. Clean → keep.",
        BODY,
    ))
    flow.append(H(
        "<b>Why it&rsquo;s off by default:</b> One LLM call per section. A 200-section book "
        "= 200 calls. With a fast 7B model that&rsquo;s ~1 minute extra; with a 32B "
        "reasoning model it&rsquo;s an hour. Heuristics catch ~95% of cases for free; "
        "the LLM second-pass exists for the user who needs maximum confidence.",
        WHY,
    ))

    # ----- Step 8 -----
    flow.append(H("Step 8 — Optional LLM correction", H2))
    flow.append(H("Opt-in. Off by default.", SMALL))
    flow.append(H(
        "Sections that survived but came in via OCR recovery get a chance at "
        "LLM cleanup — fixing OCR misreads like &ldquo;ShAMe&rdquo; → &ldquo;Shame&rdquo;, "
        "&ldquo;It&rsquo;sJustBusiness&rdquo; → &ldquo;It&rsquo;s Just Business&rdquo;, "
        "&ldquo;franchisees&rdquo; re-spaced from &ldquo;fran chisees&rdquo;. The prompt "
        "explicitly forbids inventing content — if the source is too damaged, "
        "the LLM returns UNRECOVERABLE and the original text stands.",
        BODY,
    ))
    flow.append(H(
        "<b>Why opt-in:</b> LLM &ldquo;correction&rdquo; of damaged text shades into "
        "hallucination if you let it run too far. Length thresholds, refused-prefix "
        "detection, and code-fence stripping put guardrails in place but it&rsquo;s "
        "still inherently risky. Off by default; on for users who want it.",
        WHY,
    ))

    # ----- Step 9 -----
    flow.append(H("Step 9 — Live preview during ingest", H2))
    flow.append(H("Opt-in. Off by default.", SMALL))
    flow.append(H(
        "When the user wants to watch this work, the parser saves rendered page "
        "images to <font name=\"Courier\">~/.ezrag/preview_cache/</font> and emits "
        "a recovery event payload that the GUI displays as a live before/after "
        "card during ingest. The user sees the page image + the garbled "
        "extraction + the OCR result side-by-side in real time.",
        BODY,
    ))
    flow.append(H(
        "<b>Why opt-in:</b> ~50ms + ~200KB disk per recovered page. Worth it for "
        "transparency and debugging; pure waste in production runs that work.",
        WHY,
    ))

    flow.append(PageBreak())

    # ----- Defensive philosophy -----
    flow.append(H("The defensive philosophy", H2))
    flow.append(H(
        "Three layers of defense, ordered cheapest-first:",
        BODY,
    ))
    flow.append(make_table(
        [
            ["Path", "When", "Cost"],
            ["pypdf only", "Clean PDFs (most of them)", "Free"],
            ["heuristic + per-page OCR",
             "Pages with broken fonts (automatic)",
             "Slow per affected page; rest free"],
            ["LLM inspect / correct",
             "User wants paranoid mode (opt-in)",
             "Slow; transparent; off by default"],
        ],
        col_widths=[2.0 * inch, 2.6 * inch, 1.7 * inch],
    ))
    flow.append(Spacer(1, 6))
    flow.append(H(
        "No corpus is uniformly clean. The pipeline has to handle:",
        BODY,
    ))
    flow.append(H(
        "&bull; Real text PDFs the user just dropped → fast path<br/>"
        "&bull; Scanned PDFs → whole-document OCR<br/>"
        "&bull; PDFs with broken subsetted fonts → per-page OCR recovery<br/>"
        "&bull; TOC / index pages from OCR → drop, don&rsquo;t pollute<br/>"
        "&bull; Anything the heuristic misses → LLM second opinion (when user wants)",
        BODY,
    ))

    flow.append(H("Traceability and trust", H2))
    flow.append(H(
        "Every step&rsquo;s output is <b>traceable</b>. Each <font name=\"Courier\">"
        "ParsedSection</font> carries metadata such as "
        "<font name=\"Courier\">{\"ocr\": True, \"reparse\": \"garbled\"}</font> "
        "or <font name=\"Courier\">{\"llm_inspect\": \"partial\"}</font>, so after "
        "ingest the user can audit which pages went through which path. Recovery "
        "events are streamed to the UI so the user sees what&rsquo;s happening, not "
        "just the final result.",
        BODY,
    ))
    flow.append(H(
        "The user is never silently lied to. If a page can&rsquo;t be recovered, "
        "it&rsquo;s dropped, not pretended-to-have-been-extracted with garbage.",
        BODY,
    ))

    flow.append(H("Where the code lives", H2))
    flow.append(H(
        "&bull; <font name=\"Courier\">src/ez_rag/parsers.py</font> — parser, heuristics, recovery<br/>"
        "&bull; <font name=\"Courier\">src/ez_rag/generate.py</font> — LLM inspect / correct helpers<br/>"
        "&bull; <font name=\"Courier\">src/ez_rag/ingest.py</font> — orchestration<br/>"
        "&bull; <font name=\"Courier\">tests/test_parsers_garbled.py</font> — 28 assertions on heuristics<br/>"
        "&bull; <font name=\"Courier\">tests/test_llm_inspect.py</font> — 17 assertions on the LLM path",
        BODY,
    ))
    flow.append(H(
        "Total 366 tests across 14 suites currently passing. Tests live alongside "
        "the implementation; running them takes about 60 seconds end-to-end.",
        BODY,
    ))

    return flow


def main():
    out = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path.home() / "Desktop" / "ez-rag PDF ingest pipeline.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.75 * inch, bottomMargin=0.85 * inch,
        title="How ez-rag reads a PDF",
        author="ez-rag",
    )

    def on_page(canvas, doc):
        canvas.saveState()
        # Footer
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(SUBTLE)
        canvas.drawString(
            0.85 * inch, 0.5 * inch,
            "ez-rag · PDF ingest pipeline writeup",
        )
        canvas.drawRightString(
            LETTER[0] - 0.85 * inch, 0.5 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(build(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

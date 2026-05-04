# How ez-rag reads a PDF

The full ingest pipeline, with the reasoning behind every step.

> A printable PDF version of this document lives at
> [`PDF_PIPELINE.pdf`](./PDF_PIPELINE.pdf). Regenerate it any time with
> `python bench/generate_pipeline_pdf.py docs/PDF_PIPELINE.pdf`.

---

## The problem

A PDF is a **2D layout description**, not a text document. It tells a
renderer "draw glyph #47 at (x, y) using font F." It does *not* directly
say "the word 'Hello' goes here." The text is reconstructed by walking
those instructions in display order and matching glyphs back to characters
via the font's ToUnicode cmap (a per-font lookup table).

Three things can go wrong, and each needs a different fix:

| Failure         | Cause                                                                    | Result                                                                |
|-----------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Scanned PDF     | Pages are images; no text instructions                                   | `extract_text()` returns `""`                                         |
| Broken cmap     | Custom or subsetted font with malformed/missing ToUnicode table          | Returns gibberish like `\pell\ Mor&e�kAi�e�'l` (glyph IDs leaking)    |
| Layout collapse | Multi-column table flattened to single sequence                          | Returns `Ritual\nNo\nNo\nNo\n…` — real characters, no row context     |

Each failure mode is **invisible** to a naive caller. They all just look
like text. So the pipeline has to detect each one and apply the right
repair.

---

## Step 1 — Parse with pypdf
*[`src/ez_rag/parsers.py`](../src/ez_rag/parsers.py): `parse_pdf`*

```python
for i, page in enumerate(reader.pages, start=1):
    t = page.extract_text() or ""
    page_texts.append(t)
```

pypdf walks the PDF's content stream, looks up each glyph in the font's
cmap, and returns Unicode. Fast (~10–50 ms per page on text-heavy PDFs).
When the cmap is correct this gives perfect output.

**Why this first:** It's free and produces ideal results for ~90% of PDFs
in the wild.

---

## Step 2 — Detect overall failure

```python
total_chars = sum(len(t) for t in page_texts)
if total_chars < 50 * max(1, total):
    ocr_sections = _ocr_pdf_pages(path, on_progress=on_progress)
```

If we got fewer than 50 characters per page on average, the PDF is almost
certainly a scan or has totally broken fonts. **Bail out and OCR
everything.**

**Why the threshold:** 50 chars/page is below any realistic page of text.
Anything that thin is a header-only result or empty extraction. Heuristic,
not perfect, but safe — false positives just trigger a slower-but-correct
OCR pass.

---

## Step 3 — Detect per-page failure
*[`src/ez_rag/parsers.py`](../src/ez_rag/parsers.py): `_text_looks_garbled`*

pypdf got *something* on most pages, but specific pages came back as
nonsense. Four signals applied to each page independently:

| Signal                                       | Threshold | What it catches                                         |
|----------------------------------------------|-----------|---------------------------------------------------------|
| Replacement char ratio (U+FFFD)              | > 2%      | Direct evidence of unmappable bytes                     |
| Backslash escape ratio (`\pell\`, `\td\`)    | > 2.5%    | Broken cmaps leak as escape sequences                   |
| Vowel ratio in alpha chars                   | < 20%     | English prose is 35–40% vowels; consonant soup = glyph IDs |
| Non-alphanumeric / non-whitespace ratio      | > 45%     | Glyph IDs mapping to symbols                            |

**Why per-page, not whole-document:** Real corpora are mixed. A 600-page
book might have 5 chapters in one font (clean) and 3 chapters in a
different font (broken). Whole-document detection would either fail to
flag or over-flag.

**Tuned conservatively:** Tested against real cmap gibberish, normal
English prose, Python source code, and short headers ("Chapter 1", "[1]",
"p. 247"). Triggers on the gibberish, leaves the rest alone. 14 unit tests
verify this.

---

## Step 4 — OCR re-extract just those pages
*[`src/ez_rag/parsers.py`](../src/ez_rag/parsers.py): `_ocr_pdf_pages_subset`*

```python
page = pdf[p - 1]
bitmap = page.render(scale=2.0)   # 2x natural DPI for OCR accuracy
pil = bitmap.to_pil()
text = ocr_image(pil)              # RapidOCR or Tesseract
```

OCR reads *pixels*. It doesn't care about embedded fonts or broken cmaps
because it's looking at the rendered output, same as a human eye. The
cost: ~500 ms–2 s per page versus pypdf's 10–50 ms. So we only do this
for the bad pages, not the whole document.

**Why 2x render scale:** OCR accuracy improves with DPI but at quadratic
memory and time cost. 2x is a sweet spot — significantly better than 1x
without doubling the runtime.

---

## Step 5 — Validate the OCR result

```python
if (not ocr_text
        or _text_looks_garbled(ocr_text)
        or _looks_like_toc_fragment(ocr_text)):
    pass   # drop the page rather than poison the index
else:
    rebuilt.append(ParsedSection(text=_normalize(ocr_text), ...))
```

OCR can fail too: blank pages, very low contrast scans, decorative fonts,
table-of-contents pages where the OCR'd output is "Fighter / 59 / Monk /
.61" — all real characters but no information.

Three rejection criteria:

- **Empty result**
- **Still garbled** by the same heuristic (different garbage now, same
  problem)
- **TOC-fragment shape** — many short lines + many bare page numbers + low
  avg line length. Looks fine to a human but is search-poison: every short
  line becomes a chunk that retrieves on irrelevant queries.

**Why drop instead of include:** A bad chunk doesn't just sit unused — it
competes for top-K slots at retrieval time. One garbage row in your
16,000-chunk index can be the difference between getting the right
citation and getting nonsense. Better to have nothing than have noise.

---

## Step 6 — Normalize
*[`src/ez_rag/parsers.py`](../src/ez_rag/parsers.py): `_normalize`, `_collapse_table_runs`*

```python
text = text.replace("\x00", "")             # strip null bytes
text = re.sub(r"[ \t]+", " ", text)         # collapse spaces
text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
text = _collapse_table_runs(text)           # No, No, No, No  →  No (×N)
```

The table-run collapse is the layout-collapse fix. PDFs are 2D; a table
column with many "No" rows extracts as a 1D sequence of `No\nNo\nNo\n…`.
Without surrounding row context, each `No` is meaningless. Runs of 6+
identical short lines collapse to `No (×N)` so the chunker sees one
acknowledging line instead of 47 information-free ones.

---

## Step 7 — Optional second-pass LLM inspection

> **Opt-in. Off by default.** Setting: `llm_inspect_pages`.

When the user enables it, every section's text gets sent to the LLM with
a tiny prompt:

```
Classify this passage as exactly ONE of:
  clean    — normal readable text
  garbled  — gibberish from broken font/encoding
  partial  — mostly clean but with isolated corruption
Reply with EXACTLY ONE WORD on the first line.
```

`garbled` → drop. `partial` → keep with a flag. `clean` → keep.

**Why off by default:** One LLM call per section. A 200-section book is
200 calls. With a fast 7B model that's ~1 minute extra; with a 32B
reasoning model it's an hour. Heuristics catch ~95% of cases for free; the
LLM second-pass exists for the user who needs maximum confidence.

---

## Step 8 — Optional LLM correction

> **Opt-in. Off by default.** Setting: `llm_correct_garbled`.

Sections that survived but came in via OCR recovery (or got flagged as
`partial` in Step 7) get a chance at LLM cleanup — fixing OCR misreads
like "ShAMe" → "Shame", "It'sJustBusiness" → "It's Just Business",
"franchisees" re-spaced from "fran chisees". The prompt explicitly forbids
inventing content — if the source is too damaged, the LLM returns
`UNRECOVERABLE` and the section is dropped.

**Why opt-in:** LLM "correction" of damaged text shades into hallucination
if you let it run too far. Length thresholds, refused-prefix detection,
and code-fence stripping put guardrails in place but it's still inherently
risky. Off by default; on for users who want it.

---

## Step 9 — Live preview during ingest

> **Opt-in. Off by default.** Setting: `preview_garbled_recoveries`.

When the user wants to watch this work, the parser saves rendered page
images to `~/.ezrag/preview_cache/` and emits a recovery event payload
that the GUI displays as a live before/after card during ingest. The user
sees the page image + the garbled extraction + the OCR result side-by-side
in real time.

**Why opt-in:** ~50 ms + ~200 KB disk per recovered page. Worth it for
transparency and debugging; pure waste in production runs that work.

---

## The defensive philosophy

Three layers of defense, ordered cheapest-first:

| Path                       | When                                       | Cost                                   |
|----------------------------|--------------------------------------------|----------------------------------------|
| pypdf only                 | Clean PDFs (most of them)                  | Free                                   |
| heuristic + per-page OCR   | Pages with broken fonts (automatic)        | Slow per affected page; rest free      |
| LLM inspect / correct      | User wants paranoid mode (opt-in)          | Slow; transparent; off by default      |

No corpus is uniformly clean. The pipeline has to handle:

- Real text PDFs the user just dropped → fast path
- Scanned PDFs → whole-document OCR
- PDFs with broken subsetted fonts → per-page OCR recovery
- TOC / index pages from OCR → drop, don't pollute
- Anything the heuristic misses → LLM second opinion (when user wants)

---

## Traceability and trust

Every step's output is **traceable**. Each `ParsedSection` carries
metadata such as `{"ocr": True, "reparse": "garbled"}` or
`{"llm_inspect": "partial"}` or `{"llm_corrected": True}`, so after ingest
the user can audit which pages went through which path. Recovery events
are streamed to the UI so the user sees what's happening, not just the
final result.

The user is never silently lied to. If a page can't be recovered, it's
dropped, not pretended-to-have-been-extracted with garbage.

---

## Where the code lives

- [`src/ez_rag/parsers.py`](../src/ez_rag/parsers.py) — parser, heuristics, recovery
- [`src/ez_rag/generate.py`](../src/ez_rag/generate.py) — `inspect_text_quality`, `correct_garbled_text`
- [`src/ez_rag/ingest.py`](../src/ez_rag/ingest.py) — orchestration
- [`tests/test_parsers_garbled.py`](../tests/test_parsers_garbled.py) — heuristic + TOC + table-run assertions
- [`tests/test_llm_inspect.py`](../tests/test_llm_inspect.py) — LLM-inspect path
- [`tests/test_llm_correct.py`](../tests/test_llm_correct.py) — LLM-correct path
- [`bench/generate_pipeline_pdf.py`](../bench/generate_pipeline_pdf.py) — regenerates this doc as a styled PDF

Tests live alongside the implementation; the full suite runs in about a
minute.

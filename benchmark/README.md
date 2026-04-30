# ez-rag benchmark

Reproducible end-to-end benchmark of ez-rag using public, redistributable documents.

## TL;DR

```bash
# Runs today, even before ez-rag is built (uses a built-in BM25 reference engine):
python run_benchmark.py

# Once ez-rag is installed and on PATH:
python run_benchmark.py --engine ez-rag
```

A markdown report and a `summary_*.json` file are written to `reports/`.

## What it measures

For both engines:

- **Top-1 doc match** — was the gold document the highest-ranked result?
- **Recall@5 / Recall@10** — was the gold document in the top-5 / top-10?
- **Substring grounding** — do required answer substrings appear anywhere in the top-5 retrieved chunks?
- **Ingest timings** — per-doc parse time, total parse time, indexing time.

For the OCR path (independent of retrieval):

- **CER** (character error rate) of the OCR output vs. the known string rendered into a synthetic screenshot.

End-to-end LLM grading (faithfulness / answer relevance) is in the plan for the `ez-rag` engine — the reference engine intentionally has no LLM so the harness is runnable on any machine with no GPU.

## Engines

### `--engine reference` (default)

A self-contained pipeline:

- Parsers: `pypdf` for PDFs (falls back to `pdfminer.six` if installed), `bs4` for HTML, raw text for `.txt` / `.csv`, `pytesseract` + Pillow for images.
- Index: pure-Python BM25 over word-tokenized chunks (~220 words, 30-word overlap).
- Retrieval: BM25 only.

Runnable today; validates ingestion + retrieval mechanics. **No GPU, no model downloads.**

Optional packages improve coverage:

```bash
pip install requests pypdf beautifulsoup4 lxml Pillow pytesseract
```

For OCR, install Tesseract:

- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- **macOS**: `brew install tesseract`
- **Linux**: `apt-get install tesseract-ocr`

If a parser dependency is missing, the corresponding documents are reported as parsed-empty and the script continues.

### `--engine ez-rag`

Shells out to `ez-rag` (must be on PATH). The runner:

1. Copies corpus into `<workspace>/docs/`.
2. Runs `ez-rag init` and `ez-rag ingest`.
3. For each question, runs `ez-rag ask "…" --json --top-k K` and parses the result.

This will work once ez-rag's CLI exposes the `--json` retrieval payload described in `PLAN.md` §7.

## Corpus

See `corpus_manifest.json` for the canonical list. All items are public-domain or permissively licensed; the manifest pins URLs and the runner records sha256 hashes per run for reproducibility checking.

| Source | Tests |
|---|---|
| arXiv 1706.03762 (Attention Is All You Need) | PDF text |
| arXiv 1810.04805 (BERT) | PDF text |
| arXiv 2005.11401 (RAG) | long PDF |
| Wikipedia: Retrieval-augmented generation | HTML extraction |
| Wikipedia: Transformer (deep learning architecture) | HTML extraction |
| Project Gutenberg: Pride and Prejudice | long-form TXT |
| US Constitution (archives.gov) | structured PDF |
| IRS Form 1040 instructions | long PDF with tables |
| data.gov sample CSV (synthesized fallback if URL is gated) | CSV |
| Synthetic screenshot rendered locally | OCR |

If a URL becomes unreachable, the runner logs `MISS` for that item and excludes its questions from the run rather than failing.

## Questions

`questions.json` holds 12 curated Q/A pairs. Each entry has:

- `gold_doc_id` — the corpus id where the answer should be retrievable.
- `gold_substrings` — every listed substring must appear in the top-5 retrieved chunks (case-insensitive).
- `any_of` — at least one of these substring groups must also appear (used when there are multiple valid phrasings).

Add or modify questions freely — the runner reads this file each run.

## Outputs

- `reports/report_<engine>_<utc>.md` — human-readable per-run report.
- `reports/summary_<engine>_<utc>.json` — machine-readable summary suitable for CI gating or trend plotting.

## CI usage

```bash
python run_benchmark.py --engine reference
jq '.metrics.recall_at_5_pct >= 80' reports/summary_reference_*.json
```

## Reproducibility notes

- Network access is required on first run to download the corpus; subsequent runs are offline-cached in `<workspace>/corpus/`.
- The runner records sha256 of every file it actually used. If a hash drifts (publisher edited the document), copy the new hash into `corpus_manifest.json` after manual review.
- Tokenization, chunking, and BM25 parameters are fixed in `run_benchmark.py`. Don't tune the reference engine to chase numbers — its job is to be a stable baseline against which the real ez-rag engine is compared.

# sample_data — public-domain RAG corpus

A curated bundle of **factual, freely-usable government documents** for
benchmarking and demoing ez-rag. Multiple formats (PDF, XLSX, CSV,
DOCX) across multiple states. Roughly 100 MB total when fully fetched.

## Quick start

```bash
# Linux / macOS / Windows (Python 3.11+)
python sample_data/fetch.py

# Optional: choose a smaller subset
python sample_data/fetch.py --topic geology
python sample_data/fetch.py --max-mb 30
```

The script reads `sample_data/curation.json` for the file list. Edit
that file to add or remove documents.

## What's included (default curation)

The default set is centered on **multi-state geology** (your example),
plus a broader public-domain mix for format diversity:

### Geology — multi-state (~50 MB)
- **USGS** professional papers and open-file reports covering the
  Appalachian basin, Eastern US groundwater, Mississippi Valley
  mineralization, and Western US tectonics. PDFs.
- **Ohio Geological Survey** publications including statewide
  bedrock geology summaries. PDFs.
- **USGS Mineral Resources Data System (MRDS)** state-level exports
  for Ohio, Pennsylvania, West Virginia, Kentucky, and New York.
  CSVs.

### Energy — multi-state (~20 MB)
- **EIA State Energy Data System** consumption / production /
  expenditure data. XLSX workbooks (1960–present).
- **EIA Annual Energy Outlook** narrative report. PDF.

### Labor & demographics — multi-state (~15 MB)
- **BLS Quarterly Census of Employment and Wages (QCEW)** state-
  level industry data. CSV.
- **Census ACS 5-year estimates** state-level demographic tables.
  CSV.

### Format-diverse converted documents (~10 MB)
- A handful of PDFs converted to DOCX (using `pandoc` if available;
  falls back to text-extraction-into-docx via `python-docx` if not).
- Markdown-formatted versions of selected geology articles.

### Reference text (~5 MB)
- Project Gutenberg geology and natural history texts (public
  domain, US copyright expired). Plain text + EPUB.

## Licensing

**All default sources are public-domain or explicitly free for
unrestricted reuse:**

| Source | Status |
|---|---|
| USGS publications | US federal government work — public domain |
| Ohio Geological Survey | State government open data — public domain |
| EIA datasets | US federal — public domain |
| BLS data | US federal — public domain |
| Census Bureau data | US federal — public domain |
| Project Gutenberg | US copyright expired — public domain |

This means you can ingest, redistribute, transform, and benchmark
against this data with no legal worries. The same is **not** true of
arbitrary third-party PDFs — keep that in mind if you replace items
in `curation.json`.

## File layout after fetch

```
sample_data/
  README.md                 # this file
  curation.json             # the URL list (edit to customize)
  fetch.py                  # cross-platform downloader
  fetched/                  # populated by fetch.py — gitignored
    geology/
      pdf/
        usgs-pp-xxxx.pdf
        oh-bedrock-geology.pdf
        ...
      csv/
        mrds-oh.csv
        mrds-pa.csv
        ...
    energy/
      xlsx/
        eia-seds-summary.xlsx
        ...
    labor/
      csv/
        bls-qcew-2024.csv
        ...
    text/
      gutenberg-geology-text.txt
      ...
    docx/
      converted-from-pdf-1.docx
      ...
  manifest.json             # what was actually fetched, sizes, hashes
```

The `fetched/` subdirectory is in `.gitignore` so 100 MB of binaries
don't end up in your repo. The `curation.json` IS committed so the
recipe is reproducible.

## What this is good for

- **Bench runs.** Point `ez-rag-bench full --corpus sample_data/fetched`
  at it and you have a realistic mixed-format corpus to test against.
- **Demos.** Show the RAG working on real factual documents instead
  of a toy 5-page set.
- **Format coverage.** Exercises every parser ez-rag has — PDF text,
  PDF tables, XLSX, CSV, DOCX, plain text — in one place.
- **Multi-state retrieval testing.** Questions like *"compare coal
  production trends in Ohio and West Virginia"* exercise hybrid
  retrieval across genuinely distinct sources.

## Adding your own data

Edit `curation.json`:

```json
{
  "url": "https://pubs.usgs.gov/of/2023/1234/of2023-1234.pdf",
  "save_as": "geology/pdf/of2023-1234.pdf",
  "format": "pdf",
  "topic": "geology",
  "states": ["WY", "CO"],
  "license": "public-domain",
  "expected_size_mb": 12.4
}
```

Re-run `fetch.py` — already-downloaded files are skipped.

## Notes / caveats

- **URL drift.** Government URLs occasionally move when agencies
  reorganize. If `fetch.py` reports a 404, look up the file's title
  on the agency's current site and update `curation.json`.
- **Bandwidth.** Default fetch is ~100 MB. Use `--max-mb` to cap it
  or `--dry-run` to see what would be fetched.
- **First-run cost.** Total download takes 5–15 minutes on a typical
  connection. Files are SHA-256-checked into `manifest.json` so
  re-fetches verify integrity.

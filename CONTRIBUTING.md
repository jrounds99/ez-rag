# Contributing

ez-rag is an experimental project. The bar for "fix" is low; the bar for "behavior change" is "did you measure it?"

## Quick start for hacking

```bash
git clone https://github.com/<owner>/ez-rag.git
cd ez-rag
python -m pip install --user -e ".[ocr,gui]"
ez-rag init test-workspace
# drop a few PDFs into test-workspace/docs/ then:
cd test-workspace
ez-rag ingest
ez-rag chat
```

## What kind of changes are welcome

- **Bug reports** with a minimal repro. Console log, OS, Python version, what
  command you ran, what you expected, what happened.
- **Bug fixes** — open a PR with a one-paragraph "before/after" describing
  what was broken.
- **Parser additions** for new file types — `parsers.py` is the place. Each
  parser returns `list[ParsedSection]`. Add a test file under `benchmark/` if
  you can.
- **Retrieval improvements** that move a number on `benchmark/bench_configs.py`
  or `benchmark/rag_compare.py`. Please attach the before/after report.
- **GUI polish** — Flet 0.84.x, no other UI deps. New widgets need a tooltip.

## What kind of changes get pushed back

- Adding a SaaS dependency. ez-rag is local-first by design.
- Adding a new optional dep heavier than ~50 MB without a clear reason.
- Replacing the SQLite index with a proper vector DB. *Maybe* eventually
  (LanceDB-shaped seam already exists), but the bar is "fits 100k+ chunks
  noticeably better than what's there."
- Changing the default chat model away from "whatever is in `cfg.llm_model`."
  The chooser is yours; we just ship sensible defaults.

## Style

- Apache-2.0 in spirit. Add a SPDX header to new files if you feel like it.
- Type hints where they pay rent.
- Docstrings on public functions, one-liner is fine.
- Don't add a comment that just restates the next line of code.
- Prefer Python stdlib + the existing minimal dep set. New deps need a reason.

## Running the benchmarks

```bash
python benchmark/run_benchmark.py            # public-doc corpus, end-to-end
python benchmark/rag_compare.py              # RAG-on / RAG-off across models
python benchmark/bench_configs.py            # retrieval-config matrix
```

If your change affects retrieval or generation, run all three. Reports land in
`benchmark/reports/`.

## Filing issues

Templates live under `.github/ISSUE_TEMPLATE/` (when we get around to writing
them). For now: clear repro + your environment is plenty.

## Note on co-authorship

Most of this code was typed by language models under the direction of a
patient human ([Justin Rounds](https://www.justinrounds.com)). If you submit a
PR drafted with help from a model, that's fine — just make sure you understand
every line and that it builds, runs, and passes the benchmarks before you hit
"open PR."

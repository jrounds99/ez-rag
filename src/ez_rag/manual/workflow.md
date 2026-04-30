# Workflow

The 60-second path from "I have a folder of documents" to "I'm chatting with them."

```
docs/  ─────────►  parse + chunk  ─────►  embed  ─────►  vector index (SQLite)
                                                              │
                                                              ▼
question  ───►  hybrid search ──►  rerank  ──►  context  ──►  LLM  ──►  answer + citations
                  (BM25 + dense)    (cross-     (top-K)        (Ollama
                                     encoder)                    or llama-cpp)
```

## 1. Make a workspace

A workspace is just a folder with a `docs/` directory inside it.

```bash
mkdir my-rag && cd my-rag
ez-rag init .
# or in the GUI: Open workspace → pick or create a folder
```

`ez-rag init` creates `docs/`, `.ezrag/config.toml`, and an empty SQLite index.

## 2. Drop documents in `docs/`

Anything ez-rag can parse:

PDF (text + scanned via OCR), DOCX, XLSX, CSV, HTML, MD, TXT/RST/LOG, EPUB,
EML, PNG / JPG / WEBP / TIFF / BMP (OCR'd).

Drag-and-drop into the docs/ folder works. So does the GUI's
**Files → Add files / Add folder** which copies into `docs/` for you.

## 3. Ingest

```bash
ez-rag ingest      # GUI: Files → Ingest
```

What happens:

1. **Detect** the file type and pick a parser.
2. **Parse** to plain text, preserving page numbers / sections / table layout
   where possible. PDFs auto-fall-back to OCR if extracted text density is too
   low. Images go through OCR.
3. **Chunk** ~512 tokens with 64-token overlap; structure-aware where the
   source provides headings.
4. *(optional)* **Contextual Retrieval**: prepend a 1-sentence summary of
   where this chunk lives in the document before embedding. Slower ingest,
   materially better recall.
5. **Embed** each chunk with the active embedder (Ollama's
   `nomic-embed-text` if reachable, else fastembed's `BAAI/bge-small-en-v1.5`).
6. **Index** into SQLite — embeddings as float32 BLOBs, plus FTS5 for BM25.

Re-running `ingest` only processes files that are new or have changed
(tracked by sha256). Force a rebuild with `ingest --force` (or **Re-ingest
(force)** in the GUI).

## 4. Ask

```bash
ez-rag ask "your question"
ez-rag chat                    # multi-turn REPL
ez-rag-gui                     # desktop app, Chat tab
```

Per-question pipeline:

1. *(optional)* **HyDE**: have the LLM write a hypothetical answer first;
   embed THAT instead of the bare question. Embeddings of an answer match the
   corpus phrasing better than embeddings of a question.
2. *(optional)* **Multi-query**: ask the LLM for 2 paraphrases; retrieve for
   each; fuse with reciprocal rank fusion.
3. **Hybrid search**: BM25 keyword search + dense cosine, fused with
   reciprocal rank fusion (RRF). Returns the top-30 candidates.
4. **Rerank** *(default ON)*: a small cross-encoder (~23 MB) scores each
   `(query, candidate)` pair jointly. Almost always the single biggest
   quality win; ~50–200 ms.
5. **Generate**: the top-K (default 8) passages are prepended to the question
   as numbered context items. The LLM answers and cites `[1]`, `[2]`, …
6. *(optional)* When **Use corpus** is OFF, retrieval is skipped entirely and
   the LLM answers from its own knowledge — useful for A/B-comparing whether
   the corpus is actually helping.

## 5. Tweak (or don't)

The defaults are tuned for "drop docs, get good answers" — `hybrid` and
`rerank` ON, everything else off. Three knobs are worth knowing:

| Setting | When to enable |
|---|---|
| **HyDE** | Vague / short questions, or questions that don't share vocabulary with the corpus. |
| **Multi-query** | Same as HyDE, plus when one question could be asked many ways. |
| **Contextual Retrieval** | Larger / structured corpora where chunks lose meaning out of context (technical docs, code, legal text). Slow ingest. |

See [retrieval](retrieval) for the full menu and what each option costs.

## 6. Run the benchmark

A reproducible RAG-on / RAG-off comparison harness lives in
`benchmark/rag_compare.py`:

```bash
python benchmark/rag_compare.py --quick        # 2 loops
python benchmark/rag_compare.py                # 4 loops
```

Reports `reports/rag_compare.md` with per-question accuracy and timing for
each model, and a JSON dump for further analysis.

## When things look wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Empty / truncated answer with deepseek-r1 | Reasoning ate `max_tokens` | Bump `max_tokens` to 4096+ |
| Bubble looks frozen during reasoning | Normal for deepseek-r1 | The "Reasoning…" panel shows live token count; wait for `content` phase |
| Wrong file ranked first | RAG with too-similar chunks | Enable **Rerank** (default), try **HyDE**, raise **Top-K** to 12 |
| Answer ignores corpus | Vector / keyword mismatch | Try **Multi-query**, lower temperature to 0.0–0.1 |
| Hallucinated `[1]` in no-RAG mode | (Was a bug) | Already fixed: no-RAG uses a separate system prompt that forbids citation markers |
| Image / scanned text not picked up | OCR not installed or off | Settings → enable OCR; check **Doctor** for RapidOCR / Tesseract |

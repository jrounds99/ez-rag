# Ingestion

```bash
ez-rag ingest                # incremental
ez-rag ingest --force        # reingest everything
ez-rag ingest --watch        # poll docs/ and reingest on change
ez-rag reindex               # drop the index, rebuild from docs/
```

Files are tracked by sha256. An unchanged file is skipped on the next ingest.

## Supported formats

| Extension | Notes |
|---|---|
| `.pdf` | Text + scanned (auto OCR fallback when char density is low) |
| `.docx` | Headings preserved as section markers; tables linearized |
| `.xlsx`, `.xlsm` | One section per sheet, rows joined with ` \| ` |
| `.csv`, `.tsv` | Flat rows |
| `.html`, `.htm`, `.xhtml` | Boilerplate (nav/header/footer/script/style) stripped |
| `.md`, `.markdown`, `.txt`, `.rst`, `.log` | Read directly |
| `.epub` | Per-chapter |
| `.eml` | Plain text body preferred; HTML body stripped |
| `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp` | OCR'd |

## Tweaking

Edit `.ezrag/config.toml`:

```toml
chunk_size = 512
chunk_overlap = 64
enable_ocr = true
ocr_provider = "auto"          # auto | rapidocr | tesseract | none
embedder_provider = "auto"     # auto | ollama | fastembed
embedder_model = "BAAI/bge-small-en-v1.5"
ollama_embed_model = "nomic-embed-text"
```

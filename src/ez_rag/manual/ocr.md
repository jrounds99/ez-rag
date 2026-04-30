# OCR (screenshots and scanned PDFs)

ez-rag OCR's image files (`.png`, `.jpg`, `.webp`, `.tiff`, `.bmp`) and scanned
PDFs automatically.

## Engine selection

It tries in this order:

1. **RapidOCR** — `pip install ez-rag[ocr]`. ONNX-based, ~80 MB, no system deps.
2. **Tesseract** — fallback if `tesseract` is on PATH and `pytesseract` is installed.
3. otherwise images are skipped silently.

Check what's available:

```bash
ez-rag doctor
```

## Scanned PDFs

If a PDF has < 50 chars per page on average, ez-rag re-renders the pages with
pypdfium2 and OCRs them. No flag needed.

## Languages

RapidOCR auto-detects script via its text-detection model. For Tesseract,
install language packs at the OS level:

* Ubuntu: `apt install tesseract-ocr-fra tesseract-ocr-deu` (etc.)
* macOS: `brew install tesseract-lang`
* Windows: bundled into the Tesseract installer

Then set in `.ezrag/config.toml`:

```toml
ocr_provider = "tesseract"
```

(per-language config is on the roadmap; today RapidOCR's auto-detect handles
most cases without flags.)

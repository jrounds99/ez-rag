"""Document parsers. Each returns a list[ParsedSection] which the chunker consumes.

A ParsedSection is text plus optional location metadata (page, section path).
We deliberately keep parsers in pure Python with light deps.
"""
from __future__ import annotations

import csv
import email
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ----- types -----------------------------------------------------------------

@dataclass
class ParsedSection:
    text: str
    page: int | None = None
    section: str = ""
    meta: dict = field(default_factory=dict)


# ----- registry --------------------------------------------------------------

ParserFn = Callable[[Path], list[ParsedSection]]
_REGISTRY: dict[str, ParserFn] = {}

def register(*exts: str):
    def deco(fn: ParserFn) -> ParserFn:
        for e in exts:
            _REGISTRY[e.lower()] = fn
        return fn
    return deco

def get_parser(path: Path) -> ParserFn | None:
    return _REGISTRY.get(path.suffix.lower())

def supported_extensions() -> set[str]:
    return set(_REGISTRY.keys())


# ----- helpers ---------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----- TXT / MD / RST --------------------------------------------------------

@register(".txt", ".md", ".markdown", ".rst", ".log")
def parse_text(path: Path) -> list[ParsedSection]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="replace")
    return [ParsedSection(text=_normalize(text))]


# ----- PDF -------------------------------------------------------------------

@register(".pdf")
def parse_pdf(path: Path) -> list[ParsedSection]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise RuntimeError("pypdf not installed; required for PDF parsing") from e
    sections: list[ParsedSection] = []
    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        page_texts.append(t)
        if t.strip():
            sections.append(ParsedSection(text=_normalize(t), page=i))

    # If we extracted almost nothing, the PDF is likely scanned. Fall back to
    # OCR over rendered pages, lazily importing OCR + a renderer.
    total_chars = sum(len(t) for t in page_texts)
    if total_chars < 50 * max(1, len(reader.pages)):
        ocr_sections = _ocr_pdf_pages(path)
        if ocr_sections:
            return ocr_sections
    return sections


def _ocr_pdf_pages(path: Path) -> list[ParsedSection]:
    """Render PDF pages to images and OCR them. Returns [] if deps missing."""
    try:
        from pypdf import PdfReader  # noqa: F401  (already required)
    except ImportError:
        return []
    # Try pypdfium2 (Apache + BSD) first; it's the lightest renderer.
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        return []
    from .ocr import ocr_image  # local import to avoid circular at top
    sections: list[ParsedSection] = []
    pdf = pdfium.PdfDocument(str(path))
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=2.0)
        pil = bitmap.to_pil()
        text = ocr_image(pil) or ""
        if text.strip():
            sections.append(ParsedSection(text=_normalize(text), page=i + 1, meta={"ocr": True}))
    return sections


# ----- DOCX ------------------------------------------------------------------

@register(".docx")
def parse_docx(path: Path) -> list[ParsedSection]:
    try:
        import docx  # type: ignore  (python-docx)
    except ImportError as e:
        raise RuntimeError("python-docx not installed; required for DOCX") from e
    doc = docx.Document(str(path))
    sections: list[ParsedSection] = []
    current_heading = ""
    buf: list[str] = []
    def flush():
        if buf:
            sections.append(ParsedSection(text=_normalize("\n".join(buf)),
                                          section=current_heading))
            buf.clear()
    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            flush()
            current_heading = para.text.strip()
            if current_heading:
                buf.append(f"# {current_heading}")
        else:
            if para.text.strip():
                buf.append(para.text)
    # Tables
    for table in doc.tables:
        rows = []
        for row in table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        if rows:
            buf.append("\n".join(rows))
    flush()
    return sections


# ----- XLSX / CSV ------------------------------------------------------------

@register(".xlsx", ".xlsm")
def parse_xlsx(path: Path) -> list[ParsedSection]:
    try:
        import openpyxl  # type: ignore
    except ImportError as e:
        raise RuntimeError("openpyxl not installed; required for XLSX") from e
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    sections: list[ParsedSection] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v) for v in row]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            sections.append(ParsedSection(
                text=_normalize("\n".join(rows)),
                section=sheet_name,
            ))
    return sections


@register(".csv", ".tsv")
def parse_csv(path: Path) -> list[ParsedSection]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            rows.append(" | ".join(row))
    return [ParsedSection(text=_normalize("\n".join(rows)))]


# ----- HTML ------------------------------------------------------------------

@register(".html", ".htm", ".xhtml")
def parse_html(path: Path) -> list[ParsedSection]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as e:
        raise RuntimeError("beautifulsoup4 not installed; required for HTML") from e
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml" if _has_lxml() else "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    text = soup.get_text("\n", strip=True)
    return [ParsedSection(text=_normalize(text))]


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


# ----- EPUB (handled by stdlib zipfile) --------------------------------------

@register(".epub")
def parse_epub(path: Path) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        BeautifulSoup = None  # type: ignore
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        names.sort()
        for name in names:
            try:
                raw = z.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            if BeautifulSoup is not None:
                soup = BeautifulSoup(raw, "lxml" if _has_lxml() else "html.parser")
                for t in soup(["script", "style", "nav"]):
                    t.decompose()
                text = soup.get_text("\n", strip=True)
            else:
                text = re.sub(r"<[^>]+>", " ", raw)
            text = _normalize(text)
            if text:
                sections.append(ParsedSection(text=text, section=name))
    return sections


# ----- EML / MBOX ------------------------------------------------------------

@register(".eml")
def parse_eml(path: Path) -> list[ParsedSection]:
    msg = email.message_from_bytes(path.read_bytes())
    parts: list[str] = []
    subj = msg.get("subject", "")
    frm = msg.get("from", "")
    to = msg.get("to", "")
    parts.append(f"Subject: {subj}\nFrom: {frm}\nTo: {to}")
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                parts.append(payload.decode(part.get_content_charset() or "utf-8",
                                            errors="replace"))
        elif ctype == "text/html":
            payload = part.get_payload(decode=True)
            if payload and not any(p for p in parts[1:]):
                try:
                    from bs4 import BeautifulSoup  # type: ignore
                    soup = BeautifulSoup(payload, "lxml" if _has_lxml() else "html.parser")
                    parts.append(soup.get_text("\n", strip=True))
                except ImportError:
                    parts.append(re.sub(r"<[^>]+>", " ",
                                        payload.decode("utf-8", errors="replace")))
    return [ParsedSection(text=_normalize("\n\n".join(parts)))]


# ----- Images (delegate to OCR module) --------------------------------------

@register(".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp")
def parse_image(path: Path) -> list[ParsedSection]:
    from .ocr import ocr_file
    text = ocr_file(path)
    if not text.strip():
        return []
    return [ParsedSection(text=_normalize(text), meta={"ocr": True})]

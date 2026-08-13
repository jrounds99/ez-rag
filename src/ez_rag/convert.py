"""Legacy Office format conversion — .doc / .xls / .ppt → modern XML.

The pre-2007 binary formats have no good pure-Python parsers, but two
free converters are commonly available:

  1. **LibreOffice** (`soffice --headless --convert-to …`) — preferred:
     free, cross-platform, no Python dependency, handles all three.
  2. **Microsoft Office COM** (Windows + Office + `pip install pywin32`)
     — fallback when LibreOffice isn't installed but Office is.

Conversions are cached under `~/.ezrag/convert_cache/` keyed by the
source file's sha256, so a legacy file is converted once ever (until
its content changes). The registered .doc/.xls/.ppt parsers call
`convert_legacy()` and then delegate to the modern-format parser, so
ingest, citations, and provenance all attribute to the ORIGINAL file.

If no backend is available, conversion raises a RuntimeError with
install instructions — ingest records the file as errored with that
message rather than silently skipping it.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

TARGETS = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}

_SOFFICE_CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/opt/homebrew/bin/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
]

# COM SaveAs format codes (Word / Excel / PowerPoint).
_COM_APPS = {
    ".doc": ("Word.Application", "Documents", 16),      # wdFormatXMLDocument
    ".xls": ("Excel.Application", "Workbooks", 51),     # xlOpenXMLWorkbook
    ".ppt": ("PowerPoint.Application", "Presentations", 24),  # ppSaveAsOpenXMLPresentation
}


def find_soffice() -> str | None:
    exe = shutil.which("soffice")
    if exe:
        return exe
    for cand in _SOFFICE_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def com_available() -> bool:
    try:
        import win32com.client  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def converter_available() -> bool:
    return find_soffice() is not None or com_available()


def _cache_dir() -> Path:
    d = Path.home() / ".ezrag" / "convert_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _convert_soffice(src: Path, target_ext: str, out_path: Path) -> None:
    soffice = find_soffice()
    assert soffice
    # Convert into a private temp dir (soffice names the output after the
    # source stem — a shared dir would collide on same-named files).
    with tempfile.TemporaryDirectory(prefix="ezrag_conv_") as td:
        proc = subprocess.run(
            [soffice, "--headless", "--norestore",
             "--convert-to", target_ext.lstrip("."),
             "--outdir", td, str(src)],
            capture_output=True, text=True, timeout=180,
        )
        produced = Path(td) / (src.stem + target_ext)
        if proc.returncode != 0 or not produced.is_file():
            raise RuntimeError(
                f"LibreOffice conversion failed for {src.name}: "
                f"{(proc.stderr or proc.stdout or '').strip()[:300]}"
            )
        shutil.move(str(produced), str(out_path))


def _convert_com(src: Path, target_ext: str, out_path: Path) -> None:
    import win32com.client  # type: ignore
    try:
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()   # parsers may run off the main thread
    except Exception:
        pass
    prog_id, collection, fmt = _COM_APPS[src.suffix.lower()]
    app = win32com.client.Dispatch(prog_id)
    try:
        # Word/Excel have a Visible property; PowerPoint errors on it.
        try:
            app.Visible = False
        except Exception:
            pass
        docs = getattr(app, collection)
        doc = docs.Open(str(src.resolve()))
        try:
            if prog_id.startswith("Word"):
                doc.SaveAs2(str(out_path.resolve()), FileFormat=fmt)
            else:
                doc.SaveAs(str(out_path.resolve()), fmt)
        finally:
            doc.Close(False)
    finally:
        try:
            app.Quit()
        except Exception:
            pass


def convert_legacy(path: Path) -> Path:
    """Convert a legacy Office file to its modern equivalent.

    Returns the cached converted file. Raises RuntimeError (with install
    guidance) when no conversion backend is available or the conversion
    fails.
    """
    ext = path.suffix.lower()
    target_ext = TARGETS.get(ext)
    if target_ext is None:
        raise ValueError(f"Not a legacy Office format: {path.name}")

    from .index import file_sha256
    sha = file_sha256(path)[:16]
    out_path = _cache_dir() / f"{sha}{target_ext}"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    if find_soffice() is not None:
        _convert_soffice(path, target_ext, out_path)
        return out_path
    if com_available():
        _convert_com(path, target_ext, out_path)
        return out_path
    raise RuntimeError(
        f"Cannot convert {path.name}: no converter available. Install "
        f"LibreOffice (free, libreoffice.org) — or, with Microsoft "
        f"Office installed, `pip install pywin32`. Alternatively save "
        f"the file as {target_ext} yourself and re-ingest."
    )

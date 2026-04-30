"""OCR pipeline.

Tries RapidOCR first (self-contained ONNX models, ~80MB), then Tesseract
(if `tesseract` is on PATH and pytesseract installed). Returns "" on failure.
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _rapidocr() -> Any | None:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        return RapidOCR()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def ocr_file(path: Path) -> str:
    """OCR an image file from disk."""
    engine = _rapidocr()
    if engine is not None:
        try:
            result, _ = engine(str(path))
            if result:
                return _join_lines(result)
        except Exception:
            pass
    if _has_tesseract():
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            return pytesseract.image_to_string(Image.open(path))
        except Exception:
            return ""
    return ""


def ocr_image(image: Any) -> str:
    """OCR a PIL.Image. Used by the scanned-PDF code path."""
    engine = _rapidocr()
    if engine is not None:
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(image)
            result, _ = engine(arr)
            if result:
                return _join_lines(result)
        except Exception:
            pass
    if _has_tesseract():
        try:
            import pytesseract  # type: ignore
            return pytesseract.image_to_string(image)
        except Exception:
            return ""
    return ""


def _join_lines(rapidocr_result) -> str:
    """RapidOCR returns [(box, text, conf), ...]. Sort top-to-bottom, left-to-right."""
    items = []
    for entry in rapidocr_result:
        if len(entry) < 2:
            continue
        box, text = entry[0], entry[1]
        try:
            ys = [pt[1] for pt in box]
            xs = [pt[0] for pt in box]
            y, x = sum(ys) / len(ys), min(xs)
        except Exception:
            y, x = 0, 0
        items.append((y, x, text))
    items.sort(key=lambda t: (round(t[0] / 12), t[1]))
    return "\n".join(t for _, _, t in items)


def status() -> dict:
    """Used by `ez-rag doctor` to report what's available."""
    return {
        "rapidocr": _rapidocr() is not None,
        "tesseract": _has_tesseract(),
    }

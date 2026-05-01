"""Tests for the garbled-PDF-text detector + recovery pipeline.

Heuristic-only — verifies _text_looks_garbled flags the kinds of broken
font-cmap output users hit in the wild and doesn't false-positive on
short headers, normal prose, or punctuation-heavy code/spec text.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.parsers import _text_looks_garbled


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    print("\n[1] real-world garbled extraction is flagged")
    # The screenshot the user showed me — exact text from a broken PDF.
    sample1 = (
        "hAppe�d to the \\pell\\ Mor&e�kAi�e�'l "
        "Bo1A�tif1AI BAck-PAtti��, l/ewAr&'\\ I/qt "
        "Air, A�& All the re\\t ? l'M wre I \\IAbMitte& the "
        "\\pell\\ they i� \\i\\td l i�cl!A&e herei�. "
        "SeeM\\ they �ot (o\\t i� the \\h1Affle. ShAMe."
    )
    check("font-cmap garbage flagged",
          _text_looks_garbled(sample1) is True,
          f"expected True, got False on: {sample1[:60]!r}")

    print("\n[2] heavy-replacement-char text is flagged")
    sample2 = "Hello ��� world ��� " * 6
    check("replacement-heavy text flagged",
          _text_looks_garbled(sample2) is True, "")

    print("\n[3] heavy-backslash-escape text is flagged")
    sample3 = " ".join([
        "this \\page\\ has \\many\\ unmapped \\glyphs\\ in",
        "every \\sentence\\ which \\should\\ trigger \\detection\\",
        "without \\question\\ from \\heuristic\\ logic \\here\\.",
    ])
    check("backslash-heavy text flagged",
          _text_looks_garbled(sample3) is True, "")

    print("\n[4] consonant-only nonsense is flagged")
    # 4% vowels — well below the floor
    sample4 = ("zxcv bnm qwrt yphjkl mnbv cxz xcvb nm qwrtypsdfghjklzxcvbnm "
               "qwrtypsdfghjklzxcvbnm qwrtypsdfghjklzxcvbnm")
    check("very low vowel ratio flagged",
          _text_looks_garbled(sample4) is True, "")

    print("\n[5] normal English prose is NOT flagged (no false positive)")
    sample5 = (
        "Border collies are widely considered the most intelligent breed "
        "of dog. They were bred for herding sheep in the Anglo-Scottish "
        "border region. Their stare, called the eye, is a key herding "
        "tool. Border collies require significant mental stimulation; "
        "without it they may develop neurotic behaviors. The breed has "
        "consistently topped canine intelligence rankings since the 1990s."
    )
    check("clean English prose NOT flagged",
          _text_looks_garbled(sample5) is False,
          "false positive on normal English")

    print("\n[6] short headers are NOT flagged (under threshold)")
    for header in (
        "Chapter 1",
        "TABLE OF CONTENTS",
        "Spells",
        "[1]",
        "p. 247",
    ):
        check(f"short header NOT flagged: {header!r}",
              _text_looks_garbled(header) is False, "")

    print("\n[7] code / spec text with normal punctuation NOT flagged")
    sample7 = (
        "def parse_pdf(path: Path, on_progress=None) -> list[ParsedSection]:\n"
        "    sections = []\n"
        "    reader = PdfReader(str(path))\n"
        "    total = len(reader.pages)\n"
        "    for i, page in enumerate(reader.pages, start=1):\n"
        "        text = page.extract_text() or ''\n"
        "        sections.append(ParsedSection(text=text, page=i))\n"
        "    return sections\n"
    )
    check("Python code NOT flagged",
          _text_looks_garbled(sample7) is False,
          "false positive on Python source")

    print("\n[8] empty / whitespace string returns False (not garbled)")
    check("empty string -> False", _text_looks_garbled("") is False, "")
    check("whitespace -> False", _text_looks_garbled("   \n\n  ") is False, "")
    check("None-style 'no text' -> False",
          _text_looks_garbled(None or "") is False, "")

    print(f"\n=== Garbled-detector summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

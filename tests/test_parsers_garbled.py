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

from ez_rag.parsers import (
    _collapse_table_runs, _looks_like_toc_fragment, _text_looks_garbled,
)


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

    print("\n[9] table-column flattening (the 'No, No, No' case) collapses")
    # Reproduces the screenshot: a "Ritual" column extracted as a sequence
    # of identical short lines. This is what we want to compress.
    sample9 = (
        "The Spells table lists the new spells.\n"
        "Ritual\n"
        + "No\n" * 18
        + "Yes\n"
        + "No\n" * 5
    )
    out9 = _collapse_table_runs(sample9)
    check("18× 'No' run collapsed to single line",
          "No (×18)" in out9, f"got:\n{out9}")
    check("intervening 'Yes' preserved",
          "Yes" in out9, "")
    # The trailing 5× 'No' is below the threshold (need 6+) — left alone
    n_lone_no = sum(1 for line in out9.split("\n") if line.strip() == "No")
    check("sub-threshold run NOT collapsed",
          n_lone_no == 5, f"expected 5 standalone 'No' lines; got {n_lone_no}\n{out9}")

    print("\n[10] long-line repetition is NOT collapsed (only short cells)")
    sample10 = (
        "The quick brown fox jumps over the lazy dog.\n" * 8
    )
    out10 = _collapse_table_runs(sample10)
    check("long-line repetition left alone",
          "(×" not in out10, f"unexpectedly collapsed: {out10!r}")

    print("\n[11] sub-threshold short-line repetition is NOT collapsed")
    sample11 = "OK\nOK\nOK\nOK\n"   # 4 reps — under the 6 threshold
    out11 = _collapse_table_runs(sample11)
    check("4× short-line repetition left alone",
          "(×" not in out11, f"unexpectedly collapsed: {out11!r}")

    print("\n[12] mixed page with one bad column doesn't lose normal prose")
    sample12 = (
        "Border collies are remarkably intelligent dogs.\n"
        "They were bred for herding sheep.\n"
        "Ritual\n"
        + "No\n" * 10 +
        "Their stare is called the eye.\n"
    )
    out12 = _collapse_table_runs(sample12)
    check("normal prose preserved across collapsed run",
          "Border collies" in out12 and "stare is called" in out12,
          f"got:\n{out12}")
    check("the run was collapsed",
          "No (×10)" in out12, "")

    print("\n[13] empty / single-line input — no crashes")
    check("empty -> empty",
          _collapse_table_runs("") == "", "")
    check("single line -> unchanged",
          _collapse_table_runs("just one line") == "just one line", "")

    print("\n[14] TOC-fragment OCR results are flagged")
    # Reproduces the user's screenshot: an OCR'd table-of-contents page
    # that's still useless even though it's not garbled. Should be
    # dropped during recovery.
    sample14 = (
        "CONTENTS\n"
        "Preface\n"
        "Fighter\n"
        "59\n"
        "App.A: AcQ INc.\n"
        "196\n"
        "Ch.l:Acquisitions Incorporated.\n"
        "Monk\n"
        ".61\n"
        "Omin Dran.\n"
        "..196\n"
        "It'sJustBusiness.\n"
    )
    check("user-reported TOC OCR fragment flagged",
          _looks_like_toc_fragment(sample14) is True,
          f"got False on:\n{sample14}")

    print("\n[15] normal prose is NOT flagged as TOC")
    sample15 = (
        "Border collies were bred for herding sheep in the "
        "Anglo-Scottish border region. They are widely considered "
        "the most intelligent breed of dog, consistently topping "
        "canine intelligence rankings. Their stare, called the eye, "
        "is a key herding tool — they use it to control sheep "
        "without barking. Border collies require significant mental "
        "stimulation; without it they may develop neurotic behaviors."
    )
    check("normal prose NOT flagged as TOC",
          _looks_like_toc_fragment(sample15) is False, "")

    print("\n[16] short content is NOT flagged (need at least 6 lines)")
    sample16 = "Chapter 1\nIntro\nFighter\n59"
    check("short input NOT flagged",
          _looks_like_toc_fragment(sample16) is False,
          f"got True on: {sample16!r}")

    print("\n[17] bullet list with prose is NOT flagged")
    sample17 = (
        "Things to remember about border collies:\n"
        "- They need lots of exercise every day\n"
        "- Mental stimulation matters more than physical\n"
        "- They will herd children and cats given the chance\n"
        "- Their bark is rare but loud\n"
        "- They form strong bonds with one human\n"
        "- They were bred from working stock in the borderlands\n"
    )
    check("prose-y bullet list NOT flagged",
          _looks_like_toc_fragment(sample17) is False, "")

    print("\n[18] index-like page with many bare numbers IS flagged")
    sample18 = (
        "INDEX\n"
        "Tyrannosaurus rex\n"
        "47\n"
        "Triceratops\n"
        "82\n"
        "Stegosaurus\n"
        "115\n"
        "Allosaurus\n"
        "39\n"
        "Diplodocus\n"
        "201\n"
    )
    check("dinosaur index page flagged",
          _looks_like_toc_fragment(sample18) is True,
          f"got False on:\n{sample18}")

    print(f"\n=== Garbled-detector summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

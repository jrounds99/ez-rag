"""Glossary / acronym-index tests.

Covers:
  1. Acronym extraction with min-occurrence filter + well-known skip.
  2. In-corpus definition pairing, both directions, initials-validated
     (incl. stopword-skipping: "Department of Defense (DOD)").
  3. False-definition rejection (initials don't match).
  4. SKU detection + vendor adjacency; SKUs excluded from acronyms.
  5. Offline external resolution (no network): Wikipedia search links
     labeled unverified; vendor search links.
  6. End-to-end against a real ingested workspace: HTML report exists,
     is self-contained (no CDN), shows in-corpus source; JSON valid;
     add_to_corpus writes docs/_glossary.md.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.glossary import (
    build_glossary, extract_entries, render_html, resolve_external,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


ROWS = [
    ("Retrieval-Augmented Generation (RAG) improves grounding. "
     "RAG systems retrieve before answering. RAG is standard now.",
     "docs/intro.md", 1),
    ("The DOD (Department of Defense) budget grew. DOD manages bases. "
     "Compare MTBF numbers: MTBF is mean time between failures jargon.",
     "docs/gov.md", 2),
    ("We deployed the NVIDIA RTX 5090 cluster. The RTX 5090 outperforms "
     "the A770 in our tests. Intel A770 pricing varies. QQQ QQQ.",
     "docs/hw.md", 3),
    ("PDF exports work. USA-based team. False positive: General "
     "Electric (WRONG) should not pair since initials mismatch.",
     "docs/misc.md", 4),
]


def main():
    print("\n[1] extraction basics")
    entries = extract_entries(ROWS, min_occurrences=2)
    by_term = {e.term: e for e in entries}
    check("RAG found", "RAG" in by_term)
    check("MTBF found (2 uses)", "MTBF" in by_term
          and by_term["MTBF"].occurrences == 2)
    check("well-known PDF/USA skipped",
          "PDF" not in by_term and "USA" not in by_term)
    check("QQQ included at min_occurrences=2",
          "QQQ" in by_term)
    check("occurrence counting", by_term["RAG"].occurrences == 3,
          f"{by_term['RAG'].occurrences}")

    print("\n[2] in-corpus definitions")
    check("expansion (ACRO) direction",
          by_term["RAG"].defined_in_corpus
          and by_term["RAG"].definition == "Retrieval-Augmented Generation",
          f"{by_term['RAG'].definition!r}")
    check("RAG source recorded", "docs/intro.md" in by_term["RAG"].source)
    check("ACRO (expansion) + stopword skip: DOD",
          by_term["DOD"].defined_in_corpus
          and "Department of Defense" in by_term["DOD"].definition,
          f"{by_term['DOD'].definition!r}")
    check("initials-mismatch rejected",
          "WRONG" not in by_term
          or not by_term["WRONG"].defined_in_corpus)

    print("\n[3] SKUs + vendor adjacency")
    skus = {e.term: e for e in entries if e.kind == "sku"}
    check("RTX 5090 code detected as SKU ('5090' has no letters — "
          "expect '5090' skipped, 'A770' found)",
          "A770" in skus, f"{list(skus)}")
    check("A770 vendor = intel", skus.get("A770") and
          skus["A770"].vendor == "intel", f"{skus.get('A770')}")
    check("SKU not double-counted as acronym",
          "A770" not in [e.term for e in entries if e.kind == "acronym"])

    print("\n[4] offline external resolution (no network)")
    resolve_external(entries, allow_web=False)
    undefined = [e for e in entries
                 if e.kind == "acronym" and not e.defined_in_corpus]
    check("undefined acronyms got Wikipedia links",
          all("wikipedia.org" in e.external_url for e in undefined),
          f"{[(e.term, e.external_url) for e in undefined][:3]}")
    check("offline links labeled unverified",
          all("unverified" in e.external_label for e in undefined))
    check("defined entries got NO external link",
          not by_term["RAG"].external_url)
    check("SKU links vendor-scoped",
          "site%3Aintel.com" in skus["A770"].external_url,
          skus["A770"].external_url)

    print("\n[5] HTML render")
    html_text = render_html(entries, workspace_name="testws",
                             allow_web=False)
    check("self-contained (no external assets)",
          "cdn." not in html_text and "<script src" not in html_text)
    check("shows in-corpus source", "docs/intro.md" in html_text)
    check("shows external link", "wikipedia.org" in html_text)
    check("offline note present", "OFFLINE" in html_text)

    print("\n[6] end-to-end on a real workspace")
    from ez_rag.workspace import Workspace
    from ez_rag.ingest import ingest

    tmp = Path(tempfile.mkdtemp(prefix="ezrag_gloss_"))
    ws = Workspace(tmp)
    ws.initialize()
    (ws.docs_dir / "spec.md").write_text(
        "Mean Time Between Failures (MTBF) is our key metric. MTBF "
        "targets improved. The API gateway uses the API twice. "
        "Vendor: Nvidia RTX-5090A units shipped. RTX-5090A stock is low.",
        encoding="utf-8")
    cfg = ws.load_config()
    cfg.embedder_provider = "fastembed"
    ingest(ws, cfg=cfg)

    summary = build_glossary(tmp, allow_web=False, min_occurrences=2,
                              add_to_corpus=True)
    check("entries found", summary["entries"] >= 2, f"{summary}")
    check("MTBF defined in corpus", summary["defined_in_corpus"] >= 1)
    html_p = Path(summary["html"])
    check("HTML report written", html_p.is_file())
    body = html_p.read_text(encoding="utf-8")
    check("report shows definition source",
          "MTBF" in body and "spec.md" in body)
    data = json.loads(Path(summary["json"]).read_text(encoding="utf-8"))
    check("JSON valid + populated", isinstance(data, list) and len(data) >= 2)
    md = tmp / "docs" / "_glossary.md"
    check("add_to_corpus wrote docs/_glossary.md",
          md.is_file() and "MTBF" in md.read_text(encoding="utf-8"))

    print(f"\n=== glossary summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

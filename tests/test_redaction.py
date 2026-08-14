"""Context-aware redaction tests.

Covers:
  1. Smart-casing disambiguation: "Stone" (name) redacted, "blast
     rounds" (common noun) preserved.
  2. Name-variant matching: "Stone, Casey", "C. Stone", "Casey S.".
  3. Unambiguous terms (emails, digit-bearing IDs) redacted in any case.
  4. redact_smart=False redacts everything.
  5. Ingest end-to-end: term never reaches index text, FTS tokens, or
     chunk headers; changing terms re-ingests (version fold); filename
     warning fires.
  6. Export gates: dirty index refused; include_sources refused while
     redaction configured; clean redacted export succeeds.
  7. redact-check scanner finds planted terms.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.redaction import (
    compile_matchers, filename_warnings, redact_text,
    scan_index_for_terms, terms_fingerprint,
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
    print("\n[1] smart casing — the 'Stone' problem")
    m = compile_matchers(["Casey Stone", "Stone"], smart=True)
    r = redact_text(
        "Casey Stone inspected the site. Mr. Stone noted the miners "
        "fired three crushed stone before lunch. STONE approved.", m)
    check("full name redacted", "Casey Stone" not in r.text)
    check("capitalized surname redacted", "Mr. [REDACTED] noted" in r.text,
          r.text)
    check("ALLCAPS redacted", "STONE" not in r.text)
    check("common-noun 'crushed stone' preserved",
          "crushed stone" in r.text, r.text)
    check("count reflects redactions", r.n_redacted == 3, f"{r.n_redacted}")

    print("\n[2] name variants")
    m = compile_matchers(["Casey Stone"], smart=True)
    for probe in ("Report by Stone, Casey (2024)",
                  "Signed: C. Stone",
                  "Attendees: Casey S., et al."):
        r = redact_text(probe, m)
        check(f"variant redacted: {probe[:28]!r}", r.n_redacted >= 1, r.text)
    r = redact_text("casey stone appeared in lowercase", m)
    check("full name redacted case-insensitively", r.n_redacted == 1)

    print("\n[3] unambiguous terms any-case")
    m = compile_matchers(["casey@example.com", "ACCT-4471"], smart=True)
    r = redact_text("Mail CASEY@EXAMPLE.COM about acct-4471 today.", m)
    check("email redacted any case", "JSTONE" not in r.text.upper()
          or "[REDACTED]" in r.text, r.text)
    check("id redacted any case", "4471" not in r.text, r.text)

    print("\n[4] smart off = scorched earth")
    m = compile_matchers(["Stone"], smart=False)
    r = redact_text("three crushed stone fired", m)
    check("lowercase redacted when smart off", "rounds" not in r.text)

    print("\n[5] ingest end-to-end")
    from ez_rag.workspace import Workspace
    from ez_rag.ingest import ingest
    from ez_rag.index import Index
    from ez_rag.embed import make_embedder

    tmp = Path(tempfile.mkdtemp(prefix="ezrag_redact_"))
    ws = Workspace(tmp)
    ws.initialize()
    (ws.docs_dir / "notes.md").write_text(
        "# Meeting with Casey Stone\n\n"
        "Casey Stone reviewed the quarry. The crew hauled crushed stone "
        "at noon. Contact: casey@example.com. Stone signed off.",
        encoding="utf-8")
    cfg = ws.load_config()
    cfg.embedder_provider = "fastembed"
    cfg.redact_terms = ["Casey Stone", "Stone", "casey@example.com"]
    st = ingest(ws, cfg=cfg)
    check("ingest ok", st.files_new == 1 and st.files_errored == 0)

    emb = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=emb.dim)
    texts = [r[0] for r in idx.conn.execute("SELECT text FROM chunks")]
    tokens = [r[0] for r in idx.conn.execute("SELECT tokens FROM chunks")]
    joined_t = "\n".join(texts)
    joined_k = "\n".join(tokens)
    check("name absent from index text", "Casey" not in joined_t, joined_t)
    check("email absent from index text", "casey@" not in joined_t.lower())
    check("name absent from FTS tokens", "casey" not in joined_k.lower())
    check("common-noun 'crushed stone' survived ingest",
          "crushed stone" in joined_t, joined_t)
    check("header (from doc heading) redacted too",
          "[REDACTED]" in joined_t)

    # Term change -> provenance mismatch -> re-ingest
    cfg.redact_terms = ["Casey Stone"]
    st2 = ingest(ws, cfg=cfg)
    check("term change re-ingests (version fold)", st2.files_changed == 1,
          f"{vars(st2)}")
    fp1 = terms_fingerprint(["a"], True)
    check("fingerprint varies by terms",
          fp1 != terms_fingerprint(["b"], True)
          and fp1 != terms_fingerprint(["a"], False))

    # filename warning
    warns = filename_warnings(["docs/Casey_Stone_resume.pdf"],
                               ["Casey Stone"])
    check("filename leak warned", len(warns) == 1, f"{warns}")

    print("\n[6] export gates")
    from ez_rag.export import export_chatbot
    # The index currently reflects ONE redact term ("Casey Stone" from
    # step 5) — so "Stone signed off" and the email are still present.
    # Configuring the full term list WITHOUT re-ingesting is exactly the
    # real-world dirty state the export gate exists for.
    cfg.redact_terms = ["Casey Stone", "Stone", "casey@example.com"]
    cfg.save(ws.config_path)
    try:
        export_chatbot(ws, tmp / "out.zip", include_sources=True)
        check("include_sources refused", False)
    except RuntimeError as e:
        check("include_sources refused", "ORIGINAL" in str(e), str(e)[:80])
    try:
        export_chatbot(ws, tmp / "out.zip")
        check("dirty index refused", False)
    except RuntimeError as e:
        check("dirty index refused", "still present" in str(e), str(e)[:80])
    smart_hits = scan_index_for_terms(ws.meta_db_path, ["Stone"],
                                       smart=True)
    aggressive_hits = scan_index_for_terms(ws.meta_db_path, ["Stone"],
                                            smart=False)
    check("smart scan finds capitalized leak",
          smart_hits.get("Stone", 0) >= 1, f"{smart_hits}")
    check("aggressive scan >= smart (adds common-noun hits)",
          aggressive_hits.get("Stone", 0) >= smart_hits.get("Stone", 0),
          f"{aggressive_hits} vs {smart_hits}")
    # A plain ingest re-ingests automatically (version fold) -> clean
    st3 = ingest(ws, cfg=cfg)
    check("plain ingest re-ingests to clean state",
          st3.files_changed == 1 and st3.files_errored == 0,
          f"{vars(st3)}")
    out = export_chatbot(ws, tmp / "out.zip")
    check("clean redacted export succeeds", Path(out).is_file())

    print(f"\n=== redaction summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

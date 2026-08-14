"""Corpus glossary: acronyms, terms, and product SKUs — with sources.

Scans the indexed corpus and builds a defined-terms index:

  1. **Acronyms** (2–6 uppercase letters, e.g. RAG, SKU, API) are
     collected with occurrence counts and first-seen locations. For
     each one we try to find an IN-CORPUS definition using the two
     patterns technical documents actually use:
         "Retrieval-Augmented Generation (RAG)"   expansion (ACRO)
         "RAG (Retrieval-Augmented Generation)"   ACRO (expansion)
     validated by matching the acronym letters against the expansion's
     word initials (stopwords like of/and/the may be skipped).
  2. **Undefined acronyms** get an external reference: the Wikipedia
     page when it can be verified (free Wikipedia API), otherwise an
     unverified search link. With `proprietary_data = true` NO network
     calls are made — external links are constructed offline and
     labeled unverified.
  3. **Product SKUs** (letter+digit model codes like RTX 5090, X570,
     A770) are detected heuristically; when a known vendor name
     appears adjacent in the text, the entry links a vendor-scoped
     search. Always labeled unverified — we won't claim a product page
     we didn't check.

Outputs (written by `build_glossary`):
  - `<workspace>/.ezrag/glossary.json`      — machine-readable
  - `<workspace>/.ezrag/reports/glossary-<ts>.html` — the human log:
    every entry, its definition, WHERE it came from (file:page or the
    external URL), occurrence counts. Self-contained HTML, no CDNs.
  - optional `docs/_glossary.md` (`add_to_corpus=True`) so the next
    ingest makes the glossary itself retrievable ("what does XYZ
    stand for?").
"""
from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Words that may be skipped when matching acronym letters to expansion
# initials ("Department of Defense" -> DoD/DOD).
_STOPWORDS = {"of", "and", "the", "for", "in", "on", "at", "to", "a", "an"}

# Acronyms so universal they don't need definitions flagged missing.
_WELL_KNOWN = {
    "USA", "UK", "EU", "UN", "CEO", "CFO", "CTO", "FAQ", "PDF", "URL",
    "HTML", "HTTP", "HTTPS", "USB", "GPS", "AM", "PM", "TV", "DIY",
    "ID", "OK", "PS", "NB", "AKA", "ASAP", "ETA", "DNA", "IT",
}

# Vendor names we recognize adjacent to SKUs (lowercase).
_VENDORS = {
    "nvidia": "nvidia.com", "intel": "intel.com", "amd": "amd.com",
    "cisco": "cisco.com", "dell": "dell.com", "hp": "hp.com",
    "lenovo": "lenovo.com", "microsoft": "microsoft.com",
    "apple": "apple.com", "samsung": "samsung.com", "asus": "asus.com",
    "gigabyte": "gigabyte.com", "msi": "msi.com", "qualcomm": "qualcomm.com",
    "broadcom": "broadcom.com", "seagate": "seagate.com",
    "kingston": "kingston.com", "corsair": "corsair.com",
    "logitech": "logitech.com", "ibm": "ibm.com", "oracle": "oracle.com",
    "siemens": "siemens.com", "bosch": "bosch.com", "honeywell":
    "honeywell.com", "caterpillar": "caterpillar.com", "komatsu":
    "komatsu.com",
}

_ACRO_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
# SKU-ish: letters+digits mix, optionally dashed, 3-12 chars, must have
# both a letter and a digit ("RTX" alone no, "5090" alone no, "X570" yes,
# "PN-1042A" yes).
_SKU_RE = re.compile(r"\b(?=[A-Z0-9-]{3,12}\b)(?=[A-Z-]*\d)(?=\d*[A-Z])"
                     r"[A-Z0-9][A-Z0-9-]{1,11}\b")
# "Expansion (ACRO)" and "ACRO (Expansion)"
_DEF_A = re.compile(
    r"((?:[A-Z][\w'&-]*(?:\s+(?:of|and|the|for|in|on|at|to|a|an))?\s+){1,7}"
    r"[A-Z][\w'&-]*)\s*\(\s*([A-Z][A-Z0-9]{1,5})s?\s*\)")
_DEF_B = re.compile(
    r"\b([A-Z][A-Z0-9]{1,5})s?\s*\(\s*([A-Za-z][\w'&-]*"
    r"(?:\s+[\w'&-]+){0,7})\s*\)")


@dataclass
class GlossaryEntry:
    term: str
    kind: str                       # "acronym" | "sku"
    definition: str = ""            # in-corpus expansion if found
    defined_in_corpus: bool = False
    source: str = ""                # "file:page" for in-corpus defs
    external_url: str = ""
    external_label: str = ""        # "Wikipedia (verified)" etc.
    vendor: str = ""
    occurrences: int = 0
    locations: list[str] = field(default_factory=list)   # first few


def _initials_match(expansion: str, acro: str) -> bool:
    words = [w for w in re.split(r"[\s-]+", expansion) if w]
    letters = [w[0].upper() for w in words if w.lower() not in _STOPWORDS]
    all_letters = [w[0].upper() for w in words]
    target = re.sub(r"[^A-Z0-9]", "", acro.upper())
    return "".join(letters) == target or "".join(all_letters) == target


def _wikipedia_lookup(term: str, *, allow_web: bool,
                       timeout: float = 6.0) -> tuple[str, str]:
    """Return (url, label). Verified via the free Wikipedia API when web
    access is allowed; offline/unverified search link otherwise."""
    search_url = ("https://en.wikipedia.org/w/index.php?search="
                  + term.replace(" ", "+"))
    if not allow_web:
        return search_url, "Wikipedia search (unverified — offline mode)"
    try:
        import httpx
        # Wikimedia's robot policy requires a descriptive User-Agent
        # with a contact URL — generic UAs get 403.
        r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": term, "limit": 1,
                    "format": "json"},
            timeout=timeout,
            headers={"User-Agent":
                     "ez-rag-glossary/0.1 "
                     "(https://github.com/jrounds99/ez-rag) httpx"},
        )
        data = r.json()
        if data and len(data) >= 4 and data[3]:
            return data[3][0], "Wikipedia (verified)"
    except Exception:
        pass
    return search_url, "Wikipedia search (unverified)"


def _vendor_link(sku: str, vendor: str) -> tuple[str, str]:
    dom = _VENDORS.get(vendor.lower(), "")
    if dom:
        return (f"https://duckduckgo.com/?q=site%3A{dom}+"
                + sku.replace(" ", "+"),
                f"{vendor.title()} site search (unverified)")
    return (f"https://duckduckgo.com/?q={sku.replace(' ', '+')}",
            "Web search (unverified)")


def scan_corpus(db_path: Path) -> list[tuple[str, str, int]]:
    """Return [(text, path, page)] for every chunk."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT c.text, f.path, COALESCE(c.page, 0) "
            "FROM chunks c JOIN files f ON c.file_id = f.id"
        ).fetchall()
    finally:
        conn.close()


def extract_entries(rows, *, min_occurrences: int = 2) -> list[GlossaryEntry]:
    """Pure extraction (no network): acronyms + defs + SKUs."""
    acro: dict[str, GlossaryEntry] = {}
    skus: dict[str, GlossaryEntry] = {}

    for text, path, page in rows:
        loc = f"{path}" + (f" p.{page}" if page else "")
        # ---- acronym occurrences ----
        for m in _ACRO_RE.finditer(text or ""):
            t = m.group(0)
            if t in _WELL_KNOWN or t.isdigit() or len(t) < 2:
                continue
            if _SKU_RE.fullmatch(t):
                continue          # looks like a model code, not an acronym
            e = acro.setdefault(t, GlossaryEntry(term=t, kind="acronym"))
            e.occurrences += 1
            if len(e.locations) < 3 and loc not in e.locations:
                e.locations.append(loc)
        # ---- in-corpus definitions ----
        for m in _DEF_A.finditer(text or ""):
            expansion = " ".join(m.group(1).split())
            t = m.group(2)
            if _initials_match(expansion, t):
                e = acro.setdefault(t, GlossaryEntry(term=t, kind="acronym"))
                if not e.defined_in_corpus:
                    e.definition = expansion
                    e.defined_in_corpus = True
                    e.source = loc
        for m in _DEF_B.finditer(text or ""):
            t = m.group(1)
            expansion = " ".join(m.group(2).split())
            if _initials_match(expansion, t):
                e = acro.setdefault(t, GlossaryEntry(term=t, kind="acronym"))
                if not e.defined_in_corpus:
                    e.definition = expansion
                    e.defined_in_corpus = True
                    e.source = loc
        # ---- SKUs + adjacent vendor ----
        for m in _SKU_RE.finditer(text or ""):
            t = m.group(0)
            if t.isdigit() or "-" == t.strip("-") or t in _WELL_KNOWN:
                continue
            e = skus.setdefault(t, GlossaryEntry(term=t, kind="sku"))
            e.occurrences += 1
            if len(e.locations) < 3 and loc not in e.locations:
                e.locations.append(loc)
            if not e.vendor:
                window = (text[max(0, m.start() - 40):m.start()] or "").lower()
                for v in _VENDORS:
                    if v in window:
                        e.vendor = v
                        break

    out = [e for e in acro.values() if e.occurrences >= min_occurrences
           or e.defined_in_corpus]
    out += [e for e in skus.values() if e.occurrences >= min_occurrences]
    out.sort(key=lambda e: (-e.occurrences, e.term))
    return out


def resolve_external(entries: list[GlossaryEntry], *, allow_web: bool,
                      progress=None) -> None:
    """Attach external references to entries with no in-corpus source.

    Web lookups are throttled (~5/s) per Wikimedia's robot policy."""
    for i, e in enumerate(entries):
        if e.kind == "acronym" and not e.defined_in_corpus:
            if allow_web:
                time.sleep(0.2)
            e.external_url, e.external_label = _wikipedia_lookup(
                e.term, allow_web=allow_web)
        elif e.kind == "sku":
            e.external_url, e.external_label = _vendor_link(e.term, e.vendor)
        if progress:
            try:
                progress(i + 1, len(entries))
            except Exception:
                pass


def render_html(entries: list[GlossaryEntry], *, workspace_name: str,
                 allow_web: bool) -> str:
    """Self-contained HTML report (no CDNs, no external assets)."""
    ts = time.strftime("%Y-%m-%d %H:%M")
    defined = [e for e in entries if e.defined_in_corpus]
    external = [e for e in entries
                if not e.defined_in_corpus and e.kind == "acronym"]
    sku_entries = [e for e in entries if e.kind == "sku"]

    def row(e: GlossaryEntry) -> str:
        if e.defined_in_corpus:
            src = f"<span class='src'>{html.escape(e.source)}</span>"
            d = html.escape(e.definition)
        elif e.external_url:
            src = (f"<a href='{html.escape(e.external_url)}' "
                   f"target='_blank'>{html.escape(e.external_label)}</a>")
            d = html.escape(e.definition or "—")
        else:
            src, d = "—", "—"
        locs = html.escape("; ".join(e.locations[:2]))
        return (f"<tr><td><code>{html.escape(e.term)}</code></td>"
                f"<td>{d}</td><td>{src}</td>"
                f"<td class='n'>{e.occurrences}</td>"
                f"<td class='locs'>{locs}</td></tr>")

    def table(title, subset, note=""):
        if not subset:
            return f"<h2>{title}</h2><p class='dim'>None found.</p>"
        rows = "\n".join(row(e) for e in subset)
        note_html = f"<p class='dim'>{note}</p>" if note else ""
        return (f"<h2>{title} <span class='count'>{len(subset)}</span></h2>"
                f"{note_html}"
                f"<table><thead><tr><th>Term</th><th>Definition</th>"
                f"<th>Source</th><th>Uses</th><th>Seen in</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>")

    web_note = ("External links verified against Wikipedia at build time."
                if allow_web else
                "Built OFFLINE (proprietary-data mode or --no-web): "
                "external links are constructed, not verified.")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Glossary — {html.escape(workspace_name)}</title>
<style>
 body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 2rem auto;
        max-width: 1100px; padding: 0 1rem; background:#12141c;
        color:#e6e7eb; }}
 h1 {{ margin-bottom: 0; }} .dim {{ color:#9097a6; }}
 h2 {{ margin-top: 2rem; border-bottom: 1px solid #2a2e3f;
      padding-bottom: 4px; }}
 .count {{ color:#7c7bff; font-size: .8em; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ text-align: left; padding: 6px 10px;
          border-bottom: 1px solid #23263a; vertical-align: top; }}
 th {{ color:#9097a6; font-weight:600; }}
 code {{ background:#1f2230; padding:1px 6px; border-radius:4px; }}
 a {{ color:#8ab4ff; }} .n {{ text-align:right; }}
 .locs {{ color:#9097a6; font-size:12px; }}
 .src {{ color:#3ddc84; }}
</style></head><body>
<h1>Glossary &amp; acronym index</h1>
<p class="dim">Workspace: {html.escape(workspace_name)} · built {ts} ·
{len(entries)} entries · {web_note}</p>
{table("Defined in your corpus", defined,
       "Definitions extracted from the documents themselves — the "
       "Source column shows exactly where.")}
{table("Not defined in the corpus — external reference attached",
       external,
       "These acronyms appear in your documents without an in-corpus "
       "definition. Each links to Wikipedia so readers aren't left "
       "guessing.")}
{table("Product / model codes (SKUs)", sku_entries,
       "Heuristic detection; vendor links are search links, always "
       "labeled unverified.")}
<p class="dim">Generated by <code>ez-rag glossary</code>. Machine-readable
copy: <code>.ezrag/glossary.json</code></p>
</body></html>"""


def render_markdown(entries: list[GlossaryEntry]) -> str:
    """Corpus-ingestable glossary (docs/_glossary.md via add-to-corpus)."""
    lines = ["# Glossary", "",
             "Definitions and references for terms used in this corpus.",
             ""]
    for e in entries:
        if e.defined_in_corpus:
            lines.append(f"- **{e.term}** — {e.definition} "
                         f"(defined in {e.source})")
        elif e.external_url:
            lines.append(f"- **{e.term}** — see {e.external_url} "
                         f"({e.external_label})")
    return "\n".join(lines) + "\n"


def build_glossary(ws_root: Path, *, allow_web: bool = True,
                    min_occurrences: int = 2, add_to_corpus: bool = False,
                    progress=None) -> dict:
    """Full pipeline. Returns a summary dict incl. artifact paths."""
    ws_root = Path(ws_root)
    db = ws_root / ".ezrag" / "meta.sqlite"
    if not db.is_file():
        raise FileNotFoundError(
            f"No index at {db} — run `ez-rag ingest` first.")
    rows = scan_corpus(db)
    entries = extract_entries(rows, min_occurrences=min_occurrences)
    resolve_external(entries, allow_web=allow_web, progress=progress)

    out_dir = ws_root / ".ezrag" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    html_path = out_dir / f"glossary-{ts}.html"
    html_path.write_text(
        render_html(entries, workspace_name=ws_root.name,
                     allow_web=allow_web),
        encoding="utf-8")
    json_path = ws_root / ".ezrag" / "glossary.json"
    json_path.write_text(
        json.dumps([asdict(e) for e in entries], indent=2),
        encoding="utf-8")

    md_path = None
    if add_to_corpus:
        md_path = ws_root / "docs" / "_glossary.md"
        md_path.write_text(render_markdown(entries), encoding="utf-8")

    return {
        "entries": len(entries),
        "defined_in_corpus": sum(1 for e in entries if e.defined_in_corpus),
        "external": sum(1 for e in entries
                        if e.external_url and not e.defined_in_corpus),
        "skus": sum(1 for e in entries if e.kind == "sku"),
        "html": str(html_path),
        "json": str(json_path),
        "markdown": str(md_path) if md_path else "",
    }

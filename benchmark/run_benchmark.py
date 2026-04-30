#!/usr/bin/env python3
"""
ez-rag benchmark runner.

Downloads a small corpus of public documents, ingests them, runs a curated
question set, and writes a markdown report plus summary.json.

Two engines are supported:

  --engine reference   A minimal built-in pipeline (no ML, no GPU). Validates
                       parsing + retrieval mechanics. Runnable today, before
                       the ez-rag CLI exists. This is the default.

  --engine ez-rag      Shells out to the ez-rag CLI. Requires `ez-rag` on PATH.

Usage:
    python run_benchmark.py
    python run_benchmark.py --engine ez-rag --workspace /tmp/ezrag-bench
    python run_benchmark.py --no-llm --top-k 10

Optional dependencies (auto-detected; benchmark degrades gracefully):
    pip install requests pypdf beautifulsoup4 lxml Pillow pytesseract

Optional system tool for OCR:
    Tesseract OCR (https://tesseract-ocr.github.io/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
CORPUS_MANIFEST = THIS_DIR / "corpus_manifest.json"
QUESTIONS_FILE = THIS_DIR / "questions.json"
DEFAULT_WORKSPACE = THIS_DIR / "_workspace"
REPORTS_DIR = THIS_DIR / "reports"

OCR_SCREENSHOT_TEXT = (
    "EZ-RAG OCR BENCHMARK\n"
    "Line two: the quick brown fox jumps over the lazy dog.\n"
    "Line three: 1234567890."
)

# ---------- optional deps (lazy) ----------

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

# ---------- small utils ----------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------- corpus download ----------

def download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download to dest. Returns True on success. Skips if dest exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    ensure_dir(dest.parent)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ez-rag-benchmark/0.1 (+https://example.invalid/ez-rag)",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, dest.open("wb") as f:
            shutil.copyfileobj(r, f)
        return dest.stat().st_size > 0
    except Exception as e:
        log(f"  download failed: {url} -> {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def generate_synthetic_screenshot(dest: Path) -> bool:
    """Render OCR_SCREENSHOT_TEXT to a PNG using Pillow (if available)."""
    PIL = _try_import("PIL")
    if PIL is None:
        log("  Pillow not installed; skipping synthetic screenshot generation")
        return False
    from PIL import Image, ImageDraw, ImageFont  # noqa
    width, height = 900, 220
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(candidate).exists():
            try:
                font = ImageFont.truetype(candidate, 28)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    draw.multiline_text((20, 20), OCR_SCREENSHOT_TEXT, fill="black", font=font, spacing=8)
    ensure_dir(dest.parent)
    img.save(dest, "PNG")
    return True


def fetch_corpus(corpus_dir: Path, manifest: dict) -> list[dict]:
    """Materialize the corpus locally. Returns the list of items that succeeded."""
    ensure_dir(corpus_dir)
    ready: list[dict] = []
    for item in manifest["corpus"]:
        dest = corpus_dir / item["filename"]
        if item.get("generated"):
            ok = generate_synthetic_screenshot(dest)
        else:
            ok = download(item["url"], dest)
            # data.gov item is sometimes a landing page, not the CSV. Tolerate.
            if ok and item["kind"] == "csv":
                head = dest.read_bytes()[:200].lower()
                if b"<html" in head or b"<!doctype" in head:
                    log(f"  {item['filename']}: got HTML, replacing with synthetic CSV")
                    dest.write_text(
                        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
                        encoding="utf-8",
                    )
        if ok and dest.exists() and dest.stat().st_size > 0:
            item = dict(item)
            item["_local_path"] = str(dest)
            item["_sha256"] = sha256_file(dest)
            item["_bytes"] = dest.stat().st_size
            ready.append(item)
            log(f"  ok  {item['filename']} ({item['_bytes']:,} bytes)")
        else:
            log(f"  MISS {item['filename']} (skipped from this run)")
    return ready

# ---------- reference parsers ----------

def parse_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return path.read_text(encoding="latin-1", errors="replace")


def parse_pdf(path: Path) -> str:
    pypdf = _try_import("pypdf")
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(str(path))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            log(f"  pypdf failed on {path.name}: {e}")
    pdfminer = _try_import("pdfminer.high_level")
    if pdfminer is not None:
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            return extract_text(str(path)) or ""
        except Exception as e:
            log(f"  pdfminer failed on {path.name}: {e}")
    log(f"  WARN no PDF parser available for {path.name}; install pypdf")
    return ""


def parse_html(path: Path) -> str:
    bs4 = _try_import("bs4")
    raw = parse_text_file(path)
    if bs4 is None:
        return re.sub(r"<[^>]+>", " ", raw)
    from bs4 import BeautifulSoup  # type: ignore
    soup = BeautifulSoup(raw, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    return soup.get_text(" ", strip=True)


def parse_csv(path: Path) -> str:
    return parse_text_file(path)


def parse_image(path: Path) -> str:
    pytesseract = _try_import("pytesseract")
    PIL = _try_import("PIL")
    if pytesseract is None or PIL is None:
        log(f"  WARN OCR deps missing for {path.name}; install pytesseract + Pillow + Tesseract")
        return ""
    from PIL import Image  # type: ignore
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        log(f"  OCR failed on {path.name}: {e}")
        return ""


PARSERS = {
    "txt": parse_text_file,
    "pdf": parse_pdf,
    "html": parse_html,
    "csv": parse_csv,
    "image": parse_image,
}

# ---------- reference index (lexical BM25-ish) ----------

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

def tokenize(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def chunk_text(text: str, target_words: int = 220, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, target_words - overlap)
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + target_words])
        if piece.strip():
            chunks.append(piece)
        if i + target_words >= len(words):
            break
    return chunks


@dataclass
class Chunk:
    doc_id: str
    source: str
    text: str
    tokens: list[str] = field(default_factory=list)


@dataclass
class BM25Index:
    chunks: list[Chunk]
    df: Counter
    avgdl: float
    N: int

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        df: Counter = Counter()
        total = 0
        for c in chunks:
            c.tokens = tokenize(c.text)
            total += len(c.tokens)
            for t in set(c.tokens):
                df[t] += 1
        avgdl = (total / max(1, len(chunks))) if chunks else 0.0
        return cls(chunks=chunks, df=df, avgdl=avgdl, N=len(chunks))

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, Chunk]]:
        import math
        q_tokens = tokenize(query)
        scores: list[tuple[float, Chunk]] = []
        for c in self.chunks:
            tf = Counter(c.tokens)
            dl = max(1, len(c.tokens))
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                n = self.df.get(qt, 0)
                if n == 0:
                    continue
                idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                f = tf[qt]
                denom = f + k1 * (1 - b + b * dl / max(1e-9, self.avgdl))
                score += idf * (f * (k1 + 1)) / max(1e-9, denom)
            if score > 0:
                scores.append((score, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

# ---------- engines ----------

class ReferenceEngine:
    """Built-in pipeline using local parsers + BM25. No ML, no network at query time."""

    name = "reference"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.index: BM25Index | None = None
        self.timings: dict[str, float] = {}

    def ingest(self, items: list[dict]) -> None:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        per_doc_times: dict[str, float] = {}
        for item in items:
            t_doc = time.perf_counter()
            parser = PARSERS.get(item["kind"])
            if parser is None:
                log(f"  no parser for kind={item['kind']}, skipping {item['filename']}")
                continue
            text = parser(Path(item["_local_path"])) or ""
            text = re.sub(r"\s+", " ", text).strip()
            for piece in chunk_text(text):
                chunks.append(Chunk(doc_id=item["id"], source=item["filename"], text=piece))
            per_doc_times[item["id"]] = time.perf_counter() - t_doc
            log(f"  parsed {item['filename']}: {len(text):,} chars in {per_doc_times[item['id']]:.2f}s")
        t_index = time.perf_counter()
        self.index = BM25Index.build(chunks)
        self.timings["parse_total_s"] = t_index - t0
        self.timings["index_s"] = time.perf_counter() - t_index
        self.timings["chunks"] = len(chunks)
        self.timings["per_doc_s"] = per_doc_times

    def query(self, question: str, k: int = 10) -> list[dict]:
        assert self.index is not None
        hits = self.index.search(question, k=k)
        return [
            {"score": score, "doc_id": c.doc_id, "source": c.source, "text": c.text}
            for score, c in hits
        ]


class EzRagEngine:
    """Shells out to the ez-rag CLI.

    Resolution order:
      1) $EZRAG_CMD          (e.g. "ez-rag" or "python -m ez_rag.cli")
      2) `ez-rag` on PATH    (after `pipx install ez-rag`)
      3) `python -m ez_rag.cli` if the package is importable in this env
    """

    name = "ez-rag"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.timings: dict[str, float] = {}
        self._cmd = self._resolve_cmd()

    @staticmethod
    def _resolve_cmd() -> list[str]:
        override = os.environ.get("EZRAG_CMD")
        if override:
            return override.split()
        if shutil.which("ez-rag"):
            return ["ez-rag"]
        try:
            import importlib.util
            if importlib.util.find_spec("ez_rag") is not None:
                return [sys.executable, "-m", "ez_rag.cli"]
        except Exception:
            pass
        raise RuntimeError(
            "ez-rag not available. Install it (`pipx install ez-rag`) or set "
            "EZRAG_CMD, e.g. EZRAG_CMD='python -m ez_rag.cli', or run with "
            "--engine reference."
        )

    def _run(self, args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self._cmd, *args],
            cwd=self.workspace,
            check=True,
            capture_output=capture,
            text=True,
        )

    def ingest(self, items: list[dict]) -> None:
        ensure_dir(self.workspace / "docs")
        for item in items:
            shutil.copy2(item["_local_path"], self.workspace / "docs" / item["filename"])
        self._run(["init", "."], capture=False)
        t0 = time.perf_counter()
        self._run(["ingest"], capture=False)
        self.timings["ingest_s"] = time.perf_counter() - t0

    def query(self, question: str, k: int = 10) -> list[dict]:
        proc = self._run(["ask", question, "--json", "--top-k", str(k)])
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []
        results = []
        for hit in payload.get("retrieved", []):
            results.append({
                "score": hit.get("score", 0.0),
                "doc_id": hit.get("doc_id") or hit.get("source", ""),
                "source": hit.get("source", ""),
                "text": hit.get("text", ""),
            })
        return results

# ---------- evaluation ----------

@dataclass
class QResult:
    qid: str
    question: str
    gold_doc_id: str
    retrieved_doc_ids: list[str]
    rank_of_gold: int | None  # 1-indexed; None if not found
    top1_match: bool
    recall_at_5: bool
    recall_at_10: bool
    substring_hit: bool
    notes: str = ""


def evaluate(engine, questions: list[dict], top_k: int) -> list[QResult]:
    results: list[QResult] = []
    for q in questions:
        try:
            hits = engine.query(q["question"], k=top_k)
        except Exception as e:
            log(f"  q {q['id']} ERROR: {e}")
            results.append(QResult(
                qid=q["id"], question=q["question"], gold_doc_id=q["gold_doc_id"],
                retrieved_doc_ids=[], rank_of_gold=None, top1_match=False,
                recall_at_5=False, recall_at_10=False, substring_hit=False,
                notes=f"error: {e}",
            ))
            continue
        ranked_ids = [h["doc_id"] for h in hits]
        gold = q["gold_doc_id"]
        rank = next((i + 1 for i, d in enumerate(ranked_ids) if d == gold), None)
        # Substring hit: did any retrieved chunk's text contain a required substring?
        joined = "\n".join(h["text"] for h in hits[:5]).lower()
        required = [s.lower() for s in q.get("gold_substrings", [])]
        sub_required_hit = all(s in joined for s in required) if required else True
        any_of = q.get("any_of", [])
        sub_any_hit = True
        if any_of:
            sub_any_hit = any(all(s.lower() in joined for s in group) for group in any_of)
        substring_hit = sub_required_hit and sub_any_hit
        results.append(QResult(
            qid=q["id"], question=q["question"], gold_doc_id=gold,
            retrieved_doc_ids=ranked_ids,
            rank_of_gold=rank,
            top1_match=(rank == 1),
            recall_at_5=(rank is not None and rank <= 5),
            recall_at_10=(rank is not None and rank <= 10),
            substring_hit=substring_hit,
        ))
    return results


def ocr_metrics(items: list[dict]) -> dict[str, Any]:
    """Compute character-error-rate on the synthetic screenshot if present."""
    target = next((i for i in items if i.get("generated")), None)
    if target is None:
        return {"available": False, "reason": "no synthetic screenshot in run"}
    text = parse_image(Path(target["_local_path"]))
    if not text:
        return {"available": False, "reason": "OCR engine unavailable or failed"}
    expected = OCR_SCREENSHOT_TEXT
    # Levenshtein distance, pure Python.
    a, b = expected, text
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        cer = 1.0
    else:
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = cur
        cer = prev[m] / max(1, n)
    return {
        "available": True,
        "expected_chars": len(expected),
        "ocr_chars": len(text),
        "cer": round(cer, 4),
        "ocr_text_preview": text.strip()[:200],
    }

# ---------- reporting ----------

def write_report(report_dir: Path, *, engine_name: str, items: list[dict],
                 results: list[QResult], timings: dict, ocr: dict, args) -> Path:
    ensure_dir(report_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = report_dir / f"report_{engine_name}_{stamp}.md"
    json_path = report_dir / f"summary_{engine_name}_{stamp}.json"

    n = len(results)
    top1 = sum(1 for r in results if r.top1_match)
    r5 = sum(1 for r in results if r.recall_at_5)
    r10 = sum(1 for r in results if r.recall_at_10)
    sub = sum(1 for r in results if r.substring_hit)

    summary = {
        "engine": engine_name,
        "timestamp_utc": stamp,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool))},
        "corpus": [{"id": i["id"], "filename": i["filename"], "bytes": i.get("_bytes"), "sha256": i.get("_sha256")} for i in items],
        "timings": timings,
        "ocr": ocr,
        "metrics": {
            "questions": n,
            "top1": top1,
            "top1_pct": round(100 * top1 / max(1, n), 1),
            "recall_at_5": r5,
            "recall_at_5_pct": round(100 * r5 / max(1, n), 1),
            "recall_at_10": r10,
            "recall_at_10_pct": round(100 * r10 / max(1, n), 1),
            "substring_hit": sub,
            "substring_hit_pct": round(100 * sub / max(1, n), 1),
        },
        "questions": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        f"# ez-rag benchmark report",
        "",
        f"- **Engine**: `{engine_name}`",
        f"- **Timestamp (UTC)**: {stamp}",
        f"- **Python**: {sys.version.split()[0]}  |  **Platform**: {sys.platform}",
        "",
        "## Corpus",
        "",
        "| id | file | bytes | sha256 |",
        "|---|---|---:|---|",
    ]
    for it in items:
        lines.append(f"| {it['id']} | {it['filename']} | {it.get('_bytes', 0):,} | `{(it.get('_sha256') or '')[:12]}…` |")

    lines += [
        "",
        "## Timings",
        "",
        "```json",
        json.dumps(timings, indent=2, default=str),
        "```",
        "",
        "## OCR",
        "",
        "```json",
        json.dumps(ocr, indent=2),
        "```",
        "",
        "## Retrieval metrics",
        "",
        f"- Questions: **{n}**",
        f"- Top-1 doc match: **{top1}/{n}** ({summary['metrics']['top1_pct']}%)",
        f"- Recall@5 (gold doc in top-5): **{r5}/{n}** ({summary['metrics']['recall_at_5_pct']}%)",
        f"- Recall@10: **{r10}/{n}** ({summary['metrics']['recall_at_10_pct']}%)",
        f"- Substring-grounding hit (gold substrings appear in retrieved context): **{sub}/{n}** ({summary['metrics']['substring_hit_pct']}%)",
        "",
        "## Per-question",
        "",
        "| qid | gold doc | rank | top1 | r@5 | r@10 | sub | question |",
        "|---|---|---:|:--:|:--:|:--:|:--:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.qid} | {r.gold_doc_id} | {r.rank_of_gold if r.rank_of_gold else '-'} | "
            f"{'✓' if r.top1_match else ' '} | {'✓' if r.recall_at_5 else ' '} | "
            f"{'✓' if r.recall_at_10 else ' '} | {'✓' if r.substring_hit else ' '} | "
            f"{r.question} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="ez-rag benchmark runner")
    ap.add_argument("--engine", choices=["reference", "ez-rag"], default="reference")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                    help="Where the benchmark stores corpus + index")
    ap.add_argument("--top-k", type=int, default=10, help="Top-k for retrieval evaluation")
    ap.add_argument("--no-llm", action="store_true",
                    help="Reference engine never uses an LLM. This flag is accepted for parity with --engine ez-rag.")
    ap.add_argument("--keep-workspace", action="store_true",
                    help="Don't delete the corpus + index after the run")
    args = ap.parse_args()

    manifest = json.loads(CORPUS_MANIFEST.read_text())
    questions = json.loads(QUESTIONS_FILE.read_text())["questions"]

    workspace = args.workspace
    corpus_dir = workspace / "corpus"
    ensure_dir(workspace)
    ensure_dir(REPORTS_DIR)

    log(f"engine={args.engine}  workspace={workspace}")
    log("Fetching corpus…")
    items = fetch_corpus(corpus_dir, manifest)
    if not items:
        log("FATAL: no corpus items materialized. Check network connectivity.")
        return 2
    available_ids = {i["id"] for i in items}
    questions = [q for q in questions if q["gold_doc_id"] in available_ids]
    log(f"{len(items)} corpus docs, {len(questions)} answerable questions in this run")

    log("Building index…")
    if args.engine == "reference":
        engine = ReferenceEngine(workspace)
    else:
        engine = EzRagEngine(workspace)
    t0 = time.perf_counter()
    try:
        engine.ingest(items)
    except Exception as e:
        log(f"FATAL: ingest failed: {e}")
        traceback.print_exc()
        return 3
    engine.timings["ingest_total_s"] = time.perf_counter() - t0

    log("Running OCR check on synthetic screenshot…")
    ocr = ocr_metrics(items)

    log(f"Querying {len(questions)} questions…")
    t0 = time.perf_counter()
    results = evaluate(engine, questions, top_k=args.top_k)
    engine.timings["query_total_s"] = time.perf_counter() - t0

    log("Writing report…")
    md = write_report(
        REPORTS_DIR,
        engine_name=engine.name,
        items=items,
        results=results,
        timings=engine.timings,
        ocr=ocr,
        args=args,
    )
    log(f"Report written: {md}")

    n = len(results) or 1
    top1 = sum(1 for r in results if r.top1_match)
    r5 = sum(1 for r in results if r.recall_at_5)
    sub = sum(1 for r in results if r.substring_hit)
    print()
    print(f"  Top-1:        {top1}/{n} ({100*top1/n:.1f}%)")
    print(f"  Recall@5:     {r5}/{n} ({100*r5/n:.1f}%)")
    print(f"  Substring hit:{sub}/{n} ({100*sub/n:.1f}%)")
    if ocr.get("available"):
        print(f"  OCR CER:      {ocr['cer']}")
    print()

    if not args.keep_workspace and workspace == DEFAULT_WORKSPACE:
        # Only auto-clean the default workspace, never a user-supplied path.
        try:
            shutil.rmtree(workspace)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

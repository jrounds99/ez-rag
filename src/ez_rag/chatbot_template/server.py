"""Tiny stdlib HTTP server that serves the static chat page and a streaming
/api/ask endpoint.

All retrieval / generation settings come from data/config.toml. The chatbot
itself has no UI options — change behavior by re-exporting from ez-rag, or by
editing data/config.toml in place.

If the bundle was exported with `include_sources=True`, the original ingest
files live under data/sources/<rel-path> and the server exposes:
    GET /api/source?path=<rel>          → raw bytes of the file
    GET /api/page-image?path=<rel>&page=N → PNG render of one PDF page
"""
from __future__ import annotations

import io
import json
import mimetypes
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Vendored ez-rag library (just retrieve + generate + index + embed + config)
from ezrag_lib.config import Config
from ezrag_lib.embed import make_embedder
from ezrag_lib.index import Index, read_stats
from ezrag_lib.generate import (
    apply_query_modifiers, chat_answer, detect_backend,
)
from ezrag_lib.retrieve import agentic_retrieve, smart_retrieve


HOST = "127.0.0.1"
PORT = 8765


# ---- one-time setup --------------------------------------------------------
DATA = ROOT / "data"
CFG_PATH = DATA / "config.toml"
DB_PATH = DATA / "meta.sqlite"
SOURCES_DIR = DATA / "sources"
MANIFEST_PATH = ROOT / "manifest.json"

# manifest.json is written by ez_rag.export.export_chatbot. Old bundles that
# predate it just won't have it — we fall back to autodetecting.
try:
    MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
except Exception:
    MANIFEST = {}
HAS_SOURCES = bool(MANIFEST.get("include_sources")) or SOURCES_DIR.is_dir()

print(f"loading config from {CFG_PATH} ...")
CFG = Config.load(CFG_PATH)
print(f"loading embedder ({CFG.embedder_provider} / {CFG.embedder_model}) ...")
EMBEDDER = make_embedder(CFG)
print(f"opening index at {DB_PATH} ...")
INDEX = Index(DB_PATH, embed_dim=EMBEDDER.dim)
STATS = read_stats(DB_PATH) or {}
print(f"backend: {detect_backend(CFG)}  ·  model: {CFG.llm_model}")
print(f"index:   {STATS.get('files', 0)} files  ·  {STATS.get('chunks', 0)} chunks")
print(f"sources: {'bundled' if HAS_SOURCES else 'not bundled (lean export)'}")


def _resolve_source(rel: str) -> Path | None:
    """Map a citation path like 'docs\\foo.pdf' or 'docs/foo.pdf' to the
    actual file under data/sources/. Returns None if the resolved path
    escapes data/sources/ (path traversal) or the file doesn't exist.
    """
    if not HAS_SOURCES or not rel:
        return None
    rel = rel.strip().replace("\\", "/")
    # ez-rag stores citations as 'docs/<rel>' relative to ws.root. Strip
    # that prefix so we look up the bundled copy under data/sources/<rel>.
    if rel.startswith("docs/"):
        rel = rel[len("docs/"):]
    candidate = (SOURCES_DIR / rel).resolve()
    try:
        candidate.relative_to(SOURCES_DIR.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _retrieve(q: str):
    if getattr(CFG, "agentic", False):
        return agentic_retrieve(query=q, embedder=EMBEDDER, index=INDEX, cfg=CFG)
    return smart_retrieve(query=q, embedder=EMBEDDER, index=INDEX, cfg=CFG)


# ---- HTTP handler ----------------------------------------------------------

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
    ".svg":  "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write(f"  {self.address_string()} - {fmt % args}\n")

    # ----- routing -----
    def do_GET(self):
        path, _, qs = self.path.partition("?")
        if path == "/" or path == "":
            self._serve_file("chat.html")
        elif path == "/api/status":
            self._send_json({
                "backend": detect_backend(CFG),
                "model":   CFG.llm_model,
                "files":   STATS.get("files", 0),
                "chunks":  STATS.get("chunks", 0),
                "embedder": EMBEDDER.name,
                "has_sources": HAS_SOURCES,
            })
        elif path == "/api/source":
            self._handle_source(urllib.parse.parse_qs(qs))
        elif path == "/api/page-image":
            self._handle_page_image(urllib.parse.parse_qs(qs))
        else:
            # Static files only — no path traversal allowed.
            rel = path.lstrip("/")
            self._serve_file(rel)

    def do_POST(self):
        if self.path == "/api/ask":
            self._handle_ask()
        else:
            self.send_error(404)

    # ----- helpers -----
    def _serve_file(self, rel: str):
        candidate = (ROOT / rel).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not candidate.is_file():
            self.send_error(404, "Not found")
            return
        ct = CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _handle_source(self, qs: dict):
        """Return the raw bytes of an ingested source file. Citations carry
        the relative path; we resolve under data/sources/ and refuse path
        traversal."""
        rel = (qs.get("path") or [""])[0]
        if not HAS_SOURCES:
            self.send_error(404, "Source files not bundled in this export "
                                  "(re-export with 'Include source files' "
                                  "to enable preview).")
            return
        f = _resolve_source(rel)
        if f is None:
            self.send_error(404, f"Not found in bundle: {rel}")
            return
        ct, _ = mimetypes.guess_type(str(f))
        body = f.read_bytes()
        self._send_bytes(body, ct or "application/octet-stream")

    def _handle_page_image(self, qs: dict):
        """Render a single PDF page to PNG and return it. Used by the chat
        UI to show what a citation actually looks like on the page."""
        rel = (qs.get("path") or [""])[0]
        try:
            page = int((qs.get("page") or ["0"])[0])
        except ValueError:
            page = 0
        if not HAS_SOURCES:
            self.send_error(404, "Source files not bundled.")
            return
        if page <= 0:
            self.send_error(400, "page must be a positive integer")
            return
        f = _resolve_source(rel)
        if f is None or f.suffix.lower() != ".pdf":
            self.send_error(404, f"Not a bundled PDF: {rel}")
            return
        try:
            import pypdfium2 as pdfium
        except ImportError:
            self.send_error(501,
                "pypdfium2 not installed — re-run requirements install "
                "or `pip install pypdfium2` so the chatbot can render "
                "PDF pages.")
            return
        try:
            pdf = pdfium.PdfDocument(str(f))
            if page > len(pdf):
                self.send_error(404,
                    f"Page {page} > document length {len(pdf)}")
                return
            # 2.5x render — high enough for good zoom without massive bytes.
            # The chat UI lets the user zoom further client-side.
            bitmap = pdf[page - 1].render(scale=2.5)
            pil = bitmap.to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG", optimize=True)
            self._send_bytes(buf.getvalue(), "image/png")
        except Exception as e:
            self.send_error(500, f"Render failed: {type(e).__name__}: {e}")

    def _handle_ask(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            history = payload.get("history") or []
            if not history:
                self.send_error(400, "history required")
                return
            latest = history[-1]
            if latest.get("role") != "user":
                self.send_error(400, "last turn must be user")
                return
            raw_q = latest.get("content") or ""
            question = apply_query_modifiers(raw_q, CFG)
        except Exception as e:
            self.send_error(400, f"bad request: {e}")
            return

        # Stream back NDJSON events (one JSON object per line).
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

        try:
            hits = _retrieve(question) if CFG.use_rag else []
            if hits:
                emit({"kind": "citations", "items": [
                    {"path": h.path, "page": h.page, "section": h.section,
                     "text": h.text[:240]}
                    for h in hits
                ]})

            # Build trimmed history for the LLM (everything except the last
            # user turn, which chat_answer rebuilds with hits).
            trimmed = history[:-1]
            stream = chat_answer(history=trimmed, latest_question=question,
                                 hits=hits, cfg=CFG, stream=True)
            # Streaming returns an iterator of (kind, text); non-streaming
            # returns Answer (when backend == 'none').
            if hasattr(stream, "__iter__") and not hasattr(stream, "text"):
                for kind, text in stream:
                    emit({"kind": kind, "text": text})
            else:
                emit({"kind": "content", "text": stream.text})
        except Exception as e:
            emit({"kind": "error", "text": f"{type(e).__name__}: {e}"})


# ---- main ------------------------------------------------------------------

def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"\nez-rag chatbot ready at {url}")
    print("Press Ctrl+C to stop.\n")

    # Open browser once the server is listening.
    threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
        server.shutdown()


if __name__ == "__main__":
    main()

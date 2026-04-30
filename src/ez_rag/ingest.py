"""Top-level ingest pipeline: walk docs/, parse, chunk, embed, upsert."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import Chunk, chunk_sections
from .config import Config
from .embed import Embedder, make_embedder
from .generate import contextualize_chunk, detect_backend
from .index import Index, file_sha256
from .parsers import ParsedSection, get_parser, supported_extensions
from .workspace import Workspace


PARSER_VERSION = "1"
CHUNKER_VERSION = "2"  # bumped for contextual retrieval support


@dataclass
class IngestStats:
    files_seen: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_skipped_unchanged: int = 0
    files_unsupported: int = 0
    files_errored: int = 0
    chunks_added: int = 0
    files_removed: int = 0
    seconds: float = 0.0
    errors: list[tuple[str, str]] = None  # type: ignore

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


ProgressCb = Callable[[str, str], None]   # (path, status_text)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def _tokenize(text: str) -> str:
    return " ".join(w.lower() for w in WORD_RE.findall(text))


def _walk_docs(docs_dir: Path) -> list[Path]:
    exts = supported_extensions()
    out: list[Path] = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def ingest(
    ws: Workspace,
    *,
    cfg: Config | None = None,
    force: bool = False,
    progress: ProgressCb | None = None,
) -> IngestStats:
    cfg = cfg or ws.load_config()
    stats = IngestStats()
    t0 = time.perf_counter()

    embedder = make_embedder(cfg)
    index = Index(ws.meta_db_path, embed_dim=embedder.dim)

    docs_dir = ws.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = _walk_docs(docs_dir)
    stats.files_seen = len(files)

    present_rel: set[str] = set()
    for path in files:
        rel = str(path.relative_to(ws.root))
        present_rel.add(rel)
        try:
            stat = path.stat()
            sha = file_sha256(path)
            existing = index.file_state(rel)
            if (
                not force
                and existing is not None
                and existing.sha256 == sha
            ):
                stats.files_skipped_unchanged += 1
                if progress:
                    progress(rel, "unchanged")
                continue
            if progress:
                progress(rel, "parsing")
            parser = get_parser(path)
            if parser is None:
                stats.files_unsupported += 1
                continue
            sections = parser(path)
            if not sections:
                if progress:
                    progress(rel, "empty")
                continue
            chunks = chunk_sections(
                sections,
                chunk_tokens=cfg.chunk_size,
                overlap_tokens=cfg.chunk_overlap,
            )
            if not chunks:
                if progress:
                    progress(rel, "no chunks")
                continue
            # Optional Anthropic-style Contextual Retrieval: prepend a
            # 1-sentence chunk-context summary BEFORE embedding. We embed
            # the augmented text but DO NOT alter chunk.text (so the
            # original passage still displays/cites cleanly).
            embed_texts = [c.text for c in chunks]
            if cfg.enable_contextual and detect_backend(cfg) != "none":
                if progress:
                    progress(rel, f"contextualizing {len(chunks)} chunks")
                full_text = "\n\n".join(s.text for s in sections)
                embed_texts = [
                    contextualize_chunk(c.text, full_text, cfg) for c in chunks
                ]
            if progress:
                progress(rel, f"embedding {len(chunks)} chunks")
            vecs = embedder.embed(embed_texts)
            rows = []
            for c, v in zip(chunks, vecs):
                rows.append((c.ord, c.page, c.section, c.text, _tokenize(c.text), v))
            index.replace_file(
                path=rel,
                sha256=sha,
                bytes_=stat.st_size,
                mtime=stat.st_mtime,
                parser_version=PARSER_VERSION,
                chunker_version=CHUNKER_VERSION,
                embedder=embedder.name,
                chunks=rows,
            )
            stats.chunks_added += len(rows)
            if existing is None:
                stats.files_new += 1
            else:
                stats.files_changed += 1
            if progress:
                progress(rel, f"ok ({len(rows)} chunks)")
        except Exception as e:
            stats.files_errored += 1
            stats.errors.append((str(path), repr(e)))
            if progress:
                progress(str(path), f"ERROR: {e}")

    stats.files_removed = index.delete_missing(present_rel)
    stats.seconds = time.perf_counter() - t0
    return stats

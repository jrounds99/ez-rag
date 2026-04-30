"""Top-level ingest pipeline: walk docs/, parse, chunk, embed, upsert."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
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


# ============================================================================
# Progress reporting
# ============================================================================

@dataclass
class IngestProgress:
    """Snapshot emitted by `ingest()` so UIs can render rich progress."""
    # Current event
    current_path: str = ""
    status: str = ""               # "parsing" | "embedding" | "ok" | "skipped" | "error"
    page: int | None = None
    snippet: str = ""              # 200-char preview of current/last chunk

    # Run-so-far totals (file-level granularity)
    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0            # bytes of fully-processed files
    bytes_total: int = 0
    chunks_done: int = 0

    # Timing / rate
    elapsed_s: float = 0.0
    eta_s: float | None = None     # remaining seconds, None until rate is known
    rate_bps: float = 0.0          # processed bytes per second (rolling)

    # Index growth
    db_bytes: int = 0              # current size of meta.sqlite on disk

    @property
    def bytes_pct(self) -> float:
        return (self.bytes_done / self.bytes_total) if self.bytes_total > 0 else 0.0

    @property
    def files_pct(self) -> float:
        return (self.files_done / self.files_total) if self.files_total > 0 else 0.0


# Backward-compat: progress callbacks may take either (path, status)
# or a single IngestProgress.
ProgressCb = Callable[..., None]


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


def _emit(progress: ProgressCb | None, snap: IngestProgress) -> None:
    """Send a progress snapshot. Tolerates both signatures —
    `progress(IngestProgress)` and the legacy `progress(path, status)`."""
    if progress is None:
        return
    try:
        progress(snap)
    except TypeError:
        progress(snap.current_path, snap.status)


def _db_size(path: Path) -> int:
    total = 0
    for sib in (path, path.with_suffix(path.suffix + "-wal"),
                path.with_suffix(path.suffix + "-shm")):
        try:
            total += sib.stat().st_size
        except OSError:
            pass
    return total


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

    bytes_done = 0
    files_done = 0
    chunks_done = 0

    def _early_snap(status: str, files_total: int = 0, bytes_total: int = 0) -> IngestProgress:
        elapsed = time.perf_counter() - t0
        return IngestProgress(
            status=status,
            files_done=files_done, files_total=files_total,
            bytes_done=bytes_done, bytes_total=bytes_total,
            chunks_done=chunks_done,
            elapsed_s=elapsed,
            db_bytes=_db_size(ws.meta_db_path),
        )

    # Pre-ingest narration so the UI never looks frozen during setup.
    _emit(progress, _early_snap("starting…"))
    _emit(progress, _early_snap("loading embedder (downloads on first use)"))
    embedder = make_embedder(cfg)
    _emit(progress, _early_snap(f"embedder ready ({embedder.name}, {embedder.dim}d)"))
    _emit(progress, _early_snap("opening vector index"))
    index = Index(ws.meta_db_path, embed_dim=embedder.dim)
    _emit(progress, _early_snap("scanning docs/"))

    docs_dir = ws.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = _walk_docs(docs_dir)
    stats.files_seen = len(files)
    _emit(progress, _early_snap(
        f"found {len(files)} files — measuring", files_total=len(files),
    ))
    bytes_total = sum(p.stat().st_size for p in files)
    _emit(progress, _early_snap(
        f"{len(files)} files · {bytes_total/1e6:.1f} MB to process",
        files_total=len(files), bytes_total=bytes_total,
    ))

    def snapshot(*, current_path: str = "", status: str = "",
                 page: int | None = None, snippet: str = "") -> IngestProgress:
        elapsed = time.perf_counter() - t0
        rate = (bytes_done / elapsed) if elapsed > 0 and bytes_done > 0 else 0.0
        eta = ((bytes_total - bytes_done) / rate) if rate > 0 else None
        return IngestProgress(
            current_path=current_path, status=status, page=page,
            snippet=snippet,
            files_done=files_done, files_total=len(files),
            bytes_done=bytes_done, bytes_total=bytes_total,
            chunks_done=chunks_done,
            elapsed_s=elapsed, eta_s=eta, rate_bps=rate,
            db_bytes=_db_size(ws.meta_db_path),
        )

    # Initial emission so UIs can size the bar.
    _emit(progress, snapshot(status="starting"))

    present_rel: set[str] = set()
    for path in files:
        rel = str(path.relative_to(ws.root))
        present_rel.add(rel)
        file_size = path.stat().st_size
        try:
            sha = file_sha256(path)
            existing = index.file_state(rel)
            if (
                not force
                and existing is not None
                and existing.sha256 == sha
            ):
                stats.files_skipped_unchanged += 1
                bytes_done += file_size
                files_done += 1
                _emit(progress, snapshot(current_path=rel, status="unchanged"))
                continue
            _emit(progress, snapshot(current_path=rel, status="parsing"))
            parser = get_parser(path)
            if parser is None:
                stats.files_unsupported += 1
                bytes_done += file_size
                files_done += 1
                continue
            sections = parser(path)
            if not sections:
                bytes_done += file_size
                files_done += 1
                _emit(progress, snapshot(current_path=rel, status="empty"))
                continue
            chunks = chunk_sections(
                sections,
                chunk_tokens=cfg.chunk_size,
                overlap_tokens=cfg.chunk_overlap,
            )
            if not chunks:
                bytes_done += file_size
                files_done += 1
                _emit(progress, snapshot(current_path=rel, status="no chunks"))
                continue
            # Pick a representative snippet (early-mid chunk) for live display.
            sample = chunks[min(len(chunks) - 1, len(chunks) // 2)]
            snippet = (sample.text or "").strip().replace("\n", " ")[:240]

            # Optional Anthropic-style Contextual Retrieval.
            embed_texts = [c.text for c in chunks]
            if cfg.enable_contextual and detect_backend(cfg) != "none":
                _emit(progress, snapshot(
                    current_path=rel,
                    status=f"contextualizing {len(chunks)} chunks",
                    page=sample.page, snippet=snippet,
                ))
                full_text = "\n\n".join(s.text for s in sections)
                embed_texts = [
                    contextualize_chunk(c.text, full_text, cfg) for c in chunks
                ]
            _emit(progress, snapshot(
                current_path=rel,
                status=f"embedding {len(chunks)} chunks",
                page=sample.page, snippet=snippet,
            ))
            # Embed in mini-batches so the UI updates during long embedding
            # passes (big PDFs with hundreds of chunks). For tiny files this
            # is still effectively a single call.
            BATCH = 16
            import numpy as np
            vec_chunks = []
            for i in range(0, len(embed_texts), BATCH):
                batch = embed_texts[i : i + BATCH]
                vec_chunks.append(np.asarray(embedder.embed(batch),
                                             dtype=np.float32))
                if len(embed_texts) > BATCH:
                    _emit(progress, snapshot(
                        current_path=rel,
                        status=f"embedding {min(i + BATCH, len(embed_texts))}"
                               f"/{len(embed_texts)} chunks",
                        page=sample.page, snippet=snippet,
                    ))
            vecs = np.concatenate(vec_chunks, axis=0) if vec_chunks else \
                   np.zeros((0, embedder.dim), dtype=np.float32)
            rows = []
            for c, v in zip(chunks, vecs):
                rows.append((c.ord, c.page, c.section, c.text, _tokenize(c.text), v))
            index.replace_file(
                path=rel,
                sha256=sha,
                bytes_=file_size,
                mtime=path.stat().st_mtime,
                parser_version=PARSER_VERSION,
                chunker_version=CHUNKER_VERSION,
                embedder=embedder.name,
                chunks=rows,
            )
            stats.chunks_added += len(rows)
            chunks_done += len(rows)
            if existing is None:
                stats.files_new += 1
            else:
                stats.files_changed += 1
            bytes_done += file_size
            files_done += 1
            _emit(progress, snapshot(
                current_path=rel,
                status=f"ok ({len(rows)} chunks)",
                page=sample.page, snippet=snippet,
            ))
        except Exception as e:
            stats.files_errored += 1
            stats.errors.append((str(path), repr(e)))
            bytes_done += file_size
            files_done += 1
            _emit(progress, snapshot(current_path=rel,
                                     status=f"ERROR: {e}"))

    stats.files_removed = index.delete_missing(present_rel)
    stats.seconds = time.perf_counter() - t0
    _emit(progress, snapshot(status="done"))
    return stats

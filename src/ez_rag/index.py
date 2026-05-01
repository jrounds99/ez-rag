"""SQLite-backed metadata + vector store. Embeddings stored as BLOBs of float32.

Suitable for tens of thousands of chunks; for larger corpora swap in LanceDB later.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    mtime REAL NOT NULL,
    parser_version TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    embedder TEXT NOT NULL,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    page INTEGER,
    section TEXT,
    text TEXT NOT NULL,
    tokens TEXT NOT NULL,           -- whitespace-joined lowercased word tokens for BM25
    embedding BLOB NOT NULL,
    UNIQUE(file_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, tokens, content='chunks', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, tokens) VALUES (new.id, new.text, new.tokens);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, tokens) VALUES ('delete', old.id, old.text, old.tokens);
END;
"""

# Idempotent column adds for upgrades from older schemas. SQLite has no
# `ADD COLUMN IF NOT EXISTS`, so we tolerate the OperationalError.
_MIGRATIONS = [
    "ALTER TABLE files ADD COLUMN chapters_json TEXT",
]


@dataclass
class FileRow:
    id: int
    path: str
    sha256: str
    bytes: int
    mtime: float
    n_chunks: int


@dataclass
class Hit:
    chunk_id: int
    file_id: int
    path: str
    page: int | None
    section: str
    text: str
    score: float
    source_kind: str   # "vec" | "fts" | "hybrid"


class Index:
    def __init__(self, db_path: Path, embed_dim: int):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the GUI worker thread and the
        # ThreadingHTTPServer request handlers can share one connection.
        # WAL mode + a write lock on mutating ops keeps this safe.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        for stmt in _MIGRATIONS:
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists; older DB already migrated
        self.conn.commit()
        import threading
        self._write_lock = threading.Lock()
        # Prefer the dimension already on disk over whatever we were told.
        existing = self._detect_dim()
        self.embed_dim = existing if existing is not None else embed_dim

    def _detect_dim(self) -> int | None:
        row = self.conn.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()
        if not row or not row[0]:
            return None
        return len(row[0]) // 4  # float32

    # ----- file lifecycle ---------------------------------------------------

    def file_state(self, path: str) -> FileRow | None:
        row = self.conn.execute(
            "SELECT id, path, sha256, bytes, mtime, n_chunks FROM files WHERE path = ?",
            (path,),
        ).fetchone()
        if not row:
            return None
        return FileRow(*row)

    def replace_file(
        self,
        *,
        path: str,
        sha256: str,
        bytes_: int,
        mtime: float,
        parser_version: str,
        chunker_version: str,
        embedder: str,
        chunks: list[tuple[int, int | None, str, str, str, np.ndarray]],
        chapters: list[dict] | None = None,
    ) -> int:
        """Insert/replace a file's chunks atomically. chunks tuples:
        (ord, page, section, text, tokens, vec).

        `chapters` is an optional list of `{"title", "start_ord", "end_ord",
        "start_page", "end_page"}` dicts persisted as JSON on the file row
        and consumed by chapter-aware retrieval.
        """
        import json as _json
        chapters_json = _json.dumps(chapters) if chapters else None
        with self._write_lock, self.conn:
            existing = self.file_state(path)
            if existing:
                self.conn.execute("DELETE FROM files WHERE id = ?", (existing.id,))
            cur = self.conn.execute(
                """INSERT INTO files(path, sha256, bytes, mtime, parser_version,
                                     chunker_version, embedder, n_chunks,
                                     chapters_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (path, sha256, bytes_, mtime, parser_version, chunker_version,
                 embedder, len(chunks), chapters_json),
            )
            file_id = cur.lastrowid
            self.conn.executemany(
                """INSERT INTO chunks(file_id, ord, page, section, text, tokens, embedding)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (file_id, ord_, page, section, text, tokens, _to_blob(vec))
                    for (ord_, page, section, text, tokens, vec) in chunks
                ],
            )
            return file_id

    def chapters_for_file(self, file_id: int) -> list[dict]:
        """Return the persisted chapter list for `file_id` (empty list if
        the file has no chapter metadata)."""
        import json as _json
        row = self.conn.execute(
            "SELECT chapters_json FROM files WHERE id = ?", (file_id,),
        ).fetchone()
        if not row or not row[0]:
            return []
        try:
            return _json.loads(row[0]) or []
        except Exception:
            return []

    def delete_missing(self, present_paths: set[str]) -> int:
        cur = self.conn.execute("SELECT id, path FROM files")
        ids_to_drop = [row[0] for row in cur if row[1] not in present_paths]
        if not ids_to_drop:
            return 0
        with self._write_lock, self.conn:
            self.conn.executemany(
                "DELETE FROM files WHERE id = ?",
                [(i,) for i in ids_to_drop],
            )
        return len(ids_to_drop)

    # ----- search ----------------------------------------------------------

    def all_embeddings(self) -> tuple[np.ndarray, list[int]]:
        rows = self.conn.execute(
            "SELECT id, embedding FROM chunks"
        ).fetchall()
        if not rows:
            return np.zeros((0, self.embed_dim), dtype=np.float32), []
        ids = [r[0] for r in rows]
        mat = np.stack([_from_blob(r[1], self.embed_dim) for r in rows])
        return mat, ids

    def get_chunks(self, ids: Iterable[int]) -> list[Hit]:
        ids = list(ids)
        if not ids:
            return []
        q = (
            "SELECT c.id, c.file_id, f.path, c.page, c.section, c.text "
            "FROM chunks c JOIN files f ON c.file_id = f.id "
            f"WHERE c.id IN ({','.join('?' * len(ids))})"
        )
        rows = self.conn.execute(q, ids).fetchall()
        by_id = {r[0]: r for r in rows}
        out: list[Hit] = []
        for cid in ids:
            r = by_id.get(cid)
            if not r:
                continue
            out.append(Hit(chunk_id=r[0], file_id=r[1], path=r[2],
                           page=r[3], section=r[4] or "", text=r[5],
                           score=0.0, source_kind="vec"))
        return out

    def fts_search(self, query: str, k: int) -> list[tuple[int, float]]:
        # bm25() score: lower is better. We invert.
        try:
            rows = self.conn.execute(
                """SELECT chunks.id, bm25(chunks_fts) AS rank
                   FROM chunks_fts
                   JOIN chunks ON chunks.id = chunks_fts.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (_fts_escape(query), k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r[0], 1.0 / (1.0 + r[1])) for r in rows]

    def stats(self) -> dict:
        n_files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        size_bytes = (
            self.conn.execute("SELECT COALESCE(SUM(bytes),0) FROM files").fetchone()[0]
        )
        return {"files": n_files, "chunks": n_chunks, "doc_bytes": size_bytes}


# ----- helpers ---------------------------------------------------------------

def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(buf: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.shape[0] != dim:
        # tolerate dim drift; pad/truncate
        if arr.shape[0] > dim:
            arr = arr[:dim]
        else:
            arr = np.concatenate([arr, np.zeros(dim - arr.shape[0], dtype=np.float32)])
    return arr


def _fts_escape(q: str) -> str:
    # Strip operators that confuse FTS5; keep words.
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in q)
    parts = [p for p in cleaned.split() if p]
    if not parts:
        return '""'
    return " OR ".join(f'"{p}"' for p in parts)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_stats(db_path: Path) -> dict | None:
    """Stats without instantiating an Index (no embedder probe).

    Returns None when the workspace has no DB yet.
    """
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        size_bytes = conn.execute("SELECT COALESCE(SUM(bytes),0) FROM files").fetchone()[0]
        embedder = conn.execute(
            "SELECT embedder FROM files ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return {
            "files": n_files,
            "chunks": n_chunks,
            "doc_bytes": size_bytes,
            "last_embedder": embedder[0] if embedder else None,
        }
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()

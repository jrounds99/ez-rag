"""Workspace layout management.

A workspace is just a directory containing:
    docs/                user files (anything goes here)
    .ezrag/
        config.toml
        meta.sqlite
        index/
        cache/
        models/
        ingest.log
    conversations/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config

WORKSPACE_MARKER = ".ezrag"


@dataclass
class Workspace:
    root: Path

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def state_dir(self) -> Path:
        return self.root / WORKSPACE_MARKER

    @property
    def config_path(self) -> Path:
        return self.state_dir / "config.toml"

    @property
    def meta_db_path(self) -> Path:
        return self.state_dir / "meta.sqlite"

    @property
    def index_dir(self) -> Path:
        return self.state_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.state_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.state_dir / "models"

    @property
    def conversations_dir(self) -> Path:
        return self.root / "conversations"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "ingest.log"

    def initialize(self) -> None:
        for d in (self.docs_dir, self.state_dir, self.index_dir,
                  self.cache_dir, self.models_dir, self.conversations_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            Config().save(self.config_path)
        gitignore = self.state_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n", encoding="utf-8")

    def load_config(self) -> Config:
        return Config.load(self.config_path)

    def is_initialized(self) -> bool:
        return self.state_dir.exists() and self.config_path.exists()

    def index_size_bytes(self) -> int:
        """Total disk footprint of the SQLite index (DB + WAL + SHM)."""
        total = 0
        for p in (self.meta_db_path,
                  self.meta_db_path.with_suffix(self.meta_db_path.suffix + "-wal"),
                  self.meta_db_path.with_suffix(self.meta_db_path.suffix + "-shm")):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def export_archive(self, dest: Path) -> Path:
        """Write a `.zip` archive containing config.toml + meta.sqlite (the
        portable index). Caches and model files are intentionally excluded.

        Returns the path written. The destination is created/overwritten.
        """
        import zipfile
        dest = Path(dest)
        if not dest.suffix:
            dest = dest.with_suffix(".zip")
        dest.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Manifest so importers can sanity-check the archive
            manifest = (
                "ez-rag workspace export\n"
                f"name: {self.root.name}\n"
                f"index_bytes: {self.index_size_bytes()}\n"
            )
            zf.writestr("ez-rag-export.txt", manifest)
            if self.config_path.exists():
                zf.write(self.config_path, ".ezrag/config.toml")
            for sib in (self.meta_db_path,
                        self.meta_db_path.with_suffix(
                            self.meta_db_path.suffix + "-wal"),
                        self.meta_db_path.with_suffix(
                            self.meta_db_path.suffix + "-shm")):
                if sib.exists():
                    zf.write(sib, f".ezrag/{sib.name}")
        return dest

    @classmethod
    def import_archive(cls, archive: Path, into: Path) -> "Workspace":
        """Extract a zip produced by `export_archive()` into a NEW workspace
        directory. Refuses to overwrite an existing initialized workspace."""
        import zipfile
        archive = Path(archive)
        into = Path(into).resolve()
        if (into / WORKSPACE_MARKER / "config.toml").exists():
            raise FileExistsError(
                f"{into} already has a .ezrag/ — refusing to overwrite. "
                f"Pick an empty path."
            )
        into.mkdir(parents=True, exist_ok=True)
        ws = cls(into)
        ws.initialize()  # creates docs/, .ezrag/ scaffolding
        with zipfile.ZipFile(archive, "r") as zf:
            for name in zf.namelist():
                if name.startswith(".ezrag/"):
                    zf.extract(name, into)
        return ws


def find_workspace(start: Path | None = None) -> Workspace | None:
    """Walk upward looking for a .ezrag/ directory. Returns None if not found."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / WORKSPACE_MARKER).is_dir():
            return Workspace(candidate)
    return None


def require_workspace(start: Path | None = None) -> Workspace:
    ws = find_workspace(start)
    if ws is None:
        raise SystemExit(
            "Not inside an ez-rag workspace. Run `ez-rag init .` to create one."
        )
    return ws

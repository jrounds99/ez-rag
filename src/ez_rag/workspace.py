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

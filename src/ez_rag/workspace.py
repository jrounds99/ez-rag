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

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

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
    def create_named(
        cls,
        name: str,
        parent_dir: Path,
        *,
        source_folders: list[Path] | None = None,
        clear_default: bool = True,
    ) -> "Workspace":
        """Create a new named workspace under `parent_dir/<slug(name)>/`.

        - `source_folders`: each folder is recursively scanned for supported
          file types and copied into the new docs/.
        - `clear_default`: if True (default), starts with an empty docs/.
          When False, any pre-existing files in the target docs/ are kept.

        Returns the new initialized Workspace. Raises FileExistsError if
        a workspace with this name already exists at this parent.
        """
        from .parsers import supported_extensions

        slug = "".join(c if c.isalnum() or c in " ._-" else "-" for c in name).strip()
        slug = slug.replace("  ", " ").strip("-_. ") or "rag"
        target = (parent_dir / slug).resolve()
        if (target / WORKSPACE_MARKER / "config.toml").exists():
            raise FileExistsError(
                f"A workspace already exists at {target}. "
                "Pick a different name or open it directly."
            )
        target.mkdir(parents=True, exist_ok=True)
        ws = cls(target)
        ws.initialize()

        if clear_default:
            for child in ws.docs_dir.iterdir():
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError:
                        pass

        copied = 0
        if source_folders:
            allowed = supported_extensions()
            seen_names: set[str] = set()
            for folder in source_folders:
                folder = Path(folder)
                if not folder.is_dir():
                    continue
                for f in folder.rglob("*"):
                    if not f.is_file():
                        continue
                    if f.suffix.lower() not in allowed:
                        continue
                    # Avoid name collisions across folders by prefixing
                    # the parent folder name when needed.
                    dest_name = f.name
                    if dest_name in seen_names:
                        dest_name = f"{folder.name}__{f.name}"
                    seen_names.add(dest_name)
                    try:
                        import shutil
                        shutil.copy2(f, ws.docs_dir / dest_name)
                        copied += 1
                    except Exception:
                        pass
        return ws

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
    """Walk upward looking for a real workspace. Returns None if not found.

    Requires both `<dir>/.ezrag/` AND `<dir>/.ezrag/config.toml`. The bare-
    dir check would otherwise misidentify `~/.ezrag/` (the global config /
    preview-cache home) as a workspace whenever the user runs ez-rag from
    anywhere under their home directory.
    """
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        marker = candidate / WORKSPACE_MARKER
        if marker.is_dir() and (marker / "config.toml").exists():
            return Workspace(candidate)
    return None


def require_workspace(start: Path | None = None) -> Workspace:
    ws = find_workspace(start)
    if ws is None:
        raise SystemExit(
            "Not inside an ez-rag workspace. Run `ez-rag init .` to create one."
        )
    return ws


# ============================================================================
# Global config — settings that apply across all workspaces
# ============================================================================
# Stored at ~/.ezrag/global.toml. Currently just the default RAGs folder.

GLOBAL_CONFIG_DIR = Path.home() / ".ezrag"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "global.toml"
DEFAULT_RAGS_DIR = Path.home() / "ez-rag-workspaces"


def _read_global() -> dict:
    if not GLOBAL_CONFIG_PATH.exists():
        return {}
    try:
        return tomllib.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_global(d: dict) -> None:
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, v in d.items():
        if isinstance(v, str):
            # TOML literal strings (single-quoted) don't interpret backslashes,
            # which matters for Windows paths like C:\Users\... — quoted strings
            # would mangle them as escape sequences.
            if "'" in v:
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            else:
                lines.append(f"{k} = '{v}'")
        elif isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        else:
            lines.append(f"{k} = {v}")
    GLOBAL_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_default_rags_dir() -> Path:
    """Where 'New RAG' creates workspaces by default. User-changeable via
    `set_default_rags_dir()` or the GUI's Storage settings card."""
    g = _read_global()
    p = g.get("default_rags_dir")
    if p:
        return Path(p).expanduser()
    return DEFAULT_RAGS_DIR


def set_default_rags_dir(path: Path) -> None:
    g = _read_global()
    g["default_rags_dir"] = str(Path(path).expanduser())
    _write_global(g)


def get_theme_name() -> str:
    """Active GUI palette name. Defaults to 'dark'."""
    return _read_global().get("theme", "dark") or "dark"


def set_theme_name(name: str) -> None:
    g = _read_global()
    g["theme"] = name
    _write_global(g)


def list_managed_rags(path: Path | None = None) -> list[Workspace]:
    """All initialized workspaces under `path` (defaults to the global RAGs dir).
    Sorted by directory mtime descending so the freshest is first."""
    base = (path or get_default_rags_dir()).expanduser()
    if not base.is_dir():
        return []
    out: list[Workspace] = []
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            ws = Workspace(child)
            if ws.is_initialized():
                out.append(ws)
    except OSError:
        return []
    out.sort(key=lambda w: w.root.stat().st_mtime, reverse=True)
    return out

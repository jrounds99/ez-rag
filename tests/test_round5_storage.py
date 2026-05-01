"""Round 5: Storage management.

Covers:
  - Workspace.create_named with multi-folder source ingestion + slug behavior
  - clear_default flag
  - Name-collision handling on copy
  - export_archive + import_archive roundtrip
  - get_default_rags_dir / set_default_rags_dir global config
  - list_managed_rags discovery + ordering
  - find_workspace walks upward
  - require_workspace raises when not in a workspace
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.index import read_stats
from ez_rag.ingest import ingest
from ez_rag import workspace as wsm
from ez_rag.workspace import (
    Workspace, find_workspace, require_workspace,
    list_managed_rags, get_default_rags_dir, set_default_rags_dir,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def _redirect_global_to_tmp(tmp: Path):
    """Point the global-config helpers at a tmp dir so we never touch
    the user's real ~/.ezrag."""
    wsm.GLOBAL_CONFIG_DIR = tmp
    wsm.GLOBAL_CONFIG_PATH = tmp / "global.toml"


# ---------------------------------------------------------------------------

def test_create_named_no_sources():
    print("\n[1] create_named with no source_folders -> empty docs/")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        ws = Workspace.create_named("My RAG", tmp)
        check("workspace initialized",
              ws.is_initialized(), "")
        check("docs/ exists and is empty",
              ws.docs_dir.is_dir() and not list(ws.docs_dir.iterdir()),
              f"contents={list(ws.docs_dir.iterdir())}")
        check("slug strips spaces sanely",
              ws.root.name == "My RAG", f"got {ws.root.name!r}")
        check("config.toml exists",
              ws.config_path.exists(), "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_named_slug_sanitization():
    print("\n[2] create_named slugifies bad characters")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        ws = Workspace.create_named("Weird/Name?*", tmp)
        # forbidden characters get replaced with -
        check("slug replaces / and ? and *",
              "/" not in ws.root.name and "?" not in ws.root.name
              and "*" not in ws.root.name,
              f"got {ws.root.name!r}")
        check("slug is non-empty",
              len(ws.root.name) >= 1, f"got {ws.root.name!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_named_with_sources():
    print("\n[3] create_named imports files from source_folders")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        # Build two source folders with mixed file types.
        a = tmp / "src_a"
        b = tmp / "src_b"
        a.mkdir()
        b.mkdir()
        (a / "alpha.md").write_text("alpha", encoding="utf-8")
        (a / "junk.bin").write_bytes(b"\x00\x01")    # unsupported
        (b / "beta.md").write_text("beta", encoding="utf-8")
        (b / "nested").mkdir()
        (b / "nested" / "gamma.md").write_text("gamma", encoding="utf-8")
        ws = Workspace.create_named("multi", tmp / "rags",
                                    source_folders=[a, b])
        names = sorted(p.name for p in ws.docs_dir.iterdir() if p.is_file())
        check("alpha.md imported", "alpha.md" in names, f"names={names}")
        check("beta.md imported", "beta.md" in names, f"names={names}")
        check("nested gamma.md imported (rglob recurses)",
              "gamma.md" in names, f"names={names}")
        check("unsupported junk.bin not imported",
              "junk.bin" not in names, f"names={names}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_named_handles_collisions():
    print("\n[4] create_named prefixes parent name on filename collision")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        a = tmp / "a"
        b = tmp / "b"
        a.mkdir()
        b.mkdir()
        (a / "shared.md").write_text("from-a", encoding="utf-8")
        (b / "shared.md").write_text("from-b", encoding="utf-8")
        ws = Workspace.create_named("collide", tmp / "rags",
                                    source_folders=[a, b])
        names = sorted(p.name for p in ws.docs_dir.iterdir() if p.is_file())
        check("first copy keeps original name",
              "shared.md" in names, f"names={names}")
        # second copy is prefixed with parent folder
        check("second copy is prefixed",
              any(n.startswith("b__") for n in names),
              f"names={names}")
        check("both files present",
              len([n for n in names if "shared.md" in n]) == 2,
              f"names={names}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_named_refuses_existing():
    print("\n[5] create_named raises FileExistsError if workspace exists")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        Workspace.create_named("dup", tmp)
        try:
            Workspace.create_named("dup", tmp)
            check("raised on duplicate", False, "no exception raised")
        except FileExistsError:
            check("raised FileExistsError on duplicate", True, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_named_clear_default_flag():
    print("\n[6] clear_default=False keeps pre-existing docs/ files")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        # Pre-create a workspace dir with content in docs/, then call
        # create_named with clear_default=False on a *different* path so we
        # can independently verify the flag.
        target = tmp / "keep"
        target.mkdir()
        (target / "docs").mkdir()
        (target / "docs" / "preexisting.md").write_text("pre", encoding="utf-8")
        # initialize manually so create_named will accept it (no .ezrag/config yet)
        # actually create_named will skip the FileExistsError because no config.toml
        ws = Workspace.create_named("keep", tmp, clear_default=False)
        # When clear_default=False, our "preexisting.md" should still be there.
        # Note: our slug is "keep" so target == ws.root.
        names = [p.name for p in ws.docs_dir.iterdir() if p.is_file()]
        check("clear_default=False keeps preexisting.md",
              "preexisting.md" in names, f"names={names}")

        # Now verify clear_default=True (default) drops them.
        target2 = tmp / "wipe"
        target2.mkdir()
        (target2 / "docs").mkdir()
        (target2 / "docs" / "old.md").write_text("old", encoding="utf-8")
        ws2 = Workspace.create_named("wipe", tmp, clear_default=True)
        names2 = [p.name for p in ws2.docs_dir.iterdir() if p.is_file()]
        check("clear_default=True drops old files",
              "old.md" not in names2, f"names2={names2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_export_import_roundtrip():
    print("\n[7] export_archive + import_archive roundtrip preserves index")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        # Build + ingest a workspace
        src = Workspace.create_named("src", tmp)
        (src.docs_dir / "doc.md").write_text(
            "border collies are intelligent\n", encoding="utf-8")
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     unload_llm_during_ingest=False)
        s = ingest(src, cfg=cfg)
        check("source ingest produced chunks",
              s.chunks_added > 0, f"got {s.chunks_added}")
        archive = src.export_archive(tmp / "exported.zip")
        check("archive file exists",
              archive.exists() and archive.stat().st_size > 0,
              f"size={archive.stat().st_size if archive.exists() else 0}")

        dest = tmp / "imported"
        ws2 = Workspace.import_archive(archive, dest)
        stats_imp = read_stats(ws2.meta_db_path)
        check("imported workspace has same chunk count",
              stats_imp["chunks"] == s.chunks_added,
              f"imp={stats_imp['chunks']} src={s.chunks_added}")
        check("imported workspace has config.toml",
              ws2.config_path.exists(), "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_import_refuses_overwrite():
    print("\n[8] import_archive refuses to overwrite an initialized workspace")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        src = Workspace.create_named("src", tmp)
        archive = src.export_archive(tmp / "x.zip")
        # second create_named in same place to make it look initialized
        existing = Workspace.create_named("there", tmp)
        try:
            Workspace.import_archive(archive, existing.root)
            check("import refused overwrite", False, "no exception")
        except FileExistsError:
            check("import refused overwrite", True, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_global_default_rags_dir_roundtrip():
    print("\n[9] set_default_rags_dir + get_default_rags_dir roundtrip")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    saved_dir = wsm.GLOBAL_CONFIG_DIR
    saved_path = wsm.GLOBAL_CONFIG_PATH
    try:
        _redirect_global_to_tmp(tmp)
        # initially nothing set -> default
        out = get_default_rags_dir()
        check("default returns DEFAULT_RAGS_DIR when unset",
              out == wsm.DEFAULT_RAGS_DIR,
              f"got {out!r}")
        # set + read
        custom = tmp / "my-rags"
        set_default_rags_dir(custom)
        check("global.toml was written",
              wsm.GLOBAL_CONFIG_PATH.exists(), "")
        out2 = get_default_rags_dir()
        check("get returns the value we set",
              out2 == custom, f"got {out2!r} expected {custom!r}")
    finally:
        wsm.GLOBAL_CONFIG_DIR = saved_dir
        wsm.GLOBAL_CONFIG_PATH = saved_path
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_managed_rags():
    print("\n[10] list_managed_rags discovers initialized workspaces")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        # Create three workspaces with slight time gaps so mtime differs
        a = Workspace.create_named("a", tmp)
        time.sleep(0.05)
        b = Workspace.create_named("b", tmp)
        time.sleep(0.05)
        c = Workspace.create_named("c", tmp)
        # Add a non-workspace dir; should be ignored
        (tmp / "not_a_workspace").mkdir()
        out = list_managed_rags(tmp)
        names = [w.root.name for w in out]
        check("found exactly 3 workspaces",
              len(out) == 3, f"got {len(out)}: {names}")
        check("non-workspace folder ignored",
              "not_a_workspace" not in names, f"names={names}")
        # Sorted by mtime descending — c should be first.
        check("most recent first",
              names[0] == "c", f"order={names}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_managed_rags_missing_dir():
    print("\n[11] list_managed_rags returns [] for missing path")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        bogus = tmp / "does-not-exist"
        out = list_managed_rags(bogus)
        check("missing dir -> []", out == [], f"got {out!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_find_workspace_walks_up():
    print("\n[12] find_workspace walks up the tree")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        ws = Workspace.create_named("up", tmp)
        deep = ws.root / "docs" / "subdir" / "more"
        deep.mkdir(parents=True)
        found = find_workspace(deep)
        check("finds workspace from deep child",
              found is not None and found.root.resolve() == ws.root.resolve(),
              f"found={found.root if found else None}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_find_workspace_returns_none_when_outside():
    print("\n[13] find_workspace returns None when nowhere on the path")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        out = find_workspace(tmp)
        check("returns None outside any .ezrag/ tree",
              out is None, f"got {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_require_workspace_raises():
    print("\n[14] require_workspace raises SystemExit when not in workspace")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        try:
            require_workspace(tmp)
            check("require_workspace raised", False, "no exception")
        except SystemExit:
            check("require_workspace raised SystemExit", True, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_index_size_bytes():
    print("\n[15] index_size_bytes accumulates db + wal + shm")
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round5_"))
    try:
        ws = Workspace.create_named("size", tmp)
        # Empty workspace yet — meta.sqlite doesn't exist until ingest.
        check("empty workspace -> 0 bytes",
              ws.index_size_bytes() == 0,
              f"got {ws.index_size_bytes()}")
        (ws.docs_dir / "x.md").write_text("hello\n", encoding="utf-8")
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     unload_llm_during_ingest=False)
        ingest(ws, cfg=cfg)
        check("after ingest -> >0 bytes",
              ws.index_size_bytes() > 0,
              f"got {ws.index_size_bytes()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

def main():
    print("[round 5] storage / multi-folder / global config")
    test_create_named_no_sources()
    test_create_named_slug_sanitization()
    test_create_named_with_sources()
    test_create_named_handles_collisions()
    test_create_named_refuses_existing()
    test_create_named_clear_default_flag()
    test_export_import_roundtrip()
    test_import_refuses_overwrite()
    test_global_default_rags_dir_roundtrip()
    test_list_managed_rags()
    test_list_managed_rags_missing_dir()
    test_find_workspace_walks_up()
    test_find_workspace_returns_none_when_outside()
    test_require_workspace_raises()
    test_index_size_bytes()

    print(f"\n=== Round 5 summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for name, det in FAIL:
            print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

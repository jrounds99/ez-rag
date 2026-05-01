"""Round 4: Ingest pipeline.

Covers:
  - IngestProgress fields populate during run
  - Idempotent re-ingest leaves index unchanged
  - File deletion triggers chunk cleanup via delete_missing
  - --force re-embeds even unchanged files
  - Per-batch embedding produces multiple progress emissions
  - unload_ollama_model is called when configured
  - Contextual retrieval pipeline calls contextualize_chunk
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder
from ez_rag.index import Index, read_stats
from ez_rag.ingest import IngestProgress, ingest
from ez_rag import generate as gen
from ez_rag import models as models_mod
from ez_rag.workspace import Workspace


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def make_tmp_ws() -> Workspace:
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round4_"))
    ws = Workspace(tmp)
    ws.initialize()
    (ws.docs_dir / "alpha.md").write_text(
        "# Alpha\nThe alpha document mentions border collies.\n"
        "It also mentions Newton's laws of motion in passing.\n",
        encoding="utf-8",
    )
    (ws.docs_dir / "beta.md").write_text(
        "# Beta\nBeta covers caramelization and the Maillard reaction.\n",
        encoding="utf-8",
    )
    return ws


# ---------------------------------------------------------------------------

def test_progress_emits_snapshots(ws, cfg):
    print("\n[1] ingest emits IngestProgress with correct field types")
    seen: list[IngestProgress] = []
    ingest(ws, cfg=cfg, progress=lambda p: seen.append(p))
    check("progress was called", len(seen) >= 1, f"n={len(seen)}")
    if not seen:
        return
    p = seen[-1]
    check("final snapshot is IngestProgress",
          isinstance(p, IngestProgress), f"type={type(p)}")
    check("files_total set", p.files_total >= 2, f"got {p.files_total}")
    check("files_done == files_total",
          p.files_done == p.files_total,
          f"done={p.files_done} total={p.files_total}")
    check("bytes_total > 0", p.bytes_total > 0, f"got {p.bytes_total}")
    check("elapsed_s > 0", p.elapsed_s > 0, f"got {p.elapsed_s}")
    check("db_bytes > 0", p.db_bytes > 0, f"got {p.db_bytes}")
    check("final status is 'done'", p.status == "done", f"got {p.status!r}")


def test_progress_legacy_callback_signature(ws, cfg):
    print("\n[2] legacy progress callback (path, status) still works")
    # blow away workspace so we get fresh ingestion events
    shutil.rmtree(ws.root, ignore_errors=True)
    ws.initialize()
    (ws.docs_dir / "x.md").write_text("hello world\n", encoding="utf-8")
    calls = []
    ingest(ws, cfg=cfg, progress=lambda path, status: calls.append((path, status)))
    check("legacy progress called",
          len(calls) >= 1, f"n={len(calls)}")
    check("calls are 2-tuples of strings",
          all(isinstance(c, tuple) and len(c) == 2 for c in calls),
          f"sample={calls[:1]}")


def test_idempotent_reingest():
    print("\n[3] re-ingest is a no-op when files unchanged")
    ws = make_tmp_ws()
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     enable_contextual=False)
        s1 = ingest(ws, cfg=cfg)
        s2 = ingest(ws, cfg=cfg)
        check("first ingest added chunks", s1.chunks_added > 0,
              f"got {s1.chunks_added}")
        check("second ingest added 0 chunks", s2.chunks_added == 0,
              f"got {s2.chunks_added}")
        check("second ingest marked all unchanged",
              s2.files_skipped_unchanged == s2.files_seen,
              f"skipped={s2.files_skipped_unchanged} seen={s2.files_seen}")
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_force_reingest():
    print("\n[4] force=True re-embeds even unchanged files")
    ws = make_tmp_ws()
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     enable_contextual=False)
        s1 = ingest(ws, cfg=cfg)
        s2 = ingest(ws, cfg=cfg, force=True)
        check("force re-ingest added the same chunk count",
              s2.chunks_added == s1.chunks_added,
              f"first={s1.chunks_added} second={s2.chunks_added}")
        check("force re-ingest skipped 0 unchanged",
              s2.files_skipped_unchanged == 0,
              f"skipped={s2.files_skipped_unchanged}")
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_deleted_file_removed_from_index():
    print("\n[5] removing a docs file deletes its chunks on next ingest")
    ws = make_tmp_ws()
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5")
        s1 = ingest(ws, cfg=cfg)
        chunk_count_before = read_stats(ws.meta_db_path)["chunks"]
        # delete one of the docs
        (ws.docs_dir / "alpha.md").unlink()
        s2 = ingest(ws, cfg=cfg)
        chunk_count_after = read_stats(ws.meta_db_path)["chunks"]
        check("delete_missing reported removed file",
              s2.files_removed == 1, f"got {s2.files_removed}")
        check("index lost chunks for the removed file",
              chunk_count_after < chunk_count_before,
              f"before={chunk_count_before} after={chunk_count_after}")
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_progress_includes_intermediate_statuses():
    print("\n[6] progress emits 'parsing' and 'embedding' statuses")
    ws = make_tmp_ws()
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     embed_batch_size=1,    # force per-chunk emissions
                     chunk_size=80, chunk_overlap=10)
        statuses = []
        ingest(ws, cfg=cfg, progress=lambda p: statuses.append(p.status))
        check("emitted 'parsing' status",
              any("parsing" in s for s in statuses), f"statuses={statuses}")
        check("emitted 'embedding' status",
              any("embedding" in s for s in statuses), f"statuses={statuses}")
        check("emitted final 'done'",
              statuses[-1] == "done", f"last={statuses[-1]!r}")
        check("non-empty 'starting' setup messages emitted",
              any("starting" in s or "loading" in s or "scanning" in s
                  for s in statuses),
              "expected setup narration")
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_unload_called_when_configured(monkey_safe=True):
    print("\n[7] unload_ollama_model called only when configured + ollama backend")
    ws = make_tmp_ws()
    saved_unload = models_mod.unload_ollama_model
    saved_detect = gen.detect_backend
    # patch the module-level binding ingest imported
    import ez_rag.ingest as ing_mod
    saved_ing_unload = ing_mod.unload_ollama_model
    saved_ing_detect = ing_mod.detect_backend
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     unload_llm_during_ingest=True,
                     enable_contextual=False)
        # Branch A: backend=ollama, contextual off, unload on -> unload called
        calls_a = []
        def fake_unload_a(url, tag, **k):
            calls_a.append((url, tag))
            return True
        ing_mod.unload_ollama_model = fake_unload_a
        ing_mod.detect_backend = lambda c: "ollama"
        ingest(ws, cfg=cfg)
        check("unload called when contextual off + ollama backend",
              len(calls_a) == 1, f"calls={calls_a}")
        check("unload received configured model tag",
              calls_a and calls_a[0][1] == cfg.llm_model,
              f"calls={calls_a}")

        # Branch B: backend=ollama, contextual ON -> NOT unloaded
        shutil.rmtree(ws.root, ignore_errors=True)
        ws.initialize()
        (ws.docs_dir / "x.md").write_text("hello\n", encoding="utf-8")
        calls_b = []
        def fake_unload_b(url, tag, **k):
            calls_b.append(tag)
            return True
        ing_mod.unload_ollama_model = fake_unload_b
        cfg2 = Config(embedder_provider="fastembed",
                      embedder_model="BAAI/bge-small-en-v1.5",
                      unload_llm_during_ingest=True,
                      enable_contextual=True)
        # contextual=True triggers contextualize path, stub it to no-op
        import ez_rag.ingest as ingmod
        saved_ctx = ingmod.contextualize_chunk
        ingmod.contextualize_chunk = lambda text, full, cfg: text
        try:
            ingest(ws, cfg=cfg2)
        finally:
            ingmod.contextualize_chunk = saved_ctx
        check("unload NOT called when contextual is ON",
              len(calls_b) == 0, f"calls={calls_b}")

        # Branch C: backend=none -> NOT unloaded
        shutil.rmtree(ws.root, ignore_errors=True)
        ws.initialize()
        (ws.docs_dir / "x.md").write_text("hello\n", encoding="utf-8")
        calls_c = []
        ing_mod.unload_ollama_model = lambda url, tag, **k: calls_c.append(tag) or True
        ing_mod.detect_backend = lambda c: "none"
        ingest(ws, cfg=cfg)
        check("unload NOT called when backend is 'none'",
              len(calls_c) == 0, f"calls={calls_c}")
    finally:
        models_mod.unload_ollama_model = saved_unload
        gen.detect_backend = saved_detect
        ing_mod.unload_ollama_model = saved_ing_unload
        ing_mod.detect_backend = saved_ing_detect
        shutil.rmtree(ws.root, ignore_errors=True)


def test_contextual_pipeline_invokes_contextualize_chunk():
    print("\n[8] enable_contextual=True calls contextualize_chunk per chunk")
    ws = make_tmp_ws()
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     enable_contextual=True,
                     chunk_size=80, chunk_overlap=10,
                     unload_llm_during_ingest=False)
        import ez_rag.ingest as ingmod
        # Stub backend detection so the contextual branch fires.
        saved_detect = ingmod.detect_backend
        ingmod.detect_backend = lambda c: "ollama"
        # Stub contextualize_chunk to count calls.
        saved_ctx = ingmod.contextualize_chunk
        n = {"calls": 0}
        def fake_ctx(text, doc, cfg):
            n["calls"] += 1
            return f"[ctx] {text}"
        ingmod.contextualize_chunk = fake_ctx
        try:
            stats = ingest(ws, cfg=cfg)
        finally:
            ingmod.detect_backend = saved_detect
            ingmod.contextualize_chunk = saved_ctx
        check("contextualize_chunk was called per chunk",
              n["calls"] == stats.chunks_added,
              f"calls={n['calls']} chunks={stats.chunks_added}")
        check("at least one call happened",
              n["calls"] >= 1, "")
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


def test_embed_batches_emit_progress():
    print("\n[9] embed_batch_size triggers per-batch progress emissions")
    # Use small batch size + small chunks so we get many batches
    ws = Path(tempfile.mkdtemp(prefix="ezrag_round4_batch_"))
    w = Workspace(ws)
    w.initialize()
    # write a large-ish doc so it produces many chunks
    body = "\n\n".join(
        f"Paragraph {i}: " + "border collie sheep herding " * 30
        for i in range(8)
    )
    (w.docs_dir / "big.md").write_text(body, encoding="utf-8")
    try:
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5",
                     chunk_size=40, chunk_overlap=0,
                     embed_batch_size=2)
        statuses = []
        stats = ingest(w, cfg=cfg, progress=lambda p: statuses.append(p.status))
        embed_msgs = [s for s in statuses if "embedding" in s]
        check("emitted multiple embedding progress lines",
              len(embed_msgs) >= 2,
              f"got {len(embed_msgs)} embedding emissions: {embed_msgs[:5]}")
        check("ingest still produced chunks",
              stats.chunks_added > 0, f"got {stats.chunks_added}")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_unsupported_files_counted():
    print("\n[10] unsupported file extensions are counted, not crashed")
    ws = make_tmp_ws()
    try:
        # Add a binary file with an unsupported extension
        (ws.docs_dir / "weirdo.xyz").write_bytes(b"\x00\x01\x02 not text \xff")
        cfg = Config(embedder_provider="fastembed",
                     embedder_model="BAAI/bge-small-en-v1.5")
        stats = ingest(ws, cfg=cfg)
        # .xyz is filtered out by _walk_docs, so it isn't even seen.
        check("unsupported file is filtered before walk",
              stats.files_seen == 2,
              f"saw {stats.files_seen}, expected 2")
        check("no errors", stats.files_errored == 0,
              f"errors={stats.errors}")
    finally:
        shutil.rmtree(ws.root, ignore_errors=True)


# ---------------------------------------------------------------------------

def main():
    ws = make_tmp_ws()
    cfg = Config(
        embedder_provider="fastembed",
        embedder_model="BAAI/bge-small-en-v1.5",
        unload_llm_during_ingest=False,    # don't touch ollama by default
        enable_contextual=False,
    )
    print(f"[setup] tmp workspace: {ws.root}")
    try:
        test_progress_emits_snapshots(ws, cfg)
        test_progress_legacy_callback_signature(ws, cfg)
        test_idempotent_reingest()
        test_force_reingest()
        test_deleted_file_removed_from_index()
        test_progress_includes_intermediate_statuses()
        test_unload_called_when_configured()
        test_contextual_pipeline_invokes_contextualize_chunk()
        test_embed_batches_emit_progress()
        test_unsupported_files_counted()
    finally:
        try:
            shutil.rmtree(ws.root, ignore_errors=True)
        except Exception:
            pass

    print(f"\n=== Round 4 summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for name, det in FAIL:
            print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

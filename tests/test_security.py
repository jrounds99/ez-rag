"""Proprietary-data mode tests: local-only enforcement + workspace lock.

Covers:
  1. is_local_url matrix (loopback / RFC-1918 / public / DNS names).
  2. check_local_only + check_agent_provider enforcement semantics.
  3. Encrypt/decrypt round-trip + wrong-passphrase rejection.
  4. Full workspace lock/unlock: real index -> lock (plaintext gone,
     WAL folded) -> operations refuse -> unlock -> retrieval works.
  5. Chat path blocks a non-local LLM URL when the mode is on.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.security import (
    ProprietaryDataViolation, WorkspaceLockedError, WrongPassphraseError,
    check_agent_provider, check_local_only, decrypt_file, encrypt_file,
    is_local_url, is_locked, lock_workspace, require_unlocked,
    unlock_workspace,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    print("\n[1] is_local_url matrix")
    local_yes = [
        "http://127.0.0.1:11434", "http://localhost:11434",
        "https://localhost", "http://10.0.0.5:11434",
        "http://192.168.1.20:11434", "http://172.16.0.9:8080",
        "http://[::1]:11434",
    ]
    local_no = [
        "https://api.openai.com/v1", "https://api.anthropic.com",
        "http://8.8.8.8:11434", "https://my-ollama.example.com",
        "http://172.32.0.1:11434",   # just OUTSIDE the 172.16/12 block
        "", "not a url",
    ]
    for u in local_yes:
        check(f"local: {u or '(empty)'}", is_local_url(u))
    for u in local_no:
        check(f"non-local: {u or '(empty)'}", not is_local_url(u))

    print("\n[2] enforcement semantics")
    cfg = Config()
    check_local_only(cfg, "https://api.openai.com", "x")   # off -> no-op
    check("mode OFF -> any URL allowed", True)
    cfg.proprietary_data = True
    check_local_only(cfg, "http://127.0.0.1:11434", "x")
    check("mode ON -> local URL allowed", True)
    try:
        check_local_only(cfg, "https://api.openai.com", "LLM endpoint")
        check("mode ON -> public URL blocked", False)
    except ProprietaryDataViolation as e:
        check("mode ON -> public URL blocked", "Proprietary" in str(e))
    cfg.agent_provider = "openai"
    try:
        check_agent_provider(cfg)
        check("mode ON -> cloud agent blocked", False)
    except ProprietaryDataViolation:
        check("mode ON -> cloud agent blocked", True)
    cfg.agent_provider = "same"
    check_agent_provider(cfg)
    check("mode ON -> agent 'same' allowed", True)

    print("\n[3] encrypt/decrypt round-trip")
    d = Path(tempfile.mkdtemp(prefix="ezrag_sec_"))
    src = d / "plain.bin"
    src.write_bytes(b"corpus text " * 5000)
    enc = d / "cipher.enc"
    encrypt_file(src, enc, "hunter2hunter2")
    check("ciphertext differs from plaintext",
          enc.read_bytes()[9:] != src.read_bytes())
    check("magic header present", enc.read_bytes()[:9] == b"EZRAGENC1")
    out = d / "roundtrip.bin"
    decrypt_file(enc, out, "hunter2hunter2")
    check("round-trip identical", out.read_bytes() == src.read_bytes())
    try:
        decrypt_file(enc, d / "x.bin", "wrong-passphrase")
        check("wrong passphrase rejected", False)
    except WrongPassphraseError:
        check("wrong passphrase rejected", True)

    print("\n[4] workspace lock/unlock end-to-end")
    from ez_rag.workspace import Workspace
    from ez_rag.ingest import ingest
    from ez_rag.index import Index
    from ez_rag.embed import make_embedder
    from ez_rag.retrieve import hybrid_search

    tmp = Path(tempfile.mkdtemp(prefix="ezrag_lockws_"))
    ws = Workspace(tmp)
    ws.initialize()
    (ws.docs_dir / "secret.md").write_text(
        "Project Aurora launches in March under codename BLUEBIRD.",
        encoding="utf-8")
    cfg = ws.load_config()
    cfg.embedder_provider = "fastembed"
    ingest(ws, cfg=cfg)

    # Corpus text lives in db+WAL until checkpoint — check both.
    plaintext = (tmp / ".ezrag" / "meta.sqlite").read_bytes()
    wal = tmp / ".ezrag" / "meta.sqlite-wal"
    if wal.exists():
        plaintext += wal.read_bytes()
    check("index contains corpus text before lock",
          b"BLUEBIRD" in plaintext)

    # Drop any lingering in-process SQLite handles (ingest's Index) so
    # the lock can fold the WAL — mirrors a fresh `ez-rag lock` process.
    import gc
    gc.collect()

    lock_workspace(tmp, "s3cret-passphrase")
    check("locked state detected", is_locked(tmp))
    check("plaintext db gone", not (tmp / ".ezrag" / "meta.sqlite").exists())
    check("no WAL/SHM left behind",
          not (tmp / ".ezrag" / "meta.sqlite-wal").exists()
          and not (tmp / ".ezrag" / "meta.sqlite-shm").exists())
    enc_bytes = (tmp / ".ezrag" / "meta.sqlite.enc").read_bytes()
    check("ciphertext does not leak corpus text",
          b"BLUEBIRD" not in enc_bytes)

    try:
        require_unlocked(tmp)
        check("require_unlocked refuses while locked", False)
    except WorkspaceLockedError:
        check("require_unlocked refuses while locked", True)
    try:
        ingest(ws, cfg=cfg)
        check("ingest refuses while locked", False)
    except WorkspaceLockedError:
        check("ingest refuses while locked", True)

    try:
        unlock_workspace(tmp, "wrong")
        check("unlock rejects wrong passphrase", False)
    except WrongPassphraseError:
        check("unlock rejects wrong passphrase", True)
    check("still locked after failed unlock", is_locked(tmp))

    unlock_workspace(tmp, "s3cret-passphrase")
    check("unlocked", not is_locked(tmp))
    emb = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=emb.dim)
    hits = hybrid_search(query="What is the codename?",
                          embedder=emb, index=idx, k=2)
    check("retrieval works after unlock",
          bool(hits) and "BLUEBIRD" in hits[0].text)

    print("\n[5] chat path blocks non-local LLM URL")
    from ez_rag import generate as gen
    cfg2 = Config()
    cfg2.proprietary_data = True
    cfg2.llm_url = "https://sneaky-exfil.example.com"
    cfg2.llm_model = "qwen2.5:7b"
    try:
        gen._ollama_chat(cfg2, [{"role": "user", "content": "hi"}])
        check("non-local chat URL blocked", False)
    except ProprietaryDataViolation:
        check("non-local chat URL blocked", True)
    except Exception as e:
        check("non-local chat URL blocked", False,
              f"wrong exception: {type(e).__name__}: {e}")

    print(f"\n=== security summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

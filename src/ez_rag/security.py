"""Proprietary-data mode: local-only enforcement + workspace encryption.

Two independent protections, both driven by `cfg.proprietary_data`:

1. **Local-only enforcement.** Every LLM/embedding endpoint must resolve
   to this machine or the private LAN (loopback or RFC-1918). Cloud
   agent providers (OpenAI/Anthropic) are refused outright. The result:
   with the flip ON, no prompt, chunk, or document text can be sent to
   a public endpoint, even by misconfiguration.

2. **Encryption at rest (workspace lock).** `ez-rag lock` encrypts the
   index — `meta.sqlite`, which contains every extracted chunk of your
   documents plus their embeddings — with AES-256-GCM. The key is
   derived from your passphrase via scrypt (n=2^15, r=8, p=1), so
   offline brute-force is expensive. `ez-rag unlock` restores it.
   While LOCKED, ingest and chat refuse to run (and cannot accidentally
   create a fresh empty index over the encrypted one).

HONEST LIMITS (documented, not hidden):
  - While UNLOCKED, the index is plaintext on disk — SQLite (and FTS5
    search) can't operate on ciphertext. Pair the lock with full-disk
    encryption (BitLocker / FileVault / LUKS) for protection while
    you work.
  - The lock covers what EZ-RAG generates (the index). Your original
    documents in docs/ are yours to protect — they were on your disk
    before ez-rag saw them.
  - Passphrases are never stored. Lose it and the index is gone —
    re-ingest from the source documents to rebuild.
"""
from __future__ import annotations

import ipaddress
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

MAGIC = b"EZRAGENC1"
_SALT_LEN = 16
_NONCE_LEN = 12


class ProprietaryDataViolation(RuntimeError):
    """A cloud/non-local endpoint was blocked by proprietary-data mode."""


class WorkspaceLockedError(RuntimeError):
    """Operation refused because the workspace index is encrypted."""


class WrongPassphraseError(RuntimeError):
    pass


# ============================================================================
# Local-only enforcement
# ============================================================================

def is_local_url(url: str) -> bool:
    """True if `url` points at this machine or the private LAN.

    Accepts loopback (localhost / 127.x / ::1) and RFC-1918 private
    ranges (10/8, 172.16/12, 192.168/16) — data stays on hardware you
    control. Everything else (public IPs, DNS names) is non-local.
    """
    try:
        host = urlparse(url if "://" in url else f"http://{url}").hostname
    except Exception:
        return False
    if not host:
        return False
    if host.lower() in ("localhost",):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        # A DNS hostname — can't verify where it points without trusting
        # DNS, which an exfiltration attempt could control. Refuse.
        return False


def check_local_only(cfg, url: str, what: str = "endpoint") -> None:
    """Raise ProprietaryDataViolation when proprietary mode is on and
    `url` isn't local. No-op when the mode is off."""
    if not getattr(cfg, "proprietary_data", False):
        return
    if not is_local_url(url or ""):
        raise ProprietaryDataViolation(
            f"Proprietary-data mode: refusing non-local {what} "
            f"'{url}'. Only localhost / private-LAN endpoints are "
            f"allowed while `proprietary_data = true`."
        )


def check_agent_provider(cfg) -> None:
    """Refuse cloud agent providers in proprietary mode."""
    if not getattr(cfg, "proprietary_data", False):
        return
    provider = (getattr(cfg, "agent_provider", "same") or "same").lower()
    if provider in ("openai", "anthropic"):
        raise ProprietaryDataViolation(
            f"Proprietary-data mode: agent provider '{provider}' sends "
            f"retrieved context to a cloud API and is disabled. Set "
            f"agent_provider = 'same' to use your local model."
        )


# ============================================================================
# Encryption at rest
# ============================================================================

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    kdf = Scrypt(salt=salt, length=32, n=2 ** 15, r=8, p=1)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_file(src: Path, dst: Path, passphrase: str) -> None:
    """AES-256-GCM encrypt src -> dst (MAGIC | salt | nonce | ciphertext)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = _derive_key(passphrase, salt)
    data = src.read_bytes()
    ct = AESGCM(key).encrypt(nonce, data, MAGIC)
    dst.write_bytes(MAGIC + salt + nonce + ct)


def decrypt_file(src: Path, dst: Path, passphrase: str) -> None:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    blob = src.read_bytes()
    if not blob.startswith(MAGIC):
        raise ValueError(f"{src.name} is not an ez-rag encrypted file")
    off = len(MAGIC)
    salt = blob[off:off + _SALT_LEN]
    nonce = blob[off + _SALT_LEN:off + _SALT_LEN + _NONCE_LEN]
    ct = blob[off + _SALT_LEN + _NONCE_LEN:]
    key = _derive_key(passphrase, salt)
    try:
        data = AESGCM(key).decrypt(nonce, ct, MAGIC)
    except InvalidTag:
        raise WrongPassphraseError(
            "Wrong passphrase (or the encrypted file is corrupted)."
        ) from None
    dst.write_bytes(data)


# ============================================================================
# Workspace lock / unlock
# ============================================================================

def _db_path(ws_root: Path) -> Path:
    return Path(ws_root) / ".ezrag" / "meta.sqlite"


def _enc_path(ws_root: Path) -> Path:
    return _db_path(ws_root).with_suffix(".sqlite.enc")


def is_locked(ws_root: Path) -> bool:
    return _enc_path(ws_root).is_file() and not _db_path(ws_root).is_file()


def require_unlocked(ws_root: Path) -> None:
    if is_locked(ws_root):
        raise WorkspaceLockedError(
            "This workspace's index is encrypted (proprietary-data lock). "
            "Unlock it first: `ez-rag unlock` (CLI) or Settings → "
            "Proprietary data (GUI)."
        )


def lock_workspace(ws_root: Path, passphrase: str) -> Path:
    """Encrypt the index. Returns the .enc path.

    Checkpoints the SQLite WAL first so the single encrypted file holds
    the complete database, then removes the plaintext db + WAL/SHM.
    """
    ws_root = Path(ws_root)
    db = _db_path(ws_root)
    enc = _enc_path(ws_root)
    if is_locked(ws_root):
        return enc
    if not db.is_file():
        raise FileNotFoundError(f"No index at {db} — nothing to lock.")
    if not passphrase or len(passphrase) < 8:
        raise ValueError("Passphrase must be at least 8 characters.")

    # Fold WAL into the main db file so nothing is left behind plaintext.
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # Best-effort: leaving WAL mode makes the db self-contained even
        # if a straggler connection blocks the journal-mode change (the
        # TRUNCATE checkpoint above already emptied the -wal file).
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()

    # Drop the in-memory embedding matrix — it holds the corpus vectors.
    try:
        from .index import invalidate_matrix_cache
        invalidate_matrix_cache(db)
    except Exception:
        pass

    encrypt_file(db, enc, passphrase)
    # Verify round-trip integrity BEFORE deleting the plaintext.
    probe = enc.with_suffix(".probe")
    try:
        decrypt_file(enc, probe, passphrase)
        if probe.read_bytes() != db.read_bytes():
            raise RuntimeError("Encryption verification failed — "
                               "plaintext left in place.")
    finally:
        probe.unlink(missing_ok=True)

    db.unlink()
    for sib in (db.with_suffix(".sqlite-wal"), db.with_suffix(".sqlite-shm")):
        sib.unlink(missing_ok=True)
    return enc


def unlock_workspace(ws_root: Path, passphrase: str) -> Path:
    """Decrypt the index back into place. Returns the db path."""
    ws_root = Path(ws_root)
    db = _db_path(ws_root)
    enc = _enc_path(ws_root)
    if not enc.is_file():
        raise FileNotFoundError(f"No encrypted index at {enc}.")
    if db.is_file():
        raise RuntimeError(
            f"Both {db.name} and {enc.name} exist — refusing to "
            f"overwrite. Move one aside and retry."
        )
    decrypt_file(enc, db, passphrase)     # raises WrongPassphraseError
    enc.unlink()
    return db

# Proprietary data mode

The "my documents must never leave this machine" flip, for corpora you
genuinely cannot risk: client files, unreleased research, contracts,
anything under NDA. One config switch plus an encrypted lock for the
index. Everything here is free and local, like the rest of ez-rag.

## TL;DR

```toml
# <workspace>/.ezrag/config.toml
proprietary_data = true
```

```bash
ez-rag lock      # encrypt the index when you step away
ez-rag unlock    # decrypt it to work again
```

GUI: Settings → **Proprietary data** — the switch, plus Lock/Unlock
with a passphrase field.

## What the switch enforces

With `proprietary_data = true`, ez-rag becomes actively hostile to
data leaving your hardware:

1. **Local-only endpoints.** Every LLM and embedding call must target
   `localhost` or a private-LAN address (RFC-1918: `10.x`,
   `172.16-31.x`, `192.168.x`). Anything else — a public IP, any DNS
   hostname — raises `ProprietaryDataViolation` *instead of sending
   your text*. DNS names are refused even if they currently resolve
   locally, because DNS is exactly what an exfiltration attempt would
   control.
2. **No cloud agent providers.** The agentic-retrieval upgrades
   (`agent_provider = openai / anthropic`) send retrieved context to
   cloud APIs, so they're refused; agentic mode silently uses your
   local model instead.
3. **Belt and suspenders.** These checks live at the network
   chokepoints (`_ollama_chat`, the embedder constructor, the cloud
   completion functions), not in the UI — a bad config file, an
   applied preset, or a future feature can't route around them.

What the switch does *not* block: pulling models from ollama.com
(inbound download, none of your data attached) and the optional
compression feature's one-time tokenizer download (same — see
docs/COMPRESSION.md, which traces every connection it makes).

## The workspace lock (encryption at rest)

Your index — `.ezrag/meta.sqlite` — contains **every extracted chunk
of every document plus its embeddings**. Anyone who copies that file
has your corpus. The lock encrypts it:

- **Cipher:** AES-256-GCM (authenticated — tampering is detected, not
  silently decrypted).
- **Key:** derived from your passphrase with scrypt (n=2¹⁵, r=8, p=1),
  a memory-hard KDF that makes offline brute-force expensive.
- **Before encrypting:** the SQLite WAL is checkpointed and truncated
  so no plaintext survives in journal side-files; the in-memory
  embedding cache is dropped; the round-trip is verified against the
  original bytes *before* the plaintext is deleted.
- **While locked:** `ingest`, `ask`, and `chat` refuse with a clear
  message — and cannot accidentally create a fresh empty index over
  the encrypted one.

```
$ ez-rag lock
Passphrase (min 8 chars): ********
Locked. Index encrypted at meta.sqlite.enc. Unlock with: ez-rag unlock
```

## Context-aware redaction (remove data at ingest)

For data that shouldn't exist in the index at all — your name, an
email, an account number — list it in config and it is removed from
every chunk BEFORE tokenization and embedding:

```toml
redact_terms = ["Casey Stone", "casey@example.com"]
redact_replacement = "[REDACTED]"   # default
redact_smart = true                  # default
```

Because redaction happens before anything derived from the text
exists, the terms never reach the index text, the FTS search tokens,
or the embedding vectors — so they can't surface in answers and can't
ship in a distributed export.

**Context-aware means the ambiguity is handled.** A surname like
"Stone" is also an ordinary word; blind replacement would mangle
"the crew hauled crushed stone". The rules:

- Multi-word names are redacted case-insensitively everywhere, and
  their real-world variants match too: `Stone, Casey` · `C. Stone`
  · `Casey S.`
- Terms with `@` or digits (emails, IDs, phone numbers) are
  unambiguous → redacted in any case.
- A lone ambiguous token uses **smart casing**: `Rounds` (capitalized,
  how names appear) is redacted; `crushed stone` (lowercase common
  noun) is preserved. Set `redact_smart = false` to redact every
  occurrence regardless.

**Distribution protection.** `export_chatbot` verifies the index is
actually clean with the same matching rules (terms added after the
last ingest are caught with a "re-ingest first" error), and refuses
`include_sources` while redaction is configured — the original
documents contain the data by definition. Editing the term list
re-ingests affected files automatically on the next ingest.
Verify anytime with `ez-rag redact-check` (aggressive scan — flags
even common-noun occurrences for your review).

**Limits:** filenames containing a term are warned about at ingest
(paths are stored and shown in citations — rename the file); the
original documents in docs/ keep the term, by definition.

## Honest limits — read this part

Security features that overpromise are worse than none, so:

- **While UNLOCKED, the index is plaintext on disk.** SQLite and FTS5
  cannot search ciphertext; encrypting per-row would break retrieval.
  The lock protects the index *when you're not using it* — laptop in a
  bag, shared machine, backup drives. For protection while working,
  pair it with full-disk encryption (BitLocker on Windows, FileVault
  on macOS, LUKS on Linux) — which you should be running anyway.
- **The lock covers what ez-rag generates.** Your original documents
  in `docs/` were on your disk before ez-rag saw them; encrypt them
  with the same full-disk tools. (Sidecar `.ezrag-meta.toml` files
  next to your documents also stay plaintext — they contain titles and
  topics, not document text.)
- **No passphrase recovery.** It isn't stored anywhere, which is the
  point. If you lose it, delete `meta.sqlite.enc` and re-ingest from
  your source documents — the index is always rebuildable.
- **RAM is out of scope.** While chatting, chunks pass through process
  memory and the GPU like any local inference. That's inherent to
  using the data.

## Threat model in one table

| Threat | Covered? | By what |
|---|---|---|
| Misconfigured URL sends chunks to a cloud API | ✅ | local-only enforcement |
| Cloud agent provider leaks retrieved context | ✅ | provider refusal |
| Stolen laptop / copied drive (workspace locked) | ✅ | AES-256-GCM + scrypt |
| Stolen laptop while workspace unlocked | ⚠️ | use full-disk encryption |
| Malware on your machine while you work | ❌ | out of scope for any app |
| Ollama itself phoning home | n/a | Ollama is local; block egress at the firewall for certainty |

## Verification

35 automated tests (`tests/test_security.py`) cover the URL matrix,
enforcement semantics, encrypt/decrypt round-trips, wrong-passphrase
rejection, the full lock/unlock lifecycle against a real index
(including "ciphertext must not contain corpus text"), and the chat
path refusing a non-local URL with the mode on.

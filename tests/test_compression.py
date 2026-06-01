"""Context-compression integration tests (headroom-ai seam).

These test the INTEGRATION CONTRACT, not headroom itself:

  1. Off by default — `cfg.compress_context` is False; the seam is a
     pure no-op that returns the caller's exact message list.
  2. Fail-open — if compression raises, or headroom is missing, or it
     returns garbage, the ORIGINAL messages come back unchanged. A
     broken compressor must never break a chat.
  3. Happy path — a well-behaved compressor's output is passed through.
  4. Config round-trips the new fields.

No real headroom install is required: we monkeypatch the resolved
compress function so the tests run anywhere (CI, Windows, Linux).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag import compression


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


MESSAGES = [
    {"role": "system", "content": "You are helpful. Cite [n]."},
    {"role": "user", "content":
        "Question: test?\n\nContext:\n[1] " + ("alpha beta gamma " * 100)
        + "\n\nAnswer with citations."},
]


class _FakeResult:
    def __init__(self, messages):
        self.messages = messages
        self.tokens_before = 1000
        self.tokens_after = 600
        self.tokens_saved = 400
        self.compression_ratio = 0.4
        self.transforms_applied = ["router:mixed:0.40"]


def _reset_cache():
    compression._COMPRESS_FN = None


def test_off_by_default():
    print("\n[1] off by default -> pure no-op")
    cfg = Config()
    check("compress_context defaults False", cfg.compress_context is False)
    out = compression.maybe_compress_messages(cfg, MESSAGES)
    check("off path returns SAME object", out is MESSAGES)


def test_happy_path():
    print("\n[2] happy path -> compressed messages passed through")
    _reset_cache()
    compressed = [
        MESSAGES[0],
        {"role": "user", "content": "Question: test?\n\nContext:\n[1] alpha"},
    ]
    compression._COMPRESS_FN = lambda messages, **kw: _FakeResult(compressed)
    cfg = Config()
    cfg.compress_context = True
    out = compression.maybe_compress_messages(cfg, MESSAGES)
    check("returns the compressor output", out is compressed)
    check("output shorter than input",
          len(out[1]["content"]) < len(MESSAGES[1]["content"]))
    _reset_cache()


def test_fail_open_on_exception():
    print("\n[3] fail-open when compressor raises")
    _reset_cache()

    def _boom(messages, **kw):
        raise RuntimeError("compressor exploded")

    compression._COMPRESS_FN = _boom
    cfg = Config()
    cfg.compress_context = True
    out = compression.maybe_compress_messages(cfg, MESSAGES)
    check("exception -> original messages", out is MESSAGES)
    _reset_cache()


def test_fail_open_on_garbage():
    print("\n[4] fail-open when compressor returns garbage")
    _reset_cache()

    class _Bad:
        messages = []          # empty -> must be rejected

    compression._COMPRESS_FN = lambda messages, **kw: _Bad()
    cfg = Config()
    cfg.compress_context = True
    out = compression.maybe_compress_messages(cfg, MESSAGES)
    check("empty result -> original messages", out is MESSAGES)

    compression._COMPRESS_FN = lambda messages, **kw: None  # no .messages
    out2 = compression.maybe_compress_messages(cfg, MESSAGES)
    check("None result -> original messages", out2 is MESSAGES)
    _reset_cache()


def test_fail_open_when_missing():
    print("\n[5] fail-open when headroom isn't installed")
    _reset_cache()
    compression._COMPRESS_FN = False    # simulate "import failed"
    cfg = Config()
    cfg.compress_context = True
    out = compression.maybe_compress_messages(cfg, MESSAGES)
    check("missing headroom -> original messages", out is MESSAGES)
    check("compression_available() False when missing",
          compression.compression_available() is False)
    _reset_cache()


def test_stats_helper():
    print("\n[6] compression_stats reports metrics without breaking flow")
    _reset_cache()
    compression._COMPRESS_FN = lambda messages, **kw: _FakeResult(MESSAGES)
    cfg = Config()
    stats = compression.compression_stats(cfg, MESSAGES)
    check("stats has tokens_saved", stats["tokens_saved"] == 400)
    check("stats marks compressed", stats["compressed"] is True)
    check("stats ratio captured", stats["compression_ratio"] == 0.4)
    _reset_cache()

    compression._COMPRESS_FN = False
    stats2 = compression.compression_stats(cfg, MESSAGES)
    check("stats unavailable -> error noted",
          stats2["available"] is False and stats2["error"])
    _reset_cache()


def test_config_roundtrip(tmp_path=None):
    print("\n[7] config round-trips compression fields")
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="ezrag_compress_cfg_"))
    cfg = Config()
    cfg.compress_context = True
    cfg.compress_context_min_tokens = 333
    cfg.compress_context_target_ratio = 0.45
    p = d / "config.toml"
    cfg.save(p)
    loaded = Config.load(p)
    check("compress_context round-trips", loaded.compress_context is True)
    check("min_tokens round-trips", loaded.compress_context_min_tokens == 333)
    check("target_ratio round-trips",
          abs(loaded.compress_context_target_ratio - 0.45) < 1e-9)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def main():
    test_off_by_default()
    test_happy_path()
    test_fail_open_on_exception()
    test_fail_open_on_garbage()
    test_fail_open_when_missing()
    test_stats_helper()
    test_config_roundtrip()
    print(f"\n=== compression summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

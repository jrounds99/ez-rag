"""Tests for generate.inspect_text_quality — the opt-in LLM-assisted
garbled-text detector.

We stub _llm_complete so no Ollama is required and behavior is
deterministic across the various response shapes models actually emit.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag import generate as gen


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def with_stub_llm(response: str):
    """Helper: install a stub _llm_complete that returns `response`,
    return a context-manager-like (set, restore) tuple."""
    saved = gen._llm_complete
    gen._llm_complete = lambda c, p, max_tokens=16: response
    return saved


def restore(saved):
    gen._llm_complete = saved


def test_empty_input_returns_unknown():
    print("\n[1] empty / whitespace input -> unknown without LLM call")
    saved = gen._llm_complete
    called = {"n": 0}
    def stub(c, p, max_tokens=16):
        called["n"] += 1
        return "clean"
    gen._llm_complete = stub
    try:
        cfg = Config()
        for x in ("", "   ", "\n\n\t"):
            r = gen.inspect_text_quality(x, cfg)
            check(f"input {x!r} -> unknown",
                  r["state"] == "unknown", f"got {r}")
        check("LLM was not called for empty inputs",
              called["n"] == 0, f"got {called['n']} calls")
    finally:
        gen._llm_complete = saved


def test_no_backend_returns_unknown():
    print("\n[2] no LLM backend -> unknown (degrades gracefully)")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "none"
    try:
        r = gen.inspect_text_quality("Some real text", Config())
        check("no backend -> unknown", r["state"] == "unknown",
              f"got {r}")
    finally:
        gen.detect_backend = saved_detect


def test_clean_response_parsed():
    print("\n[3] LLM says 'clean' -> state='clean'")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = with_stub_llm("clean")
    try:
        r = gen.inspect_text_quality("Border collies are smart.", Config())
        check("clean reply -> clean", r["state"] == "clean", f"got {r}")
    finally:
        restore(saved_llm)
        gen.detect_backend = saved_detect


def test_garbled_response_parsed():
    print("\n[4] LLM says 'garbled' -> state='garbled'")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = with_stub_llm("garbled")
    try:
        r = gen.inspect_text_quality("hAppe�d to the \\pell\\", Config())
        check("garbled reply -> garbled", r["state"] == "garbled",
              f"got {r}")
    finally:
        restore(saved_llm)
        gen.detect_backend = saved_detect


def test_partial_response_parsed():
    print("\n[5] LLM says 'partial' -> state='partial'")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = with_stub_llm("partial")
    try:
        r = gen.inspect_text_quality("mostly clean but...", Config())
        check("partial reply -> partial", r["state"] == "partial",
              f"got {r}")
    finally:
        restore(saved_llm)
        gen.detect_backend = saved_detect


def test_strips_punctuation_around_verdict():
    print("\n[6] handles trailing punctuation / case / extra words")
    cases = [
        ("Clean.", "clean"),
        ("Garbled,", "garbled"),
        ("PARTIAL", "partial"),
        ("clean — this passage reads normally", "clean"),
        ("garbled - the text is gibberish", "garbled"),
    ]
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    for raw, expected in cases:
        saved_llm = with_stub_llm(raw)
        try:
            r = gen.inspect_text_quality("some text here", Config())
            check(f"raw={raw!r} -> {expected}",
                  r["state"] == expected, f"got {r}")
        finally:
            restore(saved_llm)
    gen.detect_backend = saved_detect


def test_unparseable_response_returns_unknown():
    print("\n[7] unparseable LLM reply -> unknown (don't penalize section)")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    for raw in ("yes the text seems fine", "I'm not sure about this", ""):
        saved_llm = with_stub_llm(raw)
        try:
            r = gen.inspect_text_quality("text", Config())
            check(f"raw={raw!r} -> unknown",
                  r["state"] == "unknown", f"got {r}")
        finally:
            restore(saved_llm)
    gen.detect_backend = saved_detect


def test_truncates_long_input():
    print("\n[8] long input is truncated before being sent to LLM")
    captured = {}
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = gen._llm_complete

    def stub(c, p, max_tokens=16):
        captured["prompt"] = p
        return "clean"
    gen._llm_complete = stub
    try:
        # 50 KB of text — should be truncated to ~1500 chars in the prompt
        big = "lorem ipsum dolor sit amet " * 2000
        gen.inspect_text_quality(big, Config())
        # Prompt has wrapping text + the sample; total should be well
        # under 2500 chars, not 50KB
        check("prompt was truncated",
              len(captured.get("prompt", "")) < 2500,
              f"got prompt of {len(captured.get('prompt', ''))} chars")
    finally:
        gen._llm_complete = saved_llm
        gen.detect_backend = saved_detect


def main():
    test_empty_input_returns_unknown()
    test_no_backend_returns_unknown()
    test_clean_response_parsed()
    test_garbled_response_parsed()
    test_partial_response_parsed()
    test_strips_punctuation_around_verdict()
    test_unparseable_response_returns_unknown()
    test_truncates_long_input()

    print(f"\n=== LLM-inspect summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

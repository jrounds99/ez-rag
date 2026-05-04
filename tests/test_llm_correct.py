"""Tests for generate.correct_garbled_text — the opt-in LLM-assisted
text-cleanup pass for OCR-recovered or partially-garbled sections.

We stub _llm_complete so no Ollama is required and behavior is
deterministic across the response shapes models actually emit.
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
    saved = gen._llm_complete
    gen._llm_complete = lambda c, p, max_tokens=16: response
    return saved


def restore(saved):
    gen._llm_complete = saved


def test_empty_input_returns_none():
    print("\n[1] empty / whitespace input -> None without LLM call")
    saved = gen._llm_complete
    called = {"n": 0}
    def stub(c, p, max_tokens=16):
        called["n"] += 1
        return "anything"
    gen._llm_complete = stub
    try:
        cfg = Config()
        for x in ("", "   ", "\n\n\t"):
            r = gen.correct_garbled_text(x, cfg)
            check(f"input {x!r} -> None", r is None, f"got {r!r}")
        check("LLM was not called for empty inputs",
              called["n"] == 0, f"got {called['n']} calls")
    finally:
        gen._llm_complete = saved


def test_no_backend_returns_none():
    print("\n[2] no LLM backend -> None (degrades gracefully)")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "none"
    try:
        r = gen.correct_garbled_text("noisy garbled text", Config())
        check("no backend -> None", r is None, f"got {r!r}")
    finally:
        gen.detect_backend = saved_detect


def test_clean_correction_returned():
    print("\n[3] LLM returns a clean correction -> string returned")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    cleaned = (
        "Border collies are remarkably intelligent dogs. They were "
        "bred for herding sheep in the Anglo-Scottish border region."
    )
    saved_llm = with_stub_llm(cleaned)
    try:
        r = gen.correct_garbled_text(
            "Border c0llies are remark@bly intelligent d0gs. They w3re "
            "bred for herding sheep in the Anglo-Scottish border region.",
            Config(),
        )
        check("returns the cleaned text", r == cleaned, f"got {r!r}")
    finally:
        restore(saved_llm)
        gen.detect_backend = saved_detect


def test_unrecoverable_response_returns_none():
    print("\n[4] LLM says UNRECOVERABLE -> None (don't keep poison)")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    for raw in ("UNRECOVERABLE",
                "UNRECOVERABLE\n",
                "Unrecoverable - too damaged to clean",
                "UNRECOVERABLE: nothing salvageable here"):
        saved_llm = with_stub_llm(raw)
        try:
            r = gen.correct_garbled_text("hopeless gibberish", Config())
            check(f"raw={raw[:30]!r} -> None", r is None, f"got {r!r}")
        finally:
            restore(saved_llm)
    gen.detect_backend = saved_detect


def test_empty_llm_response_returns_none():
    print("\n[5] empty LLM reply -> None")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    for raw in ("", "   ", "\n\n"):
        saved_llm = with_stub_llm(raw)
        try:
            r = gen.correct_garbled_text("text", Config())
            check(f"raw={raw!r} -> None", r is None, f"got {r!r}")
        finally:
            restore(saved_llm)
    gen.detect_backend = saved_detect


def test_too_short_correction_returns_none():
    print("\n[6] very short cleanup output -> None (likely refusal)")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = with_stub_llm("nope")
    try:
        r = gen.correct_garbled_text(
            "this is a longer noisy passage that should be cleaned up",
            Config(),
        )
        check("4-char output -> None", r is None, f"got {r!r}")
    finally:
        restore(saved_llm)
        gen.detect_backend = saved_detect


def test_descriptive_response_rejected():
    print("\n[7] LLM describes input instead of correcting -> None")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    descriptive = [
        "This passage appears to be heavily garbled and likely contains "
        "OCR errors that prevent meaningful reconstruction.",
        "The passage contains broken font encoding and cannot be reliably "
        "cleaned without more context from the source document.",
        "The text shows signs of OCR misreads throughout the entire body.",
        "Here is what I can make out from the noisy passage given.",
        "I cannot reliably clean this passage as too much is corrupted.",
    ]
    for raw in descriptive:
        saved_llm = with_stub_llm(raw)
        try:
            r = gen.correct_garbled_text("noisy text", Config())
            check(f"descriptive intro -> None ({raw[:30]!r})",
                  r is None, f"got {r!r}")
        finally:
            restore(saved_llm)
    gen.detect_backend = saved_detect


def test_strips_code_fences():
    print("\n[8] LLM wraps result in ``` fences -> fences stripped")
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    body = (
        "Border collies were bred for herding sheep in the "
        "Anglo-Scottish border region."
    )
    fenced_cases = [
        f"```\n{body}\n```",
        f"```text\n{body}\n```",
        f"```\n{body}",                     # opening fence only
    ]
    for raw in fenced_cases:
        saved_llm = with_stub_llm(raw)
        try:
            r = gen.correct_garbled_text("noisy", Config())
            check(f"fences stripped from {raw[:20]!r}",
                  r == body, f"got {r!r}")
        finally:
            restore(saved_llm)
    gen.detect_backend = saved_detect


def test_truncates_long_input():
    print("\n[9] long input is truncated before being sent to LLM")
    captured = {}
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = gen._llm_complete

    def stub(c, p, max_tokens=16):
        captured["prompt"] = p
        return "Border collies are clever working dogs bred for herding."
    gen._llm_complete = stub
    try:
        big = "lorem ipsum dolor sit amet " * 5000
        gen.correct_garbled_text(big, Config(), context="some context")
        # Sample is capped at 2500 chars; prompt has framing + ctx + sample
        # so should be well under 5KB even with overhead.
        check("prompt was truncated",
              len(captured.get("prompt", "")) < 5000,
              f"got prompt of {len(captured.get('prompt', ''))} chars")
    finally:
        gen._llm_complete = saved_llm
        gen.detect_backend = saved_detect


def test_context_included_when_provided():
    print("\n[10] surrounding context is included in the prompt")
    captured = {}
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = gen._llm_complete

    def stub(c, p, max_tokens=16):
        captured["prompt"] = p
        return "A clean reconstruction of the passage suitable for indexing."
    gen._llm_complete = stub
    try:
        gen.correct_garbled_text(
            "noisy passage",
            Config(),
            context="border collies herd sheep using their stare",
        )
        prompt = captured.get("prompt", "")
        check("context block appears in prompt",
              "border collies herd sheep" in prompt,
              "context not found in prompt")
        check("noisy passage appears in prompt",
              "noisy passage" in prompt, "passage not found")
    finally:
        gen._llm_complete = saved_llm
        gen.detect_backend = saved_detect


def test_no_context_omits_context_block():
    print("\n[11] empty context -> no surrounding-context block in prompt")
    captured = {}
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    saved_llm = gen._llm_complete

    def stub(c, p, max_tokens=16):
        captured["prompt"] = p
        return "Cleaned passage, plenty long enough to pass the length floor."
    gen._llm_complete = stub
    try:
        gen.correct_garbled_text("text", Config())  # context defaults to ""
        prompt = captured.get("prompt", "")
        check("no SURROUNDING CONTEXT marker present",
              "SURROUNDING CONTEXT" not in prompt,
              "context block leaked into prompt")
    finally:
        gen._llm_complete = saved_llm
        gen.detect_backend = saved_detect


def main():
    test_empty_input_returns_none()
    test_no_backend_returns_none()
    test_clean_correction_returned()
    test_unrecoverable_response_returns_none()
    test_empty_llm_response_returns_none()
    test_too_short_correction_returns_none()
    test_descriptive_response_rejected()
    test_strips_code_fences()
    test_truncates_long_input()
    test_context_included_when_provided()
    test_no_context_omits_context_block()

    print(f"\n=== LLM-correct summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

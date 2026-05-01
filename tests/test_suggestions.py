"""Tests for generate.generate_question_suggestions — the LLM-driven
chat-welcome suggestions.

We stub `_llm_complete` so no Ollama is required.
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


def test_returns_empty_for_empty_excerpts():
    print("\n[1] empty excerpts -> []")
    out = gen.generate_question_suggestions([], Config(), n=3)
    check("empty -> []", out == [], f"got {out}")


def test_passes_excerpts_to_llm():
    print("\n[2] passes excerpts into prompt")
    captured = {}
    saved = gen._llm_complete

    def stub(c, p, max_tokens=240):
        captured["prompt"] = p
        return ("What was the migration process?\n"
                "Which models tested best?\n"
                "Did the rerank help?")

    gen._llm_complete = stub
    try:
        out = gen.generate_question_suggestions(
            ["Border collies herd sheep using eye contact.",
             "The fastembed library supports several models."],
            Config(), n=3,
        )
        check("LLM prompt contains the excerpts",
              "Border collies" in captured.get("prompt", "")
              and "fastembed" in captured.get("prompt", ""),
              f"prompt head={captured.get('prompt', '')[:200]!r}")
        check("returned 3 suggestions", len(out) == 3, f"got {out}")
        check("first is the herd question",
              out[0].startswith("What was"), f"got {out[0]!r}")
    finally:
        gen._llm_complete = saved


def test_strips_numbering_and_quotes():
    print("\n[3] strips numbering / bullets / quotes from each line")
    saved = gen._llm_complete
    gen._llm_complete = lambda c, p, max_tokens=240: (
        '1. "Why did the rerank help on small corpora?"\n'
        '- Which retrieval mode is fastest in production?\n'
        '* Does HyDE actually help on technical docs?\n'
    )
    try:
        out = gen.generate_question_suggestions(["x"] * 3, Config(), n=3)
        check("numbering removed",
              not out[0].startswith("1."), f"got {out[0]!r}")
        check("quotes removed",
              not out[0].startswith('"'), f"got {out[0]!r}")
        check("bullets removed",
              not out[1].startswith("-"), f"got {out[1]!r}")
    finally:
        gen._llm_complete = saved


def test_filters_generic_templates():
    print("\n[4] filters generic templated suggestions")
    saved = gen._llm_complete
    gen._llm_complete = lambda c, p, max_tokens=240: (
        "Summarize the corpus.\n"
        "What topics are covered?\n"
        "List the documents and their main points.\n"
        "What is the load test methodology used in section 3?\n"
    )
    try:
        out = gen.generate_question_suggestions(["x"], Config(), n=3)
        check("generic 'summarize' rejected",
              all(not q.lower().startswith("summarize ") for q in out),
              f"got {out}")
        check("generic 'what topics' rejected",
              "what topics are covered?" not in [q.lower() for q in out],
              f"got {out}")
        check("kept the good one",
              any("load test methodology" in q for q in out),
              f"got {out}")
    finally:
        gen._llm_complete = saved


def test_returns_empty_on_llm_failure():
    print("\n[5] empty LLM response -> []")
    saved = gen._llm_complete
    gen._llm_complete = lambda c, p, max_tokens=240: ""
    try:
        out = gen.generate_question_suggestions(["x"], Config(), n=3)
        check("empty response -> []", out == [], f"got {out}")
    finally:
        gen._llm_complete = saved


def test_caps_excerpt_length():
    print("\n[6] caps excerpt budget so prompts don't blow up")
    captured = {}
    saved = gen._llm_complete

    def stub(c, p, max_tokens=240):
        captured["prompt"] = p
        return "A question?\nB question?\nC question?"

    gen._llm_complete = stub
    try:
        # 100 huge excerpts of ~5k chars each — total 500k. The function
        # should cap at ~3500 chars.
        excerpts = ["lorem ipsum " * 500] * 100
        gen.generate_question_suggestions(excerpts, Config(), n=3)
        prompt_len = len(captured.get("prompt", ""))
        check("prompt stays under ~6k chars",
              prompt_len < 6000, f"got {prompt_len}")
    finally:
        gen._llm_complete = saved


def main():
    test_returns_empty_for_empty_excerpts()
    test_passes_excerpts_to_llm()
    test_strips_numbering_and_quotes()
    test_filters_generic_templates()
    test_returns_empty_on_llm_failure()
    test_caps_excerpt_length()

    print(f"\n=== Suggestions summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

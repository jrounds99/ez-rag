"""Tests for the Ollama error-message translator."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.generate import _classify_ollama_error, _explain_ollama_error


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def expl(body: str, *, msgs=None) -> str:
    cfg = Config(llm_model="qwen3:35b", llm_url="http://127.0.0.1:11434")
    return _explain_ollama_error(
        Exception("upstream"), cfg=cfg,
        messages=msgs or [{"role": "user", "content": "hi"}],
        body=body,
    )


def main():
    print("\n[1] model not found / 404")
    out = expl('{"error":"model \\"qwen3:35b\\" not found"}')
    check("mentions ollama pull",
          "ollama pull qwen3:35b" in out, f"got {out!r}")

    print("\n[2] connection refused / server down")
    out = expl("All connection attempts failed")
    check("suggests starting Ollama",
          "Ollama" in out and "running" in out.lower(),
          f"got {out!r}")

    print("\n[3] OOM / VRAM exhausted")
    out = expl("CUDA out of memory")
    check("suggests smaller model or smaller prompt",
          "smaller model" in out.lower() or "smaller" in out.lower(),
          f"got {out!r}")
    check("includes prompt size estimate",
          "tokens" in out, f"got {out!r}")

    print("\n[4] context length exceeded")
    out = expl("input context length 8192 exceeds the model's context")
    check("recommends lowering top-k",
          "Top-K" in out, f"got {out!r}")
    check("recommends turning off chapter expansion",
          "Expand-to-chapter" in out, f"got {out!r}")
    check("recommends context window 0",
          "Context window" in out, f"got {out!r}")

    print("\n[5] generic 500")
    out = expl("Server error '500 Internal Server Error' for url '...'")
    check("mentions Ollama 500",
          "500" in out, f"got {out!r}")
    check("hints at context overflow as common cause",
          "context window" in out.lower(), f"got {out!r}")

    print("\n[6] prompt size estimate scales with messages")
    big_msgs = [{"role": "user", "content": "x" * 40_000}]
    out = expl("HTTP 500", msgs=big_msgs)
    check("size estimate present and large",
          "10,000 tokens" in out or "tokens" in out, f"got {out!r}")
    # ~40_000 chars / 4 = 10_000 tokens estimate
    check("estimate contains comma-formatted tokens",
          ",000 tokens" in out, f"got {out[-100:]!r}")

    print("\n[7] 'unable to load model' guidance")
    out = expl(
        '{"error":"unable to load model: '
        'C:\\\\Users\\\\j\\\\.ollama\\\\models\\\\blobs\\\\sha256-f5ee"}'
    )
    check("points at the Reload model button",
          "Reload model" in out, f"got {out!r}")
    check("points at the Free all VRAM button",
          "Free all VRAM" in out, f"got {out!r}")
    check("points at the Update Ollama button",
          "Update Ollama" in out, f"got {out!r}")
    check("clarifies it's unrelated to ingest",
          "ingest" in out.lower(), f"got {out!r}")

    print("\n[A] _classify_ollama_error labels each case")
    cases = {
        "load_failure":     'unable to load model: blob/sha-...',
        "oom":              'CUDA out of memory while allocating',
        "context_overflow": 'input exceeds context length 8192',
        "model_not_found":  '404: model "qwen3:35b" not found',
        "server_down":      'All connection attempts failed (connection refused)',
        "generic":          'who knows what went wrong',
    }
    for expected, body in cases.items():
        got = _classify_ollama_error(body)
        check(f"classify '{expected}' from {body[:30]!r}",
              got == expected, f"got {got!r}")

    print("\n[8] unknown error falls through with size hint")
    out = expl("weird error nobody has seen")
    check("preserves raw message",
          "weird error nobody has seen" in out, f"got {out!r}")
    check("appends size hint",
          "tokens" in out, f"got {out!r}")

    print(f"\n=== Ollama-error summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

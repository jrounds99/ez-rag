"""Ollama backend benchmark.

Runs N rounds against each of M models with a realistic RAG-style prompt
and reports load time, time-to-first-token, total wall time, output token
count, and tokens/sec. Reasoning models (`thinking` field separate from
`content`) are reported separately so totals don't conflate the two.

Run:
    python bench/bench_ollama.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

URL = "http://127.0.0.1:11434"
RUNS_PER_MODEL = 10

# Five distinct sizes spanning 3 orders of magnitude. Each is on disk and
# was confirmed via `ollama list`. Adjust as the user pulls more.
MODELS = [
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "qwen2.5:3b",
    "llama3.2-vision:latest",
    "deepseek-r1:32b",
]

# RAG-style prompt — a real question against synthetic context that's
# representative of an ez-rag retrieval (8 chunks, ~80 words each = ~1k
# tokens of context + a real question).
SYSTEM = (
    "You are a careful technical assistant. Cite passages [1], [2] when "
    "drawing facts from them. Be concise."
)
CONTEXT = """
[1] Border Collies were bred in the Anglo-Scottish border region for
herding sheep. Their intelligence is consistently rated as the highest
among dog breeds in standardized canine intelligence tests.

[2] The Maillard reaction begins around 140 °C and is responsible for the
brown crust on baked bread and seared meat. It's distinct from
caramelization, which is the thermal breakdown of sugars beginning around
160 °C.

[3] In retrieval-augmented generation, the dense embedder maps a query to
a vector in the same space as the indexed passages. Cosine similarity is
used to rank candidates, and a cross-encoder reranker is often applied to
the top-K to refine relevance.

[4] BM25 is a probabilistic ranking function that scores documents based
on term frequency and inverse document frequency, with length
normalization. Reciprocal Rank Fusion (RRF) is commonly used to combine
rankings from BM25 and dense retrieval.

[5] Quantization reduces model weight precision from 16 bits to 4 or 8
bits, dramatically cutting VRAM footprint. Q4_K_M is a popular GGML
mixed-precision format that retains most of the quality of FP16 at
roughly a quarter of the memory cost.

[6] The KV cache stores attention keys and values from prior tokens so
the model doesn't recompute them at each step. KV cache memory grows
linearly with context length and is often the dominant memory cost
during long generations.

[7] Newton's second law states force equals mass times acceleration,
F = ma. It's one of three laws describing the relationship between an
object's motion and the forces acting on it.

[8] The speed of light in vacuum is exactly 299,792,458 meters per
second. This constant, denoted c, is fundamental to Einstein's special
relativity, published in 1905.
""".strip()

QUESTION = (
    "Summarize the retrieval pipeline described in passages [3] and [4]. "
    "Two sentences."
)


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

@dataclass
class Run:
    model: str
    load_s: float = 0.0           # response.load_duration / 1e9 (Ollama field)
    eval_s: float = 0.0           # response.eval_duration / 1e9
    prompt_eval_s: float = 0.0    # response.prompt_eval_duration / 1e9
    ttft_s: float = 0.0           # wall-clock time to first non-empty content chunk
    total_wall_s: float = 0.0     # wall-clock total
    eval_count: int = 0           # output tokens per Ollama
    prompt_eval_count: int = 0    # input tokens
    content_chars: int = 0        # rendered output text length
    thinking_chars: int = 0       # reasoning text length (deepseek-r1 etc.)

    @property
    def tokens_per_sec(self) -> float:
        return self.eval_count / self.eval_s if self.eval_s > 0 else 0.0


def warmup(model: str) -> None:
    """First request to any model loads weights into VRAM (slow). We do a
    1-token generate to absorb that cost outside the measurements."""
    try:
        httpx.post(
            f"{URL}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=300.0,
        ).raise_for_status()
    except Exception as e:
        print(f"  ! warmup failed: {e}")


def stop(model: str) -> None:
    """Force the model out of VRAM between models so we don't double-load.
    keep_alive=0 makes Ollama unload immediately after the response."""
    try:
        httpx.post(
            f"{URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0,
                  "stream": False},
            timeout=10.0,
        )
    except Exception:
        pass


def run_once(model: str) -> Run | None:
    """Run a single chat completion against `model` and return measurements.

    Uses streaming so we can record the wall-clock time-to-first-token in
    addition to Ollama's internal counters.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Question: {QUESTION}\n\nContext:\n{CONTEXT}"},
    ]

    r = Run(model=model)
    t0 = time.perf_counter()
    first_seen = None
    final_obj = None

    try:
        with httpx.stream(
            "POST", f"{URL}/api/chat",
            json={"model": model, "messages": messages, "stream": True,
                  "options": {"temperature": 0.2, "num_predict": 512}},
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msg = obj.get("message", {}) or {}
                think = msg.get("thinking") or ""
                content = msg.get("content") or ""
                if think:
                    r.thinking_chars += len(think)
                if content:
                    if first_seen is None:
                        first_seen = time.perf_counter()
                    r.content_chars += len(content)
                if obj.get("done"):
                    final_obj = obj
                    break
    except Exception as e:
        print(f"  ! request failed: {e}")
        return None

    r.total_wall_s = time.perf_counter() - t0
    r.ttft_s = (first_seen - t0) if first_seen else r.total_wall_s
    if final_obj:
        r.load_s = (final_obj.get("load_duration") or 0) / 1e9
        r.prompt_eval_s = (final_obj.get("prompt_eval_duration") or 0) / 1e9
        r.eval_s = (final_obj.get("eval_duration") or 0) / 1e9
        r.eval_count = int(final_obj.get("eval_count") or 0)
        r.prompt_eval_count = int(final_obj.get("prompt_eval_count") or 0)
    return r


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def stdev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def fmt_row(label, vals_tps, vals_ttft, vals_eval, vals_load,
            prompt_tok, content_chars):
    return (
        f"  {label:<24}  "
        f"{median(vals_tps):>7.1f} ± {stdev(vals_tps):>5.1f} tok/s   "
        f"TTFT {median(vals_ttft):>5.2f}s   "
        f"eval {median(vals_eval):>5.2f}s   "
        f"load {median(vals_load):>5.2f}s   "
        f"prompt {prompt_tok:>5} tok  "
        f"out ~{content_chars:>4} chars"
    )


def main():
    print(f"Ollama benchmark — {RUNS_PER_MODEL} runs per model, "
          f"streaming /api/chat, prompt ~1.0k tokens\n")

    # quick connectivity probe
    try:
        httpx.get(f"{URL}/api/tags", timeout=3.0).raise_for_status()
    except Exception as e:
        print(f"  Ollama unreachable at {URL}: {e}")
        return 1

    all_results: dict[str, list[Run]] = {}
    for model in MODELS:
        print(f"--- {model} -----------------------------------------------")
        print("  warming up…", flush=True)
        warmup(model)
        runs: list[Run] = []
        for i in range(RUNS_PER_MODEL):
            r = run_once(model)
            if r is None:
                continue
            runs.append(r)
            print(f"  run {i+1:>2}/{RUNS_PER_MODEL}: "
                  f"{r.tokens_per_sec:>6.1f} tok/s  "
                  f"TTFT {r.ttft_s:>5.2f}s  total {r.total_wall_s:>5.2f}s  "
                  f"eval {r.eval_count:>3} tok"
                  + (f"  thinking {r.thinking_chars}c"
                     if r.thinking_chars else ""))
        all_results[model] = runs
        # Free this model from VRAM before loading the next one — otherwise
        # Ollama can double-occupy and slow / fail subsequent loads.
        stop(model)
        print()

    # ---- summary ---------------------------------------------------------
    print("\n========================================================")
    print(f"SUMMARY — {RUNS_PER_MODEL} runs per model, median ± stdev")
    print("========================================================\n")
    print(f"  {'model':<24}  {'tok/s':>15}      {'TTFT':>5}    "
          f"{'eval':>5}    {'load':>5}    {'prompt':>9}    output")
    print("  " + "-" * 110)
    for model, runs in all_results.items():
        if not runs:
            print(f"  {model:<24}  (no runs)")
            continue
        print(fmt_row(
            model,
            [r.tokens_per_sec for r in runs],
            [r.ttft_s for r in runs],
            [r.eval_s for r in runs],
            [r.load_s for r in runs],
            int(median([r.prompt_eval_count for r in runs])),
            int(median([r.content_chars for r in runs])),
        ))

    # JSON dump for later analysis
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps({
        m: [r.__dict__ for r in runs] for m, runs in all_results.items()
    }, indent=2), encoding="utf-8")
    print(f"\n  raw results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Performance-optimization benchmark for Ollama.

Tests methods we hadn't tried against the baseline to see if any are
meaningful on this hardware (RTX 5090 + 16-thread CPU). Each variant
restarts Ollama with different env vars / num_ctx settings and runs N
identical chat completions.

Run:
    python bench/bench_optimizations.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

URL = "http://127.0.0.1:11434"
MODEL = "deepseek-r1:32b"     # most memory-bound model the user has → most likely to show deltas
RUNS_PER_VARIANT = 5

SYSTEM = (
    "You are a careful technical assistant. Cite passages [1], [2] when "
    "drawing facts from them. Be concise."
)
# Reuse the same prompt as the baseline so deltas are directly comparable.
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
# Variants — each is (label, env vars, per-request options)
# ---------------------------------------------------------------------------
VARIANTS = [
    ("baseline (no flash, kv=f16, ctx=2048)",
     {}, {"num_ctx": 2048}),
    ("flash attention",
     {"OLLAMA_FLASH_ATTENTION": "1"}, {"num_ctx": 2048}),
    ("flash + KV cache q8_0",
     {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"},
     {"num_ctx": 2048}),
    ("flash + KV cache q4_0",
     {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q4_0"},
     {"num_ctx": 2048}),
    ("flash + KV q8_0 + num_ctx=8192",
     {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"},
     {"num_ctx": 8192}),
    ("flash + KV q8_0 + num_batch=1024",
     {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"},
     {"num_ctx": 2048, "num_batch": 1024}),
]


# ---------------------------------------------------------------------------
# Ollama lifecycle (restart with different env vars per variant)
# ---------------------------------------------------------------------------

def stop_ollama():
    """Kill any running ollama.exe processes so we can start with fresh env."""
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    time.sleep(2)


def start_ollama(env_overrides: dict[str, str]):
    """Start `ollama serve` with the given env vars set in its environment."""
    env = os.environ.copy()
    env.update(env_overrides)
    # Detached so this script doesn't hold the Ollama server's stdio.
    creation = 0
    if os.name == "nt":
        creation = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        creationflags=creation,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for /api/tags to come up
    for _ in range(30):
        time.sleep(1)
        try:
            r = httpx.get(f"{URL}/api/tags", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class Run:
    tok_per_sec: float = 0.0
    eval_count: int = 0
    eval_s: float = 0.0
    ttft_s: float = 0.0
    total_s: float = 0.0
    thinking_chars: int = 0


def warmup(model: str, options: dict):
    try:
        httpx.post(
            f"{URL}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False,
                  "options": {**options, "num_predict": 1}},
            timeout=300.0,
        ).raise_for_status()
    except Exception as e:
        print(f"  ! warmup failed: {e}")


def stop_model(model: str):
    try:
        httpx.post(
            f"{URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0,
                  "stream": False},
            timeout=10.0,
        )
    except Exception:
        pass


def run_once(model: str, options: dict) -> Run | None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Question: {QUESTION}\n\nContext:\n{CONTEXT}"},
    ]
    r = Run()
    t0 = time.perf_counter()
    first = None
    final = None
    try:
        with httpx.stream(
            "POST", f"{URL}/api/chat",
            json={"model": model, "messages": messages, "stream": True,
                  "options": {**options, "temperature": 0.2,
                              "num_predict": 512}},
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
                if msg.get("thinking"):
                    r.thinking_chars += len(msg["thinking"])
                if msg.get("content") and first is None:
                    first = time.perf_counter()
                if obj.get("done"):
                    final = obj
                    break
    except Exception as e:
        print(f"  ! request failed: {e}")
        return None

    r.total_s = time.perf_counter() - t0
    r.ttft_s = (first - t0) if first else r.total_s
    if final:
        r.eval_s = (final.get("eval_duration") or 0) / 1e9
        r.eval_count = int(final.get("eval_count") or 0)
        r.tok_per_sec = r.eval_count / r.eval_s if r.eval_s > 0 else 0.0
    return r


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def stdev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main():
    print(f"Optimization benchmark — model={MODEL}, {RUNS_PER_VARIANT} runs/variant")
    print()

    results: dict[str, list[Run]] = {}

    for label, env, options in VARIANTS:
        print(f"--- {label}")
        print(f"    env={env or '(none)'}  options={options}")
        stop_ollama()
        if not start_ollama(env):
            print("  ! Ollama failed to start under these env vars — skipping")
            continue
        warmup(MODEL, options)
        runs: list[Run] = []
        for i in range(RUNS_PER_VARIANT):
            r = run_once(MODEL, options)
            if r is not None:
                runs.append(r)
                print(f"    run {i+1}: {r.tok_per_sec:>5.1f} tok/s  "
                      f"TTFT {r.ttft_s:>5.2f}s  eval {r.eval_count} tok"
                      + (f"  thinking {r.thinking_chars}c"
                         if r.thinking_chars else ""))
        stop_model(MODEL)
        results[label] = runs
        print()

    # Restart Ollama in default state so the user's GUI keeps working.
    print("Restoring default Ollama (no env overrides) ...")
    stop_ollama()
    start_ollama({})

    print("\n========================================================")
    print(f"OPTIMIZATION SUMMARY — {MODEL}, {RUNS_PER_VARIANT} runs each")
    print("========================================================\n")
    base_tps = 0.0
    if results.get(VARIANTS[0][0]):
        base_tps = median([r.tok_per_sec for r in results[VARIANTS[0][0]]])
    print(f"  {'variant':<42}  {'tok/s':>15}  {'Δ vs base':>10}")
    print("  " + "-" * 80)
    for label, _, _ in VARIANTS:
        runs = results.get(label, [])
        if not runs:
            print(f"  {label:<42}  (no runs)")
            continue
        tps = median([r.tok_per_sec for r in runs])
        sd = stdev([r.tok_per_sec for r in runs])
        delta = ""
        if base_tps and label != VARIANTS[0][0]:
            pct = 100.0 * (tps - base_tps) / base_tps
            sign = "+" if pct >= 0 else ""
            delta = f"{sign}{pct:>5.1f}%"
        print(f"  {label:<42}  {tps:>7.1f} ± {sd:>4.1f}   {delta:>10}")

    out = Path(__file__).parent / "results_optimizations.json"
    out.write_text(json.dumps({
        label: [r.__dict__ for r in runs]
        for label, runs in results.items()
    }, indent=2), encoding="utf-8")
    print(f"\n  raw results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Context compression (optional)

ez-rag can compress the retrieved context before it reaches the LLM,
cutting prompt tokens — and therefore prompt-processing time, KV-cache
memory, and (if you point ez-rag at a hosted API) input-token cost —
while preserving the answer-relevant content. It's powered by
[headroom-ai](https://github.com/chopratejas/headroom) and is **off by
default, optional, and fail-open**.

## TL;DR

```bash
pip install "ez-rag[compress]"      # pulls headroom-ai (+ its ML model)
```

```toml
# <workspace>/.ezrag/config.toml
compress_context = true
```

That's it. Every `ask` / `chat` now compresses the retrieved context
first. If headroom isn't installed or hits any error, ez-rag silently
sends the original context — a broken compressor can never break a chat.

## What it does

On the Ohio geology benchmark (240 real ez-rag retrieval prompts: 3
embedders × 20 questions × 4 context depths), headroom compressed the
context by **~49% overall** (820k of 1.68M tokens) with **negligible
answer-quality change** under an LLM judge. Deeper retrieval (more
top-k chunks) compresses more, because there's more redundant material
to prune. See [the impact report](../headroom_bench/headroom_impact_report.pdf)
and [BENCH.md](BENCH.md#compression-bench) for the full numbers and method.

The mechanism: headroom's ContentRouter classifies the prompt and, for
RAG prose, runs a relevance scorer (ModernBERT) that keeps the
answer-bearing sentences and prunes redundancy. Originals are retained
(reversible compression), so nothing is lost — the model just reads the
same facts in fewer tokens.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `compress_context` | `false` | Master switch. |
| `compress_context_min_tokens` | `250` | Don't compress prompts smaller than this (tiny contexts compress poorly). |
| `compress_context_target_ratio` | `0.0` | `0.0` = let headroom decide. e.g. `0.5` = aim to keep ~50% of tokens. Higher pressure = more savings, more risk of dropping detail. |

## When to use it

- **Long contexts.** Savings grow with context size. If you run
  `top_k` ≥ 10 or have large chunks, compression pays off most.
- **Hosted backends.** If you point ez-rag at an OpenAI-compatible API,
  the token reduction is a direct bill reduction on input tokens.
- **VRAM-tight local runs.** A shorter prompt means a smaller KV cache
  and faster prompt eval (time-to-first-token).

## When to skip it

- **Tiny contexts.** Below ~250 tokens headroom declines anyway.
- **You can't spare the dependency.** headroom pulls in `torch` +
  `transformers` for its relevance model (~hundreds of MB). That's why
  it's a separate extra and not a core dependency — ez-rag's core stays
  lightweight.

## Privacy & network behavior

ez-rag is local-first and sends no telemetry. The optional compression
feature preserves that — **your prompts and documents never leave your
machine.** One transparency note, though:

- **One-time tokenizer download.** For token counting, headroom uses
  [tiktoken](https://github.com/openai/tiktoken), which downloads its
  BPE vocabulary file (`o200k_base`, ~2.5 MB) from
  `openaipublic.blob.core.windows.net` the first time it's used, then
  caches it permanently. This is a *download* — a plain GET for a
  vocabulary file. **No prompt text, context, or user data is uploaded.**
  We verified this by tracing every socket connection the `compress()`
  call makes (`headroom_bench/_net_probe.py`): the only non-loopback
  connection is that tiktoken vocab fetch.
- **Relevance model.** The compression model
  (`answerdotai/ModernBERT-base`) downloads once from Hugging Face and
  is cached locally; inference is fully offline thereafter.
- **No headroom cloud.** Headroom's optional cloud features (proxy,
  cross-agent memory, leaderboard) require `HEADROOM_API_KEY` and are
  **never** touched by ez-rag's integration — we call only the local
  `headroom.compress()` function.

### Running fully offline / air-gapped

After one online run the caches are warm and everything works offline.
To pre-seed for an air-gapped box:

```bash
# pre-cache the tiktoken vocab + the relevance model on a connected box
export TIKTOKEN_CACHE_DIR=~/.cache/ezrag-tiktoken
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
python -c "from headroom import compress; compress([{'role':'user','content':'x'*2000}], compress_user_messages=True)"
# then copy ~/.cache/ezrag-tiktoken and ~/.cache/huggingface to the target
```

## Safety / reversibility

- **Off by default.** Nothing changes until you set `compress_context`.
- **Optional dependency.** Without `ez-rag[compress]`, the seam no-ops.
- **Fail-open.** Any exception in compression returns the original
  messages. Verified by 16 integration tests
  (`tests/test_compression.py`) covering the off-path, happy-path, and
  every fail-open branch (missing package, raised exception, empty/None
  result).
- **Reversible content.** headroom keeps originals and can re-expand on
  demand.

## Implementation

The whole seam is one small module,
[`src/ez_rag/compression.py`](../src/ez_rag/compression.py), called
right after message assembly in `generate.answer()` and
`generate.chat_answer()`. It lazily imports `headroom.compress`, caches
the result, and wraps everything in a fail-open guard.

## Reproducing the benchmark

The evaluation harness lives in [`headroom_bench/`](../headroom_bench/):

```bash
# 1) build realistic ez-rag prompts from the Ohio corpus (Windows/native)
python headroom_bench/build_cases.py

# 2) compress them with real headroom (Linux/WSL — needs the ML extra)
python3 headroom_bench/compress_wsl.py --in cases.json --out compressed.json

# 3) quality check: orig vs compressed answers, judged (needs Ollama)
python headroom_bench/quality_check.py --sample 40

# 4) build the PDF
python headroom_bench/make_report.py
```

> Note: headroom currently ships prebuilt wheels for Linux + macOS
> (cp310–cp313). On Windows / Python 3.14 there's no wheel yet, so the
> compression step runs under WSL in the harness above. The ez-rag
> integration itself is pure-Python and platform-agnostic — it just
> no-ops if headroom can't load.

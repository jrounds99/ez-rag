# Retrieval

Everything that happens between "user types a question" and "LLM sees context."

## Defaults (smart, tuned for quality)

| Stage | On by default | Cost | Lift |
|---|:---:|---|---|
| Hybrid search (BM25 + dense, RRF) | ✅ | almost free | high |
| Cross-encoder rerank | ✅ | ~0.9 s / query | **very high** (only option that moved the needle in our matrix) |
| HyDE | ☐ | +1 LLM call (~3 s) | corpus-dependent |
| Multi-query | ☐ | +1 LLM call (~3 s) | corpus-dependent |
| Context window (±N neighbors) | ☐ | DB query, ~ms | qualitative — gives the LLM more context per hit |
| MMR diversity | ☐ | ~70 ms | useful only when retrieved chunks are redundant |
| Contextual Retrieval (ingest-time) | ☐ | slow ingest (one LLM call per chunk) | high on technical/structured docs |
| Larger reranker (`BAAI/bge-reranker-base`) | ☐ | swap; +280 MB download | accuracy similar on small corpora |

The pipeline runs in this order. Each enabled stage feeds the next.

### Empirical results

We tested every option on/off across 12 questions (mix of easy / paraphrased /
vague / multi-hop) on an 8-doc test corpus. Full table:
[`benchmark/reports/bench_configs.md`](../../benchmark/reports/bench_configs.md).

The headline: **only rerank moved the accuracy numbers** on a small corpus.
HyDE / Multi-query / MMR / Window are designed for harder conditions
(big corpora, paraphrase-heavy, redundant text, narrative docs) — they're
neutral or negative on a small clean corpus, and worth turning on when the
specific failure mode appears.

Run the matrix yourself any time:

```bash
python benchmark/bench_configs.py
```

## Hybrid search

Every query runs through both:

- **Dense** — cosine similarity between the query embedding and every chunk
  embedding. Strong on paraphrasing, weak on rare keywords.
- **Sparse (BM25)** — SQLite FTS5 keyword match. Strong on rare keywords
  and named entities, weak on paraphrasing.

Results are fused with **Reciprocal Rank Fusion** (RRF, `k=60`):
`score(d) = Σᵢ 1 / (k + rankᵢ(d))`. Documents ranked highly by either method
bubble up; documents ranked highly by both bubble up faster.

Turn off with **Hybrid: OFF** if you want pure-dense (rarely a good idea).

## Cross-encoder rerank *(the biggest single lift)*

Hybrid search is a *bi-encoder* — query and passages are embedded independently
and compared by cosine. A *cross-encoder* feeds `[query, passage]` into a
single transformer that outputs one relevance score per pair. Massively better
discrimination at the cost of running the model once per candidate (we only
rerank the top-30 candidates after RRF).

ez-rag ships with `Xenova/ms-marco-MiniLM-L-6-v2` (~23 MB ONNX) — fast,
English-focused. To swap, set `rerank_model` in `.ezrag/config.toml`. Good
alternatives:

- `BAAI/bge-reranker-base` (~280 MB) — multilingual, higher quality, slower
- `BAAI/bge-reranker-v2-m3` — current SOTA, larger

When ON, `top_k` results come from the reranker, NOT from RRF. So a chunk
that wasn't in the top-K of the dense search can still end up #1 in the
final list — that's the point.

## HyDE — Hypothetical Document Embeddings

When you embed a *question* (e.g. "What's the password reset policy?"), it
often doesn't match the *answer* phrasing in your docs ("Users may reset
their password by..."). HyDE fixes this:

1. Send the question to the LLM with a small prompt like *"Write a 2-sentence
   answer, even if you have to guess."*
2. Embed the answer.
3. Use that embedding for retrieval.

Cost: one extra LLM call (~0.5–5 s depending on model). Quality lift varies
by domain — biggest when corpus and questions use different vocabularies.

Enable in **Settings → Retrieval → HyDE**.

## Multi-query

Asks the LLM for 2 alternative phrasings of the question, retrieves for each
plus the original, and fuses with RRF. Helps when one concept can be asked
many ways ("CEO" vs "chief executive" vs "head of company").

Cost: same as HyDE (one extra LLM call), plus more embedding work (3×
retrievals).

Enable in **Settings → Retrieval → Multi-query**.

## Contextual Retrieval *(at ingest time)*

[Anthropic's technique](https://www.anthropic.com/news/contextual-retrieval).
Before embedding each chunk, prepend a 1-sentence summary placing it in
context: *"This chunk is from the Q3 financial report, section 'Revenue by
region', discussing growth in the EMEA segment."*

This makes embeddings of out-of-context chunks (e.g. a paragraph that just
says *"It grew 14% year-over-year."*) carry their parent-document semantics.

Reported lift: **~49% reduction in retrieval failures**, ~67% with
reranking added on top.

Cost: one LLM call per chunk during ingest. For a 100-chunk corpus on
deepseek-r1:32b that's ~10 minutes. On qwen2.5:3b about 30 seconds. Fine on
small/static corpora; expensive to recompute every time docs change.

Enable in **Settings → Ingest → Contextual Retrieval**, then **Re-ingest
(force)**.

## Top-K

How many passages are pasted into the LLM's context window. Defaults to 8.
Tradeoffs:

- Higher top-K = more chance of including the right passage, but more noise
  the LLM has to filter.
- Lower top-K = sharper context, but more chance of missing the answer.

Reranking changes the math: when reranking is on, top-3 with rerank often
beats top-10 without.

Reasonable values: 5–10 for casual chat, 15–30 for harder questions on big
corpora.

## How to tell what's working

Open the **Files** tab and look at the chunk count. Then ask a known
question and check:

- **Citations** — the chips below the answer show which passages were
  used. Click them to see the exact text.
- **`Use RAG` toggle** — turn it OFF, ask the same question. Compare. If
  the corpus is actually helping, the difference will be obvious.
- **Benchmark** — `python benchmark/rag_compare.py` runs a curated Q-set
  through both modes and prints accuracy + timing.

## Agentic mode

A single ON/OFF toggle that turns plain retrieval into an iterative
retrieve → reflect → re-search loop:

1. Run the normal retrieval pipeline (hybrid + rerank, plus whatever else
   you've enabled).
2. The LLM looks at the top hits and decides: *sufficient* or *not?*
3. If not sufficient, it generates 1–2 alternative search queries.
4. We retrieve for each alternative, fuse all results with RRF, and rerank
   once with the original question.

It's deliberately brute-force — no fancy framework, no fixed taxonomy of
question types. The LLM just gets to ask for more if the first try isn't
good enough.

**Cost.** Adds ≥ 1 short LLM call per cycle (default 2 cycles). On a 32B
local model that's ~10–20 s of overhead. On a hosted small model
(GPT-4o-mini, Claude Haiku) it's ~1–3 s.

**Where the agent's calls go.** Three providers:

- `same` *(default)* — uses the chat model. No setup. Quality is bounded by
  whatever's powering chat.
- `openai` — any OpenAI-compatible endpoint. Set `agent_api_key` and
  optionally `agent_base_url` to point at OpenAI / Groq / Together /
  Fireworks / vLLM / etc. `agent_model` defaults to `gpt-4o-mini`.
- `anthropic` — direct to api.anthropic.com. `agent_model` defaults to
  `claude-haiku-4-5-20251001`.

Useful when your local model is slow but you want a smarter agent step:
flip the chat model to a fast 3B for streaming answers, set the agent
provider to a hosted small model for the reflection.

**When to enable.** When plain retrieval seems to be giving you adjacent-
but-not-quite-right passages, especially on big or messy corpora. If
hybrid+rerank already nails the answer, agentic adds latency for nothing.

## Query modifiers

Persistent prefix / suffix / negative-traits text that wraps every question
when the chat-tab **Modifiers** checkbox is on. Edited in
**Settings → Query modifiers**, applied at composer time.

Format used:

```
{prefix}

{your question}

{suffix}

Avoid: {negatives}
```

Used for both retrieval (the embedder sees the augmented text) and the LLM
call (the augmented text becomes the user message). Toggle off the
checkbox in the chat composer to skip them per-query.

Examples:

| Use case | Prefix | Suffix | Negatives |
|---|---|---|---|
| D&D rules expert persona | "You are a D&D 5e rules expert." | "Cite the rulebook and page if possible." | "homebrew, opinions" |
| Concise output | (empty) | "Answer in 2 sentences max." | "lists, headings" |
| Legal disclaimer | (empty) | "End with: 'This is not legal advice.'" | (empty) |

## Choosing a recipe

| Use case | Recipe |
|---|---|
| Personal notes, casual queries | Defaults (hybrid + rerank). Don't touch anything. |
| Technical docs, code, legal | Defaults + **Contextual Retrieval** at ingest |
| Vague / short questions | Defaults + **HyDE** |
| Concept-heavy, multi-phrasing | Defaults + **Multi-query** |
| Massive corpus | Raise **Top-K** to 15–20, keep **Rerank** ON |
| Fast iteration / prototyping | Defaults are already plenty fast |

When in doubt: **defaults**.

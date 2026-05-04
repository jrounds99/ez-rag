# Plan: Cloud providers, per-process priority, custom agentic workflows

**Status:** PLANNING — not implemented.
**Scope:** Generalize ez-rag's LLM dispatch so ANY call (chat, retrieval
HyDE, ingest contextualization, agentic reflection, judge, etc.) can be
configured to use:

- A single local Ollama (today's behavior — preserved as the default)
- A cloud provider (Ollama Cloud, OpenAI-compatible, Anthropic)
- A **priority chain** — try provider A, fall back to B on failure
- **Custom agentic workflows** that mix local + cloud per step, with
  guaranteed local fallback if cloud fails

The free, local-only path stays the default. Cloud is opt-in, off by
default, and ez-rag never makes a paid call without an explicit
per-provider opt-in switch + an API key the user typed in.

---

## Why this matters

ez-rag already has provider plumbing for "agent" calls only
(`cfg.agent_provider`, `agent_complete()` → `same | openai |
anthropic`). It's narrow:

- Only the agentic-retrieval reflection loop uses it
- One global setting, can't differ per process
- No fallback — if the cloud call fails, the agent fails
- No Ollama Cloud (the new "Ollama Turbo" / cloud-hosted Ollama
  models) as a first-class option

Three concrete user pains this plan solves:

1. **"I want Ollama Cloud for chat but local for ingest contextualization."**
   Today: not possible — they share the same `cfg.llm_model`.
2. **"I want cloud-first, but if the API rate-limits or my key dies,
   keep working on local without me having to flip a switch."**
   Today: hard switch only.
3. **"For my agentic workflow, step 1 (broad reflection) should use a
   cheap fast cloud model, step 2 (final answer with retrieved corpus)
   should use my local model so the corpus content never leaves the
   machine."**
   Today: every step uses one provider.

---

## Part A — Provider abstraction

### A1. Provider model

Every LLM call goes through a `Provider` interface. The provider knows
how to talk to one backend.

```python
@dataclass
class Provider:
    id: str               # "ollama-local" | "ollama-cloud" | "openai" |
                          # "openai-compat" | "anthropic"
    name: str             # human-readable label
    base_url: str
    auth_kind: str        # "none" | "bearer" | "header:x-api-key"
    api_key: str          # blank for "none"
    chat_endpoint: str    # path appended to base_url
    streams: bool         # supports SSE streaming
    models_endpoint: str  # for the model picker
    privacy_label: str    # "local" | "cloud-hosted" | "third-party"
    notes: str

class Backend(Protocol):
    def chat(self, cfg, messages, options) -> str: ...
    def chat_stream(self, cfg, messages, options) -> Iterator[Tuple[str, str]]: ...
    def list_models(self) -> list[str]: ...
    def health_check(self) -> tuple[bool, str]: ...
```

Built-in provider definitions:

```python
BUILTIN_PROVIDERS = {
    "ollama-local": Provider(
        id="ollama-local",
        name="Ollama (local)",
        base_url="http://127.0.0.1:11434",
        auth_kind="none", api_key="",
        chat_endpoint="/api/chat",
        streams=True, privacy_label="local",
        notes="Runs entirely on your machine. Free.",
    ),
    "ollama-cloud": Provider(
        id="ollama-cloud",
        name="Ollama Cloud",
        base_url="https://ollama.com",
        auth_kind="bearer", api_key="",
        chat_endpoint="/api/chat",
        streams=True, privacy_label="cloud-hosted",
        notes="Cloud-hosted Ollama. Same /api/chat shape, "
              "Bearer-token authenticated. PAID — use your own key.",
    ),
    "openai": Provider(
        id="openai", name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth_kind="bearer", api_key="",
        chat_endpoint="/chat/completions",
        streams=True, privacy_label="third-party",
        notes="OpenAI-hosted. PAID. Sends prompt + corpus content "
              "to OpenAI servers.",
    ),
    "anthropic": Provider(
        id="anthropic", name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        auth_kind="header:x-api-key", api_key="",
        chat_endpoint="/messages",
        streams=True, privacy_label="third-party",
        notes="Claude API. PAID. Same privacy implications as OpenAI.",
    ),
    "openai-compat": Provider(
        id="openai-compat", name="OpenAI-compatible (custom)",
        base_url="", auth_kind="bearer", api_key="",
        chat_endpoint="/chat/completions",
        streams=True, privacy_label="third-party",
        notes="Generic OpenAI-compatible endpoint — Together, Groq, "
              "Fireworks, Mistral La Plateforme, vLLM, llama.cpp "
              "server, etc.",
    ),
}
```

User-defined providers extend this set (saved to
`<workspace>/.ezrag/providers.toml`).

### A2. Why Ollama Cloud is special

The /api/chat shape is identical to local Ollama. Adding it is a
two-line change in the existing `_ollama_chat` if we just parameterize
`base_url` and (optionally) an `Authorization: Bearer ...` header.
The current code hardcodes `cfg.llm_url` and no auth — generalizing
that is the single biggest unlock.

References: [Ollama Cloud docs](https://docs.ollama.com/cloud) confirm
the API key is set via `OLLAMA_API_KEY` env var or an
`Authorization: Bearer` header, and the endpoints (`/api/chat`,
`/api/generate`) are identical to local. ([Ollama blog — cloud models](https://ollama.com/blog/cloud-models))

### A3. Where the abstraction lives

New module: `src/ez_rag/providers.py`. Contains:
- `Provider` dataclass
- `BUILTIN_PROVIDERS` dict (above)
- `load_user_providers(workspace)` — read providers.toml
- `OllamaBackend`, `OpenAIBackend`, `AnthropicBackend` — three
  classes implementing the `Backend` protocol
- `make_backend(provider) -> Backend` — factory
- `chat(provider, model, messages, options)` — top-level dispatch
  function used everywhere

Existing `_ollama_chat`, `_ollama_chat_stream`, `_llm_complete` etc.
in `generate.py` get refactored to call `providers.chat(...)` instead
of inlining the httpx call. Behavior unchanged when provider is
`ollama-local` and api_key is blank — that's the migration safety net.

---

## Part B — Per-process priority chains

### B1. What's a "process"?

Anywhere ez-rag calls an LLM. Listed in current code:

| Process ID | Where | Today's setting |
|---|---|---|
| `chat` | `chat_answer()` user-facing | `cfg.llm_model` |
| `chat_no_rag` | RAG-off path | `cfg.llm_model` |
| `hyde` | HyDE query expansion | `cfg.llm_model` |
| `multi_query` | query paraphrasing | `cfg.llm_model` |
| `agent_reflect` | agentic retrieval reflection | `cfg.agent_*` |
| `crag_filter` | chunk relevance filter (opt-in) | `cfg.llm_model` |
| `ingest_inspect` | per-section quality check | `cfg.llm_model` |
| `ingest_correct` | OCR cleanup | `cfg.llm_model` |
| `ingest_contextual` | Anthropic-style chunk context | `cfg.llm_model` |
| `ingest_scan` | (planned in INGEST_INTELLIGENCE) | TBD |
| `judge` | LLM-as-judge benchmarking | `cfg.llm_model` |
| `embed` | text → vectors | `cfg.embedder_model` (separate) |
| `rerank` | cross-encoder rerank | local fastembed (separate) |

### B2. Priority chain spec

Each process gets a chain — a list of (provider_id, model_tag) pairs
tried in order until one succeeds:

```toml
# <workspace>/.ezrag/process_routing.toml

[chat]
chain = [
  { provider = "ollama-local", model = "qwen2.5:7b" },
]

[chat_alt_cloud_first]
chain = [
  { provider = "ollama-cloud", model = "qwen3-coder:480b-cloud" },
  { provider = "ollama-local", model = "qwen2.5:7b" },   # fallback
]

[hyde]
chain = [
  # HyDE is short + frequent → cheapest fast model first
  { provider = "ollama-local", model = "qwen2.5:1.5b" },
]

[ingest_inspect]
chain = [
  # Long batch job; cloud preferred for speed; local fallback if quota
  { provider = "ollama-cloud", model = "gpt-oss:20b-cloud" },
  { provider = "ollama-local", model = "qwen2.5:7b" },
]

[ingest_contextual]
chain = [
  # Lots of calls — local only, opt-in cloud requires explicit flag
  { provider = "ollama-local", model = "qwen2.5:1.5b" },
]
```

### B3. Three priority presets

For users who don't want to hand-edit the TOML:

| Preset | What it does |
|---|---|
| **Local only** (default) | Every chain has exactly one entry: `ollama-local` with `cfg.llm_model`. No cloud calls ever. |
| **Local first** | Each chain is `[local, cloud_default]`. Cloud only fires when local errors. |
| **Cloud first** | Each chain is `[cloud_default, local]`. Cloud is preferred; local takes over on quota/network failure. |
| **Custom** | The user has hand-edited `process_routing.toml`. ez-rag won't touch it. |

`cloud_default` defaults to the first cloud provider that has an API
key configured, with model = `cfg.cloud_default_model`.

### B4. Fallback semantics

```python
def call(process_id, messages, options=None) -> ProviderResult:
    chain = ROUTING.get_chain(process_id)
    last_err = None
    for step in chain:
        provider = PROVIDERS.get(step.provider)
        if provider is None:
            continue
        if not _is_available(provider):   # auth-key missing, daemon down
            continue
        try:
            return ProviderResult(
                text=providers.chat(provider, step.model, messages, options),
                provider=step.provider, model=step.model,
                fell_back=(step is not chain[0]),
            )
        except (ProviderRateLimit, ProviderUnavailable, ProviderTimeout) as e:
            last_err = e
            continue
    raise NoProviderAvailable(last_err, chain)
```

Failure classification:
- `ProviderUnavailable` — connection refused, DNS, daemon not running
- `ProviderAuth` — 401/403 — DON'T retry the same provider
- `ProviderRateLimit` — 429 → fall back to next
- `ProviderTimeout` — fall back
- `ProviderModelNotFound` — fall back
- `ProviderInvalidRequest` — DON'T fall back (deterministic bug, falling back hides it)
- `ProviderOOM` — local Ollama OOM → fall back to cloud if present

### B5. Privacy gate

When a chain step would send corpus content to a cloud provider, the
GUI shows a one-time per-workspace confirmation:

```
The current process (ingest_contextual) is configured to use
"OpenAI" as a fallback. Running this will send chunks of your
indexed documents to api.openai.com.

Allow cloud calls for this workspace?
  [No, local-only for this workspace]   [Yes, allow cloud]
```

Stored in `workspace.toml` as `cloud_calls_allowed = true`. False by
default. If false, every chain is automatically reduced to local-only
steps.

---

## Part C — Custom agentic workflows

### C1. What's a workflow?

A named, ordered sequence of LLM steps. Each step has:
- An `id` (logged for debugging)
- A `process` (one of the IDs in B1)
- A `prompt_template` (with `{vars}` from prior steps)
- An `output_var` name
- An optional `if` condition to skip the step

Workflows live in `<workspace>/.ezrag/workflows/<name>.toml`.

### C2. Example workflow — "draft + verify"

```toml
# .ezrag/workflows/draft-and-verify.toml
name = "draft-and-verify"
description = "Cloud draft, local final pass that grounds in retrieved corpus."

[[steps]]
id = "rough_draft"
process = "agent_reflect"          # uses chain B2 for agent_reflect
prompt = """
Question: {question}

Without seeing any source documents, write a 4-sentence
hypothetical answer that DESCRIBES what the answer should
contain. Don't make up specifics. Output only the description.
"""
output_var = "draft"

[[steps]]
id = "retrieve"
process = "_retrieve"              # built-in non-LLM step
inputs = ["draft", "question"]
output_var = "hits"

[[steps]]
id = "final"
process = "chat"                   # uses chain B2 for chat (local-first)
prompt = """
Use the following context excerpts to answer the user's
question. The earlier draft is for reference only — do not
copy claims from it that aren't supported by the excerpts.

DRAFT (for reference): {draft}

CONTEXT:
{hits_formatted}

QUESTION: {question}
"""
output_var = "answer"

[returns]
text = "{answer}"
citations = "{hits}"
```

Wired into the chat send flow: when a user asks a question and a
workflow is selected as their default, ez-rag executes the workflow
instead of the standard `smart_retrieve → chat_answer` path.

### C3. The local-fallback guarantee

Every step that uses a cloud-capable process has an implicit safety
clause: if every cloud step in its chain fails AND `cloud_required`
isn't set on the step, the call falls through to the local tail of
the chain. The workflow keeps moving instead of erroring out.

If a step **must** succeed on cloud (e.g. it depends on a model only
available there), set `cloud_required = true` and the workflow
errors with a clear message instead of silently using a weaker
local model.

```toml
[[steps]]
id = "use_huge_cloud_model"
process = "agent_reflect"
cloud_required = true              # error if no cloud available
prompt = "..."
```

### C4. Workflow runner

New module: `src/ez_rag/workflow.py`. Contains:

- `Workflow` dataclass (loaded from TOML)
- `Step` dataclass with `process`, `prompt`, `inputs`, `output_var`
- `WorkflowContext` — accumulator dict for inter-step variables
- `run(workflow, question, cfg, status_cb=None) -> Answer`
- `_render(template, ctx) -> str` — Jinja-like substitution
- Built-in non-LLM step types: `_retrieve`, `_rerank`, `_format_hits`

The runner streams progress events via `status_cb` so the GUI's chat
workflow chip strip can show step-by-step pulses (already wired to
display retrieval/rerank/etc.; just add step IDs).

### C5. Built-in workflows shipped with ez-rag

Out of the box, ship 4 reference workflows:

1. **default** — current behavior. `smart_retrieve → chat_answer`. Single
   process. Local only.
2. **draft-and-verify** — example C2 above. Cloud draft, local final.
3. **multi-step-reasoning** — for multi-hop questions. Step 1 generates
   sub-questions; step 2 retrieves for each; step 3 synthesizes.
4. **fact-check** — answer once, then run a second LLM call to
   verify each claim against the retrieved chunks; flag unsupported
   claims.

Users can copy and edit these as starting points.

---

## Part D — UI integration

### D1. Settings → Cloud providers card (new)

```
┌───────────────────────────────────────────────────────────┐
│ CLOUD PROVIDERS                                           │
├───────────────────────────────────────────────────────────┤
│  ✓ Ollama (local)            127.0.0.1:11434     [test] │
│      Free · always available                             │
│                                                          │
│  ○ Ollama Cloud              ollama.com                  │
│      API key  [................] [show]      [test]    │
│      $ paid · cloud-hosted                              │
│      Default model [qwen3-coder:480b-cloud         ▼]   │
│                                                          │
│  ○ OpenAI                    api.openai.com              │
│      API key  [................] [show]      [test]    │
│      $ paid · third-party                                │
│      Default model [gpt-4.1-mini                   ▼]   │
│                                                          │
│  ○ Anthropic                 api.anthropic.com           │
│      API key  [................] [show]      [test]    │
│      $ paid · third-party                                │
│      Default model [claude-haiku-4-5               ▼]   │
│                                                          │
│  + Add custom OpenAI-compatible endpoint                 │
│                                                          │
│  ☐ Allow cloud calls in this workspace                  │
│      Required before any process can route to cloud.    │
└───────────────────────────────────────────────────────────┘
```

API keys stored in `<workspace>/.ezrag/secrets.toml` (added to
`.gitignore` template; encrypted at rest is a stretch goal).

### D2. Settings → Process priorities (new card)

```
┌───────────────────────────────────────────────────────────┐
│ PROCESS PRIORITIES                                        │
│ Preset:  ( ) local only  (•) local first  ( ) cloud first│
│          ( ) custom (edit process_routing.toml)          │
├───────────────────────────────────────────────────────────┤
│ Process              | Provider chain                     │
│ ────────────────────────────────────────────────────────── │
│ chat                 | local: qwen2.5:7b  →  cloud: …    │
│ hyde                 | local: qwen2.5:1.5b               │
│ ingest_inspect       | local: qwen2.5:7b  →  cloud: …    │
│ ingest_correct       | local: qwen2.5:7b  →  cloud: …    │
│ ingest_contextual    | local: qwen2.5:1.5b   (no cloud)  │
│ agent_reflect        | cloud: …  →  local: qwen2.5:7b    │
│ judge                | local: qwen2.5:7b                  │
└───────────────────────────────────────────────────────────┘
```

Click a row → modal where the user re-orders / adds / removes
provider+model pairs in the chain. The "no cloud" badge appears
when the user has explicitly forbidden cloud for a process even
though it'd be allowed by their preset.

### D3. Workflow builder modal

A simple TOML editor with:
- Syntax highlighting for the workflow schema
- A "validate" button that parses + renders the DAG
- A "test run" button that runs the workflow against a sample
  question without committing to it
- A "save & set as default" toggle

Out-of-scope for v1: a fully visual node-based editor. TOML is
expressive enough and stays version-controllable.

---

## Part E — Cost & safety

### E1. Token / cost tracking

Every cloud call emits a `UsageRecord` to
`<workspace>/.ezrag/usage.jsonl`:

```json
{"ts":"2026-05-04T12:30:01Z","process":"agent_reflect","provider":"openai",
 "model":"gpt-4.1-mini","prompt_tokens":1832,"completion_tokens":412,
 "estimated_cost_usd":0.0023}
```

Costs are estimated from a per-provider pricing table baked into
ez-rag (updatable). The GUI footer shows a per-session running total
when any cloud call is enabled.

### E2. Pre-flight cost estimate

Before executing an expensive workflow (anything with an estimated
cost > $0.05 per run), prompt:

```
This workflow will make ~24 cloud calls. Estimated cost
on your selected providers: $0.18.

[Cancel]   [Run once]   [Always allow up to $1.00/run]
```

### E3. Hard budget cutoff

Optional `cloud_budget_usd_per_session` config. When set, any cloud
call that would push the session total over the budget falls through
to the local tail of its chain instead.

### E4. Rate-limit handling

429s are treated as a cleanly recoverable failure that falls back to
the next chain step. ez-rag also enforces a small per-provider RPM
ceiling to avoid hammering a key right after a billing failure.

### E5. Privacy disclosure

The HTML benchmark report (and any user-facing doc) shows the
`privacy_label` for each provider. Local stays "local". Anything
else is clearly tagged. The GUI header shows a "🔒 local only" or
"☁ mixed" indicator based on whether any process in the active
preset routes to cloud.

---

## Part F — Phasing & risks

### F1. Suggested phases

| Phase | Deliverable | Effort | Risk |
|---|---|---|---|
| 1 | `providers.py` + `OllamaBackend` (local + cloud) | small | low |
| 2 | Refactor `_ollama_chat` etc. to use `providers.chat()` | medium | medium (touches every call site) |
| 3 | `OpenAIBackend` + `AnthropicBackend` | small | low |
| 4 | `process_routing.toml` + chain resolver | medium | low |
| 5 | Settings → Cloud providers UI card | medium | low |
| 6 | Settings → Process priorities UI card | medium | medium |
| 7 | `workflow.py` runner + 4 built-in workflows | medium | medium |
| 8 | Workflow builder modal | medium | low |
| 9 | Usage tracking + cost meter + budget cutoff | small | low |
| 10 | Privacy gate + indicators throughout UI | small | low |

Phases 1–4 are the foundation — once they ship, every subsequent
phase is additive. Phase 7 (workflows) doesn't block anything; could
ship in any order after 4.

### F2. Open design questions

1. **Where do API keys live?**
   `<workspace>/.ezrag/secrets.toml` with explicit gitignore. Encrypted
   at rest via OS keychain (Windows Credential Manager / macOS
   Keychain / libsecret) is a v2 nice-to-have. **Tentative:** plaintext
   in v1, document the gitignore prominently, add OS-keychain support
   in v2.

2. **Should the embedder route through this provider system?**
   Embedders (qwen3-embedding, nomic-embed-text, fastembed) are a
   different shape — `/api/embeddings` not `/api/chat`. **Tentative:**
   defer for v1; embedding stays on Ollama only. v2 adds an
   `EmbedProvider` analogue.

3. **Per-call cost estimation accuracy.**
   Pricing tables drift. **Mitigation:** the table is a TOML file
   shipped with releases, plus a periodic update mechanism.

4. **What about reasoning models on cloud (e.g. o1, claude with
   extended thinking)?**
   They emit `reasoning` separately from `content`. The provider
   abstraction needs to surface that distinction. **Tentative:** the
   `Backend.chat_stream()` yields `(kind, text)` like the existing
   Ollama path, with new kinds `"reasoning"` for cloud models.

5. **Workflow templating language scope.**
   String substitution alone or full Jinja2? **Tentative:** start with
   simple `{var}` substitution + `{var|truncate(N)}` filter. Add
   real Jinja2 if real workflows demand it.

### F3. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| User accidentally sends sensitive corpus to cloud | high | Two-gate system: per-workspace flag + per-process explicit cloud step |
| Runaway cost from a misconfigured workflow | high | Mandatory cost preview + optional hard budget cap |
| Provider API drift breaks ez-rag | medium | Each backend isolates its API; pin SDK / API versions; integration test per provider |
| Local fallback behaves differently from cloud, confusing the user | medium | Result objects carry `provider` + `model` + `fell_back` flags; UI shows the actual provider used for each answer |
| API key leaks via logs | medium | Redact `Authorization` headers in all log output; lint test |
| Latency variance hurts UX | low | UI shows per-step provider + latency in the workflow chip strip |

### F4. Backwards compatibility

- Workspaces without `process_routing.toml` get `local-only` preset
  auto-applied. Behavior identical to today.
- The existing `cfg.agent_*` fields are read once at upgrade time
  and migrated into a `process_routing.toml` entry for `agent_reflect`,
  then ignored. No surprise behavior change.
- Existing slash-command shapes (`Use RAG` toggle, etc.) are
  preserved — they map to the `chat` process chain.

---

## Glossary

- **Provider** — the *service* that hosts an LLM (Ollama local,
  Ollama Cloud, OpenAI, Anthropic, etc.)
- **Backend** — the Python class that knows how to talk to one
  provider's API
- **Process** — a named LLM-using activity in ez-rag (e.g. `chat`,
  `hyde`, `ingest_inspect`)
- **Chain** — an ordered list of (provider, model) pairs tried in
  sequence with fallback
- **Preset** — a named bundle of chains (`local-only`, `local-first`,
  `cloud-first`)
- **Workflow** — a multi-step user-defined LLM pipeline; each step
  has its own process and prompt template
- **Privacy label** — `local` / `cloud-hosted` / `third-party`,
  surfaced in UI for every provider

---

## Sources

- [Ollama Cloud documentation](https://docs.ollama.com/cloud) —
  authentication, endpoints, model list
- [Ollama blog — cloud models announcement](https://ollama.com/blog/cloud-models)
  — confirms /api/chat shape parity
- [Ollama API reference (DeepWiki)](https://deepwiki.com/ollama/ollama/3-api-reference)
  — endpoints + parameter shapes
- [Ollama main API docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- ez-rag's existing `agent_complete()` in `src/ez_rag/generate.py`
  — the existing OpenAI/Anthropic plumbing being generalized
- ez-rag's existing `cfg.agent_*` config fields — being migrated to
  per-process routing

---

*Last updated: 2026-05-03. Update whenever a new provider lands or
a workflow ships.*

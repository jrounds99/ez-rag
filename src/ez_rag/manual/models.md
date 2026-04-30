# Models

ez-rag picks an inference backend at runtime in this order:

1. **Ollama** — if `ollama serve` is reachable at `llm_url` (default `http://127.0.0.1:11434`).
2. **llama-cpp-python** — if installed (`pip install llama-cpp-python`) and `llm_provider="llama-cpp"` with `llm_model` pointing to a `.gguf` file.
3. **none** — retrieval-only mode; passages are returned but no LLM-written answer.

Same picker for embeddings: Ollama → fastembed → fail.

## Quick recommendations by VRAM

| GPU | LLM (Q4_K_M) | Embedder |
|---|---|---|
| 24 GB+ | `qwen2.5:14b-instruct` or `gemma3:27b` | `bge-m3` |
| 16 GB | `qwen2.5:7b-instruct` or `qwen3:8b` | `nomic-embed-text` |
| 8 GB | `qwen2.5:3b` or `phi4-mini` | `nomic-embed-text` |
| no GPU, 16 GB RAM | `phi4-mini` or `llama3.2:3b` | `nomic-embed-text` |
| no GPU, 8 GB RAM | retrieval-only mode | `BAAI/bge-small-en-v1.5` (fastembed) |

```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

## Browse the entire Ollama library

In the GUI, **Settings → Browse Ollama library** opens a searchable list of every public model on `ollama.com/library` (≈230 models, fetched and cached for 6 hours).

Each model shows:
- Capability badges: `tools` `vision` `embedding` `thinking` `audio` `cloud`
- A short description and total pull count
- Clickable size chips (`8b`, `70b`, `405b`, …) — clicking one fills the Tag field with `<name>:<size>`
- A search bar (matches name or description)
- Capability filter chips (All · LLMs · Vision · Reasoning · Embedding)

Click **Pull** to download. A progress bar streams `status   X.XX / Y.YY GB` while the layers come down. When it finishes, the new model is auto-selected in the LLM/embedder dropdown — hit **Save settings** to persist.

### VRAM estimates and fit color-coding

Every size chip is annotated with an estimated VRAM number and color-coded against your GPU:

- 🟢 **green** — fits comfortably (≤ 85% of total VRAM)
- 🟠 **amber** — tight (85% – 105%)
- 🔴 **red** — won't fit
- gray — no NVIDIA GPU detected (estimate shown anyway, no fit decision made)

Hover the chip for a tooltip with the exact numbers and the GPU you have.

#### How the estimate is computed

```
VRAM ≈  params × bits_per_param / 8           # weights
      + params × 0.05 × (context / 4K)        # KV cache (rough)
      + 0.5 GB                                # framework/CUDA overhead
```

`bits_per_param` per quant (Ollama defaults to **Q4_K_M ≈ 4.5 bits/param**):

| Quant | bits | Quant | bits |
|---|---:|---|---:|
| Q2_K  | 3.0 | Q5_K_M | 5.5 |
| Q3_K_M | 3.8 | Q6_K   | 6.6 |
| Q4_K_S | 4.3 | Q8_0   | 8.5 |
| Q4_K_M | 4.5 | F16    | 16.0 |
| Q5_K_S | 5.4 | F32    | 32.0 |

So, at Q4_K_M with a 4K context:

| Params | est. VRAM |
|---|---:|
| 270 M | ~0.7 GB |
| 1 B   | ~1.1 GB |
| 3 B   | ~2.3 GB |
| 7 B   | ~4.8 GB |
| 8 B   | ~5.4 GB |
| 14 B  | ~9.1 GB |
| 32 B  | ~20 GB  |
| 70 B  | ~43 GB  |
| 405 B | ~249 GB |

#### Caveats

These are **estimates**, not exact figures. The real number depends on:

- **Quantization** — most Ollama tags are Q4_K_M, but some are Q4_0 / Q3 / Q5. Click a chip and the tooltip assumes Q4_K_M; once pulled, the dropdown shows the exact on-disk size.
- **Context window** — 4K assumed. A 32K context adds ~1–2 GB on a 7B, much more on bigger models. Increase via `max_tokens` in Settings.
- **MoE models** — Qwen3-MoE etc. only load active params into VRAM, so an MoE "7b" is lighter than a dense 7b.
- **Vision/multimodal** — vision adapters add a few hundred MB on top.
- **Other GPU users** — VRAM detection uses `nvidia-smi --query-gpu=memory.total`. If something else is already using the GPU, your *available* VRAM is lower.
- **Apple Silicon / AMD** — `nvidia-smi` isn't present, so estimates display without a fit color.

For an accurate post-pull number, run `ollama ps` to see what Ollama actually has resident.

## Use a local GGUF instead of Ollama

If you'd rather run a `.gguf` directly (e.g. a fine-tune you have on disk) without going through Ollama:

1. **Settings → Use local GGUF…** opens a file picker.
2. Pick a `.gguf` file. The LLM dropdown switches to that path; `llm_provider` is auto-set to `llama-cpp`.
3. Install the runtime once: `pip install llama-cpp-python` (or `pipx inject ez-rag llama-cpp-python` if you used pipx).
4. Hit **Save settings**.

`llama-cpp-python` ships prebuilt wheels for CPU and CUDA on Windows / macOS / Linux.

## Use the model dropdowns directly

Both LLM and Ollama-embed-model fields are dropdowns of currently-pulled tags:

- Each option shows `<name>    <on-disk size>`, e.g. `qwen2.5:3b   1.8 GB`.
- The refresh ⟳ icon next to the LLM dropdown re-queries Ollama (useful after pulling from a terminal).
- Names ending in `:latest` are normalized away on display (`nomic-embed-text:latest` → `nomic-embed-text`).

## fastembed alternative

If Ollama isn't running, ez-rag falls back to **fastembed** (ONNX-based, no torch). The GUI's `fastembed model` dropdown lists supported models:

- `BAAI/bge-small-en-v1.5` (default, ~130 MB, fast on CPU)
- `BAAI/bge-base-en-v1.5` (440 MB)
- `BAAI/bge-large-en-v1.5` (1.34 GB, best quality)
- `intfloat/e5-small-v2` / `e5-base-v2`
- `intfloat/multilingual-e5-base` / `multilingual-e5-large`
- `jinaai/jina-embeddings-v2-base-en`
- `nomic-ai/nomic-embed-text-v1.5`
- `mixedbread-ai/mxbai-embed-large-v1`
- `snowflake/snowflake-arctic-embed-s` / `-m`
- `sentence-transformers/all-MiniLM-L6-v2` (90 MB, tiny)

First use of a fastembed model downloads it from HuggingFace into `~/.cache/fastembed`.

## CLI alternatives

The GUI is a frontend over `ez-rag` — everything is also reachable from the shell:

```bash
ollama pull qwen2.5:7b-instruct       # pull a model
ollama list                           # what's installed
ez-rag models                         # what ez-rag is configured to use
ez-rag doctor                         # what backends/deps are available
```

Edit `.ezrag/config.toml` directly if you'd rather not click:

```toml
llm_provider = "auto"            # auto | ollama | llama-cpp | none
llm_model    = "qwen2.5:7b-instruct"
llm_url      = "http://127.0.0.1:11434"

embedder_provider   = "auto"      # auto | ollama | fastembed
embedder_model      = "BAAI/bge-small-en-v1.5"   # used when fastembed
ollama_embed_model  = "nomic-embed-text"          # used when ollama
```

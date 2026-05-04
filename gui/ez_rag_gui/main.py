"""ez-rag GUI — Flet desktop app.

Runs the same library code as the CLI; nothing here re-implements business logic.
Primary view is Chat. Files / Settings / Doctor are the supporting tabs in a
NavigationRail on the left.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import flet as ft

from ez_rag.config import Config
from ez_rag.embed import make_embedder, clear_embedder_cache
from ez_rag.generate import (
    answer as gen_answer, chat_answer, detect_backend,
)
from ez_rag.index import Index, read_stats
from ez_rag.ingest import ingest
from ez_rag.models import (
    FASTEMBED_MODELS, RECOMMENDED_EMBED, RECOMMENDED_LLM,
    LibraryModel, OllamaModel, delete_ollama_model,
    detect_total_vram_gb, estimate_vram_gb, fetch_ollama_library,
    fmt_size, fmt_vram_gb, is_embed_capable, list_ollama_models,
    list_running_models, pull_ollama_model, search_library,
    unload_ollama_model, unload_running_models, vram_fit,
)
from ez_rag.generate import apply_query_modifiers
from ez_rag.parsers import supported_extensions
from ez_rag.retrieve import agentic_retrieve, hybrid_search, smart_retrieve
from ez_rag.multi_gpu import (
    GPU_INDEX_AUTO, GpuDaemon, ModelAssignment, RoutingTable,
    derive_default_table, load_routing_table, save_routing_table,
    set_active_table,
)
from ez_rag.daemon_supervisor import (
    DaemonSupervisor, HealthEvent, LoadedModel, SpawnError,
    detect_external, health_check_once, query_loaded_models,
)
from ez_rag.gpu_detect import detect_gpus, DetectedGpu
from ez_rag.workspace import (
    Workspace, find_workspace,
    get_default_rags_dir, set_default_rags_dir, list_managed_rags,
    get_theme_name, set_theme_name,
)


# ============================================================================
# Theme — palettes loaded from themes.toml at startup. The chosen palette
# name is persisted via workspace.get_theme_name() / set_theme_name(). Module-
# level color constants below are *populated* by `_load_theme()` in app() so
# every widget sees the active palette at construction time. To switch
# themes the app saves config and restarts (live theme switching would
# require re-walking the entire control tree).
# ============================================================================

import sys as _sys
if _sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:  # pragma: no cover
    import tomli as _tomllib  # type: ignore

THEMES_FILE = Path(__file__).parent / "themes.toml"

# Default palette baked into the source so a missing/corrupt themes.toml
# never breaks the app. Mirror of the [dark] entry.
_FALLBACK_PALETTE = {
    "accent":         "#7C7BFF",
    "accent_soft":    "#5856D6",
    "bg":             "#0F1115",
    "surface":        "#171922",
    "surface_hi":     "#1F2230",
    "on_surface":     "#E6E7EB",
    "on_surface_dim": "#9097A6",
    "success":        "#3DDC84",
    "warning":        "#F6B042",
    "danger":         "#F75A68",
    "user_bubble":    "#272A39",
    "assist_bubble":  "#1A1D29",
    "chip_bg":        "#23263A",
}


def load_themes() -> dict[str, dict[str, str]]:
    """Read themes.toml. Returns {palette_name: {key: hex, ...}}.
    Always includes a 'dark' palette (the in-source fallback) even if the
    file is missing or unreadable."""
    out: dict[str, dict[str, str]] = {"dark": dict(_FALLBACK_PALETTE)}
    try:
        with THEMES_FILE.open("rb") as f:
            data = _tomllib.load(f)
        for name, palette in data.items():
            if isinstance(palette, dict):
                # Merge with fallback so partial palettes still work.
                merged = dict(_FALLBACK_PALETTE)
                merged.update({k: str(v) for k, v in palette.items()
                               if isinstance(v, str)})
                out[name] = merged
    except Exception:
        pass
    return out


def _apply_palette(palette: dict[str, str]) -> None:
    """Mutate the module-level color globals from the chosen palette."""
    g = globals()
    g["ACCENT"] = palette["accent"]
    g["ACCENT_SOFT"] = palette["accent_soft"]
    g["BG_DARK"] = palette["bg"]
    g["SURFACE_DARK"] = palette["surface"]
    g["SURFACE_DARK_HI"] = palette["surface_hi"]
    g["ON_SURFACE"] = palette["on_surface"]
    g["ON_SURFACE_DIM"] = palette["on_surface_dim"]
    g["SUCCESS"] = palette["success"]
    g["WARNING"] = palette["warning"]
    g["DANGER"] = palette["danger"]
    g["USER_BUBBLE"] = palette["user_bubble"]
    g["ASSIST_BUBBLE"] = palette["assist_bubble"]
    g["CHIP_BG"] = palette["chip_bg"]


# Initial bind to the fallback so module imports work before app() runs.
_apply_palette(_FALLBACK_PALETTE)

RECENTS_FILE = Path.home() / ".ezrag_recents.txt"
MAX_RECENTS = 8


# ============================================================================
# Tooltip text
# ============================================================================
# One source of truth so the manual pages and the GUI stay in sync.

TIP = {
    # Header
    "open_workspace": "Open or create a different workspace folder. Each "
                      "workspace has its own docs/, index, and config.",
    "new_rag":        "Create a new named RAG: pick a name, a parent folder, "
                      "and one or more source folders to import documents "
                      "from. Each RAG is a self-contained workspace with its "
                      "own index — switch between them via the dropdown.",
    "rag_dropdown":   "Switch between recently-opened RAGs. Each is a "
                      "self-contained workspace. Use '+ New RAG…' to create "
                      "another one.",
    "files_pill":     "Files indexed in this workspace and total number of "
                      "embedded text chunks.",
    "backend_pill":   "Active LLM backend and model. Click Settings to change.",
    "help_btn":       "Help, keyboard shortcuts, and topic-by-topic guides.",
    "about_btn":      "About ez-rag, credits, and the people who got blamed.",

    # Chat tab
    "chat_input":     "Type your question. Enter sends. Shift+Enter inserts a "
                      "newline.",
    "send_btn":       "Send the message (Enter).",
    "stop_btn":       "Stop generation. The partial answer is kept.",
    "clear_btn":      "Start a fresh conversation. The model loses prior turns.",
    "free_vram_btn":   "Evict every model currently resident in Ollama's "
                      "VRAM (`ollama stop` for each loaded tag). Use when "
                      "you suspect a model is stuck, before switching "
                      "models, or when you need GPU memory back for "
                      "something else. The next chat will transparently "
                      "reload on demand. If a soft unload doesn't help, "
                      "kill ollama.exe from Task Manager.",
    "test_model_btn":  "Quick health check — sends a 1-token request to "
                      "your configured LLM and prints the verbatim result "
                      "(or the exact Ollama error) into the chat. Useful "
                      "when a specific tag like qwen3.6:35b refuses to "
                      "work and you want to see *why* without sending a "
                      "real question.",
    "reload_model_btn": "Force-redownload the configured LLM. Unloads "
                      "everything from VRAM, deletes the on-disk blob, then "
                      "re-pulls it from ollama.com. Use this if the model "
                      "is failing to load (the 'unable to load model' "
                      "error) — it's the equivalent of running  ollama rm "
                      "<tag>  then  ollama pull <tag>  in a terminal.",
    "rag_toggle":     "USE RAG — when ON, ez-rag retrieves from your corpus "
                      "before answering and cites passages [1], [2], …\n"
                      "When OFF, the question goes straight to the LLM with no "
                      "retrieval. Useful for A/B testing how much the corpus "
                      "actually helps.",
    "modifier_toggle": "Apply your configured query prefix / suffix / negatives "
                      "to this question. Configure them in Settings → Query "
                      "modifiers.",
    "agentic":        "Agentic retrieval — the LLM looks at the initial hits "
                      "and (if needed) generates 1–2 follow-up search queries "
                      "to fill in gaps. Slower but often catches things plain "
                      "retrieval misses. Uses your local model by default; "
                      "configure an API key/provider below to use a different "
                      "model just for the agent steps.",
    "agent_provider": "Where the agent calls go: 'same' uses the chat model. "
                      "'openai' hits any OpenAI-compatible endpoint (OpenAI, "
                      "Groq, Together, Fireworks, vLLM, …). 'anthropic' hits "
                      "api.anthropic.com.",
    "agent_model":    "Model name for agent calls. Blank = use the chat "
                      "model. For OpenAI: e.g. gpt-4o-mini. For Anthropic: "
                      "e.g. claude-haiku-4-5-20251001.",
    "agent_api_key":  "API key for the agent provider. Stored in plaintext "
                      "in this workspace's config.toml — keep the workspace "
                      "private.",
    "agent_base_url": "OpenAI-compatible base URL. Default is OpenAI's. "
                      "Examples: https://api.groq.com/openai/v1, "
                      "https://api.together.xyz/v1.",
    "query_prefix":   "Text added BEFORE every question (when the chat-tab "
                      "toggle is on). Useful for persona / role priming.",
    "query_suffix":   "Text added AFTER every question. Useful for output "
                      "formatting instructions.",
    "query_negatives":"Things the model should avoid. Appended as 'Avoid: …' "
                      "to every question.",
    "unload_llm":     "Before ingest starts, ask Ollama to evict the chat "
                      "LLM from VRAM. The embedder is small and doesn't need "
                      "the whole GPU. Auto-skipped if Contextual Retrieval "
                      "is ON (we need the LLM loaded for per-chunk summaries).",
    "embed_batch":    "How many chunks to embed per call. Bigger = faster "
                      "throughput on a strong GPU, more memory in flight. "
                      "16 is a balanced default; on a 5090 you can comfortably "
                      "go to 64.",
    "num_batch":      "Ollama prompt-eval batch size. Bigger = faster TTFT "
                      "(prompt processing) at the cost of more peak memory. "
                      "Empirically 1024 measured -23% TTFT on a 32B model "
                      "vs the default 512 in our benchmark.",
    "num_ctx":        "Tokens of context Ollama allocates for the model. "
                      "0 = let Ollama pick the model's default. Larger = "
                      "more VRAM upfront but no per-token speed cost when "
                      "actual sequences are short.",
    "parallel_workers": "Reserved for future parallel parsing. Currently "
                      "no effect; leave at 1.",
    "default_rags_dir": "Where 'New RAG' creates workspaces by default. "
                      "Each RAG you create here also shows up in the RAG "
                      "dropdown automatically.",
    "manage_rags":    "Browse, open, export, or delete every RAG stored in "
                      "the default folder.",
    "export_chatbot": "Bundle this RAG and a runnable chatbot into a single "
                      "zip. Cross-platform (Windows / macOS / Linux). The "
                      "recipient needs Python and Ollama with the configured "
                      "model pulled. Settings, theme, and index are baked in.",
    "include_sources": "Bundle the original ingested files (PDFs, HTML, "
                      "screenshots) under data/sources/ so the chatbot can "
                      "render PDF page previews and show original images "
                      "when a citation chip is clicked. Off by default "
                      "because PDF-heavy workspaces can be multi-GB.",
    "theme_palette":  "GUI color palette. Light, dark, midnight, forest, "
                      "solarized, nord, dracula, rosé pine, rainbow. Edit "
                      "themes.toml to add your own.",
    "theme_apply":    "Save the chosen palette to ~/.ezrag/global.toml. "
                      "Takes effect on the next launch (or click Restart now).",
    "theme_restart":  "Save and re-launch ez-rag immediately so the new "
                      "palette takes effect.",

    # Files tab
    "add_files":      "Copy files from anywhere on disk into this workspace's "
                      "docs/ folder.",
    "add_folder":     "Recursively import every supported file from a folder.",
    "open_docs":      "Open docs/ in the OS file manager — useful for "
                      "drag-and-drop.",
    "refresh_files":  "Re-scan docs/ for any new files (without ingesting).",
    "ingest_btn":     "Parse → chunk → embed every file in docs/ that is new or "
                      "has changed since the last ingest. Idempotent.",
    "reingest_btn":   "Force re-process EVERY file even if unchanged. Use after "
                      "changing chunk size, embedder, or contextual retrieval.",

    # Settings — Ingest card
    "chunk_size":     "Tokens per text chunk. Larger = more context per chunk; "
                      "smaller = more chunks but more granular matching. 512 is "
                      "the standard default.",
    "chunk_overlap":  "Tokens repeated between consecutive chunks so a fact "
                      "split across the boundary is still retrievable. ~64 is "
                      "typical (~12% overlap).",
    "enable_ocr":     "Run OCR on PNG/JPG/WEBP/TIFF/BMP images and scanned "
                      "PDFs. Off = images skipped during ingest.",
    "contextual":     "Anthropic-style chunk-context summaries: prepends a "
                      "1–2-sentence summary to each chunk before embedding. "
                      "Slower ingest (one LLM call per chunk), but materially "
                      "better recall on technical/structured docs.",
    "preview_recoveries": "When the parser detects a garbled page and "
                      "re-extracts it via OCR, show a live before/after "
                      "card during ingest — the page rendered as an "
                      "image, the original garbled text, and the OCR "
                      "result side-by-side. Useful for sanity-checking "
                      "that recovery is actually working. Costs ~50ms "
                      "+ ~200KB disk per recovered page.",
    "llm_inspect_pages": "After parsing, send each section's text to the "
                      "LLM with a 'is this garbled?' prompt. Garbled "
                      "sections are dropped before chunking, so the index "
                      "doesn't get poisoned with font-cmap nonsense the "
                      "heuristic detector missed. EXPENSIVE — one LLM "
                      "call per section, so a 200-section book is 200 "
                      "calls (use a small/fast LLM during ingest, e.g. "
                      "qwen2.5:7b, not a reasoning model). Off by default.",
    "llm_correct_garbled": "Send OCR-recovered or LLM-flagged 'partial' "
                      "sections back to the LLM for a best-effort cleanup "
                      "pass before they're indexed. Fixes obvious OCR "
                      "misreads ('ShAMe' → 'Shame', missing spaces in run-"
                      "together words). The LLM may reject as "
                      "UNRECOVERABLE — those sections are dropped instead "
                      "of poisoning the index. Independent of LLM Inspect; "
                      "turn both on for the most aggressive recovery. "
                      "EXPENSIVE — one extra LLM call per questionable "
                      "section. Off by default.",

    # Settings — Retrieval card
    "use_corpus":     "Same as the toggle above the chat — when OFF, retrieval "
                      "is skipped entirely and the LLM answers from its own "
                      "knowledge.",
    "top_k":          "Number of passages retrieved per question. Higher = "
                      "richer context but slower and noisier. 8 is balanced.",
    "hybrid":         "Combine BM25 keyword search with dense vector search via "
                      "reciprocal rank fusion. Almost always better than either "
                      "alone — recommended on.",
    "rerank":         "Cross-encoder reranking — score the top candidates with a "
                      "small (~23 MB) model that judges (query, passage) pairs "
                      "jointly. Almost always the highest-impact retrieval "
                      "improvement; ~50–200 ms latency. ON by default.",
    "use_hyde":       "HyDE — have the LLM write a hypothetical answer first "
                      "and embed THAT for retrieval. Embeddings of an answer "
                      "match the corpus phrasing better than embeddings of a "
                      "bare question. Adds one extra LLM call per query.",
    "multi_query":    "Multi-query — ask the LLM for 2 paraphrases of your "
                      "question, retrieve for each, fuse with reciprocal rank "
                      "fusion. Helps for vague or short questions. Adds one "
                      "extra LLM call per query.",
    "context_window": "Neighbor expansion — when a chunk is retrieved, also "
                      "include the chunks immediately before and after it from "
                      "the same file. 0 = off, 1 = ±1 chunk, 2 = ±2. Useful "
                      "for narrative/long-form docs where the matching chunk "
                      "alone lacks enough context. Multiplies tokens sent to "
                      "the LLM by (2*window+1).",
    "use_mmr":        "MMR — Maximum Marginal Relevance. Rebalances the final "
                      "top-K to be diverse rather than near-duplicates. Helps "
                      "when the corpus has many similar chunks; harmless on "
                      "small/diverse corpora. ~70 ms overhead.",
    "expand_to_chapter": "Replace each retrieved chunk with the entire "
                      "chapter it belongs to. Best for 'summarize the rules "
                      "around X' / 'list everything about Y' questions. "
                      "Chapter boundaries come from PDF bookmarks or heading "
                      "styles. Skipped per-hit when a chapter exceeds the "
                      "cap below.",
    "chapter_max_chars": "Hard limit on expanded chapter size, in characters. "
                      "If a chapter is bigger than this we keep the original "
                      "chunk instead so a single hit can't blow your context "
                      "window. ~4 chars ≈ 1 token.",
    "mmr_lambda":     "MMR balance: 1.0 = pure relevance (same as off), "
                      "0.0 = pure diversity, 0.5 = balanced.",

    # Settings — LLM card
    "llm_model":      "Which Ollama tag handles chat. Pick from installed "
                      "models. Tags ending in :latest are normalized.",
    "refresh_models": "Re-query Ollama for the list of installed models.",
    "browse_llm":     "Browse the entire Ollama library (~230 public models) "
                      "with VRAM-fit estimates. Click any size chip to fill the "
                      "tag, then Pull.",
    "use_gguf":       "Use a local .gguf file instead of Ollama. Switches the "
                      "backend to llama-cpp-python — install with "
                      "`pip install llama-cpp-python`.",
    "ollama_url":     "URL where the Ollama daemon is reachable. Default is the "
                      "local install at 127.0.0.1:11434.",
    "temperature":    "Sampling temperature. 0.0 = deterministic / factual, "
                      "1.0 = creative / varied. 0.2 is a sane factual default.",
    "max_tokens":     "Maximum tokens the model can generate per answer. "
                      "Reasoning models like deepseek-r1 need 4096+ — the "
                      "thinking phase eats from this budget.",

    # Settings — Embedder card
    "embed_provider": "auto = use Ollama if reachable, fall back to fastembed.\n"
                      "ollama = always use the Ollama embed model below.\n"
                      "fastembed = always use the local CPU embedder below.",
    "ollama_embed":   "Embedding model when using Ollama. Pick from installed "
                      "models, or use Browse to pull a new one.",
    "browse_embed":   "Browse the Ollama library filtered to embedding models.",
    "embed_model":    "Embedding model when using fastembed (CPU-friendly, no "
                      "Ollama needed). Downloaded on first use to "
                      "~/.cache/fastembed.",

    # Save / Reset
    "save_btn":       "Persist the current settings to .ezrag/config.toml in "
                      "this workspace.",
    "reset_btn":      "Discard unsaved changes and reload from config.toml.",

    # Doctor
    "doctor_refresh": "Re-scan the environment.",

    # Citation chip
    "citation_chip":  "Click to view the exact passage that was retrieved.",
}


# ============================================================================
# Help and credits content
# ============================================================================

HELP_MD = """\
# ez-rag — quick reference

## What it does

Drop documents into a workspace folder. ez-rag parses, chunks, and embeds them.
When you ask a question, it retrieves the most relevant passages and feeds them
to the LLM as supplementary context.

The model is the primary intelligence — it answers from its own knowledge.
The corpus is **secondary reference material** the model can cite when relevant.
Toggle the **Use RAG** switch off (top of the Chat tab) to test the model
with no retrieval.

## Terminology

These three words are not synonyms — they refer to different layers of the
same system. Knowing the difference makes everything else in this manual
clearer.

| Term | Refers to |
|---|---|
| **Corpus** | The collection of source documents — your `docs/` folder. From linguistics, Latin for "body." A corpus is *the input*. |
| **Index** | The searchable structure ez-rag builds from the corpus — chunks + embeddings + BM25, all stored in `.ezrag/meta.sqlite`. *The processed data*. |
| **RAG** | Retrieval-Augmented Generation. *The whole technique*: corpus + index + retrieval logic + LLM, working together. The "RAG" dropdown at the top picks which workspace (= which corpus + index + frozen settings) is active. |

The Chat tab toggle is labelled **"Use RAG"** — when ON, your question is augmented with retrieved passages from the corpus before the LLM answers; when OFF, the question goes straight to the LLM with no retrieval (useful for A/B comparison).

Other recurring terms:

- **Embedder** / **embedding model** — the small model that turns text into vectors. Used at *both* ingest time (one vector per chunk) and query time (one vector per question). Swapping it forces a full re-ingest.
- **Embedding** / **vector** / **dense vector** — the numeric output of the embedder. Texts with similar meaning land at similar coordinates in vector space.
- **Reranker** / **cross-encoder** — a small *second* model that scores `(query, passage)` pairs jointly. Runs on the top candidates after initial retrieval. Biggest single quality lift in most pipelines; cheap.
- **Chunk** — a piece of a document the embedder sees. Default ~512 tokens. The thing actually retrieved.
- **Hybrid retrieval** — fuses BM25 (keyword) and dense (vector) results via Reciprocal Rank Fusion. Default ON.
- **Chapter** (ez-rag specific) — a contiguous span of chunks belonging to one section, derived from PDF bookmarks or heading styles. Used by the *Expand to chapter* retrieval mode.

## Tabs

- **Chat** — multi-turn conversation. Citation chips show which passages were used.
- **Files** — manage `docs/` and run ingest.
- **Settings** — chunk size, top-K, models, OCR. Hover any field for an explanation.
- **Doctor** — what's installed, what's reachable.

## Keyboard shortcuts

| Key | Action |
|---|---|
| Enter | Send the chat message |
| Shift + Enter | Insert a newline in the input |
| Ctrl + N | Switch to Chat tab |
| Ctrl + I | Switch to Files tab |
| Ctrl + , | Switch to Settings tab |

## Retrieval pipeline (smart defaults)

```
question → [HyDE]? → [multi-query]? → hybrid (BM25 + dense)
                                           ↓
                                       cross-encoder rerank → top-K → LLM
```

ON by default: **Hybrid** + **Rerank**. Both are essentially free in
quality terms and cost ~100 ms total. Opt-in (Settings → Retrieval):

- **HyDE** — LLM writes a hypothetical answer, embed *that* (helps when
  question and corpus use different phrasings)
- **Multi-query** — LLM generates 2 paraphrases, fuse with RRF (helps for
  vague or short questions)
- **Contextual Retrieval** (at ingest) — Anthropic-style chunk-context
  summaries. ~50 % fewer retrieval failures, but slow ingest.

Type `ez-rag help retrieval` (or the same topic in this overlay) for the
full menu of options and when to use them.

## Common workflows

**Add documents**: Files tab → *Add files* or *Add folder* (or just drop them
into the docs/ folder yourself), then *Ingest*.

**Pull a different model**: Settings → *Browse Ollama library*. Filter by
capability (LLMs / Vision / Reasoning / Embedding). Each size chip shows
estimated VRAM and color-codes whether it fits your GPU.

**Use a local GGUF instead of Ollama**: Settings → *Use local GGUF…*. Install
`llama-cpp-python` once.

**Test with vs without RAG**: toggle *Use RAG* in the Chat tab header.
Same question, both modes — direct comparison.

**Run as an OpenAI-compatible API**: from a terminal, `ez-rag serve --port 11533`,
then point any OpenAI client at `http://127.0.0.1:11533/v1`.

## Where data lives

```
<workspace>/
├── docs/                     your files
└── .ezrag/
    ├── config.toml           settings
    ├── meta.sqlite           index (files, chunks, FTS, embeddings)
    └── ingest.log
```

Nothing leaves your machine after the first model download.

## Troubleshooting

- **No LLM answers** → install Ollama from ollama.com, then *Browse Ollama
  library* and pull e.g. `qwen2.5:7b-instruct`. Or `pip install llama-cpp-python`
  for a local GGUF.
- **Reasoning model gives empty answers** → bump *Max tokens* in Settings.
  deepseek-r1 needs ≥4096 (thinking eats budget).
- **Wrong file ranked first** → try increasing *Top-K* and/or enabling
  *Rerank* and *Contextual Retrieval*.
- **Image / scanned PDF text not picked up** → make sure *OCR images / scanned
  PDFs* is on; check Doctor for RapidOCR / Tesseract status.
"""

CREDITS_MD = """\
# Credits

**ez-rag** — drop documents in a folder, chat with them.

## Made by

[**Justin Rounds**](https://www.justinrounds.com) — designer, sounding board,
and the human ultimately responsible for whatever this is.

## Co-written with

A revolving cast of large language models. They wrote a lot of the code, took
none of the blame for the bugs, and at no point asked for breaks. We won't
name them — they know who they are.

## Standing on the shoulders of

- **Ollama** and **llama.cpp** — for making local inference look easy
- **Flet** — desktop GUI in plain Python, no Electron in sight
- **fastembed** + **RapidOCR** — embeddings and OCR that don't pull in 4 GB of
  PyTorch
- **pypdf**, **python-docx**, **openpyxl**, **beautifulsoup4** — the patient
  middle layer of the data world
- **SQLite + FTS5 + numpy** — still the right answer ten years later

## Special thanks to

- Every LLM that has helped someone debug a regex at 2 a.m.
- The maintainers of unglamorous middle-layer libraries
- Coke Zero, mostly
- That screenshot you accidentally pasted instead of the URL — it turned out to
  be exactly what was needed
- You, for reading this far

## License

Apache 2.0 in spirit. Use it, fork it, name your goldfish after it.
"""


def _theme(page: ft.Page) -> None:
    page.theme = ft.Theme(
        color_scheme_seed=ACCENT,
        font_family="Inter",
        visual_density=ft.VisualDensity.COMFORTABLE,
        use_material3=True,
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed=ACCENT,
        font_family="Inter",
        use_material3=True,
    )


# ============================================================================
# State
# ============================================================================

@dataclass
class ChatTurn:
    role: str           # "user" | "assistant"
    text: str = ""
    thinking: str = ""  # reasoning-model `<think>` content (deepseek-r1 etc.)
    citations: list = field(default_factory=list)   # list[Hit]
    streaming: bool = False
    md_ctrl: object = None        # ft.Markdown reference for in-place updates
    thinking_md: object = None    # ft.Markdown for the reasoning section
    thinking_box: object = None   # ft.Container wrapping thinking section
    thinking_header: object = None  # ft.Text header (so we can show "thinking..."
                                    # vs "Reasoning · 1234 chars")
    chips_row: object = None
    # Inline recovery actions, rendered as buttons under the bubble.
    # list of (label, icon, callable) — callable takes no args.
    actions: list = field(default_factory=list)
    bubble: object = None


@dataclass
class AppState:
    page: ft.Page
    ws: Workspace | None = None
    cfg: Config = field(default_factory=Config)
    turns: list[ChatTurn] = field(default_factory=list)
    streaming: bool = False
    stop_flag: bool = False
    # Cached LLM-generated chat-welcome suggestions, keyed by workspace path.
    # None = not requested yet, [] = requested and got nothing back.
    suggestions: dict = field(default_factory=dict)
    suggestions_loading: bool = False
    # Active background Ollama pulls: tag -> {pct, completed, total, status,
    # started_at}. Lives at app level (not per-dialog) so the pull can
    # outlive the dialog that started it. Surfaces in the header badge
    # and resyncs the dialog when reopened.
    active_pulls: dict = field(default_factory=dict)


# ============================================================================
# Recents
# ============================================================================

def load_recents() -> list[Path]:
    if not RECENTS_FILE.exists():
        return []
    out = []
    for line in RECENTS_FILE.read_text(encoding="utf-8").splitlines():
        p = Path(line.strip())
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def add_recent(path: Path) -> None:
    cur = [p for p in load_recents() if p.resolve() != path.resolve()]
    cur.insert(0, path)
    cur = cur[:MAX_RECENTS]
    RECENTS_FILE.write_text(
        "\n".join(str(p) for p in cur) + "\n", encoding="utf-8"
    )


# ============================================================================
# Reusable widgets
# ============================================================================

# ---------------------------------------------------------------------------
# Win32 paint-message kicker
# ---------------------------------------------------------------------------
# Flet 0.84 / Flutter Desktop on Windows pauses repaints when nothing is
# triggering a frame. `page.update()` from a background thread sets the
# value but the renderer doesn't actually flush until the OS sends a
# WM_PAINT — which happens when you click in the window, alt-tab, or
# even Win+S (the snipping tool briefly covers the window). The fix is
# to send our own WM_PAINT periodically by calling RedrawWindow on our
# Flutter HWND. Same effect as Win+S but invisible.
#
# Best-effort + cached. Returns None and silently no-ops on non-Windows
# or if the HWND can't be found.

_FLUTTER_HWND: int | None = None
# Flutter desktop's window class name on Windows. The Flutter team uses
# this exact identifier for the embedded view window — confirmed in the
# flutter/engine source. Flet ships flet-desktop (a Flutter app) as a
# separate process, so its window is NOT owned by our Python process —
# we have to find it by class name across all processes.
_FLUTTER_WIN_CLASS = "FLUTTER_RUNNER_WIN32_WINDOW"


def _find_flutter_hwnd() -> int | None:
    """Locate the Flutter view window. Looks for the FLUTTER_RUNNER_WIN32_WINDOW
    class across all processes (since flet-desktop runs out-of-process).
    Falls back to title-prefix matching ("ez-rag") if class lookup fails.
    Caches on first success; retries on every call until found."""
    global _FLUTTER_HWND
    if _FLUTTER_HWND is not None:
        # Validate cache — handle the case where the window was closed
        # and recreated (re-launch via RAG-switch, theme change, etc.).
        try:
            import ctypes
            if ctypes.windll.user32.IsWindow(_FLUTTER_HWND):
                return _FLUTTER_HWND
        except Exception:
            pass
        _FLUTTER_HWND = None
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32

        # Strategy 1: FindWindowW by class name. Fast and works regardless
        # of process ownership.
        hwnd = user32.FindWindowW(_FLUTTER_WIN_CLASS, None)
        if hwnd:
            _FLUTTER_HWND = int(hwnd)
            return _FLUTTER_HWND

        # Strategy 2: enumerate all top-level windows, match by class
        # OR by visible title starting with 'ez-rag'. Some Flet versions
        # may use a different class name (FLUTTER_VIEW, etc.).
        candidates: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _enum_proc(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            # Class name check
            buf = ctypes.create_unicode_buffer(256)
            n = user32.GetClassNameW(hwnd, buf, 256)
            cls = buf.value if n > 0 else ""
            if "FLUTTER" in cls.upper():
                candidates.append(int(hwnd))
                return True
            # Title check
            tlen = user32.GetWindowTextLengthW(hwnd)
            if tlen > 0:
                tbuf = ctypes.create_unicode_buffer(tlen + 1)
                user32.GetWindowTextW(hwnd, tbuf, tlen + 1)
                if tbuf.value and tbuf.value.lower().startswith("ez-rag"):
                    candidates.append(int(hwnd))
            return True

        user32.EnumWindows(_enum_proc, 0)
        if candidates:
            _FLUTTER_HWND = candidates[0]
    except Exception:
        pass
    return _FLUTTER_HWND


def force_window_redraw() -> bool:
    """Force a Flutter Desktop repaint by invalidating the Flutter view
    window and pumping a UpdateWindow. Same effect as Win+S / alt-tab
    waking the app — but invisible.

    Background: Flutter on Windows handles WM_PAINT by doing nothing
    (default WindowProc), and Flet's `page.update()` from a background
    thread queues the value change but doesn't itself trigger a frame.
    The OS only sends paint messages on input/focus events. So we send
    one ourselves, which is exactly what alt-tab + alt-tab back does.
    See flutter/flutter#75319 and flutter/flutter#102030.

    Returns True on success. No-op on non-Windows or if the HWND can't
    be found yet."""
    if os.name != "nt":
        return False
    hwnd = _find_flutter_hwnd()
    if not hwnd:
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        # Invalidate the entire client area + force an immediate paint
        # for all child windows. RDW_FRAME also includes the non-client
        # area for good measure. RDW_ERASE forces background re-paint.
        # 0x0001 = RDW_INVALIDATE
        # 0x0004 = RDW_ERASE
        # 0x0080 = RDW_ALLCHILDREN
        # 0x0100 = RDW_UPDATENOW
        # 0x0400 = RDW_FRAME
        flags = 0x0001 | 0x0004 | 0x0080 | 0x0100 | 0x0400
        user32.RedrawWindow(hwnd, None, None, flags)
        # Belt-and-suspenders: explicit InvalidateRect + UpdateWindow.
        # Some Flutter engine builds respond to one but not the other.
        user32.InvalidateRect(hwnd, None, False)
        user32.UpdateWindow(hwnd)
        return True
    except Exception:
        return False


def modal_heartbeat() -> ft.Control:
    """Tiny always-spinning ProgressRing meant to be embedded inside any
    long-lived modal overlay.

    Flutter Desktop on Windows pauses repaints when no input arrives and
    nothing in the visible widget tree is animating. The sysmon footer's
    heartbeat covers the main page, but modal overlays render in their
    own stack and don't always inherit those frame ticks. Drop one of
    these into any overlay where progress (download bytes, ingest pages,
    etc.) needs to keep ticking without user clicks.

    The ring is 8x8 px and dim — barely visible, but its rotation forces
    Flutter to keep rendering the overlay's subtree at vsync rate.
    """
    return ft.ProgressRing(
        width=8, height=8, stroke_width=1,
        color=ft.Colors.with_opacity(0.5, ACCENT),
        bgcolor="transparent",
        tooltip="Live indicator — keeps this dialog repainting so "
                "progress stays current without clicking.",
    )


def status_pill(text: str, color: str, *, dot: bool = True) -> ft.Container:
    return ft.Container(
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        bgcolor=ft.Colors.with_opacity(0.12, color),
        border_radius=999,
        content=ft.Row(
            [
                ft.Container(width=8, height=8, bgcolor=color,
                             border_radius=999) if dot else ft.Container(),
                ft.Text(text, size=12, color=color, weight=ft.FontWeight.W_500),
            ],
            spacing=6,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def section_card(title: str, *children: ft.Control,
                 padding: int = 18) -> ft.Container:
    return ft.Container(
        bgcolor=SURFACE_DARK,
        border=ft.border.all(1, "#262938"),
        border_radius=12,
        padding=padding,
        content=ft.Column(
            [
                ft.Text(title, size=11, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE_DIM),
                ft.Container(height=10),
                *children,
            ],
            spacing=8,
        ),
    )


def empty_state(
    *,
    icon: str,
    title: str,
    subtitle: str,
    actions: list[ft.Control] | None = None,
) -> ft.Container:
    return ft.Container(
        expand=True,
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            [
                ft.Icon(icon, size=64, color=ON_SURFACE_DIM),
                ft.Container(height=12),
                ft.Text(title, size=22, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE),
                ft.Text(subtitle, size=14, color=ON_SURFACE_DIM,
                        text_align=ft.TextAlign.CENTER),
                ft.Container(height=20),
                ft.Row(actions or [], alignment=ft.MainAxisAlignment.CENTER,
                       spacing=12),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
    )


def file_icon(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".pdf":  ft.Icons.PICTURE_AS_PDF,
        ".docx": ft.Icons.DESCRIPTION,
        ".xlsx": ft.Icons.TABLE_CHART,
        ".xlsm": ft.Icons.TABLE_CHART,
        ".csv":  ft.Icons.TABLE_VIEW,
        ".tsv":  ft.Icons.TABLE_VIEW,
        ".html": ft.Icons.HTML,
        ".htm":  ft.Icons.HTML,
        ".md":   ft.Icons.NOTES,
        ".markdown": ft.Icons.NOTES,
        ".txt":  ft.Icons.NOTES,
        ".rst":  ft.Icons.NOTES,
        ".log":  ft.Icons.NOTES,
        ".epub": ft.Icons.MENU_BOOK,
        ".eml":  ft.Icons.MAIL,
        ".png":  ft.Icons.IMAGE,
        ".jpg":  ft.Icons.IMAGE,
        ".jpeg": ft.Icons.IMAGE,
        ".webp": ft.Icons.IMAGE,
        ".tiff": ft.Icons.IMAGE,
        ".bmp":  ft.Icons.IMAGE,
    }.get(ext, ft.Icons.INSERT_DRIVE_FILE)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ============================================================================
# Citation card / dialog
# ============================================================================

def citation_chip(idx: int, hit, on_click) -> ft.Container:
    label = hit.path.split("/")[-1].split("\\")[-1]
    page_str = f" · p.{hit.page}" if hit.page else ""
    chip = ft.Container(
        bgcolor=CHIP_BG,
        border=ft.border.all(1, "#2E3245"),
        border_radius=999,
        padding=ft.padding.symmetric(horizontal=10, vertical=6),
        on_click=lambda _: on_click(idx, hit),
        content=ft.Row(
            [
                ft.Container(
                    width=20, height=20, border_radius=999,
                    bgcolor=ft.Colors.with_opacity(0.25, ACCENT),
                    alignment=ft.Alignment.CENTER,
                    content=ft.Text(str(idx), size=11,
                                    weight=ft.FontWeight.W_700, color=ACCENT),
                ),
                ft.Text(label, size=12, color=ON_SURFACE,
                        weight=ft.FontWeight.W_500, no_wrap=True),
                ft.Text(page_str, size=12, color=ON_SURFACE_DIM, no_wrap=True),
            ],
            tight=True, spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    chip.tooltip = f"{hit.path}{page_str}\nClick to view source"
    return chip


# ============================================================================
# Header
# ============================================================================

def build_header(state: AppState, *, on_open_workspace, on_create_rag,
                 on_pick_workspace, refresh_status):
    workspace_text = ft.Text("(no workspace)", size=14, color=ON_SURFACE_DIM)
    backend_pill = status_pill("offline", DANGER)
    backend_pill.tooltip = TIP["backend_pill"]
    files_pill = status_pill("0 files", ACCENT)
    files_pill.tooltip = TIP["files_pill"]

    # Recent-RAGs dropdown — switch between named workspaces, plus an action
    # item to spawn the create-RAG overlay.
    NEW_RAG_KEY = "__new_rag__"
    OPEN_FOLDER_KEY = "__open_folder__"
    rag_dropdown = ft.Dropdown(
        label="RAG",
        width=240,
        dense=True,
        tooltip=TIP["rag_dropdown"],
        options=[],
    )

    def refresh_rag_dropdown():
        opts: list[ft.dropdown.Option] = []
        seen = set()
        if state.ws is not None:
            cur = str(state.ws.root)
            opts.append(ft.dropdown.Option(
                key=cur, text=f"● {state.ws.root.name}",
            ))
            seen.add(cur)
        # 1) recents (most recently opened)
        for p in load_recents():
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            opts.append(ft.dropdown.Option(key=sp, text=p.name))
        # 2) every RAG present in the default RAGs folder
        for w in list_managed_rags():
            sp = str(w.root)
            if sp in seen:
                continue
            seen.add(sp)
            opts.append(ft.dropdown.Option(key=sp, text=w.root.name))
        opts.append(ft.dropdown.Option(key=NEW_RAG_KEY,
                                       text="+ New RAG…"))
        opts.append(ft.dropdown.Option(key=OPEN_FOLDER_KEY,
                                       text="+ Open existing folder…"))
        rag_dropdown.options = opts
        if state.ws is not None:
            rag_dropdown.value = str(state.ws.root)

    def on_rag_changed(e):
        v = e.control.value
        if v == NEW_RAG_KEY:
            # Reset to current workspace value visually, then open wizard.
            rag_dropdown.value = (str(state.ws.root) if state.ws else "")
            state.page.update()
            on_create_rag()
        elif v == OPEN_FOLDER_KEY:
            rag_dropdown.value = (str(state.ws.root) if state.ws else "")
            state.page.update()
            on_open_workspace(None)
        elif v and (state.ws is None or str(state.ws.root) != v):
            on_pick_workspace(Path(v))

    rag_dropdown.on_change = on_rag_changed

    btn_new_rag = ft.IconButton(
        icon=ft.Icons.AUTO_AWESOME,
        tooltip=TIP["new_rag"],
        on_click=lambda _: on_create_rag(),
    )
    btn_open = ft.OutlinedButton(
        "Open folder",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=on_open_workspace,
        tooltip=TIP["open_workspace"],
    )
    btn_help = ft.IconButton(
        icon=ft.Icons.HELP_OUTLINE,
        tooltip=TIP["help_btn"],
        on_click=lambda _: open_info_overlay(
            state.page, title="Help",
            body_md=HELP_MD, icon=ft.Icons.HELP_OUTLINE,
        ),
    )
    # Pull badge — visible only while a model download is running. Hidden
    # otherwise. Lets the user see a download still progressing even after
    # they've dismissed the pull dialog. Updated by the sysmon watchdog
    # tick (already running at 1 Hz to keep Flutter painting).
    pull_badge_text = ft.Text(
        "", size=11, color="#FFFFFF",
        weight=ft.FontWeight.W_700,
    )
    pull_badge = ft.Container(
        visible=False,
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        bgcolor=ACCENT, border_radius=999,
        tooltip="Active model downloads — keeps running even if you close "
                "the pull dialog. Clears when complete.",
        content=ft.Row([
            ft.Icon(ft.Icons.CLOUD_DOWNLOAD, size=14, color="#FFFFFF"),
            pull_badge_text,
        ], spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
    )

    btn_about = ft.IconButton(
        icon=ft.Icons.INFO_OUTLINE,
        tooltip=TIP["about_btn"],
        on_click=lambda _: open_info_overlay(
            state.page, title="About & Credits",
            body_md=CREDITS_MD, icon=ft.Icons.AUTO_AWESOME,
        ),
    )

    bar = ft.Container(
        bgcolor=SURFACE_DARK,
        padding=ft.padding.symmetric(horizontal=20, vertical=12),
        border=ft.border.only(bottom=ft.BorderSide(1, "#22253333")),
        content=ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(
                            width=28, height=28, border_radius=8,
                            bgcolor=ACCENT, alignment=ft.Alignment.CENTER,
                            content=ft.Text("ez", size=13,
                                            weight=ft.FontWeight.W_900,
                                            color="#FFFFFF"),
                        ),
                        ft.Text("ez-rag", size=16,
                                weight=ft.FontWeight.W_700,
                                color=ON_SURFACE),
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Container(width=20),
                ft.VerticalDivider(width=1, color="#22253355"),
                ft.Container(width=10),
                ft.Icon(ft.Icons.FOLDER_OUTLINED, size=16,
                        color=ON_SURFACE_DIM),
                workspace_text,
                ft.Container(expand=True),
                files_pill,
                backend_pill,
                pull_badge,
                rag_dropdown,
                btn_new_rag,
                btn_open,
                btn_help,
                btn_about,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
    )

    def update():
        refresh_rag_dropdown()
        if state.ws is None:
            workspace_text.value = "(no workspace)"
            workspace_text.color = ON_SURFACE_DIM
            backend_pill.content = status_pill("offline", DANGER).content
            files_pill.content = status_pill("—", ON_SURFACE_DIM).content
            return
        # Workspace name
        workspace_text.value = state.ws.root.name
        workspace_text.color = ON_SURFACE

        backend = detect_backend(state.cfg)
        if backend == "ollama":
            backend_pill.content = status_pill(
                f"Ollama · {state.cfg.llm_model}", SUCCESS).content
        elif backend == "llama-cpp":
            backend_pill.content = status_pill(
                f"llama.cpp · {Path(state.cfg.llm_model).name}", SUCCESS).content
        else:
            backend_pill.content = status_pill("retrieval-only", WARNING).content

        s = read_stats(state.ws.meta_db_path)
        if s is None or s["files"] == 0:
            files_pill.content = status_pill("no docs", ON_SURFACE_DIM).content
        else:
            files_pill.content = status_pill(
                f"{s['files']} files · {s['chunks']} chunks", ACCENT).content

    def refresh_pull_badge():
        """Refresh the header download badge from state.active_pulls.
        Called by the sysmon watchdog at 1 Hz so progress stays live even
        when the pull dialog is closed."""
        pulls = state.active_pulls
        if not pulls:
            pull_badge.visible = False
            return
        pull_badge.visible = True
        # Take the first (or only) active pull's progress for the label.
        # Multi-pull is rare; we just show "N downloading" if so.
        if len(pulls) == 1:
            tag, info = next(iter(pulls.items()))
            pct = info.get("pct")
            if pct is not None:
                pull_badge_text.value = f"{tag}  {int(pct * 100)}%"
            else:
                pull_badge_text.value = f"{tag}  {info.get('status', 'pulling')}"
        else:
            pull_badge_text.value = f"{len(pulls)} downloads"

    return bar, update, refresh_pull_badge


# ============================================================================
# Chat view
# ============================================================================

def build_chat_view(state: AppState, *, refresh_status,
                    on_open_workspace, on_open_files,
                    open_pull_dialog_cb):
    # Chat-view-scoped FilePicker for citation-image downloads + chapter
    # PDF exports. Registered as a service so save dialogs work. Lives
    # here (not borrowed from settings view) so we don't reach across
    # closures — that was the "name 'rags_dir_picker' is not defined"
    # bug a user hit when clicking Download on a citation page render.
    download_picker = ft.FilePicker()
    state.page.services.append(download_picker)

    # Follow-mode state for the chat list. When the user is at (or near)
    # the bottom we auto-scroll new content into view; when they've
    # scrolled up to read backscroll we leave them alone instead of
    # yanking them down on every streamed token.
    chat_follow_state = {"follow": True}

    def _on_chat_scroll(e):
        # Flet 0.84 ListView OnScrollEvent exposes pixels and max_scroll_extent.
        # We're "following" when within 80 px of the bottom.
        try:
            pixels = float(getattr(e, "pixels", 0) or 0)
            max_extent = float(getattr(e, "max_scroll_extent", 0) or 0)
            if max_extent <= 0:
                chat_follow_state["follow"] = True
                return
            chat_follow_state["follow"] = (max_extent - pixels) <= 80
        except Exception:
            pass

    chat_list = ft.ListView(
        expand=True, spacing=12, padding=ft.padding.all(20),
        # auto_scroll=True only fires on new appends, not in-place
        # updates — we manage scrolling manually via _scroll_to_bottom.
        auto_scroll=False,
        on_scroll=_on_chat_scroll,
    )
    input_field = ft.TextField(
        hint_text="Ask anything…   Enter to send · Shift+Enter for newline",
        multiline=True, min_lines=1, max_lines=6, expand=True,
        shift_enter=True,
        border_color="#2A2D3D",
        focused_border_color=ACCENT,
        cursor_color=ACCENT,
        text_size=14,
        bgcolor=SURFACE_DARK_HI,
        content_padding=ft.padding.symmetric(horizontal=14, vertical=12),
        tooltip=TIP["chat_input"],
    )
    send_btn = ft.IconButton(
        icon=ft.Icons.ARROW_UPWARD,
        bgcolor=ACCENT,
        icon_color="#FFFFFF",
        tooltip=TIP["send_btn"],
    )
    stop_btn = ft.IconButton(
        icon=ft.Icons.STOP_CIRCLE_OUTLINED,
        icon_color=DANGER,
        tooltip=TIP["stop_btn"],
        visible=False,
    )

    # RAG on/off toggle for the chat header
    rag_switch = ft.Switch(
        value=state.cfg.use_rag,
        active_color=ACCENT,
        tooltip=TIP["rag_toggle"],
    )
    rag_label = ft.Text(
        "Use RAG" if state.cfg.use_rag else "Bypass RAG (model only)",
        size=12, color=ON_SURFACE_DIM, weight=ft.FontWeight.W_600,
    )

    # Per-query "apply modifiers" checkbox for the chat composer
    modifier_check = ft.Checkbox(
        value=state.cfg.apply_query_modifiers,
        active_color=ACCENT,
        label="Modifiers",
        label_style=ft.TextStyle(size=11, color=ON_SURFACE_DIM),
        tooltip=TIP["modifier_toggle"],
    )

    def on_modifier_toggle(e):
        state.cfg.apply_query_modifiers = bool(e.control.value)
        if state.ws is not None:
            try:
                state.cfg.save(state.ws.config_path)
            except Exception:
                pass

    modifier_check.on_change = on_modifier_toggle

    def on_rag_toggle(e):
        state.cfg.use_rag = bool(e.control.value)
        rag_label.value = (
            "Use RAG" if state.cfg.use_rag
            else "Bypass corpus (model only)"
        )
        rag_label.color = ACCENT if state.cfg.use_rag else WARNING
        # Persist if a workspace is open
        if state.ws is not None:
            try:
                state.cfg.save(state.ws.config_path)
            except Exception:
                pass
        state.page.update()

    rag_switch.on_change = on_rag_toggle

    # Source dialog (shown when a citation chip is clicked)
    source_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Source"),
        content=ft.Container(width=720, height=480, content=ft.Text("")),
    )

    def _try_render_page_image(hit) -> ft.Control | None:
        """Render the cited PDF page → zoomable image control with download.
        Returns None for any failure path (non-PDF, missing renderer, broken
        file, etc.). Must never raise — citation chips are user-clicked and
        a crash here would brick the chat dialog."""
        try:
            if not hit.path.lower().endswith(".pdf"):
                return None
            if not hit.page or state.ws is None:
                return None
            from ez_rag.preview import render_pdf_page
            abs_path = (state.ws.root / hit.path).resolve()
            img_path = render_pdf_page(abs_path, int(hit.page))
            if img_path is None:
                return None

            # InteractiveViewer gives pan + pinch-to-zoom on the rendered
            # page (mouse-wheel zoom + click-drag pan on desktop). The
            # image itself is already rendered at 2.5x natural scale so
            # zoom-in stays sharp.
            inner_img = ft.Image(
                src=str(img_path),
                fit="contain",
                border_radius=4,
            )
            viewer = ft.InteractiveViewer(
                min_scale=0.5, max_scale=8.0,
                content=ft.Container(
                    content=inner_img, alignment=ft.Alignment.CENTER,
                ),
                expand=True,
            )

            def download_clicked(_=None):
                async def _do():
                    src = Path(hit.path).stem
                    suggested = f"{src}-p{hit.page}.png"
                    try:
                        dest = await download_picker.save_file(
                            dialog_title="Save page image",
                            file_name=suggested,
                            allowed_extensions=["png"],
                        )
                    except Exception as ex:
                        _toast(state.page, f"Save dialog failed: {ex}")
                        return
                    if not dest:
                        return
                    try:
                        shutil.copy2(img_path, dest)
                        _toast(state.page, f"Saved → {Path(dest).name}")
                    except Exception as ex:
                        _toast(state.page, f"Save failed: {ex}")
                state.page.run_task(_do)

            def open_chapter_in_browser(_=None):
                """Extract a windowed PDF centered on the cited page and
                open it in the user's default browser. The extract has:
                  - Page 1: the cited page (cover)
                  - Page 2: notes explaining the layout
                  - Pages 3..: ±7 source pages around the cited page
                Total ~17 pages instead of dumping the whole source.

                Replaces the old chapter-based extract — chapter
                metadata was unreliable on PDFs without bookmarks
                (e.g. D&D Basic Rules indexed as one 320-page chapter,
                so the chapter button used to dump the whole book).

                Cached under ~/.ezrag/chapter_cache/ so re-clicking the
                same page+window reuses the file.
                """
                def _bg():
                    if state.ws is None:
                        _toast(state.page, "Open a workspace first.")
                        return
                    try:
                        from ez_rag.preview import (
                            extract_pdf_window, window_cache_path_for,
                        )
                        if not hit.page or hit.page <= 0:
                            _toast(state.page,
                                "This hit has no page number — "
                                "windowed extract needs a target page.")
                            return
                        WINDOW = 7
                        abs_pdf = (state.ws.root / hit.path).resolve()
                        if not abs_pdf.is_file():
                            _toast(state.page,
                                f"Source PDF not found: {hit.path}")
                            return
                        cache_path = window_cache_path_for(
                            abs_pdf, hit.page, WINDOW,
                        )
                        title = (
                            f"{abs_pdf.stem} — page {hit.page} ±{WINDOW}"
                        )
                        if not cache_path.exists():
                            out = extract_pdf_window(
                                abs_pdf, hit.page, cache_path,
                                window=WINDOW, title=title,
                            )
                            if out is None:
                                _toast(state.page,
                                    "Windowed extract failed. pypdf may "
                                    "be missing, or the page is outside "
                                    "the document.")
                                return
                        import webbrowser
                        url = cache_path.resolve().as_uri()
                        if not webbrowser.open(url):
                            _toast(state.page,
                                f"Couldn't launch browser. PDF is at: "
                                f"{cache_path}")
                            return
                        _toast(state.page,
                            f"Opened page {hit.page} ±{WINDOW} from "
                            f"{abs_pdf.name} in your browser.")
                    except Exception as ex:
                        _toast(state.page,
                                f"Windowed extract failed: {ex}")
                state.page.run_thread(_bg)

            # Toolbar — cleaner now that chapter routes through the browser
            # (no save dialog needed). Page label on the left, two clearly
            # differentiated actions on the right: a primary "Chapter"
            # action (the new flow) and a secondary "Save image" action.
            toolbar = ft.Row([
                ft.Icon(ft.Icons.PICTURE_AS_PDF, size=14,
                        color=ON_SURFACE_DIM),
                ft.Text(f"page {hit.page} · 2.5× render",
                        size=11, color=ON_SURFACE_DIM),
                ft.Container(expand=True),
                ft.FilledTonalButton(
                    "Chapter preview",
                    icon=ft.Icons.AUTO_STORIES,
                    on_click=open_chapter_in_browser,
                    tooltip=("Experimental — extract just the chapter "
                             "containing this passage and open it inline "
                             "in your default browser. Use the browser's "
                             "built-in toolbar to save, print, or copy "
                             "text. Boundaries come from the source "
                             "PDF's bookmarks; quality varies."),
                ),
                ft.OutlinedButton(
                    "Save image", icon=ft.Icons.IMAGE,
                    on_click=download_clicked,
                    tooltip="Save this rendered page as a PNG.",
                ),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=6)

            return ft.Container(
                bgcolor=SURFACE_DARK_HI, border_radius=8, padding=8,
                content=ft.Column([
                    toolbar,
                    ft.Container(
                        content=viewer,
                        bgcolor="#0F0F12",
                        border=ft.border.all(1, "#1E2130"),
                        border_radius=6,
                        expand=True,
                    ),
                    ft.Text(
                        "Scroll to zoom · drag to pan · double-click to reset",
                        size=10, color=ON_SURFACE_DIM, italic=True,
                    ),
                ], spacing=6, expand=True),
                expand=True,
            )
        except Exception as ex:
            print(f"[ez-rag] page-preview render failed: {ex!r}")
            return None

    def open_source(idx: int, hit):
        # Wrap the whole dialog build so a render failure shows a toast and
        # falls back to text-only — it must never leave the user stuck on an
        # unrecoverable error overlay.
        try:
            sec = f"  ·  {hit.section}" if hit.section else ""
            source_dialog.title = ft.Row(
                [
                    ft.Container(
                        width=28, height=28, border_radius=999,
                        bgcolor=ft.Colors.with_opacity(0.25, ACCENT),
                        alignment=ft.Alignment.CENTER,
                        content=ft.Text(str(idx), size=12,
                                        weight=ft.FontWeight.W_700, color=ACCENT),
                    ),
                    ft.Column([
                        ft.Text(Path(hit.path).name, size=14,
                                weight=ft.FontWeight.W_700),
                        ft.Text(f"{hit.path}{sec}{('  ·  page ' + str(hit.page)) if hit.page else ''}",
                                size=11, color=ON_SURFACE_DIM),
                    ], spacing=2, tight=True),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12,
            )

            page_image_ctrl = _try_render_page_image(hit)

            text_panel = ft.Container(
                bgcolor=SURFACE_DARK_HI, border_radius=8, padding=14,
                content=ft.Column([
                    ft.Text(hit.text, size=13, selectable=True,
                            color=ON_SURFACE),
                ], scroll=ft.ScrollMode.AUTO, expand=True),
                expand=True,
            )

            # Side-by-side layout when we have a rendered page; text-only otherwise.
            if page_image_ctrl is not None:
                body = ft.Row([
                    ft.Container(content=page_image_ctrl, expand=1),
                    ft.Container(content=text_panel, expand=1),
                ], expand=True, spacing=10)
                dialog_w = 980
            else:
                body = text_panel
                dialog_w = 720

            source_dialog.content = ft.Container(
                width=dialog_w, height=520, padding=10, content=body,
            )
            source_dialog.actions = [
                ft.TextButton("Close",
                              on_click=lambda _: state.page.pop_dialog()),
            ]
            state.page.show_dialog(source_dialog)
        except Exception as ex:
            print(f"[ez-rag] source dialog failed: {ex!r}")
            _toast(state.page, f"Couldn't open source: {ex}")

    # ------------- bubble rendering --------------------------------------

    def _actions_for_embedder_mismatch(*, last_question: str) -> list:
        """Recovery actions for an EmbedderMismatchError.

        Two paths: re-ingest (slow but always works) or open Settings so
        the user can flip the embedder back to whatever built the index.
        Plus a Retry once they've fixed it.
        """
        def reingest():
            on_open_files()
            _toast(state.page,
                "Files tab opened — click 'Re-ingest (force)' to rebuild "
                "every chunk vector with the current embedder.")

        def open_settings():
            # `set_view` is exposed on state by the parent app() — fall
            # back to a toast if not bound (e.g. when called from a unit
            # test harness).
            try:
                state.page.run_thread(lambda: None)  # touch
            except Exception:
                pass
            _toast(state.page,
                "Open Settings → Embedder and switch back to whatever "
                "built the index, then retry.")

        def retry_question():
            input_field.value = last_question
            state.page.update()
            send_clicked()

        return [
            ("Re-ingest now", ft.Icons.REFRESH, reingest, True),
            ("Open Settings", ft.Icons.SETTINGS, open_settings, False),
            ("Retry question", ft.Icons.SEND, retry_question, False),
        ]

    def _actions_for_ollama_error(ex, *, last_question: str) -> list:
        """Build inline recovery buttons for an OllamaChatError. Each
        action is a (label, icon, callable) tuple consumed by `_bubble`.

        The actions don't return — they kick off background work and
        toast/refresh as needed. Callers store the list on `ChatTurn`.
        """
        from ez_rag.generate import OllamaChatError
        kind = getattr(ex, "kind", "generic")
        cfg = state.cfg
        url = cfg.llm_url
        model = cfg.llm_model
        out: list = []

        def repair_model():
            # rm + pull the offending tag. Re-uses the existing pull dialog
            # so the user gets a streaming progress bar for free.
            def _do():
                try:
                    delete_ollama_model(url, model)
                except Exception:
                    pass
                _toast(state.page,
                       f"Removed {model} — opening pull dialog to redownload")
            state.page.run_thread(_do)
            # Open the pull dialog with the tag pre-filled so the user can
            # start the re-pull with one click.
            opener = open_pull_dialog_cb.get("fn")
            if opener is None:
                _toast(state.page,
                       "Pull dialog not ready yet — try Settings → "
                       "Browse Ollama library.")
                return
            try:
                opener("llm", None, prefill_tag=model)
            except Exception as ex2:
                _toast(state.page, f"Couldn't open pull dialog: {ex2}")

        def free_vram():
            def _do():
                try:
                    out_tags = unload_running_models(url)
                except Exception:
                    out_tags = []
                if out_tags:
                    _toast(state.page,
                           f"Unloaded from VRAM: {', '.join(out_tags)}")
                else:
                    _toast(state.page,
                           "Nothing was loaded — nothing to unload.")
            state.page.run_thread(_do)

        def open_ollama_download():
            try:
                import webbrowser
                webbrowser.open("https://ollama.com/download")
            except Exception:
                pass

        def retry_question():
            input_field.value = last_question
            state.page.update()
            send_clicked()

        # Each entry: (label, icon, callable, primary?)
        # `primary=True` renders as a filled accent button so the recovery
        # path is visually obvious even at a glance.
        if kind == "load_failure":
            out.append(("Reload model", ft.Icons.CLOUD_DOWNLOAD,
                        repair_model, True))
            out.append(("Free all VRAM", ft.Icons.MEMORY, free_vram, False))
            out.append(("Update Ollama", ft.Icons.OPEN_IN_NEW,
                        open_ollama_download, False))
            out.append(("Retry question", ft.Icons.REFRESH,
                        retry_question, False))
        elif kind == "oom":
            out.append(("Free all VRAM", ft.Icons.MEMORY, free_vram, True))
            out.append(("Retry question", ft.Icons.REFRESH,
                        retry_question, False))
        elif kind == "context_overflow":
            out.append(("Retry question", ft.Icons.REFRESH,
                        retry_question, True))
        elif kind == "model_not_found":
            out.append(("Pull this model", ft.Icons.CLOUD_DOWNLOAD,
                        repair_model, True))
        elif kind == "server_down":
            out.append(("Retry question", ft.Icons.REFRESH,
                        retry_question, True))
            out.append(("Open Ollama download", ft.Icons.OPEN_IN_NEW,
                        open_ollama_download, False))
        else:
            out.append(("Retry question", ft.Icons.REFRESH,
                        retry_question, True))

        return out

    def _build_chips(turn: ChatTurn) -> ft.Row:
        return ft.Row(
            [citation_chip(i + 1, h, open_source)
             for i, h in enumerate(turn.citations)],
            wrap=True, spacing=6, run_spacing=6,
        )

    def _bubble(turn: ChatTurn) -> ft.Control:
        is_user = turn.role == "user"

        md = ft.Markdown(
            turn.text or ("…" if turn.streaming else ""),
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
            code_theme="atom-one-dark",
        )
        turn.md_ctrl = md

        # Optional "Reasoning" panel for deepseek-r1 / qwen3-reasoner / etc.
        # Always created for assistant turns; hidden until thinking arrives.
        thinking_md = ft.Markdown(
            (turn.thinking or "").strip() or "*(reasoning…)*",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
        )
        thinking_header = ft.Text(
            "Reasoning…", size=11,
            color=ON_SURFACE_DIM, weight=ft.FontWeight.W_700,
        )
        thinking_box = ft.Container(
            visible=(not is_user) and bool(turn.thinking or turn.streaming),
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            bgcolor="#13151E",
            border=ft.border.all(1, "#1E2130"),
            border_radius=8,
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.PSYCHOLOGY_OUTLINED, size=14,
                            color=ON_SURFACE_DIM),
                    thinking_header,
                ], spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(
                    content=thinking_md,
                    padding=ft.padding.only(top=4, left=18),
                ),
            ], spacing=2, tight=True),
        )
        turn.thinking_md = thinking_md
        turn.thinking_header = thinking_header
        turn.thinking_box = thinking_box

        chips_row = _build_chips(turn) if turn.citations and not is_user \
            else ft.Row([], visible=False)
        turn.chips_row = chips_row

        # Inline recovery actions (e.g. "Reload model" on a load failure).
        # Each entry is (label, icon, callable, primary?). The callable
        # runs on click. Primary actions render as a filled accent button
        # so the recommended recovery path is impossible to miss.
        actions_row = ft.Row([], visible=False, wrap=True, spacing=8)
        if turn.actions and not is_user:
            actions_row.visible = True
            for entry in turn.actions:
                # Backward-compat: tolerate 3-tuples that older test paths
                # may have constructed.
                if len(entry) == 4:
                    label, icon, cb, primary = entry
                else:
                    label, icon, cb = entry
                    primary = False
                handler = (lambda fn=cb: lambda _: fn())(cb)
                if primary:
                    btn = ft.FilledButton(
                        label, icon=icon, on_click=handler,
                        bgcolor=ACCENT, color="#FFFFFF",
                    )
                else:
                    btn = ft.FilledTonalButton(
                        label, icon=icon, on_click=handler,
                    )
                actions_row.controls.append(btn)

        bubble_items = []
        if not is_user:
            bubble_items.append(thinking_box)
        bubble_items.append(md)
        bubble_items.append(chips_row)
        if actions_row.visible:
            bubble_items.append(actions_row)

        bubble_col = ft.Column(bubble_items, spacing=10, tight=True)
        bubble = ft.Container(
            content=bubble_col,
            padding=ft.padding.symmetric(horizontal=16, vertical=12),
            bgcolor=USER_BUBBLE if is_user else ASSIST_BUBBLE,
            border_radius=14,
            border=ft.border.all(1, "#262938"),
        )
        avatar = ft.Container(
            width=28, height=28, border_radius=999,
            bgcolor=ACCENT_SOFT if is_user else SURFACE_DARK_HI,
            alignment=ft.Alignment.CENTER,
            content=ft.Icon(
                ft.Icons.PERSON if is_user else ft.Icons.AUTO_AWESOME,
                size=16, color="#FFFFFF" if is_user else ACCENT,
            ),
        )
        outer = ft.Row(
            [avatar, ft.Container(content=bubble, expand=True)],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
        )
        turn.bubble = outer
        return outer

    def _scroll_to_bottom(duration: int = 120, *, force: bool = False):
        """Best-effort jump to the latest message.

        Flet's ListView.auto_scroll handles new appends but not full rebuilds
        or in-place updates of existing children, so we trigger explicitly.

        When the user has scrolled up (chat_follow_state["follow"] is False),
        we skip the scroll so they can read backscroll without being yanked
        down on every streamed token. `force=True` overrides this — used
        after submitting a new question, where we always want to land at
        the bottom.

        From Flet 0.84 worker-thread context, scroll_to alone doesn't
        always paint without an explicit update on the ListView; we call
        both for reliability.
        """
        if not force and not chat_follow_state.get("follow", True):
            return
        try:
            chat_list.scroll_to(offset=-1, duration=duration)
        except Exception:
            pass
        try:
            chat_list.update()
        except Exception:
            pass

    def render_chat(*, force_scroll: bool = False):
        """Full re-render. Call when turns are added/removed."""
        chat_list.controls.clear()
        if not state.turns:
            chat_list.controls.append(_chat_welcome(state))
        else:
            for t in state.turns:
                chat_list.controls.append(_bubble(t))
        state.page.update()
        _scroll_to_bottom(force=force_scroll)

    def update_streaming_assistant(turn: ChatTurn):
        """In-place update for the streaming bubble. Avoids full rebuilds."""
        if turn.md_ctrl is None:
            render_chat()
            return

        # Reasoning panel
        if turn.thinking_box is not None:
            has_thinking = bool(turn.thinking)
            turn.thinking_box.visible = has_thinking or (
                turn.streaming and not turn.text
            )
            if has_thinking and turn.thinking_md is not None:
                # Show last ~1200 chars while streaming so it's readable, full
                # text once done.
                text = turn.thinking
                if turn.streaming and len(text) > 1200:
                    text = "…" + text[-1200:]
                turn.thinking_md.value = text or "*(reasoning…)*"
            if turn.thinking_header is not None:
                if turn.streaming and not turn.text:
                    turn.thinking_header.value = (
                        f"Reasoning… ({len(turn.thinking):,} chars)"
                    )
                elif turn.thinking:
                    turn.thinking_header.value = (
                        f"Reasoning · {len(turn.thinking):,} chars"
                    )
            try:
                turn.thinking_box.update()
            except Exception:
                pass

        # Main answer
        turn.md_ctrl.value = turn.text or (
            "" if turn.streaming and turn.thinking else (
                "…" if turn.streaming else ""
            )
        )
        try:
            turn.md_ctrl.update()
        except Exception:
            state.page.update()

        # Auto-scroll the chat list as the bubble grows.
        _scroll_to_bottom(duration=80)

    # ------------- send / stream ------------------------------------------

    def set_busy(busy: bool):
        state.streaming = busy
        send_btn.visible = not busy
        stop_btn.visible = busy
        input_field.disabled = busy
        state.page.update()

    def stop_clicked(_):
        state.stop_flag = True

    stop_btn.on_click = stop_clicked

    def send_clicked(_=None):
        if state.streaming:
            return
        if state.ws is None:
            _toast(state.page, "Open a workspace first")
            return
        text = (input_field.value or "").strip()
        if not text:
            return
        input_field.value = ""
        state.turns.append(ChatTurn(role="user", text=text))
        assistant = ChatTurn(role="assistant", text="", streaming=True)
        state.turns.append(assistant)
        # Refresh the workflow chip strip so it reflects current cfg
        # (in case Settings were changed since the chat view was built).
        render_chat_workflow_chips()
        # Snap back to follow mode whenever the user sends a new
        # question — they always want to see their own message + the
        # incoming reply, even if they had scrolled up reading earlier.
        chat_follow_state["follow"] = True
        render_chat(force_scroll=True)
        set_busy(True)

        def worker():
            try:
                # Apply prefix/suffix/negatives if the per-query toggle is on
                effective_q = apply_query_modifiers(
                    text, state.cfg,
                    workspace_root=(state.ws.root if state.ws else None),
                )

                # Honor the "Use RAG" toggle. When OFF we skip embedding +
                # retrieval entirely, sending the question straight to the LLM.
                if state.cfg.use_rag:
                    embedder = make_embedder(state.cfg)
                    idx = Index(state.ws.meta_db_path, embed_dim=embedder.dim)

                    def agent_status(msg):
                        # Surface agent steps in the streaming bubble while we
                        # wait for retrieval to complete.
                        assistant.text = f"_{msg}…_"
                        update_streaming_assistant(assistant)
                        # Best-effort match agent step → chip id.
                        s = (msg or "").lower()
                        if "retriev" in s or "search" in s:
                            chat_set_active_stage("hybrid_search")
                        elif "rerank" in s:
                            chat_set_active_stage("rerank")
                        elif "reflect" in s or "agent" in s:
                            chat_set_active_stage("query_expand")

                    def retrieval_status(stage_id: str):
                        # Mirror smart_retrieve's stage events to the
                        # chat workflow chip strip.
                        chat_set_active_stage(
                            None if stage_id == "done" else stage_id
                        )

                    if state.cfg.agentic:
                        chat_set_active_stage("query_expand")
                        hits = agentic_retrieve(
                            query=effective_q, embedder=embedder, index=idx,
                            cfg=state.cfg, status_cb=agent_status,
                        )
                        # Clear the temporary status text — chat_answer will
                        # populate the real reply.
                        assistant.text = ""
                        update_streaming_assistant(assistant)
                    else:
                        hits = smart_retrieve(
                            query=effective_q, embedder=embedder,
                            index=idx, cfg=state.cfg,
                            status_cb=retrieval_status,
                        )
                else:
                    hits = []
                assistant.citations = hits
                backend = detect_backend(state.cfg)

                # Build the conversation BEFORE the latest user turn we just appended.
                # state.turns ends with [..., user(text), assistant(empty)]; drop both.
                history = []
                for t in state.turns[:-2]:
                    history.append({"role": t.role, "content": t.text})

                if backend == "none":
                    if not hits:
                        assistant.text = (
                            "_No LLM detected and no matching content in this corpus._\n\n"
                            "Install Ollama and pull a model "
                            "(`ollama pull qwen2.5:3b`) to chat."
                        )
                    else:
                        ans = chat_answer(
                            history=history, latest_question=effective_q,
                            hits=hits, cfg=state.cfg, stream=False,
                            workspace_root=(state.ws.root if state.ws
                                            else None),
                        )
                        assistant.text = ans.text  # type: ignore
                    update_streaming_assistant(assistant)
                else:
                    chat_set_active_stage("generate")
                    state.stop_flag = False
                    last_render = 0.0
                    for kind, piece in chat_answer(
                        history=history, latest_question=effective_q,
                        hits=hits, cfg=state.cfg, stream=True,
                        workspace_root=(state.ws.root if state.ws
                                         else None),
                    ):  # type: ignore
                        if state.stop_flag:
                            assistant.text += "\n\n_[stopped]_"
                            break
                        if kind == "thinking":
                            assistant.thinking += piece
                        else:
                            assistant.text += piece
                        now = time.monotonic()
                        if now - last_render > 0.06:
                            update_streaming_assistant(assistant)
                            last_render = now
                    update_streaming_assistant(assistant)
                assistant.streaming = False
                chat_clear_pulse()
                # Final rebuild to splice in citation chips below the answer.
                render_chat()
            except Exception as ex:
                # Render the friendly OllamaChatError text as plain markdown
                # rather than wrapping it in italics — it's already
                # multi-line guidance the user needs to read.
                from ez_rag.generate import OllamaChatError
                from ez_rag.retrieve import EmbedderMismatchError
                if isinstance(ex, OllamaChatError):
                    assistant.text = (
                        "**Couldn't get a reply from the LLM.**\n\n"
                        f"{ex}"
                    )
                    assistant.actions = _actions_for_ollama_error(
                        ex, last_question=text,
                    )
                elif isinstance(ex, EmbedderMismatchError):
                    assistant.text = (
                        "**Embedder mismatch — your index is unusable with "
                        "the current embedder setting.**\n\n"
                        f"{ex}"
                    )
                    assistant.actions = _actions_for_embedder_mismatch(
                        last_question=text,
                    )
                else:
                    assistant.text = f"_Error: {ex}_"
                assistant.streaming = False
                render_chat()
            finally:
                chat_clear_pulse()
                set_busy(False)
                refresh_status()

        state.page.run_thread(worker)

    send_btn.on_click = send_clicked

    # Ctrl+Enter to send.
    def on_input_submit(_):
        send_clicked()
    input_field.on_submit = on_input_submit

    def clear_chat(_):
        state.turns.clear()
        render_chat()

    def _test_current_model():
        """Quick load-and-generate test against the configured chat LLM.

        Surfaces the exact Ollama error verbatim — useful when a model
        like qwen3.6:35b is misbehaving and the chat path's wrapped
        OllamaChatError translation is hiding the underlying detail.
        Posts a fresh assistant turn with the result so it's visible in
        chat history alongside any previous errors.
        """
        if state.ws is None:
            _toast(state.page, "Open a workspace first.")
            return
        cfg = state.cfg
        model = cfg.llm_model or "(unset)"
        # Build a placeholder turn that we'll fill in
        diag = ChatTurn(role="assistant",
                        text=f"_Testing **{model}**…_",
                        streaming=True)
        state.turns.append(diag)
        render_chat()

        def _bg():
            import httpx
            url = cfg.llm_url.rstrip("/")
            t0 = time.monotonic()
            verdict = []
            verdict.append(f"## Diagnostic: `{model}`\n")
            verdict.append(f"- backend URL: `{url}`")
            # 1. Is Ollama reachable?
            try:
                r = httpx.get(f"{url}/api/tags", timeout=3.0)
                r.raise_for_status()
                tags = [m.get("name", "") for m in r.json().get("models", [])]
                verdict.append(f"- /api/tags: **OK** ({len(tags)} model(s) installed)")
                pulled = model in tags
                verdict.append(
                    f"- model installed locally: "
                    + ("**yes**" if pulled else "**no** — run *Reload model* to pull it")
                )
            except Exception as e:
                verdict.append(f"- /api/tags: **FAIL** — {type(e).__name__}: {e}")
                diag.text = "\n".join(verdict) + (
                    "\n\nOllama isn't reachable. Make sure the Ollama app is "
                    "running, or check that `OLLAMA_HOST` matches the URL above."
                )
                diag.streaming = False
                render_chat()
                return

            # 2. Try a tiny generate — this is what fails for partial-blob
            # / corrupt-model cases.
            verdict.append("- attempting 1-token generate…")
            try:
                t1 = time.monotonic()
                r = httpx.post(
                    f"{url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "ping",
                        "stream": False,
                        "options": {"num_predict": 1, "temperature": 0.0},
                    },
                    timeout=300.0,
                )
                load_s = time.monotonic() - t1
                if r.status_code >= 400:
                    body = r.text[:600]
                    verdict.append(f"- generate: **FAIL** (HTTP {r.status_code})")
                    verdict.append("\n```\n" + body.strip() + "\n```")
                    # Tell the user which actions to try
                    low = body.lower()
                    if "unable to load model" in low:
                        verdict.append(
                            "\n**Likely fix**: corrupt blob. Click "
                            "**Reload model** above to `ollama rm` + "
                            "`ollama pull` this tag.")
                    elif "out of memory" in low or "cuda" in low:
                        verdict.append(
                            "\n**Likely fix**: VRAM exhausted. Click "
                            "**Free all VRAM** (Reload model dialog) or "
                            "pick a smaller model in Settings.")
                    elif "404" in low or "not found" in low:
                        verdict.append(
                            f"\n**Likely fix**: model not pulled. Run "
                            f"`ollama pull {model}` or use Reload model.")
                else:
                    elapsed = time.monotonic() - t0
                    data = r.json()
                    eval_count = int(data.get("eval_count") or 0)
                    eval_dur = (data.get("eval_duration") or 0) / 1e9
                    tok_per_s = (eval_count / eval_dur) if eval_dur > 0 else 0
                    verdict.append(
                        f"- generate: **OK** "
                        f"(load {load_s:.2f}s · {tok_per_s:.0f} tok/s · "
                        f"{elapsed:.2f}s total)"
                    )
                    verdict.append(
                        "\n**Model is working.** If chat still fails, the "
                        "issue is prompt-size / context-overflow rather "
                        "than a load problem — try lowering Top-K or "
                        "disabling Expand-to-chapter in Settings."
                    )
            except Exception as e:
                verdict.append(f"- generate: **FAIL** ({type(e).__name__}) — {e}")
            diag.text = "\n".join(verdict)
            diag.streaming = False
            render_chat()

        state.page.run_thread(_bg)

    def _free_vram():
        """Evict every model Ollama currently has resident in VRAM.

        Surfaces what got unloaded as a toast so the user can confirm.
        Handles the 'stuck model' case by also offering a hard taskkill
        of ollama.exe when the soft unload returns nothing (or the user
        still suspects it's wedged).
        """
        if state.ws is None:
            _toast(state.page, "Open a workspace first.")
            return
        url = state.cfg.llm_url

        def _do():
            # First check what's actually loaded so we can show the user
            # what we're about to do.
            try:
                resident = list_running_models(url)
            except Exception:
                resident = []
            if not resident:
                _toast(state.page,
                    "No models currently resident in VRAM (per /api/ps). "
                    "If you still suspect Ollama is stuck, click 'Reload "
                    "model' to force a clean rm + pull, or kill ollama.exe "
                    "from Task Manager / `taskkill /F /IM ollama.exe`.")
                return

            try:
                unloaded = unload_running_models(url)
            except Exception as ex:
                _toast(state.page, f"Unload failed: {ex}")
                return
            if unloaded:
                _toast(state.page,
                    f"Freed VRAM: unloaded {', '.join(unloaded)}. "
                    "Next chat will reload on demand.")
            else:
                _toast(state.page,
                    f"Ollama said {len(resident)} model(s) are resident "
                    "but didn't unload them. Try 'Reload model' or kill "
                    "ollama.exe from Task Manager.")

        state.page.run_thread(_do)

    def _reload_current_model():
        """Proactive 'reload model' — same flow the error bubble uses, but
        available before any question fails. Frees VRAM, removes the
        on-disk blob, and opens the pull dialog with the current tag pre-
        filled so one click starts the redownload.
        """
        if state.ws is None:
            _toast(state.page, "Open a workspace first.")
            return
        cfg = state.cfg
        if not cfg.llm_model:
            _toast(state.page, "No LLM model set in Settings.")
            return

        # Show a confirmation overlay first since this re-downloads ~ multi-GB.
        confirm_overlay = ft.Container(
            expand=True, visible=True,
            bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
            alignment=ft.Alignment.CENTER,
        )

        def _close_confirm(_=None):
            confirm_overlay.visible = False
            try:
                if confirm_overlay in state.page.overlay:
                    state.page.overlay.remove(confirm_overlay)
            except Exception:
                pass
            state.page.update()

        def _do_reload(_=None):
            _close_confirm()
            tag = cfg.llm_model

            def _bg():
                # 1. Evict everything currently loaded so the pull-and-load
                #    has a clean slate.
                try:
                    unloaded = unload_running_models(cfg.llm_url)
                    if unloaded:
                        _toast(state.page,
                               f"Freed VRAM: {', '.join(unloaded)}")
                except Exception:
                    pass
                # 2. Delete the on-disk blob so the pull does a fresh download
                #    (this is the key step for the 'unable to load model' bug).
                try:
                    delete_ollama_model(cfg.llm_url, tag)
                except Exception:
                    pass

            state.page.run_thread(_bg)

            # 3. Open the pull dialog with the tag pre-filled so the user
            #    just has to click Pull and watch the progress bar.
            opener = open_pull_dialog_cb.get("fn")
            if opener is None:
                _toast(state.page,
                       "Pull dialog not ready yet — try Settings → "
                       "Browse Ollama library.")
                return
            try:
                opener("llm", None, prefill_tag=tag)
            except Exception as ex:
                _toast(state.page, f"Couldn't open pull dialog: {ex}")

        confirm_overlay.content = ft.Container(
            width=480, padding=20,
            bgcolor=SURFACE_DARK,
            border=ft.border.all(1, "#262938"),
            border_radius=14,
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.AUTORENEW, color=ACCENT, size=22),
                    ft.Text("Reload model", size=16,
                            weight=ft.FontWeight.W_700, color=ON_SURFACE),
                ], spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Text(
                    f"This will:\n"
                    f"  1. Unload every model currently in VRAM\n"
                    f"  2. Delete '{cfg.llm_model}' from disk\n"
                    f"  3. Re-download it from ollama.com\n\n"
                    "Use this when the model fails to load or behaves "
                    "oddly. The on-disk blob is multi-GB, so the download "
                    "takes a while.",
                    size=12, color=ON_SURFACE_DIM,
                ),
                ft.Row([
                    ft.Container(expand=True),
                    ft.TextButton("Cancel", on_click=_close_confirm),
                    ft.FilledButton(
                        "Reload model",
                        icon=ft.Icons.AUTORENEW,
                        bgcolor=ACCENT, color="#FFFFFF",
                        on_click=_do_reload,
                    ),
                ], spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=14, tight=True),
        )
        state.page.overlay.append(confirm_overlay)
        state.page.update()

    composer = ft.Container(
        padding=ft.padding.only(left=20, right=20, top=10, bottom=16),
        content=ft.Column([
            ft.Container(
                bgcolor=SURFACE_DARK,
                border=ft.border.all(1, "#262938"),
                border_radius=14,
                padding=ft.padding.only(left=14, right=8, top=4, bottom=4),
                content=ft.Row(
                    [
                        input_field,
                        ft.Container(
                            content=ft.Row([stop_btn, send_btn], spacing=4),
                            padding=ft.padding.only(top=4, bottom=4),
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.END,
                ),
            ),
            ft.Row([
                ft.Container(expand=True),
                modifier_check,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=4, tight=True),
    )

    chat_toolbar = ft.Container(
        padding=ft.padding.symmetric(horizontal=20, vertical=10),
        content=ft.Row(
            [
                ft.Text("Chat", size=18, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE),
                ft.Container(expand=True),
                ft.Container(
                    padding=ft.padding.symmetric(horizontal=10, vertical=6),
                    bgcolor=SURFACE_DARK,
                    border=ft.border.all(1, "#262938"),
                    border_radius=999,
                    tooltip=TIP["rag_toggle"],
                    content=ft.Row([
                        ft.Icon(ft.Icons.STORAGE_OUTLINED, size=14,
                                color=ON_SURFACE_DIM),
                        rag_label,
                        rag_switch,
                    ], spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    tight=True),
                ),
                ft.TextButton(
                    "Test model",
                    icon=ft.Icons.HEALTH_AND_SAFETY,
                    on_click=lambda _: _test_current_model(),
                    tooltip=TIP["test_model_btn"],
                ),
                ft.TextButton(
                    "Free VRAM",
                    icon=ft.Icons.MEMORY,
                    on_click=lambda _: _free_vram(),
                    tooltip=TIP["free_vram_btn"],
                ),
                ft.TextButton(
                    "Reload model",
                    icon=ft.Icons.AUTORENEW,
                    on_click=lambda _: _reload_current_model(),
                    tooltip=TIP["reload_model_btn"],
                ),
                ft.TextButton(
                    "Clear", icon=ft.Icons.RESTART_ALT,
                    on_click=clear_chat,
                    tooltip=TIP["clear_btn"],
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        ),
    )

    # ---------- Query / answer workflow chip strip ----------
    # Mirror of the ingest pipeline strip — shows which retrieval +
    # generation stages are active for the current cfg, hovers explain
    # what each does, and the currently-running stage pulses while a
    # query is in flight. Driven by status_cb events from smart_retrieve
    # plus a "generate" stage during streaming.
    chat_workflow_chips_by_id: dict[str, ft.Container] = {}
    chat_workflow_state = {"active_stage": None, "pulse_phase": False}

    def _chat_make_chip(stage_id, label, active, *, kind="auto", tooltip=""):
        if kind == "auto":
            fg = ACCENT
            bg = ft.Colors.with_opacity(0.10, ACCENT)
            border_color = ft.Colors.with_opacity(0.45, ACCENT)
            icon = ft.Icons.CHECK_CIRCLE
        elif active:
            fg = "#FFFFFF"
            bg = ACCENT
            border_color = ACCENT
            icon = ft.Icons.CHECK_CIRCLE
        else:
            fg = ON_SURFACE_DIM
            bg = "transparent"
            border_color = "#2A2E3F"
            icon = ft.Icons.RADIO_BUTTON_UNCHECKED
        chip = ft.Container(
            content=ft.Row([
                ft.Icon(icon, size=12, color=fg),
                ft.Text(label, size=10, color=fg,
                         weight=ft.FontWeight.W_700),
            ], spacing=4, tight=True,
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=bg,
            border=ft.border.all(1, border_color),
            border_radius=12,
            padding=ft.padding.symmetric(horizontal=8, vertical=3),
            tooltip=tooltip,
            opacity=1.0,
            animate_opacity=ft.Animation(450, ft.AnimationCurve.EASE_IN_OUT),
            data={"id": stage_id, "active": active, "kind": kind},
        )
        chat_workflow_chips_by_id[stage_id] = chip
        return chip

    def _chat_arrow():
        return ft.Icon(ft.Icons.ARROW_FORWARD, size=12, color="#3D4156")

    chat_workflow_row = ft.Row(
        [], spacing=4, wrap=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    chat_workflow_label = ft.Text(
        "QUERY PIPELINE", size=10, color=ON_SURFACE_DIM,
        weight=ft.FontWeight.W_700,
    )
    chat_workflow_legend = ft.Text(
        "Hover a stage for details. The active stage pulses while answering.",
        size=10, color=ON_SURFACE_DIM, italic=True,
    )
    chat_workflow_card = ft.Container(
        padding=ft.padding.symmetric(horizontal=10, vertical=8),
        margin=ft.margin.symmetric(horizontal=20, vertical=4),
        bgcolor="#13151E",
        border=ft.border.all(1, "#1E2130"),
        border_radius=8,
        content=ft.Column(
            [chat_workflow_label, chat_workflow_row, chat_workflow_legend],
            spacing=6, tight=True),
    )

    CHAT_STAGE_INFO = {
        "query_expand": (
            "Optional. When enabled (auto-list mode, HyDE, or multi-query), "
            "the LLM rewrites or expands your question before search. "
            "For 'list X / give examples' queries, generates an entity-rich "
            "hypothetical passage that retrieves stat-blocks and tables "
            "instead of explanatory prose. Costs one extra LLM call."
        ),
        "use_rag": (
            "Always on while RAG is enabled. Toggle it off in the toolbar "
            "to ask the LLM directly with no corpus retrieval."
        ),
        "hybrid_search": (
            "Always on. BM25 keyword search + dense vector search run in "
            "parallel and are fused via Reciprocal Rank Fusion. The single "
            "biggest quality lift over either method alone."
        ),
        "rerank": (
            "Optional. Cross-encoder model re-scores the top candidates "
            "with full attention to (query, passage) pairs — catches "
            "relevance that embedding similarity misses. Highest-ROI "
            "retrieval improvement per ~50–200 ms latency."
        ),
        "diversify": (
            "Optional. Caps the number of chunks returned from any single "
            "source file (default 3) so the LLM sees varied evidence "
            "instead of letting one PDF dominate top-K. Forces grounding "
            "across multiple sources."
        ),
        "expand": (
            "Optional. After top-K is selected, expand each hit's text — "
            "either to its full chapter (capped by chapter_max_chars) or "
            "with ±N neighbor chunks. Bigger context but risks dilution."
        ),
        "reorder": (
            "Optional. Reorders retrieved chunks so the most-relevant ones "
            "are at the START and END of the prompt — combats the "
            "well-documented 'lost in the middle' effect where LLMs "
            "ignore content in the middle of long contexts. Free quality "
            "lift, no extra calls."
        ),
        "generate": (
            "Always on. The LLM produces the final answer using the "
            "retrieved chunks as context. For list-style queries the "
            "system prompt switches to extraction mode to force a clean "
            "list of named items with citations."
        ),
    }

    def render_chat_workflow_chips():
        chat_workflow_chips_by_id.clear()
        c = state.cfg or Config()

        def tip(stage_id: str, on: bool, *, always_on: bool = False) -> str:
            if always_on:
                head = "ON (always)"
            else:
                head = "ON" if on else "OFF"
            return f"{head} · {CHAT_STAGE_INFO[stage_id]}"

        chips: list[ft.Control] = []
        # Query expansion (HyDE / multi-query / auto-list)
        qe_on = (
            bool(getattr(c, "use_hyde", False))
            or bool(getattr(c, "multi_query", False))
            or bool(getattr(c, "auto_list_mode", True))
        )
        chips.append(_chat_make_chip(
            "query_expand", "query expand", qe_on, kind="opt",
            tooltip=tip("query_expand", qe_on)))
        chips.append(_chat_arrow())
        # Hybrid search (always on)
        chips.append(_chat_make_chip(
            "hybrid_search", "hybrid search", True, kind="auto",
            tooltip=tip("hybrid_search", True, always_on=True)))
        # Rerank (opt)
        chips.append(_chat_arrow())
        rr_on = bool(getattr(c, "rerank", True))
        chips.append(_chat_make_chip(
            "rerank", "rerank", rr_on, kind="opt",
            tooltip=tip("rerank", rr_on)))
        # Diversify (opt)
        chips.append(_chat_arrow())
        div_on = int(getattr(c, "diversify_per_source", 3) or 0) > 0
        chips.append(_chat_make_chip(
            "diversify", "diversify", div_on, kind="opt",
            tooltip=tip("diversify", div_on)))
        # Expand chapter / neighbor (opt)
        chips.append(_chat_arrow())
        exp_on = (bool(getattr(c, "expand_to_chapter", False))
                   or int(getattr(c, "context_window", 0) or 0) > 0)
        chips.append(_chat_make_chip(
            "expand", "expand", exp_on, kind="opt",
            tooltip=tip("expand", exp_on)))
        # Reorder (opt)
        chips.append(_chat_arrow())
        ro_on = bool(getattr(c, "reorder_for_attention", True))
        chips.append(_chat_make_chip(
            "reorder", "reorder", ro_on, kind="opt",
            tooltip=tip("reorder", ro_on)))
        # Generate (always on)
        chips.append(_chat_arrow())
        chips.append(_chat_make_chip(
            "generate", "generate", True, kind="auto",
            tooltip=tip("generate", True, always_on=True)))
        chat_workflow_row.controls = chips

    render_chat_workflow_chips()

    def chat_set_active_stage(stage_id: str | None):
        """Light up a chip and dim the others. None = clear all pulses."""
        prev = chat_workflow_state["active_stage"]
        if prev and prev in chat_workflow_chips_by_id:
            chat_workflow_chips_by_id[prev].opacity = 1.0
        chat_workflow_state["active_stage"] = stage_id
        if stage_id and stage_id in chat_workflow_chips_by_id:
            chat_workflow_chips_by_id[stage_id].opacity = 0.45
        try:
            state.page.update()
        except Exception:
            pass

    def chat_clear_pulse():
        for chip in chat_workflow_chips_by_id.values():
            chip.opacity = 1.0
        chat_workflow_state["active_stage"] = None
        try:
            state.page.update()
        except Exception:
            pass

    container = ft.Container(
        bgcolor=BG_DARK,
        expand=True,
        content=ft.Column(
            [chat_toolbar, chat_workflow_card,
             ft.Divider(height=1, color="#1E2130"),
             chat_list, composer],
            expand=True, spacing=0,
        ),
    )

    def _chat_welcome(state: AppState) -> ft.Control:
        if state.ws is None:
            return empty_state(
                icon=ft.Icons.AUTO_AWESOME,
                title="Welcome to ez-rag",
                subtitle="Open or create a workspace to start chatting with your documents.",
                actions=[
                    ft.FilledButton(
                        "Open workspace", icon=ft.Icons.FOLDER_OPEN,
                        on_click=on_open_workspace,
                        bgcolor=ACCENT, color="#FFFFFF",
                    ),
                ],
            )
        s = read_stats(state.ws.meta_db_path)
        if s is None or s["files"] == 0:
            return empty_state(
                icon=ft.Icons.UPLOAD_FILE,
                title="Add some documents",
                subtitle=f"Drop files into  {state.ws.docs_dir}\nor use the Files tab to add them, then run Ingest.",
                actions=[
                    ft.FilledButton(
                        "Open Files", icon=ft.Icons.FOLDER,
                        on_click=lambda _: on_open_files(),
                        bgcolor=ACCENT, color="#FFFFFF",
                    ),
                ],
            )
        backend = detect_backend(state.cfg)
        ws_key = str(state.ws.root)
        cached = state.suggestions.get(ws_key)  # None | [] | list[str]

        def fill(text: str):
            input_field.value = text
            state.page.update()
            try:
                input_field.focus()
            except Exception:
                pass

        # ----- LLM-generated suggestions, opt-in & cached per workspace -----
        suggestion_status = ft.Text(
            "" if cached is None else (
                "(no suggestions returned — try again)" if cached == [] else ""
            ),
            size=11, color=ON_SURFACE_DIM, italic=True,
        )
        suggestions_col = ft.Column(
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        def _render_suggestions(items: list[str]):
            suggestions_col.controls.clear()
            for q in items:
                suggestions_col.controls.append(
                    ft.Container(
                        content=ft.Text(q, size=13, color=ON_SURFACE),
                        bgcolor=SURFACE_DARK,
                        border=ft.border.all(1, "#262938"),
                        border_radius=10,
                        padding=ft.padding.symmetric(horizontal=14, vertical=10),
                        on_click=(lambda q=q: lambda _: fill(q))(q),
                        width=560,
                        ink=True,
                    )
                )

        if cached:
            _render_suggestions(cached)

        def _request_suggestions(_=None):
            if state.suggestions_loading:
                return
            if backend == "none":
                suggestion_status.value = (
                    "No LLM detected — install Ollama and pull a model "
                    "to use suggestions."
                )
                state.page.update()
                return
            state.suggestions_loading = True
            suggestion_status.value = "Asking the model for ideas…"
            suggest_btn.disabled = True
            state.page.update()

            def _worker():
                items: list[str] = []
                try:
                    # Sample diverse-ish chunks from the index.
                    import random
                    from ez_rag.generate import generate_question_suggestions
                    conn = sqlite3.connect(str(state.ws.meta_db_path))
                    try:
                        rows = conn.execute(
                            "SELECT text FROM chunks "
                            "WHERE LENGTH(text) > 80 "
                            "ORDER BY RANDOM() LIMIT 12"
                        ).fetchall()
                    finally:
                        conn.close()
                    excerpts = [r[0] for r in rows if r[0]]
                    items = generate_question_suggestions(
                        excerpts, state.cfg, n=3,
                    )
                except Exception as ex:
                    suggestion_status.value = f"Suggestion error: {ex}"
                finally:
                    state.suggestions[ws_key] = items
                    state.suggestions_loading = False
                    suggest_btn.disabled = False
                    if items:
                        _render_suggestions(items)
                        suggestion_status.value = ""
                    elif suggestion_status.value.startswith("Asking"):
                        suggestion_status.value = (
                            "Model didn't return usable suggestions — "
                            "click Suggest again."
                        )
                    state.page.update()

            state.page.run_thread(_worker)

        suggest_btn = ft.OutlinedButton(
            "Suggest questions" if cached is None else "Refresh suggestions",
            icon=ft.Icons.LIGHTBULB_OUTLINE,
            on_click=_request_suggestions,
            tooltip=("Ask the LLM to propose 3 specific questions you could "
                     "ask about your corpus, based on a random sample of "
                     "indexed text."),
        )

        return ft.Container(
            expand=True,
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.AUTO_AWESOME, size=48, color=ACCENT),
                    ft.Container(height=10),
                    ft.Text(f"Ready · {s['files']} files · {s['chunks']} chunks",
                            size=20, weight=ft.FontWeight.W_700,
                            color=ON_SURFACE),
                    ft.Text(
                        f"Backend: {backend if backend != 'none' else 'retrieval-only (no LLM detected)'}",
                        size=12, color=ON_SURFACE_DIM,
                    ),
                    ft.Container(height=14),
                    ft.Text(
                        "Ask anything about your documents.",
                        size=13, color=ON_SURFACE_DIM,
                    ),
                    ft.Container(height=14),
                    suggest_btn,
                    ft.Container(height=8),
                    suggestion_status,
                    suggestions_col,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
        )

    return container, render_chat, input_field


# ============================================================================
# Files view
# ============================================================================

def build_files_view(state: AppState, *, refresh_status, refresh_files_cb):
    pick_files_picker = ft.FilePicker()
    pick_dir_picker = ft.FilePicker()
    state.page.services.append(pick_files_picker)
    state.page.services.append(pick_dir_picker)

    file_rows = ft.Column([], spacing=4, expand=True, scroll=ft.ScrollMode.AUTO)
    summary = ft.Text("", size=12, color=ON_SURFACE_DIM)
    ingest_progress = ft.ProgressBar(
        value=0, color=ACCENT, bgcolor="#262938", visible=False,
    )

    # Workflow chip strip — visual map of which ingest stages are active
    # for the current config. Each chip lights up (accent fill) when its
    # gate is on, dims when off, and pulses when that stage is the
    # currently-running one. Re-rendered after Settings save and when
    # ingest starts.
    #
    # Each chip is a Container with `data` = {"id": stage_id, "active": bool}
    # so the watchdog can find it later and toggle opacity for the pulse.
    workflow_chips_by_id: dict[str, ft.Container] = {}

    def _make_chip(stage_id, label, active, *, kind="auto", tooltip=""):
        # kind="auto" → always-on stages get a subdued check + ACCENT
        #               outline so they read as "always runs"
        # kind="opt"  → user-toggleable stages get full ACCENT fill when
        #               on, dim outline when off
        if kind == "auto":
            fg = ACCENT
            bg = ft.Colors.with_opacity(0.10, ACCENT)
            border_color = ft.Colors.with_opacity(0.45, ACCENT)
            icon = ft.Icons.CHECK_CIRCLE
        elif active:
            fg = "#FFFFFF"
            bg = ACCENT
            border_color = ACCENT
            icon = ft.Icons.CHECK_CIRCLE
        else:
            fg = ON_SURFACE_DIM
            bg = "transparent"
            border_color = "#2A2E3F"
            icon = ft.Icons.RADIO_BUTTON_UNCHECKED
        chip = ft.Container(
            content=ft.Row([
                ft.Icon(icon, size=12, color=fg),
                ft.Text(label, size=10, color=fg,
                         weight=ft.FontWeight.W_700),
            ], spacing=4, tight=True,
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=bg,
            border=ft.border.all(1, border_color),
            border_radius=12,
            padding=ft.padding.symmetric(horizontal=8, vertical=3),
            tooltip=tooltip,
            opacity=1.0,
            # Smooth fade between pulse states so the blink looks
            # designed, not janky.
            animate_opacity=ft.Animation(450, ft.AnimationCurve.EASE_IN_OUT),
            data={"id": stage_id, "active": active, "kind": kind,
                  "base_bg": bg, "base_border": border_color},
        )
        workflow_chips_by_id[stage_id] = chip
        return chip

    def _arrow():
        return ft.Icon(ft.Icons.ARROW_FORWARD, size=12,
                        color="#3D4156")

    workflow_row = ft.Row(
        [], spacing=4, wrap=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    workflow_label = ft.Text(
        "PIPELINE", size=10, color=ON_SURFACE_DIM,
        weight=ft.FontWeight.W_700,
    )
    workflow_legend = ft.Text(
        "Hover a stage for details. The active stage pulses while ingest runs.",
        size=10, color=ON_SURFACE_DIM, italic=True,
    )
    workflow_card = ft.Container(
        padding=ft.padding.symmetric(horizontal=10, vertical=8),
        bgcolor="#13151E",
        border=ft.border.all(1, "#1E2130"),
        border_radius=8,
        content=ft.Column([workflow_label, workflow_row, workflow_legend],
                           spacing=6, tight=True),
    )

    # Tooltips per stage. Two halves: what the stage does + when it runs.
    # The current on/off state is appended dynamically in render_workflow_chips
    # so users can hover and see "ON · this stage…" or "OFF · this stage…"
    STAGE_INFO = {
        "pypdf": (
            "Always on. The first pass — pypdf walks the PDF's content "
            "stream and extracts text via the font's ToUnicode cmap. Fast "
            "(~10–50 ms/page). Produces perfect output on ~90% of PDFs."
        ),
        "garbled": (
            "Always on. Per-page heuristic detector. Flags pypdf output "
            "as garbled when replacement-char ratio, backslash-escape "
            "ratio, low vowel ratio, or symbol-soup ratio breach safe "
            "thresholds. Triggers OCR fallback for just those pages."
        ),
        "ocr": (
            "Optional. When pypdf returns nothing or the heuristic flags "
            "a page as garbled, OCR re-extracts that page from a 2× "
            "rendered image (RapidOCR primary, Tesseract fallback). "
            "Slow (~500 ms–2 s per page), so only runs on bad pages."
        ),
        "llm_inspect": (
            "Optional. After parsing, sends each section to the LLM "
            "with a 'is this clean / garbled / partial?' prompt. "
            "Garbled → drop. Partial → keep + flag for correction. "
            "EXPENSIVE — one LLM call per section."
        ),
        "llm_correct": (
            "Optional. Sends OCR-recovered, partial-flagged, or "
            "questionable sections back to the LLM for a best-effort "
            "cleanup pass. UNRECOVERABLE responses are dropped instead "
            "of poisoning the index. EXPENSIVE — one extra LLM call "
            "per questionable section. Also makes upstream parsing "
            "permissive: TOC fragments and still-garbled OCR are sent "
            "to the LLM instead of being dropped silently."
        ),
        "contextual": (
            "Optional. Anthropic-style Contextual Retrieval — the LLM "
            "writes a one-sentence situational prefix for every chunk "
            "before embedding. Materially better recall on technical "
            "docs. EXPENSIVE — one LLM call per chunk; a 200-page book "
            "is hundreds of calls."
        ),
        "preview": (
            "Display-only. When on, the parser saves rendered page "
            "images and emits before/after/corrected payloads so the "
            "GUI can show a live recovery card for each fixed page. "
            "Costs ~50 ms + ~200 KB disk per recovered page."
        ),
    }

    def render_workflow_chips():
        workflow_chips_by_id.clear()
        c = state.cfg or Config()

        def tip(stage_id: str, on: bool, *, always_on: bool = False) -> str:
            if always_on:
                head = "ON (always)"
            else:
                head = "ON" if on else "OFF"
            return f"{head} · {STAGE_INFO[stage_id]}"

        chips: list[ft.Control] = []
        # Always-on backbone — these stages run on every PDF.
        chips += [_make_chip("pypdf", "pypdf", True, kind="auto",
                              tooltip=tip("pypdf", True, always_on=True)),
                  _arrow()]
        chips += [_make_chip("garbled", "garbled detector", True,
                              kind="auto",
                              tooltip=tip("garbled", True, always_on=True)),
                  _arrow()]
        # OCR fallback (toggleable).
        ocr_on = bool(getattr(c, "enable_ocr", True))
        chips.append(_make_chip("ocr", "OCR fallback", ocr_on, kind="opt",
                                 tooltip=tip("ocr", ocr_on)))
        # LLM inspect (opt-in).
        chips.append(_arrow())
        ins_on = bool(getattr(c, "llm_inspect_pages", False))
        chips.append(_make_chip("llm_inspect", "LLM inspect", ins_on,
                                 kind="opt",
                                 tooltip=tip("llm_inspect", ins_on)))
        # LLM correct (opt-in).
        chips.append(_arrow())
        cor_on = bool(getattr(c, "llm_correct_garbled", False))
        chips.append(_make_chip("llm_correct", "LLM correct", cor_on,
                                 kind="opt",
                                 tooltip=tip("llm_correct", cor_on)))
        # Contextual retrieval (opt-in, costly).
        chips.append(_arrow())
        ctx_on = bool(getattr(c, "enable_contextual", False))
        chips.append(_make_chip("contextual", "Contextual", ctx_on,
                                 kind="opt",
                                 tooltip=tip("contextual", ctx_on)))
        # Live preview (display-only flag, separate cluster).
        chips.append(ft.Container(width=10))
        prev_on = bool(getattr(c, "preview_garbled_recoveries", False))
        chips.append(_make_chip("preview", "Live preview", prev_on,
                                 kind="opt",
                                 tooltip=tip("preview", prev_on)))
        workflow_row.controls = chips

    render_workflow_chips()

    ingest_status = ft.Text("", size=12, color=ON_SURFACE_DIM)
    ingest_meta = ft.Text(
        "", size=11, color=ON_SURFACE_DIM, weight=ft.FontWeight.W_500,
    )  # files / bytes / ETA / db size line
    ingest_snippet_path = ft.Text(
        "", size=11, color=ACCENT, weight=ft.FontWeight.W_700,
    )
    ingest_snippet_text = ft.Text(
        "", size=12, color=ON_SURFACE_DIM, italic=True, max_lines=4,
    )
    ingest_snippet_card = ft.Container(
        visible=False,
        padding=ft.padding.symmetric(horizontal=12, vertical=10),
        bgcolor="#13151E",
        border=ft.border.all(1, "#1E2130"),
        border_radius=8,
        content=ft.Column([
            ingest_snippet_path,
            ingest_snippet_text,
        ], spacing=4, tight=True),
    )

    # ----- Garbled-page recovery preview card -----------------------------
    # Visible only when an OCR recovery event arrives during ingest. Shows
    # a thumbnail of the page + side-by-side before (garbled) / after
    # (OCR'd) text so the user can see what got fixed in real time.
    # Powered by IngestProgress.recovery payloads emitted by the parser
    # when cfg.preview_garbled_recoveries is True.
    ingest_recovery_header = ft.Text(
        "", size=12, color=WARNING, weight=ft.FontWeight.W_700,
    )
    ingest_recovery_image = ft.Image(
        src="", fit="contain", width=200, height=260, border_radius=4,
    )
    ingest_recovery_image_wrap = ft.Container(
        content=ingest_recovery_image,
        bgcolor="white", border_radius=6, padding=4, width=210, height=270,
    )
    ingest_recovery_before = ft.Text(
        "", size=11, color=ON_SURFACE_DIM, max_lines=12,
        font_family="monospace",
    )
    ingest_recovery_after = ft.Text(
        "", size=11, color=ON_SURFACE, max_lines=12,
    )
    # Third panel — populated when LLM correction runs on the page.
    # Hidden by default; flipped on by the watchdog when a "correction"
    # recovery event arrives for the currently-displayed page.
    ingest_recovery_corrected = ft.Text(
        "", size=11, color=ON_SURFACE, max_lines=12,
    )
    ingest_recovery_corrected_label = ft.Text(
        "After — LLM-corrected text",
        size=10, color=ACCENT, weight=ft.FontWeight.W_700,
    )
    ingest_recovery_corrected_box = ft.Container(
        content=ingest_recovery_corrected,
        bgcolor=ft.Colors.with_opacity(0.10, ACCENT),
        border=ft.border.all(1, ft.Colors.with_opacity(0.3, ACCENT)),
        border_radius=4, padding=8, expand=True,
    )
    ingest_recovery_corrected_label.visible = False
    ingest_recovery_corrected_box.visible = False
    ingest_recovery_card = ft.Container(
        visible=False,
        padding=ft.padding.symmetric(horizontal=12, vertical=10),
        bgcolor="#1F1A0E",   # subtle amber-tinted to match WARNING
        border=ft.border.all(1, WARNING),
        border_radius=8,
        content=ft.Column([
            ingest_recovery_header,
            ft.Row([
                ingest_recovery_image_wrap,
                ft.Column([
                    ft.Text("Before — what pypdf extracted (garbled)",
                            size=10, color=DANGER,
                            weight=ft.FontWeight.W_700),
                    ft.Container(
                        content=ingest_recovery_before,
                        bgcolor=ft.Colors.with_opacity(0.10, DANGER),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.3, DANGER)),
                        border_radius=4, padding=8, expand=True,
                    ),
                    ft.Text("After — re-extracted via OCR",
                            size=10, color=SUCCESS,
                            weight=ft.FontWeight.W_700),
                    ft.Container(
                        content=ingest_recovery_after,
                        bgcolor=ft.Colors.with_opacity(0.10, SUCCESS),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.3, SUCCESS)),
                        border_radius=4, padding=8, expand=True,
                    ),
                    ingest_recovery_corrected_label,
                    ingest_recovery_corrected_box,
                ], spacing=4, expand=True, tight=True),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=8, tight=True),
    )

    ingest_log = ft.ListView(spacing=2, padding=8, height=120, auto_scroll=True)

    # ----- "Done" summary card, shown only after ingest completes -----------
    ingest_done_headline = ft.Text(
        "Ingest complete", size=14, color=SUCCESS,
        weight=ft.FontWeight.W_700,
    )
    ingest_done_summary = ft.Text(
        "", size=12, color=ON_SURFACE,
    )
    ingest_done_meta = ft.Text(
        "", size=11, color=ON_SURFACE_DIM, italic=True,
    )
    ingest_done_card = ft.Container(
        visible=False,
        padding=ft.padding.symmetric(horizontal=14, vertical=12),
        bgcolor="#13251A",   # subtle green-tinted dark — works against most palettes
        border=ft.border.all(1, SUCCESS),
        border_radius=10,
        content=ft.Row([
            ft.Icon(ft.Icons.CHECK_CIRCLE, color=SUCCESS, size=24),
            ft.Container(width=4),
            ft.Column([
                ingest_done_headline,
                ingest_done_summary,
                ingest_done_meta,
            ], spacing=2, tight=True, expand=True),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
    )

    def reset_ingest_panel_for_run():
        """Hide the 'done' card and dim/clear stale state from the previous
        run so a fresh ingest starts with a clean slate."""
        ingest_done_card.visible = False
        ingest_status.color = ON_SURFACE_DIM
        ingest_status.weight = None
        ingest_status.size = 12
        ingest_snippet_card.visible = False
        ingest_recovery_card.visible = False

    def show_ingest_done(stats, elapsed_s: float):
        """Replace the in-flight UI with a clear 'finished' state.

        Hides the snippet/countdown card (its progress text becomes nonsense
        once nothing is running), promotes the status to a success-colored
        headline, and surfaces a tidy summary block.
        """
        # Promote status text
        ingest_status.value = "Finished"
        ingest_status.color = SUCCESS
        ingest_status.weight = ft.FontWeight.W_700
        ingest_status.size = 14

        # Build a human summary; suppress zero-count clauses for readability.
        bits = []
        if stats.files_new:
            bits.append(f"{stats.files_new} new")
        if stats.files_changed:
            bits.append(f"{stats.files_changed} changed")
        if stats.files_skipped_unchanged:
            bits.append(f"{stats.files_skipped_unchanged} unchanged")
        if stats.files_removed:
            bits.append(f"{stats.files_removed} removed")
        if stats.files_errored:
            bits.append(f"{stats.files_errored} errored")
        files_summary = " · ".join(bits) if bits else "no changes"
        ingest_done_summary.value = (
            f"{stats.chunks_added:,} chunks added  ·  {files_summary}"
        )

        # Format elapsed nicely (h:mm:ss for long runs)
        e = max(0.0, elapsed_s)
        if e >= 3600:
            h = int(e // 3600); m = int((e % 3600) // 60); s = int(e % 60)
            elapsed_str = f"{h}h{m:02d}m{s:02d}s"
        elif e >= 60:
            elapsed_str = f"{int(e // 60)}m{int(e % 60):02d}s"
        else:
            elapsed_str = f"{e:.1f}s"
        files_total = stats.files_seen
        ingest_done_meta.value = (
            f"{files_total} files scanned · elapsed {elapsed_str}"
        )

        # Switch off the in-flight card; show the done card.
        ingest_snippet_card.visible = False
        ingest_done_card.visible = True
        ingest_progress.visible = False

        # Bookend the streaming log so it's clear the activity has stopped.
        ingest_log.controls.append(ft.Text(
            f"  ✓ done — {stats.chunks_added:,} chunks, {elapsed_str}",
            size=11, color=SUCCESS,
            weight=ft.FontWeight.W_700, font_family="monospace",
        ))

    def add_files_clicked(_):
        if state.ws is None:
            _toast(state.page, "Open a workspace first")
            return

        async def _pick():
            try:
                files = await pick_files_picker.pick_files(allow_multiple=True)
            except Exception as ex:
                _toast(state.page, f"file picker failed: {ex}")
                return
            if not files:
                return
            copied = 0
            for f in files:
                try:
                    shutil.copy2(f.path, state.ws.docs_dir / Path(f.path).name)
                    copied += 1
                except Exception as ex:
                    _toast(state.page, f"copy failed {f.path}: {ex}")
            _toast(state.page, f"Added {copied} file(s) to docs/")
            refresh_files()
            refresh_status()

        state.page.run_task(_pick)

    def add_folder_clicked(_):
        if state.ws is None:
            _toast(state.page, "Open a workspace first")
            return

        async def _pick():
            try:
                path = await pick_dir_picker.get_directory_path(
                    dialog_title="Pick a folder to import",
                )
            except Exception as ex:
                _toast(state.page, f"folder picker failed: {ex}")
                return
            if not path:
                return
            src = Path(path)
            copied = 0
            for f in src.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in supported_extensions():
                    continue
                try:
                    shutil.copy2(f, state.ws.docs_dir / f.name)
                    copied += 1
                except Exception:
                    pass
            _toast(state.page, f"Imported {copied} files from {src}")
            refresh_files()
            refresh_status()

        state.page.run_task(_pick)

    def open_docs_folder(_):
        if state.ws is None:
            return
        path = state.ws.docs_dir
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as ex:
            _toast(state.page, f"open failed: {ex}")

    def file_row(f: Path, indexed_meta: dict | None) -> ft.Control:
        rel = f.relative_to(state.ws.docs_dir) if state.ws else f
        size = f.stat().st_size
        chunks = (indexed_meta or {}).get("n_chunks")
        chunks_chip = (
            ft.Container(
                content=ft.Text(f"{chunks} chunks", size=11,
                                color=SUCCESS,
                                weight=ft.FontWeight.W_600),
                bgcolor=ft.Colors.with_opacity(0.15, SUCCESS),
                padding=ft.padding.symmetric(horizontal=8, vertical=2),
                border_radius=999,
            ) if chunks
            else ft.Container(
                content=ft.Text("not indexed", size=11,
                                color=ON_SURFACE_DIM),
                bgcolor="#1E2130",
                padding=ft.padding.symmetric(horizontal=8, vertical=2),
                border_radius=999,
            )
        )
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=12, vertical=10),
            border_radius=8,
            bgcolor=SURFACE_DARK,
            border=ft.border.all(1, "#262938"),
            content=ft.Row(
                [
                    ft.Icon(file_icon(f), size=20, color=ACCENT),
                    ft.Column([
                        ft.Text(str(rel), size=13, color=ON_SURFACE,
                                weight=ft.FontWeight.W_500),
                        ft.Text(fmt_bytes(size), size=11,
                                color=ON_SURFACE_DIM),
                    ], spacing=2, tight=True, expand=True),
                    chunks_chip,
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=12,
            ),
        )

    def refresh_files():
        # Re-render the pipeline chips here too — the workspace's config
        # might have just loaded in (set_workspace_path calls this), so
        # the chips need to mirror whatever the user actually has saved
        # rather than the default Config the view was built with.
        render_workflow_chips()
        file_rows.controls.clear()
        if state.ws is None:
            file_rows.controls.append(ft.Text(
                "Open a workspace to manage its files.",
                color=ON_SURFACE_DIM, italic=True,
            ))
            summary.value = ""
            state.page.update()
            return
        files = sorted(p for p in state.ws.docs_dir.rglob("*") if p.is_file())
        # pull indexed chunk counts in one query
        indexed: dict[str, dict] = {}
        if state.ws.meta_db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(state.ws.meta_db_path))
                rows = conn.execute(
                    "SELECT path, n_chunks FROM files"
                ).fetchall()
                conn.close()
                indexed = {r[0]: {"n_chunks": r[1]} for r in rows}
            except Exception:
                pass
        for f in files[:1000]:
            rel = str(f.relative_to(state.ws.root))
            file_rows.controls.append(file_row(f, indexed.get(rel)))
        summary.value = f"{len(files)} file(s) in docs/"
        state.page.update()

    refresh_files_cb["fn"] = refresh_files

    # ----- Ingest controls ------------------------------------------------

    def do_ingest(force: bool):
        if state.ws is None:
            _toast(state.page, "Open a workspace first")
            return
        if state.streaming:
            _toast(state.page, "Already busy")
            return
        ingest_log.controls.clear()
        reset_ingest_panel_for_run()
        # Re-render the workflow chips so the user sees the active config
        # at the top of the run, even if Settings were toggled mid-session.
        render_workflow_chips()
        ingest_progress.visible = True
        ingest_progress.value = None
        # More descriptive than "Starting…" — tells the user what's
        # actually happening in the first few hundred ms before ingest()
        # has time to emit its own status. ingest()'s very first call
        # (within microseconds of being invoked) overwrites this.
        ingest_status.value = "Spinning up worker thread…"
        ingest_meta.value = "ingest is about to begin — initial setup runs on a background thread"
        state.page.update()

        # Shared state between progress callback, watchdog, and worker.
        ingest_state = {
            "last_log_path": "",
            "last_snippet_t": 0.0,
            "last_bytes_change_t": time.monotonic(),
            "last_bytes": 0,
            "last_page": 0,
            "last_status": "",
            "last_stall_log_t": 0.0,
            "stall_severity": 0,         # 0=ok, 1=slow, 2=stalled
            "current_prog": None,        # latest IngestProgress snapshot
            "last_recovery": None,       # latest IngestProgress.recovery payload
            "last_recovery_applied": None,  # what's actually painted on the card
            # Accumulated panel state per (file, page) — merges OCR and
            # subsequent LLM-correction events so all three panels can stay
            # in sync. Maps "<file>::<page>" → dict of {kind, image_path,
            # before, after, corrected, unrecoverable}.
            "recovery_pages": {},
            "current_recovery_key": None,  # which (file, page) is on screen now
            # Workflow-chip pulse state: id of the chip currently lit
            # ("pypdf" / "ocr" / "llm_inspect" / etc.) and the boolean
            # toggled by the watchdog every tick to drive the blink.
            "active_stage": None,
            "pulse_phase": False,
            "started_at": time.monotonic(),
            "running": True,
        }

        SLOW_THRESHOLD_S = 10     # no progress for this long → log "slow"
        STALL_THRESHOLD_S = 45    # → log "stalled" warning
        WATCHDOG_TICK_S = 0.5     # how often the heartbeat runs

        def fmt_bytes(n: int) -> str:
            n = float(n)
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024 or unit == "GB":
                    return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} {unit}"
                n /= 1024
            return f"{n:.1f} GB"

        def fmt_eta(s):
            if s is None or s <= 0:
                return "—"
            if s >= 3600:
                return f"{int(s/3600)}h{int((s%3600)/60):02d}m"
            return f"{int(s/60)}:{int(s%60):02d}"

        def fmt_meta_line(prog, *, override_elapsed: float | None = None,
                           extra: str = "") -> str:
            elapsed = override_elapsed if override_elapsed is not None else prog.elapsed_s
            return (
                f"{prog.files_done}/{prog.files_total} files  ·  "
                f"{fmt_bytes(prog.bytes_done)} / {fmt_bytes(prog.bytes_total)}  ·  "
                f"{fmt_bytes(int(prog.rate_bps))}/s  ·  "
                f"elapsed {fmt_eta(elapsed)}  ·  ETA {fmt_eta(prog.eta_s)}  ·  "
                f"{prog.chunks_done} chunks  ·  index {fmt_bytes(prog.db_bytes)}"
                + (f"  ·  {extra}" if extra else "")
            )

        def progress_cb(prog):
            # Stall detection considers any forward signal — bytes finishing
            # a file, page index advancing inside a long parse, or the status
            # text changing (e.g. "parsing" → "embedding 32/200 chunks").
            advanced = (
                prog.bytes_done > ingest_state["last_bytes"]
                or (prog.page or 0) > ingest_state["last_page"]
                or prog.status != ingest_state["last_status"]
            )
            if advanced:
                ingest_state["last_bytes_change_t"] = time.monotonic()
                if ingest_state["stall_severity"] > 0:
                    ingest_log.controls.append(ft.Text(
                        f"  ▶ resumed: {Path(prog.current_path).name}",
                        size=11, color=SUCCESS, font_family="monospace",
                    ))
                    ingest_state["stall_severity"] = 0
            ingest_state["last_bytes"] = prog.bytes_done
            ingest_state["last_page"] = prog.page or 0
            ingest_state["last_status"] = prog.status
            ingest_state["current_prog"] = prog

            # Bar
            if prog.bytes_total > 0:
                ingest_progress.value = min(1.0, prog.bytes_pct)
            ingest_status.value = (
                f"{prog.status} — {Path(prog.current_path).name}"
                if prog.current_path
                else prog.status
            )
            ingest_meta.value = fmt_meta_line(prog)

            # File-transition log line
            if prog.current_path and prog.current_path != ingest_state["last_log_path"]:
                ingest_state["last_log_path"] = prog.current_path
                ingest_log.controls.append(
                    ft.Text(
                        f"{prog.status[:18]:>18}  {Path(prog.current_path).name}",
                        size=11, color=ON_SURFACE_DIM,
                        font_family="monospace",
                    )
                )

            # Snippet card refreshes whenever the ingest pipeline sends a
            # new sample (chunk text changing during contextualization,
            # page-image preview, etc.). No countdown — the snippet just
            # updates as new samples arrive, which on a busy run is
            # multiple times per second.
            now = time.monotonic()
            if prog.snippet:
                page_str = f"  ·  page {prog.page}" if prog.page else ""
                ingest_snippet_path.value = (
                    f"{Path(prog.current_path).name}{page_str}"
                )
                ingest_snippet_text.value = "“" + prog.snippet + "…”"
                ingest_snippet_card.visible = True
                ingest_state["last_snippet_t"] = now

            # Garbled-page recovery preview: just stash the latest payload
            # in shared state. The async watchdog actually paints the
            # card — same reason all other dynamic values do, since
            # page.update() from a worker thread doesn't reliably
            # trigger a frame on Windows Flutter Desktop.
            rec = getattr(prog, "recovery", None)
            if rec:
                ingest_state["last_recovery"] = rec

            state.page.update()

        async def watchdog():
            """Async watchdog. MUST be async + scheduled via page.run_task,
            not page.run_thread — `page.update()` only paints reliably
            from coroutines on Flet's asyncio loop. From a sync thread it
            queues the change and waits for an OS paint event before
            flushing, which is exactly the "doesn't refresh until I
            alt-tab" symptom."""
            import asyncio
            while ingest_state["running"]:
                await asyncio.sleep(WATCHDOG_TICK_S)
                if not ingest_state["running"]:
                    break
                prog = ingest_state["current_prog"]
                if prog is None:
                    continue
                now = time.monotonic()
                live_elapsed = now - ingest_state["started_at"]
                since_change = now - ingest_state["last_bytes_change_t"]

                stall_msg = ""
                if since_change >= STALL_THRESHOLD_S:
                    stall_msg = f"⚠ stalled ({int(since_change)}s)"
                    severity = 2
                elif since_change >= SLOW_THRESHOLD_S:
                    stall_msg = f"⏳ slow ({int(since_change)}s)"
                    severity = 1
                else:
                    severity = 0

                # Re-apply the LATEST status text + bar from prog, in case
                # progress_cb's own page.update() got dropped on the floor
                # by Flet 0.84 (the well-known "background-thread updates
                # don't always paint without a window event" issue). The
                # watchdog runs on page.run_thread() context, which paints
                # reliably, so we mirror the worker's state here. Without
                # this, the status text only repaints when the user pokes
                # the window (Win+S, click, focus change, etc.).
                ingest_status.value = (
                    f"{prog.status} — {Path(prog.current_path).name}"
                    if prog.current_path else prog.status
                )
                if prog.bytes_total > 0:
                    ingest_progress.value = min(1.0, prog.bytes_pct)

                ingest_meta.value = fmt_meta_line(
                    prog, override_elapsed=live_elapsed, extra=stall_msg,
                )

                # Pipeline pulse — figure out which stage is currently
                # running from the status text and pulse that chip's
                # opacity. Substring-matched (cheap + tolerant of the
                # status-text variants ingest emits).
                status_l = (prog.status or "").lower()
                running = ingest_state.get("running")
                active_stage = None
                if running and prog.current_path:
                    if ("ocr'ing" in status_l
                            or "recovered garbled page" in status_l
                            or "via ocr" in status_l):
                        active_stage = "ocr"
                    elif "llm-inspecting" in status_l or "llm inspect" in status_l:
                        active_stage = "llm_inspect"
                    elif ("llm-correcting" in status_l
                            or "llm corrected" in status_l
                            or "llm declined" in status_l
                            or "llm correction" in status_l):
                        active_stage = "llm_correct"
                    elif "contextualizing" in status_l:
                        active_stage = "contextual"
                    elif "parsing" in status_l or "parsed" in status_l:
                        active_stage = "pypdf"
                    elif "preview" in status_l:
                        active_stage = "preview"

                # Switch which chip is "active" — un-pulse the previous one.
                prev_stage = ingest_state.get("active_stage")
                if active_stage != prev_stage:
                    if prev_stage and prev_stage in workflow_chips_by_id:
                        workflow_chips_by_id[prev_stage].opacity = 1.0
                    ingest_state["active_stage"] = active_stage

                # Toggle pulse phase every tick. Setting opacity drives
                # the animate_opacity transition we configured on the
                # chip — a 450ms ease-in-out fade between 0.45 and 1.0.
                ingest_state["pulse_phase"] = not ingest_state["pulse_phase"]
                if active_stage and active_stage in workflow_chips_by_id:
                    chip = workflow_chips_by_id[active_stage]
                    chip.opacity = 0.45 if ingest_state["pulse_phase"] else 1.0

                # Mirror the latest recovery event onto the card. Only
                # touch the controls when the payload actually changed,
                # so we don't repaint identical content every tick.
                #
                # Two event kinds share this channel:
                #   kind="ocr"        — fired during parse, has image+before+after
                #   kind="correction" — fired after parse when LLM correction
                #                       runs; has before (=pre-correction text)
                #                       + after (=cleaned text)
                # We merge per-page so a correction event for a previously-
                # OCR'd page just adds the third panel without clobbering
                # the image and original before/after.
                rec = ingest_state.get("last_recovery")
                if rec and rec is not ingest_state.get("last_recovery_applied"):
                    file_key = rec.get("file") or prog.current_path or ""
                    page_key = rec.get("page")
                    key = f"{file_key}::{page_key}"
                    page_state = ingest_state["recovery_pages"].setdefault(
                        key, {"file": file_key, "page": page_key}
                    )
                    kind = rec.get("kind", "ocr")
                    if kind == "correction":
                        page_state["corrected"] = rec.get("after") or ""
                        page_state["unrecoverable"] = bool(
                            rec.get("unrecoverable")
                        )
                        # If we have no prior OCR record, also keep the
                        # before-text from the correction event so the
                        # card has something to show in the "before" slot.
                        if "before" not in page_state:
                            page_state["before"] = rec.get("before") or ""
                            page_state["after"] = rec.get("before") or ""
                    else:
                        page_state["image_path"] = rec.get("image_path") or ""
                        page_state["before"] = rec.get("before") or ""
                        page_state["after"] = rec.get("after") or ""

                    fname = Path(file_key).name
                    has_correction = "corrected" in page_state
                    if has_correction:
                        ingest_recovery_header.value = (
                            f"Recovered + corrected page {page_key} of {fname}"
                        )
                    else:
                        ingest_recovery_header.value = (
                            f"Recovered page {page_key} of {fname} via OCR"
                        )
                    img_path = page_state.get("image_path") or ""
                    if img_path:
                        ingest_recovery_image.src = img_path
                        ingest_recovery_image_wrap.visible = True
                    else:
                        ingest_recovery_image_wrap.visible = False
                    before = (page_state.get("before") or "").strip()
                    after = (page_state.get("after") or "").strip()
                    ingest_recovery_before.value = (
                        before[:1200] + ("…" if len(before) > 1200 else "")
                    ) or "(no text — extraction failed entirely)"
                    ingest_recovery_after.value = (
                        after[:1200] + ("…" if len(after) > 1200 else "")
                    ) or "(OCR also returned nothing — page dropped)"

                    if has_correction:
                        if page_state.get("unrecoverable"):
                            ingest_recovery_corrected_label.value = (
                                "After — LLM declined "
                                "(UNRECOVERABLE; original kept)"
                            )
                            ingest_recovery_corrected.value = (
                                "(no cleanup applied)"
                            )
                        else:
                            ingest_recovery_corrected_label.value = (
                                "After — LLM-corrected text"
                            )
                            corrected = (page_state.get("corrected")
                                         or "").strip()
                            ingest_recovery_corrected.value = (
                                corrected[:1200]
                                + ("…" if len(corrected) > 1200 else "")
                            ) or "(empty)"
                        ingest_recovery_corrected_label.visible = True
                        ingest_recovery_corrected_box.visible = True
                    else:
                        ingest_recovery_corrected_label.visible = False
                        ingest_recovery_corrected_box.visible = False

                    ingest_recovery_card.visible = True
                    ingest_state["current_recovery_key"] = key
                    ingest_state["last_recovery_applied"] = rec

                # Update the OS window title so the user can see ingest
                # progress in the taskbar / dock even when ez-rag is
                # minimized, in the background, or on another tab. Flet
                # propagates title changes regardless of focus state, and
                # on Windows the SetWindowText syscall happens to also
                # nudge Flutter into flushing pending paints — bonus.
                try:
                    if prog.files_total:
                        pct = int(100 * prog.bytes_pct)
                        state.page.title = (
                            f"ez-rag — ingesting {prog.files_done}"
                            f"/{prog.files_total} ({pct}%)"
                        )
                    else:
                        # Even before files_total is known, set title so
                        # the taskbar shows we're not frozen.
                        state.page.title = (
                            f"ez-rag — {prog.status[:40]}"
                            if prog.status else "ez-rag — ingesting…"
                        )
                except Exception:
                    pass

                # Log the slow/stall transition once per severity escalation,
                # then every 30 s while it persists.
                if severity > 0 and (
                    severity > ingest_state["stall_severity"]
                    or (now - ingest_state["last_stall_log_t"]) >= 30.0
                ):
                    file = Path(prog.current_path).name if prog.current_path else "—"
                    color = WARNING if severity == 1 else DANGER
                    label = "slow" if severity == 1 else "stalled"
                    ingest_log.controls.append(ft.Text(
                        f"  {'⏳' if severity == 1 else '⚠'} {label}: "
                        f"{file} — {int(since_change)}s with no byte progress "
                        f"(status: {prog.status[:30]})",
                        size=11, color=color, font_family="monospace",
                    ))
                    ingest_state["last_stall_log_t"] = now
                    ingest_state["stall_severity"] = severity

                try:
                    state.page.update()
                except Exception:
                    pass
                # NUCLEAR: force a Win32 paint message so the queued value
                # changes from the worker thread actually render. Without
                # this, on Windows the user has to alt-tab / Win+S / click
                # the window to see updates — Flutter Desktop pauses
                # repaints when nothing is triggering a frame and Flet's
                # background-thread updates don't always wake it.
                force_window_redraw()

        async def worker():
            """Async worker — runs the synchronous `ingest()` call on a
            thread via asyncio.to_thread so blocking I/O doesn't pin the
            event loop, but the wrapping context is async so any
            page.update() calls we make here go through the working path."""
            import asyncio
            try:
                stats = await asyncio.to_thread(
                    ingest, state.ws, cfg=state.cfg, force=force,
                    progress=progress_cb,
                )
                show_ingest_done(stats, stats.seconds)
                if stats.errors:
                    ingest_log.controls.append(ft.Text(
                        f"{len(stats.errors)} error(s):", color=DANGER, size=12,
                    ))
                    for path, err in stats.errors[:5]:
                        ingest_log.controls.append(ft.Text(
                            f"  {Path(path).name}: {err}", color=DANGER, size=11,
                        ))
            except Exception as ex:
                ingest_status.value = f"Failed: {ex}"
                ingest_status.color = DANGER
                ingest_status.weight = ft.FontWeight.W_700
                ingest_done_card.visible = False
            finally:
                ingest_state["running"] = False
                ingest_progress.visible = False
                # Stop the pulse and restore every chip to full opacity.
                ingest_state["active_stage"] = None
                for chip in workflow_chips_by_id.values():
                    chip.opacity = 1.0
                try:
                    state.page.title = "ez-rag"
                except Exception:
                    pass
                refresh_files()
                refresh_status()
                state.page.update()

        # Both run via page.run_task — async coroutines on Flet's asyncio
        # loop. This is the only context where page.update() reliably
        # triggers a Flutter frame on Windows. Using page.run_thread()
        # (which we did before) puts the call on a thread pool that's
        # outside the event loop, and updates from there only paint when
        # an OS event (focus, click, alt-tab, Win+S) wakes the renderer.
        state.page.run_task(watchdog)
        state.page.run_task(worker)

    ingest_card = section_card(
        "INGEST",
        ft.Row([
            ft.FilledButton("Ingest", icon=ft.Icons.PLAY_ARROW,
                            on_click=lambda _: do_ingest(False),
                            bgcolor=ACCENT, color="#FFFFFF",
                            tooltip=TIP["ingest_btn"]),
            ft.OutlinedButton("Re-ingest (force)", icon=ft.Icons.REFRESH,
                              on_click=lambda _: do_ingest(True),
                              tooltip=TIP["reingest_btn"]),
        ], spacing=10),
        workflow_card,
        ingest_progress,
        ingest_status,
        ingest_meta,
        ingest_done_card,
        ingest_recovery_card,
        ingest_snippet_card,
        ft.Container(
            content=ingest_log,
            bgcolor=BG_DARK,
            border_radius=8,
            border=ft.border.all(1, "#222637"),
            padding=4,
        ),
    )

    docs_card = section_card(
        "DOCS",
        ft.Row([
            ft.FilledButton("Add files", icon=ft.Icons.UPLOAD_FILE,
                            on_click=add_files_clicked,
                            bgcolor=ACCENT, color="#FFFFFF",
                            tooltip=TIP["add_files"]),
            ft.OutlinedButton("Add folder", icon=ft.Icons.CREATE_NEW_FOLDER,
                              on_click=add_folder_clicked,
                              tooltip=TIP["add_folder"]),
            ft.TextButton("Open in Explorer", icon=ft.Icons.OPEN_IN_NEW,
                          on_click=open_docs_folder,
                          tooltip=TIP["open_docs"]),
            ft.Container(expand=True),
            ft.IconButton(icon=ft.Icons.REFRESH,
                          tooltip=TIP["refresh_files"],
                          on_click=lambda _: refresh_files()),
            summary,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Container(
            content=file_rows,
            bgcolor=BG_DARK,
            border_radius=8,
            border=ft.border.all(1, "#222637"),
            padding=8,
            expand=True,
        ),
        padding=18,
    )

    # Split between the file list (left) and the status / ingest pane
    # (right). Stepped through five fixed presets via two arrow buttons
    # on a center divider. Hardcoded because Flet 0.84 silently ignores
    # post-construction width / expand mutations on Container — the
    # bulletproof workaround is to rebuild the row's controls list with
    # NEW Container instances, which is exactly what apply_split does.
    SPLIT_PRESETS = [0.20, 0.35, 0.55, 0.70, 0.82]
    split_state = {"idx": 2}   # default = balanced (0.55)

    def _make_pane(content, flex):
        return ft.Container(content=content, expand=flex)

    def _split_panes():
        ratio = SPLIT_PRESETS[split_state["idx"]]
        left = max(1, int(ratio * 100))
        right = max(1, 100 - left)
        return _make_pane(docs_card, left), _make_pane(ingest_card, right)

    docs_pane, ingest_pane = _split_panes()

    def can_grow_left():
        return split_state["idx"] < len(SPLIT_PRESETS) - 1

    def can_grow_right():
        return split_state["idx"] > 0

    def step(direction: int):
        # +1 grows the LEFT pane (files); -1 grows the RIGHT pane (status).
        new_idx = max(0, min(len(SPLIT_PRESETS) - 1,
                              split_state["idx"] + direction))
        if new_idx == split_state["idx"]:
            return
        split_state["idx"] = new_idx
        nonlocal docs_pane, ingest_pane
        docs_pane, ingest_pane = _split_panes()
        # Rebuild the row's controls list — NEW Container instances force
        # Flet to relayout. Mutating expand on existing instances doesn't.
        split_row.controls = [docs_pane, divider, ingest_pane]
        update_arrow_state()
        try:
            split_row.update()
        except Exception:
            state.page.update()

    def update_arrow_state():
        btn_left.disabled = not can_grow_left()
        btn_right.disabled = not can_grow_right()
        btn_left.opacity = 1.0 if can_grow_left() else 0.35
        btn_right.opacity = 1.0 if can_grow_right() else 0.35
        ratio_label.value = (
            f"{int(SPLIT_PRESETS[split_state['idx']]*100)}"
            f" / {int((1-SPLIT_PRESETS[split_state['idx']])*100)}"
        )

    btn_left = ft.IconButton(
        icon=ft.Icons.CHEVRON_LEFT, icon_size=16, icon_color=ON_SURFACE,
        bgcolor="#1E2130",
        tooltip="Grow file list (shrink status pane)",
        on_click=lambda _: step(+1),
        style=ft.ButtonStyle(shape=ft.CircleBorder(), padding=4),
    )
    btn_right = ft.IconButton(
        icon=ft.Icons.CHEVRON_RIGHT, icon_size=16, icon_color=ON_SURFACE,
        bgcolor="#1E2130",
        tooltip="Grow status pane (shrink file list)",
        on_click=lambda _: step(-1),
        style=ft.ButtonStyle(shape=ft.CircleBorder(), padding=4),
    )
    ratio_label = ft.Text(
        "", size=9, color=ON_SURFACE_DIM, weight=ft.FontWeight.W_700,
    )

    divider_rail_top = ft.Container(width=2, expand=True,
                                     bgcolor="#2A2E3F", border_radius=1)
    divider_rail_bot = ft.Container(width=2, expand=True,
                                     bgcolor="#2A2E3F", border_radius=1)
    divider = ft.Container(
        width=24,
        content=ft.Column(
            [
                divider_rail_top,
                ft.Container(
                    content=ft.Column(
                        [btn_left, ratio_label, btn_right],
                        spacing=2, tight=True,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    bgcolor="#13151E",
                    border=ft.border.all(1, "#262938"),
                    border_radius=14,
                    padding=ft.padding.symmetric(horizontal=2, vertical=4),
                ),
                divider_rail_bot,
            ],
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        ),
    )

    split_row = ft.Row(
        [docs_pane, divider, ingest_pane],
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )

    update_arrow_state()

    container = ft.Container(
        bgcolor=BG_DARK,
        expand=True,
        padding=20,
        content=ft.Column(
            [
                ft.Text("Files", size=18, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE),
                ft.Container(height=12),
                split_row,
            ],
            expand=True,
        ),
    )

    return container, refresh_files


# ============================================================================
# Settings view
# ============================================================================

def _norm_tag(tag: str) -> str:
    """Ollama treats `foo` and `foo:latest` as the same. Normalize for matching."""
    if not tag:
        return tag
    return tag[:-len(":latest")] if tag.endswith(":latest") else tag


def _model_dropdown_options(
    installed: list,
    current: str,
    *,
    embed_only: bool = False,
) -> list[ft.dropdown.Option]:
    """Options for an LLM/embed model dropdown.

    Tags ending in `:latest` are stored under their bare name so the dropdown
    value (which may be either form in the config) matches.
    """
    options: list[ft.dropdown.Option] = []
    seen: set[str] = set()
    for m in installed:
        if embed_only and not is_embed_capable(m.tag):
            continue
        if not embed_only and is_embed_capable(m.tag):
            continue
        key = _norm_tag(m.tag)
        if key in seen:
            continue
        seen.add(key)
        options.append(ft.dropdown.Option(
            key=key, text=f"{key}    {fmt_size(m.size)}",
        ))
    norm_current = _norm_tag(current)
    if current and norm_current not in seen:
        is_path = ("/" in current) or ("\\" in current)
        label = (Path(current).name + "  (local GGUF)") if is_path \
            else f"{current}  (not pulled)"
        options.append(ft.dropdown.Option(key=current, text=label))
    return options


def build_settings_view(state: AppState, *, refresh_status, on_pick_workspace):
    page = state.page

    # Local FilePicker for choosing a GGUF file
    gguf_picker = ft.FilePicker()
    page.services.append(gguf_picker)
    rags_dir_picker = ft.FilePicker()
    page.services.append(rags_dir_picker)

    def _build_appearance_card(_state: AppState) -> ft.Control:
        themes = load_themes()
        names = sorted(themes.keys())
        active = get_theme_name() if get_theme_name() in themes else "dark"

        # A small swatch row so the user can preview the selected palette
        # without restarting.
        swatch_row = ft.Row([], spacing=4, wrap=True)

        def render_swatches(name: str):
            swatch_row.controls.clear()
            pal = themes.get(name) or themes["dark"]
            for key in ("accent", "accent_soft", "bg", "surface", "surface_hi",
                        "success", "warning", "danger", "user_bubble",
                        "assist_bubble", "chip_bg"):
                swatch_row.controls.append(ft.Container(
                    width=18, height=18, bgcolor=pal[key], border_radius=4,
                    tooltip=f"{key}: {pal[key]}",
                    border=ft.border.all(1, "#0008"),
                ))
            try:
                page.update()
            except Exception:
                pass

        def fmt_label(n: str) -> str:
            return n.replace("_", " ").title()

        dropdown = ft.Dropdown(
            label="Palette",
            value=active,
            options=[ft.dropdown.Option(n, fmt_label(n)) for n in names],
            width=240, dense=True,
            tooltip=TIP["theme_palette"],
        )
        dropdown.on_change = lambda e: render_swatches(e.control.value)

        status_text = ft.Text("", size=11, color=ON_SURFACE_DIM, italic=True)

        def apply_theme(_):
            chosen = dropdown.value or "dark"
            if chosen == get_theme_name():
                status_text.value = "Already active. Pick a different palette."
                page.update()
                return
            set_theme_name(chosen)
            status_text.value = (
                f"Saved · '{fmt_label(chosen)}' applies on next launch. "
                "Click Restart now."
            )
            page.update()

        def restart_now(_):
            try:
                # Save first in case user picked-but-didn't-press Apply.
                if dropdown.value and dropdown.value != get_theme_name():
                    set_theme_name(dropdown.value)
                # On Windows, sys.executable points at python; sys.argv[0]
                # is the script path. Re-execing the current process is the
                # simplest way to fully tear down and rebuild Flet's UI.
                py = _sys.executable
                os.execv(py, [py, *_sys.argv])
            except Exception as ex:
                _toast(page, f"Restart failed: {ex}. Close and reopen ez-rag.")

        render_swatches(active)

        return section_card(
            "APPEARANCE",
            ft.Row([
                ft.Icon(ft.Icons.PALETTE, size=16, color=ACCENT),
                ft.Text("Theme palette", size=11,
                        color=ON_SURFACE_DIM, weight=ft.FontWeight.W_700),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            dropdown,
            swatch_row,
            ft.Row([
                ft.FilledButton("Apply", icon=ft.Icons.CHECK,
                                on_click=apply_theme,
                                bgcolor=ACCENT, color="#FFFFFF",
                                tooltip=TIP["theme_apply"]),
                ft.OutlinedButton("Restart now", icon=ft.Icons.RESTART_ALT,
                                  on_click=restart_now,
                                  tooltip=TIP["theme_restart"]),
            ], spacing=8),
            status_text,
            ft.Text(
                f"Edit {THEMES_FILE.name} to add or tweak palettes — RGB hex "
                "values, one TOML table per theme.",
                size=11, color=ON_SURFACE_DIM, italic=True,
            ),
        )

    def _build_storage_card(_state: AppState) -> ft.Control:
        path_text = ft.Text(
            str(get_default_rags_dir()),
            size=12, color=ACCENT, weight=ft.FontWeight.W_600,
            selectable=True,
        )

        def refresh_path():
            path_text.value = str(get_default_rags_dir())
            try:
                page.update()
            except Exception:
                pass

        def pick_dir(_):
            async def _do():
                p = await rags_dir_picker.get_directory_path(
                    dialog_title="Where should new RAGs live?",
                )
                if p:
                    set_default_rags_dir(Path(p))
                    refresh_path()
                    _toast(page, f"Default RAGs folder → {p}")
            page.run_task(_do)

        def open_in_explorer(_):
            try:
                import subprocess
                p = get_default_rags_dir()
                p.mkdir(parents=True, exist_ok=True)
                if os.name == "nt":
                    os.startfile(p)  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(p)])
            except Exception as ex:
                _toast(page, f"Open failed: {ex}")

        def manage_rags(_):
            open_manage_rags_overlay(
                page,
                on_open_workspace=on_pick_workspace,
                on_change=refresh_path,
            )

        def export_chatbot_clicked(_):
            if state.ws is None or not state.ws.is_initialized():
                _toast(page, "Open a workspace first.")
                return
            if not state.ws.meta_db_path.exists():
                _toast(page, "No index yet — run an ingest first.")
                return
            from ez_rag.export import estimate_sources_size, export_chatbot
            n_files, n_bytes = estimate_sources_size(state.ws)
            mb = n_bytes / (1024 * 1024)

            # ----- confirm overlay with options -----
            overlay = ft.Container(
                expand=True, visible=True,
                bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
                alignment=ft.Alignment.CENTER,
            )

            include_sources_cb = ft.Checkbox(
                label=f"Include source files  ({n_files} files, "
                      f"{mb:.1f} MB)",
                value=False, active_color=ACCENT,
                tooltip=TIP["include_sources"],
            )

            def _close(_=None):
                overlay.visible = False
                try:
                    if overlay in page.overlay:
                        page.overlay.remove(overlay)
                except Exception:
                    pass
                page.update()

            async def _confirm(_=None):
                include = bool(include_sources_cb.value)
                _close()
                # File picker for the destination zip
                default_name = (
                    f"{state.ws.root.name}-chatbot"
                    + ("-with-sources" if include else "") + ".zip"
                )
                try:
                    p = await rags_dir_picker.save_file(
                        dialog_title="Save chatbot bundle",
                        file_name=default_name,
                        allowed_extensions=["zip"],
                    )
                except Exception as ex:
                    _toast(page, f"Save dialog failed: {ex}")
                    return
                if not p:
                    return
                # Spin off the actual export on a background thread —
                # bundling 2 GB of PDFs would otherwise freeze the UI for
                # 10+ seconds.
                def _bg():
                    try:
                        themes = load_themes()
                        palette = themes.get(get_theme_name()) or themes["dark"]
                        out = export_chatbot(
                            state.ws, Path(p),
                            palette=palette,
                            title=state.ws.root.name,
                            include_sources=include,
                        )
                        size_mb = out.stat().st_size / (1024 * 1024)
                        _toast(page,
                            f"Chatbot exported → {out.name} ({size_mb:.1f} MB)"
                        )
                    except Exception as ex:
                        _toast(page, f"Export failed: {ex}")
                page.run_thread(_bg)

            overlay.content = ft.Container(
                width=520, padding=20,
                bgcolor=SURFACE_DARK,
                border=ft.border.all(1, "#262938"),
                border_radius=14,
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.IOS_SHARE, color=ACCENT, size=22),
                        ft.Text("Export chatbot", size=16,
                                weight=ft.FontWeight.W_700, color=ON_SURFACE),
                    ], spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Text(
                        "The chatbot index, vendored library, web UI, and "
                        "launchers are always bundled. Source files are "
                        "optional — bundle them if you want citation chips "
                        "in the chatbot to render PDF pages and original "
                        "screenshots.",
                        size=12, color=ON_SURFACE_DIM,
                    ),
                    include_sources_cb,
                    ft.Text(
                        ("With sources: ~"
                         f"{mb + 30:.0f} MB total (estimate). "
                         "Without sources: ~30 MB. "
                         "Citations show chunk text either way."),
                        size=11, color=ON_SURFACE_DIM, italic=True,
                    ),
                    ft.Row([
                        ft.Container(expand=True),
                        ft.TextButton("Cancel", on_click=_close),
                        ft.FilledButton(
                            "Export…", icon=ft.Icons.DOWNLOAD,
                            on_click=lambda e: page.run_task(_confirm),
                            bgcolor=ACCENT, color="#FFFFFF",
                        ),
                    ], spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ], spacing=14, tight=True),
            )
            page.overlay.append(overlay)
            page.update()

        return section_card(
            "STORAGE",
            ft.Row([
                ft.Icon(ft.Icons.FOLDER_OPEN, size=16, color=ACCENT),
                ft.Text("Default RAGs folder:", size=11,
                        color=ON_SURFACE_DIM, weight=ft.FontWeight.W_700),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            path_text,
            ft.Row([
                ft.OutlinedButton("Change…", icon=ft.Icons.EDIT,
                                  on_click=pick_dir,
                                  tooltip=TIP["default_rags_dir"]),
                ft.OutlinedButton("Open in Explorer", icon=ft.Icons.OPEN_IN_NEW,
                                  on_click=open_in_explorer),
                ft.FilledButton("Manage RAGs…", icon=ft.Icons.STORAGE,
                                on_click=manage_rags,
                                bgcolor=ACCENT, color="#FFFFFF",
                                tooltip=TIP["manage_rags"]),
            ], spacing=8, wrap=True),
            ft.Container(height=4),
            ft.Row([
                ft.Icon(ft.Icons.IOS_SHARE, size=16, color=ACCENT),
                ft.Text("Portable chatbot:", size=11,
                        color=ON_SURFACE_DIM, weight=ft.FontWeight.W_700),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(
                "Bundle this RAG with a runnable chatbot (Windows / macOS / "
                "Linux). Settings, model choice, and index are baked in — the "
                "recipient just runs the launcher.",
                size=11, color=ON_SURFACE_DIM, italic=True,
            ),
            ft.Row([
                ft.FilledButton("Export Chatbot…", icon=ft.Icons.DOWNLOAD,
                                on_click=export_chatbot_clicked,
                                bgcolor=ACCENT, color="#FFFFFF",
                                tooltip=TIP["export_chatbot"]),
            ], spacing=8),
        )

    chunk_size = ft.TextField(label="Chunk size (tokens)", value="512",
                              width=180, dense=True, tooltip=TIP["chunk_size"])
    chunk_overlap = ft.TextField(label="Chunk overlap", value="64",
                                 width=180, dense=True,
                                 tooltip=TIP["chunk_overlap"])
    top_k = ft.TextField(label="Top-K", value="8", width=120, dense=True,
                         tooltip=TIP["top_k"])
    hybrid = ft.Switch(label="Hybrid (BM25 + dense)", value=True,
                       active_color=ACCENT, tooltip=TIP["hybrid"])
    rerank = ft.Switch(label="Rerank (cross-encoder)", value=True,
                       active_color=ACCENT, tooltip=TIP["rerank"])
    use_hyde_sw = ft.Switch(label="HyDE", value=False,
                            active_color=ACCENT, tooltip=TIP["use_hyde"])
    multi_query_sw = ft.Switch(label="Multi-query", value=False,
                               active_color=ACCENT, tooltip=TIP["multi_query"])
    context_window_field = ft.TextField(
        label="Context window (±N chunks)", value="0", width=200, dense=True,
        tooltip=TIP["context_window"],
    )
    use_mmr_sw = ft.Switch(label="MMR diversity", value=False,
                           active_color=ACCENT, tooltip=TIP["use_mmr"])
    mmr_lambda_field = ft.TextField(
        label="MMR λ (0–1)", value="0.5", width=140, dense=True,
        tooltip=TIP["mmr_lambda"],
    )
    expand_chapter_sw = ft.Switch(
        label="Expand to chapter", value=False, active_color=ACCENT,
        tooltip=TIP["expand_to_chapter"],
    )
    chapter_max_chars_field = ft.TextField(
        label="Chapter cap (chars)", value="16000", width=180, dense=True,
        tooltip=TIP["chapter_max_chars"],
    )
    use_corpus = ft.Switch(label="Use RAG", value=True,
                           active_color=ACCENT, tooltip=TIP["use_corpus"])
    unload_llm_sw = ft.Switch(
        label="Unload LLM during ingest",
        value=True, active_color=ACCENT, tooltip=TIP["unload_llm"],
    )
    embed_batch_field = ft.TextField(
        label="Embed batch size", value="16", width=160, dense=True,
        tooltip=TIP["embed_batch"],
    )
    num_batch_field = ft.TextField(
        label="LLM num_batch", value="1024", width=160, dense=True,
        tooltip=TIP["num_batch"],
    )
    num_ctx_field = ft.TextField(
        label="LLM num_ctx (0=auto)", value="0", width=180, dense=True,
        tooltip=TIP["num_ctx"],
    )
    agentic_sw = ft.Switch(label="Agentic mode", value=False,
                           active_color=ACCENT, tooltip=TIP["agentic"])

    # Agent provider config
    agent_provider_dd = ft.Dropdown(
        label="Agent provider", value="same", width=200, dense=True,
        options=[ft.dropdown.Option(o) for o in ("same", "openai", "anthropic")],
        tooltip=TIP["agent_provider"],
    )
    agent_model_field = ft.TextField(
        label="Agent model (blank = chat model)", value="", width=320, dense=True,
        tooltip=TIP["agent_model"],
    )
    agent_api_key_field = ft.TextField(
        label="API key", value="", width=320, dense=True, password=True,
        can_reveal_password=True, tooltip=TIP["agent_api_key"],
    )
    agent_base_url_field = ft.TextField(
        label="Base URL (OpenAI-compat)", value="https://api.openai.com/v1",
        width=400, dense=True, tooltip=TIP["agent_base_url"],
    )

    # Query modifiers
    query_prefix_field = ft.TextField(
        label="Query prefix", value="", width=440, dense=True,
        multiline=True, min_lines=1, max_lines=4,
        tooltip=TIP["query_prefix"],
    )
    query_suffix_field = ft.TextField(
        label="Query suffix", value="", width=440, dense=True,
        multiline=True, min_lines=1, max_lines=4,
        tooltip=TIP["query_suffix"],
    )
    query_negatives_field = ft.TextField(
        label="Negative traits (Avoid: …)", value="", width=440, dense=True,
        multiline=True, min_lines=1, max_lines=4,
        tooltip=TIP["query_negatives"],
    )
    use_file_metadata_sw = ft.Switch(
        label="Per-file metadata (.ezrag-meta.toml sidecars)",
        value=True,
        active_color=ACCENT,
        tooltip=(
            "When ON, ez-rag reads <filename>.ezrag-meta.toml sidecars "
            "alongside your source documents (or in <workspace>/.ezrag/"
            "file_meta/) and applies per-file query_prefix / query_suffix "
            "/ query_negatives on top of the workspace-level ones above. "
            "Three scope rules:\n"
            "  • global → applies to every query\n"
            "  • topic-aware → applies when the question mentions a "
            "discovered topic\n"
            "  • file-only → applies only when this file is top-1\n\n"
            "Auto-populate sidecars by running:\n"
            "  python -m ez_rag.cli scan <workspace>\n\n"
            "Hand-edit the .ezrag-meta.toml.draft files to taste, then "
            "rename to remove the .draft extension to activate."
        ),
    )
    enable_ocr = ft.Switch(label="OCR images / scanned PDFs", value=True,
                           active_color=ACCENT, tooltip=TIP["enable_ocr"])
    contextual = ft.Switch(label="Contextual Retrieval (slower ingest, better recall)",
                           value=False, active_color=ACCENT,
                           tooltip=TIP["contextual"])
    llm_inspect_pages_sw = ft.Switch(
        label="LLM inspect pages (very slow — drops garbled text)",
        value=False, active_color=ACCENT,
        tooltip=TIP["llm_inspect_pages"],
    )
    llm_correct_garbled_sw = ft.Switch(
        label="LLM correct questionable sections (slow — repairs OCR/partial text)",
        value=False, active_color=ACCENT,
        tooltip=TIP["llm_correct_garbled"],
    )
    preview_recoveries_sw = ft.Switch(
        label="Preview garbled-page recoveries during ingest",
        value=False, active_color=ACCENT,
        tooltip=TIP["preview_recoveries"],
    )

    # ---- model dropdowns -------------------------------------------------
    llm_model = ft.Dropdown(label="LLM model", value="qwen2.5:7b-instruct",
                            width=440, dense=True, options=[],
                            tooltip=TIP["llm_model"])
    ollama_url = ft.TextField(label="Ollama URL",
                              value="http://127.0.0.1:11434",
                              width=440, dense=True,
                              tooltip=TIP["ollama_url"])
    embed_provider = ft.Dropdown(
        label="Embedder provider", value="auto", width=200, dense=True,
        options=[
            ft.dropdown.Option("auto"),
            ft.dropdown.Option("ollama"),
            ft.dropdown.Option("fastembed"),
        ],
        tooltip=TIP["embed_provider"],
    )
    embed_model = ft.Dropdown(
        label="fastembed model", value="BAAI/bge-small-en-v1.5",
        width=440, dense=True,
        options=[ft.dropdown.Option(m) for m in FASTEMBED_MODELS],
        tooltip=TIP["embed_model"],
    )
    ollama_embed_model = ft.Dropdown(
        label="Ollama embed model", value="nomic-embed-text",
        width=440, dense=True, options=[],
        tooltip=TIP["ollama_embed"],
    )
    temperature = ft.TextField(label="Temperature", value="0.2",
                               width=120, dense=True,
                               tooltip=TIP["temperature"])
    max_tokens = ft.TextField(label="Max tokens", value="4096",
                              width=120, dense=True,
                              tooltip=TIP["max_tokens"])

    refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH, tooltip=TIP["refresh_models"],
    )
    browse_llm_btn = ft.OutlinedButton(
        "Browse Ollama library", icon=ft.Icons.CLOUD_DOWNLOAD,
        tooltip=TIP["browse_llm"],
    )
    gguf_btn = ft.OutlinedButton(
        "Use local GGUF...", icon=ft.Icons.FOLDER_OPEN,
        tooltip=TIP["use_gguf"],
    )
    browse_embed_btn = ft.OutlinedButton(
        "Browse Ollama library", icon=ft.Icons.CLOUD_DOWNLOAD,
        tooltip=TIP["browse_embed"],
    )

    # ---- helpers ----------------------------------------------------------

    def refresh_model_dropdowns():
        installed = list_ollama_models(ollama_url.value or state.cfg.llm_url)
        llm_model.options = _model_dropdown_options(
            installed, llm_model.value, embed_only=False,
        )
        ollama_embed_model.options = _model_dropdown_options(
            installed, ollama_embed_model.value, embed_only=True,
        )
        page.update()

    refresh_btn.on_click = lambda _: refresh_model_dropdowns()
    browse_llm_btn.on_click = lambda _: open_pull_dialog("llm", llm_model)
    browse_embed_btn.on_click = lambda _: open_pull_dialog("embed", ollama_embed_model)

    def open_gguf_picker(_):
        async def _pick():
            try:
                files = await gguf_picker.pick_files(
                    dialog_title="Pick a GGUF model file",
                    allowed_extensions=["gguf"],
                )
            except Exception as ex:
                _toast(page, f"file picker failed: {ex}")
                return
            if not files:
                return
            f = files[0]
            llm_model.value = f.path
            llm_model.options = _model_dropdown_options(
                list_ollama_models(ollama_url.value or state.cfg.llm_url),
                f.path, embed_only=False,
            )
            state.cfg.llm_provider = "llama-cpp"
            page.update()
            _toast(page, f"Will use local GGUF: {Path(f.path).name}\n"
                         f"(set llm_provider=llama-cpp; install with "
                         f"`pip install llama-cpp-python`)")
        page.run_task(_pick)
    gguf_btn.on_click = open_gguf_picker

    # ---- Pull-a-model overlay (NOT an AlertDialog) ------------------------
    # Flet 0.84's DialogControl machinery doesn't reliably propagate
    # dialog.open=False from a worker thread, so we use a plain visibility-
    # toggled Container in page.overlay instead. Open/close from any thread.

    def open_pull_dialog(kind: str, target_dropdown=None, *,
                         prefill_tag: str | None = None):
        """Show the Ollama library browser. `target_dropdown` is optional —
        when provided, picking a tag also sets that dropdown's value (used
        from the LLM / embedder model fields). `prefill_tag` lets callers
        skip the browse step entirely and go straight to a pull (used by
        the chat-error 'Re-pull this model' action).
        """
        capability_filter = "embedding" if kind == "embed" else None
        gpu_total_vram = detect_total_vram_gb()

        # NB: this TextField sits directly in the dialog's Column, so we
        # MUST NOT pass expand=True — that means "grab leftover vertical
        # space" inside a Column, which is what was making the search bar
        # stretch and shove the list off-screen.
        search_field = ft.TextField(
            hint_text="Search the Ollama library…",
            prefix_icon=ft.Icons.SEARCH,
            dense=True,
            border_color="#2A2D3D",
            focused_border_color=ACCENT,
            cursor_color=ACCENT,
            text_size=13,
        )
        tag_field = ft.TextField(
            label="Tag",
            hint_text="qwen2.5:7b-instruct  (auto-fills when you click a size below)",
            value=prefill_tag or "",
            expand=True, dense=True,
            border_color="#2A2D3D",
            focused_border_color=ACCENT,
            cursor_color=ACCENT,
        )
        progress = ft.ProgressBar(value=0, color=ACCENT,
                                  bgcolor="#262938", visible=False)
        progress_text = ft.Text("", size=12, color=ON_SURFACE_DIM)
        status_line = ft.Text("Loading library…", size=12, color=ON_SURFACE_DIM)

        # Capability filter chips
        cap_options = [
            ("all",       "All",       None),
            ("llms",      "LLMs",      "tools" if kind == "llm" else None),
            ("vision",    "Vision",    "vision"),
            ("thinking",  "Reasoning", "thinking"),
            ("embedding", "Embedding", "embedding"),
        ]
        if kind == "embed":
            cap_options = [("embedding", "Embedding", "embedding"),
                           ("all", "All", None)]

        active_cap_key = capability_filter or ("embedding" if kind == "embed" else None)

        def cap_chip(key: str, label: str, cap: str | None):
            is_active = (key == active_cap_key) or (
                key == "all" and active_cap_key is None
            )
            return ft.Container(
                key=f"cap:{key}",
                padding=ft.padding.symmetric(horizontal=12, vertical=6),
                bgcolor=ACCENT if is_active else SURFACE_DARK_HI,
                border=ft.border.all(1, ACCENT if is_active else "#2A2D3D"),
                border_radius=999,
                on_click=(lambda c=cap: lambda _: set_capability(c))(cap),
                content=ft.Text(
                    label, size=12,
                    color="#FFFFFF" if is_active else ON_SURFACE_DIM,
                    weight=ft.FontWeight.W_600,
                ),
            )

        cap_row = ft.Row(spacing=6, wrap=True)
        list_view = ft.ListView(
            spacing=6, height=420, padding=ft.padding.only(right=4),
        )

        library_state = {"models": [], "loaded": False}

        def render_caps():
            cap_row.controls = [
                cap_chip(k, lbl, c) for (k, lbl, c) in cap_options
            ]

        def cap_badges(caps: list[str]) -> list[ft.Control]:
            colors = {
                "tools":     "#5856D6",
                "vision":    "#3DDC84",
                "embedding": "#7C7BFF",
                "thinking":  "#F6B042",
                "audio":     "#F75A68",
                "cloud":     "#9097A6",
            }
            out = []
            for c in caps:
                color = colors.get(c, ON_SURFACE_DIM)
                out.append(ft.Container(
                    padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    bgcolor=ft.Colors.with_opacity(0.18, color),
                    border_radius=4,
                    content=ft.Text(c, size=10, color=color,
                                    weight=ft.FontWeight.W_700),
                ))
            return out

        def size_chip(name: str, sz: str):
            tag = f"{name}:{sz}"
            vram = estimate_vram_gb(sz)
            fit = vram_fit(vram, gpu_total_vram)
            # Colors per fit class
            border_color = "#2A2D3D"
            text_color = ACCENT
            if fit == "fits":
                border_color = SUCCESS
                text_color = SUCCESS
            elif fit == "tight":
                border_color = WARNING
                text_color = WARNING
            elif fit == "over":
                border_color = DANGER
                text_color = DANGER

            chip_text_parts = [
                ft.Text(sz, size=11, color=text_color,
                        weight=ft.FontWeight.W_700),
            ]
            if vram is not None:
                chip_text_parts.append(ft.Text(
                    f"  {fmt_vram_gb(vram)}", size=11,
                    color=ON_SURFACE_DIM, weight=ft.FontWeight.W_500,
                ))

            tooltip = f"Use {tag}"
            if vram is not None:
                tooltip += f"\n~{vram:.1f} GB VRAM @ Q4_K_M"
                if gpu_total_vram is not None:
                    tooltip += (
                        f"\nyour GPU: {gpu_total_vram:.1f} GB ({fit})"
                    )
            return ft.Container(
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                bgcolor="#1E2130",
                border=ft.border.all(1, border_color),
                border_radius=999,
                on_click=(lambda t=tag: lambda _: select_tag(t))(tag),
                content=ft.Row(chip_text_parts, tight=True, spacing=0),
                tooltip=tooltip,
            )

        def model_row(m: LibraryModel) -> ft.Control:
            top_row = [
                ft.Text(m.name, size=14, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE),
                *cap_badges(m.capabilities),
                ft.Container(expand=True),
                ft.Text(f"{m.pulls} pulls" if m.pulls else "",
                        size=11, color=ON_SURFACE_DIM),
            ]
            controls: list[ft.Control] = [
                ft.Row(top_row, spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ]
            if m.description:
                controls.append(ft.Text(m.description, size=12,
                                        color=ON_SURFACE_DIM, max_lines=2))
            if m.sizes:
                controls.append(ft.Row(
                    [size_chip(m.name, sz) for sz in m.sizes],
                    spacing=6, wrap=True,
                ))
            return ft.Container(
                padding=ft.padding.symmetric(horizontal=14, vertical=10),
                bgcolor=SURFACE_DARK_HI,
                border=ft.border.all(1, "#262938"),
                border_radius=10,
                on_click=(
                    None if m.sizes
                    else (lambda n=m.name: lambda _: select_tag(n))(m.name)
                ),
                content=ft.Column(controls, spacing=8, tight=True),
            )

        def render_list():
            list_view.controls.clear()
            if not library_state["loaded"]:
                list_view.controls.append(
                    ft.Container(
                        padding=20, alignment=ft.Alignment.CENTER,
                        content=ft.Row([
                            ft.ProgressRing(width=18, height=18, color=ACCENT),
                            ft.Text(status_line.value, color=ON_SURFACE_DIM),
                        ], alignment=ft.MainAxisAlignment.CENTER, spacing=8),
                    )
                )
                page.update()
                return
            models = search_library(
                library_state["models"], search_field.value or "",
                capability=active_cap_key if active_cap_key != "all" else None,
            )
            if not models:
                list_view.controls.append(ft.Text(
                    "No matches.", color=ON_SURFACE_DIM, italic=True,
                ))
            else:
                for m in models[:200]:
                    list_view.controls.append(model_row(m))
                if len(models) > 200:
                    list_view.controls.append(ft.Text(
                        f"…and {len(models) - 200} more (refine search)",
                        color=ON_SURFACE_DIM, italic=True, size=12,
                    ))
            page.update()

        def set_capability(cap: str | None):
            nonlocal active_cap_key
            active_cap_key = cap if cap is not None else "all"
            render_caps()
            render_list()

        def select_tag(t: str):
            tag_field.value = t
            page.update()

        def fetch_worker():
            try:
                models = fetch_ollama_library()
                library_state["models"] = models
                library_state["loaded"] = True
                base = f"{len(models)} models on ollama.com/library"
                if gpu_total_vram is not None:
                    base += (
                        f"  ·  GPU: {gpu_total_vram:.1f} GB"
                        "  ·  size chips colored: green = fits, "
                        "amber = tight, red = won't fit"
                    )
                else:
                    base += "  ·  no NVIDIA GPU detected (estimates shown anyway)"
                status_line.value = base
            except Exception as e:
                library_state["loaded"] = True
                status_line.value = f"Couldn't fetch library ({e}). " \
                                    "Type a tag below to pull anyway."
            render_list()

        def on_search_change(_):
            render_list()

        search_field.on_change = on_search_change

        # Buttons
        pull_btn = ft.FilledButton(
            "Pull", icon=ft.Icons.DOWNLOAD,
            bgcolor=ACCENT, color="#FFFFFF",
        )

        # Defined later after `overlay` is built — captured by closure below.
        def close_dialog(_=None):
            overlay.visible = False
            # Remove from page overlay so reopening doesn't stack zombies.
            try:
                if overlay in page.overlay:
                    page.overlay.remove(overlay)
            except Exception:
                pass
            page.update()

        cancel_btn = ft.TextButton("Close", on_click=close_dialog)

        render_caps()

        # Header row of the modal panel
        title_row = ft.Row([
            ft.Icon(ft.Icons.CLOUD_DOWNLOAD, color=ACCENT),
            ft.Text(
                f"Browse Ollama models — pull a "
                f"{'embedding' if kind == 'embed' else 'LLM'}",
                weight=ft.FontWeight.W_700, size=16, color=ON_SURFACE,
            ),
            ft.Container(expand=True),
            ft.IconButton(icon=ft.Icons.CLOSE, on_click=close_dialog,
                          tooltip="Close"),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

        modal_panel = ft.Container(
            bgcolor=SURFACE_DARK,
            border=ft.border.all(1, "#262938"),
            border_radius=14,
            padding=20,
            width=860,
            content=ft.Column([
                title_row,
                ft.Divider(color="#262938"),
                search_field,
                cap_row,
                status_line,
                ft.Container(
                    content=list_view,
                    bgcolor=BG_DARK,
                    border=ft.border.all(1, "#222637"),
                    border_radius=10,
                    padding=8,
                ),
                ft.Divider(color="#262938"),
                ft.Row([tag_field, pull_btn], spacing=10,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                progress,
                # Heartbeat lives next to the progress text so the modal's
                # frame loop stays alive across multi-minute downloads —
                # otherwise the % / MB/s / ETA values only repaint when
                # the user clicks somewhere.
                ft.Row([modal_heartbeat(), progress_text],
                       spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([ft.Container(expand=True), cancel_btn],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=10, tight=True),
        )

        # Full-screen scrim with the panel centered. Clicking the scrim closes.
        overlay = ft.Container(
            expand=True,
            visible=True,
            bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
            alignment=ft.Alignment.CENTER,
            on_click=close_dialog,
            content=ft.Container(
                # Inner wrapper: catches click events so clicking inside the
                # panel doesn't bubble to the scrim's on_click (and close it).
                on_click=lambda e: None,
                content=modal_panel,
            ),
        )

        def do_pull(_):
            tag = (tag_field.value or "").strip()
            if not tag:
                _toast(page, "Click a size chip or type a tag first")
                return
            if tag in state.active_pulls:
                _toast(page,
                    f"{tag} is already being pulled — close this dialog "
                    "and check the header badge for progress.")
                return
            pull_btn.disabled = True
            cancel_btn.disabled = True
            search_field.disabled = True
            progress.visible = True
            progress.value = None
            progress_text.value = "Starting…"
            # Register the pull at app-level state so it survives if the
            # dialog is dismissed mid-pull. The header badge polls this.
            state.active_pulls[tag] = {
                "completed": 0, "total": 0, "status": "starting",
                "pct": 0.0, "started_at": time.monotonic(),
            }
            page.update()

            def safe_update(**fields):
                """Update dialog widgets but never raise if they're already
                detached (because the user closed the dialog)."""
                try:
                    if "progress" in fields:
                        progress.value = fields["progress"]
                    if "text" in fields:
                        progress_text.value = fields["text"]
                    page.update()
                except Exception:
                    pass

            def worker():
                start_t = time.monotonic()
                last_t = start_t
                last_completed = 0
                rates: list[float] = []  # rolling window of bytes/sec
                last_paint_t = 0.0
                try:
                    for evt in pull_ollama_model(
                        ollama_url.value or state.cfg.llm_url, tag,
                    ):
                        status = evt.get("status", "")
                        total = int(evt.get("total") or 0)
                        completed = int(evt.get("completed") or 0)
                        now = time.monotonic()

                        # Update rolling rate
                        dt = now - last_t
                        if dt > 0 and completed > last_completed:
                            rates.append((completed - last_completed) / dt)
                            if len(rates) > 12:
                                rates.pop(0)
                        avg_rate = sum(rates) / len(rates) if rates else 0.0
                        last_t = now
                        last_completed = completed

                        # Mirror to app-level state so the header badge +
                        # any reopened dialog can read live progress.
                        pct = (completed / total) if total else None
                        state.active_pulls[tag] = {
                            "completed": completed,
                            "total": total,
                            "status": status,
                            "pct": pct,
                            "rate_bps": avg_rate,
                            "started_at": start_t,
                        }

                        if total and completed:
                            done_g, tot_g = completed / 1e9, total / 1e9
                            rate_str = (
                                f"{avg_rate / 1e6:.1f} MB/s"
                                if avg_rate >= 1_000_000
                                else f"{avg_rate / 1e3:.0f} KB/s" if avg_rate > 0
                                else "—"
                            )
                            if avg_rate > 0:
                                eta_s = max(0, (total - completed) / avg_rate)
                                if eta_s >= 3600:
                                    eta_str = f"{int(eta_s/3600)}h{int((eta_s%3600)/60):02d}m"
                                else:
                                    eta_str = f"{int(eta_s/60)}:{int(eta_s%60):02d}"
                            else:
                                eta_str = "—"
                            text = (
                                f"{status}  ·  {pct*100:.1f}%  ·  "
                                f"{done_g:.2f} / {tot_g:.2f} GB  ·  "
                                f"{rate_str}  ·  ETA {eta_str}"
                            )
                            if now - last_paint_t > 0.1 or status == "success":
                                safe_update(progress=min(1.0, pct), text=text)
                                last_paint_t = now
                        else:
                            if now - last_paint_t > 0.5 or status == "success":
                                safe_update(
                                    progress=None if status != "success" else 1.0,
                                    text=status,
                                )
                                last_paint_t = now

                        if status == "success":
                            break

                    elapsed = time.monotonic() - start_t
                    elapsed_str = (f"{elapsed:.1f}s" if elapsed < 60
                                   else f"{int(elapsed//60)}m{int(elapsed%60):02d}s")
                    safe_update(progress=1.0,
                                text=f"✓ Pulled {tag} in {elapsed_str}")
                    # Always toast on success — works whether the dialog is
                    # still open or was dismissed mid-download.
                    _toast(page, f"✓ Downloaded {tag} ({elapsed_str})")
                    try:
                        refresh_model_dropdowns()
                        if target_dropdown is not None:
                            target_dropdown.value = _norm_tag(tag)
                        page.update()
                    except Exception:
                        pass
                    try:
                        close_dialog()
                    except Exception:
                        pass
                except Exception as e:
                    safe_update(progress=0, text=f"Error: {e}")
                    _toast(page, f"✗ Pull failed for {tag}: {e}")
                finally:
                    # Re-enable controls (no-op if dialog closed) and remove
                    # this pull from active state so the badge clears.
                    try:
                        pull_btn.disabled = False
                        cancel_btn.disabled = False
                        search_field.disabled = False
                        page.update()
                    except Exception:
                        pass
                    state.active_pulls.pop(tag, None)

            page.run_thread(worker)

        pull_btn.on_click = do_pull
        # Add the overlay to the page; it's visible immediately.
        page.overlay.append(overlay)
        page.update()
        # Show loading state immediately, then kick off the fetch.
        render_list()
        page.run_thread(fetch_worker)

    # GGUF picker result is handled inline via page.run_task in
    # open_gguf_picker (above) — Flet 0.84 file pickers are async.
    # Dropdowns no longer trigger pull/picker actions; explicit buttons do.

    def load_from_cfg():
        c = state.cfg
        chunk_size.value = str(c.chunk_size)
        chunk_overlap.value = str(c.chunk_overlap)
        top_k.value = str(c.top_k)
        hybrid.value = c.hybrid
        rerank.value = c.rerank
        use_hyde_sw.value = c.use_hyde
        multi_query_sw.value = c.multi_query
        context_window_field.value = str(c.context_window)
        use_mmr_sw.value = c.use_mmr
        mmr_lambda_field.value = str(c.mmr_lambda)
        expand_chapter_sw.value = getattr(c, "expand_to_chapter", False)
        chapter_max_chars_field.value = str(getattr(c, "chapter_max_chars", 16000))
        agentic_sw.value = c.agentic
        agent_provider_dd.value = c.agent_provider
        agent_model_field.value = c.agent_model
        agent_api_key_field.value = c.agent_api_key
        agent_base_url_field.value = c.agent_base_url
        query_prefix_field.value = c.query_prefix
        query_suffix_field.value = c.query_suffix
        query_negatives_field.value = c.query_negatives
        use_file_metadata_sw.value = getattr(c, "use_file_metadata", True)
        unload_llm_sw.value = c.unload_llm_during_ingest
        embed_batch_field.value = str(c.embed_batch_size)
        num_batch_field.value = str(getattr(c, "num_batch", 1024))
        num_ctx_field.value = str(getattr(c, "num_ctx", 0))
        use_corpus.value = c.use_rag
        enable_ocr.value = c.enable_ocr
        contextual.value = c.enable_contextual
        llm_inspect_pages_sw.value = getattr(c, "llm_inspect_pages", False)
        llm_correct_garbled_sw.value = getattr(c, "llm_correct_garbled", False)
        preview_recoveries_sw.value = getattr(c, "preview_garbled_recoveries", False)
        ollama_url.value = c.llm_url
        embed_provider.value = c.embedder_provider
        embed_model.value = c.embedder_model
        temperature.value = str(c.temperature)
        max_tokens.value = str(c.max_tokens)

        # Populate dropdowns from currently-installed Ollama models
        installed = list_ollama_models(c.llm_url)
        llm_model.options = _model_dropdown_options(
            installed, c.llm_model, embed_only=False,
        )
        llm_model.value = _norm_tag(c.llm_model)
        ollama_embed_model.options = _model_dropdown_options(
            installed, c.ollama_embed_model, embed_only=True,
        )
        ollama_embed_model.value = _norm_tag(c.ollama_embed_model)

        # ---- Multi-GPU / routing-table refresh ----
        # Reload the routing table for the current workspace, probe
        # hardware, adopt any leftover managed daemons, and re-render
        # the Hardware card. set_workspace_path already installed the
        # raw table; this populates the UI.
        try:
            hw_load_from_workspace()
            hw_rescan_gpus()
            hw_render()
            # Phase 5: paint an initial /api/ps snapshot so the user
            # doesn't see an empty "Live placement" section for the
            # first 5 s, then start the polling ticker.
            try:
                initial_snapshot: dict[int, list] = {}
                for d in hw_state["table"].daemons:
                    initial_snapshot[d.gpu_index] = query_loaded_models(
                        d.url, timeout=2.0,
                    )
                hw_render_live_placement(initial_snapshot)
            except Exception:
                pass
            start_live_placement()
            start_health_watchdog()
        except Exception as ex:
            # Hardware card is purely additive — never let a probe
            # failure block Settings load.
            try:
                _toast(state.page,
                       f"Hardware probe error (continuing): {ex}")
            except Exception:
                pass

        page.update()

    def save(_):
        if state.ws is None:
            _toast(state.page, "Open a workspace first")
            return
        try:
            c = state.cfg
            # Snapshot the model fields BEFORE we mutate them so we can
            # detect a model switch and evict the old ones from VRAM. If
            # we don't do this Ollama often returns 'unable to load model'
            # on the first chat after a switch — the new model can't fit
            # because the old one is still resident.
            old_llm_url = c.llm_url
            old_llm_model = c.llm_model
            old_ollama_embed = c.ollama_embed_model
            old_embedder_provider = c.embedder_provider

            c.chunk_size = int(chunk_size.value)
            c.chunk_overlap = int(chunk_overlap.value)
            c.top_k = int(top_k.value)
            c.hybrid = bool(hybrid.value)
            c.rerank = bool(rerank.value)
            c.use_hyde = bool(use_hyde_sw.value)
            c.multi_query = bool(multi_query_sw.value)
            c.context_window = max(0, int(context_window_field.value or 0))
            c.use_mmr = bool(use_mmr_sw.value)
            try:
                c.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda_field.value or 0.5)))
            except ValueError:
                pass
            c.expand_to_chapter = bool(expand_chapter_sw.value)
            try:
                c.chapter_max_chars = max(
                    1000, int(chapter_max_chars_field.value or 16000)
                )
            except ValueError:
                pass
            c.agentic = bool(agentic_sw.value)
            c.agent_provider = agent_provider_dd.value or "same"
            c.agent_model = (agent_model_field.value or "").strip()
            c.agent_api_key = (agent_api_key_field.value or "").strip()
            c.agent_base_url = (agent_base_url_field.value or "").strip()
            c.query_prefix = query_prefix_field.value or ""
            c.query_suffix = query_suffix_field.value or ""
            c.query_negatives = query_negatives_field.value or ""
            c.use_file_metadata = bool(use_file_metadata_sw.value)
            c.unload_llm_during_ingest = bool(unload_llm_sw.value)
            try:
                c.embed_batch_size = max(1, int(embed_batch_field.value or 16))
            except ValueError:
                pass
            try:
                c.num_batch = max(64, int(num_batch_field.value or 1024))
            except ValueError:
                pass
            try:
                c.num_ctx = max(0, int(num_ctx_field.value or 0))
            except ValueError:
                pass
            c.use_rag = bool(use_corpus.value)
            c.enable_ocr = bool(enable_ocr.value)
            c.enable_contextual = bool(contextual.value)
            c.llm_inspect_pages = bool(llm_inspect_pages_sw.value)
            c.llm_correct_garbled = bool(llm_correct_garbled_sw.value)
            c.preview_garbled_recoveries = bool(preview_recoveries_sw.value)
            c.llm_model = llm_model.value or c.llm_model
            c.llm_url = ollama_url.value
            c.embedder_provider = embed_provider.value
            c.embedder_model = embed_model.value
            c.ollama_embed_model = ollama_embed_model.value or c.ollama_embed_model
            c.temperature = float(temperature.value)
            c.max_tokens = int(max_tokens.value)
            # If the chosen LLM looks like a path, switch backend
            if c.llm_model and (("/" in c.llm_model) or ("\\" in c.llm_model)):
                c.llm_provider = "llama-cpp"
            elif c.llm_provider == "llama-cpp":
                c.llm_provider = "auto"
            c.save(state.ws.config_path)
            clear_embedder_cache()

            # ----- VRAM hygiene on model switch -----
            # Build the set of "models we still want resident" so we don't
            # accidentally evict the new selections. Then unload anything
            # else, which catches stale chat models and old embedders.
            llm_changed = (c.llm_model != old_llm_model
                           or c.llm_url != old_llm_url)
            embedder_changed = (
                c.ollama_embed_model != old_ollama_embed
                or c.embedder_provider != old_embedder_provider
            )

            unloaded: list[str] = []
            if llm_changed or embedder_changed:
                # Run on a thread so a slow Ollama doesn't freeze the GUI.
                def _evict():
                    keep: set[str] = set()
                    # Keep the new selections, in case Ollama autostarts them.
                    if c.llm_model:
                        keep.add(c.llm_model)
                    if c.embedder_provider == "ollama" and c.ollama_embed_model:
                        keep.add(c.ollama_embed_model)
                    try:
                        out = unload_running_models(c.llm_url, except_=keep)
                    except Exception:
                        out = []
                    # Also explicitly unload the OLD specific tags in case
                    # the user is on a different Ollama URL or /api/ps
                    # didn't return them.
                    for old_tag in (old_llm_model, old_ollama_embed):
                        if old_tag and old_tag not in keep and old_tag not in out:
                            try:
                                if unload_ollama_model(old_llm_url, old_tag):
                                    out.append(old_tag)
                            except Exception:
                                pass
                    if out:
                        _toast(state.page,
                               f"Unloaded from VRAM: {', '.join(out)}")
                state.page.run_thread(_evict)

            _toast(state.page, "Settings saved")
            refresh_status()
            # NB: don't call render_workflow_chips() here — it's defined
            # inside build_files_view's closure and isn't visible from
            # the Settings save handler. The Files-tab and chat-tab
            # workflow chip strips re-render on workspace open, ingest
            # start, and chat send, so the user will see updated chips
            # next time they hit one of those — no need to wire a
            # cross-view callback for instant refresh.
            state.page.update()
        except ValueError as ex:
            _toast(state.page, f"Bad value: {ex}")

    # ========================================================================
    # HARDWARE / GPU ROUTING card (Phases 4 + 5 of multi-GPU plan)
    # ========================================================================
    # State held in a closure dict so spawn/stop handlers can mutate it
    # and the re-render reads from one source of truth.
    hw_state: dict = {
        "table": RoutingTable(),
        "supervisor": DaemonSupervisor(),
        "detected_gpus": [],            # list[DetectedGpu]
        "external": None,                # ExternalDetection or None
    }

    # ----- Containers we'll re-render into -----
    hw_gpu_list = ft.Column(spacing=4, tight=True)
    hw_daemon_list = ft.Column(spacing=4, tight=True)
    hw_assignment_list = ft.Column(spacing=4, tight=True)
    hw_status_text = ft.Text("", size=11, color=ON_SURFACE_DIM, italic=True)

    hw_spawn_managed_sw = ft.Switch(
        label="Spawn managed daemons for additional GPUs",
        value=True,
        active_color=ACCENT,
        tooltip=("When ON, ez-rag spawns one ollama serve per GPU you "
                 "tick, each pinned to that GPU via CUDA_VISIBLE_DEVICES. "
                 "Per-model GPU pinning requires this. Mutually exclusive "
                 "with OLLAMA_SCHED_SPREAD mode."),
    )
    hw_sched_spread_sw = ft.Switch(
        label="Use OLLAMA_SCHED_SPREAD across all GPUs (single-daemon mode)",
        value=False,
        active_color=ACCENT,
        tooltip=("When ON, ez-rag sets OLLAMA_SCHED_SPREAD=1 on the "
                 "single external daemon so it splits model layers "
                 "across all visible GPUs by free VRAM. No per-model "
                 "pinning. Mutually exclusive with managed-spawn mode."),
    )

    def _ws_root_or_none():
        return state.ws.root if state.ws is not None else None

    def hw_load_from_workspace():
        """Read the routing table off disk for the current workspace
        and snapshot it into hw_state['table']."""
        ws_root = _ws_root_or_none()
        if ws_root is None:
            hw_state["table"] = RoutingTable()
            return
        try:
            hw_state["table"] = load_routing_table(ws_root)
        except Exception:
            hw_state["table"] = RoutingTable()

    def hw_save_to_workspace():
        """Persist the in-memory table + activate it for the resolver."""
        ws_root = _ws_root_or_none()
        if ws_root is None:
            return
        try:
            save_routing_table(ws_root, hw_state["table"])
            set_active_table(hw_state["table"])
        except Exception as ex:
            _toast(state.page, f"Saving routing table failed: {ex}")

    def hw_rescan_gpus():
        """Probe hardware + adopt any leftover daemons from a previous
        run. Updates hw_state in place."""
        try:
            hw_state["detected_gpus"] = detect_gpus()
        except Exception:
            hw_state["detected_gpus"] = []
        try:
            external_url = (state.cfg.llm_url
                            if state.cfg else "http://127.0.0.1:11434")
            hw_state["external"] = detect_external(external_url)
        except Exception:
            hw_state["external"] = None
        # Adopt previously-spawned managed daemons (if our PID files
        # are still pointing at live processes).
        try:
            adopted = hw_state["supervisor"].adopt_previous()
            for d in adopted:
                hw_state["table"].upsert_daemon(d)
        except Exception:
            pass
        # Make sure the external daemon has a slot in the table at GPU 0
        # so single-GPU users see at least one daemon listed.
        ext = hw_state["external"]
        if ext and ext.reachable:
            existing = hw_state["table"].daemon_for_gpu(0)
            if existing is None:
                # Use the first detected GPU's name if we have it.
                gpus = hw_state["detected_gpus"]
                gpu_name = gpus[0].name if gpus else ""
                vram = gpus[0].vram_total_mb if gpus else 0
                hw_state["table"].upsert_daemon(GpuDaemon(
                    gpu_index=0, gpu_name=gpu_name,
                    vram_total_mb=int(vram or 0),
                    url=ext.url, pid=None, managed=False,
                    notes="external daemon (auto-detected)",
                ))
        hw_save_to_workspace()

    def _gpu_row_label(gpu: DetectedGpu) -> str:
        vram_gb = (gpu.vram_total_mb or 0) / 1024.0
        return (f"GPU {gpu.index} · {gpu.name or gpu.vendor.upper()} · "
                f"{vram_gb:.0f} GB · {gpu.runtime}")

    def _make_spawn_btn(gpu_index: int, gpu_name: str,
                         vram_total_mb: int, disabled: bool):
        def _click(_e=None):
            def _bg():
                try:
                    daemon = hw_state["supervisor"].ensure_running(
                        gpu_index=gpu_index, gpu_name=gpu_name,
                        vram_total_mb=vram_total_mb,
                    )
                    hw_state["table"].upsert_daemon(daemon)
                    hw_save_to_workspace()
                    _toast(state.page,
                           f"Spawned daemon for GPU {gpu_index} at "
                           f"{daemon.url}")
                except SpawnError as ex:
                    _toast(state.page, f"Spawn failed: {ex}")
                except Exception as ex:
                    _toast(state.page, f"Spawn failed: {ex}")
                hw_render()
            state.page.run_thread(_bg)
        return ft.OutlinedButton(
            "Spawn daemon", icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
            on_click=_click, disabled=disabled,
            tooltip=("Start a new ollama serve pinned to this GPU on the "
                     "next free port. Models pinned to this GPU will route "
                     "to it.") if not disabled else (
                "Already running" if disabled else ""
            ),
        )

    def _make_stop_btn(gpu_index: int):
        def _click(_e=None):
            def _bg():
                try:
                    hw_state["supervisor"].shutdown(gpu_index)
                    hw_state["table"].remove_daemon(gpu_index)
                    hw_save_to_workspace()
                    _toast(state.page,
                           f"Stopped daemon for GPU {gpu_index}")
                except Exception as ex:
                    _toast(state.page, f"Stop failed: {ex}")
                hw_render()
            state.page.run_thread(_bg)
        return ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE_OUTLINED, icon_color=DANGER,
            on_click=_click,
            tooltip="Stop this managed daemon",
        )

    def hw_render():
        """Re-render the Hardware card from hw_state."""
        # ----- Detected GPUs -----
        hw_gpu_list.controls.clear()
        gpus = hw_state["detected_gpus"]
        if not gpus:
            hw_gpu_list.controls.append(ft.Text(
                "No compatible GPU detected — ez-rag will run in "
                "CPU mode.",
                size=12, color=ON_SURFACE_DIM, italic=True,
            ))
        for g in gpus:
            daemon_for_g = hw_state["table"].daemon_for_gpu(g.index)
            has_daemon = daemon_for_g is not None
            spawn_disabled = (
                has_daemon
                or not bool(hw_spawn_managed_sw.value)
                or g.index == 0   # GPU 0 is the external daemon's slot
            )
            spawn_btn = _make_spawn_btn(
                g.index, g.name or "", int(g.vram_total_mb or 0),
                spawn_disabled,
            )
            indicator = ft.Container(
                width=8, height=8, border_radius=999,
                bgcolor=(SUCCESS if g.is_compatible else DANGER),
                margin=ft.margin.only(right=8, top=6),
            )
            row = ft.Row(
                [
                    indicator,
                    ft.Text(_gpu_row_label(g), size=12,
                             color=ON_SURFACE,
                             expand=True),
                    spawn_btn,
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            )
            hw_gpu_list.controls.append(row)

        # ----- Daemons -----
        hw_daemon_list.controls.clear()
        if not hw_state["table"].daemons:
            hw_daemon_list.controls.append(ft.Text(
                "No daemons registered yet. Click 'Re-scan' to detect "
                "your existing ollama serve, or 'Spawn daemon' for a "
                "GPU above.",
                size=12, color=ON_SURFACE_DIM, italic=True,
            ))
        for d in sorted(hw_state["table"].daemons,
                         key=lambda x: x.gpu_index):
            tag = "managed" if d.managed else "external"
            tag_color = ACCENT if d.managed else SUCCESS
            line_left = ft.Row([
                ft.Container(
                    content=ft.Text(tag, size=10,
                                     color="#FFFFFF",
                                     weight=ft.FontWeight.W_700),
                    bgcolor=tag_color,
                    padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    border_radius=999,
                ),
                ft.Text(
                    f"GPU {d.gpu_index}: {d.gpu_name or '?'} · "
                    f"{d.url}"
                    + (f" · pid {d.pid}" if d.pid else ""),
                    size=12, color=ON_SURFACE,
                ),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)
            row_controls: list[ft.Control] = [
                ft.Container(content=line_left, expand=True),
            ]
            if d.managed:
                row_controls.append(_make_stop_btn(d.gpu_index))
            hw_daemon_list.controls.append(ft.Row(
                row_controls,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ))

        # ----- Per-model assignment table (Phase 5) -----
        hw_render_assignments()

        # ----- Toggle states from table -----
        hw_spawn_managed_sw.value = bool(
            hw_state["table"].spawn_managed_daemons
        )
        hw_sched_spread_sw.value = bool(
            hw_state["table"].use_sched_spread
        )

        # ----- Status line -----
        ext = hw_state["external"]
        if ext is None or not ext.reachable:
            hw_status_text.value = (
                "External Ollama: NOT detected. ez-rag will fall back "
                "to localhost:11434 by default."
            )
        else:
            hw_status_text.value = (
                f"External Ollama: {ext.url} (version {ext.version})"
            )

        try:
            state.page.update()
        except Exception:
            pass

    # ----- Per-model assignment table -----

    def hw_render_assignments():
        """Render the per-model GPU assignment rows (Phase 5)."""
        hw_assignment_list.controls.clear()
        gpus = hw_state["detected_gpus"]
        if not gpus or len(gpus) < 2:
            hw_assignment_list.controls.append(ft.Text(
                "Per-model GPU assignment becomes useful with 2+ GPUs. "
                "Until then, every model uses the default GPU.",
                size=12, color=ON_SURFACE_DIM, italic=True,
            ))
            return

        # Build the dropdown options (one entry per registered daemon
        # plus the auto sentinel and the default-GPU option).
        gpu_options: list[ft.dropdown.Option] = [
            ft.dropdown.Option(
                key=str(GPU_INDEX_AUTO), text="auto (pick at runtime)",
            ),
        ]
        for g in gpus:
            label = (f"GPU {g.index} · "
                     f"{g.name or g.vendor.upper()} · "
                     f"{(g.vram_total_mb or 0) / 1024:.0f} GB")
            gpu_options.append(ft.dropdown.Option(
                key=str(g.index), text=label,
            ))

        # Header row
        hw_assignment_list.controls.append(ft.Row([
            ft.Text("MODEL", size=10, color=ON_SURFACE_DIM,
                    weight=ft.FontWeight.W_700, expand=2),
            ft.Text("ROLE", size=10, color=ON_SURFACE_DIM,
                    weight=ft.FontWeight.W_700, width=80),
            ft.Text("GPU", size=10, color=ON_SURFACE_DIM,
                    weight=ft.FontWeight.W_700, expand=2),
            ft.Container(width=40),    # for the remove button
        ], spacing=8))

        # Each existing assignment
        for a in list(hw_state["table"].assignments):
            def _on_change(model_tag, role):
                def handler(e):
                    try:
                        new_idx = int(e.control.value)
                    except (TypeError, ValueError):
                        return
                    hw_state["table"].upsert_assignment(
                        model_tag, new_idx, role=role,
                    )
                    hw_save_to_workspace()
                return handler

            def _on_remove(model_tag, role):
                def handler(_e=None):
                    hw_state["table"].remove_assignment(
                        model_tag, role=role,
                    )
                    hw_save_to_workspace()
                    hw_render()
                return handler

            dd = ft.Dropdown(
                value=str(a.gpu_index),
                options=gpu_options,
                expand=2, dense=True,
                on_change=_on_change(a.model_tag, a.role),
            )
            hw_assignment_list.controls.append(ft.Row([
                ft.Text(a.model_tag, size=12, color=ON_SURFACE,
                         expand=2,
                         font_family="monospace"),
                ft.Text(a.role, size=11, color=ON_SURFACE_DIM, width=80),
                dd,
                ft.IconButton(
                    icon=ft.Icons.CLOSE, icon_size=14,
                    icon_color=DANGER,
                    tooltip="Remove this assignment",
                    on_click=_on_remove(a.model_tag, a.role),
                ),
            ], spacing=8,
              vertical_alignment=ft.CrossAxisAlignment.CENTER))

        # "Add" row — model field + role dropdown + GPU dropdown + + button
        new_model_field = ft.TextField(
            hint_text="model tag (e.g. qwen2.5:14b)",
            dense=True, expand=2, text_size=12,
        )
        new_role_dd = ft.Dropdown(
            value="any",
            options=[
                ft.dropdown.Option("any"),
                ft.dropdown.Option("chat"),
                ft.dropdown.Option("embed"),
            ],
            width=80, dense=True,
        )
        new_gpu_dd = ft.Dropdown(
            value=str(GPU_INDEX_AUTO),
            options=gpu_options,
            expand=2, dense=True,
        )
        def _on_add(_e=None):
            tag = (new_model_field.value or "").strip()
            if not tag:
                _toast(state.page, "Enter a model tag")
                return
            try:
                gpu_idx = int(new_gpu_dd.value)
            except (TypeError, ValueError):
                gpu_idx = GPU_INDEX_AUTO
            hw_state["table"].upsert_assignment(
                tag, gpu_idx, role=new_role_dd.value or "any",
            )
            hw_save_to_workspace()
            hw_render()
        hw_assignment_list.controls.append(ft.Row([
            new_model_field, new_role_dd, new_gpu_dd,
            ft.IconButton(
                icon=ft.Icons.ADD, icon_color=ACCENT,
                tooltip="Add assignment",
                on_click=_on_add,
            ),
        ], spacing=8,
          vertical_alignment=ft.CrossAxisAlignment.CENTER))

    # ----- Wire toggle handlers -----
    def _on_spawn_managed_change(_e=None):
        # Mutually exclusive with sched-spread
        if hw_spawn_managed_sw.value and hw_sched_spread_sw.value:
            hw_sched_spread_sw.value = False
        hw_state["table"].spawn_managed_daemons = bool(
            hw_spawn_managed_sw.value
        )
        hw_state["table"].use_sched_spread = bool(
            hw_sched_spread_sw.value
        )
        hw_save_to_workspace()
        hw_render()

    def _on_sched_spread_change(_e=None):
        if hw_sched_spread_sw.value and hw_spawn_managed_sw.value:
            hw_spawn_managed_sw.value = False
        hw_state["table"].spawn_managed_daemons = bool(
            hw_spawn_managed_sw.value
        )
        hw_state["table"].use_sched_spread = bool(
            hw_sched_spread_sw.value
        )
        hw_save_to_workspace()
        hw_render()

    hw_spawn_managed_sw.on_change = _on_spawn_managed_change
    hw_sched_spread_sw.on_change = _on_sched_spread_change

    rescan_btn = ft.OutlinedButton(
        "Re-scan hardware", icon=ft.Icons.REFRESH,
        on_click=lambda _e=None: (hw_rescan_gpus(), hw_render()),
        tooltip=("Probe nvidia-smi / rocm-smi / xpu-smi again, look "
                 "for an external ollama daemon, and adopt any managed "
                 "daemons left over from a previous ez-rag run."),
    )

    # ----- Live placement panel (Phase 5) -----
    # Polls /api/ps on each registered daemon every ~5 s while
    # Settings is visible. Tells the user which models are currently
    # resident on which GPU, with VRAM use + expiry.
    hw_live_placement = ft.Column(spacing=4, tight=True)
    hw_live_placement_status = ft.Text(
        "", size=11, color=ON_SURFACE_DIM, italic=True,
    )
    hw_live_state: dict = {
        "polling": False,    # set True while the ticker is running
        "last_update": 0.0,
    }

    def _fmt_bytes(n: int) -> str:
        if not n:
            return "0 B"
        for unit, thresh in (("GB", 1 << 30), ("MB", 1 << 20),
                             ("KB", 1 << 10)):
            if n >= thresh:
                return f"{n / thresh:.1f} {unit}"
        return f"{n} B"

    def _fmt_expires(iso_ts: str) -> str:
        """Return a human-readable 'expires in Xm' from an ISO ts.
        Empty string = no keep-alive timer."""
        if not iso_ts:
            return ""
        from datetime import datetime, timezone
        try:
            # Ollama may emit either a timezone-aware or naive UTC string.
            ts = iso_ts.replace("Z", "+00:00")
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            secs = int((t - now).total_seconds())
            if secs <= 0:
                return "expiring"
            if secs < 60:
                return f"{secs}s left"
            if secs < 3600:
                return f"{secs // 60}m left"
            return f"{secs // 3600}h{(secs % 3600) // 60:02d}m left"
        except Exception:
            return ""

    def hw_render_live_placement(snapshot: dict[int, list]):
        """Re-render the live placement column from a snapshot:
            { gpu_index: [LoadedModel, …], … }
        """
        hw_live_placement.controls.clear()
        if not hw_state["table"].daemons:
            hw_live_placement.controls.append(ft.Text(
                "(no daemons registered yet — re-scan to detect)",
                size=12, color=ON_SURFACE_DIM, italic=True,
            ))
            return

        any_loaded = False
        for d in sorted(hw_state["table"].daemons,
                         key=lambda x: x.gpu_index):
            models = snapshot.get(d.gpu_index, [])
            header = ft.Row([
                ft.Icon(ft.Icons.MEMORY, size=14, color=ACCENT),
                ft.Text(
                    f"GPU {d.gpu_index} · {d.gpu_name or '?'} · {d.url}",
                    size=12, color=ON_SURFACE,
                    weight=ft.FontWeight.W_700,
                ),
                ft.Container(expand=True),
                ft.Text(
                    f"{len(models)} model(s) loaded" if models
                    else "idle",
                    size=11,
                    color=(SUCCESS if models else ON_SURFACE_DIM),
                ),
            ], spacing=6,
              vertical_alignment=ft.CrossAxisAlignment.CENTER)
            hw_live_placement.controls.append(header)

            if not models:
                hw_live_placement.controls.append(ft.Container(
                    margin=ft.margin.only(left=20),
                    content=ft.Text(
                        "no models resident on this daemon",
                        size=11, color=ON_SURFACE_DIM, italic=True,
                    ),
                ))
                continue

            any_loaded = True
            for m in models:
                vram_pct = (
                    (m.size_vram_bytes / m.size_bytes * 100)
                    if m.size_bytes else 0
                )
                where = "GPU" if m.size_vram_bytes > 0 else "CPU"
                expires = _fmt_expires(m.expires_at)
                line = ft.Row([
                    ft.Container(width=12),   # indent
                    ft.Text(
                        m.name, size=12, color=ON_SURFACE,
                        font_family="monospace", expand=2,
                    ),
                    ft.Container(
                        content=ft.Text(where, size=10,
                                         color="#FFFFFF",
                                         weight=ft.FontWeight.W_700),
                        bgcolor=(SUCCESS if where == "GPU" else WARNING),
                        padding=ft.padding.symmetric(
                            horizontal=6, vertical=1,
                        ),
                        border_radius=999,
                    ),
                    ft.Text(
                        f"{_fmt_bytes(m.size_vram_bytes)} VRAM"
                        + (f" / {_fmt_bytes(m.size_bytes)}"
                           if m.size_bytes != m.size_vram_bytes else "")
                        + (f"  ({vram_pct:.0f}% on GPU)"
                           if 0 < vram_pct < 100 else ""),
                        size=11, color=ON_SURFACE_DIM,
                    ),
                    ft.Container(expand=True),
                    ft.Text(expires, size=11, color=ON_SURFACE_DIM),
                ], spacing=8,
                  vertical_alignment=ft.CrossAxisAlignment.CENTER)
                hw_live_placement.controls.append(line)

        if not any_loaded:
            hw_live_placement_status.value = (
                "Updated just now · no models currently loaded "
                "(daemons load on first request)"
            )
        else:
            hw_live_placement_status.value = (
                f"Updated just now · refreshing every 5s"
            )
        try:
            state.page.update()
        except Exception:
            pass

    async def hw_live_placement_ticker():
        """Async polling loop. Owned by the supervisor for the
        lifetime of this Settings view. Stops when hw_live_state
        ['polling'] flips False (e.g. workspace switch)."""
        import asyncio
        while hw_live_state.get("polling", False):
            try:
                snapshot: dict[int, list] = {}
                for d in hw_state["table"].daemons:
                    snapshot[d.gpu_index] = query_loaded_models(
                        d.url, timeout=2.0,
                    )
                hw_render_live_placement(snapshot)
                hw_live_state["last_update"] = time.monotonic()
            except Exception:
                # never let a polling glitch kill the ticker
                pass
            await asyncio.sleep(5.0)

    def start_live_placement():
        if hw_live_state.get("polling"):
            return
        hw_live_state["polling"] = True
        try:
            state.page.run_task(hw_live_placement_ticker)
        except Exception:
            hw_live_state["polling"] = False

    def stop_live_placement():
        hw_live_state["polling"] = False

    # ----- Health-check watchdog (Phase 7) -----
    # Sweeps every registered daemon every HEALTH_CHECK_INTERVAL_S
    # seconds. When a daemon stops responding consecutively, removes
    # it from the routing table + demotes its assignments to AUTO so
    # the picker reroutes. Restores them when the daemon recovers.
    hw_health_state: dict = {
        "running": False,
        "fail_counts": {},          # gpu_index -> consecutive misses
        "stranded": {},             # (model_tag, role) -> original_gpu
    }

    async def hw_health_ticker():
        import asyncio
        while hw_health_state.get("running", False):
            try:
                events = health_check_once(
                    hw_state["table"],
                    fail_counts=hw_health_state["fail_counts"],
                    stranded_backup=hw_health_state["stranded"],
                )
                if events:
                    # Persist mutated table + repaint UI.
                    hw_save_to_workspace()
                    hw_render()
                    for ev in events:
                        if ev.kind == "down":
                            _toast(state.page,
                                   f"⚠ Daemon for GPU {ev.gpu_index} "
                                   f"stopped responding — "
                                   f"assignments demoted to auto.")
                        elif ev.kind == "back":
                            _toast(state.page,
                                   f"✓ Daemon recovered: "
                                   f"GPU {ev.gpu_index}")
            except Exception:
                # Watchdog never crashes the GUI on a probe glitch.
                pass
            await asyncio.sleep(8.0)

    def start_health_watchdog():
        if hw_health_state.get("running"):
            return
        hw_health_state["running"] = True
        try:
            state.page.run_task(hw_health_ticker)
        except Exception:
            hw_health_state["running"] = False

    def stop_health_watchdog():
        hw_health_state["running"] = False

    hardware_card = section_card(
        "HARDWARE / GPU ROUTING",
        ft.Row([
            ft.Text("Detected GPUs", size=12,
                    weight=ft.FontWeight.W_700,
                    color=ON_SURFACE),
            ft.Container(expand=True),
            rescan_btn,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        hw_gpu_list,
        ft.Divider(height=1, color="#262938"),
        ft.Text("Daemons", size=12,
                weight=ft.FontWeight.W_700,
                color=ON_SURFACE),
        hw_daemon_list,
        hw_status_text,
        ft.Divider(height=1, color="#262938"),
        ft.Text("Per-model GPU assignment", size=12,
                weight=ft.FontWeight.W_700,
                color=ON_SURFACE),
        hw_assignment_list,
        ft.Divider(height=1, color="#262938"),
        ft.Row([
            ft.Text("Live placement", size=12,
                    weight=ft.FontWeight.W_700,
                    color=ON_SURFACE),
            ft.Container(expand=True),
            hw_live_placement_status,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        hw_live_placement,
        ft.Divider(height=1, color="#262938"),
        hw_spawn_managed_sw,
        hw_sched_spread_sw,
    )

    # Initial population — done lazily on the first Settings open via
    # load_settings (defined below). hw_render() is called from there.

    body = ft.ListView(
        spacing=14,
        padding=ft.padding.symmetric(horizontal=20, vertical=20),
        expand=True,
        controls=[
            ft.Text("Settings", size=18, weight=ft.FontWeight.W_700,
                    color=ON_SURFACE),
            hardware_card,
            ft.Row(
                [
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "INGEST",
                            ft.Row([chunk_size, chunk_overlap], spacing=10, wrap=True),
                            enable_ocr,
                            contextual,
                            llm_inspect_pages_sw,
                            llm_correct_garbled_sw,
                            preview_recoveries_sw,
                        ),
                    ),
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "RETRIEVAL",
                            use_corpus,
                            ft.Row([top_k, hybrid, rerank], spacing=14,
                                   wrap=True,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Row([use_hyde_sw, multi_query_sw, agentic_sw],
                                   spacing=14, wrap=True,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Row([context_window_field, use_mmr_sw,
                                    mmr_lambda_field], spacing=14,
                                   wrap=True,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Row([expand_chapter_sw,
                                    chapter_max_chars_field], spacing=14,
                                   wrap=True,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        ),
                    ),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "LLM",
                            ft.Row([llm_model, refresh_btn], spacing=4,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Row([browse_llm_btn, gguf_btn], spacing=8,
                                   wrap=True),
                            ollama_url,
                            ft.Row([temperature, max_tokens], spacing=10, wrap=True),
                        ),
                    ),
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "EMBEDDER",
                            embed_provider,
                            ollama_embed_model,
                            browse_embed_btn,
                            embed_model,
                        ),
                    ),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "AGENT (used when 'Agentic mode' is ON)",
                            agent_provider_dd,
                            agent_model_field,
                            agent_api_key_field,
                            agent_base_url_field,
                        ),
                    ),
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "QUERY MODIFIERS (toggle in chat composer)",
                            use_file_metadata_sw,
                            query_prefix_field,
                            query_suffix_field,
                            query_negatives_field,
                        ),
                    ),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "INGEST + INFERENCE PERFORMANCE",
                            unload_llm_sw,
                            ft.Row([embed_batch_field, num_batch_field,
                                    num_ctx_field],
                                   wrap=True, spacing=10,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Text(
                                "Higher batch sizes use more memory but ingest "
                                "faster on a strong GPU. Unloading the chat LLM "
                                "is auto-skipped when Contextual Retrieval is on. "
                                "num_batch=1024 measured -23% TTFT on a 32B model "
                                "vs default 512. For maximum chat speed also set "
                                "OLLAMA_FLASH_ATTENTION=1 in the shell that "
                                "starts `ollama serve` — see Doctor for the full "
                                "recommendation.",
                                size=11, color=ON_SURFACE_DIM, italic=True,
                            ),
                        ),
                    ),
                    ft.Container(
                        expand=1,
                        content=_build_storage_card(state),
                    ),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [
                    ft.Container(
                        expand=1,
                        content=_build_appearance_card(state),
                    ),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [
                    ft.FilledButton("Save settings", icon=ft.Icons.CHECK,
                                    on_click=save, bgcolor=ACCENT, color="#FFFFFF",
                                    tooltip=TIP["save_btn"]),
                    ft.TextButton("Reset", icon=ft.Icons.UNDO,
                                  on_click=lambda _: load_from_cfg(),
                                  tooltip=TIP["reset_btn"]),
                ],
                spacing=10,
            ),
            ft.Container(height=24),
        ],
    )

    container = ft.Container(bgcolor=BG_DARK, expand=True, content=body)
    # Expose open_pull_dialog so other views (e.g. the chat error
    # 'Re-pull this model' action) can trigger a pull without re-implementing
    # the whole dialog. The chat view captures this via a shared callback dict.
    return container, load_from_cfg, open_pull_dialog


# ============================================================================
# Doctor view
# ============================================================================

def build_doctor_view(state: AppState):
    rows = ft.Column([], spacing=8)

    def refresh(_=None):
        from ez_rag import ocr as ocr_mod
        rows.controls.clear()
        ok = lambda: ft.Icon(ft.Icons.CHECK_CIRCLE, color=SUCCESS, size=18)
        bad = lambda: ft.Icon(ft.Icons.CANCEL, color=DANGER, size=18)
        warn = lambda: ft.Icon(ft.Icons.INFO, color=WARNING, size=18)

        def row(label, status_ok, value, hint=""):
            return ft.Container(
                padding=ft.padding.symmetric(horizontal=14, vertical=10),
                bgcolor=SURFACE_DARK,
                border=ft.border.all(1, "#262938"),
                border_radius=10,
                content=ft.Row([
                    ok() if status_ok is True else (bad() if status_ok is False else warn()),
                    ft.Column([
                        ft.Text(label, size=13, weight=ft.FontWeight.W_600,
                                color=ON_SURFACE),
                        ft.Text(hint, size=11, color=ON_SURFACE_DIM),
                    ], spacing=2, tight=True, expand=True),
                    ft.Text(value, size=12, color=ON_SURFACE_DIM),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            )

        backend = detect_backend(state.cfg)
        rows.controls.append(row(
            "LLM backend", backend != "none",
            backend if backend != "none" else "none",
            "ollama serve  /  pip install ez-rag[llm]" if backend == "none" else "",
        ))

        try:
            import fastembed  # noqa
            rows.controls.append(row("fastembed", True, "installed"))
        except ImportError:
            rows.controls.append(row("fastembed", False, "missing"))

        try:
            import llama_cpp  # noqa
            rows.controls.append(row("llama-cpp-python", True, "installed",
                                     "for local GGUF inference"))
        except ImportError:
            rows.controls.append(row("llama-cpp-python", "warn", "not installed",
                                     "optional — Ollama is the easier path"))

        s = ocr_mod.status()
        rows.controls.append(row(
            "RapidOCR", s["rapidocr"], "ok" if s["rapidocr"] else "missing",
            "pip install ez-rag[ocr]" if not s["rapidocr"] else "",
        ))
        rows.controls.append(row(
            "Tesseract", s["tesseract"], "ok" if s["tesseract"] else "missing",
            "fallback OCR engine" if s["tesseract"] else "optional",
        ))
        try:
            from importlib.metadata import version as _v
            rows.controls.append(row("Flet (GUI)", True, _v("flet")))
        except Exception:
            rows.controls.append(row("Flet (GUI)", False, "missing"))

        import sys
        rows.controls.append(row(
            "Python", True,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({sys.platform})",
        ))

        # ----- performance recommendations ------------------------------
        # These are environment variables that affect Ollama itself, not
        # ez-rag — they have to be set in the shell that launched
        # `ollama serve`. ez-rag can only detect them and recommend
        # changes. Empirically (deepseek-r1:32b, RTX 5090) flash + KV
        # quant gave +6% throughput; with our num_batch=1024 default
        # (already applied per-request) the total gain was ~+8%.
        flash = os.environ.get("OLLAMA_FLASH_ATTENTION", "")
        kvtype = os.environ.get("OLLAMA_KV_CACHE_TYPE", "")
        flash_on = flash in ("1", "true", "TRUE", "True")
        kv_q = kvtype in ("q4_0", "q8_0")

        rows.controls.append(row(
            "Ollama: Flash attention",
            flash_on,
            f"OLLAMA_FLASH_ATTENTION={flash or '(unset)'}",
            "+~4% throughput. Set OLLAMA_FLASH_ATTENTION=1 in the shell "
            "that launches `ollama serve` and restart Ollama."
            if not flash_on else "enabled",
        ))
        rows.controls.append(row(
            "Ollama: KV cache quantization",
            kv_q,
            f"OLLAMA_KV_CACHE_TYPE={kvtype or '(unset → f16)'}",
            "Halves KV memory at no measurable speed cost (q8_0) or "
            "quarters it (q4_0). Set OLLAMA_KV_CACHE_TYPE=q8_0 in the "
            "shell launching `ollama serve`. Pairs with flash attention."
            if not kv_q else "enabled",
        ))
        rows.controls.append(row(
            "Per-request num_batch",
            int(getattr(state.cfg, "num_batch", 0) or 0) >= 1024,
            f"num_batch={getattr(state.cfg, 'num_batch', 0) or 0}",
            "1024 measured -23% TTFT vs default 512. Tune in Settings → "
            "Performance.",
        ))

        # ----- Embedder vs index dimension check -----
        # Crucial: if the configured embedder's vector size differs from
        # the index's, every retrieval will throw the matmul-mismatch
        # error. Surface this BEFORE the user hits it in chat.
        if state.ws is not None and state.ws.meta_db_path.exists():
            idx_stats = read_stats(state.ws.meta_db_path) or {}
            idx_embedder = idx_stats.get("last_embedder") or ""
            cur_embedder_label = ""
            cur_dim = None
            try:
                from ez_rag.embed import make_embedder as _mk
                emb = _mk(state.cfg)
                cur_embedder_label = emb.name
                cur_dim = emb.dim
            except Exception as _e:
                cur_embedder_label = f"unloadable ({_e})"
            # Read index dim straight from a chunk row.
            idx_dim = None
            try:
                import sqlite3 as _s
                conn = _s.connect(str(state.ws.meta_db_path))
                r = conn.execute(
                    "SELECT embedding FROM chunks LIMIT 1"
                ).fetchone()
                if r and r[0]:
                    idx_dim = len(r[0]) // 4  # float32
                conn.close()
            except Exception:
                pass
            if idx_dim is None:
                # No chunks yet — nothing to mismatch. Skip the row.
                pass
            elif cur_dim is None:
                rows.controls.append(row(
                    "Embedder match", "warn",
                    f"index={idx_dim}-d, current=?",
                    "Couldn't load the configured embedder — see error above.",
                ))
            elif cur_dim != idx_dim:
                rows.controls.append(row(
                    "Embedder match", False,
                    f"index={idx_dim}-d ({idx_embedder or 'unknown'}), "
                    f"current={cur_dim}-d ({cur_embedder_label})",
                    "MISMATCH — chat will fail with a matmul error. "
                    "Either re-ingest with the current embedder (Files tab "
                    "→ Re-ingest force) or switch the embedder back in "
                    "Settings → Embedder.",
                ))
            else:
                rows.controls.append(row(
                    "Embedder match", True,
                    f"{idx_dim}-d ({cur_embedder_label})",
                    "Index dimensions match the active embedder.",
                ))

        state.page.update()

    container = ft.Container(
        bgcolor=BG_DARK,
        expand=True,
        padding=20,
        content=ft.Column([
            ft.Row([
                ft.Text("Doctor", size=18, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE),
                ft.Container(expand=True),
                ft.IconButton(icon=ft.Icons.REFRESH, on_click=refresh,
                              tooltip=TIP["doctor_refresh"]),
            ]),
            ft.Container(height=8),
            rows,
        ], scroll=ft.ScrollMode.AUTO, expand=True),
    )
    return container, refresh


# ============================================================================
# Welcome view (no workspace open)
# ============================================================================

def build_welcome(state: AppState, *, on_open_workspace):
    recents_col = ft.Column([], spacing=8, expand=True,
                            scroll=ft.ScrollMode.AUTO)

    def render_recents():
        recents_col.controls.clear()
        recents = load_recents()
        if not recents:
            recents_col.controls.append(ft.Text(
                "No recent workspaces yet.", color=ON_SURFACE_DIM, italic=True,
            ))
            return
        for p in recents:
            recents_col.controls.append(ft.Container(
                padding=ft.padding.symmetric(horizontal=14, vertical=12),
                bgcolor=SURFACE_DARK,
                border=ft.border.all(1, "#262938"),
                border_radius=10,
                on_click=(lambda p=p: lambda _: on_open_workspace(p))(p),
                content=ft.Row([
                    ft.Icon(ft.Icons.FOLDER, size=20, color=ACCENT),
                    ft.Column([
                        ft.Text(p.name, size=14, weight=ft.FontWeight.W_600,
                                color=ON_SURFACE),
                        ft.Text(str(p), size=11, color=ON_SURFACE_DIM),
                    ], spacing=2, tight=True, expand=True),
                    ft.Icon(ft.Icons.ARROW_FORWARD_IOS,
                            size=14, color=ON_SURFACE_DIM),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            ))

    container = ft.Container(
        bgcolor=BG_DARK, expand=True, alignment=ft.Alignment.CENTER,
        content=ft.Container(
            width=560,
            padding=20,
            content=ft.Column([
                ft.Container(
                    width=64, height=64, border_radius=16, bgcolor=ACCENT,
                    alignment=ft.Alignment.CENTER,
                    content=ft.Text("ez", size=28,
                                    weight=ft.FontWeight.W_900,
                                    color="#FFFFFF"),
                ),
                ft.Container(height=18),
                ft.Text("ez-rag", size=32, weight=ft.FontWeight.W_800,
                        color=ON_SURFACE),
                ft.Text("Drop documents in a folder. Chat with them.",
                        size=14, color=ON_SURFACE_DIM),
                ft.Container(height=24),
                ft.Row([
                    ft.FilledButton(
                        "Open workspace", icon=ft.Icons.FOLDER_OPEN,
                        on_click=lambda _: on_open_workspace(None),
                        bgcolor=ACCENT, color="#FFFFFF",
                    ),
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(height=24),
                ft.Text("RECENT", size=11,
                        weight=ft.FontWeight.W_700,
                        color=ON_SURFACE_DIM),
                ft.Container(height=6),
                recents_col,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
        ),
    )
    return container, render_recents


# ============================================================================
# Toast helper
# ============================================================================

def open_manage_rags_overlay(
    page: ft.Page,
    *,
    on_open_workspace,
    on_change=None,
) -> None:
    """A panel that lists every RAG in the default folder with Open / Export
    / Delete actions, plus an Import-zip button to drop a previously-exported
    archive into the folder as a new RAG."""
    overlay_ref: dict = {}

    def close_it(_=None):
        ov = overlay_ref.get("overlay")
        if ov is None:
            return
        ov.visible = False
        try:
            if ov in page.overlay:
                page.overlay.remove(ov)
            for picker in (zip_picker,):
                if picker in page.services:
                    page.services.remove(picker)
        except Exception:
            pass
        page.update()

    rows_col = ft.Column([], spacing=6, scroll=ft.ScrollMode.AUTO)
    info_text = ft.Text("", size=12, color=ON_SURFACE_DIM)
    zip_picker = ft.FilePicker()
    page.services.append(zip_picker)

    def fmt_bytes(n: int) -> str:
        n = float(n)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"

    def refresh_rows():
        rags = list_managed_rags()
        rows_col.controls.clear()
        if not rags:
            rows_col.controls.append(ft.Text(
                f"No RAGs found in {get_default_rags_dir()}.\n"
                "Create one with the ⨁ icon in the header, or change the "
                "Default RAGs folder in Settings → Storage.",
                size=12, color=ON_SURFACE_DIM, italic=True,
            ))
            info_text.value = ""
            page.update()
            return
        info_text.value = (
            f"{len(rags)} RAG(s) in {get_default_rags_dir()}"
        )
        for ws in rags:
            from ez_rag.index import read_stats
            stats = read_stats(ws.meta_db_path) or {}
            n_files = stats.get("files", 0)
            n_chunks = stats.get("chunks", 0)
            idx_size = ws.index_size_bytes()

            def make_open(p):
                return lambda _: (close_it(), on_open_workspace(p))

            def make_export(w=ws):
                async def _do():
                    out_path = await zip_picker.save_file(
                        dialog_title=f"Export {w.root.name} as…",
                        file_name=f"{w.root.name}.zip",
                        allowed_extensions=["zip"],
                    )
                    if not out_path:
                        return
                    try:
                        dest = w.export_archive(Path(out_path))
                        _toast(page, f"Exported → {dest.name} "
                                     f"({dest.stat().st_size:,} bytes)")
                    except Exception as ex:
                        _toast(page, f"Export failed: {ex}")
                return lambda _: page.run_task(_do)

            def make_delete(w=ws):
                def _delete(_):
                    try:
                        import shutil
                        shutil.rmtree(w.root)
                        _toast(page, f"Deleted {w.root.name}")
                        refresh_rows()
                        if on_change:
                            on_change()
                    except Exception as ex:
                        _toast(page, f"Delete failed: {ex}")
                return _delete

            row = ft.Container(
                padding=ft.padding.symmetric(horizontal=12, vertical=10),
                bgcolor=SURFACE_DARK_HI,
                border=ft.border.all(1, "#262938"),
                border_radius=10,
                content=ft.Row([
                    ft.Icon(ft.Icons.FOLDER, size=18, color=ACCENT),
                    ft.Column([
                        ft.Text(ws.root.name, size=13,
                                weight=ft.FontWeight.W_700, color=ON_SURFACE),
                        ft.Text(
                            f"{n_files} files · {n_chunks} chunks · "
                            f"index {fmt_bytes(idx_size)}",
                            size=11, color=ON_SURFACE_DIM,
                        ),
                    ], spacing=2, tight=True, expand=True),
                    ft.OutlinedButton("Open", icon=ft.Icons.PLAY_ARROW,
                                      on_click=make_open(ws.root)),
                    ft.OutlinedButton("Export", icon=ft.Icons.DOWNLOAD,
                                      on_click=make_export()),
                    ft.OutlinedButton("Delete", icon=ft.Icons.DELETE,
                                      on_click=make_delete()),
                ], spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
            )
            rows_col.controls.append(row)
        page.update()

    def import_zip(_):
        async def _do():
            files = await zip_picker.pick_files(
                dialog_title="Pick a .zip exported from ez-rag",
                allowed_extensions=["zip"],
            )
            if not files:
                return
            arc = Path(files[0].path)
            target_parent = get_default_rags_dir()
            into = target_parent / arc.stem
            if into.exists():
                into = target_parent / f"{arc.stem}-imported"
            try:
                target_parent.mkdir(parents=True, exist_ok=True)
                Workspace.import_archive(arc, into)
                _toast(page, f"Imported → {into.name}")
                refresh_rows()
                if on_change:
                    on_change()
            except Exception as ex:
                _toast(page, f"Import failed: {ex}")
        page.run_task(_do)

    title_row = ft.Row([
        ft.Icon(ft.Icons.STORAGE, color=ACCENT),
        ft.Text("Manage RAGs", size=18, weight=ft.FontWeight.W_700,
                color=ON_SURFACE),
        ft.Container(expand=True),
        ft.OutlinedButton("Import .zip…", icon=ft.Icons.UPLOAD_FILE,
                          on_click=import_zip),
        ft.IconButton(icon=ft.Icons.REFRESH, on_click=lambda _: refresh_rows(),
                      tooltip="Re-scan default folder"),
        ft.IconButton(icon=ft.Icons.CLOSE, on_click=close_it,
                      tooltip="Close"),
    ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=8)

    panel = ft.Container(
        bgcolor=SURFACE_DARK,
        border=ft.border.all(1, "#262938"),
        border_radius=14,
        padding=20,
        width=820,
        height=600,
        content=ft.Column([
            title_row,
            info_text,
            ft.Divider(color="#262938"),
            ft.Container(content=rows_col, expand=True),
        ], spacing=10, expand=True),
    )

    overlay = ft.Container(
        expand=True, visible=True,
        bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
        alignment=ft.Alignment.CENTER,
        on_click=close_it,
        content=ft.Container(on_click=lambda e: None, content=panel),
    )
    overlay_ref["overlay"] = overlay
    page.overlay.append(overlay)
    refresh_rows()
    page.update()


def open_create_rag_overlay(
    page: ft.Page,
    *,
    on_created,                 # callable(Path) — receives the new workspace path
    default_parent: Path | None = None,
) -> None:
    """A modal that lets the user create a named RAG by picking N source
    folders. Each folder's supported files are copied into the new
    workspace's docs/. The user picks a base parent dir + name; we slug it.
    """
    parent = default_parent or get_default_rags_dir()

    overlay_ref: dict = {}
    def close_it(_=None):
        ov = overlay_ref.get("overlay")
        if ov is None:
            return
        ov.visible = False
        try:
            if ov in page.overlay:
                page.overlay.remove(ov)
            for picker in (parent_picker, source_picker):
                if picker in page.services:
                    page.services.remove(picker)
        except Exception:
            pass
        page.update()

    name_field = ft.TextField(
        label="RAG name", hint_text="e.g. D&D rulebooks",
        value="", expand=True, dense=True,
        border_color="#2A2D3D",
        focused_border_color=ACCENT,
        cursor_color=ACCENT,
    )
    parent_field = ft.TextField(
        label="Parent folder (will create a subfolder named after the RAG)",
        value=str(parent), expand=True, dense=True,
        border_color="#2A2D3D", focused_border_color=ACCENT,
    )
    parent_picker = ft.FilePicker()
    source_picker = ft.FilePicker()
    page.services.append(parent_picker)
    page.services.append(source_picker)

    def pick_parent(_):
        async def _do():
            picked = await parent_picker.get_directory_path(
                dialog_title="Where should this RAG live?",
            )
            if picked:
                parent_field.value = picked
                page.update()
        page.run_task(_do)

    sources: list[Path] = []
    sources_col = ft.Column([], spacing=4, tight=True)

    def render_sources():
        sources_col.controls.clear()
        if not sources:
            sources_col.controls.append(ft.Text(
                "No source folders yet — click 'Add folder…' to import documents.",
                size=12, color=ON_SURFACE_DIM, italic=True,
            ))
        else:
            for i, p in enumerate(sources):
                row = ft.Row([
                    ft.Icon(ft.Icons.FOLDER, size=16, color=ACCENT),
                    ft.Text(str(p), size=12, color=ON_SURFACE, expand=True,
                            no_wrap=False),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE, icon_size=14,
                        tooltip="Remove",
                        on_click=(lambda idx=i: lambda _: (
                            sources.pop(idx), render_sources(), page.update()
                        ))(i),
                    ),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=8)
                sources_col.controls.append(row)
        page.update()

    def add_source(_):
        async def _do():
            picked = await source_picker.get_directory_path(
                dialog_title="Pick a folder of documents",
            )
            if picked:
                p = Path(picked)
                if p not in sources:
                    sources.append(p)
                    render_sources()
        page.run_task(_do)

    blank_check = ft.Checkbox(
        value=True,
        active_color=ACCENT,
        label="Start with an empty docs/ (recommended)",
        label_style=ft.TextStyle(size=12, color=ON_SURFACE_DIM),
    )

    status_line = ft.Text("", size=12, color=ON_SURFACE_DIM)
    create_btn = ft.FilledButton(
        "Create RAG", icon=ft.Icons.AUTO_AWESOME,
        bgcolor=ACCENT, color="#FFFFFF",
    )
    cancel_btn = ft.TextButton("Cancel", on_click=close_it)

    render_sources()

    def do_create(_):
        nm = (name_field.value or "").strip()
        if not nm:
            status_line.value = "Pick a name for the RAG first."
            page.update()
            return
        prnt = (parent_field.value or "").strip()
        if not prnt:
            status_line.value = "Pick a parent folder first."
            page.update()
            return
        try:
            ws = Workspace.create_named(
                name=nm,
                parent_dir=Path(prnt),
                source_folders=sources,
                clear_default=bool(blank_check.value),
            )
        except FileExistsError as e:
            status_line.value = str(e)
            page.update()
            return
        except Exception as e:
            status_line.value = f"Error: {e}"
            page.update()
            return
        status_line.value = f"Created {ws.root.name} — opening…"
        page.update()
        close_it()
        on_created(ws.root)

    create_btn.on_click = do_create

    title_row = ft.Row([
        ft.Icon(ft.Icons.AUTO_AWESOME, color=ACCENT),
        ft.Text("Create a new RAG", size=18, weight=ft.FontWeight.W_700,
                color=ON_SURFACE),
        ft.Container(expand=True),
        ft.IconButton(icon=ft.Icons.CLOSE, on_click=close_it,
                      tooltip="Close"),
    ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

    panel = ft.Container(
        bgcolor=SURFACE_DARK,
        border=ft.border.all(1, "#262938"),
        border_radius=14,
        padding=20,
        width=720,
        content=ft.Column([
            title_row,
            ft.Divider(color="#262938"),
            name_field,
            ft.Row([parent_field,
                    ft.OutlinedButton("Browse…", icon=ft.Icons.FOLDER_OPEN,
                                      on_click=pick_parent)],
                   vertical_alignment=ft.CrossAxisAlignment.END, spacing=8),
            ft.Container(height=8),
            ft.Text("SOURCE FOLDERS", size=11, weight=ft.FontWeight.W_700,
                    color=ON_SURFACE_DIM),
            ft.Container(
                padding=ft.padding.symmetric(horizontal=12, vertical=10),
                bgcolor=BG_DARK,
                border=ft.border.all(1, "#222637"),
                border_radius=8,
                content=sources_col,
            ),
            ft.Row([
                ft.OutlinedButton("Add folder…", icon=ft.Icons.CREATE_NEW_FOLDER,
                                  on_click=add_source),
                ft.Container(expand=True),
                blank_check,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Divider(color="#262938"),
            status_line,
            ft.Row([
                ft.Container(expand=True),
                cancel_btn,
                create_btn,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
        ], spacing=10, tight=True),
    )

    overlay = ft.Container(
        expand=True, visible=True,
        bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
        alignment=ft.Alignment.CENTER,
        on_click=close_it,
        content=ft.Container(on_click=lambda e: None, content=panel),
    )
    overlay_ref["overlay"] = overlay
    page.overlay.append(overlay)
    page.update()


def open_info_overlay(page: ft.Page, *, title: str, body_md: str,
                      icon=None, accent: str = ACCENT) -> None:
    """Open a centered modal showing markdown content (Help, Credits, etc.).

    Uses the same plain-Container-on-overlay pattern as the model browser
    so it works reliably from any thread.
    """
    overlay_ref: dict = {}

    def close_it(_=None):
        ov = overlay_ref.get("overlay")
        if ov is None:
            return
        ov.visible = False
        try:
            if ov in page.overlay:
                page.overlay.remove(ov)
        except Exception:
            pass
        page.update()

    body = ft.Markdown(
        body_md,
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
        on_tap_link=lambda e: page.launch_url(e.data),
        code_theme="atom-one-dark",
    )

    title_row = ft.Row([
        ft.Icon(icon, color=accent) if icon else ft.Container(),
        ft.Text(title, weight=ft.FontWeight.W_700, size=18, color=ON_SURFACE),
        ft.Container(expand=True),
        ft.IconButton(icon=ft.Icons.CLOSE, on_click=close_it,
                      tooltip="Close (Esc)"),
    ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

    panel = ft.Container(
        bgcolor=SURFACE_DARK,
        border=ft.border.all(1, "#262938"),
        border_radius=14,
        padding=20,
        width=820,
        height=640,
        content=ft.Column([
            title_row,
            ft.Divider(color="#262938"),
            ft.Container(
                content=ft.Column([body], scroll=ft.ScrollMode.AUTO,
                                  expand=True),
                expand=True,
            ),
            ft.Row([
                ft.Container(expand=True),
                ft.TextButton("Close", on_click=close_it),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=10, expand=True),
    )

    overlay = ft.Container(
        expand=True, visible=True,
        bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
        alignment=ft.Alignment.CENTER,
        on_click=close_it,
        content=ft.Container(on_click=lambda e: None, content=panel),
    )
    overlay_ref["overlay"] = overlay
    page.overlay.append(overlay)
    page.update()


def _toast(page: ft.Page, msg: str) -> None:
    sb = ft.SnackBar(
        content=ft.Text(msg, color="#FFFFFF"),
        bgcolor=SURFACE_DARK_HI,
        duration=2500,
    )
    try:
        page.show_dialog(sb)
    except Exception:
        # Last-ditch — just log to stderr, never crash the GUI on a toast.
        print(f"[ez-rag] {msg}")


# ============================================================================
# System telemetry status bar
# ============================================================================

def _build_sysmon_bar(state: AppState, *,
                      refresh_pull_badge_cb=None) -> ft.Control:
    """Pinned footer showing CPU / RAM / GPU / VRAM / temps in real time.

    Sampling runs on a background thread at 1 Hz. Fields the host can't
    supply (no NVIDIA driver, no CPU temp on Windows, etc.) are hidden
    rather than shown as 'N/A' clutter.
    """
    from ez_rag.sysmon import (
        Sample, fmt_gb, fmt_mb_as_gb, fmt_pct, fmt_power_w, fmt_temp_c, sample,
    )

    # ----- color helpers -------------------------------------------------
    def _color_for_pct(pct: float | None) -> str:
        if pct is None:
            return ON_SURFACE_DIM
        if pct >= 95:
            return DANGER
        if pct >= 80:
            return WARNING
        return ON_SURFACE

    def _color_for_temp(t: float | None) -> str:
        if t is None:
            return ON_SURFACE_DIM
        if t >= 85:
            return DANGER
        if t >= 75:
            return WARNING
        return ON_SURFACE

    def _segment(label: str, value_text: ft.Text,
                 *, value_extra: ft.Control | None = None,
                 tip: str = "") -> ft.Control:
        kids = [
            ft.Text(label, size=10, color=ON_SURFACE_DIM,
                    weight=ft.FontWeight.W_700),
            value_text,
        ]
        if value_extra is not None:
            kids.append(value_extra)
        return ft.Container(
            tooltip=tip,
            padding=ft.padding.symmetric(horizontal=10, vertical=2),
            content=ft.Row(kids, spacing=6,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    # Mini progress bar (used for CPU/RAM/GPU compute/VRAM percentages)
    def _mini_bar() -> ft.ProgressBar:
        return ft.ProgressBar(
            value=0, color=ACCENT, bgcolor="#1E2130",
            width=80, height=4,
        )

    # ----- segment widgets -----------------------------------------------
    cpu_text = ft.Text("—", size=11, color=ON_SURFACE,
                       weight=ft.FontWeight.W_600, font_family="monospace")
    cpu_bar = _mini_bar()
    cpu_seg = _segment("CPU", cpu_text, value_extra=cpu_bar,
                       tip="System CPU load (rolling) and core count.")

    cpu_temp_text = ft.Text("—", size=11, color=ON_SURFACE,
                            font_family="monospace")
    cpu_temp_seg = _segment("CPU °", cpu_temp_text,
                            tip="CPU package temperature. Often unavailable "
                                "on Windows without elevated permissions.")

    ram_text = ft.Text("—", size=11, color=ON_SURFACE,
                       weight=ft.FontWeight.W_600, font_family="monospace")
    ram_bar = _mini_bar()
    ram_seg = _segment("RAM", ram_text, value_extra=ram_bar,
                       tip="System RAM used / total.")

    gpu_text = ft.Text("—", size=11, color=ON_SURFACE,
                       weight=ft.FontWeight.W_600, font_family="monospace")
    gpu_bar = _mini_bar()
    gpu_seg = _segment("GPU", gpu_text, value_extra=gpu_bar,
                       tip="NVIDIA GPU compute utilization.")

    vram_text = ft.Text("—", size=11, color=ON_SURFACE,
                        weight=ft.FontWeight.W_600, font_family="monospace")
    vram_bar = _mini_bar()
    vram_seg = _segment("VRAM", vram_text, value_extra=vram_bar,
                        tip="GPU video memory used / total. The LLM weights "
                            "+ KV cache live here.")

    gpu_temp_text = ft.Text("—", size=11, color=ON_SURFACE,
                            font_family="monospace")
    gpu_temp_seg = _segment("GPU °", gpu_temp_text,
                            tip="GPU edge temperature.")

    gpu_power_text = ft.Text("—", size=11, color=ON_SURFACE,
                             font_family="monospace")
    gpu_power_seg = _segment("Power", gpu_power_text,
                             tip="Live GPU power draw.")

    sep = lambda: ft.Container(
        width=1, height=14, bgcolor=SURFACE_DARK_HI,
        margin=ft.margin.symmetric(horizontal=2),
    )

    # Layout — segments separated by thin vertical dividers. Whole bar is
    # one row that horizontally scrolls on narrow windows so nothing gets
    # truncated mysteriously.
    bar_row = ft.Row(
        [
            cpu_seg, sep(),
            cpu_temp_seg, sep(),
            ram_seg, sep(),
            gpu_seg, sep(),
            vram_seg, sep(),
            gpu_temp_seg, sep(),
            gpu_power_seg,
        ],
        spacing=4,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
    )

    # ----- frame-loop heartbeat ------------------------------------------
    # Flutter Desktop on Windows pauses repaints when the window has no
    # user input, which means page.update() from background threads only
    # gets painted on the next user click. Keeping a small element
    # animating forces vsync to stay engaged — the indicator below ticks
    # twice a second whether or not the user is touching the window, which
    # makes ingest progress / sysmon stats feel live.
    heartbeat = ft.ProgressRing(
        width=10, height=10, stroke_width=1.5,
        color=ft.Colors.with_opacity(0.6, ACCENT),
        bgcolor="transparent",
        tooltip="Live indicator — keeps the UI repainting so progress "
                "tickers stay current even when you're not clicking.",
    )

    bar_container = ft.Container(
        content=ft.Row([heartbeat, bar_row], spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        bgcolor=SURFACE_DARK,
        border=ft.border.only(top=ft.BorderSide(1, "#1E2130")),
        height=28,
    )

    # ----- background sampler --------------------------------------------
    sysmon_state = {"running": True, "interval_s": 1.0}
    state.page.on_close = (lambda *a, **k:
        sysmon_state.__setitem__("running", False))

    def _apply(s: Sample):
        # CPU
        if s.cpu_pct is not None:
            cores = f" / {s.cpu_count}c" if s.cpu_count else ""
            cpu_text.value = f"{fmt_pct(s.cpu_pct)}{cores}"
            cpu_text.color = _color_for_pct(s.cpu_pct)
            cpu_bar.value = max(0.0, min(1.0, s.cpu_pct / 100.0))
            cpu_bar.color = cpu_text.color
            cpu_seg.visible = True
        else:
            cpu_seg.visible = False

        # CPU temp — hide if not available (Windows default)
        if s.cpu_temp_c is not None:
            cpu_temp_text.value = fmt_temp_c(s.cpu_temp_c)
            cpu_temp_text.color = _color_for_temp(s.cpu_temp_c)
            cpu_temp_seg.visible = True
        else:
            cpu_temp_seg.visible = False

        # RAM
        if s.ram_pct is not None:
            ram_text.value = (
                f"{fmt_gb(s.ram_used_gb)} / {fmt_gb(s.ram_total_gb)} "
                f"({fmt_pct(s.ram_pct)})"
            )
            ram_text.color = _color_for_pct(s.ram_pct)
            ram_bar.value = max(0.0, min(1.0, s.ram_pct / 100.0))
            ram_bar.color = ram_text.color
            ram_seg.visible = True
        else:
            ram_seg.visible = False

        # GPU (first GPU only — multi-GPU users with nvidia-smi can read
        # the rest from the OS; the status bar stays single-line).
        gpu = s.gpus[0] if s.gpus else None
        if gpu and gpu.util_pct is not None:
            gpu_text.value = fmt_pct(gpu.util_pct)
            gpu_text.color = _color_for_pct(gpu.util_pct)
            gpu_bar.value = max(0.0, min(1.0, (gpu.util_pct or 0) / 100.0))
            gpu_bar.color = gpu_text.color
            gpu_seg.visible = True
        else:
            gpu_seg.visible = False

        if gpu and gpu.vram_used_mb is not None and gpu.vram_total_mb:
            vram_pct = gpu.vram_pct
            vram_text.value = (
                f"{fmt_mb_as_gb(gpu.vram_used_mb)} / "
                f"{fmt_mb_as_gb(gpu.vram_total_mb)} "
                f"({fmt_pct(vram_pct)})"
            )
            vram_text.color = _color_for_pct(vram_pct)
            vram_bar.value = max(0.0, min(1.0, (vram_pct or 0) / 100.0))
            vram_bar.color = vram_text.color
            vram_seg.visible = True
        else:
            vram_seg.visible = False

        if gpu and gpu.temp_c is not None:
            gpu_temp_text.value = fmt_temp_c(gpu.temp_c)
            gpu_temp_text.color = _color_for_temp(gpu.temp_c)
            gpu_temp_seg.visible = True
        else:
            gpu_temp_seg.visible = False

        if gpu and gpu.power_w is not None:
            gpu_power_text.value = fmt_power_w(gpu.power_w)
            gpu_power_seg.visible = True
        else:
            gpu_power_seg.visible = False

    async def _sampler_async():
        """Async version of the sysmon ticker.

        IMPORTANT: this MUST be async + scheduled via page.run_task, not
        page.run_thread. Flet's `page.update()` only reliably paints when
        called from a coroutine running on the asyncio loop. The same
        call from a vanilla thread (page.run_thread) just queues the
        change and waits for an OS paint event (focus change, alt-tab,
        Win+S) before flushing. See flet-dev/flet#3571 / #4829.
        """
        import asyncio
        # Burn one sample on a thread so we don't block the event loop
        # while psutil initializes its first cpu_percent baseline.
        await asyncio.to_thread(sample)
        while sysmon_state["running"]:
            await asyncio.sleep(sysmon_state["interval_s"])
            if not sysmon_state["running"]:
                break
            try:
                # nvidia-smi takes ~5-30 ms — push it to a thread so we
                # don't block the event loop.
                snap = await asyncio.to_thread(sample)
                _apply(snap)
                if refresh_pull_badge_cb:
                    cb = refresh_pull_badge_cb.get("fn")
                    if cb is not None:
                        try: cb()
                        except Exception: pass
                state.page.update()
                # Belt-and-suspenders Win32 paint message. Cheap.
                force_window_redraw()
            except Exception:
                pass

    # page.run_task() schedules the coroutine on Flet's asyncio loop —
    # the ONLY context where page.update() reliably triggers a frame.
    state.page.run_task(_sampler_async)

    return bar_container


# ============================================================================
# Main
# ============================================================================

def app(page: ft.Page):
    # Resolve the active palette and apply BEFORE any widgets are built
    # below — those widgets bake the colors into their constructor args.
    themes = load_themes()
    theme_name = get_theme_name()
    if theme_name not in themes:
        theme_name = "dark"
    _apply_palette(themes[theme_name])

    # Sweep stale citation page-image previews so the cache doesn't grow
    # forever. 3 days of inactivity → eviction (mtime is touched on each
    # cache hit so frequently-viewed previews effectively get a rolling
    # lease). One-shot at startup, never blocks rendering.
    try:
        from ez_rag.preview import sweep_old_previews
        sweep_old_previews()
    except Exception:
        pass

    page.title = "ez-rag"
    # Light palettes need ThemeMode.LIGHT so Flet's auto-rendered material
    # widgets (date pickers, default scrollbars, etc.) match.
    is_light = theme_name in ("light", "solarized_light")
    page.theme_mode = ft.ThemeMode.LIGHT if is_light else ft.ThemeMode.DARK
    page.bgcolor = BG_DARK
    page.padding = 0
    page.window.width = 1280
    page.window.height = 820
    page.window.min_width = 980
    page.window.min_height = 640
    _theme(page)

    state = AppState(page=page)

    # ---- workspace folder picker -----------------------------------------
    ws_picker = ft.FilePicker()
    page.services.append(ws_picker)

    refresh_files_cb: dict = {}
    refresh_pull_badge_cb: dict = {}
    refresh_status_cb: dict = {}
    refresh_doctor_cb: dict = {}
    welcome_render_cb: dict = {}

    # Rail rebuilder — set later (when the rail itself is constructed),
    # called here so the nav reflects "have a workspace yet?".
    rebuild_rail_cb: dict = {}

    def set_workspace_path(path: Path):
        ws = Workspace(path)
        if not ws.is_initialized():
            ws.initialize()
        state.ws = ws
        state.cfg = ws.load_config()
        add_recent(ws.root)
        clear_embedder_cache()
        # Multi-GPU routing: load this workspace's routing table and
        # install it as the active table so every Ollama call routes
        # through it. If the file doesn't exist yet (single-GPU users
        # / fresh workspace) the table is empty and resolve_url falls
        # back to cfg.llm_url — preserving today's behavior.
        try:
            table = load_routing_table(ws.root)
            set_active_table(table)
        except Exception:
            set_active_table(None)
        if "fn" in refresh_files_cb:
            refresh_files_cb["fn"]()
        if "fn" in load_settings_cb:
            load_settings_cb["fn"]()
        if "fn" in refresh_status_cb:
            refresh_status_cb["fn"]()
        if "fn" in refresh_doctor_cb:
            refresh_doctor_cb["fn"]()
        if "fn" in rebuild_rail_cb:
            rebuild_rail_cb["fn"]()
        switch_view(0)
        render_chat_cb["fn"]()
        page.update()

    def on_open_workspace(path_or_event=None):
        # Either we got a Path (from recents) or a click event.
        if isinstance(path_or_event, Path):
            set_workspace_path(path_or_event)
            return

        # Flet 0.84 FilePicker methods are async — must be awaited.
        async def _pick():
            try:
                path = await ws_picker.get_directory_path(
                    dialog_title="Pick or create a workspace folder",
                )
            except Exception as ex:
                _toast(page, f"Folder picker failed: {ex}")
                return
            if path:
                set_workspace_path(Path(path))

        page.run_task(_pick)

    # ---- header ----------------------------------------------------------

    def go_to_files():
        switch_view(1)

    def refresh_status():
        update_header()
        if "fn" in refresh_files_cb:
            refresh_files_cb["fn"]()

    def on_create_rag_action(_=None):
        open_create_rag_overlay(
            page,
            on_created=lambda new_path: set_workspace_path(new_path),
        )

    header_bar, update_header, refresh_pull_badge = build_header(
        state,
        on_open_workspace=lambda _=None: on_open_workspace(),
        on_create_rag=on_create_rag_action,
        on_pick_workspace=set_workspace_path,
        refresh_status=refresh_status,
    )
    refresh_status_cb["fn"] = update_header
    # Expose the badge refresher so the sysmon watchdog can tick it once a
    # second alongside its own samples — that way the user sees download
    # progress in the header even with the pull dialog closed.
    refresh_pull_badge_cb["fn"] = refresh_pull_badge

    # ---- views -----------------------------------------------------------

    # Will be populated below once build_settings_view returns. Chat view
    # error actions look it up at click-time, so it can be late-bound.
    open_pull_dialog_cb: dict = {}

    chat_view, render_chat, chat_input = build_chat_view(
        state,
        refresh_status=refresh_status,
        on_open_workspace=lambda _=None: on_open_workspace(),
        on_open_files=go_to_files,
        open_pull_dialog_cb=open_pull_dialog_cb,
    )
    render_chat_cb = {"fn": render_chat}

    files_view, refresh_files = build_files_view(
        state, refresh_status=refresh_status,
        refresh_files_cb=refresh_files_cb,
    )
    settings_view, load_settings, open_pull_dialog = build_settings_view(
        state, refresh_status=refresh_status,
        on_pick_workspace=set_workspace_path,
    )
    load_settings_cb = {"fn": load_settings}
    # Now that the settings view has been constructed, hand its
    # `open_pull_dialog` to the chat view's error-action factory.
    open_pull_dialog_cb["fn"] = open_pull_dialog
    doctor_view, refresh_doctor = build_doctor_view(state)
    refresh_doctor_cb["fn"] = refresh_doctor

    welcome_view, render_recents = build_welcome(
        state, on_open_workspace=on_open_workspace,
    )
    welcome_render_cb["fn"] = render_recents
    render_recents()

    # ---- nav rail --------------------------------------------------------

    # All possible rail destinations. We pick a subset based on whether
    # a workspace is open — Files / Settings / Doctor depend on a real
    # workspace, so we hide them until the user has one. Otherwise the
    # tabs would be there but every action would just toast "Open a
    # workspace first."
    _RAIL_FULL = [
        ft.NavigationRailDestination(
            icon=ft.Icons.CHAT_OUTLINED,
            selected_icon=ft.Icons.CHAT,
            label="Chat",
        ),
        ft.NavigationRailDestination(
            icon=ft.Icons.FOLDER_OUTLINED,
            selected_icon=ft.Icons.FOLDER,
            label="Files",
        ),
        ft.NavigationRailDestination(
            icon=ft.Icons.SETTINGS_OUTLINED,
            selected_icon=ft.Icons.SETTINGS,
            label="Settings",
        ),
        ft.NavigationRailDestination(
            icon=ft.Icons.MEDICAL_SERVICES_OUTLINED,
            selected_icon=ft.Icons.MEDICAL_SERVICES,
            label="Doctor",
        ),
    ]
    # When no workspace is open, only "Chat" (which routes to the welcome
    # screen with the Open / New RAG buttons) is reachable.
    _RAIL_NO_WORKSPACE = _RAIL_FULL[:1]

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        bgcolor=SURFACE_DARK,
        leading=ft.Container(height=8),
        destinations=_RAIL_NO_WORKSPACE,
    )

    def rebuild_rail():
        """Show Files / Settings / Doctor only when there's a workspace
        to act on. Called whenever set_workspace_path() runs."""
        new_dests = _RAIL_FULL if state.ws is not None else _RAIL_NO_WORKSPACE
        rail.destinations = new_dests
        # Clamp selection if we just shrank the rail (e.g. user was on
        # Settings and closed the workspace — collapse them to Chat).
        if rail.selected_index is None or rail.selected_index >= len(new_dests):
            rail.selected_index = 0
    rebuild_rail_cb["fn"] = rebuild_rail

    # ---- view stack ------------------------------------------------------

    main_area = ft.AnimatedSwitcher(
        chat_view,
        transition=ft.AnimatedSwitcherTransition.FADE,
        duration=200,
    )

    def switch_view(idx: int):
        # Clamp into the currently-visible destinations. Keyboard
        # shortcuts and external callers can request a tab the rail
        # doesn't currently show (e.g. Files when no workspace) — quietly
        # collapse to the welcome screen instead of breaking the rail.
        n_visible = len(rail.destinations or [])
        if idx >= n_visible:
            idx = 0
        rail.selected_index = idx
        if state.ws is None:
            main_area.content = welcome_view
        else:
            main_area.content = [chat_view, files_view, settings_view, doctor_view][idx]
        if idx == 0 and state.ws is not None:
            render_chat()
        elif idx == 1:
            refresh_files()
        elif idx == 2:
            load_settings()
        elif idx == 3:
            refresh_doctor()
        page.update()

    rail.on_change = lambda e: switch_view(int(e.control.selected_index))

    # ---- keyboard --------------------------------------------------------

    def on_keyboard(e: ft.KeyboardEvent):
        if not e.ctrl:
            return
        if e.key == ",":
            switch_view(2)
        elif e.key.lower() == "i":
            switch_view(1)
        elif e.key.lower() == "n":
            switch_view(0)

    page.on_keyboard_event = on_keyboard

    # ---- system telemetry status bar -------------------------------------
    # Pinned footer showing CPU / RAM / GPU / VRAM / temps in real time so
    # the user can see how loaded their machine is during ingest + chat.
    sysbar = _build_sysmon_bar(
        state, refresh_pull_badge_cb=refresh_pull_badge_cb,
    )

    # ---- assemble --------------------------------------------------------

    body = ft.Row(
        [
            rail,
            ft.VerticalDivider(width=1, color="#1E2130"),
            ft.Container(content=main_area, expand=True, bgcolor=BG_DARK),
        ],
        expand=True, spacing=0,
    )
    page.add(ft.Column([header_bar, body, sysbar],
                       expand=True, spacing=0))

    # If we're already inside a workspace, open it.
    pre = find_workspace()
    if pre is not None:
        set_workspace_path(pre.root)
    else:
        main_area.content = welcome_view
        page.update()


def run():  # pragma: no cover
    """Entry point for `ez-rag-gui`."""
    ft.app(target=app)


if __name__ == "__main__":  # pragma: no cover
    run()

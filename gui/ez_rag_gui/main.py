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

    chat_list = ft.ListView(
        expand=True, spacing=12, padding=ft.padding.all(20), auto_scroll=True,
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

            def download_chapter_clicked(_=None):
                """Experimental — extract just the chapter that contains
                this hit and save it as a standalone PDF. Uses the
                chapter metadata persisted at ingest time (PDF outline
                or section headings). Quality depends on how clean the
                source PDF's bookmarks are.
                """
                async def _do():
                    if state.ws is None:
                        _toast(state.page, "Open a workspace first.")
                        return
                    try:
                        from ez_rag.preview import extract_pdf_pages
                        from ez_rag.embed import make_embedder
                        from ez_rag.index import Index
                        from ez_rag.chapters import find_chapter
                        embedder = make_embedder(state.cfg)
                        idx = Index(state.ws.meta_db_path,
                                    embed_dim=embedder.dim)
                        chapters = idx.chapters_for_file(hit.file_id)
                        if not chapters:
                            _toast(state.page,
                                "No chapter metadata for this file. "
                                "Re-ingest after a recent ez-rag update "
                                "to populate it.")
                            return
                        # Look up the chunk's ord, then the chapter that
                        # contains it.
                        row = idx.conn.execute(
                            "SELECT ord FROM chunks WHERE id = ?",
                            (hit.chunk_id,),
                        ).fetchone()
                        ord_ = row[0] if row else None
                        ch = find_chapter(chapters, ord_) if ord_ is not None else None
                        if ch is None:
                            _toast(state.page,
                                "Couldn't locate the chapter for this "
                                "passage. Try Re-ingest (force).")
                            return
                        sp = ch.get("start_page")
                        ep = ch.get("end_page")
                        if not sp or not ep:
                            _toast(state.page,
                                "Chapter has no page range — can't "
                                "extract as PDF. (Non-PDF source?)")
                            return
                        title = ch.get("title") or f"chapter-{sp}-{ep}"
                        # Sanitize for filename
                        safe_title = "".join(
                            c if c.isalnum() or c in " ._-" else "_"
                            for c in title
                        ).strip()
                        suggested = (
                            f"{Path(hit.path).stem} - {safe_title} "
                            f"(pp {sp}-{ep}).pdf"
                        )
                    except Exception as ex:
                        _toast(state.page, f"Lookup failed: {ex}")
                        return

                    try:
                        dest = await download_picker.save_file(
                            dialog_title="Save chapter as PDF",
                            file_name=suggested,
                            allowed_extensions=["pdf"],
                        )
                    except Exception as ex:
                        _toast(state.page, f"Save dialog failed: {ex}")
                        return
                    if not dest:
                        return

                    def _bg():
                        abs_pdf = (state.ws.root / hit.path).resolve()
                        out = extract_pdf_pages(
                            abs_pdf, sp, ep, Path(dest), title=title,
                        )
                        if out is None:
                            _toast(state.page,
                                "Chapter extract failed. pypdf may be "
                                "missing, or the page range exceeded "
                                "the document.")
                        else:
                            pages = ep - sp + 1
                            _toast(state.page,
                                f"Saved chapter '{title}' "
                                f"({pages} pages) → {Path(out).name}")
                    state.page.run_thread(_bg)
                state.page.run_task(_do)

            toolbar = ft.Row([
                ft.Text(f"page {hit.page} · 2.5x render",
                        size=11, color=ON_SURFACE_DIM),
                ft.Container(expand=True),
                ft.OutlinedButton(
                    "Chapter (experimental)",
                    icon=ft.Icons.AUTO_STORIES,
                    on_click=download_chapter_clicked,
                    tooltip=("Save just the chapter containing this "
                             "passage as a standalone PDF. Boundaries "
                             "come from the source PDF's bookmarks "
                             "(or section headings) — quality varies."),
                ),
                ft.OutlinedButton(
                    "Download", icon=ft.Icons.DOWNLOAD,
                    on_click=download_clicked,
                    tooltip="Save this page image as a PNG.",
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

    def _scroll_to_bottom(duration: int = 120):
        """Best-effort jump to the latest message.

        Flet's ListView.auto_scroll handles new appends but not full rebuilds
        or in-place updates of existing children, so we trigger explicitly.
        """
        try:
            chat_list.scroll_to(offset=-1, duration=duration)
        except Exception:
            pass

    def render_chat():
        """Full re-render. Call when turns are added/removed."""
        chat_list.controls.clear()
        if not state.turns:
            chat_list.controls.append(_chat_welcome(state))
        else:
            for t in state.turns:
                chat_list.controls.append(_bubble(t))
        state.page.update()
        _scroll_to_bottom()

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
        render_chat()
        set_busy(True)

        def worker():
            try:
                # Apply prefix/suffix/negatives if the per-query toggle is on
                effective_q = apply_query_modifiers(text, state.cfg)

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

                    if state.cfg.agentic:
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
                        )
                        assistant.text = ans.text  # type: ignore
                    update_streaming_assistant(assistant)
                else:
                    state.stop_flag = False
                    last_render = 0.0
                    for kind, piece in chat_answer(
                        history=history, latest_question=effective_q,
                        hits=hits, cfg=state.cfg, stream=True,
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

    container = ft.Container(
        bgcolor=BG_DARK,
        expand=True,
        content=ft.Column(
            [chat_toolbar, ft.Divider(height=1, color="#1E2130"),
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
        ingest_progress,
        ingest_status,
        ingest_meta,
        ingest_done_card,
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

    container = ft.Container(
        bgcolor=BG_DARK,
        expand=True,
        padding=20,
        content=ft.Column(
            [
                ft.Text("Files", size=18, weight=ft.FontWeight.W_700,
                        color=ON_SURFACE),
                ft.Container(height=12),
                ft.Row([
                    ft.Container(content=docs_card, expand=2),
                    ft.Container(content=ingest_card, expand=1),
                ], spacing=14, vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                expand=True),
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
    enable_ocr = ft.Switch(label="OCR images / scanned PDFs", value=True,
                           active_color=ACCENT, tooltip=TIP["enable_ocr"])
    contextual = ft.Switch(label="Contextual Retrieval (slower ingest, better recall)",
                           value=False, active_color=ACCENT,
                           tooltip=TIP["contextual"])

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
        unload_llm_sw.value = c.unload_llm_during_ingest
        embed_batch_field.value = str(c.embed_batch_size)
        num_batch_field.value = str(getattr(c, "num_batch", 1024))
        num_ctx_field.value = str(getattr(c, "num_ctx", 0))
        use_corpus.value = c.use_rag
        enable_ocr.value = c.enable_ocr
        contextual.value = c.enable_contextual
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
        except ValueError as ex:
            _toast(state.page, f"Bad value: {ex}")

    body = ft.ListView(
        spacing=14,
        padding=ft.padding.symmetric(horizontal=20, vertical=20),
        expand=True,
        controls=[
            ft.Text("Settings", size=18, weight=ft.FontWeight.W_700,
                    color=ON_SURFACE),
            ft.Row(
                [
                    ft.Container(
                        expand=1,
                        content=section_card(
                            "INGEST",
                            ft.Row([chunk_size, chunk_overlap], spacing=10, wrap=True),
                            enable_ocr,
                            contextual,
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

    def set_workspace_path(path: Path):
        ws = Workspace(path)
        if not ws.is_initialized():
            ws.initialize()
        state.ws = ws
        state.cfg = ws.load_config()
        add_recent(ws.root)
        clear_embedder_cache()
        if "fn" in refresh_files_cb:
            refresh_files_cb["fn"]()
        if "fn" in load_settings_cb:
            load_settings_cb["fn"]()
        if "fn" in refresh_status_cb:
            refresh_status_cb["fn"]()
        if "fn" in refresh_doctor_cb:
            refresh_doctor_cb["fn"]()
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

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        bgcolor=SURFACE_DARK,
        leading=ft.Container(height=8),
        destinations=[
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
        ],
    )

    # ---- view stack ------------------------------------------------------

    main_area = ft.AnimatedSwitcher(
        chat_view,
        transition=ft.AnimatedSwitcherTransition.FADE,
        duration=200,
    )

    def switch_view(idx: int):
        rail.selected_index = idx
        if state.ws is None and idx != 0:
            # Force welcome when no workspace.
            main_area.content = welcome_view
        elif state.ws is None:
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

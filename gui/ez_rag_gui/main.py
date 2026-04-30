"""ez-rag GUI — Flet desktop app.

Runs the same library code as the CLI; nothing here re-implements business logic.
Primary view is Chat. Files / Settings / Doctor are the supporting tabs in a
NavigationRail on the left.
"""
from __future__ import annotations

import os
import shutil
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
    pull_ollama_model, search_library, vram_fit,
)
from ez_rag.parsers import supported_extensions
from ez_rag.retrieve import hybrid_search, smart_retrieve
from ez_rag.workspace import Workspace, find_workspace


# ============================================================================
# Theme
# ============================================================================

ACCENT = "#7C7BFF"            # indigo accent
ACCENT_SOFT = "#5856D6"
BG_DARK = "#0F1115"           # near-black, slight blue
SURFACE_DARK = "#171922"
SURFACE_DARK_HI = "#1F2230"
ON_SURFACE = "#E6E7EB"
ON_SURFACE_DIM = "#9097A6"
SUCCESS = "#3DDC84"
WARNING = "#F6B042"
DANGER = "#F75A68"
USER_BUBBLE = "#272A39"
ASSIST_BUBBLE = "#1A1D29"
CHIP_BG = "#23263A"

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
    "rag_toggle":     "USE CORPUS — when ON, ez-rag retrieves from your docs "
                      "before answering and cites passages [1], [2], …\n"
                      "When OFF, the question goes straight to the LLM with no "
                      "retrieval. Useful for A/B testing how much the corpus "
                      "actually helps.",

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
Toggle the **Use corpus** switch off (top of the Chat tab) to test the model
with no retrieval.

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

**Test with vs without RAG**: toggle *Use corpus* in the Chat tab header.
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
    bubble: object = None


@dataclass
class AppState:
    page: ft.Page
    ws: Workspace | None = None
    cfg: Config = field(default_factory=Config)
    turns: list[ChatTurn] = field(default_factory=list)
    streaming: bool = False
    stop_flag: bool = False


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

def build_header(state: AppState, *, on_open_workspace, refresh_status):
    workspace_text = ft.Text("(no workspace)", size=14, color=ON_SURFACE_DIM)
    backend_pill = status_pill("offline", DANGER)
    backend_pill.tooltip = TIP["backend_pill"]
    files_pill = status_pill("0 files", ACCENT)
    files_pill.tooltip = TIP["files_pill"]
    btn_open = ft.OutlinedButton(
        "Open workspace",
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
                btn_open,
                btn_help,
                btn_about,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
    )

    def update():
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

    return bar, update


# ============================================================================
# Chat view
# ============================================================================

def build_chat_view(state: AppState, *, refresh_status,
                    on_open_workspace, on_open_files):
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
        "Use corpus" if state.cfg.use_rag else "Bypass corpus (model only)",
        size=12, color=ON_SURFACE_DIM, weight=ft.FontWeight.W_600,
    )

    def on_rag_toggle(e):
        state.cfg.use_rag = bool(e.control.value)
        rag_label.value = (
            "Use corpus" if state.cfg.use_rag
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

    def open_source(idx: int, hit):
        loc = hit.path + (f"  ·  page {hit.page}" if hit.page else "")
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
        source_dialog.content = ft.Container(
            width=720, height=480, padding=10,
            content=ft.Column([
                ft.Container(
                    bgcolor=SURFACE_DARK_HI, border_radius=8, padding=14,
                    content=ft.Column([
                        ft.Text(hit.text, size=13, selectable=True,
                                color=ON_SURFACE),
                    ], scroll=ft.ScrollMode.AUTO, expand=True),
                    expand=True,
                ),
            ], expand=True),
        )
        source_dialog.actions = [
            ft.TextButton("Close",
                          on_click=lambda _: state.page.pop_dialog()),
        ]
        state.page.show_dialog(source_dialog)

    # ------------- bubble rendering --------------------------------------

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

        bubble_items = []
        if not is_user:
            bubble_items.append(thinking_box)
        bubble_items.append(md)
        bubble_items.append(chips_row)

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

    def render_chat():
        """Full re-render. Call when turns are added/removed."""
        chat_list.controls.clear()
        if not state.turns:
            chat_list.controls.append(_chat_welcome(state))
        else:
            for t in state.turns:
                chat_list.controls.append(_bubble(t))
        state.page.update()

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
        try:
            chat_list.scroll_to(offset=-1, duration=80)
        except Exception:
            pass

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
                # Honor the "Use corpus" toggle. When OFF we skip embedding +
                # retrieval entirely, sending the question straight to the LLM
                # with no context.
                if state.cfg.use_rag:
                    embedder = make_embedder(state.cfg)
                    idx = Index(state.ws.meta_db_path, embed_dim=embedder.dim)
                    # smart_retrieve honors hybrid / rerank / hyde / multi_query
                    hits = smart_retrieve(
                        query=text, embedder=embedder, index=idx, cfg=state.cfg,
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
                            history=history, latest_question=text,
                            hits=hits, cfg=state.cfg, stream=False,
                        )
                        assistant.text = ans.text  # type: ignore
                    update_streaming_assistant(assistant)
                else:
                    state.stop_flag = False
                    last_render = 0.0
                    for kind, piece in chat_answer(
                        history=history, latest_question=text,
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

    composer = ft.Container(
        padding=ft.padding.only(left=20, right=20, top=10, bottom=16),
        content=ft.Container(
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
        prompt_text = (
            "Try one of:" if s["files"] > 0 else
            "Add documents to start asking questions."
        )
        suggestions = [
            "Summarize the corpus.",
            "What topics are covered?",
            "List the documents and their main points.",
        ]

        def fill(text: str):
            input_field.value = text
            state.page.update()

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
                    ft.Container(height=18),
                    ft.Text(prompt_text, size=12, color=ON_SURFACE_DIM),
                    ft.Container(height=8),
                    ft.Column(
                        [
                            ft.Container(
                                content=ft.Text(s, size=13, color=ON_SURFACE),
                                bgcolor=SURFACE_DARK,
                                border=ft.border.all(1, "#262938"),
                                border_radius=10,
                                padding=ft.padding.symmetric(horizontal=14, vertical=10),
                                on_click=(lambda s=s: lambda _: fill(s))(s),
                                width=520,
                            )
                            for s in suggestions
                        ],
                        spacing=8,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
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
        ingest_progress.visible = True
        ingest_progress.value = None
        ingest_status.value = "Starting…"
        ingest_meta.value = ""
        ingest_snippet_card.visible = False
        state.page.update()

        last_log_path = {"v": ""}
        last_snippet_t = {"v": 0.0}

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

        def progress_cb(prog):
            # Determinate progress bar (bytes-based)
            if prog.bytes_total > 0:
                ingest_progress.value = min(1.0, prog.bytes_pct)
            ingest_status.value = (
                f"{prog.status} — {Path(prog.current_path).name}"
                if prog.current_path
                else prog.status
            )
            ingest_meta.value = (
                f"{prog.files_done}/{prog.files_total} files  ·  "
                f"{fmt_bytes(prog.bytes_done)} / {fmt_bytes(prog.bytes_total)}  ·  "
                f"{fmt_bytes(int(prog.rate_bps))}/s  ·  ETA {fmt_eta(prog.eta_s)}  ·  "
                f"{prog.chunks_done} chunks  ·  index {fmt_bytes(prog.db_bytes)}"
            )

            # Append a one-line log entry only on file transitions
            if prog.current_path and prog.current_path != last_log_path["v"]:
                last_log_path["v"] = prog.current_path
                ingest_log.controls.append(
                    ft.Text(
                        f"{prog.status[:18]:>18}  {Path(prog.current_path).name}",
                        size=11, color=ON_SURFACE_DIM,
                        font_family="monospace",
                    )
                )

            # Show a snippet card every ~15 s while parsing/embedding
            now = time.monotonic()
            if (prog.snippet and
                    (now - last_snippet_t["v"]) >= 15.0):
                page_str = f"  ·  page {prog.page}" if prog.page else ""
                ingest_snippet_path.value = (
                    f"{Path(prog.current_path).name}{page_str}"
                )
                ingest_snippet_text.value = "“" + prog.snippet + "…”"
                ingest_snippet_card.visible = True
                last_snippet_t["v"] = now

            state.page.update()

        def worker():
            try:
                stats = ingest(state.ws, cfg=state.cfg, force=force,
                               progress=progress_cb)
                ingest_status.value = (
                    f"Done · {stats.files_new} new · "
                    f"{stats.files_changed} changed · "
                    f"{stats.files_skipped_unchanged} unchanged · "
                    f"{stats.chunks_added} chunks in "
                    f"{stats.seconds:.1f}s"
                )
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
            finally:
                ingest_progress.visible = False
                refresh_files()
                refresh_status()
                state.page.update()

        state.page.run_thread(worker)

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


def build_settings_view(state: AppState, *, refresh_status):
    page = state.page

    # Local FilePicker for choosing a GGUF file
    gguf_picker = ft.FilePicker()
    page.services.append(gguf_picker)

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
    use_corpus = ft.Switch(label="Use corpus (RAG)", value=True,
                           active_color=ACCENT, tooltip=TIP["use_corpus"])
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

    def open_pull_dialog(kind: str, target_dropdown: ft.Dropdown):
        capability_filter = "embedding" if kind == "embed" else None
        gpu_total_vram = detect_total_vram_gb()

        search_field = ft.TextField(
            hint_text="Search the Ollama library…",
            prefix_icon=ft.Icons.SEARCH,
            expand=True, dense=True,
            border_color="#2A2D3D",
            focused_border_color=ACCENT,
            cursor_color=ACCENT,
            text_size=13,
        )
        tag_field = ft.TextField(
            label="Tag",
            hint_text="qwen2.5:7b-instruct  (auto-fills when you click a size below)",
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
                progress_text,
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
            pull_btn.disabled = True
            cancel_btn.disabled = True
            search_field.disabled = True
            progress.visible = True
            progress.value = None
            progress_text.value = "Starting…"
            page.update()

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

                        if total and completed:
                            pct = completed / total
                            progress.value = min(1.0, pct)
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
                            progress_text.value = (
                                f"{status}  ·  {pct*100:.1f}%  ·  "
                                f"{done_g:.2f} / {tot_g:.2f} GB  ·  "
                                f"{rate_str}  ·  ETA {eta_str}"
                            )
                        else:
                            progress_text.value = status
                            if status not in ("success",):
                                progress.value = None

                        # Repaint at most ~10 fps so we don't flood Flet.
                        if now - last_paint_t > 0.1 or status == "success":
                            page.update()
                            last_paint_t = now

                        if status == "success":
                            break

                    elapsed = time.monotonic() - start_t
                    progress.value = 1.0
                    progress_text.value = (
                        f"✓ Pulled {tag} in {elapsed:.1f}s"
                        if elapsed < 60
                        else f"✓ Pulled {tag} in {int(elapsed//60)}m{int(elapsed%60):02d}s"
                    )
                    pull_btn.disabled = False
                    cancel_btn.disabled = False
                    search_field.disabled = False
                    page.update()
                    refresh_model_dropdowns()
                    target_dropdown.value = _norm_tag(tag)
                    page.update()
                    # Auto-close — overlay visibility toggles propagate fine
                    # from a worker thread (unlike DialogControl).
                    close_dialog()
                except Exception as e:
                    progress.value = 0
                    progress_text.value = f"Error: {e}"
                    pull_btn.disabled = False
                    cancel_btn.disabled = False
                    search_field.disabled = False
                    page.update()

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
                            ft.Row([use_hyde_sw, multi_query_sw], spacing=14,
                                   wrap=True,
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Row([context_window_field, use_mmr_sw,
                                    mmr_lambda_field], spacing=14,
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
    return container, load_from_cfg


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
# Main
# ============================================================================

def app(page: ft.Page):
    page.title = "ez-rag"
    page.theme_mode = ft.ThemeMode.DARK
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

    header_bar, update_header = build_header(
        state,
        on_open_workspace=lambda _=None: on_open_workspace(),
        refresh_status=refresh_status,
    )
    refresh_status_cb["fn"] = update_header

    # ---- views -----------------------------------------------------------

    chat_view, render_chat, chat_input = build_chat_view(
        state,
        refresh_status=refresh_status,
        on_open_workspace=lambda _=None: on_open_workspace(),
        on_open_files=go_to_files,
    )
    render_chat_cb = {"fn": render_chat}

    files_view, refresh_files = build_files_view(
        state, refresh_status=refresh_status,
        refresh_files_cb=refresh_files_cb,
    )
    settings_view, load_settings = build_settings_view(
        state, refresh_status=refresh_status,
    )
    load_settings_cb = {"fn": load_settings}
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

    # ---- assemble --------------------------------------------------------

    body = ft.Row(
        [
            rail,
            ft.VerticalDivider(width=1, color="#1E2130"),
            ft.Container(content=main_area, expand=True, bgcolor=BG_DARK),
        ],
        expand=True, spacing=0,
    )
    page.add(ft.Column([header_bar, body], expand=True, spacing=0))

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

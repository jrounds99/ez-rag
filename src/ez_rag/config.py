"""Workspace + global configuration. Stored as TOML in <workspace>/.ezrag/config.toml."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class Config:
    # Models
    llm_provider: str = "auto"          # "auto" | "ollama" | "llama-cpp" | "none"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_url: str = "http://127.0.0.1:11434"  # Ollama default
    embedder_provider: str = "auto"     # "auto" | "ollama" | "fastembed"
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    ollama_embed_model: str = "nomic-embed-text"

    # Ingest
    chunk_size: int = 512
    chunk_overlap: int = 64
    enable_ocr: bool = True
    ocr_provider: str = "auto"          # "auto" | "rapidocr" | "tesseract" | "none"
    enable_contextual: bool = False     # Anthropic-style chunk context (slower, better recall)
    # LLM-assisted text-quality inspection. When ON, every parsed section
    # is sent to the LLM during ingest with a "is this garbled?" prompt;
    # garbled sections are dropped. Catches font-cmap or OCR failures the
    # heuristic detector missed. EXPENSIVE — one LLM call per section,
    # so a 200-section book is 200 calls. Default OFF.
    llm_inspect_pages: bool = False
    # LLM-assisted correction of questionable text. When ON, sections the
    # heuristic flags as garbled (or that the LLM inspector flags as
    # "partial") are sent to the LLM for a best-effort cleanup pass before
    # being indexed. The LLM may also reject as UNRECOVERABLE, in which
    # case the section is dropped. Costs an extra LLM call per flagged
    # section; off by default. Independent of llm_inspect_pages — turn
    # both on for the most aggressive recovery.
    llm_correct_garbled: bool = False
    # Live preview of garbled-page recoveries during ingest. When ON, the
    # parser saves a 2× page render + before/after text excerpts so the
    # GUI can show a real-time "what we saw vs. what we fixed" card as
    # ingest works through bad pages. Costs ~50ms + ~200KB disk per
    # recovered page; off by default to keep ingest as lean as possible.
    preview_garbled_recoveries: bool = False

    # Ingest performance — flip these on for big corpora
    unload_llm_during_ingest: bool = True   # frees VRAM if contextual is OFF
    parallel_workers: int = 1               # parser/chunker process pool (1 = sequential)
    embed_batch_size: int = 16              # bigger = faster on a strong GPU

    # Retrieval
    use_rag: bool = True               # off = ask the LLM directly
    top_k: int = 8
    hybrid: bool = True                # BM25 + dense fused with RRF
    rerank: bool = True                # cross-encoder rerank — biggest single quality lift
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    use_hyde: bool = False             # query expansion via hypothetical answer
    multi_query: bool = False          # fan out to N LLM-generated paraphrases
    # Auto-detect "list X / name some X / give examples of X" queries
    # and route them through entity-rich HyDE + an extraction-only
    # system prompt. Dramatically improves answers to open-ended
    # exploratory questions ("list NPCs from the books", "interesting
    # magic items for level 5", etc.) on small/medium models.
    # On by default — costs one extra LLM call per detected query.
    auto_list_mode: bool = True
    # Cap chunks per source file in retrieval results. 0 = no cap.
    # 3 is a strong default — forces the LLM to ground answers across
    # multiple sources instead of letting one PDF dominate top-K.
    diversify_per_source: int = 3
    # Reorder retrieved chunks so highest-rank are at start AND end of
    # the prompt — combats the "lost in the middle" effect on LONG
    # contexts. On our bench (qwen2.5:7b, ~16 KB context after expand)
    # the LLM-as-judge slightly preferred original RRF order (10.70 vs
    # 10.30), so OFF by default. Worth turning ON for very long
    # contexts (>32 KB) where the middle attention drop is real.
    reorder_for_attention: bool = False
    # CRAG-style chunk relevance filter — after retrieval, ask the LLM
    # in ONE batched call which retrieved chunks are actually relevant
    # to the query and drop the rest. Costs +1 small LLM call per query.
    # Off by default; turn on for noisy corpora where retrieval pulls in
    # off-topic chunks (e.g. a corpus mixing rules + adventure modules).
    crag_filter: bool = False
    # Hard cap for auto-sized num_ctx (0 = let model's native max win).
    # Lower this if you're VRAM-constrained — 8192 keeps memory modest
    # at the cost of dropping prompt content past ~6 KB.
    num_ctx_cap: int = 0
    context_window: int = 0            # include N neighbor chunks per hit (0 = off)
    use_mmr: bool = False              # diversify retrieved chunks via MMR
    mmr_lambda: float = 0.5            # 1.0 = pure relevance, 0.0 = pure diversity
    expand_to_chapter: bool = False    # replace each hit's text with its full chapter
    # Chapter expansion cap. 8000 chars (~2K tokens) tested as the sweet
    # spot — bigger values caused the LLM to summarize chapters instead
    # of extracting from them. List queries auto-tighten this further.
    chapter_max_chars: int = 8000

    # Agentic retrieval — LLM-driven iterative search.
    # When ON, the LLM evaluates the initial retrieval and (if needed) generates
    # follow-up queries; results are fused with RRF + reranked once at the end.
    agentic: bool = False              # single on/off — sane defaults below
    agent_max_iterations: int = 2      # retrieve→reflect cycles
    agent_provider: str = "same"       # same | openai | anthropic
    agent_model: str = ""              # blank = use cfg.llm_model
    agent_api_key: str = ""            # for openai / anthropic
    agent_base_url: str = "https://api.openai.com/v1"  # OpenAI-compatible endpoint

    # Generation
    max_tokens: int = 4096
    temperature: float = 0.2
    # Tokens of context Ollama allocates for the model. 0 = use Ollama's
    # default for the model. Larger = more memory, no per-token speed cost
    # if actual sequence is short.
    num_ctx: int = 0
    # Prompt-eval batch size. Bigger = faster TTFT (prompt processing) at
    # the cost of more peak memory. Default 512 (Ollama's default); 1024
    # was measured to give ~+8% throughput / -23% TTFT on a 32B model.
    num_batch: int = 1024

    # Server
    serve_host: str = "127.0.0.1"
    serve_port: int = 11533

    # Query modifiers — applied to every question when the chat-tab toggle
    # is on (it mirrors `apply_query_modifiers`).
    apply_query_modifiers: bool = True
    query_prefix: str = ""             # text prepended to each question
    query_suffix: str = ""             # text appended
    query_negatives: str = ""          # added as "Avoid: …" constraint
    # Per-file metadata sidecars — when True, `<file>.ezrag-meta.toml`
    # files alongside source documents (or under <workspace>/.ezrag/
    # file_meta/) are read at query time. Modifiers with scope="global"
    # are merged into every query; "topic-aware" / "file-only" scopes
    # are applied post-retrieval.
    use_file_metadata: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            return cls()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                cfg.extra[k] = v
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# ez-rag configuration", ""]
        for k, v in asdict(self).items():
            if k == "extra":
                continue
            if isinstance(v, str):
                # TOML literal strings (single-quoted) don't interpret
                # backslashes — needed for Windows paths and any user-typed
                # query modifiers. Fall back to escaped quoted form when the
                # value contains a single quote.
                if "'" in v:
                    escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                    lines.append(f'{k} = "{escaped}"')
                else:
                    lines.append(f"{k} = '{v}'")
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                lines.append(f"{k} = {v}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

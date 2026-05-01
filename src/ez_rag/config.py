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
    context_window: int = 0            # include N neighbor chunks per hit (0 = off)
    use_mmr: bool = False              # diversify retrieved chunks via MMR
    mmr_lambda: float = 0.5            # 1.0 = pure relevance, 0.0 = pure diversity
    expand_to_chapter: bool = False    # replace each hit's text with its full chapter
    chapter_max_chars: int = 16000     # safety cap so big chapters don't blow context

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

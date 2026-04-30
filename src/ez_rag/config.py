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

    # Generation
    max_tokens: int = 4096
    temperature: float = 0.2

    # Server
    serve_host: str = "127.0.0.1"
    serve_port: int = 11533

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
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                lines.append(f"{k} = {v}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

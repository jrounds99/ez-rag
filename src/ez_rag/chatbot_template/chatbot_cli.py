"""CLI chatbot — same retrieval and generation as server.py, terminal UI.

Usage:
    python chatbot_cli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ezrag_lib.config import Config
from ezrag_lib.embed import make_embedder
from ezrag_lib.index import Index, read_stats
from ezrag_lib.generate import (
    apply_query_modifiers, chat_answer, detect_backend,
)
from ezrag_lib.retrieve import agentic_retrieve, smart_retrieve


DATA = ROOT / "data"


def main():
    cfg = Config.load(DATA / "config.toml")
    print(f"loading embedder ({cfg.embedder_provider} / {cfg.embedder_model}) ...")
    embedder = make_embedder(cfg)
    print(f"opening index ({DATA / 'meta.sqlite'}) ...")
    index = Index(DATA / "meta.sqlite", embed_dim=embedder.dim)
    stats = read_stats(DATA / "meta.sqlite") or {}

    backend = detect_backend(cfg)
    print()
    print(f"  backend : {backend}")
    print(f"  model   : {cfg.llm_model}")
    print(f"  index   : {stats.get('files', 0)} files, "
          f"{stats.get('chunks', 0)} chunks")
    print(f"  RAG     : {'on' if cfg.use_rag else 'off'}")
    if backend == "none":
        print()
        print("  WARNING: no LLM backend detected. Install Ollama and pull")
        print(f"           the model ({cfg.llm_model!r}), then re-run.")
    print()
    print("Type a question and press Enter. Ctrl+C or 'exit' to quit.")
    print()

    history: list[dict] = []
    try:
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", ":q"):
                break

            question = apply_query_modifiers(q, cfg)

            if cfg.use_rag:
                if getattr(cfg, "agentic", False):
                    hits = agentic_retrieve(query=question, embedder=embedder,
                                            index=index, cfg=cfg)
                else:
                    hits = smart_retrieve(query=question, embedder=embedder,
                                          index=index, cfg=cfg)
            else:
                hits = []

            history.append({"role": "user", "content": q})
            print("\nbot> ", end="", flush=True)

            stream = chat_answer(history=history[:-1], latest_question=question,
                                 hits=hits, cfg=cfg, stream=True)

            answer_text = ""
            in_thinking = False
            if hasattr(stream, "__iter__") and not hasattr(stream, "text"):
                for kind, text in stream:
                    if kind == "thinking":
                        if not in_thinking:
                            print("\n  [reasoning]\n  ", end="", flush=True)
                            in_thinking = True
                        print(text, end="", flush=True)
                    else:
                        if in_thinking:
                            print("\n", end="")
                            in_thinking = False
                        print(text, end="", flush=True)
                        answer_text += text
            else:
                answer_text = stream.text
                print(answer_text, end="")

            history.append({"role": "assistant", "content": answer_text})
            print()

            if hits:
                print("\n  citations:")
                for i, h in enumerate(hits, 1):
                    page = f" p.{h.page}" if h.page else ""
                    print(f"    [{i}] {h.path}{page}")
            print()
    except KeyboardInterrupt:
        pass

    print("bye.")


if __name__ == "__main__":
    main()

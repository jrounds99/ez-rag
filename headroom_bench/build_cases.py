"""Build realistic ez-rag RAG prompts for the headroom compression bench.

For each (embedder workspace x question x context-depth) we run real
ez-rag retrieval over the ingested Ohio corpus and assemble the exact
(system, user) messages ez-rag would send to the LLM. Output: a JSON
list of cases that the WSL-side headroom compressor consumes.

We retrieve once per (embedder, question) at the max depth, then
truncate to each smaller depth to keep retrieval cheap. The user
prompt is built with ez-rag's own `_build_user_prompt`, so the
compression test reflects production prompts exactly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder, clear_embedder_cache
from ez_rag.generate import SYSTEM_PROMPT_RAG, _build_user_prompt
from ez_rag.index import Index
from ez_rag.retrieve import smart_retrieve
from ez_rag.workspace import Workspace

WORKSPACES = [
    ("qwen3-emb-8b", REPO / "bench" / "_ohio-workspaces" / "ws-qwen3-emb-8b"),
    ("nomic-text",   REPO / "bench" / "_ohio-workspaces" / "ws-nomic-text"),
    ("bge-m3",       REPO / "bench" / "_ohio-workspaces" / "ws-bge-m3"),
]
DEPTHS = [5, 10, 15, 20]          # context sizes (top-k) per case
MAX_DEPTH = max(DEPTHS)


def main() -> int:
    questions = json.loads(
        (REPO / "bench" / "ohio_questions.json").read_text(encoding="utf-8")
    )["questions"]

    cases: list[dict] = []
    case_id = 0
    for emb_label, ws_root in WORKSPACES:
        if not (ws_root / ".ezrag" / "meta.sqlite").is_file():
            print(f"[skip] {emb_label}: no index at {ws_root}")
            continue
        print(f"[{emb_label}] loading workspace {ws_root}")
        ws = Workspace(ws_root)
        cfg = ws.load_config()
        cfg.top_k = MAX_DEPTH
        clear_embedder_cache()
        embedder = make_embedder(cfg)
        index = Index(ws.meta_db_path, embed_dim=embedder.dim)

        for q_obj in questions:
            q = q_obj["q"]
            cat = q_obj.get("category", "?")
            t0 = time.perf_counter()
            try:
                hits = smart_retrieve(query=q, embedder=embedder,
                                       index=index, cfg=cfg)
            except Exception as ex:
                print(f"  [retrieval err] {q[:40]}: {ex}")
                continue
            dt = time.perf_counter() - t0
            print(f"  [{emb_label}] {dt:4.1f}s {len(hits):2d} hits — {q[:50]}")

            for depth in DEPTHS:
                sub = hits[:depth]
                if not sub:
                    continue
                user_prompt = _build_user_prompt(q, sub)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_RAG},
                    {"role": "user", "content": user_prompt},
                ]
                cases.append({
                    "case_id": case_id,
                    "embedder": emb_label,
                    "question": q,
                    "category": cat,
                    "depth": depth,
                    "n_hits": len(sub),
                    "messages": messages,
                    "sources": [
                        f"{getattr(h, 'path', '?')}:p{getattr(h, 'page', '?')}"
                        for h in sub
                    ],
                })
                case_id += 1

    out = REPO / "headroom_bench" / "cases.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {len(cases)} cases -> {out}")
    # quick depth histogram
    from collections import Counter
    by_depth = Counter(c["depth"] for c in cases)
    by_emb = Counter(c["embedder"] for c in cases)
    print(f"     by depth   : {dict(sorted(by_depth.items()))}")
    print(f"     by embedder: {dict(by_emb)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

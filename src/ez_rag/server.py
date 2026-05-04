"""Tiny OpenAI-compatible HTTP server using only the stdlib.

Exposes:
    POST /v1/chat/completions

Each call retrieves from the workspace and answers grounded in the corpus.
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .embed import make_embedder
from .generate import answer as gen_answer
from .index import Index
from .retrieve import hybrid_search
from .workspace import Workspace


def run_server(ws: Workspace, host: str, port: int) -> None:
    cfg = ws.load_config()
    embedder = make_embedder(cfg)
    index = Index(ws.meta_db_path, embed_dim=embedder.dim)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass  # quiet

        def _json(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": cfg.llm_model, "object": "model"}
                ]})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid json"})
                return
            messages = body.get("messages", [])
            user = next((m["content"] for m in reversed(messages)
                         if m.get("role") == "user"), "")
            if not user:
                self._json(400, {"error": "no user message"})
                return
            hits = hybrid_search(query=user, embedder=embedder,
                                 index=index, k=cfg.top_k)
            ans = gen_answer(question=user, hits=hits, cfg=cfg, stream=False)
            text = ans.text  # type: ignore
            self._json(200, {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": cfg.llm_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "ez_rag_citations": [
                    {"path": h.path, "page": h.page, "score": h.score}
                    for h in hits
                ],
            })

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"ez-rag serving on http://{host}:{port}  (POST /v1/chat/completions)")
    print("Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
        httpd.server_close()

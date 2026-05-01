"""Round 3: Agentic retrieval + query modifiers + agent provider dispatch.

External providers (OpenAI, Anthropic) are stubbed at the httpx layer so no
network is touched. The 'same' provider path is exercised by stubbing the
backend detector + chat helper.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.embed import make_embedder
from ez_rag.index import Index
from ez_rag.ingest import ingest
from ez_rag import generate as gen
from ez_rag.retrieve import agentic_retrieve, smart_retrieve
from ez_rag.workspace import Workspace


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


DOCS = {
    "dogs.md": (
        "Border Collies are intelligent herding dogs.\n\n"
        "Golden Retrievers are gentle family dogs.\n"
    ),
    "cooking.md": (
        "Sourdough relies on wild yeast.\n\n"
        "Maillard reaction creates browning at 140 °C.\n"
    ),
    "physics.md": (
        "Speed of light is 299792458 m/s.\n\n"
        "Newton's second law: F=ma.\n"
    ),
}


def make_tmp_ws() -> Workspace:
    tmp = Path(tempfile.mkdtemp(prefix="ezrag_round3_"))
    ws = Workspace(tmp)
    ws.initialize()
    for name, body in DOCS.items():
        (ws.docs_dir / name).write_text(body, encoding="utf-8")
    return ws


# ---------------------------------------------------------------------------
# Query modifiers
# ---------------------------------------------------------------------------

def test_apply_modifiers_off():
    print("\n[1] apply_query_modifiers=False -> bare question")
    cfg = Config(apply_query_modifiers=False,
                 query_prefix="PREFIX", query_suffix="SUFFIX",
                 query_negatives="NEG")
    out = gen.apply_query_modifiers("hi", cfg)
    check("returns bare question", out == "hi", f"got {out!r}")


def test_apply_modifiers_full():
    print("\n[2] apply_query_modifiers full pipeline")
    cfg = Config(apply_query_modifiers=True,
                 query_prefix="Be concise.",
                 query_suffix="Cite passages.",
                 query_negatives="speculation, hedging")
    out = gen.apply_query_modifiers("what is X?", cfg)
    check("contains question", "what is X?" in out, f"got {out!r}")
    check("contains prefix", "Be concise." in out, "")
    check("contains suffix", "Cite passages." in out, "")
    check("contains Avoid: directive",
          "Avoid: speculation, hedging" in out, f"got {out!r}")
    # ordering: prefix, question, suffix, avoid
    parts = out.split("\n\n")
    check("ordering: prefix first",
          parts[0] == "Be concise.", f"parts={parts}")
    check("ordering: question second",
          parts[1] == "what is X?", f"parts={parts}")


def test_apply_modifiers_partial():
    print("\n[3] only some modifiers set")
    cfg = Config(apply_query_modifiers=True, query_prefix="Hi.",
                 query_suffix="", query_negatives="")
    out = gen.apply_query_modifiers("q", cfg)
    check("prefix-only output",
          out == "Hi.\n\nq", f"got {out!r}")

    cfg = Config(apply_query_modifiers=True, query_prefix="",
                 query_suffix="", query_negatives="be_brief")
    out = gen.apply_query_modifiers("q", cfg)
    check("negatives-only -> Avoid: be_brief",
          out == "q\n\nAvoid: be_brief", f"got {out!r}")


def test_apply_modifiers_whitespace_only_treated_as_blank():
    print("\n[4] whitespace-only modifier strings are skipped")
    cfg = Config(apply_query_modifiers=True,
                 query_prefix="   ", query_suffix="\n\t",
                 query_negatives="  ")
    out = gen.apply_query_modifiers("q", cfg)
    check("whitespace-only modifiers ignored",
          out == "q", f"got {out!r}")


# ---------------------------------------------------------------------------
# Agent provider dispatch
# ---------------------------------------------------------------------------

def test_agent_complete_falls_back_to_local():
    print("\n[5] agent_complete falls back to local backend when no API key")
    saved_detect = gen.detect_backend
    saved_chat = gen._ollama_chat
    gen.detect_backend = lambda cfg: "ollama"
    gen._ollama_chat = lambda cfg, msgs: "local-answer"
    try:
        cfg = Config(agent_provider="openai", agent_api_key="",  # blank key
                     agent_model="")
        out = gen.agent_complete(cfg, [{"role": "user", "content": "hi"}])
        check("falls back to local when key blank",
              out == "local-answer", f"got {out!r}")
    finally:
        gen.detect_backend = saved_detect
        gen._ollama_chat = saved_chat


def test_agent_complete_openai_path():
    print("\n[6] agent_complete dispatches to OpenAI-compatible endpoint")
    captured = {}

    class FakeResp:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200
        def raise_for_status(self): pass
        def json(self): return self.payload

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return FakeResp({"choices": [{"message": {"content": "openai-out"}}]})

    saved = gen.httpx.post
    gen.httpx.post = fake_post
    try:
        cfg = Config(agent_provider="openai", agent_api_key="sk-test",
                     agent_model="gpt-4o-mini",
                     agent_base_url="https://api.openai.com/v1")
        out = gen.agent_complete(cfg, [{"role": "user", "content": "hi"}])
        check("returned OpenAI text",
              out == "openai-out", f"got {out!r}")
        check("hit chat/completions URL",
              "/chat/completions" in captured["url"],
              f"url={captured.get('url')!r}")
        check("Bearer auth header set",
              captured["headers"].get("Authorization") == "Bearer sk-test",
              f"hdrs={captured.get('headers')}")
        check("model from cfg.agent_model",
              captured["body"]["model"] == "gpt-4o-mini",
              f"body model={captured['body'].get('model')!r}")
    finally:
        gen.httpx.post = saved


def test_agent_complete_anthropic_path():
    print("\n[7] agent_complete dispatches to Anthropic with correct shape")
    captured = {}

    class FakeResp:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200
        def raise_for_status(self): pass
        def json(self): return self.payload

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return FakeResp({"content": [{"type": "text", "text": "anthropic-out"}]})

    saved = gen.httpx.post
    gen.httpx.post = fake_post
    try:
        cfg = Config(agent_provider="anthropic", agent_api_key="sk-ant-test",
                     agent_model="claude-haiku-4-5-20251001")
        msgs = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        out = gen.agent_complete(cfg, msgs)
        check("returned anthropic text",
              out == "anthropic-out", f"got {out!r}")
        check("hit messages URL",
              captured["url"].endswith("/v1/messages"),
              f"url={captured['url']!r}")
        check("x-api-key header",
              captured["headers"].get("x-api-key") == "sk-ant-test",
              f"hdrs={captured['headers']}")
        check("system pulled out of messages",
              captured["body"].get("system") == "be brief",
              f"system={captured['body'].get('system')!r}")
        check("body messages don't contain system",
              all(m["role"] != "system" for m in captured["body"]["messages"]),
              "system leaked into messages array")
    finally:
        gen.httpx.post = saved


def test_agent_complete_swallows_provider_errors():
    print("\n[8] agent_complete returns '' on provider error and falls back")
    saved_post = gen.httpx.post

    def boom(*a, **k):
        raise RuntimeError("boom")
    gen.httpx.post = boom
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda cfg: "none"
    try:
        cfg = Config(agent_provider="openai", agent_api_key="sk-test")
        out = gen.agent_complete(cfg, [{"role": "user", "content": "hi"}])
        check("returns empty on failure (no crash)",
              out == "", f"got {out!r}")
    finally:
        gen.httpx.post = saved_post
        gen.detect_backend = saved_detect


# ---------------------------------------------------------------------------
# Agentic retrieval
# ---------------------------------------------------------------------------

def test_agentic_returns_initial_when_iterations_zero(ws, cfg):
    print("\n[9] agent_max_iterations=0 -> initial retrieval only")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    cfg2 = Config(**{**cfg.__dict__, "agent_max_iterations": 0})
    hits = agentic_retrieve(query="border collie", embedder=embedder,
                            index=idx, cfg=cfg2)
    check("agent with 0 iterations returns hits",
          len(hits) >= 1, f"got {len(hits)}")


def test_agentic_returns_initial_when_no_backend(ws, cfg):
    print("\n[10] no LLM backend -> agent degrades to plain retrieval")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    saved = gen.detect_backend
    gen.detect_backend = lambda c: "none"
    try:
        hits = agentic_retrieve(query="border collie", embedder=embedder,
                                index=idx, cfg=cfg)
        check("agent without backend returns hits anyway",
              len(hits) >= 1, f"got {len(hits)}")
    finally:
        gen.detect_backend = saved


def test_agentic_breaks_on_sufficient(ws, cfg):
    print("\n[11] agent stops early when LLM returns SUFFICIENT")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    saved_detect = gen.detect_backend
    saved_agent = gen.agent_complete
    calls = {"n": 0}
    gen.detect_backend = lambda c: "ollama"

    def fake_agent(cfg, messages, max_tokens=300):
        calls["n"] += 1
        return "SUFFICIENT"
    gen.agent_complete = fake_agent
    # patch the binding in retrieve module too (it imported the symbol)
    import ez_rag.retrieve as rt
    saved_rt_agent = rt.agent_complete
    rt.agent_complete = fake_agent
    try:
        cfg2 = Config(**{**cfg.__dict__, "agent_max_iterations": 3})
        hits = agentic_retrieve(query="dogs", embedder=embedder, index=idx,
                                cfg=cfg2)
        check("agent broke after 1 reflection",
              calls["n"] == 1, f"calls={calls['n']}")
        check("still returns hits",
              len(hits) >= 1, f"got {len(hits)}")
    finally:
        gen.detect_backend = saved_detect
        gen.agent_complete = saved_agent
        rt.agent_complete = saved_rt_agent


def test_agentic_status_callback_called(ws, cfg):
    print("\n[12] status_cb is invoked at each step")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "none"
    msgs = []
    try:
        agentic_retrieve(query="dogs", embedder=embedder, index=idx,
                        cfg=cfg,
                        status_cb=lambda m: msgs.append(m))
        # With backend=none we only get the "initial retrieval" message.
        check("status_cb called at least once",
              len(msgs) >= 1, f"msgs={msgs}")
        check("first status mentions initial",
              "initial" in (msgs[0] if msgs else ""),
              f"first={msgs[0] if msgs else ''!r}")
    finally:
        gen.detect_backend = saved_detect


def test_agentic_runs_followups(ws, cfg):
    print("\n[13] agent fans out to follow-up queries when not sufficient")
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    saved_detect = gen.detect_backend
    gen.detect_backend = lambda c: "ollama"
    import ez_rag.retrieve as rt
    saved_rt_agent = rt.agent_complete

    # First call: insufficient → return 2 follow-ups
    # Second call: SUFFICIENT
    responses = iter([
        "speed of light constant\nF=ma newton's law\n",
        "SUFFICIENT",
    ])

    def fake_agent(cfg, messages, max_tokens=300):
        return next(responses, "SUFFICIENT")
    rt.agent_complete = fake_agent
    msgs = []
    try:
        cfg2 = Config(**{**cfg.__dict__, "agent_max_iterations": 2,
                         "rerank": False})
        hits = agentic_retrieve(query="science",
                                embedder=embedder, index=idx, cfg=cfg2,
                                status_cb=lambda m: msgs.append(m))
        check("agent emitted refining status",
              any("refining" in m for m in msgs), f"msgs={msgs}")
        check("agent retrieved hits across queries",
              len(hits) >= 1, f"got {len(hits)}")
        # Should now include physics-related results, since we asked about
        # follow-up "speed of light" / "F=ma" topics.
        paths = {h.path for h in hits}
        check("hits include physics doc",
              any("physics" in p.lower() for p in paths),
              f"paths={paths}")
    finally:
        gen.detect_backend = saved_detect
        rt.agent_complete = saved_rt_agent


def main():
    ws = make_tmp_ws()
    cfg = Config(
        embedder_provider="fastembed",
        embedder_model="BAAI/bge-small-en-v1.5",
        rerank=False, hybrid=True, top_k=4,
        chunk_size=80, chunk_overlap=10,
    )
    print(f"[setup] tmp workspace: {ws.root}")
    try:
        stats = ingest(ws, cfg=cfg)
        print(f"[setup] ingested files={stats.files_seen} "
              f"chunks={stats.chunks_added}")
        # Modifier tests don't need the workspace
        test_apply_modifiers_off()
        test_apply_modifiers_full()
        test_apply_modifiers_partial()
        test_apply_modifiers_whitespace_only_treated_as_blank()
        # Provider dispatch
        test_agent_complete_falls_back_to_local()
        test_agent_complete_openai_path()
        test_agent_complete_anthropic_path()
        test_agent_complete_swallows_provider_errors()
        # Agentic retrieval
        test_agentic_returns_initial_when_iterations_zero(ws, cfg)
        test_agentic_returns_initial_when_no_backend(ws, cfg)
        test_agentic_breaks_on_sufficient(ws, cfg)
        test_agentic_status_callback_called(ws, cfg)
        test_agentic_runs_followups(ws, cfg)
    finally:
        try:
            shutil.rmtree(ws.root, ignore_errors=True)
        except Exception:
            pass

    print(f"\n=== Round 3 summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for name, det in FAIL:
            print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

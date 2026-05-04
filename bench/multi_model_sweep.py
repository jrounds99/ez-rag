"""Multi-model RAG quality sweep.

Goal: find the parameter-count knee where answer quality starts to
fall off, holding retrieval constant. The motivating question: at what
size does a small/cheap model stop being good enough? (Important for
VRAM-constrained deployments — the user framed this as a "worldwide
memory shortage" question.)

Methodology
-----------
1. **Hold retrieval constant.** Run smart_retrieve ONCE per question
   using the user's configured embedder + the configured HyDE model
   (qwen2.5:7b in this workspace). Save the hits.
2. **Sweep models.** For each model under test, swap the chat model
   into cfg, hand it the same question + same hits, and capture the
   answer.
3. **Same prompt + options for every model.** Auto-num_ctx kicks in
   per model based on the model's own native max context.
4. **Judge with one strong model.** Use qwen2.5:7b as the LLM judge
   with the same 0-12 rubric (addresses, specificity, grounded,
   on_topic) so scores are comparable across the sweep.

Output
------
- JSON: every (model, question, answer, sources, seconds) row
- Markdown report with: per-model averages, quality-vs-params plot
  (ASCII), VRAM-vs-quality plot (ASCII), the "knee" identified
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx

from ez_rag.config import Config
from ez_rag.embed import make_embedder
from ez_rag.generate import (
    SYSTEM_PROMPT_LIST_EXTRACTION, SYSTEM_PROMPT_RAG, _build_user_prompt,
    _is_list_query, _ollama_chat, detect_backend,
)
from ez_rag.index import Index
from ez_rag.retrieve import smart_retrieve
from ez_rag.workspace import Workspace


# ============================================================================
# Models under test
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    tag: str            # ollama tag
    params_b: float     # billion-parameter count for plotting
    family: str         # "qwen2.5" / "llama3" / "phi" / "mistral" / "deepseek-r1"
    notes: str = ""


MODELS: list[ModelSpec] = [
    ModelSpec("qwen2.5:0.5b",    0.50, "qwen2.5"),
    ModelSpec("llama3.2:1b",     1.20, "llama3"),
    ModelSpec("qwen2.5:1.5b",    1.50, "qwen2.5"),
    ModelSpec("deepseek-r1:1.5b", 1.80, "deepseek-r1",
               "reasoning model (uses <think> blocks)"),
    ModelSpec("llama3.2:3b",     3.00, "llama3"),
    ModelSpec("qwen2.5:3b",      3.10, "qwen2.5"),
    ModelSpec("phi4-mini",       3.80, "phi"),
    ModelSpec("mistral:7b",      7.00, "mistral"),
    ModelSpec("qwen2.5:7b",      7.60, "qwen2.5"),
    ModelSpec("llama3.1:8b",     8.00, "llama3"),
    ModelSpec("qwen2.5:14b",    14.00, "qwen2.5"),
    ModelSpec("deepseek-r1:32b", 32.80, "deepseek-r1",
               "reasoning model (uses <think> blocks)"),
]


# ============================================================================
# Question set — same 30 questions across all models for fair comparison
# ============================================================================

@dataclass
class Question:
    text: str
    category: str

QUESTIONS: list[Question] = [
    # Rule lookups (specific rule + page expected)
    Question("How does the grappling rule work in 5e?", "rule"),
    Question("What is the AC of plate armor?", "rule"),
    Question("How many spells can a level 3 wizard prepare?", "rule"),
    Question("What does the prone condition do?", "rule"),
    Question("How does concentration work for spells?", "rule"),
    Question("What are the rules for opportunity attacks?", "rule"),
    Question("What does cover do mechanically?", "rule"),
    Question("How do death saving throws work?", "rule"),
    Question("What is the difference between a ritual spell and a normal spell?",
              "rule"),
    Question("How does the dodge action work?", "rule"),

    # Comparisons (require synthesis)
    Question("Compare the Fighter and Barbarian at level 5 — strengths and weaknesses.",
              "comparison"),
    Question("How does the Wizard's spellcasting differ from the Sorcerer's?",
              "comparison"),
    Question("What's the difference between a Cleric and a Paladin?", "comparison"),
    Question("Compare the Rogue and the Monk for unarmed combat.", "comparison"),
    Question("Druid versus Ranger — when would you pick each?", "comparison"),
    Question("What's the difference between divine and arcane magic?", "comparison"),
    Question("Sneak Attack vs Smite — how do they compare in damage output?",
              "comparison"),

    # Exploratory / list (where the model has to extract named entities)
    Question("List some unique-sounding NPCs from the books with their names.",
              "exploratory"),
    Question("What are some interesting magical items a DM could give a level 5 party?",
              "exploratory"),
    Question("List notable villains and antagonists from the published adventures.",
              "exploratory"),
    Question("What are some classic D&D dungeons or locations a new DM could borrow?",
              "exploratory"),
    Question("Give examples of memorable taverns or inns mentioned in any of the books.",
              "exploratory"),
    Question("List interesting deities or gods that PCs could worship.",
              "exploratory"),
    Question("What are the most iconic dragons in D&D 5e lore?", "exploratory"),
    Question("Give examples of legendary artifacts in 5e.", "exploratory"),

    # Multi-step / generative
    Question("If I have a level 5 Battle Master Fighter, what feats should I consider next?",
              "multi-step"),
    Question("Walk me through resolving a contested grapple between a Fighter and a giant spider.",
              "multi-step"),
    Question("What are the steps to cast Fireball at 4th level?", "multi-step"),
    Question("Give me a checklist for prepping a 3-hour D&D session.",
              "multi-step"),
    Question("Plan a one-shot adventure set in a haunted manor.", "multi-step"),
]


# ============================================================================
# Helpers
# ============================================================================

@dataclass
class Result:
    model: str
    params_b: float
    family: str
    question: str
    category: str
    answer: str
    sources: list[str]
    seconds: float
    err: str = ""


def installed_models(url: str) -> set[str]:
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=5.0)
        if r.status_code != 200:
            return set()
        return {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        return set()


def normalize_tag(tag: str) -> str:
    return tag if ":" in tag else f"{tag}:latest"


def model_present(url: str, tag: str, installed: set[str]) -> bool:
    return tag in installed or normalize_tag(tag) in installed


def pull_model(url: str, tag: str) -> bool:
    """Stream pull a model. Returns True on success."""
    try:
        with httpx.stream(
            "POST",
            url.rstrip("/") + "/api/pull",
            json={"name": tag},
            timeout=None,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("status") == "success":
                    return True
                if obj.get("error"):
                    print(f"  pull error: {obj['error']}")
                    return False
        return True
    except Exception as ex:
        print(f"  pull failed: {ex}")
        return False


def unload_model(url: str, tag: str) -> None:
    """Tell Ollama to evict the model from VRAM (best-effort)."""
    try:
        httpx.post(
            url.rstrip("/") + "/api/generate",
            json={"model": tag, "prompt": "", "keep_alive": 0},
            timeout=10.0,
        )
    except Exception:
        pass


def precompute_retrieval(workspace: Workspace, cfg: Config,
                          questions: list[Question]) -> dict[str, list]:
    """Run retrieval ONCE per question. Returns {question_text: hits}."""
    embedder = make_embedder(cfg)
    index = Index(workspace.meta_db_path, embed_dim=embedder.dim)
    out: dict[str, list] = {}
    for i, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        hits = smart_retrieve(
            query=q.text, embedder=embedder, index=index, cfg=cfg,
        )
        dt = time.perf_counter() - t0
        out[q.text] = hits
        print(f"  [{i}/{len(questions)}] retrieved {len(hits)} hits in "
               f"{dt:.1f}s — {q.text[:55]}…")
    return out


def answer_with_model(*, question: str, hits, cfg: Config) -> str:
    """Generate an answer using the chat model in cfg with the supplied hits."""
    user_prompt = _build_user_prompt(question, hits)
    sys_prompt = SYSTEM_PROMPT_RAG
    if hits and getattr(cfg, "auto_list_mode", True) and _is_list_query(question):
        sys_prompt = SYSTEM_PROMPT_LIST_EXTRACTION
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _ollama_chat(cfg, messages)


def strip_thinking(text: str) -> str:
    """Reasoning models (deepseek-r1) wrap their internal monologue in
    <think>…</think> tags. Strip them so the judge sees only the actual
    answer."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=r"C:\Users\jroun\Desktop\dnd books\2014 (5e)")
    ap.add_argument("--out-dir", default=str(ROOT / "bench" / "reports"))
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--skip-pull", action="store_true",
                     help="Skip pulling missing models (test only locally available)")
    ap.add_argument("--unload-between", action="store_true",
                     help="Evict each model from VRAM after its run")
    args = ap.parse_args()

    ws = Workspace(Path(args.workspace))
    if not ws.is_initialized():
        print(f"workspace not initialized: {args.workspace}")
        return 1
    cfg = ws.load_config()
    if detect_backend(cfg) == "none":
        print("Ollama not reachable")
        return 1

    questions = QUESTIONS
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    models = MODELS
    if args.limit_models:
        models = models[: args.limit_models]

    print(f"Workspace: {args.workspace}")
    print(f"Embedder:  {cfg.ollama_embed_model or cfg.embedder_model}")
    print(f"Models under test: {len(models)}")
    print(f"Questions:         {len(questions)}")
    print()

    # ---- Pull missing models ----
    installed = installed_models(cfg.llm_url)
    missing = [m for m in models if not model_present(cfg.llm_url, m.tag, installed)]
    if missing and not args.skip_pull:
        print(f"Pulling {len(missing)} missing model(s)…")
        for m in missing:
            print(f"  pulling {m.tag} (~{m.params_b}B params)…")
            if not pull_model(cfg.llm_url, m.tag):
                print(f"  FAILED to pull {m.tag} — will skip")
        installed = installed_models(cfg.llm_url)

    runnable = [m for m in models
                 if model_present(cfg.llm_url, m.tag, installed)]
    skipped = [m for m in models if m not in runnable]
    if skipped:
        print(f"Skipping {len(skipped)} unavailable: "
               f"{[m.tag for m in skipped]}")

    if not runnable:
        print("No runnable models — exiting.")
        return 1

    # ---- Pre-compute retrieval (using cfg's chat model for HyDE) ----
    print("\nPre-computing retrieval (chat model for HyDE = "
           f"{cfg.llm_model})…")
    hits_by_q = precompute_retrieval(ws, cfg, questions)
    # Reduce hits to a serializable form for the JSON dump.
    hits_serialized = {
        q: [{"chunk_id": h.chunk_id, "path": h.path, "page": h.page,
             "section": h.section, "text": (h.text or "")[:300]}
            for h in v]
        for q, v in hits_by_q.items()
    }

    # ---- Sweep ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"multimodel-{stamp}.json"

    results: list[Result] = []

    for mi, m in enumerate(runnable, start=1):
        print(f"\n[{mi}/{len(runnable)}] === {m.tag} ({m.params_b}B "
               f"params, {m.family}) ===")
        # Override chat model only — embedder stays the same.
        cfg_m = Config(**{k: v for k, v in cfg.__dict__.items()
                          if k != "extra"})
        cfg_m.llm_model = m.tag

        m_t0 = time.perf_counter()
        for qi, q in enumerate(questions, start=1):
            hits = hits_by_q[q.text]
            t0 = time.perf_counter()
            err = ""
            ans = ""
            try:
                ans = answer_with_model(
                    question=q.text, hits=hits, cfg=cfg_m,
                )
                ans = strip_thinking(ans)
            except Exception as ex:
                err = f"{type(ex).__name__}: {ex}"
            dt = time.perf_counter() - t0
            r = Result(
                model=m.tag, params_b=m.params_b, family=m.family,
                question=q.text, category=q.category,
                answer=ans, sources=[
                    f"{h.path or h.section}:p{h.page or '?'}"
                    for h in hits[:5]
                ],
                seconds=dt, err=err,
            )
            results.append(r)
            flag = f"  ERR: {err}" if err else ""
            print(f"  [{qi:>2}/{len(questions)}] {dt:5.1f}s  "
                   f"{q.text[:60]}{flag}")

            # Periodic JSON dump so a crash doesn't lose data
            if len(results) % 10 == 0:
                json_path.write_text(json.dumps(
                    {"results": [asdict(x) for x in results],
                     "hits": hits_serialized,
                     "models_run": [asdict(m) for m in runnable]},
                    indent=2,
                ), encoding="utf-8")

        m_dt = time.perf_counter() - m_t0
        print(f"  -- {m.tag} done in {m_dt:.0f}s "
               f"({m_dt/len(questions):.1f}s/q)")

        if args.unload_between:
            unload_model(cfg.llm_url, m.tag)

    # Final dump
    json_path.write_text(json.dumps(
        {"results": [asdict(x) for x in results],
         "hits": hits_serialized,
         "models_run": [asdict(m) for m in runnable]},
        indent=2,
    ), encoding="utf-8")

    print(f"\n[OK] Wrote {len(results)} results to {json_path}")
    print(f"\nNext: judge with bench/judge_eval.py and analyze with "
           "bench/multi_model_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

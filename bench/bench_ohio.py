"""Multi-embedder × multi-model bench against the Ohio geology corpus.

For each embedder, ingests the Ohio sample corpus into its own
workspace, then sweeps every chat model with the same retrieval. All
generations get judged with the LLM-as-judge rubric. Result: a JSON
bundle that the heatmap/efficiency renderer turns into HTML.

Design:
  - Embedders are tested by re-ingesting the corpus per embedder
    (full-fidelity comparison). Cheap because the corpus is small.
  - Each (embedder × chat-model × question) cell gets one generation
    attempt; on failure we retry once with the same input. Two
    failures → that model is marked failed for the embedder and
    subsequent questions are skipped.
  - Power is sampled throughout via bench/power.PowerSampler so we
    can compute quality-per-watt-second alongside quality-per-second.

Usage:
    python -X utf8 bench/bench_ohio.py
    python -X utf8 bench/bench_ohio.py --limit-questions 8 --limit-models 6
    python -X utf8 bench/bench_ohio.py --skip-pull --no-power

Output: `bench/reports/ohio-<UTC-timestamp>/`
  - manifest.json
  - sweep.json (raw answers per cell)
  - judged.json (judge scores)
  - power_samples.csv (if --power)
  - report.html (heatmap + efficiency)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import httpx

from ez_rag.config import Config
from ez_rag.embed import make_embedder, clear_embedder_cache
from ez_rag.generate import (
    SYSTEM_PROMPT_LIST_EXTRACTION, SYSTEM_PROMPT_RAG, _build_user_prompt,
    _is_list_query, _ollama_chat, detect_backend,
)
from ez_rag.index import Index
from ez_rag.ingest import ingest
from ez_rag.retrieve import smart_retrieve
from ez_rag.workspace import Workspace

# Power sampling — reuses bench/power.py from the bench-suite work.
try:
    from bench.power import PowerSampler
    HAS_POWER = True
except ImportError:
    HAS_POWER = False


# ============================================================================
# Bench config
# ============================================================================

EMBEDDERS = [
    # (provider, model_or_tag, label)
    # Ollama-hosted embedders are what most users will run; fastembed
    # is a CPU fallback. The bench tests whatever's installed.
    ("ollama", "qwen3-embedding:8b", "qwen3-emb-8b"),
    ("ollama", "nomic-embed-text", "nomic-text"),
    ("ollama", "bge-m3:567m", "bge-m3"),
]

CHAT_MODELS = [
    # (tag, params_b, family) — 22 models spanning 6 families,
    # 0.5B → 32B, including new Qwen3 + Gemma3 + Granite3 + OLMo2.
    # Smallest first so a failure pattern surfaces fast.

    # Sub-1B
    ("qwen2.5:0.5b",      0.5, "qwen2.5"),
    ("qwen3:0.6b",        0.6, "qwen3"),

    # 1B class
    ("llama3.2:1b",       1.2, "llama3"),
    ("qwen2.5:1.5b",      1.5, "qwen2.5"),
    ("qwen3:1.7b",        1.7, "qwen3"),
    ("deepseek-r1:1.5b",  1.8, "deepseek-r1"),

    # 2B class
    ("gemma2:2b",         2.6, "gemma"),
    ("granite3.3:2b",     2.5, "granite"),

    # 3B class
    ("llama3.2:3b",       3.0, "llama3"),
    ("qwen2.5:3b",        3.1, "qwen2.5"),
    ("phi4-mini",         3.8, "phi"),
    ("qwen3:4b",          4.0, "qwen3"),
    ("gemma3:4b",         4.3, "gemma"),

    # 7-9B class
    ("mistral:7b",        7.0, "mistral"),
    ("qwen2.5:7b",        7.6, "qwen2.5"),
    ("qwen3:8b",          8.2, "qwen3"),
    ("granite3.3:8b",     8.2, "granite"),
    ("llama3.1:8b",       8.0, "llama3"),
    ("gemma2:9b",         9.0, "gemma"),

    # 12-14B class
    ("mistral-nemo:12b", 12.0, "mistral"),
    ("qwen2.5:14b",      14.0, "qwen2.5"),
    ("qwen3:14b",        14.8, "qwen3"),

    # Reasoning / 32B
    ("deepseek-r1:32b",  32.8, "deepseek-r1"),
]

# Two retrieval strategies — the bench reports both so the user can
# see whether the bumped settings actually buy them anything.
STRATEGIES = [
    {
        "name": "default",
        "overrides": {},
    },
    {
        "name": "list-mode",
        "overrides": {"auto_list_mode": True, "top_k": 12},
    },
]


# ============================================================================
# Result types
# ============================================================================

@dataclass
class CellResult:
    """One (embedder × chat-model × strategy × question) cell."""
    embedder: str
    chat_model: str
    params_b: float
    family: str
    strategy: str
    question: str
    category: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    attempts: int = 0
    # v2: token economics
    prompt_tokens: int = 0
    eval_tokens: int = 0
    tokens_per_sec: float = 0.0
    # v2: gold-snippet match (rule-based correctness check)
    gold_score: Optional[int] = None  # 0..max_required, None = no gold for Q
    gold_max: Optional[int] = None
    gold_missing: list[str] = field(default_factory=list)


# ============================================================================
# Helpers
# ============================================================================

def _say(msg: str) -> None:
    print(msg, flush=True)


def installed_models(url: str) -> set[str]:
    try:
        r = httpx.get(url.rstrip("/") + "/api/tags", timeout=5.0)
        if r.status_code != 200:
            return set()
        return {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        return set()


def model_present(installed: set[str], tag: str) -> bool:
    if tag in installed:
        return True
    if (tag + ":latest") in installed:
        return True
    if tag.endswith(":latest") and tag[:-7] in installed:
        return True
    return False


def pull_model(url: str, tag: str) -> bool:
    """Stream-pull. Returns True on success."""
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
                    _say(f"  pull error: {obj['error']}")
                    return False
        return True
    except Exception as ex:
        _say(f"  pull failed: {ex}")
        return False


def unload_model(url: str, tag: str) -> None:
    """Tell Ollama to evict a model from VRAM (best-effort)."""
    try:
        httpx.post(
            url.rstrip("/") + "/api/generate",
            json={"model": tag, "prompt": "", "keep_alive": 0},
            timeout=10.0,
        )
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Gold-snippet must-contain checks (rule-based correctness alongside the
# rubric judge). Each entry is a list of substring groups; an answer
# scores 1 point per group where ANY synonym appears (case-insensitive).
# Only the most factual/objective questions get gold checks — open-ended
# ones rely on the rubric judge.
# ----------------------------------------------------------------------------
GOLD_SNIPPETS = {
    "Which mineral commodity is Ohio's largest by production value?": [
        # The 2023 Ohio Mineral Industries report names construction
        # aggregates / limestone & dolomite as #1 by value.
        ["limestone", "dolomite", "aggregate", "construction aggregate",
         "stone"],
    ],
    "When was the Ohio Geological Survey founded and who led it?": [
        ["1837", "1838"],
        ["mather", "william mather", "w. w. mather", "w.w. mather"],
    ],
    "What are the main coal-producing regions of Ohio?": [
        ["eastern", "southeastern", "appalachian"],
        ["pennsylvanian", "permian"],
    ],
    "Which industrial minerals are produced in Ohio today?": [
        ["limestone", "dolomite", "salt", "sand", "gravel", "clay",
         "gypsum", "sandstone"],
    ],
    "Tell me about the paving-brick industry in Ohio.": [
        ["clay", "shale"],
        ["nelsonville", "athens", "perry", "tuscarawas"],
    ],
    "How does Ohio's geology relate to the broader Appalachian Basin?": [
        ["appalachian basin", "basin"],
        ["sedimentary", "paleozoic", "devonian", "pennsylvanian",
         "ordovician", "silurian"],
    ],
    "Compare coal resources between Ohio and West Virginia.": [
        ["west virginia", "wv"],
        ["bituminous", "pennsylvanian"],
    ],
    "What's the difference between bedrock geology in eastern vs western Ohio?": [
        ["eastern"],
        ["western"],
        ["pennsylvanian", "permian", "ordovician", "silurian", "devonian",
         "younger", "older"],
    ],
}


def score_gold(question: str, answer: str) -> Optional[tuple[int, int, list[str]]]:
    """Return (got, max, missing_groups) or None if no gold for this Q."""
    groups = GOLD_SNIPPETS.get(question)
    if not groups:
        return None
    a = (answer or "").lower()
    got = 0
    missing: list[str] = []
    for g in groups:
        if any(syn.lower() in a for syn in g):
            got += 1
        else:
            missing.append(g[0])
    return got, len(groups), missing


def strip_thinking(text: str) -> str:
    """Reasoning models wrap CoT in <think>…</think>. Strip so the
    judge scores the actual answer.

    v2 fallback: if the cleaned result is empty BUT the raw text had
    <think> tags, return the text after the LAST </think> if any, else
    the unstripped raw (better than empty — qwen3 sometimes emits the
    final answer inside the thinking block when num_predict runs out).
    """
    import re
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned:
        # Also strip lone tags that survived the paired-strip pass
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.DOTALL).strip()
        return cleaned
    # All content was inside (possibly unclosed) <think>… — try recovery
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail
    # Last resort: strip just the opening tag and return whatever's left
    return re.sub(r"</?think>", "", text, flags=re.DOTALL).strip()


def _ollama_chat_with_meta(*, url: str, model: str, messages: list[dict],
                            options: dict, think: bool = False,
                            timeout_s: float = 300.0) -> dict:
    """Bench-local /api/chat that captures token counts + thinking flag.

    Returns: {"content": str, "thinking": str, "prompt_tokens": int,
              "eval_tokens": int, "total_duration_ns": int,
              "eval_duration_ns": int}
    `think=False` disables CoT for fair apples-to-apples comparison
    on reasoning models (qwen3, deepseek-r1).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": bool(think),
    }
    r = httpx.post(url.rstrip("/") + "/api/chat", json=payload,
                   timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    msg = body.get("message", {}) or {}
    return {
        "content": msg.get("content", "") or "",
        "thinking": msg.get("thinking", "") or "",
        "prompt_tokens": int(body.get("prompt_eval_count", 0) or 0),
        "eval_tokens": int(body.get("eval_count", 0) or 0),
        "total_duration_ns": int(body.get("total_duration", 0) or 0),
        "eval_duration_ns": int(body.get("eval_duration", 0) or 0),
    }


# ============================================================================
# Workspace setup per embedder
# ============================================================================

def build_workspace_for_embedder(
    *, embedder_provider: str, embedder_tag: str, label: str,
    base_dir: Path, corpus_dir: Path, ollama_url: str,
    skip_pull: bool = False,
) -> Workspace:
    """Create / refresh a workspace under base_dir/<label>/ ingested
    with the given embedder. Returns the ready-to-query workspace."""
    ws_root = base_dir / f"ws-{label}"
    ws_root.mkdir(parents=True, exist_ok=True)
    docs_dir = ws_root / "docs"
    if not docs_dir.exists() or not any(docs_dir.iterdir()):
        # Symlink would be cleaner; on Windows we copy to be safe
        docs_dir.mkdir(parents=True, exist_ok=True)
        for src in corpus_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() not in (".pdf", ".docx", ".xlsx",
                                            ".csv", ".txt"):
                continue
            rel = src.relative_to(corpus_dir)
            dst = docs_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

    ws = Workspace(ws_root)
    if not ws.is_initialized():
        ws.initialize()

    # Configure for this embedder
    cfg = ws.load_config()
    cfg.embedder_provider = embedder_provider
    if embedder_provider == "ollama":
        cfg.ollama_embed_model = embedder_tag
    else:
        cfg.embedder_model = embedder_tag
    cfg.llm_url = ollama_url
    # Use a fast small model for HyDE during retrieval pre-compute
    cfg.llm_model = "qwen2.5:1.5b"
    # Conservative ingest settings for fast bench iteration
    cfg.chunk_size = 512
    cfg.chunk_overlap = 64
    cfg.embed_batch_size = 32
    cfg.parallel_workers = 2
    cfg.enable_ocr = True
    cfg.enable_contextual = False
    cfg.llm_inspect_pages = False
    cfg.llm_correct_garbled = False
    cfg.save(ws.config_path)

    return ws


def ingest_workspace(ws: Workspace, *, force: bool = False,
                      skip_pull: bool = False) -> dict:
    """Run ingest. Returns stats dict."""
    cfg = ws.load_config()

    # Make sure the embedder model is present
    if cfg.embedder_provider == "ollama":
        installed = installed_models(cfg.llm_url)
        if not model_present(installed, cfg.ollama_embed_model):
            if skip_pull:
                raise RuntimeError(
                    f"Embedder {cfg.ollama_embed_model} not installed and "
                    "--skip-pull was set. Pull manually or remove the flag."
                )
            _say(f"  pulling embedder {cfg.ollama_embed_model}…")
            if not pull_model(cfg.llm_url, cfg.ollama_embed_model):
                raise RuntimeError(
                    f"Failed to pull embedder {cfg.ollama_embed_model}"
                )

    clear_embedder_cache()
    t0 = time.perf_counter()
    last_status = {"status": "starting"}

    def _on_progress(prog):
        try:
            last_status["status"] = prog.status
            last_status["files_done"] = prog.files_done
            last_status["chunks_done"] = prog.chunks_done
        except Exception:
            pass

    stats = ingest(ws, cfg=cfg, force=force, progress=_on_progress)
    duration = time.perf_counter() - t0
    return {
        "files_seen": stats.files_seen,
        "files_new": stats.files_new,
        "files_changed": stats.files_changed,
        "files_skipped_unchanged": stats.files_skipped_unchanged,
        "files_unsupported": stats.files_unsupported,
        "files_errored": stats.files_errored,
        "chunks_added": stats.chunks_added,
        "duration_s": round(duration, 1),
    }


# ============================================================================
# Generation with retry-once-then-skip
# ============================================================================

def answer_with_retry(*, question: str, hits, cfg: Config,
                      timeout_s: float = 180.0) -> tuple[str, str, int, dict]:
    """Try the answer call up to twice. Returns (answer, error, attempts, meta).

    meta contains token counts + tokens/sec for the successful attempt.
    v2: passes think=False to Ollama (disables reasoning CoT for a
    fair, content-only comparison). Empty content also counts as a
    soft failure that triggers retry.
    """
    last_err = ""
    user_prompt = _build_user_prompt(question, hits)
    sys_prompt = SYSTEM_PROMPT_RAG if hits else SYSTEM_PROMPT_RAG
    if (hits and getattr(cfg, "auto_list_mode", True)
            and _is_list_query(question)):
        sys_prompt = SYSTEM_PROMPT_LIST_EXTRACTION
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Build options the same way generate.py does (autosize num_ctx).
    from ez_rag.generate import _ollama_options as _opts
    options = _opts(cfg, messages)
    # Ensure num_predict is generous so reasoning models that ignore
    # think=False still finish. 4096 is the cfg default; bump to 2048
    # min to cover smaller user-set caps.
    options["num_predict"] = max(int(options.get("num_predict", 0) or 0),
                                  2048)

    for attempt in (1, 2):
        try:
            t0 = time.perf_counter()
            res = _ollama_chat_with_meta(
                url=cfg.llm_url, model=cfg.llm_model,
                messages=messages, options=options,
                think=False, timeout_s=timeout_s,
            )
            dt = time.perf_counter() - t0
            text = strip_thinking(res["content"])
            if not text:
                # Empty answer — try the thinking field as fallback
                text = strip_thinking(res.get("thinking", ""))
            if not text:
                # Still empty — retry once if we have an attempt left
                last_err = "empty answer"
                if attempt == 1:
                    _say(f"      attempt 1: empty answer — retrying once…")
                    time.sleep(2.0)
                    continue
                return "", last_err, attempt, {}
            tps = (res["eval_tokens"] / (res["eval_duration_ns"] / 1e9)
                   if res["eval_duration_ns"] > 0 else 0.0)
            meta = {
                "prompt_tokens": res["prompt_tokens"],
                "eval_tokens": res["eval_tokens"],
                "tokens_per_sec": round(tps, 2),
                "wall_seconds": round(dt, 2),
            }
            return text, "", attempt, meta
        except Exception as ex:
            last_err = f"{type(ex).__name__}: {ex}"
            if attempt == 1:
                _say(f"      attempt 1 failed ({last_err}) — retrying once…")
                time.sleep(2.0)
                continue
    return "", last_err, 2, {}


# ============================================================================
# Main run loop
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",
                     default=str(REPO_ROOT / "sample_data" / "fetched"
                                  / "geology"),
                     help="Path to the corpus dir (geology subset by default).")
    ap.add_argument("--out-dir",
                     default=str(REPO_ROOT / "bench" / "reports"),
                     help="Where to write the result bundle.")
    ap.add_argument("--workspaces-dir",
                     default=str(REPO_ROOT / "bench" / "_ohio-workspaces"),
                     help="Where to put one ws per embedder.")
    ap.add_argument("--ollama-url",
                     default="http://127.0.0.1:11434")
    ap.add_argument("--judge", default="qwen2.5:7b")
    ap.add_argument("--limit-models", type=int, default=0,
                     help="Cap chat-model count (0 = all).")
    ap.add_argument("--limit-questions", type=int, default=0,
                     help="Cap question count (0 = all).")
    ap.add_argument("--limit-embedders", type=int, default=0,
                     help="Cap embedder count (0 = all).")
    ap.add_argument("--strategies", default="default",
                     help="Comma-separated strategy names. Default: 'default'. "
                          "Available: " + ", ".join(s["name"] for s in STRATEGIES))
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--reuse-workspaces", action="store_true",
                     help="Skip re-ingest if workspace exists.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus).resolve()
    if not corpus_dir.is_dir():
        _say(f"[!] Corpus not found: {corpus_dir}")
        _say(f"    Run: python sample_data/fetch.py")
        return 1

    # ----- Output dir -----
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"ohio-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir = Path(args.workspaces_dir)
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    # ----- Resolve scope -----
    embedders = EMBEDDERS[:]
    if args.limit_embedders:
        embedders = embedders[: args.limit_embedders]
    chat_models = CHAT_MODELS[:]
    if args.limit_models:
        chat_models = chat_models[: args.limit_models]

    questions_path = Path(__file__).resolve().parent / "ohio_questions.json"
    qdata = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = qdata["questions"]
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    # Filter strategies
    enabled_strat_names = {s.strip()
                            for s in args.strategies.split(",") if s.strip()}
    strategies = [s for s in STRATEGIES
                   if s["name"] in enabled_strat_names]
    if not strategies:
        strategies = [STRATEGIES[0]]

    _say(f"\n>>> Ohio bench: writing to {out_dir}")
    _say(f"    corpus      : {corpus_dir}")
    _say(f"    embedders   : {len(embedders)}  "
         f"({', '.join(e[2] for e in embedders)})")
    _say(f"    chat models : {len(chat_models)}")
    _say(f"    strategies  : {[s['name'] for s in strategies]}")
    _say(f"    questions   : {len(questions)}")

    # ----- Power sampler -----
    sampler = None
    if HAS_POWER and not args.no_power:
        sampler = PowerSampler(interval_s=0.5)
        sampler.start()

    # ----- Pull every chat model up-front -----
    if not args.skip_pull:
        installed = installed_models(args.ollama_url)
        missing = [m for m in chat_models
                    if not model_present(installed, m[0])]
        if missing:
            _say(f"\n>>> Pulling {len(missing)} missing chat model(s)…")
            for tag, _, _ in missing:
                _say(f"    pulling {tag}")
                if not pull_model(args.ollama_url, tag):
                    _say(f"    [!] failed to pull {tag} — will skip later")

    # ----- Per-embedder loop -----
    all_cells: list[CellResult] = []
    embedder_stats: dict[str, dict] = {}

    for ei, (provider, embed_tag, embed_label) in enumerate(embedders, start=1):
        _say(f"\n{'='*70}")
        _say(f"[Embedder {ei}/{len(embedders)}] {embed_label} "
             f"({provider}: {embed_tag})")
        _say('='*70)

        seg_label = f"embedder.{embed_label}"
        if sampler:
            sampler.set_segment(seg_label)

        # Build / refresh workspace
        try:
            ws = build_workspace_for_embedder(
                embedder_provider=provider, embedder_tag=embed_tag,
                label=embed_label, base_dir=workspaces_dir,
                corpus_dir=corpus_dir, ollama_url=args.ollama_url,
                skip_pull=args.skip_pull,
            )
        except Exception as ex:
            _say(f"  [!] workspace setup failed: {ex}")
            embedder_stats[embed_label] = {"error": str(ex)}
            continue

        # Ingest if needed
        ingest_seg = f"ingest.{embed_label}"
        if sampler:
            sampler.set_segment(ingest_seg)
        try:
            if args.reuse_workspaces and (ws.meta_db_path.is_file()
                                            and ws.meta_db_path.stat().st_size > 0):
                _say(f"  [reuse-workspaces] skipping ingest")
                ingest_stats = {"skipped": True}
            else:
                _say(f"  ingesting…")
                ingest_stats = ingest_workspace(
                    ws, force=False, skip_pull=args.skip_pull,
                )
                _say(f"  ingest: {ingest_stats}")
        except Exception as ex:
            _say(f"  [!] ingest failed: {ex}")
            embedder_stats[embed_label] = {
                "error": f"ingest: {ex}",
                "trace": traceback.format_exc(),
            }
            continue

        embedder_stats[embed_label] = {
            "provider": provider,
            "tag": embed_tag,
            "ingest": ingest_stats,
        }

        # Pre-compute retrieval per (strategy, question) so we don't
        # waste an embed call per chat model.
        cfg_base = ws.load_config()
        embedder_obj = make_embedder(cfg_base)
        index = Index(ws.meta_db_path, embed_dim=embedder_obj.dim)

        retrieval_seg = f"retrieve.{embed_label}"
        if sampler:
            sampler.set_segment(retrieval_seg)
        retrieval_cache: dict[tuple[str, str], list] = {}
        for strat in strategies:
            cfg = Config(**{k: v for k, v in cfg_base.__dict__.items()
                            if k != "extra"})
            for k, v in strat["overrides"].items():
                setattr(cfg, k, v)
            for q_obj in questions:
                q = q_obj["q"]
                t0 = time.perf_counter()
                try:
                    hits = smart_retrieve(
                        query=q, embedder=embedder_obj,
                        index=index, cfg=cfg,
                    )
                except Exception as ex:
                    _say(f"  retrieval err for {strat['name']}/{q[:40]}…: {ex}")
                    hits = []
                dt = time.perf_counter() - t0
                retrieval_cache[(strat["name"], q)] = hits
                _say(f"  retrieval [{strat['name']:8s}] {dt:5.1f}s "
                     f"({len(hits)} hits) — {q[:55]}")

        # Per chat model — generate + retry-once-skip
        for mi, (chat_tag, params_b, family) in enumerate(chat_models, start=1):
            installed = installed_models(args.ollama_url)
            if not model_present(installed, chat_tag):
                _say(f"\n  [{mi}/{len(chat_models)}] SKIP {chat_tag} (not installed)")
                continue

            _say(f"\n  [{mi}/{len(chat_models)}] {chat_tag} "
                 f"({params_b}B, {family})")
            failed_count = 0
            skipped_remaining = False

            for strat in strategies:
                cfg_m = Config(**{k: v for k, v in cfg_base.__dict__.items()
                                   if k != "extra"})
                for k, v in strat["overrides"].items():
                    setattr(cfg_m, k, v)
                cfg_m.llm_model = chat_tag
                cfg_m.llm_url = args.ollama_url

                for qi, q_obj in enumerate(questions, start=1):
                    if skipped_remaining:
                        all_cells.append(CellResult(
                            embedder=embed_label,
                            chat_model=chat_tag,
                            params_b=params_b, family=family,
                            strategy=strat["name"],
                            question=q_obj["q"],
                            category=q_obj.get("category", "?"),
                            error="skipped (model failed twice)",
                            attempts=0,
                        ))
                        continue

                    if sampler:
                        sampler.set_segment(
                            f"answer.{embed_label}.{chat_tag}"
                        )
                    hits = retrieval_cache.get(
                        (strat["name"], q_obj["q"]), []
                    )
                    t0 = time.perf_counter()
                    answer, err, attempts, meta = answer_with_retry(
                        question=q_obj["q"], hits=hits, cfg=cfg_m,
                    )
                    dt = time.perf_counter() - t0
                    gold = score_gold(q_obj["q"], answer)
                    cell = CellResult(
                        embedder=embed_label,
                        chat_model=chat_tag,
                        params_b=params_b, family=family,
                        strategy=strat["name"],
                        question=q_obj["q"],
                        category=q_obj.get("category", "?"),
                        answer=answer,
                        sources=[
                            f"{getattr(h, 'path', '?')}:p"
                            f"{getattr(h, 'page', '?')}"
                            for h in hits[:5]
                        ],
                        seconds=round(dt, 2),
                        error=err,
                        attempts=attempts,
                        prompt_tokens=meta.get("prompt_tokens", 0),
                        eval_tokens=meta.get("eval_tokens", 0),
                        tokens_per_sec=meta.get("tokens_per_sec", 0.0),
                        gold_score=gold[0] if gold else None,
                        gold_max=gold[1] if gold else None,
                        gold_missing=gold[2] if gold else [],
                    )
                    all_cells.append(cell)
                    flag = ""
                    if err:
                        failed_count += 1
                        flag = f"  ERR: {err[:60]}"
                    _say(f"    [{strat['name']:8s} q{qi:>2}] {dt:>5.1f}s  "
                         f"{q_obj['q'][:55]}{flag}")

                    # Periodic checkpoint dump
                    if len(all_cells) % 25 == 0:
                        _checkpoint(out_dir, all_cells, embedder_stats,
                                     ollama_url=args.ollama_url,
                                     bench_args=vars(args))

                    # Two-failure cutoff for this model at this embedder
                    if failed_count >= 2:
                        _say(f"    [!] {chat_tag} failed twice — "
                             "skipping remaining questions for this model")
                        skipped_remaining = True

            # Free this model's VRAM before moving to the next
            unload_model(args.ollama_url, chat_tag)

    # ----- Final dumps -----
    if sampler:
        sampler.stop()

    _checkpoint(out_dir, all_cells, embedder_stats,
                 ollama_url=args.ollama_url, bench_args=vars(args))
    if sampler:
        sampler.write_csv(out_dir / "power_samples.csv")
        (out_dir / "power_summary.json").write_text(
            json.dumps(sampler.summary(), indent=2),
            encoding="utf-8",
        )

    _say(f"\n>>> Sweep complete. {len(all_cells)} cells. Judging…")

    # ----- Judge -----
    judged_path = _run_judge(
        sweep_path=out_dir / "sweep.json",
        out_path=out_dir / "judged.json",
        ollama_url=args.ollama_url,
        judge_model=args.judge,
    )

    # ----- Render report -----
    if judged_path is not None:
        try:
            # Make the project root importable so `bench.*` resolves whether
            # the script was launched as `python bench/bench_ohio.py` or
            # `python -m bench.bench_ohio`.
            import sys as _sys
            _root = str(Path(__file__).resolve().parent.parent)
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from bench.ohio_html import render_report
            html_path = render_report(
                judged_path=judged_path,
                manifest_path=out_dir / "manifest.json",
                out_html=out_dir / "report.html",
                power_summary=(out_dir / "power_summary.json"
                               if (out_dir / "power_summary.json").is_file()
                               else None),
            )
            _say(f"\n[OK] Report: {html_path}")
        except Exception as ex:
            _say(f"[!] HTML render failed: {ex}")

    _say(f"\n[OK] Bundle: {out_dir}")
    return 0


_SYSINFO_CACHE: dict = {}


def _gather_sysinfo_once(ollama_url: str) -> dict:
    """Cache sysinfo for the run — probing GPUs every 25 cells is wasteful."""
    if _SYSINFO_CACHE:
        return _SYSINFO_CACHE
    # Make repo root importable so `bench.sysinfo` resolves whether the
    # script is launched as `python bench/bench_ohio.py` or via `-m`.
    _root = str(REPO_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        from bench.sysinfo import gather_system_info, to_dict
        info = gather_system_info(ollama_url=ollama_url)
        _SYSINFO_CACHE.update(to_dict(info))
    except Exception as ex:
        _SYSINFO_CACHE["error"] = f"sysinfo failed: {ex}"
    return _SYSINFO_CACHE


def _checkpoint(out_dir: Path, cells: list[CellResult],
                embedder_stats: dict, *, ollama_url: str = "",
                bench_args: Optional[dict] = None) -> None:
    """Persist the in-progress results so a crash mid-run doesn't lose data."""
    sweep_path = out_dir / "sweep.json"
    sweep_path.write_text(
        json.dumps([asdict(c) for c in cells], indent=2),
        encoding="utf-8",
    )
    manifest = {
        "version": 2,
        "started_at": time.time(),
        "embedders": embedder_stats,
        "cells_total": len(cells),
        "cells_ok": sum(1 for c in cells if not c.error),
        "cells_error": sum(1 for c in cells if c.error),
        "system": _gather_sysinfo_once(ollama_url) if ollama_url else {},
        "bench_args": bench_args or {},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def _run_judge(*, sweep_path: Path, out_path: Path,
                ollama_url: str, judge_model: str) -> Optional[Path]:
    """Shell out to bench/judge_eval.py, then move the produced
    -judged.json to the desired filename."""
    cmd = [
        sys.executable, "-X", "utf8", "-u",
        str(REPO_ROOT / "bench" / "judge_eval.py"),
        str(sweep_path),
        "--judge-model", judge_model,
        "--llm-url", ollama_url,
    ]
    _say(f"  >>> {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        _say(f"  [!] judge_eval exited {proc.returncode}")
        return None
    produced = sweep_path.with_name(sweep_path.stem + "-judged.json")
    if not produced.is_file():
        _say(f"  [!] judge_eval produced no file at {produced}")
        return None
    if produced != out_path:
        shutil.move(produced, out_path)
    return out_path


if __name__ == "__main__":
    sys.exit(main())

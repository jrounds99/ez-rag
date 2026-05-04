"""ez-rag-bench CLI entry point.

Subcommands:
  probe           Print system probe + the gating decision
  quick           Fast sanity sweep (1 model, 5 questions, no power)
  full            Run everything (ingest + retrieval + answer + judge)
  search          Just retrieval+answer (assumes ingest is done)
  ingest          Just the ingest matrix
  consolidate     Merge multiple-system bundles
  report          Render HTML / Markdown from a saved bundle

Designed to be runnable directly from a fresh git clone via:
    python -m bench.cli <subcommand> [args...]

The first thing any subcommand does is verify Ollama is reachable
and emit a one-line guidance message if not. No silent retries.
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
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================================
# Helpers
# ============================================================================

def _say(msg: str) -> None:
    print(msg, flush=True)


def _err(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr, flush=True)


def _check_ollama(url: str = "http://127.0.0.1:11434") -> bool:
    """One-shot Ollama reachability check. Prints guidance on failure."""
    try:
        import httpx
        r = httpx.get(url.rstrip("/") + "/api/version", timeout=3.0)
        if r.status_code == 200:
            ver = r.json().get("version", "?")
            _say(f"  Ollama reachable at {url} (version {ver})")
            return True
    except Exception as ex:
        _err(f"Couldn't reach Ollama at {url}: "
             f"{type(ex).__name__}: {ex}")
    _err("Install Ollama: https://ollama.com/download")
    _err("Then start it: `ollama serve` (or run the desktop app)")
    return False


def _summarize_corpus(corpus_dir: Path) -> dict:
    """Cheap walk of a corpus directory. No content read — only metadata."""
    if not corpus_dir.is_dir():
        return {"file_count": 0, "total_bytes": 0, "by_ext": {}}
    file_count = 0
    total_bytes = 0
    by_ext: dict[str, int] = {}
    for f in corpus_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            sz = f.stat().st_size
        except OSError:
            continue
        file_count += 1
        total_bytes += sz
        by_ext[f.suffix.lower()] = by_ext.get(f.suffix.lower(), 0) + 1
    return {"file_count": file_count, "total_bytes": total_bytes,
            "by_ext": by_ext}


def _models_to_test(skip_pull: bool = False,
                     user_specified: Optional[list[str]] = None,
                     ) -> list[dict]:
    """Decide which chat models the bench will run.

    Defaults to the bench/multi_model_sweep.py MODELS list, filtered
    by what's installed (skip_pull) or expanded with auto-pulls.
    """
    from bench.multi_model_sweep import MODELS
    out: list[dict] = []
    for m in MODELS:
        if user_specified and m.tag not in user_specified:
            continue
        out.append({
            "tag": m.tag,
            "params_b": m.params_b,
            "family": m.family,
            "notes": m.notes,
        })
    return out


# ============================================================================
# Subcommand: probe
# ============================================================================

def cmd_probe(args) -> int:
    from bench.sysinfo import gather_system_info, to_dict
    info = gather_system_info(ollama_url=args.ollama_url)
    print(json.dumps(to_dict(info), indent=2))

    print("\n=== Capability gating ===")
    if not info.gpus:
        print("No compatible GPU detected.")
        print("Bench will run, but everything will be CPU-only and slow.")
    for g in info.gpus:
        compat = "compatible" if (g.vram_total_mb or 0) >= 4096 \
                  else "too small"
        print(f"  GPU {g.index} · {g.name} · "
              f"{(g.vram_total_mb or 0)/1024:.0f} GB · "
              f"runtime {g.runtime} · {compat}")

    print("\nModels that the bench will test by default:")
    for m in _models_to_test():
        print(f"  {m['tag']:<22s} {m['params_b']:>5.1f}B params · "
               f"{m['family']}")

    return 0


# ============================================================================
# Subcommand: quick
# ============================================================================

def cmd_quick(args) -> int:
    """5-min sanity bench. Single model, 5 questions, no power."""
    if not _check_ollama(args.ollama_url):
        return 2

    from bench.run import make_run, stage, finalize

    cfg = {
        "mode": "quick",
        "models": ["qwen2.5:0.5b"],   # smallest, always present-ish
        "questions": 5,
        "power": False,
        "corpus": str(args.corpus) if args.corpus else None,
    }
    invocation = " ".join(sys.argv)
    run = make_run(out_dir=Path(args.out) if args.out else None,
                    corpus_dir=Path(args.corpus) if args.corpus else None,
                    cli_invocation=invocation,
                    config=cfg, ollama_url=args.ollama_url)
    _say(f"\n>>> Quick bench: writing to {run.out_dir}")
    _say(f">>> System: {run.sysinfo.cpu_model} · "
         f"{run.sysinfo.ram_total_gb:.0f} GB RAM · "
         f"{len(run.sysinfo.gpus)} GPU(s)")

    # Quick mode: just confirm we can do retrieval + one answer
    try:
        with stage(run, "smoke.search"):
            out = _run_simple_search(args.ollama_url, "qwen2.5:0.5b",
                                      questions_n=5,
                                      corpus_dir=args.corpus)
            run.stages["smoke.search"]["result"] = out
        _say("\n[OK] Quick bench complete.")
        _say(f"     Bundle: {run.out_dir}")
        return 0
    except Exception as exc:
        run.record_error("smoke.search", exc)
        bundle = run.write_diagnostic_bundle()
        _err(f"Quick bench failed: {exc}")
        _err(f"Diagnostic bundle: {bundle}")
        return 1
    finally:
        finalize(run)


def _run_simple_search(ollama_url: str, model: str, *, questions_n: int,
                        corpus_dir: Optional[Path]) -> dict:
    """Send N tiny prompts to verify Ollama answers. Doesn't touch the
    full RAG pipeline — used only by `quick` to verify plumbing."""
    import httpx
    started = time.time()
    answers: list[dict] = []
    sample_prompts = [
        "Reply with a single word: hello",
        "What is 2+2?",
        "Name a color.",
        "Reply with 'ok' if you can read this.",
        "What's the capital of France?",
    ][:questions_n]
    for p in sample_prompts:
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                ollama_url.rstrip("/") + "/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": p}],
                    "stream": False,
                    "options": {"num_predict": 50, "temperature": 0.2},
                },
                timeout=60.0,
            )
            dt = time.perf_counter() - t0
            answers.append({
                "prompt": p,
                "answer": (r.json().get("message", {}).get("content")
                           if r.status_code == 200 else f"HTTP {r.status_code}"),
                "seconds": round(dt, 2),
            })
        except Exception as exc:
            answers.append({
                "prompt": p,
                "answer": "",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - t0, 2),
            })
    return {
        "model": model,
        "n": len(answers),
        "total_seconds": round(time.time() - started, 2),
        "answers": answers,
    }


# ============================================================================
# Subcommand: search
# ============================================================================

def cmd_search(args) -> int:
    """Run the existing multi_model_sweep + judge_eval, wrapped in our
    orchestrator + power sampler."""
    if not _check_ollama(args.ollama_url):
        return 2

    from bench.run import make_run, stage, finalize

    cfg = {
        "mode": "search",
        "models": _models_to_test(),
        "judge_model": args.judge,
        "limit_questions": args.questions or 0,
        "limit_models": args.limit_models or 0,
        "power": not args.no_power,
        "corpus": str(args.corpus) if args.corpus else None,
        "skip_pull": args.skip_pull,
    }
    invocation = " ".join(sys.argv)
    run = make_run(out_dir=Path(args.out) if args.out else None,
                    corpus_dir=Path(args.corpus) if args.corpus else None,
                    cli_invocation=invocation,
                    config=cfg, ollama_url=args.ollama_url)

    if cfg["power"]:
        run.sampler.start()

    _say(f"\n>>> Bench: writing to {run.out_dir}")
    _say(f">>> System: {run.sysinfo.cpu_model} · "
         f"{run.sysinfo.ram_total_gb:.0f} GB RAM")
    for g in run.sysinfo.gpus:
        _say(f">>>   GPU {g.index}: {g.name} · "
             f"{g.vram_total_mb / 1024:.0f} GB")

    workspace = args.workspace or os.environ.get("EZRAG_WORKSPACE")
    if not workspace:
        _err("No workspace specified. Pass --workspace or set EZRAG_WORKSPACE.")
        finalize(run)
        return 2

    # Stage 1 — multi-model sweep
    sweep_json: Optional[Path] = None
    try:
        with stage(run, "search.sweep"):
            sweep_json = _run_sweep(workspace, run, args)
            run.stages["search.sweep"]["json"] = str(sweep_json)
    except Exception as exc:
        run.record_error("search.sweep", exc)
        bundle = run.write_diagnostic_bundle()
        _err(f"Sweep failed: {exc}")
        _err(f"Diagnostic bundle: {bundle}")
        finalize(run)
        return 1

    # Stage 2 — judge
    try:
        with stage(run, "search.judge"):
            judged_json = _run_judge(sweep_json, args)
            run.stages["search.judge"]["json"] = str(judged_json)
    except Exception as exc:
        run.record_error("search.judge", exc)
        bundle = run.write_diagnostic_bundle()
        _err(f"Judge failed: {exc}")
        _err(f"Diagnostic bundle: {bundle}")
        finalize(run)
        return 1

    # Stage 3 — render report
    try:
        with stage(run, "search.report"):
            html = _render_html(judged_json, run.out_dir / "report.html")
            run.stages["search.report"]["html"] = str(html)
    except Exception as exc:
        # Report-render failure shouldn't kill the bundle
        run.record_error("search.report", exc)

    finalize(run)
    _say("\n[OK] Search bench complete.")
    _say(f"     Bundle: {run.out_dir}")
    if (run.out_dir / "report.html").is_file():
        _say(f"     Report: {run.out_dir / 'report.html'}")
    return 0


def _run_sweep(workspace: str, run, args) -> Path:
    """Invoke bench/multi_model_sweep.py as a subprocess to keep crash
    isolation. Returns the path to its JSON output."""
    cmd = [
        sys.executable, "-X", "utf8", "-u",
        str(REPO_ROOT / "bench" / "multi_model_sweep.py"),
        "--workspace", workspace,
        "--out-dir", str(run.out_dir / "search"),
    ]
    if args.skip_pull:
        cmd.append("--skip-pull")
    if args.limit_questions:
        cmd += ["--limit-questions", str(args.limit_questions)]
    if args.limit_models:
        cmd += ["--limit-models", str(args.limit_models)]

    _say(f"  >>> {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"multi_model_sweep exited {proc.returncode}")

    # Find the JSON it wrote (newest multimodel-*.json under search/)
    candidates = sorted(
        (run.out_dir / "search").glob("multimodel-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("multi_model_sweep wrote no JSON")
    return candidates[0]


def _run_judge(sweep_json: Path, args) -> Path:
    cmd = [
        sys.executable, "-X", "utf8", "-u",
        str(REPO_ROOT / "bench" / "judge_eval.py"),
        str(sweep_json),
        "--judge-model", args.judge,
        "--llm-url", args.ollama_url,
    ]
    _say(f"  >>> {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"judge_eval exited {proc.returncode}")
    judged = sweep_json.with_name(sweep_json.stem + "-judged.json")
    if not judged.is_file():
        raise RuntimeError("judge_eval wrote no -judged.json")
    return judged


def _render_html(judged_json: Path, out_html: Path) -> Path:
    cmd = [
        sys.executable, "-X", "utf8", "-u",
        str(REPO_ROOT / "bench" / "multi_model_html.py"),
        str(judged_json),
        "--out", str(out_html),
    ]
    _say(f"  >>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"multi_model_html exited {proc.returncode}")
    return out_html


# ============================================================================
# Subcommand: full / ingest / consolidate / report (stubs in v1)
# ============================================================================

def cmd_full(args) -> int:
    """For v1, `full` == `search` since the ingest-bench module isn't
    written yet. Future: chain ingest + search + power summary report."""
    _say(">>> NOTE: 'full' currently runs only the search phase. "
         "Ingest matrix benchmarking is planned (Phase 5 of bench plan).")
    return cmd_search(args)


def cmd_ingest(args) -> int:
    _err("Ingest matrix benchmarking is not yet implemented.")
    _err("See docs/PLAN_BENCH_MODE.md Part C / Phase 5.")
    return 64


def cmd_consolidate(args) -> int:
    _err("Cross-system consolidation is not yet implemented.")
    _err("See docs/PLAN_BENCH_MODE.md Part E / Phase 10.")
    return 64


def cmd_report(args) -> int:
    """Rendering only — re-render HTML from an existing judged JSON."""
    judged = Path(args.judged_json)
    if not judged.is_file():
        _err(f"judged JSON not found: {judged}")
        return 1
    out = Path(args.out) if args.out else judged.with_suffix(".html")
    _render_html(judged, out)
    _say(f"[OK] Report: {out}")
    return 0


# ============================================================================
# Argparse
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ez-rag-bench",
        description=(
            "Cross-platform benchmark suite for ez-rag.\n"
            "Captures system info, runs ingest + search benchmarks, "
            "samples power, and emits a self-contained bundle."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ez-rag-bench probe\n"
            "  ez-rag-bench quick --corpus ./bench/test-corpus\n"
            "  ez-rag-bench search --workspace ~/Desktop/dnd --questions 10\n"
            "\n"
            "When something breaks, the bundle includes a diagnostic_bundle.\n"
            "zip you can drop into your AI coding agent.\n"
        ),
    )
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # probe
    sp = sub.add_parser("probe",
                         help="Print system probe + gating decision.")
    sp.set_defaults(func=cmd_probe)

    # quick
    sp = sub.add_parser("quick",
                         help="5-min sanity bench (single model, 5 prompts).")
    sp.add_argument("--corpus", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_quick)

    # search
    sp = sub.add_parser("search",
                         help="Multi-model sweep + LLM judge + report.")
    sp.add_argument("--workspace", default=None,
                     help="Path to an ez-rag workspace (already ingested). "
                          "Or set $EZRAG_WORKSPACE.")
    sp.add_argument("--corpus", default=None)
    sp.add_argument("--out", default=None)
    sp.add_argument("--judge", default="qwen2.5:7b")
    sp.add_argument("--questions", type=int, default=0,
                     help="Limit question count (0 = all).")
    sp.add_argument("--limit-models", type=int, default=0,
                     help="Limit model count (0 = all that fit).")
    sp.add_argument("--skip-pull", action="store_true")
    sp.add_argument("--no-power", action="store_true",
                     help="Disable power sampling.")
    sp.set_defaults(func=cmd_search)

    # full
    sp = sub.add_parser("full",
                         help="Run everything (currently == search).")
    sp.add_argument("--workspace", default=None)
    sp.add_argument("--corpus", default=None)
    sp.add_argument("--out", default=None)
    sp.add_argument("--judge", default="qwen2.5:7b")
    sp.add_argument("--questions", type=int, default=0)
    sp.add_argument("--limit-models", type=int, default=0)
    sp.add_argument("--skip-pull", action="store_true")
    sp.add_argument("--no-power", action="store_true")
    sp.set_defaults(func=cmd_full)

    # ingest (placeholder)
    sp = sub.add_parser("ingest", help="Ingest matrix sweep (TBD).")
    sp.set_defaults(func=cmd_ingest)

    # consolidate (placeholder)
    sp = sub.add_parser("consolidate",
                         help="Merge multiple bundles (TBD).")
    sp.add_argument("dirs", nargs="+")
    sp.set_defaults(func=cmd_consolidate)

    # report
    sp = sub.add_parser("report",
                         help="Re-render HTML from a judged JSON.")
    sp.add_argument("judged_json")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_report)

    return ap


# ============================================================================
# Entrypoint
# ============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Bench orchestrator — produces a self-contained result bundle.

This module owns the lifecycle of one bench run:
  - Create the output directory
  - Capture sysinfo, write manifest.json
  - Spin up the power sampler
  - Hand control to the various stage runners (sweep, judge, ingest)
  - Render the report
  - On crash: write diagnostic_bundle.zip with everything an AI agent
    needs to triage offline

The orchestrator is deliberately thin — it doesn't know HOW each
stage works, just that each stage:
  1. takes a handle to this orchestrator
  2. produces JSON-serializable results
  3. fires sampler.set_segment(label) so power data is attributed

This file exists so `bench/cli.py` doesn't end up doing all of it.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bench.sysinfo import (
    SystemInfo, gather_system_info, to_dict as sysinfo_to_dict,
)
from bench.power import PowerSampler


# ============================================================================
# Run handle
# ============================================================================

@dataclass
class BenchRun:
    out_dir: Path
    sysinfo: SystemInfo
    sampler: PowerSampler
    started_at: float = field(default_factory=time.time)
    bench_version: str = "0.1.0"
    cli_invocation: str = ""
    config: dict = field(default_factory=dict)
    # Stage results — populated by stage runners
    stages: dict[str, dict] = field(default_factory=dict)
    # Errors captured per stage
    errors: list[dict] = field(default_factory=list)

    # ---- Result accessors ----

    def stage_dir(self, name: str) -> Path:
        d = self.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_manifest(self) -> None:
        """Persist the current state of the run. Called repeatedly so
        a crash doesn't lose data."""
        manifest = {
            "bench_version": self.bench_version,
            "cli_invocation": self.cli_invocation,
            "config": self.config,
            "started_at": self.started_at,
            "now": time.time(),
            "duration_s": time.time() - self.started_at,
            "system_info": sysinfo_to_dict(self.sysinfo),
            "stages": self.stages,
            "errors": self.errors,
            "power_capabilities": {
                "nvidia": self.sampler.has_nvidia,
                "cpu": self.sampler.has_cpu,
            },
        }
        path = self.out_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )

    def record_stage(self, name: str, result: dict) -> None:
        self.stages[name] = result
        self.write_manifest()

    def record_error(self, stage: str, exc: BaseException) -> None:
        self.errors.append({
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "trace": traceback.format_exc(),
            "ts": time.time(),
        })
        self.write_manifest()

    # ---- Diagnostic bundle ----

    def write_diagnostic_bundle(self, *, last_log_lines: int = 200
                                 ) -> Path:
        """Zip up the manifest + sysinfo + the most recent errors so
        the user can drop the bundle into an AI coding agent."""
        bundle_path = self.out_dir / "diagnostic_bundle.zip"
        readme = self.out_dir / "diagnostic_README.md"
        readme.write_text(
            "# ez-rag-bench diagnostic bundle\n\n"
            "Open this directory in your AI coding agent (Claude Code, "
            "Cursor, etc.) and ask:\n\n"
            "    Read `diagnostic_README.md`, `manifest.json`, and the "
            "errors. Diagnose why the bench failed and propose a fix.\n\n"
            "## What's in here\n\n"
            "- `manifest.json` — system info, config, stage status, "
            "captured errors with full Python traces.\n"
            "- `*-judged.json` / `*.json` (if present) — partial results "
            "from any stages that completed before the failure.\n"
            "- `power_samples.csv` (if power sampling was on) — the "
            "wattage time-series up to the point of failure.\n\n"
            "## What's NOT in here\n\n"
            "- Your corpus content. Only file metadata (counts, "
            "extensions, sizes) was captured.\n"
            "- API keys / credentials. Manifest scrubs known sensitive "
            "fields.\n",
            encoding="utf-8",
        )

        files_to_bundle: list[Path] = []
        for candidate in (
            self.out_dir / "manifest.json",
            self.out_dir / "diagnostic_README.md",
            self.out_dir / "system_info.json",
            self.out_dir / "power_samples.csv",
            self.out_dir / "search" / "judged.json",
            self.out_dir / "search" / "raw_answers.json",
            self.out_dir / "ingest" / "runs.json",
        ):
            if candidate.is_file():
                files_to_bundle.append(candidate)

        with zipfile.ZipFile(bundle_path, "w",
                              compression=zipfile.ZIP_DEFLATED) as z:
            for f in files_to_bundle:
                z.write(f, f.relative_to(self.out_dir))
        return bundle_path


# ============================================================================
# Orchestrator
# ============================================================================

def make_run(*, out_dir: Optional[Path] = None,
              corpus_dir: Optional[Path] = None,
              cli_invocation: str = "",
              config: Optional[dict] = None,
              ollama_url: str = "http://127.0.0.1:11434",
              power_interval_s: float = 0.5,
              ) -> BenchRun:
    """Initialize a fresh BenchRun. Captures sysinfo, sets up the
    output directory, primes the power sampler (but doesn't start it
    yet — caller decides when sampling begins)."""
    if out_dir is None:
        host = socket.gethostname() or "host"
        sysid_short = ""
        try:
            si = gather_system_info(skip_disk_probe=True)
            sysid_short = si.short_system_id[:8]
        except Exception:
            sysid_short = "00000000"
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = REPO_ROOT / "bench-results" / f"{host}-{sysid_short}-{ts}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = gather_system_info(
        bench_run_id=out_dir.name,
        docs_dir=corpus_dir,
        ollama_url=ollama_url,
    )
    # Persist sysinfo immediately — useful even if the run dies early
    (out_dir / "system_info.json").write_text(
        json.dumps(sysinfo_to_dict(info), indent=2),
        encoding="utf-8",
    )

    sampler = PowerSampler(interval_s=power_interval_s)

    run = BenchRun(
        out_dir=out_dir,
        sysinfo=info,
        sampler=sampler,
        cli_invocation=cli_invocation,
        config=config or {},
    )
    run.write_manifest()
    return run


@contextmanager
def stage(run: BenchRun, name: str):
    """Context manager that:
      - tags power samples with the stage name
      - times the stage
      - writes manifest on entry/exit
      - captures any exception into the errors list and re-raises so
        the orchestrator can decide whether to keep going

    The stage entry is created on ENTER so callers can write
    `run.stages[name]["foo"] = ...` inside the block.
    """
    t0 = time.time()
    run.sampler.set_segment(name)
    # Pre-populate so the body can write into the stage dict.
    run.stages.setdefault(name, {})
    run.stages[name].setdefault("status", "running")
    run.stages[name]["started_at"] = t0
    try:
        yield
        run.stages[name]["status"] = "ok"
        run.stages[name]["duration_s"] = time.time() - t0
        run.write_manifest()
    except Exception as exc:
        run.record_error(name, exc)
        run.stages[name]["status"] = "error"
        run.stages[name]["duration_s"] = time.time() - t0
        run.write_manifest()
        raise
    finally:
        run.sampler.set_segment("")


def finalize(run: BenchRun) -> None:
    """Stop the sampler, dump CSV + summary, write final manifest."""
    run.sampler.stop()
    csv_path = run.out_dir / "power_samples.csv"
    run.sampler.write_csv(csv_path)
    summary = run.sampler.summary()
    (run.out_dir / "power_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    # Roll segment summaries into the manifest under stages
    for label, seg in summary.items():
        st = run.stages.setdefault(label, {})
        st["power"] = seg
    run.write_manifest()

"""Tests for bench/sysinfo.py, bench/power.py, bench/run.py.

Doesn't actually run the bench end-to-end (that's covered by the
CLI smoke test). Verifies:
  - sysinfo captures the right shape, never raises
  - power sampler segment marking + summary math
  - BenchRun stage() context manager pre-populates the dict
  - BenchRun writes manifest atomically + diagnostic bundle on demand
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    # ========================================================================
    # sysinfo
    # ========================================================================
    print("\n[1] sysinfo.gather_system_info — basic shape")
    from bench.sysinfo import (
        SystemInfo, gather_system_info, to_dict,
    )
    info = gather_system_info(skip_disk_probe=True)
    check("returns SystemInfo",
          isinstance(info, SystemInfo))
    check("hostname populated",
          bool(info.hostname))
    check("short_system_id is 12 hex chars",
          len(info.short_system_id) == 12
          and all(c in "0123456789abcdef" for c in info.short_system_id))
    check("os_name populated",
          bool(info.os_name))
    check("python_version populated",
          bool(info.python_version)
          and info.python_version[0].isdigit())
    check("bench_run_id populated",
          bool(info.bench_run_id))
    check("to_dict() returns JSON-serializable dict",
          isinstance(to_dict(info), dict)
          and json.dumps(to_dict(info)))   # round-trip

    print("\n[2] sysinfo never raises on weirdness")
    # Even with skip_disk_probe and a bogus Ollama URL, it should
    # complete cleanly with probe_warnings populated as needed.
    info2 = gather_system_info(
        skip_disk_probe=True,
        ollama_url="http://invalid.localhost:55555",
    )
    # Note: when the URL probe fails, _probe_ollama falls back to
    # the `ollama --version` CLI. So on a box that has the CLI
    # installed (most users), the version is still reported even
    # with an unreachable URL. Both outcomes are valid behavior.
    check("ollama probe captured a version OR 'unreachable'",
          (info2.ollama_version == "unreachable"
           or (info2.ollama_version
               and info2.ollama_version[0].isdigit())),
          f"got {info2.ollama_version!r}")
    check("doesn't raise on bad ollama url",
          isinstance(info2, SystemInfo))

    print("\n[3] same machine -> same short_system_id (stable)")
    info3 = gather_system_info(skip_disk_probe=True)
    check("stable short_system_id",
          info.short_system_id == info3.short_system_id)

    # ========================================================================
    # power
    # ========================================================================
    print("\n[4] power.PowerSampler — basic lifecycle")
    from bench.power import PowerSampler
    sampler = PowerSampler(interval_s=0.2)
    sampler.start()
    time.sleep(0.5)
    sampler.stop()
    samples = sampler.samples()
    check("got at least one sample",
          len(samples) >= 1, f"got {len(samples)} samples")
    check("sample has ts > 0",
          all(s.ts > 0 for s in samples))

    print("\n[5] segment marking")
    sampler = PowerSampler(interval_s=0.1)
    sampler.start()
    with sampler.measure("phase-a"):
        time.sleep(0.3)
    with sampler.measure("phase-b"):
        time.sleep(0.3)
    sampler.stop()
    summary = sampler.summary()
    check("two segments in summary",
          "phase-a" in summary and "phase-b" in summary)
    check("phase-a duration ≥ 0.2s",
          summary["phase-a"]["duration_s"] >= 0.2)
    check("phase-b duration ≥ 0.2s",
          summary["phase-b"]["duration_s"] >= 0.2)

    print("\n[6] CSV write")
    sampler = PowerSampler(interval_s=0.1)
    sampler.start()
    with sampler.measure("csv-test"):
        time.sleep(0.3)
    sampler.stop()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.csv"
        sampler.write_csv(path)
        check("CSV file created", path.is_file())
        text = path.read_text(encoding="utf-8")
        check("CSV has header",
              text.splitlines()[0].startswith("ts,segment"))
        check("CSV has data rows",
              len(text.splitlines()) >= 2)

    # ========================================================================
    # run.py — BenchRun + stage()
    # ========================================================================
    print("\n[7] BenchRun creates output directory + manifest")
    from bench.run import make_run, stage, finalize
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bench-out"
        run = make_run(out_dir=out, cli_invocation="test", config={"x": 1})
        check("out_dir created", out.is_dir())
        check("manifest.json exists",
              (out / "manifest.json").is_file())
        check("system_info.json exists",
              (out / "system_info.json").is_file())
        manifest = json.loads(
            (out / "manifest.json").read_text(encoding="utf-8")
        )
        check("manifest has cli_invocation",
              manifest["cli_invocation"] == "test")
        check("manifest has system_info",
              "system_info" in manifest)
        finalize(run)

    print("\n[8] stage() pre-populates the dict so body can write into it")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stage-test"
        run = make_run(out_dir=out)
        run.sampler.start()
        with stage(run, "test-stage"):
            # This is the bug from the smoke test — must work.
            run.stages["test-stage"]["my_key"] = "my_value"
        check("stage entry exists after exit",
              "test-stage" in run.stages)
        check("body's writes are preserved",
              run.stages["test-stage"].get("my_key") == "my_value")
        check("status = ok",
              run.stages["test-stage"]["status"] == "ok")
        check("duration_s is positive",
              run.stages["test-stage"]["duration_s"] >= 0)
        finalize(run)

    print("\n[9] stage() captures errors into the errors list + re-raises")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "err-test"
        run = make_run(out_dir=out)
        run.sampler.start()
        raised = False
        try:
            with stage(run, "broken-stage"):
                raise RuntimeError("synthetic")
        except RuntimeError:
            raised = True
        check("exception was re-raised", raised)
        check("error captured",
              len(run.errors) == 1
              and run.errors[0]["stage"] == "broken-stage")
        check("status = error",
              run.stages["broken-stage"]["status"] == "error")
        finalize(run)

    print("\n[10] diagnostic_bundle.zip captures the right files")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bundle-test"
        run = make_run(out_dir=out)
        run.sampler.start()
        try:
            with stage(run, "fail"):
                raise ValueError("synthetic failure")
        except ValueError:
            pass
        bundle = run.write_diagnostic_bundle()
        check("bundle file created", bundle.is_file())
        with zipfile.ZipFile(bundle) as z:
            names = set(z.namelist())
        check("bundle contains manifest.json",
              "manifest.json" in names)
        check("bundle contains system_info.json",
              "system_info.json" in names)
        check("bundle contains diagnostic_README.md",
              "diagnostic_README.md" in names)
        finalize(run)

    print(f"\n=== bench orchestrator summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

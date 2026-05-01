"""Smoke tests for the sysmon module.

These don't assert exact values (those depend on the host machine) — just
that the public API returns a populated Sample without raising and that
the format helpers handle None.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import sysmon


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    print("\n[1] sample() returns a populated Sample")
    sysmon.sample()           # warm CPU% delta
    time.sleep(0.3)
    s = sysmon.sample()
    check("sample is a Sample instance",
          isinstance(s, sysmon.Sample), f"got {type(s)}")

    print("\n[2] CPU + RAM (psutil) populated where available")
    if s.has_psutil:
        check("cpu_pct in [0, 100] when present",
              s.cpu_pct is None or 0.0 <= s.cpu_pct <= 100.0,
              f"got {s.cpu_pct}")
        check("cpu_count is positive when present",
              s.cpu_count is None or s.cpu_count > 0,
              f"got {s.cpu_count}")
        check("ram_total_gb is positive",
              s.ram_total_gb is not None and s.ram_total_gb > 0,
              f"got {s.ram_total_gb}")
        check("ram_used_gb <= ram_total_gb",
              (s.ram_used_gb or 0) <= (s.ram_total_gb or 0) + 0.5,
              f"used={s.ram_used_gb} total={s.ram_total_gb}")
    else:
        print("  SKIP  psutil not installed in this env")

    print("\n[3] GPU sample (only when nvidia-smi present)")
    if s.has_nvidia:
        gpu = s.gpus[0]
        check("GPU has a name",
              bool(gpu.name), f"got {gpu.name!r}")
        check("GPU vram_total > vram_used",
              (gpu.vram_used_mb or 0) <= (gpu.vram_total_mb or 0) + 1.0,
              f"used={gpu.vram_used_mb} total={gpu.vram_total_mb}")
        check("vram_pct computes",
              gpu.vram_pct is None or 0.0 <= gpu.vram_pct <= 100.0,
              f"got {gpu.vram_pct}")
    else:
        print("  SKIP  nvidia-smi not on PATH")

    print("\n[4] format helpers handle None")
    check("fmt_pct(None) -> dash",
          sysmon.fmt_pct(None) == "—", f"got {sysmon.fmt_pct(None)!r}")
    check("fmt_gb(None) -> dash",
          sysmon.fmt_gb(None) == "—", f"got {sysmon.fmt_gb(None)!r}")
    check("fmt_mb_as_gb(None) -> dash",
          sysmon.fmt_mb_as_gb(None) == "—",
          f"got {sysmon.fmt_mb_as_gb(None)!r}")
    check("fmt_temp_c(None) -> dash",
          sysmon.fmt_temp_c(None) == "—",
          f"got {sysmon.fmt_temp_c(None)!r}")
    check("fmt_pct(42.7) rounds",
          sysmon.fmt_pct(42.7) == "43%",
          f"got {sysmon.fmt_pct(42.7)!r}")
    check("fmt_gb(1.234, decimals=1)",
          sysmon.fmt_gb(1.234) == "1.2 GB",
          f"got {sysmon.fmt_gb(1.234)!r}")

    print("\n[5] sample() is fast (cheap enough for 1 Hz polling)")
    t0 = time.perf_counter()
    for _ in range(5):
        sysmon.sample()
    elapsed = time.perf_counter() - t0
    check("5 samples in < 2s (avg < 400ms)",
          elapsed < 2.0, f"elapsed={elapsed:.2f}s")

    print(f"\n=== Sysmon summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

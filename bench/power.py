"""Cross-platform power & utilization sampler for the bench bundle.

Background thread polls every interval_s seconds and records:
  - per-GPU power draw (W), GPU utilization (%), VRAM used (MB)
  - CPU power (Linux RAPL / macOS powermetrics / Windows: skipped)
  - System power (best-effort)

Records are kept in memory and dumped to a CSV when the sampler stops.
Each record has a `segment` label so the bench can carve up energy
use by phase (e.g. "ingest.qwen2.5:7b", "search.deepseek-r1:32b").

Public API:
    sampler = PowerSampler()
    sampler.start()
    with sampler.measure("ingest.qwen2.5:7b"):
        ...do work...
    sampler.stop()
    sampler.write_csv(path)
    sampler.summary()  -> dict per-segment energy totals

The sampler degrades gracefully:
  - No nvidia-smi → no GPU data, but CPU/system data still flows
  - No RAPL access → no CPU data, but GPU data still flows
  - Nothing works → still produces samples (with all-zero power), so
    the duration of each segment is recorded for timing analysis
"""
from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# Public types
# ============================================================================

@dataclass
class PowerSample:
    ts: float                          # epoch seconds
    segment: str                       # current label or "" for unsegmented
    gpu_idx: int                       # -1 = aggregate / non-GPU sample
    gpu_power_w: float = 0.0
    gpu_util_pct: float = 0.0
    gpu_vram_used_mb: int = 0
    cpu_power_w: float = 0.0
    system_power_w: float = 0.0


@dataclass
class SegmentSummary:
    label: str
    duration_s: float
    sample_count: int
    gpu_energy_kj: float
    cpu_energy_kj: float
    system_energy_kj: float
    gpu_avg_w: float
    gpu_peak_w: float
    gpu_util_avg_pct: float
    vram_peak_mb: int

    def total_energy_kj(self) -> float:
        return self.gpu_energy_kj + self.cpu_energy_kj


# ============================================================================
# nvidia-smi probe
# ============================================================================

def _which(name: str) -> Optional[str]:
    """Locate a binary, including common Windows install paths."""
    found = shutil.which(name)
    if found:
        return found
    if platform.system() == "Windows":
        for d in (
            r"C:\Program Files\NVIDIA Corporation\NVSMI",
            r"C:\Windows\System32",
        ):
            candidate = Path(d) / f"{name}.exe"
            if candidate.exists():
                return str(candidate)
    return None


def _probe_nvidia_once() -> list[dict]:
    """Single nvidia-smi probe. Returns list of dicts, one per GPU.
    Empty list if nvidia-smi isn't present or returns garbage."""
    smi = _which("nvidia-smi")
    if not smi:
        return []
    cmd = [
        smi,
        "--query-gpu=index,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=2.0, check=False)
        if r.returncode != 0 or not r.stdout:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    out: list[dict] = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            out.append({
                "idx": int(parts[0]),
                "power_w": float(parts[1]) if parts[1] != "[N/A]" else 0.0,
                "util_pct": float(parts[2]) if parts[2] != "[N/A]" else 0.0,
                "vram_used_mb": int(parts[3]) if parts[3] != "[N/A]" else 0,
            })
        except (ValueError, IndexError):
            continue
    return out


# ============================================================================
# CPU power — Intel RAPL on Linux (the cleanest probe)
# ============================================================================

class _RAPLProbe:
    """Intel RAPL energy probe via /sys/class/powercap/intel-rapl:0/energy_uj.

    RAPL gives a monotonically-increasing microjoule counter. We compute
    instantaneous wattage as deltaJ / deltaT between samples.
    Wraps on overflow (~2^32 µJ → ~71 minutes at 100 W). We detect and
    bridge the wrap.
    """
    def __init__(self):
        self.path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
        self.max_path = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
        self.available = self.path.is_file() and os.access(self.path, os.R_OK)
        self._last_uj: Optional[int] = None
        self._last_ts: Optional[float] = None
        self._max_uj = self._read_max()

    def _read_max(self) -> int:
        if not self.max_path.is_file():
            return 1 << 32
        try:
            return int(self.max_path.read_text().strip())
        except (OSError, ValueError):
            return 1 << 32

    def sample(self) -> float:
        """Return instantaneous wattage since the last call. Returns
        0.0 on the first call (no delta yet) or if RAPL is unavailable."""
        if not self.available:
            return 0.0
        try:
            uj = int(self.path.read_text().strip())
        except (OSError, ValueError):
            return 0.0
        now = time.monotonic()
        if self._last_uj is None or self._last_ts is None:
            self._last_uj, self._last_ts = uj, now
            return 0.0
        delta_t = now - self._last_ts
        if delta_t <= 0:
            return 0.0
        delta_uj = uj - self._last_uj
        if delta_uj < 0:
            # Counter wrapped
            delta_uj += self._max_uj
        self._last_uj, self._last_ts = uj, now
        return (delta_uj / 1_000_000) / delta_t   # joules per second = W


# ============================================================================
# CPU power — macOS via powermetrics (requires sudo, optional)
# ============================================================================

class _PowermetricsProbe:
    """Best-effort macOS CPU/GPU power. powermetrics requires root and
    streams output; we don't try to integrate that into the polling
    loop. Instead we accept that macOS power is opt-in and skip it
    silently when unavailable.
    """
    def __init__(self):
        self.available = False   # explicitly disabled in v1

    def sample(self) -> float:
        return 0.0


# ============================================================================
# CPU power — Windows
# ============================================================================

class _WindowsCpuProbe:
    """Windows doesn't expose CPU power without vendor-specific tooling.
    Skip cleanly; the report will note CPU power as unmeasured."""
    def __init__(self):
        self.available = False

    def sample(self) -> float:
        return 0.0


def _make_cpu_probe():
    sysname = platform.system()
    if sysname == "Linux":
        return _RAPLProbe()
    if sysname == "Darwin":
        return _PowermetricsProbe()
    if sysname == "Windows":
        return _WindowsCpuProbe()
    # Unknown OS — return a dud
    p = _WindowsCpuProbe()
    p.available = False
    return p


# ============================================================================
# Sampler
# ============================================================================

class PowerSampler:
    """Background-thread power sampler.

    Usage:
        sampler = PowerSampler(interval_s=0.5)
        sampler.start()
        with sampler.measure("ingest"):
            run_ingest()
        with sampler.measure("search"):
            run_search()
        sampler.stop()
        sampler.write_csv(out_dir / "power_samples.csv")
        summaries = sampler.summary()   # dict[label -> SegmentSummary]
    """
    def __init__(self, *, interval_s: float = 0.5):
        self.interval_s = max(0.1, float(interval_s))
        self._samples: list[PowerSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._segment_lock = threading.Lock()
        self._current_segment = ""
        self._current_segment_start: Optional[float] = None
        self._cpu_probe = _make_cpu_probe()
        # Closed segments: list of (label, start_ts, end_ts).
        self._segment_log: list[tuple[str, float, float]] = []
        # Probe capabilities at init for diagnostic reporting
        self.has_nvidia = bool(_probe_nvidia_once() or _which("nvidia-smi"))
        self.has_cpu = self._cpu_probe.available

    # ---- segment marking ----

    def set_segment(self, label: str) -> None:
        with self._segment_lock:
            if self._current_segment == label:
                return
            # Close the previous segment (if any).
            if (self._current_segment
                    and self._current_segment_start is not None):
                self._segment_log.append((
                    self._current_segment,
                    self._current_segment_start,
                    time.time(),
                ))
            # Open a new segment (or go to "unsegmented" when label="").
            self._current_segment = label
            self._current_segment_start = time.time() if label else None

    @contextmanager
    def measure(self, label: str):
        prev = self._current_segment
        self.set_segment(label)
        try:
            yield
        finally:
            self.set_segment(prev)

    # ---- thread lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ezrag-power-sampler",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Close any final segment using its actual start time.
        with self._segment_lock:
            if (self._current_segment
                    and self._current_segment_start is not None):
                self._segment_log.append((
                    self._current_segment,
                    self._current_segment_start,
                    time.time(),
                ))
            self._current_segment = ""
            self._current_segment_start = None

    def _run(self) -> None:
        # Prime the CPU probe so the first delta isn't huge
        self._cpu_probe.sample()
        while not self._stop.wait(self.interval_s):
            self._take_sample()

    def _take_sample(self) -> None:
        ts = time.time()
        segment = self._current_segment
        cpu_w = self._cpu_probe.sample()
        gpu_records = _probe_nvidia_once()
        if not gpu_records:
            # Still record CPU/segment timing
            self._samples.append(PowerSample(
                ts=ts, segment=segment, gpu_idx=-1,
                cpu_power_w=cpu_w,
            ))
            return
        for g in gpu_records:
            self._samples.append(PowerSample(
                ts=ts, segment=segment, gpu_idx=g["idx"],
                gpu_power_w=g["power_w"], gpu_util_pct=g["util_pct"],
                gpu_vram_used_mb=g["vram_used_mb"],
                cpu_power_w=cpu_w if g["idx"] == 0 else 0.0,
            ))

    # ---- output ----

    def samples(self) -> list[PowerSample]:
        return list(self._samples)

    def write_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "segment", "gpu_idx", "gpu_power_w", "gpu_util_pct",
                "gpu_vram_used_mb", "cpu_power_w", "system_power_w",
            ])
            for s in self._samples:
                w.writerow([
                    f"{s.ts:.3f}", s.segment, s.gpu_idx,
                    f"{s.gpu_power_w:.2f}", f"{s.gpu_util_pct:.1f}",
                    s.gpu_vram_used_mb, f"{s.cpu_power_w:.2f}",
                    f"{s.system_power_w:.2f}",
                ])

    def summary(self) -> dict:
        """Per-segment aggregated stats. Keys are segment labels.
        Energy is integrated via trapezoidal rule over the samples
        landing in that segment.

        Duration prefers the segment-log entries (set_segment start/end
        wall-clock) over sample timestamps, because sparse sampling
        otherwise under-reports short segments. When no segment log
        entry exists for a label (e.g. samples that landed before any
        set_segment call), we fall back to first-to-last sample range.
        """
        # Build duration map from segment_log: label -> total seconds
        seg_durations: dict[str, float] = {}
        for label, t0, t1 in self._segment_log:
            if t1 > t0:
                seg_durations[label] = (
                    seg_durations.get(label, 0.0) + (t1 - t0)
                )

        out: dict[str, SegmentSummary] = {}
        # Group samples by segment + gpu_idx (we treat GPU 0 as primary
        # for energy reporting; multi-GPU is summed)
        by_seg: dict[str, list[PowerSample]] = {}
        for s in self._samples:
            by_seg.setdefault(s.segment or "(unsegmented)", []).append(s)

        for label, group in by_seg.items():
            if not group:
                continue
            group.sort(key=lambda x: x.ts)
            duration = (
                seg_durations.get(label)
                or (group[-1].ts - group[0].ts)
            )
            # Integrate per-GPU power over time, sum GPUs
            gpu_indices = sorted({s.gpu_idx for s in group if s.gpu_idx >= 0})
            gpu_energy_j = 0.0
            gpu_powers: list[float] = []
            gpu_utils: list[float] = []
            vram_peak = 0
            for idx in gpu_indices:
                gs = [s for s in group if s.gpu_idx == idx]
                gs.sort(key=lambda x: x.ts)
                for i in range(1, len(gs)):
                    dt = gs[i].ts - gs[i - 1].ts
                    avg_w = (gs[i].gpu_power_w + gs[i - 1].gpu_power_w) / 2
                    gpu_energy_j += avg_w * dt
                gpu_powers.extend(s.gpu_power_w for s in gs)
                gpu_utils.extend(s.gpu_util_pct for s in gs)
                vram_peak = max(vram_peak,
                                 max((s.gpu_vram_used_mb for s in gs),
                                     default=0))

            cpu_energy_j = 0.0
            cpu_samples = [s for s in group if s.gpu_idx in (-1, 0)]
            cpu_samples.sort(key=lambda x: x.ts)
            for i in range(1, len(cpu_samples)):
                dt = cpu_samples[i].ts - cpu_samples[i - 1].ts
                avg = (cpu_samples[i].cpu_power_w
                       + cpu_samples[i - 1].cpu_power_w) / 2
                cpu_energy_j += avg * dt

            out[label] = SegmentSummary(
                label=label,
                duration_s=duration,
                sample_count=len(group),
                gpu_energy_kj=round(gpu_energy_j / 1000, 3),
                cpu_energy_kj=round(cpu_energy_j / 1000, 3),
                system_energy_kj=0.0,    # not measured in v1
                gpu_avg_w=(round(sum(gpu_powers) / len(gpu_powers), 1)
                           if gpu_powers else 0.0),
                gpu_peak_w=round(max(gpu_powers, default=0.0), 1),
                gpu_util_avg_pct=(
                    round(sum(gpu_utils) / len(gpu_utils), 1)
                    if gpu_utils else 0.0
                ),
                vram_peak_mb=vram_peak,
            )
        return {k: v.__dict__ for k, v in out.items()}


# ============================================================================
# CLI smoke test
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Smoke test the power sampler.")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--out", default="bench/sample_power.csv")
    args = ap.parse_args()

    sampler = PowerSampler(interval_s=args.interval)
    print(f"Capabilities: nvidia-smi={sampler.has_nvidia} "
          f"cpu_probe={sampler.has_cpu}")
    print(f"Sampling for {args.seconds:.1f}s...")
    sampler.start()
    with sampler.measure("smoke-test"):
        time.sleep(args.seconds)
    sampler.stop()
    out = Path(args.out)
    sampler.write_csv(out)
    summary = sampler.summary()
    import json as _json
    print(_json.dumps(summary, indent=2))
    print(f"CSV: {out} ({len(sampler.samples())} samples)")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())

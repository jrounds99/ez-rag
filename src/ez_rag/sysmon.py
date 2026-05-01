"""Lightweight system telemetry sampler for the GUI status bar.

Pulls one snapshot per call:
    - CPU percent (system-wide, since last call)
    - System RAM (used / total / percent)
    - CPU temperature (when available; Linux/Mac mostly, rarely on Windows)
    - NVIDIA GPU compute %, VRAM, temperature (via nvidia-smi)
    - Active Ollama model + estimated VRAM use, when ez-rag knows the URL

Designed to be cheap (~5 ms with NVIDIA driver, ~0.5 ms without) so the
GUI can poll once a second on the watchdog thread. Failures degrade
gracefully — fields the host can't supply come back as None and the UI
just hides them.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional


# psutil is BSD-licensed and runs on Win/Mac/Linux. Imported lazily so
# the rest of ez-rag still works on a stripped-down environment.
def _psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except ImportError:
        return None


@dataclass
class GPUSample:
    index: int
    name: str = ""
    util_pct: float | None = None       # 0..100, GPU compute %
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    temp_c: float | None = None         # GPU edge temp °C
    power_w: float | None = None        # current draw

    @property
    def vram_pct(self) -> float | None:
        if self.vram_used_mb is None or not self.vram_total_mb:
            return None
        return 100.0 * self.vram_used_mb / self.vram_total_mb


@dataclass
class Sample:
    cpu_pct: float | None = None
    cpu_temp_c: float | None = None
    cpu_count: int | None = None        # logical cores
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    ram_pct: float | None = None
    gpus: list[GPUSample] = field(default_factory=list)
    has_nvidia: bool = False
    has_psutil: bool = False


# ---- nvidia-smi ------------------------------------------------------------

_NVIDIA_SMI: str | None = None
_NVIDIA_SMI_RESOLVED = False


def _nvidia_smi_path() -> str | None:
    global _NVIDIA_SMI, _NVIDIA_SMI_RESOLVED
    if _NVIDIA_SMI_RESOLVED:
        return _NVIDIA_SMI
    _NVIDIA_SMI_RESOLVED = True
    p = shutil.which("nvidia-smi")
    if p:
        _NVIDIA_SMI = p
        return p
    # Common Windows fallback when PATH isn't populated for the GUI process
    if os.name == "nt":
        for cand in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.isfile(cand):
                _NVIDIA_SMI = cand
                return cand
    return None


_GPU_QUERY = (
    "index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)


def _sample_gpus() -> list[GPUSample]:
    smi = _nvidia_smi_path()
    if not smi:
        return []
    try:
        out = subprocess.check_output(
            [smi, f"--query-gpu={_GPU_QUERY}",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[GPUSample] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # CSV columns are in the order of _GPU_QUERY. nvidia-smi prints
        # "[Not Supported]" for missing fields — we coerce those to None.
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue

        def _f(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None

        try:
            idx = int(parts[0])
        except ValueError:
            continue
        gpus.append(GPUSample(
            index=idx,
            name=parts[1],
            util_pct=_f(parts[2]),
            vram_used_mb=_f(parts[3]),
            vram_total_mb=_f(parts[4]),
            temp_c=_f(parts[5]),
            power_w=_f(parts[6]),
        ))
    return gpus


# ---- CPU + RAM via psutil --------------------------------------------------

def _sample_cpu_ram(sample: Sample) -> None:
    ps = _psutil()
    if ps is None:
        return
    sample.has_psutil = True
    try:
        # `interval=None` reads the % since the last call, ~0 ms cost.
        sample.cpu_pct = float(ps.cpu_percent(interval=None))
        sample.cpu_count = int(ps.cpu_count(logical=True) or 0) or None
        vm = ps.virtual_memory()
        sample.ram_total_gb = vm.total / (1024 ** 3)
        sample.ram_used_gb = vm.used / (1024 ** 3)
        sample.ram_pct = float(vm.percent)
    except Exception:
        pass

    # CPU temperature — best-effort, format varies wildly by platform.
    # Returns nothing on Windows under default privileges.
    try:
        if hasattr(ps, "sensors_temperatures"):
            temps = ps.sensors_temperatures(fahrenheit=False) or {}
            sample.cpu_temp_c = _pick_cpu_temp(temps)
    except Exception:
        pass


def _pick_cpu_temp(temps: dict) -> float | None:
    """Heuristic: prefer 'coretemp' / 'k10temp' / 'cpu_thermal' entries,
    falling back to the highest reported temp from any group."""
    preferred = ("coretemp", "k10temp", "cpu_thermal", "acpitz")
    for key in preferred:
        for k, v in temps.items():
            if key in k.lower() and v:
                # Average of cores in the group
                vals = [s.current for s in v
                        if getattr(s, "current", None) is not None]
                if vals:
                    return sum(vals) / len(vals)
    # Fallback: any reading at all
    all_vals: list[float] = []
    for v in temps.values():
        for s in v or []:
            cur = getattr(s, "current", None)
            if cur is not None:
                all_vals.append(float(cur))
    if all_vals:
        return max(all_vals)
    return None


# ---- public API ------------------------------------------------------------

def sample() -> Sample:
    """Read one telemetry snapshot. Returns a populated `Sample` regardless
    of which sources were available — callers check the per-field None for
    'unavailable on this host'."""
    s = Sample()
    _sample_cpu_ram(s)
    s.gpus = _sample_gpus()
    s.has_nvidia = bool(s.gpus)
    return s


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{int(round(v))}%"


def fmt_gb(v: float | None, *, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f} GB"


def fmt_mb_as_gb(mb: float | None, *, decimals: int = 1) -> str:
    if mb is None:
        return "—"
    return f"{mb / 1024.0:.{decimals}f} GB"


def fmt_temp_c(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{int(round(v))}°C"


def fmt_power_w(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{int(round(v))} W"

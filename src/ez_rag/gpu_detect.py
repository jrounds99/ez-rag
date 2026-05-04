"""Cross-platform GPU detection.

`detect_gpus()` is the single entry point. It tries every vendor probe
ez-rag knows about (nvidia-smi, pynvml, rocm-smi, xpu-smi, system_profiler,
WMI fallback) and returns a flat list of DetectedGpu records — one per
physical card / logical device the OS exposes.

The function never raises. Probes that fail (binary missing, malformed
output, permission denied) are silently skipped; if everything fails, the
result is an empty list. Callers should treat empty as "no GPU detected,
fall back to CPU mode."

Detection precedence per vendor:
  NVIDIA     nvidia-smi   →  pynvml   →  WMI / lspci name match
  AMD        rocm-smi     →  WMI / lspci name match  →  catalog match
  Intel      xpu-smi      →  WMI / lspci name match
  Apple      system_profiler          (macOS only — not in v1 scope)

NVIDIA is fully supported and tested. AMD and Intel paths exist and follow
vendor docs, but have NOT been validated on real hardware — see the test
matrix in docs/HARDWARE.md when this lands.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .gpu_catalog import GpuSpec, find_spec, runtime_for_architecture


# ============================================================================
# Public dataclass
# ============================================================================

@dataclass
class DetectedGpu:
    """One GPU as actually present on the system at detect time."""
    index: int                          # vendor-local index (nvidia 0..N, amd 0..N)
    vendor: str                         # "nvidia" | "amd" | "intel" | "unknown"
    name: str                           # raw probe name
    matched_spec: Optional[GpuSpec]     # cross-referenced catalog entry
    vram_total_mb: int                  # 0 if unknown
    vram_free_mb: Optional[int]         # None if unknown
    driver_version: Optional[str]
    runtime: str                        # "cuda" | "rocm" | "xpu" | "cpu" | "unknown"
    detection_source: str               # "nvidia-smi" | "pynvml" | "wmi" | …
    health_notes: list[str] = field(default_factory=list)

    @property
    def vram_total_gb(self) -> int:
        return int(round(self.vram_total_mb / 1024)) if self.vram_total_mb else 0

    @property
    def is_compatible(self) -> bool:
        """True when ez-rag can plausibly accelerate on this card.

        We're conservative: 4 GB minimum, known runtime, and either a
        catalog match (so we trust the spec) or NVIDIA (where nvidia-smi
        reporting is reliable enough on its own).
        """
        if self.vram_total_mb < 4096:
            return False
        if self.runtime == "unknown":
            return False
        if self.matched_spec is None and self.vendor != "nvidia":
            return False
        return True


# ============================================================================
# Subprocess wrapper — short timeout, no exceptions
# ============================================================================

def _run(cmd: list[str], *, timeout: float = 5.0) -> Optional[str]:
    """Run a command, return stdout on success, None on any failure.

    Used for every vendor probe. We never want a missing binary or hung
    process to bubble up — detection is a best-effort enrichment, the GUI
    must keep running.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired,
            PermissionError, OSError):
        return None


def _which(name: str) -> Optional[str]:
    """Locate a binary across the usual install paths. shutil.which only
    walks $PATH, but vendor tools sometimes live in /usr/local/cuda/bin
    (CUDA toolkit) or C:\\Program Files\\NVIDIA Corporation\\NVSMI."""
    found = shutil.which(name)
    if found:
        return found
    extra_dirs = [
        "/usr/bin",
        "/usr/local/bin",
        "/usr/local/cuda/bin",
        "/opt/rocm/bin",
        "/opt/intel/oneapi/xpu-smi/bin",
        r"C:\Program Files\NVIDIA Corporation\NVSMI",
        r"C:\Windows\System32",
        r"C:\Program Files\AMD\ROCm\6.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\xpu-smi\bin",
    ]
    suffix = ".exe" if platform.system() == "Windows" else ""
    for d in extra_dirs:
        candidate = Path(d) / f"{name}{suffix}"
        if candidate.exists():
            return str(candidate)
    return None


# ============================================================================
# NVIDIA — primary path: nvidia-smi
# ============================================================================

def _detect_nvidia_smi() -> list[DetectedGpu]:
    smi = _which("nvidia-smi")
    if not smi:
        return []
    out = _run([
        smi,
        "--query-gpu=index,name,memory.total,memory.free,driver_version,"
        "compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus: list[DetectedGpu] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram_total = int(parts[2])
            vram_free = int(parts[3]) if parts[3] else None
            driver = parts[4]
            compute_cap = parts[5] if len(parts) > 5 else ""
        except ValueError:
            continue
        spec = find_spec(name, vram_mb=vram_total)
        notes: list[str] = []
        # Old driver warning — Ollama 0.5+ calls some new APIs that break
        # on driver < 535 for big models.
        try:
            major = int((driver or "").split(".")[0])
            if major and major < 535:
                notes.append(
                    f"Driver {driver} is older than 535 — some Ollama "
                    "models may fail to load."
                )
        except (ValueError, IndexError):
            pass
        if spec and spec.runtime_notes:
            notes.append(spec.runtime_notes)
        if spec and spec.legacy:
            notes.append(
                f"{spec.name} is a legacy architecture; sticking to "
                "Q4_K_M quantization is recommended."
            )
        gpus.append(DetectedGpu(
            index=idx,
            vendor="nvidia",
            name=name,
            matched_spec=spec,
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            driver_version=driver,
            runtime="cuda",
            detection_source="nvidia-smi",
            health_notes=notes,
        ))
    return gpus


# ============================================================================
# NVIDIA — fallback: pynvml (binding to libnvidia-ml)
# ============================================================================

def _detect_nvidia_pynvml() -> list[DetectedGpu]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    gpus: list[DetectedGpu] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8",
                                                                  errors="ignore")
        except Exception:
            driver = None
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name_raw = pynvml.nvmlDeviceGetName(handle)
                name = (name_raw.decode("utf-8", errors="ignore")
                         if isinstance(name_raw, bytes) else str(name_raw))
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_total_mb = int(mem.total // (1024 * 1024))
                vram_free_mb = int(mem.free // (1024 * 1024))
            except Exception:
                continue
            spec = find_spec(name, vram_mb=vram_total_mb)
            gpus.append(DetectedGpu(
                index=i,
                vendor="nvidia",
                name=name,
                matched_spec=spec,
                vram_total_mb=vram_total_mb,
                vram_free_mb=vram_free_mb,
                driver_version=driver,
                runtime="cuda",
                detection_source="pynvml",
            ))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


# ============================================================================
# AMD — primary path: rocm-smi
# ============================================================================

def _detect_rocm_smi() -> list[DetectedGpu]:
    smi = _which("rocm-smi")
    if not smi:
        return []
    # Modern rocm-smi exposes JSON output; older versions don't.
    out = _run([smi, "--showproductname", "--showmeminfo", "vram",
                 "--showdriverversion", "--json"])
    parsed: dict | None = None
    if out:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
    gpus: list[DetectedGpu] = []
    if isinstance(parsed, dict):
        # JSON shape: {"card0": {"Card series": "...", "VRAM Total Memory (B)": "...", ...}, ...}
        for key, info in parsed.items():
            if not isinstance(info, dict):
                continue
            try:
                idx = int(key.replace("card", "")) if "card" in key else 0
            except ValueError:
                idx = 0
            name = (info.get("Card series")
                    or info.get("Card model")
                    or info.get("GPU ID")
                    or "Unknown AMD GPU")
            vram_total_b = info.get("VRAM Total Memory (B)") or "0"
            vram_used_b = info.get("VRAM Total Used Memory (B)") or "0"
            try:
                vram_total_mb = int(int(vram_total_b) // (1024 * 1024))
                vram_used_mb = int(int(vram_used_b) // (1024 * 1024))
                vram_free_mb = max(0, vram_total_mb - vram_used_mb)
            except ValueError:
                vram_total_mb, vram_free_mb = 0, None
            driver = info.get("Driver version")
            spec = find_spec(name, vram_mb=vram_total_mb)
            notes: list[str] = []
            if spec and spec.runtime_notes:
                notes.append(spec.runtime_notes)
            gpus.append(DetectedGpu(
                index=idx,
                vendor="amd",
                name=name,
                matched_spec=spec,
                vram_total_mb=vram_total_mb,
                vram_free_mb=vram_free_mb,
                driver_version=driver,
                runtime="rocm",
                detection_source="rocm-smi",
                health_notes=notes,
            ))
    return gpus


# ============================================================================
# Intel — primary path: xpu-smi
# ============================================================================

def _detect_xpu_smi() -> list[DetectedGpu]:
    smi = _which("xpu-smi")
    if not smi:
        return []
    out = _run([smi, "discovery", "-j"])
    if not out:
        return []
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return []
    gpus: list[DetectedGpu] = []
    devices = parsed.get("device_list") if isinstance(parsed, dict) else None
    if not isinstance(devices, list):
        return []
    for i, dev in enumerate(devices):
        if not isinstance(dev, dict):
            continue
        name = dev.get("device_name") or "Unknown Intel GPU"
        # xpu-smi reports memory in MiB/MB or bytes depending on
        # version. Detect the unit from the key suffix:
        #   *_mib  → already in MiB (treat as MB)
        #   *_mb   → already in MB
        #   else   → bytes (older xpu-smi)
        vram_total_mb = 0
        for key in ("memory_physical_size_mib", "memory_physical_size_mb",
                    "memory_physical_size", "memory_size_mb"):
            v = dev.get(key)
            if v is None:
                continue
            try:
                k = key.lower()
                if k.endswith("_mib") or k.endswith("_mb"):
                    vram_total_mb = int(v)
                else:
                    # bytes
                    vram_total_mb = int(int(v) // (1024 * 1024))
                break
            except ValueError:
                continue
        driver = dev.get("driver_version")
        spec = find_spec(name, vram_mb=vram_total_mb)
        gpus.append(DetectedGpu(
            index=i,
            vendor="intel",
            name=name,
            matched_spec=spec,
            vram_total_mb=vram_total_mb,
            vram_free_mb=None,
            driver_version=driver,
            runtime="xpu",
            detection_source="xpu-smi",
        ))
    return gpus


# ============================================================================
# Windows fallback — WMI Win32_VideoController
# ============================================================================

def _detect_wmi() -> list[DetectedGpu]:
    """Last-resort detection on Windows when no vendor SMI tool is on PATH.

    WMI returns only the adapter name + a 32-bit AdapterRAM value (capped
    at ~4 GB regardless of actual VRAM). We use the name to look up the
    catalog and trust the catalog's vram_gb instead of WMI's truncated
    number.
    """
    if platform.system() != "Windows":
        return []
    cmd = [
        "powershell.exe", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json",
    ]
    out = _run(cmd, timeout=8.0)
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    gpus: list[DetectedGpu] = []
    nvidia_idx, amd_idx, intel_idx = 0, 0, 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("Name") or "").strip()
        if not name:
            continue
        # Skip non-GPU adapters (Microsoft Basic Display Adapter etc.)
        low = name.lower()
        if any(skip in low for skip in (
                "basic display", "remote display", "virtual",
                "hyper-v", "parsec",
        )):
            continue
        adapter_ram = entry.get("AdapterRAM") or 0
        try:
            adapter_ram_mb = int(int(adapter_ram) // (1024 * 1024))
        except (TypeError, ValueError):
            adapter_ram_mb = 0
        driver = entry.get("DriverVersion")
        # Vendor heuristic from name string.
        if "nvidia" in low or "geforce" in low or "quadro" in low:
            vendor = "nvidia"
            idx = nvidia_idx
            nvidia_idx += 1
        elif ("amd" in low or "radeon" in low or "instinct" in low):
            vendor = "amd"
            idx = amd_idx
            amd_idx += 1
        elif "intel" in low or "arc" in low:
            vendor = "intel"
            idx = intel_idx
            intel_idx += 1
        else:
            continue
        spec = find_spec(name, vram_mb=adapter_ram_mb)
        # Trust catalog VRAM over WMI's 4 GB-capped AdapterRAM.
        vram_total_mb = (spec.vram_gb * 1024) if spec else adapter_ram_mb
        runtime = (spec.runtime if spec
                   else runtime_for_architecture(""))
        if runtime == "unknown":
            runtime = {"nvidia": "cuda", "amd": "rocm",
                        "intel": "xpu"}.get(vendor, "unknown")
        notes: list[str] = []
        if not spec:
            notes.append(
                "Could not match this card against the ez-rag catalog. "
                "VRAM and capability info may be incomplete."
            )
        if spec and spec.runtime_notes:
            notes.append(spec.runtime_notes)
        gpus.append(DetectedGpu(
            index=idx,
            vendor=vendor,
            name=name,
            matched_spec=spec,
            vram_total_mb=vram_total_mb,
            vram_free_mb=None,
            driver_version=driver,
            runtime=runtime,
            detection_source="wmi",
            health_notes=notes,
        ))
    return gpus


# ============================================================================
# Master detector
# ============================================================================

def detect_gpus() -> list[DetectedGpu]:
    """Enumerate every GPU on the system.

    Combines results from every vendor probe, deduplicates by
    (vendor, name, vram_total_mb), and returns a stable-ordered list
    (NVIDIA first, then AMD, then Intel, in vendor-local index order).

    Never raises. Returns [] when nothing is found.
    """
    all_gpus: list[DetectedGpu] = []

    # NVIDIA — try nvidia-smi first; fall back to pynvml.
    nvidia = _detect_nvidia_smi()
    if not nvidia:
        nvidia = _detect_nvidia_pynvml()
    all_gpus.extend(nvidia)

    # AMD — rocm-smi.
    all_gpus.extend(_detect_rocm_smi())

    # Intel — xpu-smi.
    all_gpus.extend(_detect_xpu_smi())

    # WMI fallback fills in anything the vendor probes missed (e.g. a
    # consumer machine with no CUDA toolkit installed but a GeForce card).
    if not all_gpus and platform.system() == "Windows":
        all_gpus.extend(_detect_wmi())

    # Dedup: a single card can show up in both nvidia-smi and WMI on
    # Windows. Trust whichever probe gave us VRAM totals.
    seen: dict[tuple, DetectedGpu] = {}
    for g in all_gpus:
        key = (g.vendor, g.name, g.vram_total_mb)
        if key in seen:
            # Prefer the entry with more info (free VRAM, driver, etc).
            existing = seen[key]
            if (existing.vram_free_mb is None
                    and g.vram_free_mb is not None):
                seen[key] = g
            continue
        seen[key] = g

    # Stable order: vendor priority, then index.
    vendor_order = {"nvidia": 0, "amd": 1, "intel": 2, "unknown": 3}
    ordered = sorted(seen.values(),
                     key=lambda g: (vendor_order.get(g.vendor, 9), g.index))

    # Filter out integrated / virtual displays that slipped through
    # (some have <= 1 GB and aren't actual GPUs we want to surface).
    return [g for g in ordered
            if g.vram_total_mb >= 1024 or g.matched_spec is not None]


def primary_gpu(detected: Optional[list[DetectedGpu]] = None
                 ) -> Optional[DetectedGpu]:
    """Pick the most capable detected GPU. Used as the default selection
    when the user hasn't manually chosen one yet."""
    if detected is None:
        detected = detect_gpus()
    if not detected:
        return None
    # Compatible cards beat incompatible; within compatibility, more VRAM
    # wins; within VRAM, NVIDIA wins (most mature runtime).
    vendor_score = {"nvidia": 3, "amd": 2, "intel": 1}
    return max(
        detected,
        key=lambda g: (
            int(g.is_compatible),
            g.vram_total_mb,
            vendor_score.get(g.vendor, 0),
        ),
    )

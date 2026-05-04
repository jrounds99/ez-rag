"""Cross-platform system info capture for the bench bundle.

Single entry point: `gather_system_info()` returns a SystemInfo
dataclass with everything the bench wants captured in the bundle:
OS, CPU, RAM, GPUs, disk speed, Ollama version, ez-rag git SHA.

Every probe is wrapped in try/except — failures degrade to "unknown"
rather than crashing the bench. This module never raises.

Cross-platform priority: Linux first (the cleanest probes via /proc
and /sys), Windows second (psutil + wmi where needed), macOS third
(sysctl + system_profiler). Apple Silicon is handled via psutil's
unified-memory reporting; we don't bench Apple Silicon as a primary
target but record the info correctly when present.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ============================================================================
# Public types
# ============================================================================

@dataclass
class GpuInfoBrief:
    """Per-GPU info captured for the bundle. A trimmed subset of
    gpu_detect.DetectedGpu — we don't need the full catalog match
    here; the bench links to gpu_detect output separately.
    """
    index: int
    vendor: str                       # "nvidia" | "amd" | "intel" | "unknown"
    name: str
    vram_total_mb: int
    vram_free_mb: Optional[int]
    driver_version: Optional[str]
    runtime: str                      # "cuda" | "rocm" | "xpu" | "cpu"
    detection_source: str


@dataclass
class SystemInfo:
    # Identity
    hostname: str
    short_system_id: str              # 12-hex-char stable ID per box
    os_name: str                      # "Windows 11" / "Ubuntu 24.04" / etc.
    os_version: str                   # full version string
    os_kernel: str                    # uname -r equivalent
    os_arch: str                      # "x86_64" / "arm64"
    timezone: str
    utc_timestamp: str

    # CPU
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_threads: int
    cpu_max_freq_mhz: int

    # RAM
    ram_total_gb: float
    ram_available_gb: float
    ram_speed_mhz: int                # 0 = unknown

    # GPUs
    gpus: list[GpuInfoBrief] = field(default_factory=list)

    # Disk
    docs_disk_seq_read_mbps: float = 0.0
    docs_disk_path: str = ""

    # Tool versions
    ollama_version: str = ""
    ollama_url: str = ""
    python_version: str = ""
    ezrag_git_sha: str = ""

    # Bench config
    bench_version: str = ""
    bench_run_id: str = ""

    # Notes — list of probe-failure messages for diagnostic transparency
    probe_warnings: list[str] = field(default_factory=list)


# ============================================================================
# Helpers
# ============================================================================

def _run(cmd: list[str], *, timeout: float = 5.0) -> Optional[str]:
    """Run a command, return stdout on success, None on any failure."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired,
            PermissionError, OSError):
        return None


def _stable_system_id() -> str:
    """A short ID that's the same every run on this box.

    Uses hostname + the first MAC we can find via uuid.getnode().
    Hashed so we don't leak the MAC. 12 hex chars is plenty to
    disambiguate the user's 3 machines.
    """
    try:
        host = socket.gethostname() or "unknown"
        mac = uuid.getnode()
        seed = f"{host}|{mac}".encode("utf-8")
        return hashlib.sha256(seed).hexdigest()[:12]
    except Exception:
        return "unknown00000"


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_timezone() -> str:
    try:
        return time.strftime("%Z") or "UTC"
    except Exception:
        return "UTC"


# ============================================================================
# OS probe
# ============================================================================

def _probe_os(info: SystemInfo) -> None:
    try:
        info.os_name = platform.system()
        info.os_version = platform.version()
        info.os_kernel = platform.release()
        info.os_arch = platform.machine()
    except Exception as ex:
        info.probe_warnings.append(f"os probe: {ex}")
    # Windows-specific richer name
    if info.os_name == "Windows":
        try:
            # platform.win32_ver returns (release, version, csd, ptype)
            release, ver, csd, _ = platform.win32_ver()
            if release:
                info.os_name = f"Windows {release}"
            if ver:
                info.os_version = ver
        except Exception:
            pass
    elif info.os_name == "Darwin":
        try:
            release = platform.mac_ver()[0]
            if release:
                info.os_name = f"macOS {release}"
                info.os_version = release
        except Exception:
            pass
    elif info.os_name == "Linux":
        # Read /etc/os-release for the friendly distro name
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                kv = {}
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        kv[k] = v.strip('"').strip("'")
            pretty = kv.get("PRETTY_NAME") or kv.get("NAME")
            if pretty:
                info.os_name = pretty
                info.os_version = kv.get("VERSION_ID", info.os_version)
        except OSError:
            pass


# ============================================================================
# CPU probe
# ============================================================================

def _probe_cpu(info: SystemInfo) -> None:
    info.cpu_model = "unknown"
    info.cpu_physical_cores = 0
    info.cpu_logical_threads = 0
    info.cpu_max_freq_mhz = 0

    # Try py-cpuinfo first — most accurate brand string cross-platform
    try:
        import cpuinfo  # type: ignore
        ci = cpuinfo.get_cpu_info()
        info.cpu_model = ci.get("brand_raw") or ci.get("brand", "unknown")
    except Exception:
        # Per-platform fallbacks
        if info.os_name and info.os_name.startswith("Windows"):
            out = _run([
                "powershell.exe", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ])
            if out:
                info.cpu_model = out.strip().splitlines()[0]
        elif info.os_name and info.os_name.startswith("macOS"):
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                info.cpu_model = out.strip()
        else:
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("model name"):
                            info.cpu_model = line.split(":", 1)[1].strip()
                            break
            except OSError:
                pass

    # Cores via psutil
    try:
        import psutil  # type: ignore
        info.cpu_physical_cores = (psutil.cpu_count(logical=False)
                                     or 0)
        info.cpu_logical_threads = (psutil.cpu_count(logical=True)
                                      or 0)
        try:
            freq = psutil.cpu_freq()
            if freq and freq.max:
                info.cpu_max_freq_mhz = int(freq.max)
        except Exception:
            pass
    except ImportError:
        info.cpu_logical_threads = os.cpu_count() or 0
        info.probe_warnings.append("psutil missing — CPU info partial")


# ============================================================================
# RAM probe
# ============================================================================

def _probe_ram(info: SystemInfo) -> None:
    info.ram_total_gb = 0.0
    info.ram_available_gb = 0.0
    info.ram_speed_mhz = 0
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info.ram_total_gb = round(vm.total / 1024**3, 2)
        info.ram_available_gb = round(vm.available / 1024**3, 2)
    except ImportError:
        info.probe_warnings.append("psutil missing — RAM totals skipped")

    # RAM speed — best-effort, often missing
    if info.os_name and info.os_name.startswith("Windows"):
        out = _run([
            "powershell.exe", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_PhysicalMemory).Speed | Select -First 1",
        ])
        if out:
            try:
                info.ram_speed_mhz = int(out.strip().splitlines()[0])
            except (ValueError, IndexError):
                pass
    elif info.os_name and info.os_name.startswith("macOS"):
        out = _run(["system_profiler", "SPMemoryDataType", "-json"],
                    timeout=10.0)
        if out:
            try:
                data = json.loads(out)
                items = data.get("SPMemoryDataType", [])
                for item in items:
                    speed = item.get("dimm_speed") or item.get("dimm_type")
                    if speed:
                        # Look like "4800 MT/s" — extract digits
                        digits = "".join(c for c in str(speed)
                                          if c.isdigit())
                        if digits:
                            info.ram_speed_mhz = int(digits)
                            break
            except Exception:
                pass


# ============================================================================
# GPU probe (delegates to gpu_detect, which already exists)
# ============================================================================

def _probe_gpus(info: SystemInfo) -> None:
    try:
        # Add src/ to sys.path so we can import the project module
        # without pip-installing it. This keeps the bench standalone.
        repo_root = Path(__file__).resolve().parents[1]
        src = repo_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from ez_rag.gpu_detect import detect_gpus  # type: ignore
        for g in detect_gpus():
            info.gpus.append(GpuInfoBrief(
                index=g.index, vendor=g.vendor, name=g.name,
                vram_total_mb=g.vram_total_mb,
                vram_free_mb=g.vram_free_mb,
                driver_version=g.driver_version,
                runtime=g.runtime,
                detection_source=g.detection_source,
            ))
    except Exception as ex:
        info.probe_warnings.append(f"gpu probe: {ex}")


# ============================================================================
# Disk speed probe — sequential read of a synthetic 100 MB file
# ============================================================================

def _probe_disk_speed(info: SystemInfo, *, target_dir: Optional[Path] = None
                      ) -> None:
    if target_dir is None:
        target_dir = Path.cwd()
    info.docs_disk_path = str(target_dir.resolve())
    test_file = target_dir / ".ezrag_disk_probe.bin"
    size_mb = 100
    try:
        # Write 100 MB of random-ish bytes (not so random that it
        # gets compressed by an FS feature; just a pseudo-random pattern).
        chunk = b"\x55\xaa" * (1024 * 512)   # 1 MB
        with open(test_file, "wb") as f:
            for _ in range(size_mb):
                f.write(chunk)
        # Force OS to drop the page cache where possible. If we can't,
        # the read will be cached and the result optimistic — still
        # informative as a relative number across machines.
        try:
            if hasattr(os, "fsync"):
                with open(test_file, "rb") as f:
                    os.fsync(f.fileno())
        except OSError:
            pass

        t0 = time.perf_counter()
        bytes_read = 0
        with open(test_file, "rb") as f:
            while True:
                buf = f.read(1024 * 1024)
                if not buf:
                    break
                bytes_read += len(buf)
        dt = time.perf_counter() - t0
        if dt > 0:
            info.docs_disk_seq_read_mbps = round(
                (bytes_read / 1024 / 1024) / dt, 1
            )
    except Exception as ex:
        info.probe_warnings.append(f"disk probe: {ex}")
    finally:
        try:
            test_file.unlink()
        except OSError:
            pass


# ============================================================================
# Ollama version probe
# ============================================================================

def _probe_ollama(info: SystemInfo,
                   *, url: str = "http://127.0.0.1:11434") -> None:
    info.ollama_url = url
    try:
        import httpx  # type: ignore
        r = httpx.get(url.rstrip("/") + "/api/version", timeout=3.0)
        if r.status_code == 200:
            info.ollama_version = str(r.json().get("version", "")) or "?"
            return
    except Exception:
        pass
    # Fallback: try the CLI
    out = _run(["ollama", "--version"], timeout=3.0)
    if out:
        # "ollama version is 0.4.7" format
        parts = out.strip().split()
        if parts:
            info.ollama_version = parts[-1]
        return
    info.ollama_version = "unreachable"


# ============================================================================
# Git SHA probe
# ============================================================================

def _probe_git_sha(info: SystemInfo) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sha = _run(["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"])
    if sha:
        info.ezrag_git_sha = sha.strip()
    else:
        info.ezrag_git_sha = "no-git"


# ============================================================================
# Public API
# ============================================================================

def gather_system_info(*,
                       bench_version: str = "0.1.0",
                       bench_run_id: Optional[str] = None,
                       docs_dir: Optional[Path] = None,
                       ollama_url: str = "http://127.0.0.1:11434",
                       skip_disk_probe: bool = False,
                       ) -> SystemInfo:
    """Capture the full system info record. Never raises."""
    info = SystemInfo(
        hostname=socket.gethostname() or "unknown",
        short_system_id=_stable_system_id(),
        os_name="", os_version="", os_kernel="", os_arch="",
        timezone=_safe_timezone(),
        utc_timestamp=_now_utc_iso(),
        cpu_model="", cpu_physical_cores=0,
        cpu_logical_threads=0, cpu_max_freq_mhz=0,
        ram_total_gb=0.0, ram_available_gb=0.0, ram_speed_mhz=0,
        python_version=sys.version.split()[0],
        bench_version=bench_version,
        bench_run_id=(bench_run_id or _now_utc_iso().replace(":", "-")),
    )
    _probe_os(info)
    _probe_cpu(info)
    _probe_ram(info)
    _probe_gpus(info)
    if not skip_disk_probe:
        _probe_disk_speed(info, target_dir=docs_dir)
    _probe_ollama(info, url=ollama_url)
    _probe_git_sha(info)
    return info


def to_dict(info: SystemInfo) -> dict:
    """JSON-serializable dict (lists of dataclasses get unrolled)."""
    d = asdict(info)
    # gpus are already converted to dicts by asdict()
    return d


def write_to_file(info: SystemInfo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(info), indent=2),
                     encoding="utf-8")


# ============================================================================
# CLI smoke test
# ============================================================================

def main():
    """`python -m bench.sysinfo` prints the probe result."""
    info = gather_system_info()
    print(json.dumps(to_dict(info), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

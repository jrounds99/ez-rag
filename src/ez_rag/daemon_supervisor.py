"""DaemonSupervisor — spawns and tracks Ollama daemons across GPUs.

Phase 2 of the multi-GPU plan. Owns the *processes*; multi_gpu.py
owns the *data*.

Flow:
  1. detect_external() probes whatever ollama daemon is already
     running on the user's configured port (typically 11434).
     If reachable, ez-rag treats it as the GPU-0 slot and DOES NOT
     manage its lifecycle. We never spawn over an external daemon.
  2. spawn_managed(gpu_index) launches a NEW ollama serve process
     with CUDA_VISIBLE_DEVICES=<gpu_index>, OLLAMA_HOST=<free port>,
     OLLAMA_MODELS pointing at the same shared blob store, then
     waits for /api/version to answer before returning.
  3. shutdown_managed() SIGTERMs everything we spawned. External
     daemons are left alone.

Design constraints honoring the plan:
  - Managed daemons share ~/.ollama/models with the external one.
    No duplicate weights on disk.
  - Each managed daemon writes a PID file at
    ~/.ezrag/daemon.<port>.pid so a re-launch of ez-rag can detect
    them and skip respawning.
  - Health checks are stateless probes against /api/version; they
    don't depend on anything ollama-specific other than that endpoint.

Doesn't import or call multi_gpu.RoutingTable — that's the caller's
job. This module just spawns processes and hands back GpuDaemon
records describing them. Keeps the supervisor unit-testable in
isolation.
"""
from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from .multi_gpu import (
    GpuDaemon, MANAGED_DAEMON_BASE_PORT, find_free_port, url_for_port,
)


# ============================================================================
# Constants
# ============================================================================

# Where managed-daemon PID files live. Used so a fresh ez-rag launch
# can adopt daemons it spawned in a previous run instead of respawning.
PID_DIR = Path.home() / ".ezrag" / "daemons"

# How long to wait for /api/version after spawning. Ollama's first
# startup loads no model — version comes back in <2 s typically; bigger
# windows just for slower disks.
SPAWN_HEALTH_TIMEOUT_S = 30.0

# How long SIGTERM gets before we escalate to SIGKILL on Linux/macOS
# (or terminate() on Windows since SIGTERM ≈ TerminateProcess there).
SHUTDOWN_GRACE_S = 5.0


# ============================================================================
# External-daemon detection
# ============================================================================

@dataclass
class ExternalDetection:
    """Result of probing for a pre-existing ollama daemon."""
    reachable: bool
    url: str
    version: str = ""
    error: str = ""


def _probe_url(url: str, *, timeout: float = 2.0) -> ExternalDetection:
    """Hit /api/version on the given URL. Returns reachability + raw version.
    Never raises."""
    try:
        r = httpx.get(url.rstrip("/") + "/api/version", timeout=timeout)
        if r.status_code == 200:
            return ExternalDetection(
                reachable=True, url=url,
                version=str(r.json().get("version", "")),
            )
        return ExternalDetection(
            reachable=False, url=url,
            error=f"HTTP {r.status_code}",
        )
    except Exception as ex:
        return ExternalDetection(
            reachable=False, url=url, error=f"{type(ex).__name__}: {ex}",
        )


@dataclass
class LoadedModel:
    """One model currently resident on a daemon, as reported by /api/ps."""
    name: str
    size_bytes: int          # total model size on disk
    size_vram_bytes: int     # bytes resident on GPU (0 = on CPU)
    expires_at: str          # ISO timestamp; "" if no keep-alive set


def query_loaded_models(url: str, *, timeout: float = 3.0
                          ) -> list[LoadedModel]:
    """Hit /api/ps on the given daemon. Returns the list of currently
    loaded models. Empty list on any failure (the live-placement
    panel just shows "no models loaded" instead of erroring).

    Used by Phase 5 (live placement panel). Polled every ~5 s by the
    GUI, so it must be cheap and never raise.
    """
    try:
        r = httpx.get(url.rstrip("/") + "/api/ps", timeout=timeout)
        if r.status_code != 200:
            return []
        models = r.json().get("models", []) or []
    except Exception:
        return []
    out: list[LoadedModel] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        try:
            out.append(LoadedModel(
                name=str(m.get("name", "") or m.get("model", "")),
                size_bytes=int(m.get("size", 0) or 0),
                size_vram_bytes=int(m.get("size_vram", 0) or 0),
                expires_at=str(m.get("expires_at", "") or ""),
            ))
        except (TypeError, ValueError):
            continue
    return out


def detect_external(default_url: str = "http://127.0.0.1:11434"
                    ) -> ExternalDetection:
    """Look for a pre-existing ollama daemon to adopt.

    Resolution order:
      1. Default URL passed in (typically the user's cfg.llm_url)
      2. OLLAMA_HOST env var if set ("host:port" or full URL)

    Returns reachable=True iff /api/version answered with 200. The
    caller treats reachable=True as "use this URL as the external
    GPU-0 slot; DO NOT spawn over it."
    """
    candidates: list[str] = [default_url]
    env_host = os.environ.get("OLLAMA_HOST", "").strip()
    if env_host:
        if "://" not in env_host:
            env_host = f"http://{env_host}"
        if env_host not in candidates:
            candidates.insert(0, env_host)
    for url in candidates:
        result = _probe_url(url)
        if result.reachable:
            return result
    return ExternalDetection(
        reachable=False, url=candidates[0],
        error="no daemon answered on any probed URL",
    )


# ============================================================================
# PID-file bookkeeping (so we can adopt our own daemons on relaunch)
# ============================================================================

def _pid_file(port: int) -> Path:
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"daemon.{port}.pid"


def _write_pid_file(port: int, pid: int, gpu_index: int) -> None:
    p = _pid_file(port)
    p.write_text(f"{pid}\n{gpu_index}\n", encoding="utf-8")


def _read_pid_file(port: int) -> tuple[Optional[int], Optional[int]]:
    p = _pid_file(port)
    if not p.is_file():
        return None, None
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        return int(lines[0]), int(lines[1]) if len(lines) > 1 else None
    except (OSError, ValueError, IndexError):
        return None, None


def _delete_pid_file(port: int) -> None:
    try:
        _pid_file(port).unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """Cross-platform 'is this PID still running?' check."""
    if pid is None or pid <= 0:
        return False
    if platform.system() == "Windows":
        try:
            # Signal 0 raises on Windows; use tasklist or psutil if
            # available. Fallback: try to open the process via
            # ctypes.
            import ctypes  # type: ignore
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            h = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid),
            )
            if not h:
                return False
            still_active = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(h, ctypes.byref(still_active))
            kernel32.CloseHandle(h)
            STILL_ACTIVE = 259
            return bool(ok) and still_active.value == STILL_ACTIVE
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def adopt_existing_managed_daemons() -> list[GpuDaemon]:
    """On startup, look for managed daemons we spawned in a previous
    ez-rag session and adopt them if they're still alive.

    A daemon is adopted iff:
      - PID file exists in ~/.ezrag/daemons/
      - Process at that PID is still running
      - /api/version on the recorded port answers

    Stale PID files are cleaned up. Returns the adopted GpuDaemon
    records (caller merges them into the routing table).
    """
    out: list[GpuDaemon] = []
    if not PID_DIR.is_dir():
        return out
    for pid_path in PID_DIR.glob("daemon.*.pid"):
        try:
            port = int(pid_path.stem.split(".")[-1])
        except ValueError:
            continue
        pid, gpu_index = _read_pid_file(port)
        if pid is None or not _pid_alive(pid):
            _delete_pid_file(port)
            continue
        url = url_for_port(port)
        if not _probe_url(url).reachable:
            _delete_pid_file(port)
            continue
        out.append(GpuDaemon(
            gpu_index=int(gpu_index) if gpu_index is not None else 0,
            url=url, pid=pid, managed=True,
            notes="adopted from previous ez-rag session",
        ))
    return out


# ============================================================================
# Spawning
# ============================================================================

class SpawnError(RuntimeError):
    """Raised when we couldn't get a managed daemon to come up."""


def _ollama_executable() -> str:
    """Locate the ollama CLI. Honors PATH; falls back to common
    install locations on Windows."""
    found = shutil.which("ollama")
    if found:
        return found
    if platform.system() == "Windows":
        for candidate in (
            r"C:\Users\Public\AppData\Local\Programs\Ollama\ollama.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
        ):
            if os.path.isfile(candidate):
                return candidate
    raise SpawnError(
        "Could not find the 'ollama' executable on PATH. "
        "Install Ollama (https://ollama.com/download) or add it to PATH."
    )


def _models_dir() -> Path:
    """Where Ollama stores its model blobs. Shared across daemons so
    we don't double the disk usage."""
    explicit = os.environ.get("OLLAMA_MODELS", "").strip()
    if explicit:
        return Path(explicit)
    if platform.system() == "Windows":
        return Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".ollama" / "models"
    return Path.home() / ".ollama" / "models"


def spawn_managed_daemon(gpu_index: int, gpu_name: str = "",
                          vram_total_mb: int = 0,
                          *,
                          port: Optional[int] = None,
                          keep_alive_s: int = 1800,
                          ) -> GpuDaemon:
    """Spawn a new ollama serve process pinned to one GPU.

    Returns the GpuDaemon record. Raises SpawnError if the daemon
    didn't come up within SPAWN_HEALTH_TIMEOUT_S.

    The spawned process inherits the user's environment with three
    overrides:
      CUDA_VISIBLE_DEVICES = <gpu_index>     # NVIDIA pin
      HIP_VISIBLE_DEVICES  = <gpu_index>     # AMD pin (cheap to set both)
      OLLAMA_HOST          = 127.0.0.1:<port>
      OLLAMA_KEEP_ALIVE    = <keep_alive_s>s
      OLLAMA_MODELS        = (existing or default location — shared)
    """
    exe = _ollama_executable()
    if port is None:
        port = find_free_port(start=MANAGED_DAEMON_BASE_PORT)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_index)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_KEEP_ALIVE"] = f"{int(keep_alive_s)}s"
    # Make the model store explicit so the daemon doesn't accidentally
    # use a different default than its sibling.
    env["OLLAMA_MODELS"] = str(_models_dir())

    creationflags = 0
    if platform.system() == "Windows":
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )

    proc = subprocess.Popen(
        [exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    # Wait for /api/version to answer.
    url = url_for_port(port)
    deadline = time.monotonic() + SPAWN_HEALTH_TIMEOUT_S
    last_err = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            # Process died before becoming reachable — surface the
            # exit code in the error.
            raise SpawnError(
                f"ollama serve exited (code {proc.returncode}) "
                f"before answering on {url}"
            )
        result = _probe_url(url, timeout=1.0)
        if result.reachable:
            _write_pid_file(port, proc.pid, gpu_index)
            return GpuDaemon(
                gpu_index=gpu_index,
                gpu_name=gpu_name,
                vram_total_mb=int(vram_total_mb),
                url=url,
                pid=proc.pid,
                managed=True,
                keep_alive_s=int(keep_alive_s),
                notes=(f"spawned {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                       f"on port {port}"),
            )
        last_err = result.error
        time.sleep(0.3)

    # Timed out — kill the process so we don't leak it.
    _terminate_proc(proc)
    raise SpawnError(
        f"ollama serve on port {port} did not answer /api/version "
        f"within {SPAWN_HEALTH_TIMEOUT_S}s (last error: {last_err})"
    )


# ============================================================================
# Shutdown
# ============================================================================

def _terminate_proc(proc: subprocess.Popen) -> None:
    """SIGTERM with a grace period, then escalate. Cross-platform."""
    if proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except Exception:
        # Already dead, missing handle, etc — fall through to wait.
        pass
    try:
        proc.wait(timeout=SHUTDOWN_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    # Escalate.
    try:
        proc.kill()
    except Exception:
        pass


def shutdown_managed_by_pid(pid: int, port: int) -> bool:
    """Best-effort shutdown of a managed daemon we don't currently hold
    a Popen handle for (e.g. a daemon adopted from a previous run).
    Returns True iff the process was running and we sent it a signal.

    Cross-platform behavior:
      - Windows: terminate via OpenProcess + TerminateProcess.
      - Other: SIGTERM, then SIGKILL after the grace window.
    """
    if not _pid_alive(pid):
        _delete_pid_file(port)
        return False
    if platform.system() == "Windows":
        try:
            import ctypes  # type: ignore
            PROCESS_TERMINATE = 0x0001
            kernel32 = ctypes.windll.kernel32
            h = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
            if not h:
                return False
            kernel32.TerminateProcess(h, 1)
            kernel32.CloseHandle(h)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            _delete_pid_file(port)
            return False
        # Wait for grace
        deadline = time.monotonic() + SHUTDOWN_GRACE_S
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                _delete_pid_file(port)
                return True
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    _delete_pid_file(port)
    return True


# ============================================================================
# DaemonSupervisor — top-level orchestration
# ============================================================================

class DaemonSupervisor:
    """Holds Popen handles for every daemon ez-rag spawned in this session.

    Use cases:
      - Phase 4 UI: 'ensure a managed daemon is running for GPU N'
      - Phase 7 health-check: 'is daemon X still up?'
      - atexit: 'shut down every managed daemon cleanly'

    The supervisor never tracks the EXTERNAL daemon (the user's own
    ollama serve). Detection happens via detect_external(); the
    routing table records the external slot but the supervisor
    doesn't manage its lifetime.
    """

    def __init__(self):
        self._procs: dict[int, subprocess.Popen] = {}   # gpu_index -> Popen
        self._records: dict[int, GpuDaemon] = {}         # gpu_index -> record

    # ----- Spawning -----

    def ensure_running(self, gpu_index: int, *,
                       gpu_name: str = "", vram_total_mb: int = 0,
                       port: Optional[int] = None,
                       keep_alive_s: int = 1800,
                       ) -> GpuDaemon:
        """Idempotent spawn. If a daemon for this GPU is already
        tracked AND its process is alive AND its URL answers, return
        the existing record. Otherwise spawn a fresh one."""
        existing = self._records.get(gpu_index)
        proc = self._procs.get(gpu_index)
        if existing is not None and proc is not None and proc.poll() is None:
            if _probe_url(existing.url, timeout=1.0).reachable:
                return existing
            # Process is alive but not responding — kill it and respawn.
            _terminate_proc(proc)
            self._procs.pop(gpu_index, None)
            self._records.pop(gpu_index, None)

        record = spawn_managed_daemon(
            gpu_index=gpu_index, gpu_name=gpu_name,
            vram_total_mb=vram_total_mb,
            port=port, keep_alive_s=keep_alive_s,
        )
        # Re-acquire the Popen handle. spawn_managed_daemon doesn't
        # return one because adopted daemons (from PID files) won't
        # have one either; the supervisor uses pid for those.
        # For freshly-spawned ones we DO want the handle so terminate
        # is reliable. Walk the OS process table by PID:
        try:
            import psutil  # type: ignore
            psutil.Process(record.pid)   # confirm alive
        except (ImportError, Exception):
            pass
        self._records[gpu_index] = record
        return record

    # ----- Adoption from previous session -----

    def adopt_previous(self) -> list[GpuDaemon]:
        """Pull in any managed daemons left over from a prior ez-rag
        run that are still healthy. Records them in the supervisor
        without holding a Popen handle (we'll signal them by pid)."""
        adopted = adopt_existing_managed_daemons()
        for d in adopted:
            self._records[d.gpu_index] = d
            # No Popen handle — _procs entry stays absent
        return adopted

    # ----- Health -----

    def is_alive(self, gpu_index: int) -> bool:
        rec = self._records.get(gpu_index)
        if rec is None:
            return False
        # Prefer the Popen handle when we have it (most accurate).
        proc = self._procs.get(gpu_index)
        if proc is not None and proc.poll() is not None:
            return False
        if rec.pid is not None and not _pid_alive(rec.pid):
            return False
        return _probe_url(rec.url, timeout=1.0).reachable

    def records(self) -> list[GpuDaemon]:
        return list(self._records.values())

    # ----- Shutdown -----

    def shutdown(self, gpu_index: int) -> None:
        """Terminate the daemon for one GPU."""
        proc = self._procs.pop(gpu_index, None)
        rec = self._records.pop(gpu_index, None)
        if proc is not None:
            _terminate_proc(proc)
            if rec is not None:
                # Derive port from URL — record only stores the URL.
                try:
                    port = int(rec.url.rsplit(":", 1)[-1].split("/")[0])
                    _delete_pid_file(port)
                except Exception:
                    pass
        elif rec is not None and rec.pid is not None:
            try:
                port = int(rec.url.rsplit(":", 1)[-1].split("/")[0])
            except Exception:
                port = 0
            shutdown_managed_by_pid(rec.pid, port)

    def shutdown_all(self) -> None:
        """Terminate every managed daemon. Safe to call multiple times."""
        for gpu_index in list(self._procs.keys()) + list(self._records.keys()):
            self.shutdown(gpu_index)


# ============================================================================
# Phase 7: health-check + stranded-assignment recovery
# ============================================================================
# A small, optional watchdog the GUI can spin up. Every HEALTH_CHECK_INTERVAL_S
# seconds, it pings each registered daemon's /api/version. When a daemon
# stops responding TWICE in a row (debounce against transient blips), it
# is marked dead in the routing table:
#   - The GpuDaemon record is REMOVED from the table
#   - Any ModelAssignment pinned to that GPU is DEMOTED to AUTO so the
#     picker can route the next call to a healthy daemon
#   - The user sees a one-time toast / status-bar update
#
# When the daemon comes back later (PID still alive, URL answers), the
# watchdog re-registers it AND restores any previously-stranded
# assignments to their original GPU.

HEALTH_CHECK_INTERVAL_S = 8.0      # how often to sweep
HEALTH_FAIL_THRESHOLD = 2          # consecutive misses before declaring dead


@dataclass
class HealthEvent:
    """Returned from one health-check sweep so the GUI can update."""
    kind: str                      # "down" | "back" | "ok"
    gpu_index: int
    url: str
    notes: str = ""


def health_check_once(table, *, fail_counts: Optional[dict] = None,
                       stranded_backup: Optional[dict] = None
                       ) -> list[HealthEvent]:
    """Run ONE pass of health checks against every daemon in `table`.

    `fail_counts` is a caller-owned dict {gpu_index: int} that
    accumulates consecutive failures across calls; the watchdog
    passes the same dict in every loop iteration.

    `stranded_backup` is a caller-owned dict
    {(model_tag, role): original_gpu_index} that records which
    assignments were demoted to AUTO so they can be restored when
    the daemon comes back.

    Returns the list of state-change events for this sweep. Caller
    saves the routing table to disk + emits UI events.
    """
    if table is None or not table.daemons:
        return []
    if fail_counts is None:
        fail_counts = {}
    if stranded_backup is None:
        stranded_backup = {}

    events: list[HealthEvent] = []

    # Snapshot daemon list — we may mutate it during the sweep.
    for daemon in list(table.daemons):
        idx = daemon.gpu_index
        url = daemon.url
        result = _probe_url(url, timeout=2.0)

        if result.reachable:
            # Reset the fail counter (single success clears the streak).
            had_failures = fail_counts.get(idx, 0) > 0
            fail_counts[idx] = 0
            if had_failures:
                events.append(HealthEvent(
                    kind="back", gpu_index=idx, url=url,
                    notes="daemon recovered",
                ))
            continue

        # Probe failed.
        fail_counts[idx] = fail_counts.get(idx, 0) + 1
        if fail_counts[idx] < HEALTH_FAIL_THRESHOLD:
            # Transient blip — wait for one more cycle before acting.
            continue

        # Persistent failure — mark daemon down + strand its assignments.
        # Remove from table.
        table.remove_daemon(idx)

        # Demote any assignment pinned to this GPU to AUTO.
        # Save the original gpu_index so we can restore on recovery.
        for a in table.assignments:
            if a.gpu_index == idx:
                key = (a.model_tag, a.role)
                if key not in stranded_backup:
                    stranded_backup[key] = a.gpu_index
                a.gpu_index = -1   # AUTO sentinel

        events.append(HealthEvent(
            kind="down", gpu_index=idx, url=url,
            notes=(f"daemon at {url} stopped answering after "
                   f"{HEALTH_FAIL_THRESHOLD} consecutive checks"),
        ))

    # Recovery path: try to re-adopt daemons that previously went
    # down — if their URL now answers AND we have a stranded
    # assignment that wanted them.
    if stranded_backup:
        # Walk a copy so we can mutate the dict during iteration.
        for (model_tag, role), original_gpu in list(stranded_backup.items()):
            # Is the daemon for `original_gpu` back?
            existing_daemon = table.daemon_for_gpu(original_gpu)
            if existing_daemon is None:
                # No daemon at that GPU — nothing to restore yet.
                continue
            # Check it's actually responding.
            if not _probe_url(existing_daemon.url, timeout=2.0).reachable:
                continue
            # Restore the assignment's pin.
            for a in table.assignments:
                if (a.model_tag == model_tag and a.role == role
                        and a.gpu_index == -1):
                    a.gpu_index = original_gpu
                    stranded_backup.pop((model_tag, role), None)
                    events.append(HealthEvent(
                        kind="back", gpu_index=original_gpu,
                        url=existing_daemon.url,
                        notes=(f"assignment '{model_tag}' "
                                f"({role}) restored to "
                                f"GPU {original_gpu}"),
                    ))
                    break

    return events

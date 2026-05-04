"""Tests for daemon_supervisor — daemon detection, adoption, shutdown.

Doesn't actually spawn ollama serve (that's an integration test).
Stubs:
  - httpx.get for /api/version probes
  - PID-file IO via a temp PID_DIR
  - subprocess.Popen for spawn paths is touched at module level only
    (we test the bookkeeping, not the spawn)
  - os.kill / Windows ctypes for liveness — we run with synthetic
    PIDs that we know don't exist, plus our own PID which we know
    does exist
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import daemon_supervisor as ds
from ez_rag.daemon_supervisor import (
    DaemonSupervisor, ExternalDetection, _delete_pid_file, _pid_alive,
    _read_pid_file, _write_pid_file, adopt_existing_managed_daemons,
    detect_external,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


# ============================================================================
# Helpers
# ============================================================================

class StubProbeManager:
    """Patch _probe_url to return a canned ExternalDetection."""
    def __init__(self, mapping: dict):
        # mapping: url -> ExternalDetection
        self.mapping = mapping
        self.saved = ds._probe_url
        self.calls: list[str] = []

    def __enter__(self):
        def stub(url, *, timeout=2.0):
            self.calls.append(url)
            return self.mapping.get(url, ExternalDetection(
                reachable=False, url=url, error="stub: not in mapping",
            ))
        ds._probe_url = stub
        return self

    def __exit__(self, *args):
        ds._probe_url = self.saved


def stub_pid_dir(tmp_path: Path):
    """Point PID_DIR at a temp dir for the duration of the test."""
    saved = ds.PID_DIR
    ds.PID_DIR = tmp_path
    return saved


def restore_pid_dir(saved):
    ds.PID_DIR = saved


# ============================================================================
# Tests
# ============================================================================

def main():
    print("\n[1] detect_external — happy path")
    with StubProbeManager({
        "http://127.0.0.1:11434": ExternalDetection(
            reachable=True, url="http://127.0.0.1:11434", version="0.4.0",
        ),
    }) as sm:
        result = detect_external("http://127.0.0.1:11434")
        check("reachable=True", result.reachable)
        check("version captured", result.version == "0.4.0")
        check("only the default URL was probed",
              sm.calls == ["http://127.0.0.1:11434"])

    print("\n[2] detect_external — OLLAMA_HOST env override")
    saved_env = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = "host.lan:11500"
    try:
        with StubProbeManager({
            "http://host.lan:11500": ExternalDetection(
                reachable=True, url="http://host.lan:11500", version="0.4.1",
            ),
            "http://127.0.0.1:11434": ExternalDetection(
                reachable=False, url="http://127.0.0.1:11434",
            ),
        }) as sm:
            result = detect_external("http://127.0.0.1:11434")
            check("env-host probed FIRST",
                  sm.calls[0] == "http://host.lan:11500",
                  f"call order: {sm.calls}")
            check("env-host adopted",
                  result.reachable
                   and result.url == "http://host.lan:11500")
    finally:
        if saved_env is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = saved_env

    print("\n[3] detect_external — nothing reachable")
    with StubProbeManager({}):
        result = detect_external("http://127.0.0.1:11434")
        check("reachable=False on cold start",
              not result.reachable)
        check("error text populated",
              "no daemon" in (result.error or "").lower(),
              f"err: {result.error!r}")

    print("\n[4] _probe_url itself with bad URL (real httpx)")
    real = ds._probe_url("http://127.0.0.1:1", timeout=1.0)
    check("unreachable port -> reachable=False",
          not real.reachable)
    check("error captured (no crash)",
          bool(real.error))

    print("\n[5] PID-file write/read/delete")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            _write_pid_file(11435, 12345, gpu_index=1)
            pid, gpu = _read_pid_file(11435)
            check("pid round-trips", pid == 12345)
            check("gpu_index round-trips", gpu == 1)
            _delete_pid_file(11435)
            pid2, _ = _read_pid_file(11435)
            check("delete clears the file",
                  pid2 is None)
            # Reading a nonexistent file is safe
            pid3, _ = _read_pid_file(99999)
            check("missing pid file returns None",
                  pid3 is None)
        finally:
            restore_pid_dir(saved)

    print("\n[6] _pid_alive — own PID is alive, fake high PID isn't")
    own = os.getpid()
    check("our own PID reads as alive",
          _pid_alive(own))
    # Pick a PID that's almost certainly not in use. On Linux PID 1 IS
    # used (init), but a very high number isn't. Windows handles this
    # safely too.
    check("fake high PID reads as dead",
          not _pid_alive(7_777_777))
    check("PID 0 is not alive",
          not _pid_alive(0))
    check("negative PID is not alive",
          not _pid_alive(-1))

    print("\n[7] adopt_existing_managed_daemons — happy path")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            # Write a PID file that points at OUR running process
            # (so _pid_alive returns True), and stub the URL probe so
            # the daemon at port 11435 is "reachable."
            _write_pid_file(11435, os.getpid(), gpu_index=1)
            with StubProbeManager({
                "http://127.0.0.1:11435": ExternalDetection(
                    reachable=True, url="http://127.0.0.1:11435",
                    version="0.4.0",
                ),
            }):
                adopted = adopt_existing_managed_daemons()
                check("one daemon adopted",
                      len(adopted) == 1)
                d = adopted[0]
                check("adopted gpu_index from PID file",
                      d.gpu_index == 1)
                check("adopted PID matches",
                      d.pid == os.getpid())
                check("adopted as managed",
                      d.managed is True)
                check("adopted URL has correct port",
                      d.url == "http://127.0.0.1:11435")
        finally:
            restore_pid_dir(saved)

    print("\n[8] adopt_existing_managed_daemons — stale PID is cleaned up")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            _write_pid_file(11436, 7_777_777, gpu_index=2)  # dead PID
            adopted = adopt_existing_managed_daemons()
            check("dead-PID daemon NOT adopted",
                  len(adopted) == 0)
            check("stale PID file was deleted",
                  not (Path(tmp) / "daemon.11436.pid").exists())
        finally:
            restore_pid_dir(saved)

    print("\n[9] adopt — alive PID but unreachable URL is cleaned up")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            _write_pid_file(11437, os.getpid(), gpu_index=3)
            with StubProbeManager({}):  # nothing reachable
                adopted = adopt_existing_managed_daemons()
                check("unreachable-URL daemon NOT adopted",
                      len(adopted) == 0)
                check("stale PID file was deleted",
                      not (Path(tmp) / "daemon.11437.pid").exists())
        finally:
            restore_pid_dir(saved)

    print("\n[10] DaemonSupervisor — adopt path + records()")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            _write_pid_file(11438, os.getpid(), gpu_index=0)
            _write_pid_file(11439, os.getpid(), gpu_index=1)
            with StubProbeManager({
                "http://127.0.0.1:11438": ExternalDetection(
                    reachable=True, url="http://127.0.0.1:11438",
                ),
                "http://127.0.0.1:11439": ExternalDetection(
                    reachable=True, url="http://127.0.0.1:11439",
                ),
            }):
                sup = DaemonSupervisor()
                adopted = sup.adopt_previous()
                check("supervisor adopted 2 daemons",
                      len(adopted) == 2)
                recs = sup.records()
                check("records() returns the same",
                      len(recs) == 2)
                check("records have managed=True",
                      all(r.managed for r in recs))
        finally:
            restore_pid_dir(saved)

    print("\n[11] supervisor.is_alive on adopted daemon")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            _write_pid_file(11440, os.getpid(), gpu_index=0)
            with StubProbeManager({
                "http://127.0.0.1:11440": ExternalDetection(
                    reachable=True, url="http://127.0.0.1:11440",
                ),
            }):
                sup = DaemonSupervisor()
                sup.adopt_previous()
                check("is_alive(0) True for healthy daemon",
                      sup.is_alive(0))
                check("is_alive(99) False for unknown GPU",
                      not sup.is_alive(99))
        finally:
            restore_pid_dir(saved)

    print("\n[12] supervisor.is_alive returns False when URL goes dark")
    with tempfile.TemporaryDirectory() as tmp:
        saved = stub_pid_dir(Path(tmp))
        try:
            _write_pid_file(11441, os.getpid(), gpu_index=0)
            with StubProbeManager({
                "http://127.0.0.1:11441": ExternalDetection(
                    reachable=True, url="http://127.0.0.1:11441",
                ),
            }):
                sup = DaemonSupervisor()
                sup.adopt_previous()
            # Now the probe stub is gone — _probe_url returns the real
            # version which won't reach a daemon at 11441.
            check("is_alive flips to False when URL stops answering",
                  not sup.is_alive(0))
        finally:
            restore_pid_dir(saved)

    print(f"\n=== daemon_supervisor summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

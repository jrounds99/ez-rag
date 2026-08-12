"""Pytest runner that subprocesses each manual test script.

The other `test_*.py` files in this directory each ship their own
harness and `if __name__ == "__main__":` runner — they exit non-zero
on failure. This module parametrizes over them so `pytest tests/`
runs the whole suite in CI with a clean PASS/FAIL summary, without
forcing us to rewrite 27 files.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_THIS_FILE = Path(__file__).name


def _discover() -> list[str]:
    return sorted(
        p.name
        for p in _TESTS_DIR.glob("test_*.py")
        if p.name != _THIS_FILE
    )


@pytest.mark.parametrize("script", _discover())
def test_manual_script(script: str) -> None:
    """Run the script with the project's Python; assert exit 0.

    Output goes to a temp FILE, not pipes: scripts that spawn servers
    (export tests) pass their stdout handle to grandchildren, and a
    piped `subprocess.run(timeout=...)` then wedges in communicate()
    forever after killing only the direct child. On timeout we
    `taskkill /T` (Windows) the whole process tree so no orphaned
    server survives to squat the next run's ports.
    """
    import os
    import tempfile
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                 errors="replace") as out:
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", str(_TESTS_DIR / script)],
            stdout=out, stderr=subprocess.STDOUT, env=env,
        )
        try:
            rc = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            proc.wait(timeout=30)
            out.seek(0)
            pytest.fail(f"{script} timed out after 300s\n----- output -----\n"
                        + out.read()[-4000:])
        if rc != 0:
            out.seek(0)
            pytest.fail(f"{script} exited {rc}\n----- output -----\n"
                        + (out.read().strip() or "(empty)"))


def _kill_tree(pid: int) -> None:
    """Kill a process AND its descendants (exported chatbot servers etc.)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=30)
    else:
        try:
            import signal
            import os
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass

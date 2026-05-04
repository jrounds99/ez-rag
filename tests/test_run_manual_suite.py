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

    Output is only shown when the test fails — pytest captures stdout
    on success.
    """
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(_TESTS_DIR / script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ,
              "PYTHONIOENCODING": "utf-8",
              "PYTHONUTF8": "1"},
    )
    if proc.returncode != 0:
        # Surface stdout+stderr so the failure is actionable.
        msg_parts = [
            f"{script} exited {proc.returncode}",
            "----- stdout -----",
            proc.stdout.strip() or "(empty)",
            "----- stderr -----",
            proc.stderr.strip() or "(empty)",
        ]
        pytest.fail("\n".join(msg_parts))

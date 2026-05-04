"""Pytest configuration for ez-rag.

The 27 `test_*.py` files in this directory are self-contained scripts
that use their own assertion harness (a small `check()` helper +
PASS/FAIL lists) and run via:

    python tests/test_<name>.py

They are NOT pytest-style test modules — pytest would try to collect
the bare `def test_foo(ws, cfg)` functions and fail on missing
fixtures. So we tell pytest to skip them at collection time and add
a single runner module (`test_run_manual_suite.py`) that wraps each
one as a subprocess. This keeps the two execution styles in sync:

    pytest tests/                  # CI / single-command run
    python tests/test_<name>.py    # direct dev iteration

To add a new manual-script test, drop it in `tests/`. To add a
pytest-style test, name the file `pytest_<name>.py` (or add a
`pytest:` opt-in marker — adjust the ignore list below).
"""
from __future__ import annotations

import os
from pathlib import Path

_THIS_DIR = Path(__file__).parent

# Auto-ignore every existing `test_*.py` script EXCEPT the runner that
# subprocesses them. Anything new that starts with `test_` is
# auto-ignored too — explicit pytest tests should use a different
# prefix.
_RUNNER = "test_run_manual_suite.py"
collect_ignore = [
    f for f in os.listdir(_THIS_DIR)
    if f.startswith("test_") and f.endswith(".py") and f != _RUNNER
]

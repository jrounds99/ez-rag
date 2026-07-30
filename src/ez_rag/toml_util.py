"""Shared TOML string rendering.

One canonical implementation of the "prefer single-quoted literal
strings" rule used by every hand-editable TOML file ez-rag writes
(routing tables, per-file metadata sidecars). Single-quoted literals
keep Windows paths and URLs with backslashes round-trippable; we fall
back to escaped basic strings only when the value contains a single
quote.
"""
from __future__ import annotations


def toml_str(value: str) -> str:
    """Render a string value for TOML output."""
    if "'" in value:
        # Escape only what TOML basic-string syntax requires: \ and "
        # (control chars are vanishingly rare in our values; keep the
        # output human-readable).
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return f"'{value}'"

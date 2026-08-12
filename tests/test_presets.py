"""Preset bundle tests.

Every preset must: reference only real Config fields, apply cleanly,
report an accurate change diff, be idempotent, and carry the
user-facing explanation text ("more info") with its benchmark numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.presets import PRESETS, apply_preset, get_preset

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    print("\n[1] structural validity")
    check("at least 3 presets", len(PRESETS) >= 3, f"{len(PRESETS)}")
    ids = [p.id for p in PRESETS]
    check("ids unique", len(ids) == len(set(ids)))
    ref = Config()
    for p in PRESETS:
        unknown = [k for k in p.settings if not hasattr(ref, k)]
        check(f"{p.id}: all fields exist on Config", not unknown,
              f"unknown: {unknown}")
        check(f"{p.id}: has details text", len(p.details) > 200)
        check(f"{p.id}: has tagline", bool(p.tagline))
        check(f"{p.id}: names its models", len(p.models_needed) >= 1)

    print("\n[2] apply + diff + idempotence")
    for p in PRESETS:
        cfg = Config()
        changed = apply_preset(cfg, p.id)
        for k, v in p.settings.items():
            if hasattr(cfg, k):
                check(f"{p.id}: {k} applied", getattr(cfg, k) == v,
                      f"{getattr(cfg, k)!r} != {v!r}")
        again = apply_preset(cfg, p.id)
        check(f"{p.id}: second apply is a no-op", again == [])
        check(f"{p.id}: diff only lists real changes",
              all(old != new for _, old, new in changed))

    print("\n[3] round-trip through config.toml")
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="ezrag_preset_"))
    cfg = Config()
    apply_preset(cfg, "deep-context")
    cfg.save(d / "config.toml")
    loaded = Config.load(d / "config.toml")
    p = get_preset("deep-context")
    ok = all(getattr(loaded, k) == v for k, v in p.settings.items()
             if hasattr(loaded, k))
    check("deep-context survives save/load", ok)

    print("\n[4] error handling")
    try:
        apply_preset(Config(), "nope")
        check("unknown preset raises", False)
    except ValueError as e:
        check("unknown preset raises", "Available:" in str(e))
    check("get_preset unknown -> None", get_preset("nope") is None)
    check("get_preset case-insensitive",
          get_preset("BALANCED") is not None)

    print(f"\n=== presets summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    for name, det in FAIL:
        print(f"  FAIL  {name} :: {det}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

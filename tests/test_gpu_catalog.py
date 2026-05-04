"""Tests for gpu_catalog — the static GPU reference data.

Fully deterministic, no I/O. Verifies the catalog is well-formed, the
fuzzy matcher resolves the test-matrix cards (RTX 5060 Mobile, 5090,
3090), and VRAM-disambiguation works for multi-VRAM SKUs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.gpu_catalog import (
    CATALOG, GpuSpec, find_spec, runtime_for_architecture,
    _autotier, _normalize,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def main():
    print("\n[1] catalog is well-formed")
    check("catalog non-empty", len(CATALOG) > 100,
          f"got {len(CATALOG)} entries")
    check("all entries are GpuSpec",
          all(isinstance(s, GpuSpec) for s in CATALOG))
    check("every entry has a non-empty name",
          all(s.name for s in CATALOG))
    check("every entry has at least one alias",
          all(len(s.aliases) >= 1 for s in CATALOG))
    check("vendor is one of {nvidia, amd, intel}",
          all(s.vendor in ("nvidia", "amd", "intel") for s in CATALOG))
    check("every entry has positive VRAM",
          all(s.vram_gb > 0 for s in CATALOG))
    check("every entry has runtime field",
          all(s.runtime in ("cuda", "rocm", "hip", "xpu")
               for s in CATALOG))
    check("every entry has a tier",
          all(s.tier in ("min", "comfortable", "ample",
                          "professional", "extreme")
               for s in CATALOG))

    print("\n[2] tier auto-derivation matches VRAM thresholds")
    check("4 GB -> min", _autotier(4) == "min")
    check("6 GB -> min", _autotier(6) == "min")
    check("8 GB -> comfortable", _autotier(8) == "comfortable")
    check("12 GB -> comfortable", _autotier(12) == "comfortable")
    check("16 GB -> ample", _autotier(16) == "ample")
    check("24 GB -> ample", _autotier(24) == "ample")
    check("32 GB -> professional", _autotier(32) == "professional")
    check("48 GB -> professional", _autotier(48) == "professional")
    check("64 GB -> extreme", _autotier(64) == "extreme")
    check("96 GB -> extreme", _autotier(96) == "extreme")
    check("192 GB -> extreme", _autotier(192) == "extreme")

    print("\n[3] _normalize strips vendor prefixes consistently")
    check("'NVIDIA GeForce RTX 5090' -> 'geforce rtx 5090'",
          _normalize("NVIDIA GeForce RTX 5090") == "geforce rtx 5090")
    check("'AMD Radeon RX 7900 XTX' -> 'radeon rx 7900 xtx'",
          _normalize("AMD Radeon RX 7900 XTX") == "radeon rx 7900 xtx")
    check("collapses multiple spaces",
          _normalize("Tesla   T4") == "tesla t4")
    check("empty string -> empty",
          _normalize("") == "")

    print("\n[4] user's test-matrix cards resolve")
    # The user's three test cards: 5060 Mobile (Linux), 5090 (Windows),
    # 3090 (Linux).
    spec = find_spec("NVIDIA GeForce RTX 5090")
    check("RTX 5090 found",
          spec is not None and "5090" in spec.name,
          f"got {spec!r}")
    check("RTX 5090 is 32 GB",
          spec is not None and spec.vram_gb == 32, f"got {spec!r}")
    check("RTX 5090 is Blackwell",
          spec is not None and spec.architecture == "blackwell",
          f"got {spec!r}")
    check("RTX 5090 is professional tier",
          spec is not None and spec.tier == "professional",
          f"got {spec!r}")

    spec = find_spec("NVIDIA GeForce RTX 3090")
    check("RTX 3090 found",
          spec is not None and "3090" in spec.name and spec.vram_gb == 24)
    check("RTX 3090 is Ampere", spec.architecture == "ampere")

    spec = find_spec("NVIDIA GeForce RTX 5060 Laptop GPU")
    check("RTX 5060 Mobile found", spec is not None)
    check("RTX 5060 Mobile has laptop=True",
          spec.laptop is True, f"got {spec!r}")
    check("RTX 5060 Mobile is 8 GB",
          spec.vram_gb == 8, f"got {spec.vram_gb}")
    check("RTX 5060 Mobile is Blackwell",
          spec.architecture == "blackwell")

    print("\n[5] VRAM disambiguation for multi-VRAM SKUs")
    # RTX A2000 has 6 GB and 12 GB variants — same name, different VRAM.
    spec_6 = find_spec("NVIDIA RTX A2000", vram_mb=6 * 1024)
    spec_12 = find_spec("NVIDIA RTX A2000 12GB", vram_mb=12 * 1024)
    check("A2000 6GB resolves to 6 GB entry",
          spec_6 is not None and spec_6.vram_gb == 6, f"got {spec_6!r}")
    check("A2000 12GB resolves to 12 GB entry",
          spec_12 is not None and spec_12.vram_gb == 12, f"got {spec_12!r}")

    spec_8 = find_spec("NVIDIA GeForce RTX 4060 Ti", vram_mb=8 * 1024)
    spec_16 = find_spec("NVIDIA GeForce RTX 4060 Ti", vram_mb=16 * 1024)
    check("4060 Ti 8GB resolves correctly",
          spec_8 is not None and spec_8.vram_gb == 8)
    check("4060 Ti 16GB resolves correctly",
          spec_16 is not None and spec_16.vram_gb == 16)

    spec_a770_8 = find_spec("Intel Arc A770 Graphics", vram_mb=8 * 1024)
    spec_a770_16 = find_spec("Intel Arc A770 Graphics", vram_mb=16 * 1024)
    check("Arc A770 8GB disambiguated",
          spec_a770_8 is not None and spec_a770_8.vram_gb == 8)
    check("Arc A770 16GB disambiguated",
          spec_a770_16 is not None and spec_a770_16.vram_gb == 16)

    print("\n[6] data-center cards present")
    check("L40S in catalog", find_spec("NVIDIA L40S") is not None)
    check("L40S is 48 GB",
          find_spec("NVIDIA L40S").vram_gb == 48)
    check("L40S marked data_center",
          find_spec("NVIDIA L40S").data_center is True)
    check("A100 80GB in catalog",
          find_spec("NVIDIA A100-PCIE-80GB") is not None)
    check("H100 SXM5 in catalog",
          find_spec("NVIDIA H100 80GB HBM3") is not None)
    check("H200 in catalog", find_spec("NVIDIA H200") is not None)
    check("H200 is 141 GB",
          find_spec("NVIDIA H200").vram_gb == 141)

    print("\n[7] RTX PRO Blackwell workstation cards")
    pro = find_spec("NVIDIA RTX PRO 6000 Blackwell")
    check("RTX PRO 6000 Blackwell found", pro is not None)
    check("RTX PRO 6000 Blackwell is 96 GB",
          pro is not None and pro.vram_gb == 96)
    check("RTX PRO 6000 Blackwell is extreme tier",
          pro is not None and pro.tier == "extreme")
    # Aliases for the WS edition variants
    pro_ws = find_spec(
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
    check("RTX PRO 6000 WS Edition resolves to same family",
          pro_ws is not None and pro_ws.vram_gb == 96)

    print("\n[8] AMD cards resolve")
    xtx = find_spec("AMD Radeon RX 7900 XTX")
    check("7900 XTX found", xtx is not None and xtx.vram_gb == 24)
    check("7900 XTX is RDNA3", xtx.architecture == "rdna3")
    mi300x = find_spec("AMD Instinct MI300X")
    check("MI300X found", mi300x is not None and mi300x.vram_gb == 192)

    print("\n[9] Intel cards resolve")
    a770 = find_spec("Intel Arc A770 Graphics", vram_mb=16 * 1024)
    check("Arc A770 found", a770 is not None and a770.vendor == "intel")
    b580 = find_spec("Intel Arc B580 Graphics")
    check("Arc B580 found", b580 is not None and b580.vram_gb == 12)

    print("\n[10] unknown names return None")
    check("nonsense name returns None",
          find_spec("Made-up GPU 9999") is None)
    check("empty string returns None",
          find_spec("") is None)
    check("None-equivalent returns None",
          find_spec(None or "") is None)

    print("\n[11] runtime_for_architecture mapping")
    check("ampere -> cuda",
          runtime_for_architecture("ampere") == "cuda")
    check("blackwell -> cuda",
          runtime_for_architecture("blackwell") == "cuda")
    check("rdna3 -> rocm",
          runtime_for_architecture("rdna3") == "rocm")
    check("cdna3 -> rocm",
          runtime_for_architecture("cdna3") == "rocm")
    check("xe-hpg -> xpu",
          runtime_for_architecture("xe-hpg") == "xpu")
    check("unknown arch -> 'unknown'",
          runtime_for_architecture("imaginary") == "unknown")

    print("\n[12] legacy/datacenter flags")
    p40 = find_spec("Tesla P40")
    check("P40 marked legacy", p40.legacy is True)
    check("P40 marked data_center", p40.data_center is True)
    check("P40 has runtime_notes",
          "Pascal" in p40.runtime_notes)

    print(f"\n=== Catalog summary: {len(PASS)} pass, {len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

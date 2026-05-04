"""Tests for gpu_detect — the cross-platform GPU enumeration layer.

Mocks the subprocess wrapper (`_run`) and `_which` so we exercise parsing
without real GPU hardware. Coverage:

  - nvidia-smi happy path on the user's three test cards
  - nvidia-smi missing → falls back to pynvml stub or empty
  - rocm-smi JSON parsing
  - WMI Windows fallback
  - dedup + ordering
  - graceful no-GPU result

Detection of the live machine's hardware is NOT tested here — that's
manual smoke testing on the user's actual three cards.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag import gpu_detect as gd


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _StubManager:
    """Patch `_run` / `_which` for a single test, restore afterward."""
    def __init__(self):
        self.saved_run = gd._run
        self.saved_which = gd._which
        self.saved_platform_system = None

    def install_run(self, fn):
        gd._run = fn

    def install_which(self, fn):
        gd._which = fn

    def install_platform_system(self, value):
        import platform as _p
        self.saved_platform_system = _p.system
        _p.system = lambda: value

    def restore(self):
        gd._run = self.saved_run
        gd._which = self.saved_which
        if self.saved_platform_system is not None:
            import platform as _p
            _p.system = self.saved_platform_system


def main():
    print("\n[1] nvidia-smi parses the user's test matrix")
    # Three rows: 5060 Mobile (Linux), 5090 (Windows), 3090 (Linux)
    smi_output = (
        "0, NVIDIA GeForce RTX 5090, 32760, 28000, 555.42, 12.0\n"
        "1, NVIDIA GeForce RTX 3090, 24576, 18000, 535.86, 8.6\n"
    )
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/nvidia-smi" if n == "nvidia-smi" else None)
        sm.install_run(lambda cmd, timeout=5.0: smi_output)
        gpus = gd._detect_nvidia_smi()
        check("two GPUs parsed", len(gpus) == 2,
              f"got {len(gpus)}")
        check("first is RTX 5090",
              "5090" in gpus[0].name)
        check("5090 VRAM correct",
              gpus[0].vram_total_mb == 32760)
        check("5090 free VRAM captured",
              gpus[0].vram_free_mb == 28000)
        check("5090 driver captured",
              gpus[0].driver_version == "555.42")
        check("5090 runtime is cuda",
              gpus[0].runtime == "cuda")
        check("5090 detection_source nvidia-smi",
              gpus[0].detection_source == "nvidia-smi")
        check("5090 matched_spec is Blackwell",
              gpus[0].matched_spec is not None
               and gpus[0].matched_spec.architecture == "blackwell")
        check("3090 matched_spec is Ampere",
              gpus[1].matched_spec is not None
               and gpus[1].matched_spec.architecture == "ampere")
        check("5090 is_compatible True",
              gpus[0].is_compatible)
        check("3090 is_compatible True",
              gpus[1].is_compatible)
    finally:
        sm.restore()

    print("\n[2] nvidia-smi parses laptop variant")
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/nvidia-smi" if n == "nvidia-smi" else None)
        sm.install_run(lambda cmd, timeout=5.0:
                       "0, NVIDIA GeForce RTX 5060 Laptop GPU, 8192, 7500, "
                       "555.42, 12.0\n")
        gpus = gd._detect_nvidia_smi()
        check("laptop card parsed", len(gpus) == 1)
        check("laptop matched_spec is laptop=True",
              gpus[0].matched_spec is not None
               and gpus[0].matched_spec.laptop is True)
        check("laptop has 8 GB",
              gpus[0].vram_total_mb == 8192)
    finally:
        sm.restore()

    print("\n[3] nvidia-smi missing -> _detect_nvidia_smi returns []")
    sm = _StubManager()
    try:
        sm.install_which(lambda n: None)
        gpus = gd._detect_nvidia_smi()
        check("no smi -> empty", gpus == [])
    finally:
        sm.restore()

    print("\n[4] nvidia-smi old driver fires a health warning")
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/nvidia-smi")
        sm.install_run(lambda cmd, timeout=5.0:
                       "0, NVIDIA GeForce RTX 3060, 12288, 11000, 470.86, 8.6\n")
        gpus = gd._detect_nvidia_smi()
        check("got one GPU", len(gpus) == 1)
        check("old driver flagged",
              any("Driver" in n and "535" in n for n in gpus[0].health_notes),
              f"notes={gpus[0].health_notes}")
    finally:
        sm.restore()

    print("\n[5] rocm-smi JSON parsing")
    rocm_output = (
        '{"card0": {"Card series": "AMD Radeon RX 7900 XTX", '
        '"VRAM Total Memory (B)": "25769803776", '
        '"VRAM Total Used Memory (B)": "1073741824", '
        '"Driver version": "6.2.4"}}'
    )
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/rocm-smi" if n == "rocm-smi" else None)
        sm.install_run(lambda cmd, timeout=5.0: rocm_output)
        gpus = gd._detect_rocm_smi()
        check("AMD GPU detected", len(gpus) == 1)
        check("AMD vendor",
              gpus[0].vendor == "amd")
        check("AMD VRAM ~24 GB",
              23 * 1024 < gpus[0].vram_total_mb < 25 * 1024,
              f"got {gpus[0].vram_total_mb}")
        check("AMD free VRAM computed (~23 GB)",
              gpus[0].vram_free_mb is not None
               and gpus[0].vram_free_mb > 22 * 1024)
        check("AMD runtime is rocm",
              gpus[0].runtime == "rocm")
        check("matched 7900 XTX",
              gpus[0].matched_spec is not None
               and "7900 XTX" in gpus[0].matched_spec.name)
    finally:
        sm.restore()

    print("\n[6] rocm-smi malformed output -> []")
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/rocm-smi")
        sm.install_run(lambda cmd, timeout=5.0: "not json at all {{{")
        gpus = gd._detect_rocm_smi()
        check("malformed -> empty", gpus == [])
    finally:
        sm.restore()

    print("\n[7] xpu-smi JSON parsing")
    xpu_output = (
        '{"device_list": [{"device_name": "Intel Arc A770 Graphics", '
        '"memory_physical_size_mib": 16384, '
        '"driver_version": "31.0.101.5333"}]}'
    )
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/xpu-smi" if n == "xpu-smi" else None)
        sm.install_run(lambda cmd, timeout=5.0: xpu_output)
        gpus = gd._detect_xpu_smi()
        check("Intel GPU detected", len(gpus) == 1)
        check("Intel vendor", gpus[0].vendor == "intel")
        check("Intel runtime is xpu", gpus[0].runtime == "xpu")
        check("Intel VRAM 16 GB",
              gpus[0].vram_total_mb == 16384)
    finally:
        sm.restore()

    print("\n[8] WMI fallback parses NVIDIA + AMD + Intel")
    wmi_output = (
        '['
        '{"Name": "NVIDIA GeForce RTX 4090", "AdapterRAM": 4294967295, '
        '"DriverVersion": "555.42"},'
        '{"Name": "AMD Radeon RX 7900 XTX", "AdapterRAM": 4294967295, '
        '"DriverVersion": "23.40.27.06"},'
        '{"Name": "Intel(R) UHD Graphics 770", "AdapterRAM": 1073741824, '
        '"DriverVersion": "31.0.101.5333"},'
        '{"Name": "Microsoft Basic Display Adapter", "AdapterRAM": 0, '
        '"DriverVersion": ""}'
        ']'
    )
    sm = _StubManager()
    try:
        sm.install_platform_system("Windows")
        sm.install_run(lambda cmd, timeout=5.0: wmi_output)
        gpus = gd._detect_wmi()
        check("got 3 GPUs (basic display filtered)",
              len(gpus) == 3, f"got {len(gpus)}")
        vendors = sorted(g.vendor for g in gpus)
        check("vendors include nvidia, amd, intel",
              vendors == ["amd", "intel", "nvidia"], f"got {vendors}")
        nv = next(g for g in gpus if g.vendor == "nvidia")
        check("NVIDIA via WMI uses catalog VRAM (24 GB)",
              nv.vram_total_mb == 24 * 1024,
              f"got {nv.vram_total_mb}")
    finally:
        sm.restore()

    print("\n[9] WMI on non-Windows returns []")
    sm = _StubManager()
    try:
        sm.install_platform_system("Linux")
        gpus = gd._detect_wmi()
        check("Linux WMI -> empty", gpus == [])
    finally:
        sm.restore()

    print("\n[10] detect_gpus() merges + dedups")
    nvidia_output = (
        "0, NVIDIA GeForce RTX 5090, 32760, 28000, 555.42, 12.0\n"
    )
    wmi_output_with_5090 = (
        '[{"Name": "NVIDIA GeForce RTX 5090", '
        '"AdapterRAM": 4294967295, "DriverVersion": "555.42"}]'
    )
    sm = _StubManager()
    call_count = {"n": 0}

    def fake_run(cmd, timeout=5.0):
        call_count["n"] += 1
        cmd_s = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "nvidia-smi" in cmd_s:
            return nvidia_output
        if "Win32_VideoController" in cmd_s or "powershell" in cmd_s.lower():
            return wmi_output_with_5090
        if "rocm-smi" in cmd_s or "xpu-smi" in cmd_s:
            return None
        return None

    def fake_which(name):
        if name == "nvidia-smi":
            return "/fake/nvidia-smi"
        return None

    try:
        sm.install_which(fake_which)
        sm.install_run(fake_run)
        sm.install_platform_system("Windows")
        gpus = gd.detect_gpus()
        # nvidia-smi returns 1, WMI would also see it — dedup by
        # (vendor, name, vram). The nvidia-smi path runs first so it
        # populates the result; WMI fallback only runs if nvidia-smi
        # returns nothing.
        check("dedupe works (no double 5090)",
              len(gpus) == 1, f"got {len(gpus)}: {[g.name for g in gpus]}")
        check("vendor is nvidia",
              gpus[0].vendor == "nvidia")
        check("source is nvidia-smi (not wmi)",
              gpus[0].detection_source == "nvidia-smi")
    finally:
        sm.restore()

    print("\n[11] detect_gpus() with NO probes succeeding -> []")
    sm = _StubManager()
    try:
        sm.install_which(lambda n: None)
        sm.install_run(lambda cmd, timeout=5.0: None)
        sm.install_platform_system("Linux")
        gpus = gd.detect_gpus()
        check("no probe success -> empty", gpus == [])
    finally:
        sm.restore()

    print("\n[12] primary_gpu picks the most capable")
    sm = _StubManager()
    try:
        sm.install_which(lambda n: "/fake/nvidia-smi" if n == "nvidia-smi" else None)
        sm.install_run(lambda cmd, timeout=5.0:
                       "0, NVIDIA GeForce RTX 3060, 12288, 11000, 555.42, 8.6\n"
                       "1, NVIDIA GeForce RTX 5090, 32760, 28000, 555.42, 12.0\n")
        gpus = gd.detect_gpus()
        primary = gd.primary_gpu(gpus)
        check("primary picks the bigger card",
              primary is not None and "5090" in primary.name,
              f"got {primary.name if primary else None}")
    finally:
        sm.restore()

    print("\n[13] primary_gpu on empty list -> None")
    check("primary of [] is None",
          gd.primary_gpu([]) is None)

    print("\n[14] DetectedGpu.is_compatible thresholds")
    g_small = gd.DetectedGpu(
        index=0, vendor="nvidia",
        name="GeForce GTX 1050", matched_spec=None,
        vram_total_mb=2048, vram_free_mb=None,
        driver_version="535", runtime="cuda",
        detection_source="test",
    )
    check("2 GB card not compatible",
          not g_small.is_compatible)

    g_unknown = gd.DetectedGpu(
        index=0, vendor="amd",
        name="Mystery card", matched_spec=None,
        vram_total_mb=12 * 1024, vram_free_mb=None,
        driver_version=None, runtime="rocm",
        detection_source="wmi",
    )
    check("Unknown AMD without catalog match not compatible",
          not g_unknown.is_compatible)

    print(f"\n=== gpu_detect summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

"""Static catalog of GPUs ez-rag knows about.

Hand-curated reference data, not scraped at runtime. Every consumer +
workstation + datacenter card we expect users to plausibly run ez-rag on
is in here, with VRAM, memory bandwidth, FP16 TFLOPS, runtime support,
and a tier label that drives model recommendations.

Multi-VRAM SKUs (e.g. RTX A2000 6 GB vs 12 GB, 4060 Ti 8 GB vs 16 GB)
get separate entries with the SAME aliases. The matcher in
`gpu_detect.py` uses detected VRAM to disambiguate.

Tier convention used by the recommender:
  - min            4–6 GB    tiny LLMs only (phi4-mini, llama3.2:1b)
  - comfortable    7–12 GB   7B Q4 + embedder
  - ample          13–24 GB  14–32B Q4 / 7B at higher quant
  - professional   26–48 GB  70B Q3, contextual retrieval at scale
  - extreme        64+ GB    full precision smaller models, 70B+ Q4

Bandwidth + FP16 TFLOPS numbers are approximate (vendor spec sheets
vary by source). The recommender uses bandwidth as a rough tokens/sec
predictor; small inaccuracies don't materially change suggestions.

Architecture → runtime support tier:
  pascal           CUDA, but legacy quant kernels only (P40)
  volta/turing     full CUDA
  ampere/ada/      full CUDA + flash attention
   hopper/blackwell
  cdna1            ROCm Linux only
  cdna2/3/4        full ROCm + HIP Windows
  rdna2            HIP Windows preview / ROCm Linux
  rdna3+           full HIP Windows + ROCm Linux
  xe-hpg/xe2       Intel oneAPI / Level Zero
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuSpec:
    """One canonical entry in the catalog."""
    name: str
    aliases: tuple[str, ...]
    vendor: str               # "nvidia" | "amd" | "intel"
    family: str               # "geforce" | "rtx-pro" | "datacenter" | "jetson" |
                              # "radeon" | "radeon-pro" | "instinct" |
                              # "arc" | "intel-dc"
    architecture: str
    vram_gb: int
    bandwidth_gbps: int
    fp16_tflops: float
    runtime: str              # "cuda" | "rocm" | "hip" | "xpu"
    runtime_notes: str = ""
    laptop: bool = False
    data_center: bool = False
    legacy: bool = False
    tier: str = ""            # auto-derived if blank, see _autotier


def _autotier(vram_gb: int, *, data_center: bool = False) -> str:
    if vram_gb >= 64:
        return "extreme"
    if vram_gb >= 26:
        return "professional"
    if vram_gb >= 13:
        return "ample"
    if vram_gb >= 7:
        return "comfortable"
    return "min"


def _spec(name, aliases, vendor, family, architecture, vram_gb,
           bandwidth_gbps, fp16_tflops, runtime, *,
           runtime_notes="", laptop=False, data_center=False,
           legacy=False, tier=None) -> GpuSpec:
    return GpuSpec(
        name=name,
        aliases=tuple(aliases),
        vendor=vendor,
        family=family,
        architecture=architecture,
        vram_gb=vram_gb,
        bandwidth_gbps=bandwidth_gbps,
        fp16_tflops=fp16_tflops,
        runtime=runtime,
        runtime_notes=runtime_notes,
        laptop=laptop,
        data_center=data_center,
        legacy=legacy,
        tier=tier or _autotier(vram_gb, data_center=data_center),
    )


# ============================================================================
# NVIDIA — GeForce desktop (consumer)
# ============================================================================

_NVIDIA_GEFORCE_DESKTOP: list[GpuSpec] = [
    # ----- 30-series (Ampere) -----
    _spec("RTX 3050 6GB",
          ["NVIDIA GeForce RTX 3050 6GB", "GeForce RTX 3050 6GB"],
          "nvidia", "geforce", "ampere", 6, 168, 9.1, "cuda"),
    _spec("RTX 3050",
          ["NVIDIA GeForce RTX 3050", "GeForce RTX 3050"],
          "nvidia", "geforce", "ampere", 8, 224, 9.1, "cuda"),
    _spec("RTX 3060 8GB",
          ["NVIDIA GeForce RTX 3060 8GB"],
          "nvidia", "geforce", "ampere", 8, 240, 12.7, "cuda"),
    _spec("RTX 3060",
          ["NVIDIA GeForce RTX 3060", "GeForce RTX 3060"],
          "nvidia", "geforce", "ampere", 12, 360, 12.7, "cuda"),
    _spec("RTX 3060 Ti",
          ["NVIDIA GeForce RTX 3060 Ti", "GeForce RTX 3060 Ti"],
          "nvidia", "geforce", "ampere", 8, 448, 16.2, "cuda"),
    _spec("RTX 3070",
          ["NVIDIA GeForce RTX 3070", "GeForce RTX 3070"],
          "nvidia", "geforce", "ampere", 8, 448, 20.3, "cuda"),
    _spec("RTX 3070 Ti",
          ["NVIDIA GeForce RTX 3070 Ti", "GeForce RTX 3070 Ti"],
          "nvidia", "geforce", "ampere", 8, 608, 21.7, "cuda"),
    _spec("RTX 3080",
          ["NVIDIA GeForce RTX 3080", "GeForce RTX 3080"],
          "nvidia", "geforce", "ampere", 10, 760, 29.8, "cuda"),
    _spec("RTX 3080 12GB",
          ["NVIDIA GeForce RTX 3080 12GB"],
          "nvidia", "geforce", "ampere", 12, 912, 30.6, "cuda"),
    _spec("RTX 3080 Ti",
          ["NVIDIA GeForce RTX 3080 Ti", "GeForce RTX 3080 Ti"],
          "nvidia", "geforce", "ampere", 12, 912, 34.1, "cuda"),
    _spec("RTX 3090",
          ["NVIDIA GeForce RTX 3090", "GeForce RTX 3090"],
          "nvidia", "geforce", "ampere", 24, 936, 35.6, "cuda"),
    _spec("RTX 3090 Ti",
          ["NVIDIA GeForce RTX 3090 Ti", "GeForce RTX 3090 Ti"],
          "nvidia", "geforce", "ampere", 24, 1008, 40.0, "cuda"),

    # ----- 40-series (Ada Lovelace) -----
    _spec("RTX 4060",
          ["NVIDIA GeForce RTX 4060", "GeForce RTX 4060"],
          "nvidia", "geforce", "ada", 8, 272, 15.1, "cuda"),
    _spec("RTX 4060 Ti 8GB",
          ["NVIDIA GeForce RTX 4060 Ti", "GeForce RTX 4060 Ti"],
          "nvidia", "geforce", "ada", 8, 288, 22.1, "cuda"),
    _spec("RTX 4060 Ti 16GB",
          ["NVIDIA GeForce RTX 4060 Ti", "GeForce RTX 4060 Ti"],
          "nvidia", "geforce", "ada", 16, 288, 22.1, "cuda"),
    _spec("RTX 4070",
          ["NVIDIA GeForce RTX 4070", "GeForce RTX 4070"],
          "nvidia", "geforce", "ada", 12, 504, 29.1, "cuda"),
    _spec("RTX 4070 Super",
          ["NVIDIA GeForce RTX 4070 SUPER", "GeForce RTX 4070 SUPER",
           "NVIDIA GeForce RTX 4070 Super"],
          "nvidia", "geforce", "ada", 12, 504, 35.5, "cuda"),
    _spec("RTX 4070 Ti",
          ["NVIDIA GeForce RTX 4070 Ti", "GeForce RTX 4070 Ti"],
          "nvidia", "geforce", "ada", 12, 504, 40.1, "cuda"),
    _spec("RTX 4070 Ti Super",
          ["NVIDIA GeForce RTX 4070 Ti SUPER",
           "NVIDIA GeForce RTX 4070 Ti Super",
           "GeForce RTX 4070 Ti SUPER"],
          "nvidia", "geforce", "ada", 16, 672, 44.1, "cuda"),
    _spec("RTX 4080",
          ["NVIDIA GeForce RTX 4080", "GeForce RTX 4080"],
          "nvidia", "geforce", "ada", 16, 717, 48.7, "cuda"),
    _spec("RTX 4080 Super",
          ["NVIDIA GeForce RTX 4080 SUPER",
           "NVIDIA GeForce RTX 4080 Super"],
          "nvidia", "geforce", "ada", 16, 736, 52.2, "cuda"),
    _spec("RTX 4090",
          ["NVIDIA GeForce RTX 4090", "GeForce RTX 4090"],
          "nvidia", "geforce", "ada", 24, 1008, 82.6, "cuda"),

    # ----- 50-series (Blackwell) -----
    _spec("RTX 5060",
          ["NVIDIA GeForce RTX 5060", "GeForce RTX 5060"],
          "nvidia", "geforce", "blackwell", 8, 448, 19.2, "cuda"),
    _spec("RTX 5060 Ti 8GB",
          ["NVIDIA GeForce RTX 5060 Ti", "GeForce RTX 5060 Ti"],
          "nvidia", "geforce", "blackwell", 8, 448, 23.7, "cuda"),
    _spec("RTX 5060 Ti 16GB",
          ["NVIDIA GeForce RTX 5060 Ti", "GeForce RTX 5060 Ti"],
          "nvidia", "geforce", "blackwell", 16, 448, 23.7, "cuda"),
    _spec("RTX 5070",
          ["NVIDIA GeForce RTX 5070", "GeForce RTX 5070"],
          "nvidia", "geforce", "blackwell", 12, 672, 30.9, "cuda"),
    _spec("RTX 5070 Ti",
          ["NVIDIA GeForce RTX 5070 Ti", "GeForce RTX 5070 Ti"],
          "nvidia", "geforce", "blackwell", 16, 896, 43.9, "cuda"),
    _spec("RTX 5080",
          ["NVIDIA GeForce RTX 5080", "GeForce RTX 5080"],
          "nvidia", "geforce", "blackwell", 16, 960, 56.3, "cuda"),
    _spec("RTX 5090",
          ["NVIDIA GeForce RTX 5090", "GeForce RTX 5090"],
          "nvidia", "geforce", "blackwell", 32, 1792, 104.8, "cuda"),
]


# ============================================================================
# NVIDIA — GeForce mobile / laptop
# ============================================================================
# Mobile cards report distinctly in nvidia-smi (suffix " Laptop GPU") and
# usually have less VRAM and lower clocks than their desktop namesakes.

_NVIDIA_GEFORCE_LAPTOP: list[GpuSpec] = [
    # ----- 30-series mobile -----
    _spec("RTX 3050 Laptop",
          ["NVIDIA GeForce RTX 3050 Laptop GPU"],
          "nvidia", "geforce", "ampere", 4, 192, 7.6, "cuda", laptop=True),
    _spec("RTX 3050 Ti Laptop",
          ["NVIDIA GeForce RTX 3050 Ti Laptop GPU"],
          "nvidia", "geforce", "ampere", 4, 192, 8.0, "cuda", laptop=True),
    _spec("RTX 3060 Laptop",
          ["NVIDIA GeForce RTX 3060 Laptop GPU"],
          "nvidia", "geforce", "ampere", 6, 336, 10.9, "cuda", laptop=True),
    _spec("RTX 3070 Laptop",
          ["NVIDIA GeForce RTX 3070 Laptop GPU"],
          "nvidia", "geforce", "ampere", 8, 384, 16.6, "cuda", laptop=True),
    _spec("RTX 3070 Ti Laptop",
          ["NVIDIA GeForce RTX 3070 Ti Laptop GPU"],
          "nvidia", "geforce", "ampere", 8, 448, 17.7, "cuda", laptop=True),
    _spec("RTX 3080 Laptop 8GB",
          ["NVIDIA GeForce RTX 3080 Laptop GPU"],
          "nvidia", "geforce", "ampere", 8, 384, 20.0, "cuda", laptop=True),
    _spec("RTX 3080 Laptop 16GB",
          ["NVIDIA GeForce RTX 3080 Laptop GPU"],
          "nvidia", "geforce", "ampere", 16, 512, 21.0, "cuda", laptop=True),
    _spec("RTX 3080 Ti Laptop",
          ["NVIDIA GeForce RTX 3080 Ti Laptop GPU"],
          "nvidia", "geforce", "ampere", 16, 512, 22.7, "cuda", laptop=True),

    # ----- 40-series mobile -----
    _spec("RTX 4050 Laptop",
          ["NVIDIA GeForce RTX 4050 Laptop GPU"],
          "nvidia", "geforce", "ada", 6, 216, 8.9, "cuda", laptop=True),
    _spec("RTX 4060 Laptop",
          ["NVIDIA GeForce RTX 4060 Laptop GPU"],
          "nvidia", "geforce", "ada", 8, 256, 15.1, "cuda", laptop=True),
    _spec("RTX 4070 Laptop",
          ["NVIDIA GeForce RTX 4070 Laptop GPU"],
          "nvidia", "geforce", "ada", 8, 256, 18.4, "cuda", laptop=True),
    _spec("RTX 4080 Laptop",
          ["NVIDIA GeForce RTX 4080 Laptop GPU"],
          "nvidia", "geforce", "ada", 12, 432, 24.7, "cuda", laptop=True),
    _spec("RTX 4090 Laptop",
          ["NVIDIA GeForce RTX 4090 Laptop GPU"],
          "nvidia", "geforce", "ada", 16, 576, 32.9, "cuda", laptop=True),

    # ----- 50-series mobile (Blackwell) -----
    _spec("RTX 5060 Laptop",
          ["NVIDIA GeForce RTX 5060 Laptop GPU"],
          "nvidia", "geforce", "blackwell", 8, 320, 16.0, "cuda", laptop=True),
    _spec("RTX 5070 Laptop",
          ["NVIDIA GeForce RTX 5070 Laptop GPU"],
          "nvidia", "geforce", "blackwell", 8, 384, 24.0, "cuda", laptop=True),
    _spec("RTX 5070 Ti Laptop",
          ["NVIDIA GeForce RTX 5070 Ti Laptop GPU"],
          "nvidia", "geforce", "blackwell", 12, 512, 30.5, "cuda", laptop=True),
    _spec("RTX 5080 Laptop",
          ["NVIDIA GeForce RTX 5080 Laptop GPU"],
          "nvidia", "geforce", "blackwell", 16, 768, 39.5, "cuda", laptop=True),
    _spec("RTX 5090 Laptop",
          ["NVIDIA GeForce RTX 5090 Laptop GPU"],
          "nvidia", "geforce", "blackwell", 24, 896, 51.2, "cuda", laptop=True),
]


# ============================================================================
# NVIDIA — RTX A-series workstation (Ampere)
# ============================================================================

_NVIDIA_RTX_AMPERE: list[GpuSpec] = [
    _spec("RTX A2000 6GB",
          ["NVIDIA RTX A2000", "RTX A2000"],
          "nvidia", "rtx-pro", "ampere", 6, 288, 7.9, "cuda"),
    _spec("RTX A2000 12GB",
          ["NVIDIA RTX A2000 12GB", "RTX A2000 12GB"],
          "nvidia", "rtx-pro", "ampere", 12, 288, 7.9, "cuda"),
    _spec("RTX A4000",
          ["NVIDIA RTX A4000", "RTX A4000"],
          "nvidia", "rtx-pro", "ampere", 16, 448, 19.2, "cuda"),
    _spec("RTX A4000H",
          ["NVIDIA RTX A4000H", "RTX A4000H"],
          "nvidia", "rtx-pro", "ampere", 16, 448, 18.6, "cuda"),
    _spec("RTX A4500",
          ["NVIDIA RTX A4500", "RTX A4500"],
          "nvidia", "rtx-pro", "ampere", 20, 640, 23.7, "cuda"),
    _spec("RTX A5000",
          ["NVIDIA RTX A5000", "RTX A5000"],
          "nvidia", "rtx-pro", "ampere", 24, 768, 27.8, "cuda"),
    _spec("RTX A5500",
          ["NVIDIA RTX A5500", "RTX A5500"],
          "nvidia", "rtx-pro", "ampere", 24, 768, 34.1, "cuda"),
    _spec("RTX A6000",
          ["NVIDIA RTX A6000", "RTX A6000"],
          "nvidia", "rtx-pro", "ampere", 48, 768, 38.7, "cuda"),
]


# ============================================================================
# NVIDIA — RTX Ada workstation
# ============================================================================

_NVIDIA_RTX_ADA: list[GpuSpec] = [
    _spec("RTX 2000 Ada",
          ["NVIDIA RTX 2000 Ada Generation", "RTX 2000 Ada"],
          "nvidia", "rtx-pro", "ada", 16, 224, 12.0, "cuda"),
    _spec("RTX 4000 SFF Ada",
          ["NVIDIA RTX 4000 SFF Ada Generation", "RTX 4000 SFF Ada"],
          "nvidia", "rtx-pro", "ada", 20, 280, 19.2, "cuda"),
    _spec("RTX 4000 Ada",
          ["NVIDIA RTX 4000 Ada Generation", "RTX 4000 Ada"],
          "nvidia", "rtx-pro", "ada", 20, 360, 26.7, "cuda"),
    _spec("RTX 4500 Ada",
          ["NVIDIA RTX 4500 Ada Generation", "RTX 4500 Ada"],
          "nvidia", "rtx-pro", "ada", 24, 432, 39.6, "cuda"),
    _spec("RTX 5000 Ada",
          ["NVIDIA RTX 5000 Ada Generation", "RTX 5000 Ada"],
          "nvidia", "rtx-pro", "ada", 32, 576, 65.3, "cuda"),
    _spec("RTX 5880 Ada",
          ["NVIDIA RTX 5880 Ada Generation", "RTX 5880 Ada"],
          "nvidia", "rtx-pro", "ada", 48, 960, 80.0, "cuda"),
    _spec("RTX 6000 Ada",
          ["NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada"],
          "nvidia", "rtx-pro", "ada", 48, 960, 91.1, "cuda"),
]


# ============================================================================
# NVIDIA — RTX PRO Blackwell workstation (new naming)
# ============================================================================

_NVIDIA_RTX_PRO_BLACKWELL: list[GpuSpec] = [
    _spec("RTX PRO 4000 Blackwell",
          ["NVIDIA RTX PRO 4000 Blackwell", "RTX PRO 4000"],
          "nvidia", "rtx-pro", "blackwell", 24, 432, 38.0, "cuda"),
    _spec("RTX PRO 4500 Blackwell",
          ["NVIDIA RTX PRO 4500 Blackwell", "RTX PRO 4500"],
          "nvidia", "rtx-pro", "blackwell", 32, 576, 60.0, "cuda"),
    _spec("RTX PRO 5000 Blackwell",
          ["NVIDIA RTX PRO 5000 Blackwell", "RTX PRO 5000"],
          "nvidia", "rtx-pro", "blackwell", 48, 1152, 90.0, "cuda"),
    _spec("RTX PRO 6000 Blackwell",
          ["NVIDIA RTX PRO 6000 Blackwell", "RTX PRO 6000",
           "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
           "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"],
          "nvidia", "rtx-pro", "blackwell", 96, 1792, 120.0, "cuda"),
]


# ============================================================================
# NVIDIA — Quadro RTX (Turing, legacy but common in workstations)
# ============================================================================

_NVIDIA_QUADRO_RTX: list[GpuSpec] = [
    _spec("Quadro RTX 4000",
          ["Quadro RTX 4000", "NVIDIA Quadro RTX 4000"],
          "nvidia", "rtx-pro", "turing", 8, 416, 7.4, "cuda",
          legacy=True),
    _spec("Quadro RTX 5000",
          ["Quadro RTX 5000", "NVIDIA Quadro RTX 5000"],
          "nvidia", "rtx-pro", "turing", 16, 448, 11.2, "cuda",
          legacy=True),
    _spec("Quadro RTX 6000",
          ["Quadro RTX 6000", "NVIDIA Quadro RTX 6000"],
          "nvidia", "rtx-pro", "turing", 24, 672, 16.3, "cuda",
          legacy=True),
    _spec("Quadro RTX 8000",
          ["Quadro RTX 8000", "NVIDIA Quadro RTX 8000"],
          "nvidia", "rtx-pro", "turing", 48, 672, 16.3, "cuda",
          legacy=True),
]


# ============================================================================
# NVIDIA — datacenter / AI inference
# ============================================================================

_NVIDIA_DATACENTER: list[GpuSpec] = [
    _spec("Tesla T4",
          ["Tesla T4", "NVIDIA T4"],
          "nvidia", "datacenter", "turing", 16, 320, 8.1, "cuda",
          data_center=True),
    _spec("Tesla V100 PCIe 16GB",
          ["Tesla V100-PCIE-16GB", "Tesla V100", "NVIDIA V100 PCIe 16GB"],
          "nvidia", "datacenter", "volta", 16, 900, 28.3, "cuda",
          data_center=True),
    _spec("Tesla V100 SXM2 16GB",
          ["Tesla V100-SXM2-16GB", "NVIDIA V100 SXM2 16GB"],
          "nvidia", "datacenter", "volta", 16, 900, 31.4, "cuda",
          data_center=True),
    _spec("Tesla V100 SXM2 32GB",
          ["Tesla V100-SXM2-32GB", "NVIDIA V100 SXM2 32GB"],
          "nvidia", "datacenter", "volta", 32, 900, 31.4, "cuda",
          data_center=True),
    _spec("Tesla P40",
          ["Tesla P40", "NVIDIA Tesla P40"],
          "nvidia", "datacenter", "pascal", 24, 346, 11.8, "cuda",
          data_center=True, legacy=True,
          runtime_notes="Pascal — Ollama 0.5+ may have issues; "
                        "stick to Q4_K_M quantization."),
    _spec("A10",
          ["NVIDIA A10", "A10"],
          "nvidia", "datacenter", "ampere", 24, 600, 31.2, "cuda",
          data_center=True),
    _spec("A16",
          ["NVIDIA A16", "A16"],
          "nvidia", "datacenter", "ampere", 16, 200, 8.0, "cuda",
          data_center=True,
          runtime_notes="Quad-GPU board — appears as 4 separate "
                        "16 GB devices to nvidia-smi."),
    _spec("A30",
          ["NVIDIA A30", "A30"],
          "nvidia", "datacenter", "ampere", 24, 933, 82.0, "cuda",
          data_center=True),
    _spec("A40",
          ["NVIDIA A40", "A40"],
          "nvidia", "datacenter", "ampere", 48, 696, 37.4, "cuda",
          data_center=True),
    _spec("A100 PCIe 40GB",
          ["NVIDIA A100-PCIE-40GB", "A100-PCIE-40GB"],
          "nvidia", "datacenter", "ampere", 40, 1555, 77.9, "cuda",
          data_center=True),
    _spec("A100 PCIe 80GB",
          ["NVIDIA A100-PCIE-80GB", "A100-PCIE-80GB"],
          "nvidia", "datacenter", "ampere", 80, 1935, 77.9, "cuda",
          data_center=True),
    _spec("A100 SXM4 40GB",
          ["NVIDIA A100-SXM4-40GB", "A100-SXM4-40GB"],
          "nvidia", "datacenter", "ampere", 40, 1555, 77.9, "cuda",
          data_center=True),
    _spec("A100 SXM4 80GB",
          ["NVIDIA A100-SXM4-80GB", "A100-SXM4-80GB"],
          "nvidia", "datacenter", "ampere", 80, 2039, 77.9, "cuda",
          data_center=True),
    _spec("L4",
          ["NVIDIA L4", "L4"],
          "nvidia", "datacenter", "ada", 24, 300, 30.3, "cuda",
          data_center=True),
    _spec("L40",
          ["NVIDIA L40", "L40"],
          "nvidia", "datacenter", "ada", 48, 864, 90.5, "cuda",
          data_center=True),
    _spec("L40S",
          ["NVIDIA L40S", "L40S"],
          "nvidia", "datacenter", "ada", 48, 864, 91.6, "cuda",
          data_center=True),
    _spec("H100 PCIe 80GB",
          ["NVIDIA H100 PCIe", "H100 PCIe"],
          "nvidia", "datacenter", "hopper", 80, 2039, 204.9, "cuda",
          data_center=True),
    _spec("H100 SXM5 80GB",
          ["NVIDIA H100 80GB HBM3", "H100 SXM5"],
          "nvidia", "datacenter", "hopper", 80, 3350, 267.6, "cuda",
          data_center=True),
    _spec("H100 NVL 94GB",
          ["NVIDIA H100 NVL", "H100 NVL"],
          "nvidia", "datacenter", "hopper", 94, 3938, 267.6, "cuda",
          data_center=True),
    _spec("H200 SXM5 141GB",
          ["NVIDIA H200", "H200"],
          "nvidia", "datacenter", "hopper", 141, 4800, 267.6, "cuda",
          data_center=True),
    _spec("B100",
          ["NVIDIA B100", "B100"],
          "nvidia", "datacenter", "blackwell", 192, 8000, 884.0, "cuda",
          data_center=True),
    _spec("B200",
          ["NVIDIA B200", "B200"],
          "nvidia", "datacenter", "blackwell", 192, 8000, 1100.0, "cuda",
          data_center=True),
]


# ============================================================================
# NVIDIA — Jetson edge
# ============================================================================

_NVIDIA_JETSON: list[GpuSpec] = [
    _spec("Jetson Orin Nano",
          ["Orin (nvgpu)", "NVIDIA Jetson Orin Nano"],
          "nvidia", "jetson", "ampere", 8, 68, 1.3, "cuda",
          runtime_notes="ARM-side iGPU; CUDA via JetPack."),
    _spec("Jetson Orin NX 8GB",
          ["NVIDIA Jetson Orin NX 8GB"],
          "nvidia", "jetson", "ampere", 8, 102, 2.5, "cuda",
          runtime_notes="ARM-side iGPU; CUDA via JetPack."),
    _spec("Jetson Orin NX 16GB",
          ["NVIDIA Jetson Orin NX 16GB"],
          "nvidia", "jetson", "ampere", 16, 102, 2.5, "cuda",
          runtime_notes="ARM-side iGPU; CUDA via JetPack."),
    _spec("Jetson AGX Orin 32GB",
          ["NVIDIA Jetson AGX Orin 32GB"],
          "nvidia", "jetson", "ampere", 32, 204, 5.3, "cuda",
          runtime_notes="ARM-side iGPU; CUDA via JetPack."),
    _spec("Jetson AGX Orin 64GB",
          ["NVIDIA Jetson AGX Orin 64GB"],
          "nvidia", "jetson", "ampere", 64, 204, 5.3, "cuda",
          runtime_notes="ARM-side iGPU; CUDA via JetPack."),
]


# ============================================================================
# AMD — Radeon RX desktop (consumer)
# ============================================================================

_AMD_RADEON_DESKTOP: list[GpuSpec] = [
    # ----- RX 6000-series (RDNA2) -----
    _spec("RX 6600",
          ["AMD Radeon RX 6600", "Radeon RX 6600"],
          "amd", "radeon", "rdna2", 8, 224, 8.9, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6650 XT",
          ["AMD Radeon RX 6650 XT", "Radeon RX 6650 XT"],
          "amd", "radeon", "rdna2", 8, 280, 10.8, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6700 XT",
          ["AMD Radeon RX 6700 XT", "Radeon RX 6700 XT"],
          "amd", "radeon", "rdna2", 12, 384, 13.2, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6750 XT",
          ["AMD Radeon RX 6750 XT", "Radeon RX 6750 XT"],
          "amd", "radeon", "rdna2", 12, 432, 13.5, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6800",
          ["AMD Radeon RX 6800", "Radeon RX 6800"],
          "amd", "radeon", "rdna2", 16, 512, 16.2, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6800 XT",
          ["AMD Radeon RX 6800 XT", "Radeon RX 6800 XT"],
          "amd", "radeon", "rdna2", 16, 512, 20.7, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6900 XT",
          ["AMD Radeon RX 6900 XT", "Radeon RX 6900 XT"],
          "amd", "radeon", "rdna2", 16, 512, 23.0, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),
    _spec("RX 6950 XT",
          ["AMD Radeon RX 6950 XT", "Radeon RX 6950 XT"],
          "amd", "radeon", "rdna2", 16, 576, 23.7, "rocm",
          runtime_notes="RDNA2 — HIP Windows preview / ROCm Linux."),

    # ----- RX 7000-series (RDNA3) -----
    _spec("RX 7600",
          ["AMD Radeon RX 7600", "Radeon RX 7600"],
          "amd", "radeon", "rdna3", 8, 288, 21.5, "rocm"),
    _spec("RX 7600 XT",
          ["AMD Radeon RX 7600 XT", "Radeon RX 7600 XT"],
          "amd", "radeon", "rdna3", 16, 288, 22.6, "rocm"),
    _spec("RX 7700 XT",
          ["AMD Radeon RX 7700 XT", "Radeon RX 7700 XT"],
          "amd", "radeon", "rdna3", 12, 432, 35.2, "rocm"),
    _spec("RX 7800 XT",
          ["AMD Radeon RX 7800 XT", "Radeon RX 7800 XT"],
          "amd", "radeon", "rdna3", 16, 624, 37.3, "rocm"),
    _spec("RX 7900 GRE",
          ["AMD Radeon RX 7900 GRE", "Radeon RX 7900 GRE"],
          "amd", "radeon", "rdna3", 16, 576, 45.9, "rocm"),
    _spec("RX 7900 XT",
          ["AMD Radeon RX 7900 XT", "Radeon RX 7900 XT"],
          "amd", "radeon", "rdna3", 20, 800, 51.6, "rocm"),
    _spec("RX 7900 XTX",
          ["AMD Radeon RX 7900 XTX", "Radeon RX 7900 XTX"],
          "amd", "radeon", "rdna3", 24, 960, 61.4, "rocm"),

    # ----- RX 9000-series (RDNA4) -----
    _spec("RX 9070",
          ["AMD Radeon RX 9070", "Radeon RX 9070"],
          "amd", "radeon", "rdna4", 16, 644, 48.7, "rocm"),
    _spec("RX 9070 XT",
          ["AMD Radeon RX 9070 XT", "Radeon RX 9070 XT"],
          "amd", "radeon", "rdna4", 16, 644, 51.4, "rocm"),
]


# ============================================================================
# AMD — Radeon Pro workstation
# ============================================================================

_AMD_RADEON_PRO: list[GpuSpec] = [
    _spec("Radeon Pro W6600",
          ["AMD Radeon Pro W6600", "Radeon Pro W6600"],
          "amd", "radeon-pro", "rdna2", 8, 224, 10.4, "rocm",
          runtime_notes="RDNA2 — ROCm Linux primary."),
    _spec("Radeon Pro W6800",
          ["AMD Radeon Pro W6800", "Radeon Pro W6800"],
          "amd", "radeon-pro", "rdna2", 32, 512, 17.8, "rocm",
          runtime_notes="RDNA2 — ROCm Linux primary."),
    _spec("Radeon Pro W6900X",
          ["AMD Radeon Pro W6900X", "Radeon Pro W6900X"],
          "amd", "radeon-pro", "rdna2", 32, 512, 22.6, "rocm",
          runtime_notes="RDNA2 — Mac Pro card."),
    _spec("Radeon Pro W7600",
          ["AMD Radeon Pro W7600", "Radeon Pro W7600"],
          "amd", "radeon-pro", "rdna3", 8, 256, 20.0, "rocm"),
    _spec("Radeon Pro W7700",
          ["AMD Radeon Pro W7700", "Radeon Pro W7700"],
          "amd", "radeon-pro", "rdna3", 16, 576, 28.3, "rocm"),
    _spec("Radeon Pro W7800",
          ["AMD Radeon Pro W7800", "Radeon Pro W7800"],
          "amd", "radeon-pro", "rdna3", 32, 576, 45.2, "rocm"),
    _spec("Radeon Pro W7900",
          ["AMD Radeon Pro W7900", "Radeon Pro W7900"],
          "amd", "radeon-pro", "rdna3", 48, 864, 61.3, "rocm"),
]


# ============================================================================
# AMD — Instinct datacenter
# ============================================================================

_AMD_INSTINCT: list[GpuSpec] = [
    _spec("Instinct MI25",
          ["AMD Instinct MI25", "Instinct MI25"],
          "amd", "instinct", "vega", 16, 484, 24.6, "rocm",
          data_center=True, legacy=True,
          runtime_notes="Vega — ROCm 5.x final-supported; tight."),
    _spec("Instinct MI50",
          ["AMD Instinct MI50", "Instinct MI50"],
          "amd", "instinct", "cdna1", 32, 1024, 26.5, "rocm",
          data_center=True),
    _spec("Instinct MI60",
          ["AMD Instinct MI60", "Instinct MI60"],
          "amd", "instinct", "cdna1", 32, 1024, 29.5, "rocm",
          data_center=True),
    _spec("Instinct MI100",
          ["AMD Instinct MI100", "Instinct MI100"],
          "amd", "instinct", "cdna1", 32, 1228, 184.6, "rocm",
          data_center=True),
    _spec("Instinct MI210",
          ["AMD Instinct MI210", "Instinct MI210"],
          "amd", "instinct", "cdna2", 64, 1638, 181.0, "rocm",
          data_center=True),
    _spec("Instinct MI250",
          ["AMD Instinct MI250", "Instinct MI250"],
          "amd", "instinct", "cdna2", 128, 3277, 362.1, "rocm",
          data_center=True),
    _spec("Instinct MI250X",
          ["AMD Instinct MI250X", "Instinct MI250X"],
          "amd", "instinct", "cdna2", 128, 3277, 383.0, "rocm",
          data_center=True),
    _spec("Instinct MI300A",
          ["AMD Instinct MI300A", "Instinct MI300A"],
          "amd", "instinct", "cdna3", 128, 5325, 980.6, "rocm",
          data_center=True,
          runtime_notes="APU — unified CPU + GPU memory."),
    _spec("Instinct MI300X",
          ["AMD Instinct MI300X", "Instinct MI300X"],
          "amd", "instinct", "cdna3", 192, 5325, 1307.4, "rocm",
          data_center=True),
    _spec("Instinct MI325X",
          ["AMD Instinct MI325X", "Instinct MI325X"],
          "amd", "instinct", "cdna3", 256, 6000, 1307.4, "rocm",
          data_center=True),
    _spec("Instinct MI355X",
          ["AMD Instinct MI355X", "Instinct MI355X"],
          "amd", "instinct", "cdna4", 288, 8000, 2500.0, "rocm",
          data_center=True,
          runtime_notes="Specs preliminary."),
]


# ============================================================================
# Intel — Arc consumer + Data Center
# ============================================================================

_INTEL: list[GpuSpec] = [
    _spec("Arc A380",
          ["Intel Arc A380 Graphics", "Arc A380"],
          "intel", "arc", "xe-hpg", 6, 186, 4.2, "xpu"),
    _spec("Arc A580",
          ["Intel Arc A580 Graphics", "Arc A580"],
          "intel", "arc", "xe-hpg", 8, 512, 12.3, "xpu"),
    _spec("Arc A750",
          ["Intel Arc A750 Graphics", "Arc A750"],
          "intel", "arc", "xe-hpg", 8, 512, 17.2, "xpu"),
    _spec("Arc A770 8GB",
          ["Intel Arc A770 Graphics", "Arc A770"],
          "intel", "arc", "xe-hpg", 8, 512, 19.6, "xpu"),
    _spec("Arc A770 16GB",
          ["Intel Arc A770 Graphics", "Arc A770"],
          "intel", "arc", "xe-hpg", 16, 560, 19.6, "xpu"),
    _spec("Arc B570",
          ["Intel Arc B570 Graphics", "Arc B570"],
          "intel", "arc", "xe2", 10, 380, 18.0, "xpu"),
    _spec("Arc B580",
          ["Intel Arc B580 Graphics", "Arc B580"],
          "intel", "arc", "xe2", 12, 456, 22.0, "xpu"),
    _spec("Data Center GPU Flex 140",
          ["Intel Data Center GPU Flex 140", "Flex 140"],
          "intel", "intel-dc", "xe-hpg", 12, 256, 14.4, "xpu",
          data_center=True),
    _spec("Data Center GPU Flex 170",
          ["Intel Data Center GPU Flex 170", "Flex 170"],
          "intel", "intel-dc", "xe-hpg", 16, 576, 31.0, "xpu",
          data_center=True),
    _spec("Data Center GPU Max 1100",
          ["Intel Data Center GPU Max 1100", "Max 1100"],
          "intel", "intel-dc", "xe-hpc", 48, 1228, 96.0, "xpu",
          data_center=True),
    _spec("Data Center GPU Max 1550",
          ["Intel Data Center GPU Max 1550", "Max 1550"],
          "intel", "intel-dc", "xe-hpc", 128, 3277, 104.0, "xpu",
          data_center=True),
]


# ============================================================================
# Aggregate
# ============================================================================

CATALOG: list[GpuSpec] = [
    *_NVIDIA_GEFORCE_DESKTOP,
    *_NVIDIA_GEFORCE_LAPTOP,
    *_NVIDIA_RTX_AMPERE,
    *_NVIDIA_RTX_ADA,
    *_NVIDIA_RTX_PRO_BLACKWELL,
    *_NVIDIA_QUADRO_RTX,
    *_NVIDIA_DATACENTER,
    *_NVIDIA_JETSON,
    *_AMD_RADEON_DESKTOP,
    *_AMD_RADEON_PRO,
    *_AMD_INSTINCT,
    *_INTEL,
]


# ============================================================================
# Lookup helpers
# ============================================================================

def _normalize(name: str) -> str:
    """Lowercase, collapse whitespace, strip vendor prefixes that tend to
    appear inconsistently across nvidia-smi / WMI / lspci output."""
    if not name:
        return ""
    s = " ".join(name.split()).lower()
    for prefix in ("nvidia ", "amd ", "intel(r) ", "intel "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def find_spec(name: str, vram_mb: Optional[int] = None) -> Optional[GpuSpec]:
    """Match a detected GPU name (and optionally its detected VRAM in MB)
    to a catalog entry.

    Match strategy:
      1. Collect every spec whose name or any alias matches the
         normalized input string (exact match after normalization).
      2. If multiple candidates AND vram_mb is provided, pick the one
         whose spec.vram_gb is closest to vram_mb / 1024.
      3. If still ambiguous, return the first (insertion-order) match —
         catalog ordering puts the most common SKU first within a name.

    Returns None if no candidate matches.
    """
    target = _normalize(name)
    if not target:
        return None
    candidates: list[GpuSpec] = []
    for spec in CATALOG:
        names = (spec.name, *spec.aliases)
        if any(_normalize(n) == target for n in names):
            candidates.append(spec)
    if not candidates:
        return None
    if len(candidates) == 1 or vram_mb is None:
        return candidates[0]
    # Disambiguate by VRAM proximity.
    detected_gb = vram_mb / 1024.0
    return min(candidates, key=lambda s: abs(s.vram_gb - detected_gb))


def runtime_for_architecture(arch: str) -> str:
    """Map an architecture string to the runtime ez-rag will use to talk
    to the card. Used to fall back when a spec doesn't carry an explicit
    runtime field (shouldn't happen, but defensive)."""
    arch = arch.lower()
    if arch in ("pascal", "volta", "turing", "ampere", "ada", "hopper",
                "blackwell"):
        return "cuda"
    if arch in ("vega", "cdna1", "cdna2", "cdna3", "cdna4",
                "rdna2", "rdna3", "rdna4"):
        return "rocm"
    if arch in ("xe-hpg", "xe2", "xe-hpc"):
        return "xpu"
    return "unknown"

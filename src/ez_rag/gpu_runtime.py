"""Apply the user's selected GPU to the runtime environment.

The Settings → Hardware card lets the user pick one detected GPU. That
choice has to actually take effect — Ollama / fastembed / OCR / the
cross-encoder reranker each have their own way of being told which
device to use.

This module is the single seam where that translation happens.

  apply_selected_gpu(cfg)        Side-effect: set process env vars so
                                 child processes (Ollama spawned by us,
                                 fastembed, etc.) inherit the pin.
  make_ollama_env(cfg)           Return a dict of env overrides to pass
                                 to a manually-spawned Ollama process.
  pin_torch_device(cfg)          Set torch.cuda.set_device(idx) before
                                 cross-encoder reranker model load.
  selected_gpu_index(cfg)        Helper — extract the integer index from
                                 cfg.gpu_index (defaults to 0).

All functions are no-ops when:
  - cfg has no gpu_index / gpu_vendor set, OR
  - cfg.gpu_index < 0 (sentinel for "CPU mode"), OR
  - the corresponding library isn't installed (defensive).

Important: ez-rag does NOT manage the Ollama daemon's lifecycle in v1
— users start it themselves. So setting CUDA_VISIBLE_DEVICES at our
process scope only changes which GPU OUR Python uses (fastembed, torch).
For Ollama, we surface a doctor-tab warning when the daemon's visible
GPUs don't match the selection.
"""
from __future__ import annotations

import os
from typing import Optional


# Sentinel: when cfg.gpu_index is this value, the user has explicitly
# chosen CPU-only mode (or no GPU was detected and we fell back).
GPU_INDEX_CPU = -1


def selected_gpu_index(cfg) -> int:
    """Return the configured GPU index, defaulting to 0 when unset."""
    raw = getattr(cfg, "gpu_index", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def selected_gpu_vendor(cfg) -> str:
    return str(getattr(cfg, "gpu_vendor", "") or "").lower()


def is_cpu_mode(cfg) -> bool:
    return selected_gpu_index(cfg) == GPU_INDEX_CPU


def make_ollama_env(cfg) -> dict[str, str]:
    """Build the env-var dict to hand to a freshly-spawned Ollama
    process so it pins to the user's GPU choice.

    NVIDIA: CUDA_VISIBLE_DEVICES=<index>
    AMD:    HIP_VISIBLE_DEVICES=<index> + ROCR_VISIBLE_DEVICES (alt name)
    Intel:  ONEAPI_DEVICE_SELECTOR=level_zero:<index>
    CPU:    OLLAMA_NO_GPU=1 (deliberately disable GPU)

    Returns an empty dict if no useful pin can be derived (so callers
    can `env.update(make_ollama_env(cfg))` unconditionally).
    """
    env: dict[str, str] = {}
    if is_cpu_mode(cfg):
        env["OLLAMA_NO_GPU"] = "1"
        return env

    idx = selected_gpu_index(cfg)
    vendor = selected_gpu_vendor(cfg)
    if vendor == "nvidia":
        env["CUDA_VISIBLE_DEVICES"] = str(idx)
    elif vendor == "amd":
        env["HIP_VISIBLE_DEVICES"] = str(idx)
        env["ROCR_VISIBLE_DEVICES"] = str(idx)
    elif vendor == "intel":
        env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{idx}"
    return env


def apply_selected_gpu(cfg) -> None:
    """Mutate this Python process's os.environ so child libraries that
    inherit env vars (fastembed via onnxruntime, sentence-transformers
    via torch) honor the user's selection.

    Idempotent. Safe to call repeatedly (e.g. on every Settings save).
    """
    env_overrides = make_ollama_env(cfg)
    for k, v in env_overrides.items():
        os.environ[k] = v


def pin_torch_device(cfg) -> Optional[int]:
    """Pin the active CUDA device for torch (used by the cross-encoder
    reranker). No-op when:
      - torch isn't installed
      - CUDA isn't available
      - user is in CPU mode
      - selected vendor isn't NVIDIA (torch ROCm support is build-time)

    Returns the device index that was actually pinned, or None when no
    pin was applied.
    """
    if is_cpu_mode(cfg):
        return None
    if selected_gpu_vendor(cfg) != "nvidia":
        return None
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        idx = selected_gpu_index(cfg)
        if 0 <= idx < torch.cuda.device_count():
            torch.cuda.set_device(idx)
            return idx
    except Exception:
        pass
    return None


def onnxruntime_providers(cfg) -> list:
    """Build the providers list for onnxruntime sessions (RapidOCR uses
    these). Pins to a specific device when possible, falls back to CPU.

    Used like:
        sess = ort.InferenceSession(path,
                                     providers=onnxruntime_providers(cfg))
    """
    if is_cpu_mode(cfg):
        return ["CPUExecutionProvider"]
    vendor = selected_gpu_vendor(cfg)
    idx = selected_gpu_index(cfg)
    if vendor == "nvidia":
        return [
            ("CUDAExecutionProvider", {"device_id": idx}),
            "CPUExecutionProvider",
        ]
    if vendor == "amd":
        return [
            ("ROCMExecutionProvider", {"device_id": idx}),
            "CPUExecutionProvider",
        ]
    if vendor == "intel":
        # OpenVINOExecutionProvider is the typical Intel path.
        return [
            ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]

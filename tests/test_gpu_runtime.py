"""Tests for gpu_runtime — the env-var / device-pin layer.

Pure tests: no real GPU calls, just verifies that selecting a GPU in cfg
produces the right env-var dict for each backend.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.gpu_runtime import (
    GPU_INDEX_CPU,
    apply_selected_gpu, is_cpu_mode, make_ollama_env,
    onnxruntime_providers, selected_gpu_index, selected_gpu_vendor,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def cfg_with(*, vendor="nvidia", index=0):
    c = Config()
    setattr(c, "gpu_vendor", vendor)
    setattr(c, "gpu_index", index)
    return c


def main():
    print("\n[1] selected_gpu_index defaults & coercion")
    c = Config()
    check("default index is 0",
          selected_gpu_index(c) == 0)
    setattr(c, "gpu_index", "2")
    check("string '2' coerces to int 2",
          selected_gpu_index(c) == 2)
    setattr(c, "gpu_index", "junk")
    check("non-numeric falls back to 0",
          selected_gpu_index(c) == 0)
    setattr(c, "gpu_index", -1)
    check("CPU sentinel preserved",
          selected_gpu_index(c) == -1)

    print("\n[2] selected_gpu_vendor lowercase")
    c2 = cfg_with(vendor="NVIDIA")
    check("vendor lowercased",
          selected_gpu_vendor(c2) == "nvidia")
    c3 = Config()
    check("missing vendor -> ''",
          selected_gpu_vendor(c3) == "")

    print("\n[3] is_cpu_mode")
    c_cpu = cfg_with(index=GPU_INDEX_CPU)
    check("index=-1 -> cpu mode True",
          is_cpu_mode(c_cpu))
    check("default cfg is not cpu mode",
          not is_cpu_mode(Config()))

    print("\n[4] make_ollama_env — NVIDIA")
    env = make_ollama_env(cfg_with(vendor="nvidia", index=0))
    check("NVIDIA gets CUDA_VISIBLE_DEVICES",
          env.get("CUDA_VISIBLE_DEVICES") == "0",
          f"got {env}")
    check("NVIDIA does NOT set HIP",
          "HIP_VISIBLE_DEVICES" not in env)
    env_idx2 = make_ollama_env(cfg_with(vendor="nvidia", index=2))
    check("respects non-zero index",
          env_idx2.get("CUDA_VISIBLE_DEVICES") == "2")

    print("\n[5] make_ollama_env — AMD")
    env = make_ollama_env(cfg_with(vendor="amd", index=1))
    check("AMD sets HIP_VISIBLE_DEVICES",
          env.get("HIP_VISIBLE_DEVICES") == "1")
    check("AMD also sets ROCR_VISIBLE_DEVICES",
          env.get("ROCR_VISIBLE_DEVICES") == "1")
    check("AMD does NOT set CUDA",
          "CUDA_VISIBLE_DEVICES" not in env)

    print("\n[6] make_ollama_env — Intel")
    env = make_ollama_env(cfg_with(vendor="intel", index=0))
    check("Intel sets ONEAPI_DEVICE_SELECTOR",
          env.get("ONEAPI_DEVICE_SELECTOR") == "level_zero:0",
          f"got {env}")

    print("\n[7] make_ollama_env — CPU mode disables GPU")
    env = make_ollama_env(cfg_with(index=GPU_INDEX_CPU))
    check("CPU mode sets OLLAMA_NO_GPU=1",
          env.get("OLLAMA_NO_GPU") == "1",
          f"got {env}")
    check("CPU mode does NOT pin a GPU",
          "CUDA_VISIBLE_DEVICES" not in env
           and "HIP_VISIBLE_DEVICES" not in env)

    print("\n[8] make_ollama_env — unknown vendor returns empty")
    env = make_ollama_env(cfg_with(vendor="quantum", index=0))
    check("unknown vendor -> empty dict",
          env == {}, f"got {env}")

    print("\n[9] apply_selected_gpu mutates os.environ")
    saved_cuda = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    saved_hip = os.environ.pop("HIP_VISIBLE_DEVICES", None)
    try:
        apply_selected_gpu(cfg_with(vendor="nvidia", index=3))
        check("CUDA_VISIBLE_DEVICES set to 3",
              os.environ.get("CUDA_VISIBLE_DEVICES") == "3")
        # Switching vendors should set the new vars; the old ones are
        # left as-is (we don't aggressively clear unrelated keys).
        apply_selected_gpu(cfg_with(vendor="amd", index=0))
        check("HIP_VISIBLE_DEVICES added on switch",
              os.environ.get("HIP_VISIBLE_DEVICES") == "0")
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("OLLAMA_NO_GPU", None)
        if saved_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda
        if saved_hip is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = saved_hip

    print("\n[10] onnxruntime_providers — NVIDIA")
    p = onnxruntime_providers(cfg_with(vendor="nvidia", index=1))
    check("NVIDIA provider list starts with CUDA",
          isinstance(p[0], tuple)
          and p[0][0] == "CUDAExecutionProvider")
    check("NVIDIA device_id passed",
          p[0][1].get("device_id") == 1)
    check("CPU fallback always present",
          "CPUExecutionProvider" in p)

    print("\n[11] onnxruntime_providers — AMD / Intel / CPU")
    p_amd = onnxruntime_providers(cfg_with(vendor="amd", index=0))
    check("AMD provider list starts with ROCM",
          isinstance(p_amd[0], tuple)
          and p_amd[0][0] == "ROCMExecutionProvider")

    p_intel = onnxruntime_providers(cfg_with(vendor="intel", index=0))
    check("Intel provider list mentions OpenVINO",
          isinstance(p_intel[0], tuple)
          and p_intel[0][0] == "OpenVINOExecutionProvider")

    p_cpu = onnxruntime_providers(cfg_with(index=GPU_INDEX_CPU))
    check("CPU mode -> CPU-only",
          p_cpu == ["CPUExecutionProvider"])

    print(f"\n=== gpu_runtime summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

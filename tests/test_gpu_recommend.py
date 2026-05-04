"""Tests for gpu_recommend — the assessment + model recommendation engine.

Builds synthetic DetectedGpu records that mirror the user's test matrix
(RTX 5060 Mobile / 5090 / 3090) and a few additional VRAM tiers, then
asserts which models the recommender suggests / withholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ez_rag.config import Config
from ez_rag.gpu_catalog import find_spec
from ez_rag.gpu_detect import DetectedGpu
from ez_rag.gpu_recommend import (
    Assessment, ModelSuggestion, VramRequirement,
    assess, recommend_models, estimate_required_vram,
    _BASELINE_OVERHEAD_MB, _tier_from_headroom,
)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def fake_gpu(name, vram_gb, vendor="nvidia", driver="555.42",
              free_gb=None):
    """Build a synthetic DetectedGpu for a card whose name is in the
    catalog. Uses find_spec to stay consistent with the real lookup."""
    spec = find_spec(name, vram_mb=vram_gb * 1024)
    return DetectedGpu(
        index=0,
        vendor=vendor,
        name=name,
        matched_spec=spec,
        vram_total_mb=vram_gb * 1024,
        vram_free_mb=(free_gb * 1024) if free_gb else None,
        driver_version=driver,
        runtime={"nvidia": "cuda", "amd": "rocm",
                  "intel": "xpu"}.get(vendor, "unknown"),
        detection_source="test",
        health_notes=[],
    )


def main():
    print("\n[1] tier mapping from headroom")
    check("60 GB headroom -> extreme",
          _tier_from_headroom(60 * 1024) == "extreme")
    check("32 GB headroom -> professional",
          _tier_from_headroom(32 * 1024) == "professional")
    check("16 GB headroom -> ample",
          _tier_from_headroom(16 * 1024) == "ample")
    check("8 GB headroom -> comfortable",
          _tier_from_headroom(8 * 1024) == "comfortable")
    check("3 GB headroom -> min",
          _tier_from_headroom(3 * 1024) == "min")
    check("1 GB headroom -> insufficient",
          _tier_from_headroom(1 * 1024) == "insufficient")

    print("\n[2] assess() on user's test-matrix cards")
    g_5090 = fake_gpu("NVIDIA GeForce RTX 5090", 32)
    a_5090 = assess(g_5090)
    check("5090 is runnable", a_5090.runnable is True)
    check("5090 tier is professional",
          a_5090.tier == "professional", f"got {a_5090.tier}")
    check("5090 headroom > 28 GB",
          a_5090.headroom_mb > 28 * 1024,
          f"got {a_5090.headroom_mb}")
    check("5090 runtime blurb mentions CUDA",
          "CUDA" in a_5090.runtime_blurb)

    g_3090 = fake_gpu("NVIDIA GeForce RTX 3090", 24)
    a_3090 = assess(g_3090)
    check("3090 is runnable", a_3090.runnable is True)
    check("3090 tier is ample",
          a_3090.tier == "ample", f"got {a_3090.tier}")

    g_5060m = fake_gpu("NVIDIA GeForce RTX 5060 Laptop GPU", 8)
    a_5060m = assess(g_5060m)
    check("5060 Mobile is runnable", a_5060m.runnable is True)
    check("5060 Mobile tier is comfortable",
          a_5060m.tier == "comfortable", f"got {a_5060m.tier}")

    print("\n[3] recommend_models for 5090 (professional tier)")
    recs_5090 = recommend_models(g_5090)
    tags = [r.tag for r in recs_5090]
    check("5090 recs include qwen2.5:7b-instruct",
          "qwen2.5:7b-instruct" in tags)
    check("5090 recs include qwen2.5:14b",
          "qwen2.5:14b" in tags)
    check("5090 recs include qwen2.5:32b",
          "qwen2.5:32b" in tags)
    check("5090 recs include embedder",
          any(r.role == "embedder" for r in recs_5090))
    check("5090 recs include reranker",
          any(r.role == "reranker" for r in recs_5090))
    # 70B Q3 is ~36 GB — won't fit on 32 GB 5090
    fitting_chat = [r for r in recs_5090
                    if r.role == "chat" and r.fits]
    check("5090 fits 32B chat model",
          any("32b" in r.tag for r in fitting_chat))
    not_fitting = [r for r in recs_5090
                   if r.role == "chat" and not r.fits]
    check("5090 surfaces a 70B 'next tier' card",
          any("70b" in r.tag or "72b" in r.tag for r in not_fitting),
          f"got fits=False set: {[r.tag for r in not_fitting]}")

    print("\n[4] recommend_models for 3090 (ample tier)")
    recs_3090 = recommend_models(g_3090)
    tags = [r.tag for r in recs_3090]
    check("3090 recs include 7B",
          "qwen2.5:7b-instruct" in tags)
    check("3090 recs include 14B",
          "qwen2.5:14b" in tags)
    fitting_chat = [r for r in recs_3090
                    if r.role == "chat" and r.fits]
    fitting_tags = {r.tag for r in fitting_chat}
    # 32B at Q4 is ~20 GB — fits on 24 GB after baseline (22.5 GB headroom)
    check("3090 fits 32B Q4",
          "qwen2.5:32b" in fitting_tags,
          f"fitting tags: {fitting_tags}")

    print("\n[5] recommend_models for 5060 Mobile (comfortable tier)")
    recs_5060m = recommend_models(g_5060m)
    fitting_chat = [r for r in recs_5060m
                    if r.role == "chat" and r.fits]
    fitting_tags = {r.tag for r in fitting_chat}
    check("5060 Mobile fits a 7B",
          "qwen2.5:7b-instruct" in fitting_tags,
          f"fitting: {fitting_tags}")
    # 14B Q4 is 9.5 GB — won't fit on 8 GB after 1.5 GB baseline (6.5 GB headroom)
    check("5060 Mobile does NOT fit 14B",
          "qwen2.5:14b" not in fitting_tags,
          f"fitting: {fitting_tags}")
    # The non-fitting peek should show 14B as the next-tier unlock.
    not_fitting = [r for r in recs_5060m
                    if r.role == "chat" and not r.fits]
    check("5060 Mobile peek shows 14B as unlock",
          any("14b" in r.tag for r in not_fitting),
          f"not-fitting: {[r.tag for r in not_fitting]}")

    print("\n[6] recommend_models for tiny GPU (4 GB)")
    g_4gb = fake_gpu("NVIDIA GeForce RTX 3050", 8)  # closest catalog match
    # Override VRAM to simulate a 4 GB card.
    g_4gb.vram_total_mb = 4096
    recs = recommend_models(g_4gb)
    fitting = [r for r in recs if r.role == "chat" and r.fits]
    fitting_tags = {r.tag for r in fitting}
    check("4 GB does not fit 7B",
          "qwen2.5:7b-instruct" not in fitting_tags,
          f"fitting: {fitting_tags}")
    check("4 GB fits something tiny",
          len(fitting) >= 1, f"got: {fitting_tags}")

    print("\n[7] recommend_models for extreme tier (H100 80GB)")
    g_h100 = fake_gpu("NVIDIA H100 PCIe", 80)
    recs_h100 = recommend_models(g_h100)
    fitting_tags = {r.tag for r in recs_h100
                     if r.role == "chat" and r.fits}
    check("H100 fits 70B",
          any("70b" in t for t in fitting_tags) or "qwen2.5:72b" in fitting_tags,
          f"fitting: {fitting_tags}")
    check("H100 fits 32B comfortably",
          "qwen2.5:32b" in fitting_tags)

    print("\n[8] estimate_required_vram with default cfg (7B)")
    cfg = Config()
    req = estimate_required_vram(cfg)
    check("returns VramRequirement",
          isinstance(req, VramRequirement))
    check("7B + embedder min ~7 GB",
          5 <= req.min_vram_gb <= 9,
          f"got {req.min_vram_gb}")
    check("recommended > min",
          req.recommended_vram_gb > req.min_vram_gb,
          f"min={req.min_vram_gb} rec={req.recommended_vram_gb}")
    check("manifest carries the LLM tag",
          req.llm_model == "qwen2.5:7b-instruct")

    print("\n[9] estimate_required_vram with 32B model")
    cfg32 = Config()
    cfg32.llm_model = "qwen2.5:32b"
    req32 = estimate_required_vram(cfg32)
    check("32B min is ~22 GB",
          20 <= req32.min_vram_gb <= 24,
          f"got {req32.min_vram_gb}")

    print("\n[10] estimate_required_vram with unknown model falls back conservative")
    cfg_unknown = Config()
    cfg_unknown.llm_model = "imaginary-model:99b"
    req_unknown = estimate_required_vram(cfg_unknown)
    check("unknown model gets fallback estimate",
          req_unknown.min_vram_gb >= 5,
          f"got {req_unknown.min_vram_gb}")
    check("manifest preserves user-provided tag",
          req_unknown.llm_model == "imaginary-model:99b")

    print("\n[11] expected tokens/sec roughly tracks bandwidth")
    # 5090 (1792 GB/s) should produce a higher ceiling than 3090 (936 GB/s)
    # for the same model.
    rec_5090_7b = next((r for r in recs_5090
                         if r.tag == "qwen2.5:7b-instruct"), None)
    rec_3090_7b = next((r for r in recs_3090
                         if r.tag == "qwen2.5:7b-instruct"), None)
    check("both recommend 7B",
          rec_5090_7b is not None and rec_3090_7b is not None)
    check("5090 expected high tps > 3090 expected high tps",
          rec_5090_7b.expected_tps[1] > rec_3090_7b.expected_tps[1],
          f"5090={rec_5090_7b.expected_tps} 3090={rec_3090_7b.expected_tps}")

    print("\n[12] CPU-fallback shape")
    g_cpu = DetectedGpu(
        index=-1, vendor="unknown", name="CPU", matched_spec=None,
        vram_total_mb=0, vram_free_mb=None, driver_version=None,
        runtime="cpu", detection_source="fallback",
    )
    a_cpu = assess(g_cpu)
    check("CPU is not runnable as GPU",
          a_cpu.runnable is False, f"{a_cpu}")

    print(f"\n=== gpu_recommend summary: {len(PASS)} pass, "
          f"{len(FAIL)} fail ===")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAIL  {n} :: {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

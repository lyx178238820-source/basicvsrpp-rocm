# -*- coding: utf-8 -*-
"""Smoke test: run a real clip through the engine on the GPU.

Skipped automatically when there is no CUDA device or no checkpoint. Point the
test at a checkpoint with the BVR_WEIGHTS environment variable (defaults to
weights/lada_mosaic_restoration_model_generic_v1.2.pth).
"""
import os
from pathlib import Path

import pytest
import torch

from basicvsrpp_rocm import BasicVSRPlusPlusEngine
from basicvsrpp_rocm.engine import BVR_HIP_GRAPH_ENV

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "lada_mosaic_restoration_model_generic_v1.2.pth"
WEIGHTS = Path(os.environ.get("BVR_WEIGHTS", DEFAULT_WEIGHTS))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not WEIGHTS.is_file(),
    reason="needs a CUDA device and a BasicVSR++ checkpoint (set BVR_WEIGHTS)",
)

CLIP = 12  # short on purpose: capture warmup dominates longer graphs


def _frames(t: int, size: int = 256) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(7)
    return [torch.randint(0, 256, (3, size, size), dtype=torch.uint8, generator=gen) for _ in range(t)]


def test_engine_restores_clip() -> None:
    engine = BasicVSRPlusPlusEngine(
        checkpoint_path=str(WEIGHTS), device="cuda", max_clip_size=CLIP, fp16=True
    )
    try:
        out = engine.raw_process(_frames(CLIP))
        assert out.shape == (CLIP, 3, 256, 256)
        assert out.dtype.is_floating_point
        assert torch.isfinite(out).all()
        # The generator is unbounded (random-noise input can overshoot [0, 1]
        # slightly); only sanity-bound the magnitude here.
        assert float(out.abs().max()) < 8.0
    finally:
        engine.close()


def test_graph_output_matches_eager() -> None:
    """HIP-graph replay must be numerically identical to the eager forward."""
    eager_env = os.environ.get(BVR_HIP_GRAPH_ENV)
    try:
        os.environ[BVR_HIP_GRAPH_ENV] = "0"
        eager = BasicVSRPlusPlusEngine(
            checkpoint_path=str(WEIGHTS), device="cuda", max_clip_size=CLIP, fp16=True
        )
        os.environ[BVR_HIP_GRAPH_ENV] = "1"
        graphed = BasicVSRPlusPlusEngine(
            checkpoint_path=str(WEIGHTS), device="cuda", max_clip_size=CLIP, fp16=True
        )
        frames = _frames(CLIP)
        with torch.inference_mode():
            out_eager = eager.raw_process(frames)
            out_graph = graphed.raw_process(frames)
        assert torch.equal(out_eager, out_graph), (
            "graph replay diverged from eager output"
        )
    finally:
        if eager_env is None:
            os.environ.pop(BVR_HIP_GRAPH_ENV, None)
        else:
            os.environ[BVR_HIP_GRAPH_ENV] = eager_env

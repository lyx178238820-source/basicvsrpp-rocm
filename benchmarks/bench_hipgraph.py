# -*- coding: utf-8 -*-
"""Measure the engine's steady-state throughput, eager vs HIP-graph replay.

Usage:
    python benchmarks/bench_hipgraph.py --weights weights/lada_mosaic_restoration_model_generic_v1.2.pth \
        [--clip-frames 60] [--clips 12] [--size 256] [--no-graph]

The point of the exercise: the plain PyTorch forward is launch/dispatch bound
(~150k small ops per clip), so HIP-graph replay of the captured forward is the
single biggest win available on ROCm. The run is bounded and reports the
steady-state rate (the first clip is treated as warmup, since it also performs
MIOpen kernel selection and lazy graph capture).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


def bench(engine, clip_frames: int, clips: int, size: int) -> float:
    """Run `clips` clips; return seconds per frame over everything but the first."""
    frames = [
        torch.randint(0, 256, (3, size, size), dtype=torch.uint8)
        for _ in range(clip_frames)
    ]
    timings: list[float] = []
    for i in range(clips):
        start = time.perf_counter()
        out = engine.raw_process(frames)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        assert out.shape == (clip_frames, 3, size, size), out.shape
    steady = timings[1:] or timings
    per_frame = sum(steady) / (len(steady) * clip_frames)
    return per_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to the BasicVSR++ .pth checkpoint")
    parser.add_argument("--clip-frames", type=int, default=60)
    parser.add_argument("--clips", type=int, default=12)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--no-graph", action="store_true", help="disable HIP graphs for this run")
    args = parser.parse_args()

    from basicvsrpp_rocm import BasicVSRPlusPlusEngine

    if args.no_graph:
        import os
        os.environ["BVR_HIP_GRAPH"] = "0"

    device = torch.device("cuda")
    engine = BasicVSRPlusPlusEngine(
        checkpoint_path=args.weights,
        device=device,
        max_clip_size=args.clip_frames,
        fp16=True,
        temporal_overlap=args.overlap,
    )
    label = "eager " if args.no_graph else "graph "
    ms = bench(engine, args.clip_frames, args.clips, args.size) * 1000.0
    print(f"{label}: {ms:7.1f} ms/frame  ({1000.0 / ms:5.1f} fps)  "
          f"[T={args.clip_frames}, {args.size}x{args.size}, fp16]")
    engine.close()


if __name__ == "__main__":
    main()

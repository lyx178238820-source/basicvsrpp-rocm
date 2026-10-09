# -*- coding: utf-8 -*-
"""BasicVSR++ inference engine for AMD ROCm (the path that replaced TensorRT).

On NVIDIA, jasna cuts BasicVSR++ into TensorRT sub-engines. AMD has no TensorRT,
so this engine runs the plain PyTorch forward instead and wins the speed back
with a stack of ROCm-specific measures (see README.md for the full story):

  * fp16 weights + inputs
  * HIP graph capture of the whole forward for the clip lengths that actually
    occur (534 -> 292 ms per 60-frame 256x256 clip on an RX 7900 XT,
    bit-identical output)
  * MIOpen FIND_MODE=FAST, allocator tuning, AOTriton flash-attention
  * a VRAM budget so long jobs never spill into Windows shared memory

The engine takes a list of (C, H, W) RGB frames in [0, 255] and returns the
restored (T, C, H, W) float tensor in [0, 1].
"""
from __future__ import annotations

import logging
import os

import torch
from torch import Tensor

from basicvsrpp_rocm.models.inference import load_model
from basicvsrpp_rocm import rocm_env

logger = logging.getLogger(__name__)

# ~150k small CUDA ops per forward make the plain PyTorch path launch/dispatch
# bound: HIP-graph replay of the captured forward measured 534 -> 292 ms at
# T=60 (bit-identical output) on an RX 7900 XT (torch 2.14+rocm10.1, Windows).
#
# Graphs need static shapes AND a quiet GPU: capturing while other threads
# submit GPU work poisons their calls ("operation not permitted when stream is
# capturing" aborts them on this ROCm build). Capture therefore happens once,
# at construction time - before any worker thread exists - for the clip
# lengths that cover virtually every clip of a run: max_clip_size and
# max_clip_size - 2*temporal_overlap. Any other length (final partial clip,
# scene-cut clip) runs eager.
BVR_HIP_GRAPH_ENV = "BVR_HIP_GRAPH"
# Extra clip lengths to capture, comma separated (e.g. "52,40"). Any clip whose
# length was not captured runs eager, ~1.8x slower per frame on the same card,
# so a run whose clips end early (scene cuts) pays that on every such clip.
# Capture cost is one eager forward plus the graph's reserved activations per
# length, so add only the lengths a clip histogram actually shows.
BVR_HIP_GRAPH_LENGTHS_ENV = "BVR_HIP_GRAPH_LENGTHS"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUTHY


class BasicVSRPlusPlusEngine:
    """Load a BasicVSR++ checkpoint and restore clips on an AMD (or NVIDIA) GPU.

    Args:
        checkpoint_path: path to the ``.pth`` checkpoint (see weights/README.md).
        device: ``torch.device`` - typically ``torch.device('cuda')`` on ROCm.
        max_clip_size: the temporal length the caller will feed most often;
            graphs are captured for this length (and for
            ``max_clip_size - 2 * temporal_overlap``).
        fp16: half-precision weights/inputs (recommended on RDNA3: ~2x
            throughput, no visible quality change).
        config: optional mmengine config path / dict; ``None`` uses the
            default BasicVSR++ GAN inference config.
        temporal_overlap: frames of overlap between consecutive clips (only
            relevant for choosing the second graph length).
        apply_rocm_defaults: run :func:`basicvsrpp_rocm.rocm_env.configure_process`
            first (env vars, SDPA availability check, VRAM budget).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device | str = "cuda",
        max_clip_size: int = 60,
        fp16: bool = True,
        config: str | dict | None = None,
        temporal_overlap: int = 0,
        apply_rocm_defaults: bool = True,
    ):
        self.device = torch.device(device)
        if apply_rocm_defaults:
            rocm_env.configure_process(self.device)
        self.max_clip_size = int(max_clip_size)
        self.input_dtype = torch.float16 if fp16 else torch.float32

        self.model = load_model(config, checkpoint_path, self.device, fp16)
        logger.info(
            "BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16
        )

        self._graphs: dict[int, tuple[Tensor, Tensor, "torch.cuda.CUDAGraph"]] = {}
        self._graphs_enabled = _env_flag(BVR_HIP_GRAPH_ENV, default=True) and (
            self.device.type == "cuda"
        )
        if self._graphs_enabled:
            self._prewarm_graphs(int(temporal_overlap))

    def _prewarm_graphs(self, temporal_overlap: int) -> None:
        """Capture graphs for the dominant clip lengths before worker threads
        start (mid-pipeline capture is not safe on ROCm)."""
        lengths: list[int] = []
        t1 = self.max_clip_size
        t2 = t1 - 2 * int(temporal_overlap)
        if t1 > 1:
            lengths.append(t1)
        if 1 < t2 < t1:
            lengths.append(t2)
        for token in os.environ.get(BVR_HIP_GRAPH_LENGTHS_ENV, "").split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            extra = int(token)
            if extra > 1 and extra not in lengths:
                lengths.append(extra)
        for t in lengths:
            reserved_before = torch.cuda.memory_reserved(self.device)
            try:
                with torch.inference_mode():
                    static_input = torch.zeros(
                        (1, t, 3, 256, 256), device=self.device, dtype=self.input_dtype
                    )
                    # the eager call doubles as the capture warmup (MIOpen
                    # kernel selection must happen outside the capture)
                    self.model(inputs=static_input)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_output = self.model(inputs=static_input)
            except Exception:
                logger.warning(
                    "BasicVSR++ HIP graph capture failed for T=%d; that clip"
                    " length will run eager.", t, exc_info=True,
                )
                torch.cuda.synchronize(self.device)
                continue
            self._graphs[t] = (static_input, static_output, graph)
            logger.info(
                "BasicVSR++ HIP graph captured for T=%d (+%.0f MiB reserved)",
                t,
                (torch.cuda.memory_reserved(self.device) - reserved_before) / (1024 * 1024),
            )
        if self._graphs:
            logger.info(
                "BasicVSR++ HIP graphs ready for clip lengths %s (%s=0 disables)",
                sorted(self._graphs), BVR_HIP_GRAPH_ENV,
            )

    def close(self) -> None:
        self._graphs.clear()
        self._graphs_enabled = False
        self.model = None

    def _forward_graphed(self, batched: Tensor) -> Tensor:
        entry = self._graphs.get(int(batched.shape[1]))
        if entry is None:
            return self.model(inputs=batched)
        static_input, static_output, graph = entry
        static_input.copy_(batched)
        graph.replay()
        # static_output is a fixed buffer that the next replay overwrites;
        # callers may hold the result until long after, so hand out a copy.
        return static_output.clone()

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """Restore one clip.

        Args:
            video: list of ``(C, H, W)`` RGB tensors, values in ``[0, 255]``.
                All frames must share one spatial size (256x256 is what the
                shipped checkpoints were trained on).
        Returns:
            ``(T, C, H, W)`` float tensor on ``self.device``, nominally in
            ``[0, 1]``. The generator is unbounded, so clamp before writing
            out (the reference pipeline does the same).
        """
        with torch.inference_mode():
            stacked = torch.stack(video).to(
                device=self.device,
                dtype=self.input_dtype,
                memory_format=torch.contiguous_format,
            ).div_(255.0)
            batched = stacked.unsqueeze(0)
            return self._forward_graphed(batched).squeeze(0)

# BasicVSR++ on AMD ROCm — the engine that replaced TensorRT

A standalone inference engine that runs **BasicVSR++** video restoration on
AMD cards (ROCm). The NVIDIA side of the upstream project cuts BasicVSR++ into
**TensorRT sub-engines**; AMD has no TensorRT, and the model's core operator
(modulated deformable convolution) does not export to MIGraphX/ONNX cleanly.
This repository contains the **full source and the lessons learned** for the
replacement path: a plain PyTorch ROCm forward plus a stack of ROCm-specific
optimizations, validated on an RX 7900 XT.

> This is an engine only — **no host application code** (no pipeline, no GUI,
> no detector) and **no weights** are included.

## Measured results (RX 7900 XT / gfx1100, torch 2.14.0+rocm10.1, Windows)

| Technique | Effect |
|---|---|
| **HIP graph capture of the whole forward** | 60-frame 256² clip: **534 → 292 ms (1.83×)**, bit-identical output; this repo's benchmark measures **eager 10.3 → graph 5.0 ms/frame (2.06×, 199 fps)** |
| fp16 weights + inputs | ~2× throughput, no visible quality loss |
| AOTriton FLASH SDPA | attention-bound model: **38.3 → 18.8 ms/frame** @576² |
| VRAM budget (anti-spill) | removes the Windows shared-memory fallback collapse (1.2 fps after spill → 5.8+ steady) |
| MIOPEN_FIND_MODE=FAST | avoids repeated runtime tuning for every new convolution shape (variable T ⇒ variable effective conv batches) |

End-to-end (decode → detect → restore → encode) on a 4K source: **5.8–5.9
frames/s steady state** (x4plus second-stage restoration, fp16).

## Why "plain PyTorch + HIP graph" instead of TensorRT

- TensorRT sub-engines are NVIDIA-only; there is no AMD runtime.
- BasicVSR++'s core operator is **modulated deformable convolution**
  (`models/deformconv.py`); MIGraphX/ONNX support for it is incomplete, so the
  export route is a dead end.
- The PyTorch forward is not compute-bound, it is **launch/dispatch bound**:
  one 60-frame clip issues ~150k small kernels. Replaying the captured HIP
  graph removes that overhead in one shot — the single biggest win available
  on ROCm, with **bit-identical output**.
- On the PyTorch route the deformable conv actually lands in
  `torchvision.ops.deform_conv2d` — **torchvision's ROCm wheels ship that
  operator**, which is precisely why this pure-PyTorch path works on AMD at all.

## Usage

```python
import torch
from basicvsrpp_rocm import BasicVSRPlusPlusEngine

engine = BasicVSRPlusPlusEngine(
    checkpoint_path="weights/lada_mosaic_restoration_model_generic_v1.2.pth",
    device="cuda",
    max_clip_size=60,      # the temporal length you will feed most often
    fp16=True,
    temporal_overlap=8,    # frames of overlap between clips (2nd graph length)
)
out = engine.raw_process(frames)   # frames: list[(3, 256, 256) RGB uint8, 0..255]
# out: (T, 3, 256, 256) float tensor on cuda, nominally in [0, 1]
engine.close()
```

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `BVR_HIP_GRAPH` | `1` | `0` disables graphs, everything runs eager |
| `BVR_HIP_GRAPH_LENGTHS` | empty | extra clip lengths to capture, comma separated (e.g. `52,40`) |
| `BVR_VRAM_LIMIT_GB` / `BVR_VRAM_FRACTION` | VRAM − 500 MB | allocator budget |
| `BVR_FORCE_MATH_SDP` | empty | force the MATH attention backend (troubleshooting) |

## Lessons learned (the important part)

1. **HIP graphs are the biggest lever, with two hard constraints**: static
   shapes and a quiet GPU. Capturing while other threads submit GPU work
   poisons those calls ("operation not permitted when stream is capturing"
   kills the decode/encode threads on this ROCm build). **Capture once, at
   construction time, before any worker thread exists.**
2. **Capture only the dominant lengths**: real runs concentrate on
   `max_clip_size` and `max_clip_size − 2×overlap`; those two cover virtually
   every clip. Anything else (tail clips, scene cuts) runs eager — **~1.8×
   slower per frame**. Add lengths via `BVR_HIP_GRAPH_LENGTHS` only after a
   clip histogram says so (each length reserves its own activation memory).
3. **The eager forward before capture is not optional**: it doubles as the
   MIOpen kernel-selection warmup, which must happen outside the capture.
4. **ROCm's allocator env names differ from CUDA's**: set both
   `PYTORCH_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` (`PYTORCH_CUDA_ALLOC_CONF`
   is silently ignored on ROCm).
5. **`expandable_segments` must stay off**: it releases memory by unmapping
   address ranges, so an `empty_cache()` can pull pages out from under kernels
   another thread still has in flight → fp16 GEMM faults with "Page not
   present" (jasna issue #252). Default `expandable_segments:False`.
6. **Missing AOTriton images crash the process**: Windows ROCm wheels split the
   payload into a kernel package and an SDPA image package; with only the
   exact-arch package installed, the first FLASH launch submits a null kernel →
   `hipErrorInvalidImage` (ROCm/TheRock#7315). The engine checks image
   availability at startup and falls back to MATH explicitly (slow but correct),
   telling you to install e.g. `amd-torch-device-gfx110x`.
7. **Quantization is a net loss on this path**: fp8/int8 on the hot forward
   measured ~2× slower. Don't.
8. **Don't increase the batch**: larger inference batches hit pathological
   MIOpen kernels (upstream measured 16.5 fps at batch 2 → 0.24 fps at
   batch 4). Speed comes from graphs, not batching.
9. **VRAM budget**: the caching allocator only grows, so long jobs creep toward
   the whole card; once Windows demotes pages to shared memory (PCIe slow path)
   the frame rate collapses. `set_per_process_memory_fraction` makes the
   allocator reclaim its own cached blocks at the cap.
10. **MIOPEN_FIND_MODE=FAST**: spatial dims are fixed but the temporal length
    varies, so convolution shapes change with T; NORMAL mode re-tunes online.

## Repository layout

```
basicvsrpp-rocm/
├── basicvsrpp_rocm/
│   ├── engine.py            # the engine: fp16 + per-length HIP graph capture/replay
│   ├── rocm_env.py          # ROCm process defaults: env vars, SDPA check, VRAM budget
│   └── models/              # BasicVSR++ model definition (from mmagic / Lada, rebased imports)
│       ├── basicvsrpp_gan.py
│       ├── deformconv.py    # ModulatedDeformConv2d (the op TensorRT couldn't take)
│       ├── inference.py     # load_model (default inference config + checkpoint loading)
│       └── mmagic/          # vendored from open-mmlab/mmagic (SPDX headers kept)
├── benchmarks/bench_hipgraph.py   # bounded eager-vs-graph steady-state benchmark
├── tests/test_smoke.py            # real-weights smoke test + graph/eager equality
└── weights/README.md              # where to get the checkpoint (not included)
```

## Verifying

```bash
# smoke tests (need a GPU + weights; graph and eager output must be torch.equal)
python -m pytest tests -q

# benchmark: graph vs eager
python benchmarks/bench_hipgraph.py --weights weights/lada_mosaic_restoration_model_generic_v1.2.pth
python benchmarks/bench_hipgraph.py --weights weights/... --no-graph   # control run
```

## Requirements

- torch ≥ 2.4 (ROCm build; measured on torch 2.14.0+rocm10.1 / Windows / gfx1100)
- mmengine, numpy, torchvision (see `requirements.txt`)
- an AOTriton image package (e.g. `amd-torch-device-gfx110x`), or attention
  falls back to MATH

## Provenance & license

- `models/mmagic/**` is vendored from [open-mmlab/mmagic](https://github.com/open-mmlab/mmagic)
  (BasicVSR++ network); original SPDX headers kept (Apache-2.0 / AGPL-3.0).
- `models/inference.py` and the checkpoint format come from
  [Lada](https://github.com/qwekaze/lada) (AGPL-3.0); original SPDX headers kept.
- `engine.py` / `rocm_env.py` were extracted and adapted from the AMD build of
  [jasna](https://github.com/Kruk2/jasna).

Accordingly this repository is released under **AGPL-3.0**.

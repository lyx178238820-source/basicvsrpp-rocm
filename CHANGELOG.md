# Changelog

## 0.1.1 — 2026-10-10

### Added

- `CHANGELOG.md` (this file).
- Upstream contribution: the mmcv → torchvision deform-conv fallback shipped in
  this engine was upstreamed to OpenMMLab/mmagic as
  [open-mmlab/mmagic#2168](https://github.com/open-mmlab/mmagic/pull/2168)
  ("BasicVSR++: fall back to torchvision.ops.deform_conv2d when mmcv ops are
  unavailable", 1 file, +61/−1). Referenced from the README.

### Documented — companion project status

The engine itself is unchanged in this release. These notes record the state
of the jasna AMD build it was extracted from (its full test suite: **2494
tests, one session, all green on an RX 7900 XT**, 4 min 50 s):

- Fixed a silent-failure bug: decode-thread errors were re-raised into
  `threading.excepthook` instead of the pipeline's error channel, so a failed
  decode ended the job "normally" and wrote a truncated output file
  (`jasna/pipeline_threads.py`; errors now land in `error_holder`, which the
  parent re-raises).
- Guarded a division-by-zero crash in the streaming encoder's fps log line
  (`jasna/streaming_pipeline.py`).
- `--min-detection-duration` default returned 4 → 2 (the GUI slider and
  tooltips had been left inconsistent with it), so real 2–3-frame mosaics are
  no longer skipped.
- A new full portable build (2026-10-10) was cut from that state, carrying the
  per-GPU wheels: gfx1100 (exact) + gfx110x (family attention images) +
  gfx1201 (RDNA4) + `rocm_sdk_libraries`; other architectures (RX 6000
  gfx1030–1036, iGPUs gfx1103/115x) self-install their exact-arch wheels via
  `setup-rocm101-gpu.ps1`.

## 0.1.0 — 2026-10-09

- Initial public release:
  - `BasicVSRPlusPlusEngine` — fp16 + per-length HIP graph capture; measured
    10.3 ms/frame eager → 5.0 ms/frame graph at T=60/256²/fp16 (2.06×),
    bit-identical output;
  - `rocm_env` — MIOpen / allocator / AOTriton process defaults, SDPA
    availability check with explicit MATH fallback, VRAM budget;
  - vendored BasicVSR++ model package (mmagic / Lada) with rebased imports —
    the deformable convolution runs on `torchvision.ops.deform_conv2d`, which
    ships in torchvision's ROCm wheels;
  - bounded eager-vs-graph benchmark and smoke tests.

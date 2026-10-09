# -*- coding: utf-8 -*-
"""ROCm process defaults for the BasicVSR++ engine.

Extracted from the AMD build of jasna (accelerator.py), where these settings
were tuned on an RX 7900 XT / gfx1100 with torch 2.14+rocm10.1 on Windows.
Every knob can be overridden through the environment; the defaults are simply
setdefault calls.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import MutableMapping

import torch

logger = logging.getLogger(__name__)

#: NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
#: spatial dimensions but a variable temporal clip length (and therefore
#: variable effective convolution batches), so FAST avoids repeated runtime
#: profiling while still using MIOpen's system/user performance databases.
#:
#: Expandable segments release memory by unmapping virtual address ranges
#: rather than by a device-synchronizing free, so an empty_cache() could pull
#: pages out from under kernels another thread still had in flight - caught as
#: a restorer fp16 GEMM faulting with "Page not present" (jasna issue #252).
#: The env names cover the torch versions in the field; whichever the build
#: reads wins.
#:
#: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL opts in to the AOTriton
#: flash-attention kernels on Radeon parts that PyTorch ships as "not
#: officially validated". It is what makes FLASH/EFFICIENT SDPA usable on the
#: consumer cards this targets (see aotriton_image_dir below).
def apply_rocm_env_defaults(environ: MutableMapping[str, str]) -> None:
    environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


# --- AOTriton flash-attention availability -----------------------------------
#
# Windows ROCm wheels split the AOTriton payload across two device packages and
# neither one is sufficient on its own:
#
#   amd-torch-device-<exact arch>    -> torch/.kpack/torch_<arch>.kpack
#                                       (the torch kernels)
#   amd-torch-device-<arch family>   -> torch/lib/aotriton.images/<family>/...
#                                       (the SDPA kernel images)
#
# With only the exact-arch package installed the images are absent and the
# first FLASH/EFFICIENT SDPA launch submits a null kernel:
# hipModuleLaunchKernel is called with a null function pointer and the process
# dies with hipErrorInvalidImage (ROCm/TheRock#7315). Disabling those two
# backends keeps the run alive on the numerically identical MATH backend, but
# it is much slower - on an attention-bound model MATH costs 38.3 ms/frame
# against 18.8 ms/frame with FLASH at 576x576 (RX 7900 XT, gfx1100). So the
# backends are only switched off when the images for *this* GPU are really
# missing.
_AOTRITON_IMAGES_DIRNAME = "aotriton.images"
_AOTRITON_IMAGE_GLOB = "*/flash/attn_fwd.zip"
BVR_FORCE_MATH_SDP_ENV = "BVR_FORCE_MATH_SDP"
_TRUTHY = {"1", "true", "yes", "on"}

#: Headroom subtracted from the card's VRAM when deriving the default budget.
_VRAM_DEFAULT_HEADROOM_BYTES = 500 * 1024 * 1024
BVR_VRAM_LIMIT_GB_ENV = "BVR_VRAM_LIMIT_GB"
BVR_VRAM_FRACTION_ENV = "BVR_VRAM_FRACTION"


def _image_sets(root: Path) -> list[Path]:
    """Image directories under ``root`` that carry a complete flash kernel set."""
    if not root.is_dir():
        return []
    return sorted({marker.parent.parent for marker in root.glob(_AOTRITON_IMAGE_GLOB)})


def _device_gcn_arch() -> str:
    """``gfxNNNN`` for device 0, or an empty string when it cannot be read."""
    try:
        name = str(torch.cuda.get_device_properties(0).gcnArchName)
    except Exception:  # pragma: no cover - depends on the runtime
        return ""
    return name.split(":")[0].strip()


def aotriton_image_dir(lib_dir: Path | str | None = None, *, arch: str | None = None) -> Path | None:
    """AOTriton image directory usable by this GPU, or ``None`` when there is none.

    Image sets are named after the architecture family, so a ``gfx1100`` device
    is served by ``amd-gfx110x`` and a ``gfx1201`` by ``amd-gfx120x``.
    """
    if lib_dir is None:
        lib_dir = Path(torch.__file__).parent / "lib"
    lib_dir = Path(lib_dir)
    sets = _image_sets(lib_dir / _AOTRITON_IMAGES_DIRNAME)
    if not sets:
        return None
    if len(sets) == 1:
        return sets[0]
    arch = arch or _device_gcn_arch()
    for candidate in sets:
        if arch and arch.replace("gfx", "gfx") in candidate.name:
            return candidate
    return sets[0]


def _math_sdp_forced() -> bool:
    return os.environ.get(BVR_FORCE_MATH_SDP_ENV, "").strip().lower() in _TRUTHY


def configure_sdpa_backends() -> None:
    """Keep FLASH/EFFICIENT SDPA usable, or fall back to MATH explicitly."""
    if not torch.version.hip:
        return
    if _math_sdp_forced():
        logger.info("%s is set: using the MATH SDPA backend", BVR_FORCE_MATH_SDP_ENV)
    elif aotriton_image_dir() is not None:
        return
    else:
        logger.warning(
            "No AOTriton SDPA images for this GPU were found, so FLASH/EFFICIENT "
            "attention cannot be launched (ROCm/TheRock#7315). Falling back to the "
            "MATH backend: correct but noticeably slower. Install the image package "
            "for this architecture, e.g. 'amd-torch-device-gfx110x' for gfx1100/1101/"
            "1102/1103, to get the fast path back."
        )
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def apply_vram_budget(
    device_index: int = 0,
    *,
    headroom_bytes: int = _VRAM_DEFAULT_HEADROOM_BYTES,
) -> None:
    """Cap the CUDA caching allocator so long jobs stop ballooning.

    The allocator caches every freed block and never returns VRAM on its own,
    so a long job creeps toward the whole card; once Windows starts demoting
    pages to shared memory (PCIe slow path) the frame rate collapses.
    ``set_per_process_memory_fraction`` makes the allocator reclaim its own
    cached blocks at the cap instead of growing past it.

    Default budget: the card's VRAM minus ``headroom_bytes`` (500 MB).
    Override with ``BVR_VRAM_LIMIT_GB`` (absolute GB) or ``BVR_VRAM_FRACTION``
    ((0-1], relative to the card).
    """
    if not torch.cuda.is_available():
        return
    try:
        total = torch.cuda.get_device_properties(device_index).total_memory
        if not total:
            return
        limit = os.environ.get(BVR_VRAM_LIMIT_GB_ENV, "").strip()
        frac_env = os.environ.get(BVR_VRAM_FRACTION_ENV, "").strip()
        if limit:
            fraction = float(limit) * (1024 ** 3) / total
        elif frac_env:
            fraction = float(frac_env)
        else:
            fraction = (total - headroom_bytes) / total
        if not 0.05 <= fraction <= 0.99:
            logger.warning(
                "VRAM budget %s resolves to fraction %.2f, outside 0.05-0.99; ignored",
                limit or frac_env or "default", fraction,
            )
            return
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        try:
            name = torch.cuda.get_device_name(device_index)
        except Exception:  # noqa: BLE001
            name = "GPU"
        logger.info(
            "VRAM budget: device %d (%s) capped at %.1f GB of %.1f GB (%.0f%%); "
            "tune with %s / %s",
            device_index, name,
            total * fraction / (1024 ** 3), total / (1024 ** 3), fraction * 100,
            BVR_VRAM_LIMIT_GB_ENV, BVR_VRAM_FRACTION_ENV,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM budget not applied: %s", exc)


def release_vram_cache() -> None:
    """Return every cached-but-unused block to the driver (defragmentation).

    Live allocations are untouched; only the allocator's cold cache is
    released, which is what otherwise grows until Windows starts demoting the
    hot pages to shared memory and the frame rate collapses.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def configure_process(device: torch.device | str | None = None) -> None:
    """One-call process setup: env defaults, SDPA check, VRAM budget."""
    if torch.version.hip:
        apply_rocm_env_defaults(os.environ)
    configure_sdpa_backends()
    index = 0
    if device is not None and torch.device(device).type == "cuda":
        index = torch.device(device).index or 0
    apply_vram_budget(index)

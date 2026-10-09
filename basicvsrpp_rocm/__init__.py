# -*- coding: utf-8 -*-
"""BasicVSR++ on AMD ROCm - the engine that replaced TensorRT.

Public interface:
    BasicVSRPlusPlusEngine  - load a checkpoint, restore clips
    rocm_env                - ROCm process defaults (env, SDPA, VRAM budget)
"""
from basicvsrpp_rocm.engine import BasicVSRPlusPlusEngine

__all__ = ["BasicVSRPlusPlusEngine"]
__version__ = "0.1.0"

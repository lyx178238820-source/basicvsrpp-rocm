# Weights (not included in this repo)

The engine loads a BasicVSR++ GAN checkpoint produced by the Lada training
pipeline (`lada_mosaic_restoration_model_generic_v1.2.pth`, 74.8 MB, AGPL-3.0
licensed by the Lada authors). The checkpoint is **not** redistributed here.

- Project: <https://codeberg.org/ladaapp/lada>
- Checkpoint: <https://huggingface.co/ladaapp/lada>
  (`lada_mosaic_restoration_model_generic_v1.2.pth`)

Download it and drop it in this folder (or anywhere — you pass the path
explicitly).

```python
from basicvsrpp_rocm import BasicVSRPlusPlusEngine

engine = BasicVSRPlusPlusEngine(
    checkpoint_path="weights/lada_mosaic_restoration_model_generic_v1.2.pth",
    device="cuda",
    max_clip_size=60,     # the temporal length you will feed most often
    fp16=True,
    temporal_overlap=8,
)
out = engine.raw_process(frames)   # list of (3, 256, 256) RGB, [0, 255]
```

The default config (`basicvsrpp_rocm/models/inference.py::get_default_gan_inference_config`)
matches that checkpoint: `BasicVSRPlusPlusGan` generator with `mid_channels=64`,
`num_blocks=15`. A different stage-1/stage-2 checkpoint can be selected by
passing an mmengine config file or dict via `config=`.

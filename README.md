# BasicVSR++ on AMD ROCm —— 替代 TensorRT 引擎的独立推理引擎

在 A 卡（ROCm）上跑通并提速 **BasicVSR++** 视频修复模型的完整引擎：NVIDIA 侧上游方案是把 BasicVSR++ 切成 **TensorRT 子引擎**，而 A 卡没有 TensorRT（可变形卷积等算子也无法导出到 MIGraphX）。本仓库是替代路径的**全部源代码与经验**：纯 PyTorch ROCm 前向 + 一组针对性优化，在 RX 7900 XT 上实测把速度追平甚至超过 N 卡子引擎方案的可用水平。

> 本仓库是独立引擎，**不包含任何宿主应用代码**（无流水线/GUI/检测器），也不含权重文件。

## 实测数据（RX 7900 XT / gfx1100, torch 2.14.0+rocm10.1, Windows）

| 手段 | 效果 |
|---|---|
| **HIP graph 捕获整个前向** | 60 帧 256² 片段 **534 → 292 ms（1.83×）**，输出位一致；本仓库基准复测 **eager 10.3 → graph 5.0 ms/帧（2.06×，199 fps）** |
| fp16 权重+输入 | ~2× 吞吐，无可感知画质损失 |
| AOTriton FLASH SDPA | 注意力绑定模型 **38.3 → 18.8 ms/帧** @576² |
| 显存预算（防溢出） | 消除 Windows 共享内存回退导致的掉速（溢出后 1.2 帧/秒 → 稳态 5.8+） |
| MIOPEN_FIND_MODE=FAST | 避免每个新卷积形状重复调优（T 可变 → 卷积 batch 可变） |

嵌入完整流水线（解码→检测→修复→编码）后的稳态吞吐：4K 源 **5.8~5.9 帧/秒**（x4plus 二次修复， fp16）。

## 为什么是"纯 PyTorch 前向 + HIP graph"，而不是 TensorRT

- TensorRT 子引擎是 NVIDIA 专属，A 卡没有对应运行时；
- BasicVSR++ 的核心算子是**可变形卷积**（`ModulatedDeformConv2d`，本仓库 `models/deformconv.py`），MIGraphX/ONNX 对它的支持不完整，导出路线走不通；而 PyTorch 路线里它实际落在 `torchvision.ops.deform_conv2d` —— **torchvision 的 ROCm 轮子自带该算子的实现**，这正是整条纯 PyTorch 路线在 A 卡上能跑通的关键；
- PyTorch 前向的瓶颈不是算力而是 **launch/dispatch**：一次 60 帧片段的前向约 **15 万个小 kernel**。HIP graph 把整段前向录制成一次重放，直接消掉这部分开销 —— 这是在 A 卡上能拿到的最大单项收益，且**输出位一致**。

## 使用

```python
import torch
from basicvsrpp_rocm import BasicVSRPlusPlusEngine

engine = BasicVSRPlusPlusEngine(
    checkpoint_path="weights/lada_mosaic_restoration_model_generic_v1.2.pth",
    device="cuda",
    max_clip_size=60,      # 你最常喂的片段长度（graph 按它捕获）
    fp16=True,
    temporal_overlap=8,    # 相邻片段重叠帧数（决定第二个 graph 长度）
)
out = engine.raw_process(frames)   # frames: list[(3, 256, 256) RGB uint8, 0..255]
# out: (T, 3, 256, 256) float tensor, 0..1，位于 cuda 上
engine.close()
```

环境变量（全部可选）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `BVR_HIP_GRAPH` | `1` | `0` 关闭 graph，全部 eager |
| `BVR_HIP_GRAPH_LENGTHS` | 空 | 额外捕获的片段长度，逗号分隔（如 `52,40`） |
| `BVR_VRAM_LIMIT_GB` / `BVR_VRAM_FRACTION` | 显存−500MB | 分配器预算 |
| `BVR_FORCE_MATH_SDP` | 空 | 强制 MATH 注意力后端（排障用） |

## 经验与踩坑（重点）

1. **HIP graph 是最大杠杆，但有两个硬约束**：静态形状 + 安静 GPU。在流水线其它线程提交 GPU 任务时捕获，会以 "operation not permitted when stream is capturing" 毒化那些线程（ROCm 构建上直接弄死解码/编码线程）。所以**必须在任何工作线程启动前、构造时捕获**。
2. **只捕主导长度**：整段运行的片段长度高度集中（`max_clip_size` 与 `max_clip_size − 2×overlap` 两种覆盖几乎全部片段）。其它长度（结尾零头、切镜头）走 eager，**每帧慢约 1.8×**；有片段直方图后再用 `BVR_HIP_GRAPH_LENGTHS` 补捕，别乱加（每长度多占一份激活显存）。
3. **捕获前的 eager 前向不能省**：它兼作 MIOpen kernel 选择的预热，捕获必须在选型完成之后。
4. **ROCm 的分配器变量名与 CUDA 不同**：要同时设 `PYTORCH_ALLOC_CONF` 和 `PYTORCH_HIP_ALLOC_CONF`（CUDA 名 `PYTORCH_CUDA_ALLOC_CONF` 在 ROCm 上不生效）。
5. **`expandable_segments` 必须关**：它以取消映射方式释放内存，`empty_cache()` 会把别的线程还在用的页抽走 → fp16 GEMM 崩 "Page not present"（jasna issue #252）。默认 `expandable_segments:False`。
6. **AOTriton 图像缺失会直接崩进程**：Windows ROCm 轮子把 kernel 与 SDPA 图像分装在两个包里，只装精确定架构包时 FLASH 启动会提交空 kernel → `hipErrorInvalidImage`（ROCm/TheRock#7315）。本引擎启动时检测图像是否存在，缺失则显式回退 MATH（慢但正确），并提示装 `amd-torch-device-gfx110x` 这类图像包。
7. **量化在这条路上是负收益**：对前向热循环做 fp8/int8 实测慢近 2×，不要做。
8. **别加大 batch**：更大的推理 batch 会命中 MIOpen 的病态 kernel（上游实测 batch 2→4 从 16.5 fps 掉到 0.24 fps 的先例）。速度靠 graph，不靠 batch。
9. **显存预算**：分配器只增不减，长任务会爬满整卡；一旦 Windows 开始把页挤到共享内存（PCIe 慢路径）帧率崩塌。`set_per_process_memory_fraction` 让分配器到顶就回收自己的缓存块。
10. **MIOPEN_FIND_MODE=FAST**：空间尺寸固定、时间长度可变 → 卷积形状随 T 变化，NORMAL 模式会反复在线调优。

## 目录结构

```
basicvsrpp-rocm/
├── basicvsrpp_rocm/
│   ├── engine.py            # 引擎：fp16 + HIP graph 捕获/重放（对外接口）
│   ├── rocm_env.py          # ROCm 进程默认：env 变量、SDPA 可用性检查、显存预算
│   └── models/              # BasicVSR++ 模型定义（源自 mmagic / Lada，导入已独立）
│       ├── basicvsrpp_gan.py
│       ├── deformconv.py    # ModulatedDeformConv2d（可变形卷积，TRT 导不走的那个算子）
│       ├── inference.py     # load_model（默认推理配置 + checkpoint 加载）
│       └── mmagic/          # vendored 自 open-mmlab/mmagic（保留其 SPDX 头）
├── benchmarks/bench_hipgraph.py   # eager vs graph 有界稳态基准
├── tests/test_smoke.py            # 真实权重冒烟 + graph/eager 位一致断言
└── weights/README.md              # 权重放哪（本仓库不含权重）
```

## 验证

```bash
# 冒烟（需要 GPU + 权重；graph 与 eager 输出必须 torch.equal）
python -m pytest tests -q

# 基准：graph vs eager
python benchmarks/bench_hipgraph.py --weights weights/lada_mosaic_restoration_model_generic_v1.2.pth
python benchmarks/bench_hipgraph.py --weights weights/... --no-graph   # 对照组
```

## 环境

- torch ≥ 2.4（ROCm 构建；实测 torch 2.14.0+rocm10.1 / Windows / gfx1100）
- mmengine、numpy、torchvision（`requirements.txt`）
- AOTriton 图像包（如 `amd-torch-device-gfx110x`），否则注意力回退 MATH

## 来源与许可

- `models/mmagic/**` vendored 自 [open-mmlab/mmagic](https://github.com/open-mmlab/mmagic)（BasicVSR++ 网络），保留原 SPDX 头（Apache-2.0 / AGPL-3.0）。
- `models/inference.py` 与权重格式来自 [Lada](https://github.com/qwekaze/lada)（AGPL-3.0），保留原 SPDX 头。
- `engine.py` / `rocm_env.py` 从 jasna 的 AMD 构建中提取改写（jasna 上游：[Kruk2/jasna](https://github.com/Kruk2/jasna)）。

本仓库按上游许可要求整体以 **AGPL-3.0** 释出。

# gsplat → Apple Silicon (MPS/Metal) 训练迁移：任务规格与分工

> 目标读者：DeepSeek Harness 中负责实现代码的 agent（flash / v4.1-1017）。
> 本文档是唯一的事实来源（single source of truth）：实现者按第 5、6 节的契约写代码，按第 7 节验收，按第 9 节在 Mac 上验证。评审者（v4-pro）按同一份契约 review。

---

## 1. 项目背景与目标

为摆脱对 NVIDIA CUDA 云 GPU 的长期依赖（受制于人 + 未来做 3DGS 游戏），自研一个跑在 Apple Silicon 上的 3DGS 训练渲染器。

本次（Milestone 0）的唯一目标：让 gsplat 的纯 PyTorch 渲染路径在 Mac MPS 后端上端到端跑通训练。正确性优先，不追求速度。速度留给后续的 Metal 里程碑。

## 2. 关键发现（读源码结论）

gsplat（nerfstudio-project/gsplat，Apache 2.0 商用友好）内部有两条渲染路径：

- rasterization()（gsplat/rendering.py:234）→ 走 CUDA 融合 kernel，快但 CUDA-only。
- _rasterization()（gsplat/rendering.py:722）→ 走 gsplat/cuda/_torch_impl.py 的纯 PyTorch 实现，靠 PyTorch autograd 反传，无需手写 backward。官方 docstring 明说这条路径可作为调试新想法的 playground，无需 backward 实现。

纯 torch 路径的组件清单：

| 组件 | 位置（_torch_impl.py） | 实现 | MPS 可用 |
|---|---|---|---|
| _fully_fused_projection | 262 行 | 纯 torch | ✅ |
| _isect_tiles | 356 行 | 纯 torch（@torch.no_grad） | ✅ |
| _isect_offset_encode | 455 行 | 纯 torch | ✅ |
| _rasterize_to_pixels | 814 行 | 纯 torch，但内部调 CUDA rasterize_to_indices_in_range | ❌ |
| accumulate | 713 行 | 纯 torch，但内部调 nerfacc | ❌ |

结论：堵住 MPS 的只有两处 CUDA 依赖。补齐这两处，_rasterization() 就能在 Mac 上端到端训练。这正是本次要做的两件事（Task A / Task B）。

## 3. 环境信息

- 目标机器：Mac M2 Pro，16GB 统一内存，Apple Silicon（Metal GPU）
- 运行时：PyTorch，MPS 后端（torch.device("mps")）
- 参考代码：gsplat repo（commit 512d366，Apache 2.0），路径 gsplat/
- 关键依赖：_torch_impl.py 当前 import 了 nerfacc 和 ._wrapper（CUDA），本里程碑的目标是去掉这两个依赖

## 4. 迁移策略（三里程碑）

- M0（本次）：纯 torch 跑通 MPS。补齐 Task A + Task B，_rasterization() 在 Mac 上能训练。慢但正确，且是后续 Metal 提速的数值基准。
- M1：profile 热点，把投影 + 光栅化 forward 换成 Metal kernel 提速。
- M2：补 Metal backward，达到可用训练速度。

先完成 M0，再谈 M1/M2。任何时刻手里都要有一个能训练的系统。

## 5. Milestone 0 任务分解

### Task A：纯 torch 重写 rasterize_to_indices_in_range

它是什么：把「tile → 高斯」的索引（isect_offsets + flatten_ids）展开成「(高斯 id, 像素 id, 图像 id)」三元组，供 alpha 合成用。索引簿记，不涉及重数学。

CUDA 参考：
- 入口 gsplat/cuda/_wrapper.py:2488（rasterize_to_indices_in_range，实际调 torch.ops.gsplat.rasterize_to_indices_3dgs）
- kernel gsplat/cuda/csrc/RasterizeContributingGaussianIds.cu、RasterizeNumContributingGaussians.cu、RasterizeContributingCommon.cuh
- 纯 torch 侧调用点 gsplat/cuda/_torch_impl.py:887

实现要点：
- 输入里 isect_ids 已经在 _isect_tiles 阶段按深度排序（depth 编码在 64 位整数里），所以三元组的输出顺序天然是「近到远」，下游 accumulate 直接依赖这个顺序，不要破坏它。
- 覆盖判定：像素中心落在高斯的 2D 椭圆内（用 conics 计算 Mahalanobis 距离，阈值对应 3σ 左右；精确阈值以对齐 CUDA 输出为准）。
- transmittances 用于提前终止：像素已不透明（透射率≈0）时跳过后续高斯（这是优化项；若为保正确性可不做提前终止，但输出会多出本可跳过的三元组，需与验收标准对齐后决定）。

### Task B：纯 torch 重写 alpha 合成（去 nerfacc）

它是什么：把 accumulate 里调 nerfacc.accumulate_along_rays + render_weight_from_alpha 的部分换成纯 torch。

参考：gsplat/cuda/_torch_impl.py:713 的 accumulate 函数。它已经把 sigmas、alphas（opacity * exp(-sigma)）在纯 torch 里算好了，只把「按像素分组的前到后合成」委托给了 nerfacc。

实现要点：
- 合成公式：weight_i = T_i * α_i，其中 T_i = ∏_{j<i}(1-α_j)（前到后，同一像素内按 depth 排序）。
- 纯 torch 实现：torch.sort 按 (image_id, pixel_id, depth) 分组 → 组内 torch.cumprod(1-α) 得 T → torch.scatter_add 或等价操作做分段加权求和。
- 注意保持与 nerfacc accumulate_along_rays 完全一致的数值语义（浮点累加顺序可允许 1e-4 级误差，见验收标准）。

## 6. 接口契约（实现者必须严格遵守）

### 契约 A

    def rasterize_to_indices_in_range_torch(
        range_start: int,          # 起始 tile 批（含）
        range_end: int,            # 结束 tile 批（不含）
        transmittances: Tensor,    # [..., H, W] float32 — 当前透射率 = 1 - 累计 alpha
        means2d: Tensor,           # [..., N, 2] float32 — 投影后高斯中心（像素坐标）
        conics: Tensor,            # [..., N, 3] float32 — 逆 2D 协方差上三角 [c0, c1, c2]
        opacities: Tensor,         # [..., N] float32 — 不透明度
        image_width: int,
        image_height: int,
        tile_size: int,
        isect_offsets: Tensor,     # [..., tile_h, tile_w] int32 — 每 tile 在 flatten_ids 中的起始偏移
        flatten_ids: Tensor,       # [n_isects] int32 — 扁平索引 = image_id * N + gauss_id
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # 返回 (gs_ids [M] int64, pixel_ids [M] int64 row-major, image_ids [M] int64)

### 契约 B

    def accumulate_torch(
        means2d: Tensor,       # [..., N, 2]
        conics: Tensor,        # [..., N, 3]
        opacities: Tensor,     # [..., N]
        colors: Tensor,        # [..., N, channels]
        gaussian_ids: Tensor,  # [M]
        pixel_ids: Tensor,     # [M] row-major
        image_ids: Tensor,     # [M]
        image_width: int,
        image_height: int,
    ) -> Tuple[Tensor, Tensor]:
        # 返回 (renders [..., H, W, channels], alphas [..., H, W, 1])

约定：
- 输入 tensor 可能带任意 batch 维度（...），实现要兼容 I = prod(...) 张图。
- dtype 统一 float32 / int32，索引用 int64。
- 所有实现必须是纯 torch op（torch.*），禁止 .cuda()、torch.ops.gsplat.*、nerfacc 调用。
- 输出放到输入同一 device（MPS 上就是 mps）。

## 7. 验收标准

- Task A：与 CUDA rasterize_to_indices_3dgs 输出对齐——gs_ids / pixel_ids / image_ids 三个索引张量完全一致（这是确定性整数操作，不是浮点）。
- Task B：与 nerfacc 版本对齐——renders / alphas 相对误差 ≤ 1e-4（浮点累加顺序允许的误差）。
- 端到端：_rasterization() 在 MPS 上跑通一次前向 + 反传，loss 正常下降，无 NaN。
- 测试数据：用随机初始化的小场景（如 N=1024 高斯、256×256 图像、单相机），先生成 CUDA 参考输出（在有 CUDA 的环境或 CI 上），再在 MPS 上比对。

## 8. 分工

| 角色 | 负责 | 产出 |
|---|---|---|
| flash (v4.1-1017) | 实现 Task A + Task B 的纯 torch 代码 | 两个函数 + 对齐测试脚本 |
| v4-pro (评审) | 契约把关、代码 review、数值对齐验证、集成 | 通过的验收结论 / 返工意见 |

给 flash 的执行要求：
- 先读 _torch_impl.py 的 _rasterize_to_pixels（814 行起）和 accumulate（713 行起），理解数据流再动手。
- 优先写正确的、可读的纯 torch 版本，不追求性能（M0 是正确性里程碑）。
- 每写完一个函数，立即用对齐测试验证（见第 9 节），通过再交 review，避免返工。

## 9. 验证流程（在 Mac 上）

1. 确保 gsplat 已 clone 到 Mac 本地，PyTorch 装了 MPS 后端（torch.backends.mps.is_available() 为 True）。
2. 新建实现文件（建议 gsplat/cuda/_torch_impl_mps.py），放 Task A/B 两个纯 torch 函数。
3. 在 _torch_impl.py 里做最小侵入式替换：把 from ._wrapper import rasterize_to_indices_in_range 换成 MPS 版，把 accumulate 里的 nerfacc 调用换成 accumulate_torch（或用 monkey-patch / 环境变量开关，不改 CUDA 主路径）。
4. 写一个 test_mps.py：随机小场景 → 走 _rasterization() → 前向得 (render_colors, render_alphas) → 反传 → 检查梯度非空、无 NaN、loss 可下降。
5. 若本机无 CUDA 做参考比对，则至少满足「端到端跑通 + 无 NaN + loss 下降」；有 CUDA 参考时再补精确数值对齐。

## 10. 参考文件清单

- 纯 torch 路径：gsplat/cuda/_torch_impl.py（重点 262/356/455/713/814 行）
- CUDA 索引 kernel：gsplat/cuda/csrc/RasterizeContributingGaussianIds.cu 等
- CUDA wrapper：gsplat/cuda/_wrapper.py:2488
- 高层入口：gsplat/rendering.py:722（_rasterization）
- 数学工具：gsplat/cuda/_math.py:689（_quat_scale_to_covar_preci，纯 torch，无需改）

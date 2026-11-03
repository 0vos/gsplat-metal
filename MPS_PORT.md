# gsplat → Apple Silicon (MPS) 训练移植说明

本仓库把 [gsplat](https://github.com/nerfstudio-project/gsplat)（Apache-2.0，commit `512d366`，v1.6.0）
的**纯 PyTorch 渲染路径**移植到了 Apple Silicon 的 MPS 后端，并在 MacBook Pro (M2 Pro, 16GB) 上
用真实多视角数据（NeRF-synthetic `lego`）和 Blender 渲染的人体多视角数据完成了端到端训练。

- 不需要 NVIDIA CUDA
- 不需要编译 gsplat 的 C++/CUDA 扩展（`csrc.so`）
- 不需要 `nerfacc`
- 训练、反向传播、致密化、评估、导出 PLY 全部在 `torch.device("mps")` 上完成

---

## 1. 结论

| 运行 | 分辨率 | 步数 | 高斯数 | 测试 PSNR | 耗时 |
|---|---|---|---|---|---|
| `results/lego_100` | 100×100 | 600 | 27,080 | **23.70 dB** | 22.6 min |
| `results/lego_200` | 200×200 | 600（计划 1000，主动停在 checkpoint） | 35,746 | **22.77 dB** | ~25 min |
| `results/human_200` | 200×200 | 500 | 20,328 | **39.05 dB** | 15.3 min |

`results/*/renders/final_view*.png` 是「左 GT / 右 渲染」的并排图，`comparison.png` 是 4 个测试视角的拼图，
`turntable.gif` 是绕物体 360° 的新视角旋转（CPU 渲出来的，可直接查看高斯体的 3D 一致性）。

重建结果（左：GT，右：训练出的高斯渲染）：

- lego 200×200：`results/lego_200/comparison.png`、`results/lego_200/turntable.gif`
- 人体 200×200：`results/human_200/comparison.png`、`results/human_200/turntable.gif`

即使在这个很短的训练预算下，多视角图像也已经收敛成一个结构完整、可自由视角渲染的高斯体
（`results/*/point_cloud.ply`，可直接用任意 3DGS 查看器打开，例如 https://superspl.at/editor）。

---

## 2. 代码改动

### 2.1 新增：`gsplat/gsplat/cuda/_torch_impl_mps.py`

| 函数 | 对应 CUDA 实现 | 说明 |
|---|---|---|
| `rasterize_to_indices_in_range_torch` | `RasterizeToIndices3DGSSerialBatch.cu` | Task A。把 tile 的 `(isect_offsets, flatten_ids)` 展开成 `(gaussian, pixel, image)` 三元组。严格复刻 kernel 语义：`sigma >= 0`、`alpha >= 1/255`、`T_next <= 1e-4` 时提前终止（且排除当前高斯）、按像素分组、组内近到远。 |
| `accumulate_torch` | `nerfacc.render_weight_from_alpha` + `accumulate_along_rays` | Task B。纯 torch 前到后 alpha 合成 `weight_i = alpha_i · Π_{j<i}(1-alpha_j)`，支持任意 batch 维、任意通道数。 |
| `_isect_tiles_fast` | `isect_tiles`（CUDA）| tile 枚举的向量化纯 torch 版本。原 `_torch_impl._isect_tiles` 是 `I×N` 的 Python 双重循环，只适合做参考实现，训练时不可用。 |

实现细节：

- 索引生成按 `tile（每 16 个一组） × 深度批 × 像素` 分块向量化，只为「当前 tile 当前批」构造
  `[tiles, 256, 256]` 的临时张量，内存有上界；透射率状态按 tile 维护，跨批推进。
- 输出用 stable sort 按 `image*H*W + pixel` 排序，因此与 CUDA 的输出顺序（像素主序、组内近到远）一致。
- `accumulate_torch` 用「按像素分块的 padding + `cumprod`」计算分段累积积，避免除法下溢；
  对任意顺序的像素组都成立（组内顺序必须近到远，这一点由索引生成器保证）。

### 2.2 修改：`gsplat/gsplat/cuda/_torch_impl.py`

- `accumulate()`：非 CUDA 设备自动转到 `accumulate_torch`（CUDA 上仍走 nerfacc，行为不变）。
- `_rasterize_to_pixels()`：非 CUDA 设备自动改用 `rasterize_to_indices_in_range_torch`
  （CUDA 上仍走 `_wrapper` 的融合 kernel）。

### 2.3 修改：`gsplat/gsplat/rendering.py`

PLAN.md 只列了两处 CUDA 依赖，实际读源码还发现三处，一并处理：

1. `_rasterization()` 里的 `isect_tiles` / `isect_offset_encode` 来自 `_wrapper`（CUDA-only）。
   改用纯 torch 的 `_isect_tiles_fast` / `_isect_offset_encode`。
2. `_maybe_evaluate_sh()` 里的 `spherical_harmonics` 来自 `_wrapper`（CUDA-only）。
   非 CUDA 设备改为先用 viewmat 反解相机位置、算出视线方向，再调用
   `_torch_impl._spherical_harmonics`（与 CUDA kernel 的 dirs 约定一致）。
3. `rasterization()` 增加设备自动分发：非 CUDA 设备直接走 `_rasterization()`，并对 M0 不支持的
   特性（`with_ut`、`with_eval3d`、`rays`、lidar、鱼眼/畸变、rolling shutter、`absgrad`、
   `distributed`、`segmented`、`return_normals`、非零 `radius_clip`）抛出明确的 `NotImplementedError`。
   `packed` 参数被忽略（两种模式渲染结果等价）。

### 2.4 测试：`tests/test_mps_torch_impl.py`

没有 CUDA 环境，因此用**按 CUDA kernel 语义手写的慢速参考实现**做对齐：

- `slow_rasterize_to_indices_in_range`：逐 image/tile/pixel/batch 循环，完全照抄 kernel 逻辑。
  - 与 `rasterize_to_indices_in_range_torch` 的 `gs/pixel/image` 三个索引张量**逐元素完全一致**，
    包括分批调用（`range_start/range_end`）的情况。
  - 同时校验 `_isect_tiles_fast` 与原始 `_isect_tiles` 的 `tiles_per_gauss / isect_ids / flatten_ids`
    完全一致。
- `brute_force_render`：逐像素对全部高斯做前到后合成（不使用 tile、不使用分批）。
  - 与 `_rasterization()` 的 `renders/alphas` 最大误差 < `2e-5`（实测 ~`6e-8`）。
- MPS 与 CPU 的 `renders/alphas` 一致性（实测 ~`2e-6`）。
- SH（degree 2）+ autograd：MPS 与 CPU 的前向、以及 means/quats/scales/opacities/colors 的梯度
  相对误差 < `5e-3`（实测远小于）。

### 2.5 训练脚本：`examples/train_mps.py`

- 数据：NeRF-synthetic / Blender 格式（`transforms_train.json` + `train/*.png`，RGBA，白底合成）。
- 归一化：`similarity_from_cameras`（scene focus 居中、相机半径归一化）+ 相机旋转行再归一化。
- 初始化：多视角 alpha mask **visual hull** 体素雕刻（可选 `--init random`），在视觉外壳表面采样
  点并做 4 次多视图颜色平均；这比随机初始化收敛快很多（3DGS 原本依赖 COLMAP SfM 点）。
- 训练：`DefaultStrategy`（致密化/分裂/剪枝/透明度重置）、L1 + 0.2·SSIM、SH degree 逐步升高、
  Adam（普通版；CUDA 的 `fused=True` 在 MPS 上不可用）。
- 输出：`log.csv`、每次 eval 的 `renders/stepXXXX_viewYY.png`（GT | render 并排）、
  `point_cloud.ply`、`splats.pt`、`result.json`。

---

## 3. 踩过的坑（重要）

1. **相机归一化把旋转也缩放了**
   `examples/datasets/normalize.py::similarity_from_cameras` 返回的 4×4 矩阵把世界缩放写进了
   上 3×4 块，所以 `T @ c2w` 得到的相机旋转是「缩放过的旋转」，相机不再刚体。投影的 u/v 不受
   影响（等比缩放会抵消），但**相机空间深度会差 1/scale 倍**（lego 上约 4 倍），导致所有高斯
   投到图像上小 4 倍，训练几乎学不动。
   修复：对 `T @ c2w` 的旋转行再做归一化（等价于 gsplat 的 `transform_cameras`）。
   修复后同样的场景初始 PSNR 从 ~7 dB 提升到 ~22 dB。
2. **白底**
   NeRF-synthetic 的 PNG 带 alpha；训练和 eval 都必须用 `render + 1.0*(1-alpha)` 合成白底，
   否则空白区域是黑的，模型会学成「整体发灰」。
3. **PLAN 低估了 CUDA 依赖**
   除 Task A/B 外，`_rasterization()` 还调用了 CUDA 的 `isect_tiles` / `isect_offset_encode`，
   以及 CUDA 的 `spherical_harmonics`。这三处在 M0 里也必须替换（见 2.3）。

---

## 4. 运行方式

### 4.1 环境

任何带 MPS 后端的 PyTorch ≥ 2.4 均可（本机实测 torch 2.8.0 / Python 3.13 / macOS 26.7 / M2 Pro）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mps.txt
# 若已有带 torch 的环境，也可以：
# python3 -m venv --system-site-packages .venv
```

> 不需要 `pip install -e gsplat`（那会尝试编译 CUDA 扩展）。直接用 `PYTHONPATH` 指向仓库即可。

### 4.2 测试

```bash
PYTHONPATH=$PWD/gsplat .venv/bin/python tests/test_mps_torch_impl.py
```

### 4.3 训练

```bash
# 一键脚本（默认 200×200 / 2500 步）
./run_train_lego.sh

# 或者手动：
PYTHONPATH=$PWD/gsplat:$PWD/gsplat/examples .venv/bin/python examples/train_mps.py \
  --data_dir data/lego --out_dir results/lego_200 \
  --downscale 4 --steps 1400 --init_num_pts 30000 \
  --eval_every 200 --refine_start 200 --refine_stop 1100 \
  --refine_every 100 --reset_every 700 --sh_degree_interval 300 \
  --max_gaussians 150000
```

人体数据（灰底，200×200）：

```bash
PYTHONPATH=$PWD/gsplat:$PWD/gsplat/examples .venv/bin/python examples/train_mps.py \
  --data_dir data/human --out_dir results/human_200 --downscale 2 \
  --steps 500 --init_num_pts 20000 --init_extent 0.5 --hull_min_hits 4 \
  --bg 0.35 --empty_cache_every 1 --eval_every 250 --save_every 100 \
  --refine_start 100 --refine_stop 450 --refine_every 100 --reset_every 300 \
  --sh_degree_interval 150 --max_gaussians 60000
```

`data/lego` 是 NeRF-synthetic 的 `lego` 场景（50 张训练视图 / 20 张测试视图，800×800 RGBA），
由 `--downscale` 下采样（4 → 200×200）。

---

## 4.4 人体（人形）多视角数据集

`tools/render_human_dataset.py` 用 Blender（headless Cycles）把一个本地人形 FBX 模型渲染成
NeRF-synthetic 格式的多视角数据集（48 训练视角 + 12 测试视角，400×400，RGBA 透明背景），
再交给同一个 `train_mps.py` 训练：

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python tools/render_human_dataset.py -- \
  --fbx ~/Documents/Models/宫野志保/base_basic_pbr.fbx \
  --out data/human --res 400 --views 48 --test-views 12 --samples 24

PYTHONPATH=$PWD/gsplat:$PWD/gsplat/examples .venv/bin/python examples/train_mps.py \
  --data_dir data/human --out_dir results/human_200 --downscale 2 \
  --steps 1200 --init_num_pts 20000 --init_extent 0.5 --hull_min_hits 4 \
  --bg 0.35 --eval_every 200 --refine_start 200 --refine_stop 1000 \
  --refine_every 100 --reset_every 600 --sh_degree_interval 300 --max_gaussians 120000
```

> 人体模型（白色实验服）在纯白背景下会和背景融为一体，所以这里用灰色背景（`--bg 0.35`）渲染/合成。
> 数据集是本地生成的，`data/` 已加入 `.gitignore`；只提交渲染脚本，模型文件不随仓库分发。

## 5. 性能（M2 Pro, 16GB）

| 配置 | 每步耗时 |
|---|---|
| 100×100, 20k→27k Gaussians | 0.7–3.5 s（致密化后变慢） |
| 200×200, 20k→36k Gaussians | 1.5–4.5 s（见 `results/lego_200.log` / `human_200.log`） |

### 关于 16GB 内存的工程注意

在 M2 Pro 16GB 上连续跑较长的 MPS 训练时，Metal/MPSGraph 的编译缓存 + 系统 swap 会持续增长
（实测 swap 从 7GB 涨到 14GB，磁盘剩余空间被 swap 吃掉，最终进程被系统 SIGKILL）。这不是
纯 torch 光栅化的显存泄漏（单步 driver 内存实测约 1.5GB，`empty_cache` 后 0.15GB），而是长时间
MPSGraph 编译/系统内存压力叠加的结果。训练脚本因此提供：

- `--save_every N`：每 N 步保存 `splats.pt` / `splats_stepXXXX.pt`
- `--empty_cache_every N`：定期 `torch.mps.empty_cache()`
- `--resume PATH`：从 `splats.pt` 热启动
- `tools/eval_checkpoint.py`：从 checkpoint 恢复评估/导出 PLY（CPU 即可）

本次 `lego_200` 就是在第 600 步主动停在 checkpoint，再用 `eval_checkpoint.py` 评估和导出 PLY。
长跑建议分段（如每次 600–1000 步），或换 32GB+ 内存机器。

瓶颈几乎全在 forward 的 tile 枚举与逐批 alpha 累积（纯 torch kernel launch 开销）；backward 由
autograd 完成，相对便宜。这个速度足够做小场景/验证，离实时还很远——那属于 M1/M2（Metal kernel）。

---

## 6. 已知限制 / 后续

- M0 只支持 **classic 3DGS + pinhole** 的前向与反向。3DGUT（`with_ut`）、eval3d、lidar、
  鱼眼/正交相机、畸变模型、rolling shutter、`absgrad` 在 MPS 上会显式报 `NotImplementedError`。
- `packed` 参数被忽略（渲染结果等价，只是少了 compact 模式的省显存优化）。
- 纯 torch 路径的数值行为已与 CUDA kernel 的阈值语义对齐；由于没有 CUDA 机器，未做真实 CUDA
  张量的逐元素比对（测试用的是按 kernel 语义手写的参考实现）。
- 下一步（M1）：把投影 + tile 枚举 + 光栅化 forward 写成 Metal kernel；M2 再补 backward。

---

## 7. 文件清单

```
gsplat/gsplat/cuda/_torch_impl_mps.py     # Task A/B + 向量化 isect_tiles（新增）
gsplat/gsplat/cuda/_torch_impl.py         # 非 CUDA 设备自动改用纯 torch 实现（修改）
gsplat/gsplat/rendering.py                # _rasterization 去 CUDA 依赖 + 设备分发（修改）
examples/train_mps.py                     # MPS 训练脚本（新增）
tests/test_mps_torch_impl.py              # 数值对齐 / 梯度 / SH 测试（新增）
tools/render_human_dataset.py             # Blender 批量渲染人体多视角数据（新增）
tools/eval_checkpoint.py                  # 从 splats.pt 评估 / 导出 PLY（新增）
tools/render_turntable.py                 # 环绕视角 GIF（新增）
tools/make_comparison.py                  # GT|渲染 拼图（新增）
run_train_lego.sh                         # 一键训练（新增）
requirements-mps.txt                      # 依赖（新增）
results/lego_100/                         # 100×100 训练结果
results/lego_200/                         # 200×200 训练结果
```

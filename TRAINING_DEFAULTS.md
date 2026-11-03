# 训练默认档位（已确定）

> 决定时间：2026-11-05。以后人体/多视角 3DGS 训练按这个优先级选分辨率，避免每次重新试错。

## 默认策略

| 档位 | 用途 | 推荐步数 | 备注 |
|---|---|---|---|
| **800×800（默认）** | 日常人体重建、出图/出 PLY | 从 400 模型续训 200–300 步，或从零 600–1000 步 | 真实数据 driver 峰值约 **5.1GB**，单步 2.4–5.3s；16GB 可分段长跑 |
| 400×400 | 快速迭代、调参、稳定性优先 | 600–900 步 | 最稳，driver ~2.6GB，单步 1.3–2.1s；常作为 800 的初始化 |
| **1200×1200（按需）** | 需要极限细节时的高分辨率精修 | **只需从 800 checkpoint 续训 1–2 epoch（48–96 步）** | 起步 driver ~9.6GB，连续跑会涨到 ~18.6GB；不要单进程长跑 |

**结论：默认 800×800；有更高要求再上 1200，而且 1200 只做短程精修，不需要从头训练。**

## 推荐流程

```bash
export PYTHONPATH=$PWD/gsplat:$PWD/gsplat/examples

# 1) 400×400 基础模型（稳定、快）
.venv/bin/python examples/train_mps.py \
  --data_dir data/human --out_dir results/human_400 --downscale 1 \
  --steps 900 --init_num_pts 30000 --init_extent 0.5 --hull_min_hits 4 --bg 0.35 \
  --eval_every 200 --save_every 100 --empty_cache_every 5 \
  --refine_start 200 --refine_stop 1000 --refine_every 100 --reset_every 600 \
  --sh_degree_interval 100 --max_gaussians 150000 --grow_grad2d 1.5e-4

# 2) 默认出图档：800×800（从 400 的 900 步续训 200–300 步）
.venv/bin/python examples/train_mps.py \
  --data_dir data/human_800 --out_dir results/human_800 --downscale 1 \
  --resume results/human_400/splats_step0900.pt --start_step 900 --steps 300 \
  --eval_every 100 --save_every 100 --empty_cache_every 5 \
  --refine_start 0 --refine_stop 1400 --refine_every 100 --reset_every 1000000 \
  --sh_degree_interval 100 --max_gaussians 150000 --grow_grad2d 1.5e-4 --bg 0.35

# 3) 可选：1200×1200 精修（只跑 1–2 epoch = 48–96 步）
.venv/bin/python examples/train_mps.py \
  --data_dir data/human_1200 --out_dir results/human_1200 --downscale 1 \
  --resume results/human_800/splats_step1200.pt --start_step 1200 --steps 96 \
  --eval_every 100000 --save_every 24 --empty_cache_every 1 --log_memory_every 12 \
  --refine_start 0 --refine_stop 1450 --refine_every 100 --reset_every 1000000 \
  --sh_degree_interval 100 --max_gaussians 150000 --grow_grad2d 1.5e-4 --bg 0.35
```

> 800/1200 数据集需要按对应分辨率用 `tools/render_human_dataset.py --res 800` / `--res 1200`
> 重新渲染；400 的图像不能直接放大当 800 用。

## 为什么 1200 只做短程精修

- 1200 的活张量很小（~0.9GB），但 MPSGraph/Metal driver 会按每步变化的形状缓存 graph，
  连续训练 96 步后从 ~9.6GB 涨到 ~18.6GB，超过 16GB 物理内存；
- 从 800 模型续训 1–2 epoch 就能拿到明显的细节提升（眼睛、发丝、衣缝），继续长跑的收益递减；
- 如果确实要长跑 1200：每 96 步重启进程 + `--resume/--start_step` 分段，或等 shape bucketing /
  activation checkpointing 优化（M1/M2）。

## 相关记录

- 实测数据、内存曲线、对比图：[MPS_PORT.md](MPS_PORT.md) 第 4.5 节
- 结果目录：`results/human_400`、`results/human_800`、`results/human_1200`
- 内存曲线：`results/human_1200/memory_1200.png`

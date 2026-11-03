# gsplat-metal

在 Apple Silicon（MPS/Metal）上运行 [gsplat](https://github.com/nerfstudio-project/gsplat) 的
3D Gaussian Splatting **训练**，不依赖 CUDA / 编译扩展 / nerfacc。

- 移植说明、改动清单、数值验证与踩坑记录：[MPS_PORT.md](MPS_PORT.md)
- **训练默认档位（人体默认 800×800，1200 仅短程精修）：[TRAINING_DEFAULTS.md](TRAINING_DEFAULTS.md)**
- 一键训练（NeRF-synthetic `lego`）：`./run_train_lego.sh`
- 数值/梯度测试：`PYTHONPATH=$PWD/gsplat .venv/bin/python tests/test_mps_torch_impl.py`
- 训练脚本：`examples/train_mps.py`（lego / 人体数据均可）
- 人体多视角数据渲染（Blender）：`tools/render_human_dataset.py`
- 环绕视角 GIF / 评估 / 拼图：`tools/render_turntable.py`、`tools/eval_checkpoint.py`、`tools/make_comparison.py`
- 训练结果（PSNR、渲染图、PLY）：`results/`（已被 gitignore）
  - 人体 400×400：`results/human_400`（42.06 dB，900 步）
  - 人体 800×800（默认档）：`results/human_800`（41.29 dB，1200 步）
  - 人体 1200×1200：`results/human_1200`（2 epoch 内存测试，39.90 dB）
  - 200/400/800 清晰度对比：`results/sharpness_200_400_800.png`

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mps.txt
PYTHONPATH=$PWD/gsplat .venv/bin/python tests/test_mps_torch_impl.py
./run_train_lego.sh
```

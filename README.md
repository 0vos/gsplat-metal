# gsplat-metal

在 Apple Silicon（MPS/Metal）上运行 [gsplat](https://github.com/nerfstudio-project/gsplat) 的
3D Gaussian Splatting **训练**，不依赖 CUDA / 编译扩展 / nerfacc。

- 移植说明、改动清单、数值验证与踩坑记录：[MPS_PORT.md](MPS_PORT.md)
- 一键训练（NeRF-synthetic `lego`）：`./run_train_lego.sh`
- 数值/梯度测试：`PYTHONPATH=$PWD/gsplat .venv/bin/python tests/test_mps_torch_impl.py`
- 训练脚本：`examples/train_mps.py`（lego / 人体数据均可）
- 人体多视角数据渲染（Blender）：`tools/render_human_dataset.py`
- 环绕视角 GIF / 评估 / 拼图：`tools/render_turntable.py`、`tools/eval_checkpoint.py`、`tools/make_comparison.py`
- 训练结果（PSNR、渲染图、PLY）：`results/`（已被 gitignore）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mps.txt
PYTHONPATH=$PWD/gsplat .venv/bin/python tests/test_mps_torch_impl.py
./run_train_lego.sh
```

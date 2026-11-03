# SPDX-FileCopyrightText: Copyright (c) 2026 the Regents of the University of California, Nerfstudio Team and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a saved ``splats.pt`` checkpoint: test PSNR, side-by-side images, PLY.

Useful for recovering a run that was stopped at a checkpoint. Runs on CPU by
default (``--device cpu``) so it does not add Metal/swap pressure.

Usage::

    PYTHONPATH=gsplat:gsplat/examples .venv/bin/python tools/eval_checkpoint.py \
        --run results/lego_200 --data_dir data/lego --downscale 4 --bg 1.0
"""

import argparse
import math
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gsplat import rasterization
from gsplat.exporter import export_splats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from train_mps import load_split, normalize_cameras  # noqa: E402


def psnr(pred, gt):
    mse = F.mse_loss(pred.clamp(0, 1), gt.clamp(0, 1)).item()
    return -10.0 * math.log10(max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/lego_200")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_dir", default="data/lego")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bg", type=float, default=1.0)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--export_ply", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device
    ckpt_path = args.ckpt or os.path.join(args.run, "splats.pt")
    p = {k: v.to(device) for k, v in torch.load(ckpt_path, map_location="cpu").items()}
    means = p["means"]
    quats = p["quats"]
    scales = torch.exp(p["scales"])
    opacities = torch.sigmoid(p["opacities"])
    colors = torch.cat([p["sh0"], p["shN"]], dim=1)

    bg = (args.bg, args.bg, args.bg)
    raw = load_split(os.path.join(args.data_dir, "transforms_train.json"), args.downscale)
    T = normalize_cameras(raw["c2ws"])
    ds = load_split(
        os.path.join(args.data_dir, f"transforms_{args.split}.json"),
        args.downscale, normalize_T=T, bg_color=bg,
    )
    W, H = ds["W"], ds["H"]
    vmats = torch.linalg.inv(ds["c2ws"]).to(device)
    K = ds["K"][None].to(device)
    bgc = torch.full((1, 1, 1, 3), args.bg, device=device)

    psnrs = []
    os.makedirs(os.path.join(args.run, "renders"), exist_ok=True)
    for i in range(len(ds["images"])):
        gt = ds["images"][i].to(device)
        with torch.no_grad():
            out, alpha, _ = rasterization(
                means, quats, scales, opacities, colors,
                vmats[i : i + 1], K, W, H, sh_degree=args.sh_degree,
                packed=False, rasterize_mode="classic",
            )
            out = out + bgc * (1.0 - alpha)
        psnrs.append(psnr(out[0], gt))
        if i < args.max_images:
            canvas = torch.cat([gt, out[0].clamp(0, 1)], dim=1).cpu().numpy()
            Image.fromarray((canvas * 255).astype(np.uint8)).save(
                os.path.join(args.run, "renders", f"ckpt_{args.split}_view{i:02d}.png")
            )
    print(f"[eval] {ckpt_path}: {args.split} PSNR={np.mean(psnrs):.3f} dB "
          f"(n={len(psnrs)}, device={device}, res={W}x{H})")

    if args.export_ply:
        ply = os.path.join(args.run, "point_cloud.ply")
        export_splats(
            means=p["means"].cpu(), scales=p["scales"].cpu(), quats=p["quats"].cpu(),
            opacities=p["opacities"].cpu(), sh0=p["sh0"].cpu(), shN=p["shN"].cpu(),
            format="ply", save_to=ply,
        )
        print(f"[eval] wrote {ply}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 the Regents of the University of California, Nerfstudio Team and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render a turntable GIF from a trained gsplat checkpoint (pure-PyTorch / MPS).

The checkpoint written by ``examples/train_mps.py`` (``splats.pt``) lives in the
normalized world frame, so this script recomputes the same camera normalization
from the dataset's training transforms and orbits the object.

Usage::

    PYTHONPATH=gsplat:gsplat/examples .venv/bin/python tools/render_turntable.py \
        --run results/lego_200 --data_dir data/lego --downscale 4 \
        --frames 48 --res 200 --bg 1.0
"""

import argparse
import math
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

from gsplat import rasterization

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from train_mps import load_split, normalize_cameras  # noqa: E402


def look_at(eye, target, up):
    up = up.to(eye.device)
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    # Camera y points *down* (OpenCV), i.e. opposite the world up vector.
    y = torch.linalg.cross(x, z)
    M = torch.eye(4)
    M[:3, 0] = x
    M[:3, 1] = y
    M[:3, 2] = z
    M[:3, 3] = eye
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/lego_200")
    ap.add_argument("--data_dir", default="data/lego")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--res", type=int, default=200)
    ap.add_argument("--bg", type=float, default=1.0)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run, "turntable.gif")
    device = args.device

    ckpt = torch.load(os.path.join(args.run, "splats.pt"), map_location="cpu")
    params = {k: v.to(device) for k, v in ckpt.items()}

    bg = (args.bg, args.bg, args.bg)
    raw = load_split(os.path.join(args.data_dir, "transforms_train.json"), args.downscale)
    T = normalize_cameras(raw["c2ws"])
    tr = load_split(os.path.join(args.data_dir, "transforms_train.json"), args.downscale,
                    normalize_T=T, bg_color=bg)
    cams = tr["c2ws"][:, :3, 3]
    # The normalization rotates the world so that the cameras' up axis is
    # -c2w[:, :3, 1] (OpenCV camera y points down). Use that as world up
    # instead of assuming z-up, otherwise the turntable comes out rolled 90°.
    up = (-tr["c2ws"][:, :3, 1]).mean(dim=0)
    up = (up / up.norm()).to(device)
    tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
    if abs(float((tmp * up).sum())) > 0.9:
        tmp = torch.tensor([0.0, 0.0, 1.0], device=device)
    e1 = torch.linalg.cross(up, tmp)
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(up, e1)
    center = torch.zeros(3, device=device)
    radius = float(cams.norm(dim=-1).mean())
    height = float((cams.to(device) @ up).mean())
    K = tr["K"][None].to(device)
    W = H = args.res

    # Scale the intrinsics if the requested output resolution differs.
    if args.res != tr["W"]:
        K = K.clone()
        K[0, 0, 0] *= args.res / tr["W"]
        K[0, 1, 1] *= args.res / tr["H"]
        K[0, 0, 2] *= args.res / tr["W"]
        K[0, 1, 2] *= args.res / tr["H"]

    frames = []
    means = params["means"]
    quats = params["quats"]
    scales = torch.exp(params["scales"])
    opacities = torch.sigmoid(params["opacities"])
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    bgc = torch.full((1, 1, 1, 3), args.bg, device=device)

    for i in range(args.frames):
        theta = 2.0 * math.pi * i / args.frames
        eye = (
            center
            + radius * (math.cos(theta) * e1 + math.sin(theta) * e2)
            + height * up
        )
        c2w = look_at(eye.to(device), center.to(device), up)
        viewmat = torch.linalg.inv(c2w)[None].to(device)
        with torch.no_grad():
            out_img, alpha, _ = rasterization(
                means, quats, scales, opacities, colors,
                viewmat, K, W, H, sh_degree=args.sh_degree,
                packed=False, rasterize_mode="classic",
            )
            out_img = out_img + bgc * (1.0 - alpha)
        arr = (out_img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
        if (i + 1) % 8 == 0:
            print(f"[turntable] {i + 1}/{args.frames}", flush=True)

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"[turntable] wrote {out}")


if __name__ == "__main__":
    main()

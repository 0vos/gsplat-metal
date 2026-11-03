# SPDX-FileCopyrightText: Copyright (c) 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal 3D Gaussian Splatting trainer for Apple Silicon (MPS).

Trains on a NeRF-synthetic style dataset (Blender ``transforms_*.json`` +
``train/``/``test/`` PNGs), e.g. the ``lego`` scene shipped under ``data/``.

The rasterizer is gsplat's pure-PyTorch path: ``gsplat.rasterization()``
auto-dispatches to ``_rasterization()`` on non-CUDA devices, which uses the
pure-torch index generator + alpha compositing from
``gsplat/cuda/_torch_impl_mps.py`` and PyTorch autograd for the backward pass.

Usage (from the repository root)::

    PYTHONPATH=gsplat:gsplat/examples .venv/bin/python examples/train_mps.py \
        --data_dir data/lego --out_dir results/lego_200 --downscale 4 \
        --steps 2000

No CUDA, no nerfacc, no gsplat compiled extension.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from gsplat import rasterization
from gsplat.exporter import export_splats
from gsplat.losses import l1_loss, ssim_loss
from gsplat.strategy import DefaultStrategy

try:  # examples/datasets/normalize.py
    from datasets.normalize import similarity_from_cameras
except ImportError:  # pragma: no cover
    similarity_from_cameras = None


C0 = 0.28209479177387814


def rgb_to_sh(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / C0


def knn_mean_dist(points: Tensor, k: int = 4, chunk: int = 2048) -> Tensor:
    """Mean distance to the k-1 nearest neighbours, computed in chunks (CPU)."""
    n = points.shape[0]
    out = []
    for i in range(0, n, chunk):
        d = torch.cdist(points[i : i + chunk], points)  # [c, n]
        vals = d.topk(k + 1, largest=False, dim=1).values[:, 1:]  # skip self
        out.append(vals.pow(2).mean(dim=-1).sqrt())
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def blender_c2w(transform_matrix) -> torch.Tensor:
    """Blender camera-to-world (x right, y up, -z forward) -> OpenCV (+z fwd)."""
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    return c2w @ flip


def load_split(json_path: Path, downscale: int, normalize_T=None, bg_color=(1.0, 1.0, 1.0)):
    meta = json.loads(Path(json_path).read_text())
    angle_x = float(meta["camera_angle_x"])
    frames = meta["frames"]
    base = Path(json_path).parent

    images, masks, c2ws = [], [], []
    for fr in frames:
        rel = fr["file_path"]
        img_path = base / (rel + ("" if Path(rel).suffix else ".png"))
        im = Image.open(img_path).convert("RGBA")
        W0, H0 = im.size
        W, H = W0 // downscale, H0 // downscale
        im = im.resize((W, H), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        alpha = arr[..., 3:4]
        bg = np.asarray(bg_color, dtype=np.float32).reshape(1, 1, 3)
        rgb = arr[..., :3] * alpha + bg * (1.0 - alpha)
        images.append(torch.from_numpy(rgb).clip(0, 1))
        masks.append(torch.from_numpy(alpha[..., 0] > 0.5))
        c2ws.append(blender_c2w(fr["transform_matrix"]))

    c2ws = torch.stack(c2ws)
    W0, H0 = Image.open(
        base / (frames[0]["file_path"] + ("" if Path(frames[0]["file_path"]).suffix else ".png"))
    ).size
    W, H = W0 // downscale, H0 // downscale
    f = 0.5 * W0 / math.tan(angle_x / 2.0) / downscale
    K = torch.tensor(
        [[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    if normalize_T is not None:
        c2ws = normalize_T @ c2ws
        # ``similarity_from_cameras`` folds the world scale into the transform's
        # upper-left 3x4 block, which also scales the rotation part. Re-normalize
        # the rotation rows so the cameras stay rigid (otherwise the camera-space
        # depths are off by 1/scale and every Gaussian projects ~4x too small).
        rot_scale = c2ws[:, 0, :3].norm(dim=-1)
        c2ws[:, :3, :3] = c2ws[:, :3, :3] / rot_scale[:, None, None]

    return {
        "images": torch.stack(images),  # [N, H, W, 3]
        "masks": torch.stack(masks),  # [N, H, W]
        "c2ws": c2ws,  # [N, 4, 4]
        "K": K,
        "W": W,
        "H": H,
    }


def normalize_cameras(c2ws: torch.Tensor):
    """Similarity transform that centres the scene and puts cameras at r~1."""
    if similarity_from_cameras is not None:
        T = similarity_from_cameras(c2ws.numpy())
        return torch.from_numpy(T).float()
    # Fallback: recenter on camera mean and scale by max radius.
    center = c2ws[:, :3, 3].mean(dim=0)
    r = (c2ws[:, :3, 3] - center).norm(dim=-1).max()
    T = torch.eye(4)
    T[:3, 3] = -center
    T[:3, :] *= 1.0 / r
    return T


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def psnr(pred: Tensor, gt: Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), gt.clamp(0, 1)).item()
    return -10.0 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def visual_hull_init(
    images: Tensor,  # [V, H, W, 3]
    masks: Tensor,  # [V, H, W] bool
    c2ws: Tensor,  # [V, 4, 4]
    K: Tensor,  # [3, 3]
    n_points: int,
    extent: float = 0.7,
    res: int = 96,
    min_hits: int = 4,
    seed: int = 0,
):
    """Initialise Gaussians on the visual hull carved from the training masks.

    Random initialisation needs thousands of optimizer steps just to *move*
    Gaussians onto the object (3DGS normally starts from COLMAP SfM points).
    Carving a coarse voxel hull from the multi-view alpha masks gives us
    surface points immediately, which converges much faster on a laptop.

    Returns (points [P,3], colors [P,3]) in the normalized world frame.
    """
    V, H, W = masks.shape
    device = images.device
    lin = torch.linspace(-extent, extent, res, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # [R^3, 3]

    viewmats = torch.linalg.inv(c2ws)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    free = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
    seen = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
    vis_sum = torch.zeros(pts.shape[0], 3, device=device)
    vis_cnt = torch.zeros(pts.shape[0], device=device)
    nhit = torch.zeros(pts.shape[0], device=device)
    for v in range(V):
        cam = pts @ viewmats[v, :3, :3].T + viewmats[v, :3, 3]
        z = cam[:, 2]
        front = z > 1e-4
        zz = torch.where(front, z, torch.ones_like(z))
        u = fx * cam[:, 0] / zz + cx
        vv = fy * cam[:, 1] / zz + cy
        inside = front & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        ui = u.long().clamp(0, W - 1)
        vi = vv.long().clamp(0, H - 1)
        hit = masks[v].reshape(-1)[vi * W + ui] & inside
        free |= inside & ~hit
        seen |= hit
        vis_sum += torch.where(hit[:, None], images[v].reshape(-1, 3)[vi * W + ui], 0.0)
        vis_cnt += hit.float()
        nhit += hit.float()

    # ``nhit`` counts the views whose silhouette contains the voxel. Voxels that
    # are only ever seen by one or two (grazing) views are classic visual-hull
    # "phantoms"; require a few consistent hits to filter them out.
    hull = (nhit >= min_hits) & ~free
    hull3 = hull.reshape(res, res, res)
    pad = torch.nn.functional.pad
    h_pad = pad(hull3[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0] > 0.5
    c = h_pad[1:-1, 1:-1, 1:-1]
    nbrs = (
        h_pad[:-2, 1:-1, 1:-1] & h_pad[2:, 1:-1, 1:-1]
        & h_pad[1:-1, :-2, 1:-1] & h_pad[1:-1, 2:, 1:-1]
        & h_pad[1:-1, 1:-1, :-2] & h_pad[1:-1, 1:-1, 2:]
    )
    surface = (c & ~nbrs).reshape(-1)
    idx = surface.nonzero(as_tuple=False).reshape(-1)
    if idx.numel() < 32:
        print(f"[init] visual hull too small ({idx.numel()} surface voxels); using random init")
        g = torch.Generator(device="cpu").manual_seed(seed)
        p = (torch.rand(n_points, 3, generator=g) * 2 - 1) * extent
        col = torch.rand(n_points, 3, generator=g)
        return p, col

    g = torch.Generator(device="cpu").manual_seed(seed)
    sel = idx[torch.randperm(idx.numel(), generator=g)[:n_points]]
    if sel.numel() < n_points:  # fewer surface voxels than points: resample
        extra = idx[torch.randint(0, idx.numel(), (n_points - sel.numel(),), generator=g)]
        sel = torch.cat([sel, extra])
    pts_sel = pts[sel]
    vox = (2 * extent) / (res - 1)
    jitter = (torch.rand(pts_sel.shape, generator=g) - 0.5) * vox * 0.8
    pts_sel = (pts_sel + jitter.to(device)).clamp(-extent, extent)

    cols = vis_sum[sel] / vis_cnt[sel].clamp_min(1.0)[:, None]
    return pts_sel, cols


def build_splats(
    n: int,
    init_extent: float,
    sh_degree: int,
    device: str,
    seed: int = 0,
    points: Tensor = None,
    rgbs: Tensor = None,
):
    g = torch.Generator().manual_seed(seed)
    if points is None:
        points = (torch.rand(n, 3, generator=g) * 2 - 1) * init_extent
    if rgbs is None:
        rgbs = torch.rand(n, 3, generator=g)
    n = points.shape[0]
    dist_avg = knn_mean_dist(points, k=4)
    scales = torch.log(dist_avg.clamp_min(1e-4)).unsqueeze(-1).repeat(1, 3)
    quats = torch.randn(n, 4, generator=g)
    opacities = torch.logit(torch.full((n,), 0.1))

    colors = torch.zeros((n, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors[:, :1, :]),
            "shN": torch.nn.Parameter(colors[:, 1:, :]),
        }
    ).to(device)
    return params


def make_optimizers(params, scene_scale: float, batch_size: int = 1):
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    BS = batch_size
    opts = {}
    for name, lr in lrs.items():
        opts[name] = torch.optim.Adam(
            [{"params": params[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
    return opts


def render_one(params, viewmat, K, W, H, sh_degree, bg_color=None):
    means = params["means"]
    quats = params["quats"]
    scales = torch.exp(params["scales"])
    opacities = torch.sigmoid(params["opacities"])
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    out, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree,
        packed=False,
        rasterize_mode="classic",
        render_mode="RGB",
    )
    if bg_color is not None:
        # The Blender/NeRF-synthetic images were composited on white; empty
        # pixels have to render as that background colour, not as black.
        out = out + bg_color * (1.0 - alpha)
    return out, alpha, info


@torch.no_grad()
def evaluate(params, ds, device, sh_degree, save_prefix=None, max_images=4, args_bg=1.0):
    psnrs = []
    viewmats = torch.linalg.inv(ds["c2ws"]).to(device)
    K = ds["K"].to(device)
    for i in range(len(ds["images"])):
        gt = ds["images"][i].to(device)
        out, alpha, _ = render_one(
            params, viewmats[i : i + 1], K[None], ds["W"], ds["H"], sh_degree,
            bg_color=torch.full((1, 1, 1, 3), args_bg, device=device),
        )
        psnrs.append(psnr(out[0], gt))
        if save_prefix is not None and i < max_images:
            comp = torch.cat([gt, out[0].clamp(0, 1)], dim=1)  # side by side
            arr = (comp.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(f"{save_prefix}_view{i:02d}.png")
    return float(np.mean(psnrs)), psnrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/lego")
    ap.add_argument("--out_dir", default="results/lego_mps")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--init_num_pts", type=int, default=20000)
    ap.add_argument("--init", choices=["hull", "random"], default="hull")
    ap.add_argument("--init_extent", type=float, default=0.6)
    ap.add_argument("--hull_min_hits", type=int, default=6)
    ap.add_argument("--bg", type=float, default=1.0,
                    help="background value used to composite RGBA images and renders")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--sh_degree_interval", type=int, default=500)
    ap.add_argument("--ssim_lambda", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--refine_start", type=int, default=300)
    ap.add_argument("--refine_stop", type=int, default=1500)
    ap.add_argument("--refine_every", type=int, default=100)
    ap.add_argument("--grow_grad2d", type=float, default=2e-4)
    ap.add_argument("--reset_every", type=int, default=1000)
    ap.add_argument("--max_gaussians", type=int, default=150_000)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--empty_cache_every", type=int, default=20)
    ap.add_argument("--resume", default=None,
                    help="optional splats.pt to warm-start from")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "renders").mkdir(exist_ok=True)

    t_start = time.time()
    data_dir = Path(args.data_dir)
    bg = (args.bg, args.bg, args.bg)
    train_raw = load_split(data_dir / "transforms_train.json", args.downscale, bg_color=bg)
    T = normalize_cameras(train_raw["c2ws"])
    train = load_split(data_dir / "transforms_train.json", args.downscale, normalize_T=T, bg_color=bg)
    test = load_split(data_dir / "transforms_test.json", args.downscale, normalize_T=T, bg_color=bg)
    W, H = train["W"], train["H"]
    cams = train["c2ws"][:, :3, 3]
    scene_scale = float(cams.norm(dim=-1).max())
    print(
        f"[data] train={len(train['images'])} test={len(test['images'])} "
        f"{W}x{H} scene_scale={scene_scale:.3f} device={device}"
    )

    init_pts, init_rgbs = None, None
    if args.init == "hull":
        init_pts, init_rgbs = visual_hull_init(
            train["images"],
            train["masks"],
            train["c2ws"],
            train["K"],
            args.init_num_pts,
            extent=args.init_extent,
            min_hits=args.hull_min_hits,
            seed=args.seed,
        )
        print(f"[init] visual hull: {init_pts.shape[0]} surface points")
    params = build_splats(
        args.init_num_pts, args.init_extent, args.sh_degree, device,
        seed=args.seed, points=init_pts, rgbs=init_rgbs,
    )
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        for k, v in ckpt.items():
            if k in params and params[k].shape == v.shape:
                params[k].data.copy_(v.to(device))
        print(f"[resume] loaded {args.resume} ({params['means'].shape[0]} gaussians)")
    optimizers = make_optimizers(params, scene_scale)
    strategy = DefaultStrategy(
        absgrad=False,
        refine_start_iter=args.refine_start,
        refine_stop_iter=args.refine_stop,
        refine_every=args.refine_every,
        reset_every=args.reset_every,
        grow_grad2d=args.grow_grad2d,
        prune_opa=0.005,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    train_images = train["images"].to(device)
    train_viewmats = torch.linalg.inv(train["c2ws"]).to(device)
    train_K = train["K"].to(device)

    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "l1", "ssim", "num_gs", "psnr_test", "sec"])

    print(f"[train] steps={args.steps} init_gaussians={args.init_num_pts}")
    t0 = time.time()
    running = []
    for step in range(args.steps):
        idx = int(torch.randint(0, len(train_images), (1,)).item())
        gt = train_images[idx]
        viewmat = train_viewmats[idx : idx + 1]
        sh_degree_to_use = min(step // args.sh_degree_interval, args.sh_degree)

        out, alpha, info = render_one(
            params, viewmat, train_K[None], W, H, sh_degree_to_use,
            bg_color=torch.full((1, 1, 1, 3), args.bg, device=device),
        )
        l1 = l1_loss(out, gt[None]).mean()
        ssim_val = torch.zeros((), device=device)
        loss = l1
        if args.ssim_lambda > 0:
            ssim_val = ssim_loss(
                out.permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2)
            )
            loss = torch.lerp(l1, ssim_val, args.ssim_lambda)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")

        strategy.step_pre_backward(
            params=params, optimizers=optimizers, state=strategy_state, step=step, info=info
        )
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if params["means"].shape[0] < args.max_gaussians:
            strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                packed=False,
            )

        if (step + 1) % args.empty_cache_every == 0 and device == "mps":
            torch.mps.empty_cache()

        running.append(loss.item())
        if (step + 1) % 10 == 0:
            dt = time.time() - t0
            print(
                f"step {step + 1:4d}/{args.steps} loss={np.mean(running[-10:]):.4f} "
                f"gs={params['means'].shape[0]} {dt / 10:.2f}s/it",
                flush=True,
            )
            t0 = time.time()

        if args.save_every and (step + 1) % args.save_every == 0:
            sd = {k: v.detach().cpu() for k, v in params.items()}
            torch.save(sd, out_dir / "splats.pt")
            torch.save(sd, out_dir / f"splats_step{step + 1:04d}.pt")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            p, _ = evaluate(
                params, test, device, args.sh_degree,
                save_prefix=str(out_dir / "renders" / f"step{step + 1:04d}"),
                args_bg=args.bg,
            )
            print(f"  [eval] step {step + 1}: test PSNR={p:.2f} dB", flush=True)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step + 1, float(np.mean(running[-10:])), l1.detach().item(),
                     float(ssim_val.detach()), int(params["means"].shape[0]), p,
                     time.time() - t_start]
                )

    # Final artefacts.
    final_psnr, _ = evaluate(
        params, test, device, args.sh_degree,
        save_prefix=str(out_dir / "renders" / "final"),
        args_bg=args.bg,
    )
    print(f"[done] final test PSNR={final_psnr:.2f} dB  time={time.time() - t_start:.0f}s")

    export_splats(
        means=params["means"].detach().cpu(),
        scales=params["scales"].detach().cpu(),
        quats=params["quats"].detach().cpu(),
        opacities=params["opacities"].detach().cpu(),
        sh0=params["sh0"].detach().cpu(),
        shN=params["shN"].detach().cpu(),
        format="ply",
        save_to=str(out_dir / "point_cloud.ply"),
    )
    torch.save(
        {k: v.detach().cpu() for k, v in params.items()}, out_dir / "splats.pt"
    )
    with open(out_dir / "result.json", "w") as f:
        json.dump(
            {
                "final_psnr": final_psnr,
                "num_gaussians": int(params["means"].shape[0]),
                "scene_scale": scene_scale,
                "resolution": [H, W],
                "steps": args.steps,
                "seconds": time.time() - t_start,
                "device": device,
                "torch": torch.__version__,
            },
            f,
            indent=2,
        )
    print(f"[done] wrote {out_dir}/point_cloud.ply")


if __name__ == "__main__":
    main()

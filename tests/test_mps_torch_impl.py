# SPDX-FileCopyrightText: Copyright (c) 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the pure-PyTorch (Apple MPS) 3DGS rasterization path.

There is no CUDA device in this repo's target environment, so correctness is
established against independent, deliberately slow reference implementations
written directly from the CUDA kernel semantics:

* ``slow_rasterize_to_indices_in_range`` -- per-tile / per-pixel / per-batch
  Python loops that mirror ``RasterizeToIndices3DGSSerialBatch.cu``.
* ``brute_force_render`` -- per-pixel front-to-back alpha compositing over all
  Gaussians (no tiles, no batching, no autograd tricks).

Run with::

    PYTHONPATH=. python tests/test_mps_torch_impl.py
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math
import sys
from typing import Tuple

import torch

from gsplat.cuda._constants import ALPHA_THRESHOLD, MAX_ALPHA, TRANSMITTANCE_THRESHOLD
from gsplat.cuda._torch_impl import (
    _fully_fused_projection,
    _isect_offset_encode,
    _isect_tiles_fast,
    _quat_scale_to_covar_preci,
)
from gsplat.cuda._torch_impl_mps import (
    _isect_tiles_fast as mps_isect_tiles_fast,
    accumulate_torch,
    rasterize_to_indices_in_range_torch,
)

DEVICE = os.environ.get("TEST_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# Slow references
# ---------------------------------------------------------------------------


def slow_rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: torch.Tensor,  # [I, H, W]
    means2d: torch.Tensor,  # [I, N, 2]
    conics: torch.Tensor,  # [I, N, 3]
    opacities: torch.Tensor,  # [I, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: torch.Tensor,  # [I, th, tw]
    flatten_ids: torch.Tensor,  # [n_isects]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    I, N = means2d.shape[0], means2d.shape[1]
    th, tw = isect_offsets.shape[-2:]
    bs = tile_size * tile_size
    n_isects = flatten_ids.shape[0]
    offsets = isect_offsets.reshape(I, th * tw).to(torch.int64)
    counts = torch.zeros_like(offsets)
    for i in range(I):
        for t in range(th * tw):
            start = int(offsets[i, t])
            end = int(offsets[i, t + 1]) if t + 1 < th * tw else (
                int(offsets[i + 1, 0]) if i + 1 < I else n_isects
            )
            counts[i, t] = end - start

    gs, pix, img = [], [], []
    for i in range(I):
        for t in range(th * tw):
            off = int(offsets[i, t])
            cnt = int(counts[i, t])
            if cnt == 0:
                continue
            num_batches = (cnt + bs - 1) // bs
            b_lo = range_start
            b_hi = min(range_end, num_batches)
            if b_lo >= b_hi:
                continue
            ty, tx = t // tw, t % tw
            for dy in range(tile_size):
                py = ty * tile_size + dy
                if py >= image_height:
                    continue
                for dx in range(tile_size):
                    px = tx * tile_size + dx
                    if px >= image_width:
                        continue
                    pix_id = py * image_width + px
                    T = float(transmittances[i, py, px])
                    for b in range(b_lo, b_hi):
                        s = off + b * bs
                        e = min(off + (b + 1) * bs, off + cnt)
                        for k in range(s, e):
                            g = int(flatten_ids[k])
                            dx_ = float(means2d[i, g % N, 0]) - (px + 0.5)
                            dy_ = float(means2d[i, g % N, 1]) - (py + 0.5)
                            c = conics[i, g % N]
                            sigma = (
                                0.5 * (float(c[0]) * dx_ * dx_ + float(c[2]) * dy_ * dy_)
                                + float(c[1]) * dx_ * dy_
                            )
                            if sigma < 0.0:
                                continue
                            alpha = min(MAX_ALPHA, float(opacities[i, g % N]) * math.exp(-sigma))
                            if alpha < ALPHA_THRESHOLD:
                                continue
                            next_T = T * (1.0 - alpha)
                            if next_T <= TRANSMITTANCE_THRESHOLD:
                                break
                            gs.append(g % N)
                            pix.append(pix_id)
                            img.append(i)
                            T = next_T

    gs_t = torch.tensor(gs, dtype=torch.int64)
    pix_t = torch.tensor(pix, dtype=torch.int64)
    img_t = torch.tensor(img, dtype=torch.int64)
    # CUDA emits triples grouped by pixel id, near-to-far inside a pixel.
    combined = img_t * (image_height * image_width) + pix_t
    order = torch.argsort(combined, stable=True)
    return gs_t[order], pix_t[order], img_t[order]


def brute_force_render(
    means2d: torch.Tensor,  # [I, N, 2]
    conics: torch.Tensor,  # [I, N, 3]
    depths: torch.Tensor,  # [I, N]
    opacities: torch.Tensor,  # [I, N]
    colors: torch.Tensor,  # [I, N, C]
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel front-to-back compositing over *all* Gaussians (CPU, float32)."""
    I, N = means2d.shape[0], means2d.shape[1]
    px = (torch.arange(image_width, dtype=torch.float32) + 0.5).view(1, 1, 1, image_width)
    py = (torch.arange(image_height, dtype=torch.float32) + 0.5).view(
        1, 1, image_height, 1
    )
    dx = means2d[:, :, 0].view(I, N, 1, 1) - px  # [I, N, H, W]
    dy = means2d[:, :, 1].view(I, N, 1, 1) - py
    c = conics.view(I, N, 3, 1, 1)
    sigma = 0.5 * (c[:, :, 0] * dx * dx + c[:, :, 2] * dy * dy) + c[:, :, 1] * dx * dy
    alpha = torch.clamp_max(opacities.view(I, N, 1, 1) * torch.exp(-sigma), MAX_ALPHA)
    valid = (sigma >= 0.0) & (alpha >= ALPHA_THRESHOLD)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))

    order = torch.argsort(depths, dim=1)  # near-to-far per image
    gather = order.view(I, N, 1, 1).expand_as(alpha)
    alpha_sorted = torch.gather(alpha, 1, gather)
    valid_sorted = torch.gather(valid, 1, gather)
    colors_sorted = torch.gather(
        colors, 1, order.view(I, N, 1).expand(I, N, colors.shape[-1])
    )

    T_after = torch.cumprod(1.0 - alpha_sorted, dim=1)
    T_before = torch.cat([torch.ones_like(T_after[:, :1]), T_after[:, :-1]], dim=1)
    keep = valid_sorted & (T_after > TRANSMITTANCE_THRESHOLD)
    weights = alpha_sorted * T_before * keep
    colors_sorted = colors_sorted.view(I, N, 1, 1, -1)
    renders = (weights[..., None] * colors_sorted).sum(dim=1)  # [I, H, W, C]
    alphas = weights.sum(dim=1).unsqueeze(-1)  # [I, H, W, 1]
    return renders, alphas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scene(
    N: int = 48,
    C: int = 2,
    H: int = 40,
    W: int = 48,
    seed: int = 0,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    means = (torch.rand(N, 3) - 0.5) * 1.2  # in [-0.6, 0.6]^3
    quats = torch.randn(N, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3) * 0.06 + 0.01
    opacities = torch.rand(N) * 0.9 + 0.05
    colors = torch.rand(N, 3)

    # Look-at cameras on a circle around the origin.
    c2ws = []
    for ci in range(C):
        ang = 2 * math.pi * ci / C
        eye = torch.tensor(
            [1.6 * math.sin(ang), 0.6, 2.2 * math.cos(ang)], dtype=torch.float32
        )
        z = -eye / eye.norm()  # camera forward (gsplat/COLMAP: +z in cam space)
        up = torch.tensor([0.0, 1.0, 0.0])
        x = torch.linalg.cross(up, z)
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        Wm = torch.eye(4)
        Wm[:3, :3] = torch.stack([x, y, z])
        Wm[:3, 3] = eye
        c2ws.append(Wm)
    c2ws = torch.stack(c2ws)
    viewmats = torch.linalg.inv(c2ws)

    f = 1.1 * W
    Ks = torch.zeros(C, 3, 3)
    Ks[:, 0, 0] = f
    Ks[:, 1, 1] = f
    Ks[:, 0, 2] = W / 2
    Ks[:, 1, 2] = H / 2
    Ks[:, 2, 2] = 1.0

    out = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
    }
    return {k: v.to(device) for k, v in out.items()}


def project(scene, H, W, eps2d=0.3):
    covars, _ = _quat_scale_to_covar_preci(
        scene["quats"], scene["scales"], True, False, triu=False
    )
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        scene["means"],
        covars,
        scene["viewmats"],
        scene["Ks"],
        W,
        H,
        eps2d=eps2d,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
    )
    return radii, means2d, depths, conics


def run_case(name, N, C, H, W, tile_size=16, seed=0):
    scene_cpu = make_scene(N=N, C=C, H=H, W=W, seed=seed, device="cpu")
    radii, means2d, depths, conics = project(scene_cpu, H, W)
    opacities = scene_cpu["opacities"].expand(C, N)
    I = C

    # Sanity: vectorized tile enumeration matches the looped reference.
    from gsplat.cuda._torch_impl import _isect_tiles as _slow_isect_tiles

    tp_g, isect_ids, flatten_ids = _slow_isect_tiles(
        means2d, radii, depths, tile_size, math.ceil(W / tile_size), math.ceil(H / tile_size)
    )
    tp_f, isect_ids_f, flatten_ids_f = mps_isect_tiles_fast(
        means2d, radii, depths, tile_size, math.ceil(W / tile_size), math.ceil(H / tile_size)
    )
    assert torch.equal(tp_g, tp_f), "tiles_per_gauss mismatch"
    assert torch.equal(isect_ids, isect_ids_f), "isect_ids mismatch"
    assert torch.equal(flatten_ids, flatten_ids_f), "flatten_ids mismatch"

    th, tw = math.ceil(H / tile_size), math.ceil(W / tile_size)
    isect_offsets = _isect_offset_encode(isect_ids, I, tw, th)

    # --- Task A: index generation vs. CUDA-semantics Python reference ---
    trans = torch.ones(I, H, W, dtype=torch.float32)
    gs_ref, pix_ref, img_ref = slow_rasterize_to_indices_in_range(
        0, 10**9, trans, means2d, conics, opacities, W, H, tile_size,
        isect_offsets, flatten_ids,
    )
    gs_mps, pix_mps, img_mps = rasterize_to_indices_in_range_torch(
        0, 10**9, trans, means2d, conics, opacities, W, H, tile_size,
        isect_offsets, flatten_ids,
    )
    assert torch.equal(gs_ref, gs_mps), (
        f"{name}: gaussian ids mismatch ({gs_ref.shape} vs {gs_mps.shape})"
    )
    assert torch.equal(pix_ref, pix_mps), f"{name}: pixel ids mismatch"
    assert torch.equal(img_ref, img_mps), f"{name}: image ids mismatch"

    # Also exercise a batched (range_start/range_end) call, like the rasterizer.
    bs = tile_size * tile_size
    counts = torch.diff(
        torch.cat([isect_offsets.reshape(-1), torch.tensor([len(flatten_ids)])])
    )
    n_batches = int((counts + bs - 1).div(bs, rounding_mode="floor").max())
    for step in range(0, max(1, n_batches), 2):
        t = torch.ones(I, H, W, dtype=torch.float32) * 0.7
        ref = slow_rasterize_to_indices_in_range(
            step, step + 2, t, means2d, conics, opacities, W, H, tile_size,
            isect_offsets, flatten_ids,
        )
        got = rasterize_to_indices_in_range_torch(
            step, step + 2, t, means2d, conics, opacities, W, H, tile_size,
            isect_offsets, flatten_ids,
        )
        for a, b, label in zip(ref, got, ("gs", "pix", "img")):
            assert torch.equal(a, b), f"{name}: batched range {step} {label} mismatch"

    # --- Task B: accumulate vs. brute force ---
    # Full (single-batch) rasterization through the pipeline.
    from gsplat.rendering import _rasterization

    with torch.no_grad():
        render_mps, alpha_mps, meta = _rasterization(
            scene_cpu["means"], scene_cpu["quats"], scene_cpu["scales"],
            scene_cpu["opacities"], scene_cpu["colors"],
            scene_cpu["viewmats"], scene_cpu["Ks"], W, H,
            tile_size=tile_size,
        )
    render_bf, alpha_bf = brute_force_render(
        means2d.reshape(I, N, 2), conics.reshape(I, N, 3),
        depths.reshape(I, N), opacities.reshape(I, N),
        scene_cpu["colors"].expand(I, N, 3),
        W, H,
    )
    err = (render_mps - render_bf).abs()
    aerr = (alpha_mps - alpha_bf).abs()
    print(
        f"  {name}: pairs={gs_mps.numel()} render max_err={err.max():.3e} "
        f"mean_err={err.mean():.3e} alpha max_err={aerr.max():.3e}"
    )
    assert err.max() < 2e-5, f"{name}: render mismatch {err.max()}"
    assert aerr.max() < 2e-5, f"{name}: alpha mismatch {aerr.max()}"

    # --- accumulate must be invariant to the order of whole pixel groups ---
    # (accumulate relies on the near-to-far order *within* a pixel, which the
    # index generator guarantees; groups may be presented in any order.)
    uniq = torch.unique(combined := (img_mps * (H * W) + pix_mps), sorted=True)
    perm = torch.randperm(uniq.numel(), generator=torch.Generator().manual_seed(seed))
    new_order = []
    for u in uniq[perm].tolist():
        new_order.append((combined == u).nonzero(as_tuple=False).reshape(-1))
    idx = torch.cat(new_order)
    renders1, alphas1 = accumulate_torch(
        means2d.reshape(I, N, 2), conics.reshape(I, N, 3),
        opacities.expand(I, N), scene_cpu["colors"].expand(I, N, 3),
        gs_mps, pix_mps, img_mps, W, H,
    )
    renders2, alphas2 = accumulate_torch(
        means2d.reshape(I, N, 2), conics.reshape(I, N, 3),
        opacities.expand(I, N), scene_cpu["colors"].expand(I, N, 3),
        gs_mps[idx], pix_mps[idx], img_mps[idx], W, H,
    )
    assert (renders1 - renders2).abs().max() < 1e-6, f"{name}: accumulate group-order issue"
    print(f"  {name}: accumulate group-order invariance OK")

    # --- MPS vs CPU parity ---
    if DEVICE == "mps":
        scene_mps = make_scene(N=N, C=C, H=H, W=W, seed=seed, device="mps")
        with torch.no_grad():
            r_mps, a_mps, _ = _rasterization(
                scene_mps["means"], scene_mps["quats"], scene_mps["scales"],
                scene_mps["opacities"], scene_mps["colors"],
                scene_mps["viewmats"], scene_mps["Ks"], W, H,
                tile_size=tile_size,
            )
        d = (r_mps.cpu() - render_mps.cpu()).abs().max()
        da = (a_mps.cpu() - alpha_mps.cpu()).abs().max()
        print(f"  {name}: MPS vs CPU render max diff={d:.3e} alpha={da:.3e}")
        assert d < 5e-5 and da < 5e-5, f"{name}: MPS/CPU mismatch"


def main():
    print(f"device={DEVICE} torch={torch.__version__}")
    run_case("basic-N48", N=48, C=2, H=40, W=48)
    run_case("small-N24", N=24, C=1, H=32, W=32, seed=3)
    run_case("tile8", N=32, C=2, H=37, W=53, tile_size=8, seed=5)
    run_case("edge-H41", N=40, C=1, H=41, W=17, seed=7)
    run_sh_grad_case()
    print("ALL TESTS PASSED")




def _render_with_grads(device: str, sh_degree: int = 2, seed: int = 11):
    """Render a small SH scene and return (renders, grads dict)."""
    N, C, H, W = 32, 2, 36, 44
    scene = make_scene(N=N, C=C, H=H, W=W, seed=seed, device=device)
    torch.manual_seed(seed + 1)
    sh = ((torch.rand(N, (sh_degree + 1) ** 2, 3) - 0.5) * 0.6).to(device)
    params = {
        "means": scene["means"].clone().requires_grad_(True),
        "quats": scene["quats"].clone().requires_grad_(True),
        "scales": scene["scales"].clone().requires_grad_(True),
        "opacities": scene["opacities"].clone().requires_grad_(True),
        "colors": sh.clone().requires_grad_(True),
    }
    from gsplat.rendering import _rasterization

    out, alpha, _ = _rasterization(
        params["means"], params["quats"], params["scales"], params["opacities"],
        params["colors"], scene["viewmats"], scene["Ks"], W, H,
        sh_degree=sh_degree,
    )
    loss = (out * out).mean() + alpha.mean()
    loss.backward()
    grads = {k: v.grad.detach().clone() for k, v in params.items()}
    return out.detach(), alpha.detach(), grads


def run_sh_grad_case():
    torch.manual_seed(0)
    out_cpu, a_cpu, g_cpu = _render_with_grads("cpu")
    if DEVICE != "mps":
        print("  SH/grad: CPU only (MPS unavailable); skipping parity")
        return
    out_mps, a_mps, g_mps = _render_with_grads("mps")
    dr = (out_cpu.cpu() - out_mps.cpu()).abs().max().item()
    da = (a_cpu.cpu() - a_mps.cpu()).abs().max().item()
    print(f"  SH render CPU vs MPS: {dr:.3e} (alpha {da:.3e})")
    assert dr < 5e-5 and da < 5e-5
    for k in g_cpu:
        g1, g2 = g_cpu[k], g_mps[k].cpu()
        denom = g1.abs().max().clamp_min(1e-8)
        rel = (g1 - g2).abs().max() / denom
        print(f"  grad[{k}]: max rel diff={rel:.3e} (abs {g1.abs().max():.3e})")
        assert rel < 5e-3, f"gradient mismatch for {k}: {rel}"
    print("  SH + autograd parity OK")


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 the Regents of the University of California, Nerfstudio Team and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stitch the GT | render pairs saved by ``examples/train_mps.py`` into a grid.

Usage::

    python tools/make_comparison.py --run results/lego_200 --out results/lego_200/comparison.png
"""

import argparse
import glob
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/lego_200")
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=2)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run, "renders", f"{args.prefix}_view*.png")))
    if not paths:
        raise SystemExit(f"no renders found in {args.run}/renders with prefix {args.prefix}")
    imgs = [Image.open(p) for p in paths]
    w, h = imgs[0].size
    cols = min(args.cols, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(imgs):
        canvas.paste(im.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    out = args.out or os.path.join(args.run, "comparison.png")
    canvas.save(out)
    print(f"[comparison] wrote {out} ({canvas.size[0]}x{canvas.size[1]}) from {len(imgs)} pairs")


if __name__ == "__main__":
    main()

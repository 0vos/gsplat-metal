#!/usr/bin/env bash
# Train 3DGS on the NeRF-synthetic "lego" scene with gsplat's pure-PyTorch
# (MPS) rasterizer. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=${DATA_DIR:-data/lego}
OUT_DIR=${OUT_DIR:-results/lego_200}
STEPS=${STEPS:-1000}
DOWNSCALE=${DOWNSCALE:-4}
INIT_N=${INIT_N:-30000}

PYTHONPATH="$PWD/gsplat:$PWD/gsplat/examples" \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv/bin/python -u examples/train_mps.py \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --downscale "$DOWNSCALE" \
    --steps "$STEPS" \
    --init_num_pts "$INIT_N" \
    --eval_every 250 \
    --refine_start 300 \
    --refine_stop $((STEPS * 3 / 4)) \
    --refine_every 100 \
    --reset_every 1000 \
    --sh_degree_interval 500 \
    --save_every 200 \
    --empty_cache_every 20 \
    --max_gaussians 150000

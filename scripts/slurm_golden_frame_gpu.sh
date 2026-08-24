#!/bin/bash
# One-off GPU validation of the golden-frame regression test (both tiers).
# Usage: sbatch scripts/slurm_golden_frame_gpu.sh
# Cost: single a10g node, ~15 min (~$0.30 at $1.21/hr on gpu-med a10g).
#SBATCH -p gpu-med
#SBATCH --gres=gpu:a10g:1
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -o /data/npadman/tmp/slurm-logs/golden/%j.out
set -euo pipefail
cd /data/npadman/1-Projects/roman/roman_disperser/main
nvidia-smi --query-gpu=name --format=csv,noheader
pixi run -e cuda python -c "import jax; print('jax', jax.__version__, jax.default_backend())"
pixi run -e cuda pytest -v tests/test_golden_frame.py

#!/bin/bash
# Full 18-SCA grism sim of the line-centering test field, one a10g GPU.
#SBATCH --job-name=line_test_grism
#SBATCH --partition=gpu-med
#SBATCH --gres=gpu:a10g:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/npadman/tmp/slurm-logs/line_test/%x-%j.out
set -euo pipefail

RUN=/data/npadman/1-Projects/roman/roman_disperser/line_test_outputs/run
# Run from main/ (its cuda pixi env is installed; build_grism_image.py is
# identical on the feature branch, which only adds the catalog/checker scripts).
cd /data/npadman/1-Projects/roman/roman_disperser/main

# Persistent JAX compile cache on /data so re-runs skip the ~60s/SCA cold compile.
export JAX_COMPILATION_CACHE_DIR=/data/npadman/1-Projects/roman/roman_disperser/line_test_outputs/jax-cache

pixi run -e cuda python scripts/build_grism_image.py \
    --config "$RUN/line_test_config.yaml" \
    --pointings "$RUN/line_test_pointing.ecsv" \
    --gpu 0 \
    --cache-dir "$JAX_COMPILATION_CACHE_DIR"

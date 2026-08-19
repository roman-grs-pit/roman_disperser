#!/bin/bash
# Prepare-stage decomposition run (issue #30 follow-on to job 7144):
# sub-stage timings of prepare_galaxy_images + structural variants
# (fast-size FFT padding, precomputed PSF FFTs, fused warp scatter,
# galaxy batching). One pinned a10g.
set -euo pipefail

WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/native_deposit
WB=workbench/20260819-native-deposit
RESULTS=$WB/results/${BENCH_SUBDIR:-a10g}
PIXI=/home/npadman/.pixi/bin/pixi

export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
export JAX_COMPILATION_CACHE_DIR=/data/npadman/jax-cache-grism

cd "$WORKDIR"
mkdir -p "$RESULTS"

echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=== git: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

echo "=== prep decomposition (order 1)"
$PIXI run -e cuda python $WB/bench_prep.py \
    --n-gal 32 --repeats 2 --orders 1 \
    --fast-sizes 303,315,320,324 --precomp-size 320 --batch-sizes 4,16 \
    --tag prep-a10g --out $RESULTS/prep.json

echo "=== prep run complete"

#!/bin/bash
# GPU benchmark driver for the 2026-08-18 performance exploration.
# Runs bench_disperse.py under several environment configurations on ONE
# pinned a10g. Submitted via slurm_submit.sh; can also be run directly on a
# GPU node. Each configuration is a separate python process because the XLA
# allocator env vars are read once at JAX initialization.
set -euo pipefail

WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/perf
BENCH=workbench/20260818-perf-exploration/bench_disperse.py
RESULTS=workbench/20260818-perf-exploration/results/${BENCH_SUBDIR:-a10g}
PIXI=/home/npadman/.pixi/bin/pixi

export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
export JAX_COMPILATION_CACHE_DIR=/data/npadman/jax-cache-grism

cd "$WORKDIR"
mkdir -p "$RESULTS"

echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=== git: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

# 1. Main sweep: all variants, all orders, stars. Production-like env
#    (pixi cuda env, default JAX allocator = preallocating BFC).
echo "=== [1/4] main sweep"
$PIXI run -e cuda python $BENCH \
    --n-gal 300 --n-star 300 --repeats 2 \
    --variants baseline,noscatter,spread,local \
    --tag gpu-main --out $RESULTS/gpu-main.json

# 2. Keith's interactive env vars (the 20260803 log.txt run):
#    platform allocator + no preallocation. Baseline only, fewer galaxies
#    (expected slow).
echo "=== [2/4] platform allocator"
XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false \
$PIXI run -e cuda python $BENCH \
    --n-gal 100 --skip-stars --repeats 2 \
    --variants baseline \
    --tag gpu-platform-alloc --out $RESULTS/gpu-platform-alloc.json

# 3. Keith's batch-script env vars: cuda_malloc_async + no preallocation.
echo "=== [3/4] cuda_malloc_async"
TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false \
$PIXI run -e cuda python $BENCH \
    --n-gal 100 --skip-stars --repeats 2 \
    --variants baseline \
    --tag gpu-async-alloc --out $RESULTS/gpu-async-alloc.json

# 4. Chunk-size sensitivity (baseline deposit), orders 0 and 1.
echo "=== [4/4] chunk sweep"
for CS in 100 1000; do
    $PIXI run -e cuda python $BENCH \
        --n-gal 100 --skip-stars --repeats 2 --orders 0,1 \
        --variants baseline --chunk-size $CS \
        --tag gpu-chunk$CS --out $RESULTS/gpu-chunk$CS.json
done

echo "=== bench complete"

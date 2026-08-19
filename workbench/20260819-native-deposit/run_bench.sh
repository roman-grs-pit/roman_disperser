#!/bin/bash
# GPU benchmark driver for the native16 deposit prototype (issue #30,
# branch feature/native-deposit). Runs bench_disperse.py on ONE pinned a10g
# in the production-like env (pixi cuda env, default JAX allocator).
# Submitted via slurm_submit.sh; can also be run directly on a GPU node.
set -euo pipefail

WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/native_deposit
BENCH=workbench/20260819-native-deposit/bench_disperse.py
RESULTS=workbench/20260819-native-deposit/results/${BENCH_SUBDIR:-a10g}
PIXI=/home/npadman/.pixi/bin/pixi

export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
export JAX_COMPILATION_CACHE_DIR=/data/npadman/jax-cache-grism

cd "$WORKDIR"
mkdir -p "$RESULTS"

echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=== git: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

# Main sweep: baseline vs native16 (plus noscatter as the compute floor),
# all orders, stars for context, 3 timed repeats. The native16 equivalence
# gate (allclose rtol 1e-5 + relative-sum vs baseline) runs first per order.
$PIXI run -e cuda python $BENCH \
    --n-gal 300 --n-star 300 --repeats 3 \
    --variants baseline,noscatter,native16 \
    --tag native16-main --out $RESULTS/native16-main.json

echo "=== bench complete"

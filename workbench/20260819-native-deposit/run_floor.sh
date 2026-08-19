#!/bin/bash
# Floor-decomposition run for the native16 deposit (issue #30): per-stage
# timings (bench_stages.py) + fused end-to-end chunk-size sweep
# (bench_disperse.py, native16 only). One pinned a10g.
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

echo "=== [1/2] per-stage decomposition (order 1, chunks 500/2000)"
$PIXI run -e cuda python $WB/bench_stages.py \
    --n-gal 30 --repeats 2 --orders 1 --chunk-sizes 500,2000 \
    --tag stages-a10g --out $RESULTS/stages.json

echo "=== [2/2] fused end-to-end chunk sweep (native16, order 1)"
for CS in 1000 2000; do
    $PIXI run -e cuda python $WB/bench_disperse.py \
        --n-gal 100 --skip-stars --repeats 3 --orders 1 \
        --variants baseline,native16 --chunk-size $CS \
        --tag native16-chunk$CS --out $RESULTS/native16-chunk$CS.json
done

echo "=== floor run complete"

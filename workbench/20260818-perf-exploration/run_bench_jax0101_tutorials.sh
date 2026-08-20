#!/bin/bash
# Why does tutorials notebook 08 measure ~435 ms/source-order on the a10g
# when the perf bench's steady state is 9-10 ms/gal (jax 0.7.2 and 0.11.1)?
# Two candidates fit the single cold wall-clock: (a) compile domination
# (225 dispersions can't amortize ~6 fori compiles), or (b) the tutorials
# gpu env's jax/jaxlib 0.10.1 (conda-forge, cuda129 build) carrying the
# scatter regression measured at 15-19x on 0.11.0 (SLURM 7153) — 0.10.1
# was never benched, and upstream jax-ml/jax#39959 brackets it (0.8->0.11).
# Discriminator: steady-state timing (bench times after compile+warmup)
# under the tutorials gpu env itself, same bench/data/hardware as all
# prior probes. ~10 ms/gal => compile; ~150 ms/gal => 0.10.1 regression.
set -euo pipefail
PERF=/data/npadman/1-Projects/roman/roman_disperser/perf
TUT=/data/npadman/1-Projects/roman/roman_disperser/tutorials
PIXI=/home/npadman/.pixi/bin/pixi
export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
# Separate compilation cache: different XLA version must not poison the shared one.
export JAX_COMPILATION_CACHE_DIR=/data/npadman/tmp/jax-cache-tut-0101

cd "$PERF"
echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PIXI run --manifest-path "$TUT/pixi.toml" -e gpu python -c \
    "import jax, jaxlib, roman_disperser; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__, '|', jax.devices(), '| roman_disperser', roman_disperser.__version__)"

$PIXI run --manifest-path "$TUT/pixi.toml" -e gpu python \
    workbench/20260818-perf-exploration/bench_disperse.py \
    --n-gal 100 --n-star 100 --repeats 2 \
    --variants baseline,noscatter \
    --tag gpu-jax0101-tutorials \
    --out workbench/20260818-perf-exploration/results/a10g/gpu-jax0101-tutorials.json
echo "=== done"

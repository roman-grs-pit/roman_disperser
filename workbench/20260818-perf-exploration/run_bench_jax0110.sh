#!/bin/bash
# Test whether jax 0.11.0 — the exact version in Keith's conda env
# (roman-disp, conda list captured 2026-08-19) — reproduces the 20260803
# order-skewed slowdown on the same a10g hardware. Our 7138 probe ran
# 0.11.1 (released the day before) and was fast; this pins one patch back.
# Runs the bench twice: default allocator, then Keith's
# XLA_PYTHON_CLIENT_ALLOCATOR=platform, to catch an allocator x version
# interaction (platform alone measured ~1.25x on jax 0.7.2).
set -euo pipefail
WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/perf
VENV=/data/npadman/tmp/perf-jax0110-venv
export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
# Separate compilation cache: different XLA version must not poison the shared one.
export JAX_COMPILATION_CACHE_DIR=/data/npadman/tmp/jax-cache-perf-0110

cd "$WORKDIR"
echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
rm -rf "$VENV"
/data/npadman/1-Projects/roman/roman_disperser/perf/.pixi/envs/default/bin/python -m venv "$VENV"
source "$VENV/bin/activate"
pip -q install -U pip
pip -q install -e . astropy "jax[cuda12]==0.11.0"
python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo "=== pass 1: default allocator"
python workbench/20260818-perf-exploration/bench_disperse.py \
    --n-gal 100 --skip-stars --repeats 2 \
    --variants baseline,noscatter \
    --tag gpu-jax0110 \
    --out workbench/20260818-perf-exploration/results/a10g/gpu-jax0110.json

echo "=== pass 2: XLA_PYTHON_CLIENT_ALLOCATOR=platform (Keith's setting)"
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python workbench/20260818-perf-exploration/bench_disperse.py \
    --n-gal 100 --skip-stars --repeats 2 \
    --variants baseline,noscatter \
    --tag gpu-jax0110-platform \
    --out workbench/20260818-perf-exploration/results/a10g/gpu-jax0110-platform.json

echo "=== done"

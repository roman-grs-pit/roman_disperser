#!/bin/bash
# Test whether a NEWER jax (pip-latest, vs the pixi env's 0.7.2) reproduces
# the 20260803 run's order-dependent slowdown on the same a10g hardware.
# Keith's run used a personal conda env of unknown jax version; this probes
# the jax-version axis with everything else held fixed.
set -euo pipefail
WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/perf
VENV=/data/npadman/tmp/perf-jaxlatest-venv
export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
# Separate compilation cache: different XLA version must not poison the shared one.
export JAX_COMPILATION_CACHE_DIR=/data/npadman/tmp/jax-cache-perf-latest

cd "$WORKDIR"
echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
rm -rf "$VENV"
/data/npadman/1-Projects/roman/roman_disperser/perf/.pixi/envs/default/bin/python -m venv "$VENV"
source "$VENV/bin/activate"
pip -q install -U pip
pip -q install -e . astropy "jax[cuda12]"
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python workbench/20260818-perf-exploration/bench_disperse.py \
    --n-gal 100 --skip-stars --repeats 2 \
    --variants baseline,noscatter \
    --tag gpu-jaxlatest \
    --out workbench/20260818-perf-exploration/results/a10g/gpu-jaxlatest.json
echo "=== done"

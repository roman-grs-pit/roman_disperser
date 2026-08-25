#!/usr/bin/env bash
#
# Run the GPU-dependent tests on a GPU node.
#
# Why this exists
# ---------------
# `tests/test_disperser_gpu.py` guards every test with `has_gpu()`, so on a
# GPU-less machine (the cluster head node, a laptop) pytest reports green with
# those tests never executed. In July 2026 that let a TF32 defect ship: the
# sky->FPA rotation was exact on CPU and wrong on GPU by a median 1.84 px, and
# no test ever ran anywhere that could tell the difference.
#
# So the CPU/GPU comparison is only worth anything if something actually runs
# it on a GPU. This script is that something. It is cheap enough (~3-4 min,
# roughly $0.08 on an a10g) to run on every merge to main.
#
# It runs the WHOLE suite, deliberately. An earlier version listed only the
# three test files that seemed GPU-relevant, which covered 3 of 12 files and
# left ~173 tests -- the disperser, galaxy disperser, star disperser, PSF model,
# Sersic and catalog code, all of which execute JAX on the GPU in production --
# never once exercised on a GPU. That is the same mistake one level up: gating
# only what you already suspect. TF32 happened to live in a file that would have
# been on such a list; the next device-dependent defect need not.
#
# The one exclusion is `-m 'not slow'`, and it is not about GPU coverage: those
# tests generate STPSF PSF caches, which take hours and are CPU-bound (STPSF has
# no GPU path at all), so running them here would burn GPU-hours to exercise
# NumPy. Generate caches with scripts/generate_psf_caches.py instead.
#
# The exception to the exclusion is the golden-frame *full* tier
# (tests/test_golden_frame.py, also marked `slow`, ~1 min on an a10g): it is
# the end-to-end output guard at production wavelength sampling and part of
# the PR merge gate (CLAUDE.md), so it runs here as a second pytest step. This
# script is therefore the whole GPU half of the merge gate: the suite, then
# the golden full tier. The perf suite (benchmarks/) is separate.
#
# Cluster-specific bits (partition, GRES, log dir) are env-overridable:
#   SLURM_PARTITION, SLURM_GRES, SLURM_LOG_DIR.
#
# Usage
# -----
#   scripts/slurm_run_tests.sh                 # submit and return
#   scripts/slurm_run_tests.sh --wait          # submit and block until done
#
# The GRES is pinned to a10g deliberately: `gpu-med` mixes a10g / l4 / l40s,
# and an unqualified `--gres=gpu:1` may land on the pricier l40s. It is also
# the architecture the TF32 defect was characterised on, so a regression is
# compared against like hardware.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${SLURM_LOG_DIR:-/data/npadman/tmp/slurm-logs/tests}"
PARTITION="${SLURM_PARTITION:-gpu-med}"
GRES="${SLURM_GRES:-gpu:a10g:1}"
mkdir -p "$LOG_DIR"

WAIT_FLAG=""
if [[ "${1:-}" == "--wait" ]]; then
    WAIT_FLAG="--wait"
fi

# shellcheck disable=SC2086
sbatch $WAIT_FLAG \
    --job-name=rd-gputests \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --time=00:40:00 \
    --output="$LOG_DIR/gputests-%j.out" \
    --wrap="cd '$REPO_ROOT' && \
            pixi run -e cuda python -c 'import jax; print(\"devices:\", jax.devices())' && \
            pixi run -e cuda python -m pytest tests -v -m 'not slow' && \
            pixi run -e cuda python -m pytest -v -m slow tests/test_golden_frame.py"

echo "Submitted. Logs: $LOG_DIR/gputests-<jobid>.out"

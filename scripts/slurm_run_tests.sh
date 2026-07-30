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
# it on a GPU. This script is that something. It is cheap enough (~2-3 min,
# roughly $0.06 on an a10g) to run on every merge to main.
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
mkdir -p "$LOG_DIR"

WAIT_FLAG=""
if [[ "${1:-}" == "--wait" ]]; then
    WAIT_FLAG="--wait"
fi

# shellcheck disable=SC2086
sbatch $WAIT_FLAG \
    --job-name=rd-gputests \
    --partition=gpu-med \
    --gres=gpu:a10g:1 \
    --time=00:30:00 \
    --output="$LOG_DIR/gputests-%j.out" \
    --wrap="cd '$REPO_ROOT' && \
            pixi run -e cuda python -c 'import jax; print(\"devices:\", jax.devices())' && \
            pixi run -e cuda python -m pytest \
                tests/test_disperser_gpu.py \
                tests/test_optical_model_jax.py \
                tests/test_precision_convention.py \
                -v -m 'not slow'"

echo "Submitted. Logs: $LOG_DIR/gputests-<jobid>.out"

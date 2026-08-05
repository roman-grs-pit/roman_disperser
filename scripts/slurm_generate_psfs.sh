#!/bin/bash
# Generate PSF caches for all 18 SCAs in parallel, for one dispersing element.
#
# Spawns 18 separate Python processes (one per SCA), each running
# generate_psf_caches.py with --workers 1. This sidesteps a fork+JAX
# deadlock that hangs multiprocessing.Pool with --workers > 1 on this
# codebase.
#
# Submit (element defaults to grism; pass prism explicitly):
#   sbatch scripts/slurm_generate_psfs.sh
#   sbatch --export=ALL,ELEMENT=prism scripts/slurm_generate_psfs.sh
#
# Ported from the prism branch (where it generated the 18 PRISM caches on
# cpun-2xlg in ~35 min); made element-aware in the prism merge. The time
# limit fits the prism (1 filter/SCA); the grism generates 2 filters/SCA,
# so raise --time to ~1:30:00 for a full grism regeneration.
#
#SBATCH --job-name=psf-cache
#SBATCH --partition=cpun-2xlg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/data/npadman/tmp/slurm-logs/psf/%j.out
#SBATCH --error=/data/npadman/tmp/slurm-logs/psf/%j.err

set -euo pipefail

# Run from the repo root (the checkout this script lives in)
cd "$(dirname "$(readlink -f "$0")")/.."

ELEMENT=${ELEMENT:-grism}

# Single-thread BLAS in each worker (per-PSF threading gives only ~13%; we
# parallelize over SCAs instead).
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Shared vendored data dir if set, else the checkout's data/
CACHE_DIR=${CACHE_DIR:-${ROMAN_DISPERSER_DATA:-data}/psf_cache}
mkdir -p "$CACHE_DIR"
mkdir -p /data/npadman/tmp/slurm-logs/psf

# Audit trail
mkdir -p scripts/slurm-meta-psf
META=scripts/slurm-meta-psf/${SLURM_JOB_ID}.env
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOB_NAME=${SLURM_JOB_NAME}"
  echo "PARTITION=${SLURM_JOB_PARTITION}"
  echo "NODE=${SLURMD_NODENAME:-unknown}"
  echo "ELEMENT=${ELEMENT}"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
  echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CACHE_DIR=${CACHE_DIR}"
} > "$META"

echo "Starting 18 parallel PSF cache generators (${ELEMENT}) on $(hostname)"
echo "Audit: $META"
date -u

# Launch one process per SCA, all backgrounded
pids=()
for i in $(seq -f '%02g' 1 18); do
  DET="WFI${i}"
  LOG="/data/npadman/tmp/slurm-logs/psf/${SLURM_JOB_ID}_${DET}.log"
  pixi run python scripts/generate_psf_caches.py \
    --detectors "$DET" \
    --element "$ELEMENT" \
    --workers 1 \
    --cache-dir "$CACHE_DIR" \
    > "$LOG" 2>&1 &
  pids+=($!)
done

echo "Launched ${#pids[@]} workers, PIDs: ${pids[*]}"

# Wait for all, capturing per-worker exit codes
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "WORKER FAILED: pid $pid"
    fail=$((fail + 1))
  fi
done

date -u
echo "Done. Failures: $fail"
echo "Cache contents:"
ls -la "$CACHE_DIR" | head -60

if [[ $fail -gt 0 ]]; then
  exit 1
fi

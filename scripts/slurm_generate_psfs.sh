#!/bin/bash
# Generate PRISM PSF caches for all 18 SCAs in parallel.
#
# Spawns 18 separate Python processes (one per SCA), each running
# generate_psf_caches.py with --workers 1. This sidesteps a fork+JAX
# deadlock that hangs multiprocessing.Pool with --workers > 1 on this
# codebase.
#
# Submit:
#   sbatch scripts/slurm_generate_psfs.sh
#
#SBATCH --job-name=prism-psf
#SBATCH --partition=cpun-2xlg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/data/npadman/tmp/slurm-logs/psf/%j.out
#SBATCH --error=/data/npadman/tmp/slurm-logs/psf/%j.err

set -euo pipefail

cd /data/npadman/1-Projects/roman_disperser_prism

# Single-thread BLAS in each worker (per-PSF threading gives only ~13%; we
# parallelize over SCAs instead).
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

CACHE_DIR=data/psf_cache
mkdir -p "$CACHE_DIR"
mkdir -p /data/npadman/tmp/slurm-logs/psf

# Audit trail
META=scripts/slurm-meta-psf/${SLURM_JOB_ID}.env
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOB_NAME=${SLURM_JOB_NAME}"
  echo "PARTITION=${SLURM_JOB_PARTITION}"
  echo "NODE=${SLURMD_NODENAME:-unknown}"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
  echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CACHE_DIR=${CACHE_DIR}"
} > "$META"

echo "Starting 18 parallel PSF cache generators on $(hostname)"
echo "Audit: $META"
date -u

# Launch one process per SCA, all backgrounded
pids=()
for i in $(seq -f '%02g' 1 18); do
  DET="WFI${i}"
  LOG="/data/npadman/tmp/slurm-logs/psf/${SLURM_JOB_ID}_${DET}.log"
  pixi run python scripts/generate_psf_caches.py \
    --detectors "$DET" \
    --orders 1 \
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
ls -la "$CACHE_DIR" | head -25

if [[ $fail -gt 0 ]]; then
  exit 1
fi

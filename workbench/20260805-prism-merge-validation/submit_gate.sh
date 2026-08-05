#!/bin/bash
# Submit one validation-gate render (1 SCA, 1 pointing) on gpu-med/a10g.
#
# Usage:
#   sbatch --export=ALL,CHECKOUT=<repo dir>,CONFIG=<config.yaml>,ECSV=<pointings.ecsv> \
#       workbench/20260805-prism-merge-validation/submit_gate.sh
#
# CHECKOUT selects which code renders (the unmodified main checkout for the
# baseline, this worktree for the branch gates); CONFIG and ECSV are absolute
# paths into this workbench. ~5.5 min of GPU at $1.21/hr once warm; first run
# pays JIT compilation on top. a10g pinned: gpu-med mixes a10g/l4/l40s.
#
#SBATCH --job-name=prism-merge-gate
#SBATCH --partition=gpu-med
#SBATCH --gres=gpu:a10g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=/data/npadman/tmp/slurm-logs/prism-merge-fable/%j.out
#SBATCH --error=/data/npadman/tmp/slurm-logs/prism-merge-fable/%j.err

set -euo pipefail

: "${CHECKOUT:?set CHECKOUT to the repo checkout to render with}"
: "${CONFIG:?set CONFIG to the gate config yaml}"
: "${ECSV:?set ECSV to the one-row pointing table}"

cd "$CHECKOUT"

# Audit trail next to the outputs
OUTDIR=$(grep '^output_dir:' "$CONFIG" | awk '{print $2}')
mkdir -p "$OUTDIR" /data/npadman/tmp/slurm-logs/prism-merge-fable
META="$OUTDIR/slurm-meta-${SLURM_JOB_ID}.env"
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "NODE=${SLURMD_NODENAME:-unknown}"
  echo "CHECKOUT=${CHECKOUT}"
  echo "CONFIG=${CONFIG}"
  echo "ECSV=${ECSV}"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
  echo "GIT_DIRTY=$(git status --porcelain | head -1 | wc -l)"
  echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$META"

echo "Gate render: checkout=$CHECKOUT config=$CONFIG"
date -u
pixi run -e cuda python scripts/build_grism_image.py \
    --config "$CONFIG" --pointings "$ECSV" --force
date -u
echo "Done."

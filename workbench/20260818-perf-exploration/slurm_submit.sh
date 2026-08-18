#!/bin/bash
# Submit the perf-exploration benchmark to one pinned a10g.
# Cost: gpu-med a10g ~= $1.21/hr on-demand; expected wall < 1 hr.
# Writes an audit .env next to the results per repo convention.
set -euo pipefail

WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/perf
META=$WORKDIR/workbench/20260818-perf-exploration/results
LOGDIR=/data/npadman/tmp/slurm-logs/perf
mkdir -p "$LOGDIR" "$META"

JOB=$(sbatch --parsable \
    -J perf-bench \
    -p gpu-med --gres=gpu:a10g:1 \
    -c 8 --mem=24G -t 02:00:00 \
    -o "$LOGDIR/%j.out" \
    "$WORKDIR/workbench/20260818-perf-exploration/run_bench.sh")

cat > "$META/bench-$JOB.env" <<EOF
JOB=$JOB
PARTITION=gpu-med
GRES=gpu:a10g:1
MEM=24G
TIME=02:00:00
PROJ_ROOT=$WORKDIR
GIT_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
BRANCH=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD)
SLURM_LOG=$LOGDIR/$JOB.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
PURPOSE=perf-exploration microbenchmark (deposit variants + allocator envs)
EOF

echo "submitted job $JOB; log $LOGDIR/$JOB.out"

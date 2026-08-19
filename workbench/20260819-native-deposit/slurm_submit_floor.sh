#!/bin/bash
# Submit the native16 floor-decomposition run to one pinned a10g.
# Cost: gpu-med a10g ~= $1.21/hr on-demand; expected wall < 30 min.
set -euo pipefail

WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/native_deposit
META=$WORKDIR/workbench/20260819-native-deposit/results
LOGDIR=/data/npadman/tmp/slurm-logs/perf
mkdir -p "$LOGDIR" "$META"

JOB=$(sbatch --parsable \
    -J native16-floor \
    -p gpu-med --gres=gpu:a10g:1 \
    -c 4 --mem=24G -t 01:30:00 \
    -o "$LOGDIR/%j.out" \
    "$WORKDIR/workbench/20260819-native-deposit/run_floor.sh")

cat > "$META/bench-$JOB.env" <<EOF
JOB=$JOB
PARTITION=gpu-med
GRES=gpu:a10g:1
MEM=24G
TIME=01:30:00
PROJ_ROOT=$WORKDIR
GIT_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
BRANCH=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD)
SLURM_LOG=$LOGDIR/$JOB.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
PURPOSE=native16 floor decomposition (issue #30): per-stage timings + fused chunk sweep
EOF

echo "submitted job $JOB; log $LOGDIR/$JOB.out"

#!/usr/bin/env bash
# Archive the 2026-05-05 spectro acceptance run + pointing ECSV to the
# spinup-003131-romanisim-l3 bucket. Run from the head node. Idempotent.
#
# Source / dest layout (mirrors imaging/2026-04-30/...):
#   /mnt/roman-science/grs/acceptance-testing-20260430/acceptance-testing-spectro.sim.ecsv
#     -> s3://spinup-003131-romanisim-l3/grs/acceptance-testing-20260430/acceptance-testing-spectro.sim.ecsv
#   /mnt/.../spectro/2026-05-05/acceptance/{output,output_l2,output_l2_mosaics,slurm-meta,scripts,logs}/
#     -> s3://.../spectro/2026-05-05/acceptance/...
set -euo pipefail

PROFILE=spinup-003131-romanisim-l3
BUCKET=s3://spinup-003131-romanisim-l3
SRC_BASE=/mnt/roman-science/grs/acceptance-testing-20260430
DST_BASE=$BUCKET/grs/acceptance-testing-20260430
ACC=spectro/2026-05-05/acceptance

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "test credentials"
aws --profile $PROFILE s3 ls $DST_BASE/ > /dev/null
echo "credentials ok"

step "pointing ECSV (3 KB)"
aws --profile $PROFILE s3 cp \
    $SRC_BASE/acceptance-testing-spectro.sim.ecsv \
    $DST_BASE/acceptance-testing-spectro.sim.ecsv

# Small dirs first as a credential / layout sanity check.
for sub in output_l2_mosaics slurm-meta scripts logs; do
    step "sync $ACC/$sub"
    aws --profile $PROFILE s3 sync \
        $SRC_BASE/$ACC/$sub/ \
        $DST_BASE/$ACC/$sub/
done

# Big payloads.
step "sync $ACC/output_l2 (117 GB)"
aws --profile $PROFILE s3 sync \
    $SRC_BASE/$ACC/output_l2/ \
    $DST_BASE/$ACC/output_l2/ \
    --only-show-errors

step "sync $ACC/output (73 GB)"
aws --profile $PROFILE s3 sync \
    $SRC_BASE/$ACC/output/ \
    $DST_BASE/$ACC/output/ \
    --only-show-errors

step "done"
echo "verify counts:"
aws --profile $PROFILE s3 ls --recursive $DST_BASE/$ACC/output_l2/ | wc -l
aws --profile $PROFILE s3 ls --recursive $DST_BASE/$ACC/output/ | wc -l

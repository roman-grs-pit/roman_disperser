#!/usr/bin/env bash
#
# Run the HEALPix SED-store afterburner (scripts/healpix_repartition_seds.py)
# on SLURM: `plan` inline (seconds; reads metadata + stats shards), then one
# `fill` task per planned task as an array, then `verify` as a dependent job.
#
# Each fill task streams every source shard whole (~17 GB for the acceptance
# catalog) and holds its pixels in RAM (<= --task-mem-gb raw, default 40 GB),
# so it runs on mem-lg (r6i.4xlarge, 124 GB, ~$1.01/h).
#
# Usage
# -----
#   scripts/slurm_healpix_repartition.sh <input_catalog_dir> <output_catalog_dir> [plan args...]
#
# e.g.
#   A=/mnt/roman-science/grs/acceptance-testing-20260430
#   scripts/slurm_healpix_repartition.sh $A/catalogs_padded $A/catalogs_padded_hp
#
# Re-running is safe: `plan` refuses to overwrite an existing plan (so the
# script reuses it) and `fill` skips pixels already marked complete.
#
# Env overrides: SLURM_PARTITION (mem-lg), SLURM_MEM (120G), SLURM_TIME (2:00:00),
# SLURM_LOG_DIR (/data/npadman/tmp/slurm-logs/healpix).
#
# Audit trail: <output>/slurm-meta/healpix-<JobID>.env records what was
# submitted and against which commit.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1
    exit 2
fi
INPUT="$1"; OUTPUT="$2"; shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${SLURM_LOG_DIR:-/data/npadman/tmp/slurm-logs/healpix}"
PARTITION="${SLURM_PARTITION:-mem-lg}"
MEM="${SLURM_MEM:-120G}"
TIME="${SLURM_TIME:-2:00:00}"
SCRIPT="scripts/healpix_repartition_seds.py"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain -- "$SCRIPT")" ]]; then
    echo "ERROR: $SCRIPT has uncommitted changes; commit first (provenance)." >&2
    exit 1
fi
GIT_COMMIT="$(git rev-parse --short HEAD)"

if [[ -f "$OUTPUT/healpix_plan.parquet" ]]; then
    echo "Plan exists in $OUTPUT; reusing it."
else
    pixi run python "$SCRIPT" plan --input "$INPUT" --output "$OUTPUT" "$@"
fi
N_TASKS=$(pixi run python -c "import pyarrow.parquet as pq; \
print(int(pq.read_table('$OUTPUT/healpix_plan.parquet').column('task').to_numpy().max()) + 1)")
echo "Fill tasks: $N_TASKS"

FILL_ID=$(sbatch --parsable -p "$PARTITION" --mem="$MEM" --time="$TIME" \
    --array=0-$((N_TASKS - 1)) -J hp-fill \
    -o "$LOG_DIR/%A_%a.out" \
    --wrap "cd '$REPO_ROOT' && pixi run python $SCRIPT fill --output '$OUTPUT'")
VERIFY_ID=$(sbatch --parsable -p "$PARTITION" --mem=32G --time=1:00:00 \
    --dependency=afterok:"$FILL_ID" -J hp-verify \
    -o "$LOG_DIR/%j-verify.out" \
    --wrap "cd '$REPO_ROOT' && pixi run python $SCRIPT verify --output '$OUTPUT'")

mkdir -p "$OUTPUT/slurm-meta"
cat > "$OUTPUT/slurm-meta/healpix-$FILL_ID.env" <<EOF
INPUT=$INPUT
OUTPUT=$OUTPUT
PLAN_ARGS=$*
N_TASKS=$N_TASKS
PARTITION=$PARTITION
MEM=$MEM
TIME=$TIME
FILL_JOB=$FILL_ID
VERIFY_JOB=$VERIFY_ID
LOG_DIR=$LOG_DIR
GIT_COMMIT=$GIT_COMMIT
SUBMITTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "fill array $FILL_ID (0-$((N_TASKS - 1))), verify $VERIFY_ID; logs in $LOG_DIR"

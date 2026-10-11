#!/usr/bin/env bash
#
# A/B test of the HEALPix SED store (thread catalog-store-layout): render the
# same pointing twice on A10G, once per galaxy-SED store layout, and compare
# timings and images.
#
#   A = sim layout     acceptance-testing-20260430/catalogs_padded     (as array 10117)
#   B = healpix layout acceptance-testing-20260430/catalogs_padded_hp  (fill 11823, verify 11824)
#
# Pointing: VISIT 1 (RA 10, Dec 0, PA 0) of the 4-roll v0.15.1 rerun, selected
# with --worker-index 0 --num-workers 4 (row idx % 4 == worker), so the config,
# ECSV (whose stem seeds the noise), seed 42 and SCA set are exactly array
# 10117's worker 1. Its output and log are the third reference point:
#   /mnt/roman-science/grs/20261007_4roll-ra10dec0-v0151/spectro/2026-10-07/4roll/
#
# Code: the library comes from main's `cuda` env (main @17873e3; src/ is
# identical to this branch, and to 886e247 = v0.15.1); the *script* is this
# branch's build_dispersed_image.py, whose loader dispatches on the store's
# `layout` attr. A and B run concurrently on separate nodes so both see the
# same S3 conditions; they share the JAX disk cache, so JIT cost is common.
#
# Usage: bash submit.sh   (from the head node; ~2 x g5.2xlarge, <= ~$3)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SED_STORE="$(cd "$HERE/../.." && pwd)"
RD_MAIN=/data/npadman/1-Projects/roman/roman_disperser/main
REF_RUN=/mnt/roman-science/grs/20261007_4roll-ra10dec0-v0151
REF_DIR=$REF_RUN/spectro/2026-10-07/4roll
ECSV=$REF_RUN/4roll-ra10dec0.sim.ecsv
CAT=/mnt/roman-science/grs/acceptance-testing-20260430
OUT=/mnt/roman-science/grs/20261011_sed-store-ab
SLURM_LOG=/data/npadman/tmp/slurm-logs/sed-store-ab
PART=gpu-med; GRES=gpu:a10g:1; MEM=24G; TIME=03:00:00

[ -z "$(git -C "$SED_STORE" status --porcelain -- scripts workbench)" ] \
    || { echo "sed_store scripts/workbench dirty; commit first"; exit 1; }
[ -z "$(git -C "$RD_MAIN" status --porcelain -- src pixi.toml pixi.lock)" ] \
    || { echo "main src/env dirty"; exit 1; }
[ -z "$(git -C "$SED_STORE" diff --stat "$(git -C "$RD_MAIN" rev-parse HEAD)" HEAD -- src)" ] \
    || { echo "src differs between main and this branch"; exit 1; }
mkdir -p "$OUT/slurm-meta" "$SLURM_LOG"

for V in A B; do
    if [[ $V == A ]]; then CATDIR=$CAT/catalogs_padded; else CATDIR=$CAT/catalogs_padded_hp; fi
    D=$OUT/$V; mkdir -p "$D/logs"
    # Same config as 10117 except output_dir and catalog_dir.
    sed -e "s#^output_dir:.*#output_dir: $D/output#" \
        -e "s#^catalog_dir:.*#catalog_dir: $CATDIR#" \
        "$REF_DIR/config.yaml" > "$D/config.yaml"
    W=$OUT/slurm-meta/task-$V.sh
    cat > "$W" <<EOT
#!/usr/bin/env bash
set -euo pipefail
source $REF_DIR/scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
hostname
exec pixi run --manifest-path $RD_MAIN/pixi.toml -e cuda \\
  python $SED_STORE/scripts/build_dispersed_image.py \\
    --config $D/config.yaml --pointings $ECSV --cache-dir \$JAX_CACHE --gpu 0 \\
    --worker-index 0 --num-workers 4 --log-file $D/logs/worker.log
EOT
    chmod +x "$W"
    JOB=$(sbatch --parsable --partition=$PART --gres=$GRES --cpus-per-task=4 \
        --mem=$MEM --time=$TIME --job-name=sedab-$V \
        --output="$SLURM_LOG/%j-$V.out" "$W")
    cat > "$OUT/slurm-meta/sedab-$V-$JOB.env" <<EOT
JOB=$JOB
VARIANT=$V
CATALOG_DIR=$CATDIR
CONFIG=$D/config.yaml
ECSV=$ECSV
PARTITION=$PART
GRES=$GRES
MEM=$MEM
TIME=$TIME
SCRIPT_COMMIT=$(git -C "$SED_STORE" rev-parse HEAD)
LIB_ENV=$RD_MAIN (cuda) @ $(git -C "$RD_MAIN" rev-parse HEAD)
SLURM_LOG=$SLURM_LOG/$JOB-$V.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
    echo "$V: job $JOB  catalog $CATDIR"
done

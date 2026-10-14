#!/usr/bin/env bash
#
# Full-stack timing on the XMM HEALPix store (thread xmm-scale-perf, step 6).
#
# Code: feature/source-cuts HEAD (string sim keys + jit-payload-args + jitted
# per-batch Sersic + source_cuts). Catalog: 20260917-xmm_lss_hp (HEALPix
# re-partition of the FULL metadata.parquet; selections are made with
# source_cuts, not with filtered.metadata.parquet). Pointing: Keith's part_03
# row 0, all 18 SCAs, grism, same config as ../oomfix (cone 0.6, batch 100,
# npix 30, seed 42), A10G. Three selections, run concurrently on separate g5s:
#
#   full = no cuts                                  (cf. oomfix/fix, job 12219,
#                                                    sim-layout store: 12,759 s)
#   h27  = F158 mag_le 27 + half_light_radius > 0 for SER
#          (reproduces Keith's filtered.metadata.parquet: 12,831,050 vs
#           12,831,046 rows; the 4 are H = 27.000000 boundary rows)
#   h265 = as h27 with F158 mag_le 26.5
#
# What this measures: per-SCA SED read, JIT, dispersion ms/gal, total; GPU
# peak. Expected: SED read drops from ~500 s/SCA (sim layout) to seconds.
# Each run has its own cold JAX cache so JIT times are comparable.
#
# Usage: bash submit.sh   (head node; 3 x g5.2xlarge, ~1 h each, ~$4 total)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
REPRO=/mnt/roman-science/grs/20261012_xmm-scale-perf/repro
CAT=/mnt/roman-science/grs/catalogs/20260917-xmm_lss_hp
OUT=/mnt/roman-science/grs/20261012_xmm-scale-perf/timing
SLURM_LOG=/data/npadman/tmp/slurm-logs/xmm-scale-perf
CACHE_ROOT=/data/npadman/tmp/jax-cache-xmm-timing
PART=gpu-med; GRES=gpu:a10g:1; MEM=28G; TIME=04:00:00

[ -z "$(git -C "$WT" status --porcelain -- src scripts workbench tests)" ] \
    || { echo "worktree dirty; commit first"; exit 1; }
mkdir -p "$OUT/slurm-meta" "$SLURM_LOG"

cuts_for() {
    case $1 in
        full) echo "" ;;
        h27|h265)
            local m; [[ $1 == h27 ]] && m=27.0 || m=26.5
            cat <<EOT
source_cuts:
  - {column: F158, op: mag_le, value: $m}
  - {column: half_light_radius, op: gt, value: 0.0, types: [SER]}
EOT
            ;;
    esac
}

for V in full h27 h265; do
    D=$OUT/$V
    [ ! -e "$D/output" ] || { echo "$D/output exists; refusing to overwrite"; exit 1; }
    mkdir -p "$D/logs"
    cp "$REPRO/part_03.ecsv" "$D/part_03.ecsv"
    { cat <<EOT
# As ../oomfix configs except catalog_dir (HEALPix store) and source_cuts.
output_dir: $D/output
seed: 42
element: grism
scas: all
cone_radius: 0.6
star_batch_size: 1000
galaxy_batch_size: 100
galaxy_npix: 30
catalog_dir: $CAT
EOT
      cuts_for $V; } > "$D/config.yaml"
    W=$OUT/slurm-meta/task-$V.sh
    cat > "$W" <<EOT
#!/usr/bin/env bash
set -euo pipefail
export ROMAN_DISPERSER_DATA=/data/grs/common/refdata/roman_disperser_data
export STPSF_PATH=/data/grs/common/refdata/stpsf-data
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
hostname
exec pixi run --manifest-path $WT/pixi.toml -e cuda \\
  python $WT/scripts/build_dispersed_image.py \\
    --config $D/config.yaml --pointings $D/part_03.ecsv \\
    --cache-dir $CACHE_ROOT/$V --gpu 0 \\
    --worker-index 0 --num-workers 9 --log-file $D/logs/worker.log
EOT
    chmod +x "$W"
    JOB=$(sbatch --parsable --partition=$PART --gres=$GRES --cpus-per-task=4 \
        --mem=$MEM --time=$TIME --job-name=xmm-time-$V \
        --output="$SLURM_LOG/%j-timing-$V.out" "$W")
    cat > "$OUT/slurm-meta/timing-$V-$JOB.env" <<EOT
JOB=$JOB
VARIANT=$V
WORKTREE=$WT
CODE_COMMIT=$(git -C "$WT" rev-parse HEAD)
CATALOG_DIR=$CAT
CONFIG=$D/config.yaml
ECSV=$D/part_03.ecsv (copy of $REPRO/part_03.ecsv)
POINTING=row 0 (--worker-index 0 --num-workers 9)
PARTITION=$PART
GRES=$GRES
MEM=$MEM
TIME=$TIME
JAX_CACHE=$CACHE_ROOT/$V (cold)
SLURM_LOG=$SLURM_LOG/$JOB-timing-$V.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
    echo "submitted $V: $JOB"
done

#!/usr/bin/env bash
#
# GPU timing of the JITTED per-batch Sersic generator (thread xmm-scale-perf).
#
# Same pointing / catalog / config as ../oomfix (Keith's part_03 row 0,
# unfiltered 20260917-xmm_lss, no source_cuts, grism, batch 100, npix 30,
# seed 42, A10G), SCAs 1-4, on this worktree's HEAD (feature/source-cuts,
# which carries the jitted generator 7cfa924; source_cuts unused here).
# Compare against ../oomfix FIX (job 12219, un-jitted per-batch generator):
#   - per-order ms/galaxy (12219: 2.4/2.3-2.4/2.3)
#   - MODEL within the GPU golden tolerance (expected diffs ~1e-8 of max)
# Caveat: runs while the XMM HEALPix fill (12239) reads the same sim store,
# so galaxy-SED read times are not comparable; dispersion times are.
#
# Usage: bash submit.sh   (head node; 1 x g5.2xlarge, ~45 min, ~$1)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
REPRO=/mnt/roman-science/grs/20261012_xmm-scale-perf/repro
CAT=/mnt/roman-science/grs/catalogs/20260917-xmm_lss
OUT=/mnt/roman-science/grs/20261012_xmm-scale-perf/jitsersic
SLURM_LOG=/data/npadman/tmp/slurm-logs/xmm-scale-perf
CACHE_ROOT=/data/npadman/tmp/jax-cache-xmm-jitsersic
PART=gpu-med; GRES=gpu:a10g:1; MEM=28G

[ -z "$(git -C "$WT" status --porcelain -- src scripts workbench tests)" ] \
    || { echo "worktree dirty; commit first"; exit 1; }
[ ! -e "$OUT/run/output" ] || { echo "$OUT/run/output exists; refusing to overwrite"; exit 1; }
mkdir -p "$OUT/slurm-meta" "$SLURM_LOG"

for V in run; do
    SCAS="[1, 2, 3, 4]"; TIME=02:00:00
    D=$OUT/$V; mkdir -p "$D/logs"
    cp "$REPRO/part_03.ecsv" "$D/part_03.ecsv"
    cat > "$D/config.yaml" <<EOT
# As $REPRO/config.yaml (Keith's full_run.yaml values) except output_dir, scas.
output_dir: $D/output
seed: 42
element: grism
scas: $SCAS
cone_radius: 0.6
star_batch_size: 1000
galaxy_batch_size: 100
galaxy_npix: 30
catalog_dir: $CAT
EOT
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
        --mem=$MEM --time=$TIME --job-name=xmm-jitsersic \
        --output="$SLURM_LOG/%j-jitsersic.out" "$W")
    cat > "$OUT/slurm-meta/jitsersic-$JOB.env" <<EOT
JOB=$JOB
VARIANT=$V
WORKTREE=$WT
CODE_COMMIT=$(git -C "$WT" rev-parse HEAD)
CATALOG_DIR=$CAT
METADATA=metadata.parquet (unfiltered, as Keith)
CONFIG=$D/config.yaml
SCAS=$SCAS
ECSV=$D/part_03.ecsv (copy of $REPRO/part_03.ecsv)
POINTING=row 0 (--worker-index 0 --num-workers 9)
PARTITION=$PART
GRES=$GRES
MEM=$MEM
TIME=$TIME
JAX_CACHE=$CACHE_ROOT/$V (cold)
SLURM_LOG=$SLURM_LOG/$JOB-jitsersic.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
    echo "submitted $V: $JOB"
done

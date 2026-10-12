#!/usr/bin/env bash
#
# GPU verification of the Sersic per-batch OOM fix (thread xmm-scale-perf, step 3).
#
# Same pointing / catalog / config as ../repro (Keith's part_03 row 0, unfiltered
# 20260917-xmm_lss, grism, cone 0.6, batch 100, npix 30, seed 42, A10G), run on
# two code versions concurrently on separate g5 nodes:
#
#   base = perf/jit-payload-args @ BASE_SHA   (no fix)   SCAs 1-4
#          SCAs 1-4 did not OOM at v0.15.1 (repro 12200), so they give a
#          baseline the fix can be compared against bitwise.
#   fix  = fix/sersic-batching   @ this HEAD  (fix)      SCAs 1-6
#          SCA 5 OOMed at v0.15.1 (Keith 11334; our repro 12200 if it does);
#          1-4 compare against base, 5-6 show the dense SCAs now complete.
#
# Pass criteria on SCAs 1-4 (base vs fix):
#   - MODEL within GPU run-to-run noise, i.e. the GPU golden tolerance used by
#     ../../20261011-sed-store-ab/compare.py (allclose rtol=1e-5,
#     atol=1e-5*max|ref|; observed sim-vs-sim rel sum diff <= ~1.5e-9). Not
#     bitwise: GPU scatter-add ordering is nondeterministic. (CPU: bitwise,
#     checked on the golden scenes.)
#   - ISIM reported, not gated (last-bit MODEL noise flips Poisson draws).
#   - per-order ms/galaxy unchanged within run-to-run scatter.
# And on the fix run: GPU peak well below base's and ~flat across SCAs;
# SCAs 5-6 complete.
#
# Each version runs from its own worktree's pixi `cuda` env (src differs: the
# fix touches src/roman_disperser/pipeline.py). Cold per-run JAX caches on /data.
#
# Usage: bash submit.sh   (head node; 2 x g5.2xlarge, ~45 + ~70 min, <= ~$3)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_WT="$(cd "$HERE/../../.." && pwd)"
BASE_WT=/data/npadman/1-Projects/roman/roman_disperser/jit
BASE_SHA=6a94346
REPRO=/mnt/roman-science/grs/20261012_xmm-scale-perf/repro
CAT=/mnt/roman-science/grs/catalogs/20260917-xmm_lss
OUT=/mnt/roman-science/grs/20261012_xmm-scale-perf/oomfix
SLURM_LOG=/data/npadman/tmp/slurm-logs/xmm-scale-perf
CACHE_ROOT=/data/npadman/tmp/jax-cache-xmm-oomfix
PART=gpu-med; GRES=gpu:a10g:1; MEM=28G; TIME=03:00:00

[ -z "$(git -C "$FIX_WT" status --porcelain -- src scripts workbench tests)" ] \
    || { echo "fix worktree dirty; commit first"; exit 1; }
[ "$(git -C "$BASE_WT" rev-parse --short HEAD)" = "$BASE_SHA" ] \
    || { echo "base worktree not at $BASE_SHA"; exit 1; }
[ -z "$(git -C "$BASE_WT" status --porcelain -- src scripts)" ] \
    || { echo "base worktree dirty"; exit 1; }
[ ! -e "$OUT/base/output" ] && [ ! -e "$OUT/fix/output" ] \
    || { echo "$OUT outputs exist; refusing to overwrite"; exit 1; }
mkdir -p "$OUT/slurm-meta" "$SLURM_LOG"

for V in base fix; do
    if [[ $V == base ]]; then WT=$BASE_WT; SCAS="[1, 2, 3, 4]"; else WT=$FIX_WT; SCAS="[1, 2, 3, 4, 5, 6]"; fi
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
        --mem=$MEM --time=$TIME --job-name=xmm-oom-$V \
        --output="$SLURM_LOG/%j-oomfix-$V.out" "$W")
    cat > "$OUT/slurm-meta/oomfix-$V-$JOB.env" <<EOT
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
SLURM_LOG=$SLURM_LOG/$JOB-oomfix-$V.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
    echo "submitted $V: $JOB"
done

#!/usr/bin/env bash
#
# XMM-LSS OOM repro + speed check in OUR env (thread xmm-scale-perf, step 1).
#
# Re-renders the first pointing of Keith Buckholz's part_03 (RA 35.397461,
# Dec -4.827622, PA 245; output stem part_03_001.087.001.001.021.002), which
# OOMed at SCA 5 (36,890 galaxies, 1.98 GiB jit_true_divide) on an A10G in his
# job 11334 and completed on an L40S in his job 11635. Run dir:
#   /mnt/roman-science/grs/20261005_xmm-lss-first-run/
#
# What is held equal to Keith's run: config (full_run.yaml values: grism, all
# SCAs, cone 0.6, star batch 1000, galaxy batch 100, galaxy_npix 30, seed 42),
# catalog (20260917-xmm_lss, whose metadata.parquet the driver reads = the
# UNFILTERED 15,983,994 rows, as his runs did), pointing ECSV (a copy of his
# part_03.ecsv; its stem seeds the noise, so the name is kept), and the A10G.
# What differs: env = main's pixi `cuda` env (jax/jaxlib 0.7.2, conda-forge
# CUDA jaxlib, py3.12) instead of his conda env (jax 0.10.2, cpu jaxlib + pypi
# cuda12 plugin, py3.14); script = this branch's build_dispersed_image.py
# (v0.15.1 driver + layout dispatch + string-sim keys + memory log) instead of
# his fork (v0.15.1 driver + string-sim keys + memory log). The sim-layout
# read path is the v0.15.1 one.
#
# Expected: OOM at SCA 5 on unbatched Sersic generation if the diagnosis is
# right (the process then dies; SCAs 1-4 still give per-galaxy speed in our
# env). If it does NOT OOM, that is informative too (env-dependent footprint).
#
# Usage: bash submit.sh   (head node; 1 x g5.2xlarge, <= 4 h, <= ~$5)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SED_STORE="$(cd "$HERE/../../.." && pwd)"
RD_MAIN=/data/npadman/1-Projects/roman/roman_disperser/main
KEITH_RUN=/mnt/roman-science/grs/20261005_xmm-lss-first-run
KEITH_ECSV=/data/kbuckholz/github_dir/xmm-sim-scripts/pointings/part_03.ecsv
CAT=/mnt/roman-science/grs/catalogs/20260917-xmm_lss
OUT=/mnt/roman-science/grs/20261012_xmm-scale-perf/repro
SLURM_LOG=/data/npadman/tmp/slurm-logs/xmm-scale-perf
JAX_CACHE=$OUT/jax-cache   # cold, per-run (Keith's runs also started cold)
PART=gpu-med; GRES=gpu:a10g:1; MEM=28G; TIME=04:00:00

[ -z "$(git -C "$SED_STORE" status --porcelain -- scripts workbench)" ] \
    || { echo "sed_store scripts/workbench dirty; commit first"; exit 1; }
[ -z "$(git -C "$RD_MAIN" status --porcelain -- src pixi.toml pixi.lock)" ] \
    || { echo "main src/env dirty"; exit 1; }
[ -z "$(git -C "$SED_STORE" diff --stat "$(git -C "$RD_MAIN" rev-parse HEAD)" HEAD -- src)" ] \
    || { echo "src differs between main and this branch"; exit 1; }
[ ! -e "$OUT/output" ] || { echo "$OUT/output exists; refusing to overwrite"; exit 1; }
mkdir -p "$OUT/logs" "$OUT/slurm-meta" "$SLURM_LOG"

cp "$KEITH_ECSV" "$OUT/part_03.ecsv"
cat > "$OUT/config.yaml" <<EOT
# Mirrors $KEITH_RUN/full_run.yaml except output_dir.
output_dir: $OUT/output
seed: 42
element: grism
scas: all
cone_radius: 0.6
star_batch_size: 1000
galaxy_batch_size: 100
galaxy_npix: 30
catalog_dir: $CAT
EOT

W=$OUT/slurm-meta/task.sh
cat > "$W" <<EOT
#!/usr/bin/env bash
set -euo pipefail
export ROMAN_DISPERSER_DATA=/data/grs/common/refdata/roman_disperser_data
export STPSF_PATH=/data/grs/common/refdata/stpsf-data
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
hostname
# Row 0 of the 9-row part_03.ecsv only (round-robin: idx % 9 == 0).
exec pixi run --manifest-path $RD_MAIN/pixi.toml -e cuda \\
  python $SED_STORE/scripts/build_dispersed_image.py \\
    --config $OUT/config.yaml --pointings $OUT/part_03.ecsv \\
    --cache-dir $JAX_CACHE --gpu 0 \\
    --worker-index 0 --num-workers 9 --log-file $OUT/logs/worker.log
EOT
chmod +x "$W"
JOB=$(sbatch --parsable --partition=$PART --gres=$GRES --cpus-per-task=4 \
    --mem=$MEM --time=$TIME --job-name=xmm-repro \
    --output="$SLURM_LOG/%j-repro.out" "$W")
cat > "$OUT/slurm-meta/repro-$JOB.env" <<EOT
JOB=$JOB
CATALOG_DIR=$CAT
METADATA=metadata.parquet (unfiltered, as Keith)
CONFIG=$OUT/config.yaml
ECSV=$OUT/part_03.ecsv (copy of $KEITH_ECSV, sha256 $(sha256sum "$KEITH_ECSV" | cut -c1-16))
POINTING=row 0 (--worker-index 0 --num-workers 9)
PARTITION=$PART
GRES=$GRES
MEM=$MEM
TIME=$TIME
SCRIPT_COMMIT=$(git -C "$SED_STORE" rev-parse HEAD)
LIB_ENV=$RD_MAIN (cuda) @ $(git -C "$RD_MAIN" rev-parse HEAD)
JAX_CACHE=$JAX_CACHE
SLURM_LOG=$SLURM_LOG/$JOB-repro.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
echo "submitted $JOB  ->  $OUT"

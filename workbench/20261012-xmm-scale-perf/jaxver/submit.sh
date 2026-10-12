#!/usr/bin/env bash
#
# jax-version timing (thread xmm-scale-perf, step 7). One A10G node runs, in
# sequence, the same three tests in each of three pixi envs of this worktree:
#
#   cuda         : conda-forge jax/jaxlib 0.7.2 (the locked production env; control)
#   cuda-cf010   : conda-forge jax/jaxlib 0.10.2, CUDA 12 build
#   cuda-pypi011 : PyPI jax[cuda12]==0.11.2
#
# Tests, per env:
#   1. benchmarks/bench_deposit.py + check_perf.py (synthetic hot path; the
#      ratio gate catches a 0.11.0-class scatter-add regression; absolute gate
#      vs baselines/gpu.json, A10G @ jax 0.7.2).
#   2. pytest tests/test_golden_frame.py tests/test_disperser_gpu.py on GPU.
#      The gpu-a10g golden frames predate #40, so the known coarse flips may
#      fail in every env; only a failure pattern that differs from the 0.7.2
#      control is attributable to jax.
#   3. build_dispersed_image.py on the step-6 h27 setup (XMM HEALPix store,
#      F158<=27 + r_hl>0 SER), Keith's part_03 row 0, SCAs 1-4, cold per-env
#      JAX cache: per-SCA JIT, ms/gal, GPU peak, and images to compare
#      against step-6 job 12270 (same code, jax 0.7.2).
# Running the control on the same node removes node-to-node scatter.
#
# Usage: bash submit.sh   (head node; 1 x g5.2xlarge, ~1 h, ~$1.2)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
STEP6=/mnt/roman-science/grs/20261012_xmm-scale-perf/timing/h27
OUT=/mnt/roman-science/grs/20261012_xmm-scale-perf/jaxver
SLURM_LOG=/data/npadman/tmp/slurm-logs/xmm-scale-perf
CACHE_ROOT=/data/npadman/tmp/jax-cache-xmm-jaxver
PART=gpu-med; GRES=gpu:a10g:1; MEM=28G; TIME=02:30:00
ENVS="cuda cuda-cf010 cuda-pypi011"

[ -z "$(git -C "$WT" status --porcelain -- src scripts workbench tests benchmarks pixi.toml pixi.lock)" ] \
    || { echo "worktree dirty; commit first"; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT exists; refusing to overwrite"; exit 1; }
mkdir -p "$OUT/slurm-meta" "$SLURM_LOG"
for E in $ENVS; do
    D=$OUT/$E; mkdir -p "$D/logs"
    cp "$STEP6/part_03.ecsv" "$D/part_03.ecsv"
    sed -e "s|^output_dir: .*|output_dir: $D/output|" -e "s|^scas: .*|scas: [1, 2, 3, 4]|" \
        "$STEP6/config.yaml" > "$D/config.yaml"
done

W=$OUT/slurm-meta/task.sh
cat > "$W" <<EOT
#!/usr/bin/env bash
set -uo pipefail   # no -e: one env failing must not stop the others
export ROMAN_DISPERSER_DATA=/data/grs/common/refdata/roman_disperser_data
export STPSF_PATH=/data/grs/common/refdata/stpsf-data
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
hostname
cd $WT
for E in $ENVS; do
    D=$OUT/\$E
    echo "=== \$E \$(date -u +%FT%TZ)"
    pixi run --frozen -e \$E python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, jax.devices())"
    export JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/\$E-bench
    pixi run --frozen -e \$E python benchmarks/bench_deposit.py --tag \$E --out \$D/bench.json > \$D/logs/bench.log 2>&1
    pixi run --frozen -e \$E python benchmarks/check_perf.py \$D/bench.json 2>&1 | tee \$D/logs/check_perf.log
    unset JAX_COMPILATION_CACHE_DIR
    echo "--- golden \$(date -u +%FT%TZ)"
    pixi run --frozen -e \$E python -m pytest -q tests/test_golden_frame.py tests/test_disperser_gpu.py > \$D/logs/pytest.log 2>&1
    tail -3 \$D/logs/pytest.log
    echo "--- pipeline \$(date -u +%FT%TZ)"
    pixi run --frozen -e \$E python scripts/build_dispersed_image.py \\
        --config \$D/config.yaml --pointings \$D/part_03.ecsv \\
        --cache-dir $CACHE_ROOT/\$E --gpu 0 \\
        --worker-index 0 --num-workers 9 --log-file \$D/logs/worker.log > \$D/logs/pipeline.out 2>&1
    echo "pipeline exit \$?"; tail -4 \$D/logs/worker.log
done
echo "=== done \$(date -u +%FT%TZ)"
EOT
chmod +x "$W"
JOB=$(sbatch --parsable --partition=$PART --gres=$GRES --cpus-per-task=4 \
    --mem=$MEM --time=$TIME --job-name=xmm-jaxver \
    --output="$SLURM_LOG/%j-jaxver.out" "$W")
cat > "$OUT/slurm-meta/jaxver-$JOB.env" <<EOT
JOB=$JOB
WORKTREE=$WT
CODE_COMMIT=$(git -C "$WT" rev-parse HEAD)
ENVS=$ENVS
CATALOG/CONFIG=$STEP6/config.yaml with scas [1,2,3,4], output_dir per env
PARTITION=$PART
GRES=$GRES
MEM=$MEM
TIME=$TIME
JAX_CACHE=$CACHE_ROOT/<env> (cold)
SLURM_LOG=$SLURM_LOG/$JOB-jaxver.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
echo "submitted: $JOB"

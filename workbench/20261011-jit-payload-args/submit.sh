#!/usr/bin/env bash
#
# Measure the payload-as-argument JIT speed-up (thread jit-payload-args) on A10G.
#
#   probe : compile_probe.py on SCAs 1 2 3 at production sampling (old closure
#           path vs new argument path, compilation cache OFF), then the perf
#           suite (bench_deposit.py + check_perf.py) to check that per-source
#           runtime did not regress.
#   C     : the same pointing as the HEALPix A/B's variant B (job 12131;
#           workbench/20261011-sed-store-ab): 4-roll VISIT 1, HEALPix
#           catalog, seed 42, same config, but with THIS branch's library +
#           script, and an EMPTY JAX cache. That is the worst case. B had 12131's
#           shared cache, so its 601 s of JIT was mostly cache load + trace.
#
# Everything (library and script) comes from this worktree's `cuda` env.
# Usage: bash submit.sh   (head node; 2 x g5.2xlarge, ~$1 total)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../.." && pwd)"
REF_RUN=/mnt/roman-science/grs/20261007_4roll-ra10dec0-v0151
REF_DIR=$REF_RUN/spectro/2026-10-07/4roll
ECSV=$REF_RUN/4roll-ra10dec0.sim.ecsv
B_CFG=/mnt/roman-science/grs/20261011_sed-store-ab/B/config.yaml
OUT=/mnt/roman-science/grs/20261011_jit-payload-ab
SLURM_LOG=/data/npadman/tmp/slurm-logs/jit-payload-ab
PART=gpu-med; GRES=gpu:a10g:1; MEM=24G

[ -z "$(git -C "$WT" status --porcelain -- src scripts benchmarks workbench pixi.toml pixi.lock)" ] \
    || { echo "worktree dirty; commit first"; exit 1; }
COMMIT=$(git -C "$WT" rev-parse HEAD)
mkdir -p "$OUT/slurm-meta" "$OUT/probe" "$OUT/C/logs" "$OUT/C/jax-cache" "$SLURM_LOG"
PIXI="pixi run --manifest-path $WT/pixi.toml -e cuda"

# --- probe + perf --------------------------------------------------------
cat > "$OUT/slurm-meta/task-probe.sh" <<EOT
#!/usr/bin/env bash
set -euo pipefail
source $REF_DIR/scripts/env.sh
unset JAX_COMPILATION_CACHE_DIR
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
cd $WT
PROBE_NWL=5501 $PIXI python workbench/20261011-jit-payload-args/compile_probe.py 1 2 3 \\
    2>&1 | tee $OUT/probe/compile_probe.txt
$PIXI python benchmarks/bench_deposit.py --out $OUT/probe/bench.json 2>&1 | tail -20
$PIXI python benchmarks/check_perf.py $OUT/probe/bench.json 2>&1 | tee $OUT/probe/check_perf.txt
EOT

# --- C: end-to-end pointing ---------------------------------------------
sed -e "s#^output_dir:.*#output_dir: $OUT/C/output#" "$B_CFG" > "$OUT/C/config.yaml"
cat > "$OUT/slurm-meta/task-C.sh" <<EOT
#!/usr/bin/env bash
set -euo pipefail
source $REF_DIR/scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
exec $PIXI python $WT/scripts/build_dispersed_image.py \\
    --config $OUT/C/config.yaml --pointings $ECSV --cache-dir $OUT/C/jax-cache --gpu 0 \\
    --worker-index 0 --num-workers 4 --log-file $OUT/C/logs/worker.log
EOT

for V in probe C; do
    T=01:00:00; [[ $V == C ]] && T=02:00:00
    chmod +x "$OUT/slurm-meta/task-$V.sh"
    JOB=$(sbatch --parsable --partition=$PART --gres=$GRES --cpus-per-task=4 \
        --mem=$MEM --time=$T --job-name=jitab-$V \
        --output="$SLURM_LOG/%j-$V.out" "$OUT/slurm-meta/task-$V.sh")
    cat > "$OUT/slurm-meta/jitab-$V-$JOB.env" <<EOT
JOB=$JOB
VARIANT=$V
PARTITION=$PART
GRES=$GRES
MEM=$MEM
TIME=$T
GIT_COMMIT=$COMMIT
WORKTREE=$WT (cuda env)
CONFIG=$OUT/C/config.yaml (from $B_CFG, output_dir changed)
JAX_CACHE=$OUT/C/jax-cache (empty at submit) for C; disabled for probe
SLURM_LOG=$SLURM_LOG/$JOB-$V.out
SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOT
    echo "$V: job $JOB"
done

#!/bin/bash
# 1-SCA reproduction of the slow 20260803 run: current main-line code
# (explore/performance worktree + catalog-schema shim), same catalog +
# pointing, pixi cuda env, DEFAULT allocator, pinned a10g. Samples GPU +
# host memory alongside, since Keith reported OOMs (his allocator motive).
set -uo pipefail
WORKDIR=/data/npadman/1-Projects/roman/roman_disperser/perf
REPRO=/data/npadman/1-Projects/roman/roman_disperser/perf/workbench/20260818-perf-exploration/repro
OUT=/data/npadman/tmp/perf-repro-20260803
PIXI=/home/npadman/.pixi/bin/pixi
export ROMAN_DISPERSER_DATA=/data/npadman/3-Resources/roman_disperser_data
export JAX_COMPILATION_CACHE_DIR=/data/npadman/jax-cache-grism

cd "$WORKDIR"
echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=== git: $(git rev-parse --short HEAD)"

# Memory sampler: GPU used/total + process RSS, every 10 s.
( while true; do
    ts=$(date -u +%H:%M:%S)
    gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    rss=$(ps -o rss= -C python | sort -rn | head -1)
    echo "$ts,$gpu,${rss:-0}"
    sleep 10
  done > "$OUT/memory-trace.csv" ) &
SAMPLER=$!

$PIXI run -e cuda python scripts/build_dispersed_image.py \
    --config "$REPRO/repro-sca1.yaml" \
    --pointings /mnt/roman-science/grs/20260803_test_catalog/994-hlwas-Feb26_XMM-LSS_GRISM_pointings_1.ecsv \
    --log-file "$OUT/repro-sca1.log"
RC=$?
kill $SAMPLER 2>/dev/null
echo "=== driver exit code: $RC"
tail -40 "$OUT/repro-sca1.log"
exit $RC

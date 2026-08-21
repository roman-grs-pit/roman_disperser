#!/bin/bash
# Performance-regression sweep over the golden jax versions.
#
# For each version in golden-jax-versions.txt: build a throwaway venv,
# install this package (editable) + jax[cuda12]==<version>, run
# bench_deposit.py, then gate every result with check_perf.py.
#
# Run this on a GPU node (the gates were calibrated on GPU; see
# bench_deposit.py). Requires hydrated reference data — either
# $ROMAN_DISPERSER_DATA or <repo>/data (see roman_disperser.paths.data_dir).
#
# Environment knobs:
#   BASE_PYTHON  python used to create the venvs   (default: python3)
#   WORKDIR      scratch dir for venvs + jit caches (default: ${TMPDIR:-/tmp}/perf-golden)
#   OUTDIR       where result JSONs land            (default: benchmarks/results)
#   BENCH_ARGS   extra args passed to bench_deposit.py (e.g. "--n-gal 300")
#
# Each version gets its own JAX_COMPILATION_CACHE_DIR: different jaxlib/XLA
# versions must not share (or poison) a compilation cache.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON=${BASE_PYTHON:-python3}
WORKDIR=${WORKDIR:-${TMPDIR:-/tmp}/perf-golden}
OUTDIR=${OUTDIR:-$REPO/benchmarks/results}
BENCH_ARGS=${BENCH_ARGS:-}

# A CUDA venv is ~8 GB; node-local /tmp cannot hold three plus a pip cache
# (first a10g run died with ENOSPC). Keep peak usage to one venv: no pip
# cache, and each venv is deleted after its leg.
export PIP_NO_CACHE_DIR=1

mkdir -p "$WORKDIR" "$OUTDIR"
echo "=== node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

RESULTS=()
grep -v '^\s*#' "$REPO/benchmarks/golden-jax-versions.txt" \
    | grep -v '^\s*$' > "$WORKDIR/versions"

while read -r VER; do
    if [ "$VER" = "latest" ]; then
        SPEC="jax[cuda12]"
        TAG="jax-latest"
    else
        SPEC="jax[cuda12]==$VER"
        TAG="jax-$VER"
    fi
    VENV="$WORKDIR/venv-$TAG"
    echo "=== $TAG: $SPEC"
    rm -rf "$VENV"
    "$BASE_PYTHON" -m venv "$VENV"
    # Subshell so each version's activation and env vars stay isolated.
    (
        source "$VENV/bin/activate"
        pip -q install -U pip
        pip -q install -e "$REPO" astropy "$SPEC"
        export JAX_COMPILATION_CACHE_DIR="$WORKDIR/jax-cache-$TAG"
        python -c "import jax; print('jax', jax.__version__, jax.devices())"
        python "$REPO/benchmarks/bench_deposit.py" $BENCH_ARGS \
            --tag "$TAG" --out "$OUTDIR/$TAG.json"
    )
    rm -rf "$VENV"
    RESULTS+=("$OUTDIR/$TAG.json")
done < "$WORKDIR/versions"

echo "=== gating ${#RESULTS[@]} result(s)"
"$BASE_PYTHON" "$REPO/benchmarks/check_perf.py" "${RESULTS[@]}"

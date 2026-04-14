#!/bin/bash
#
# ==========================================================================
# Acceptance Test: Full Grism Simulation on NERSC
# ==========================================================================
#
# Runs the complete grism simulation pipeline on a single NERSC Perlmutter
# GPU node (4x A100).  Processes 32 pointings from an APT-format ECSV
# pointing table, each with 18 SCAs, using the full Galacticus 4 deg^2 mock
# catalog (all 100 sub-samples, ~1.2M galaxies + ~87k stars).
#
# The pipeline:
#   1. Builds the unified source catalog from raw Galacticus data
#   2. Pads the catalog in RA for pointings that extend beyond the native
#      2-degree RA coverage
#   3. Copies performance-critical data to ramdisk (/dev/shm)
#   4. Warms up the JAX compilation cache across 4 GPUs in parallel
#   5. Processes all 32 pointings with 4 GPUs (round-robin distribution)
#
# Ramdisk layout (~45 GB of 128 GB available):
#   /dev/shm/grism/
#     catalogs/          Padded metadata (42 MB) + seds.zarr (16 GB)
#     psf_cache/         Pre-computed PSF grids (4.3 GB)
#     jax-cache/         JIT compilation cache (~24 GB, built during warmup)
#
# Persistent output goes to $CFS for long-term storage.
#
# --------------------------------------------------------------------------
# How to run
# --------------------------------------------------------------------------
#
# Request an interactive GPU node:
#   salloc -N 1 -G 4 -C gpu -t 4:00:00 -A m4943 -q interactive
#
# Then run the script:
#   cd $SCRATCH/roman_disperser
#   bash workbench/20260414-acceptance-testing/run.sh
#
# Each step is idempotent -- it checks for existing output and skips if
# already done.  To re-run from a specific step, comment out earlier steps.
#
# --------------------------------------------------------------------------

set -euo pipefail

# ==========================================================================
# Configuration -- edit these paths for your NERSC setup
# ==========================================================================

# Where the repo is checked out
REPO_DIR="$SCRATCH/roman_disperser"

# Raw Galacticus mock (all 100 sub-samples, HDF5 + FITS index)
GALACTICUS_DIR="$CFS/m4943/grismsim/galacticus_4deg2_mock"

# Working directory for persistent artifacts (catalogs, output, logs)
WORK_DIR="$CFS/m4943/grismsim/acceptance-testing"

# Ramdisk for ephemeral high-performance data.
# Everything here is lost when the job ends -- that's fine, the expensive
# artifacts (catalog, output FITS) live on $CFS.  The JAX cache is rebuilt
# in ~5 min during warmup.
RAMDISK="/dev/shm/grism"

# Number of GPUs on this node (Perlmutter GPU nodes have 4x A100-40GB)
NGPU=4

# Pixi package cache -- keep off $HOME (tight quota on NERSC).
# $SCRATCH is fast and large; pixi can always reinstall from the lockfile.
export PIXI_CACHE_DIR="$SCRATCH/.pixi-cache"


# ==========================================================================
# Step 0: Environment
# ==========================================================================
#
# Install pixi (a conda-like package manager) if not present, then install
# the cuda environment.  Pixi uses the repo's pixi.toml to create a
# self-contained environment with JAX, CUDA, and all dependencies.
#
# PIXI_CACHE_DIR is set to $SCRATCH to avoid filling the $HOME quota.
#
echo "=== Step 0: Environment ==="

cd "$REPO_DIR"

if ! command -v pixi &> /dev/null; then
    echo "Installing pixi (single binary, no root needed)..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

pixi install -e cuda

echo "Verifying GPU access..."
pixi run -e cuda python -c "import jax; print(f'Backend: {jax.default_backend()}'); print(jax.devices())"


# ==========================================================================
# Step 1: Build the full source catalog
# ==========================================================================
#
# Converts the raw Galacticus mock (HDF5 galaxy SEDs + star catalog) into
# the unified catalog format: metadata.parquet (positions, morphology,
# flux scaling) + seds.zarr (SEDs in Zarr v3 with sharding for fast
# random access).
#
# All 100 Galacticus sub-samples are included.  Each sub-sample covers the
# same 2x2 deg^2 field (RA 9-11, Dec -1 to +1) -- they are independent
# random realizations, not spatial tiles.
#
# This is the slowest step (~TBD, depends on Galacticus I/O).  The output
# is written to $CFS and reused across runs.
#
echo "=== Step 1: Build source catalog ==="

CATALOG_DIR="$WORK_DIR/catalogs"

if [ -f "$CATALOG_DIR/metadata.parquet" ]; then
    echo "Catalog already exists at $CATALOG_DIR, skipping."
else
    mkdir -p "$CATALOG_DIR"
    pixi run -e cuda python scripts/build_source_catalog.py \
        --sims 1-100 \
        --galacticus-dir "$GALACTICUS_DIR" \
        --star-dir data/stars \
        --output-dir "$CATALOG_DIR"
fi


# ==========================================================================
# Step 2: Pad the catalog in RA
# ==========================================================================
#
# The source catalog covers RA [9, 11] (2 degrees).  Our 32 pointings span
# RA ~8.7 to ~11.2, and each pointing's cone search extends 0.6 deg beyond
# the center, so we need catalog coverage from ~8.1 to ~11.8.
#
# The pad script replicates the catalog periodically in RA to fill [8, 12].
# Only the metadata parquet is duplicated (with shifted RA values); the
# seds.zarr is symlinked since SED indices are unchanged.
#
# The periodic box boundaries (--ra-box-min/max) must match the simulation
# box, not the source positions.  Our Galacticus mock uses a [9, 11] box.
#
echo "=== Step 2: Pad catalog ==="

PADDED_DIR="$WORK_DIR/catalogs_padded"

if [ -f "$PADDED_DIR/metadata.parquet" ]; then
    echo "Padded catalog already exists at $PADDED_DIR, skipping."
else
    pixi run -e cuda python scripts/pad_catalog.py \
        --input "$CATALOG_DIR" \
        --output "$PADDED_DIR" \
        --ra-box-min 9.0 --ra-box-max 11.0 \
        --ra-min 8.0 --ra-max 12.0
fi


# ==========================================================================
# Step 3: Download PSF caches
# ==========================================================================
#
# Pre-computed PSF grids (36 files, ~4.3 GB): one per (SCA, grism order)
# combination.  Each grid contains wavelength- and position-dependent PSFs
# for trilinear interpolation during dispersion.
#
# Downloaded from a public GitHub release.  Only needs to run once.
#
echo "=== Step 3: PSF caches ==="

if [ -d "$REPO_DIR/data/psf_cache" ] && [ "$(ls $REPO_DIR/data/psf_cache/*.npz 2>/dev/null | wc -l)" -eq 36 ]; then
    echo "PSF caches already present (36 files), skipping."
else
    pixi run -e cuda python scripts/download_psf_caches.py
fi


# ==========================================================================
# Step 4: Copy data to ramdisk
# ==========================================================================
#
# Copy the padded catalog and PSF caches to /dev/shm for fast I/O during
# the simulation.  The ramdisk is node-local and ephemeral (lost when the
# job ends), but eliminates filesystem latency for the two hot data paths:
#
#   - Catalog metadata: scanned fully during cone search for each pointing
#   - Galaxy SEDs: random-access reads (sharded Zarr) during dispersion
#   - PSF caches: loaded once per SCA (~120 MB each)
#
# Total ramdisk usage: ~45 GB (of 128 GB available on Perlmutter GPU nodes)
#
echo "=== Step 4: Copy to ramdisk ==="

mkdir -p "$RAMDISK"

# Padded catalog -> ramdisk (metadata + full seds.zarr)
# The padded catalog's seds.zarr is a symlink to the original; resolve and
# copy the actual data so reads hit ramdisk, not $CFS.
if [ ! -f "$RAMDISK/catalogs/metadata.parquet" ]; then
    echo "Copying padded catalog to ramdisk (~16 GB)..."
    mkdir -p "$RAMDISK/catalogs"
    cp "$PADDED_DIR/metadata.parquet" "$RAMDISK/catalogs/"
    cp -r "$(readlink -f "$PADDED_DIR/seds.zarr")" "$RAMDISK/catalogs/seds.zarr"
fi

# PSF caches -> ramdisk (~4.3 GB)
if [ ! -d "$RAMDISK/psf_cache" ]; then
    echo "Copying PSF caches to ramdisk (~4.3 GB)..."
    cp -r "$REPO_DIR/data/psf_cache" "$RAMDISK/psf_cache"
fi

echo "Ramdisk contents:"
du -sh "$RAMDISK"/*


# ==========================================================================
# Step 5: Write pipeline config
# ==========================================================================
#
# The config YAML tells the pipeline where to find data and where to write
# output.  All I/O-hot paths point to ramdisk; output goes to $CFS.
#
# The JAX compilation cache (cache_dir) also lives on ramdisk.  It's built
# during the warmup step and reused by all 4 GPUs during processing.
#
echo "=== Step 5: Config ==="

POINTINGS="$REPO_DIR/workbench/20260414-acceptance-testing/pointings.ecsv"
OUTPUT_DIR="$WORK_DIR/output"
CACHE_DIR="$RAMDISK/jax-cache"
LOG_DIR="$WORK_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

CONFIG="$WORK_DIR/config.yaml"
cat > "$CONFIG" <<EOF
output_dir: $OUTPUT_DIR
seed: 42
scas: all
cone_radius: 0.6
star_batch_size: 1000
galaxy_batch_size: 100
galaxy_npix: 30
cache_dir: $CACHE_DIR
catalog_dir: $RAMDISK/catalogs
sensitivity_dir: $REPO_DIR/data/sensitivities
optical_model: $REPO_DIR/data/Roman_grism_OpticalModel_v0.8.yaml
psf_cache_dir: $RAMDISK/psf_cache
EOF

echo "Config written to $CONFIG"
cat "$CONFIG"


# ==========================================================================
# Step 6: Warm up the JIT compilation cache
# ==========================================================================
#
# JAX compiles specialized GPU kernels the first time each function is
# called (~10s per function, 6 functions per SCA, 18 SCAs = ~18 min cold).
# Once compiled, functions are cached to disk and load in ~2.5s each.
#
# We split the 18 SCAs across 4 GPUs to parallelize the cold compilation.
# Each GPU compiles its share of SCAs (round-robin: GPU 0 gets SCAs
# 1,5,9,13,17; GPU 1 gets 2,6,10,14,18; etc.).  All write to the same
# shared cache directory on ramdisk.
#
# After warmup, the processing step loads cached functions in ~15s/SCA
# instead of ~60s/SCA.
#
echo "=== Step 6: JIT warmup ==="

for i in $(seq 0 $((NGPU - 1))); do
    pixi run -e cuda python scripts/build_grism_image.py \
        --config "$CONFIG" \
        --warmup-only \
        --gpu $i --worker-index $i --num-workers $NGPU \
        --log-file "$LOG_DIR/warmup_gpu${i}.log" &
done
echo "Waiting for warmup (4 GPUs compiling 18 SCAs, ~5 min)..."
wait
echo "Warmup complete."

for i in $(seq 0 $((NGPU - 1))); do
    echo "  GPU $i: $(tail -1 "$LOG_DIR/warmup_gpu${i}.log")"
done


# ==========================================================================
# Step 7: Process all 32 pointings
# ==========================================================================
#
# Each GPU processes every 4th pointing (round-robin).  Within each
# pointing, all 18 SCAs are processed sequentially on the assigned GPU.
#
# Per pointing, the pipeline:
#   1. Cone search: find sources within 0.6 deg of pointing center
#   2. Per-SCA: load PSF, build dispersers (from JIT cache), select
#      sources whose traces overlap detector, generate spectra, disperse
#      stars and galaxies, apply Poisson noise, write FITS + PNG
#   3. Write mosaic PNG and metadata YAML
#
# Output directories are named by APT identifiers from the ECSV table,
# so workers never collide.  RNG keys are derived from APT identifiers
# (not pointing order), so results are identical regardless of how
# pointings are distributed across GPUs.
#
# Each GPU logs to its own file for clean separation.
#
echo "=== Step 7: Process pointings ==="

for i in $(seq 0 $((NGPU - 1))); do
    pixi run -e cuda python scripts/build_grism_image.py \
        --config "$CONFIG" \
        --pointings "$POINTINGS" \
        --gpu $i --worker-index $i --num-workers $NGPU \
        --log-file "$LOG_DIR/run_gpu${i}.log" &
done
echo "Waiting for processing (32 pointings x 18 SCAs across 4 GPUs)..."
wait
echo "Processing complete."

for i in $(seq 0 $((NGPU - 1))); do
    echo "  GPU $i: $(tail -3 "$LOG_DIR/run_gpu${i}.log")"
done


# ==========================================================================
# Done
# ==========================================================================
echo ""
echo "=== Acceptance test complete ==="
echo ""
echo "Output:   $OUTPUT_DIR"
echo "Logs:     $LOG_DIR"
echo "Config:   $CONFIG"
echo ""
n_dirs=$(ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | wc -l)
echo "Pointing directories: $n_dirs (expected 32)"
echo ""
echo "To inspect a single pointing:"
echo "  ls $OUTPUT_DIR/acceptance-testing.sim_001.001.001.001.001.001/"
echo ""
echo "Ramdisk cleanup (automatic on job exit, or manual):"
echo "  rm -rf $RAMDISK"

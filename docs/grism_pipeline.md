# Grism Pipeline (`scripts/build_grism_image.py`)

Simulates Roman Space Telescope dispersed images — G150 **grism** (default) or P127 **prism** (`element: prism`) — from a unified source catalog containing both stars and galaxies. Disperses point sources (via `star_disperser`) and extended sources (via `galaxy_disperser` with per-source Sersic morphologies) through the optical model with wavelength-dependent PSFs, applies per-SCA sensitivity curves, and produces noiseless and Poisson-sampled FITS images.

## Overview

The script has five modes:

1. **Quick mode** -- single pointing, single SCA. Runs `setup_pipeline` + `process_pointing` as a convenience wrapper.
2. **Batch mode** -- YAML config + ECSV pointing table (APT format).
3. **`--warmup-only`** -- compile JIT functions for all SCAs and exit (no dispersion).
4. **`--mosaic`** -- generates a focal-plane mosaic PNG from an existing pointing directory (no dispersion).
5. **`--generate-config`** -- writes a documented template YAML config and exits.

## Quick Start

```bash
# Quick mode: single SCA
pixi run -e cuda python scripts/build_grism_image.py \
    --pointing-ra 9.5 --pointing-dec 0.95 --pointing-pa 0.0 \
    --sca 5 --output my_field.fits --seed 42

# Batch mode: config + ECSV pointing table
pixi run -e cuda python scripts/build_grism_image.py \
    --config scripts/example_grism_config.yaml \
    --pointings pointings.ecsv

# Generate a template config
pixi run -e cuda python scripts/build_grism_image.py \
    --generate-config my_config.yaml

# Warmup JIT cache (single GPU)
pixi run -e cuda python scripts/build_grism_image.py \
    --config scripts/example_grism_config.yaml \
    --warmup-only --gpu 0 --cache-dir /path/to/jax-cache

# Generate mosaic from existing pointing directory
pixi run -e cuda python scripts/build_grism_image.py \
    --mosaic /path/to/pointing_dir
```

## Output Files

### Directory Structure (Batch Mode)

Output directories are named from the ECSV filename and APT identifiers:

```
output_dir/
  {ecsv_basename}_{plan}.{pass}.{segment}.{observation}.{visit}.{exposure}/
    grism_{dirname}_detSCA01.fits
    grism_{dirname}_detSCA01.png
    ...
    grism_{dirname}_detSCA18.fits
    grism_{dirname}_detSCA18.png
    grism_{dirname}_mosaic.png
    grism_{dirname}_sources.parquet
    grism_{dirname}_meta.yaml
```

For example, with `--pointings galacticus.ecsv` and a row with plan=1, pass=1,
segment=1, observation=1, visit=1, exposure=1, the directory would be
`galacticus_001.001.001.001.001.001/`.

Quick mode writes a single FITS, PNG, source manifest, and metadata YAML at the user-specified path.

### FITS Structure

| Extension | Name    | Description |
|-----------|---------|-------------|
| 0         | PRIMARY | Empty data; metadata headers only |
| 1         | MODEL   | Noiseless count-rate image (counts/s), float32 [4088x4088] |
| 2         | ISIM    | Poisson-sampled image (counts), float32 [4088x4088] |

**Primary header fields:**

| Keyword    | Description |
|------------|-------------|
| WFICENRA   | Pointing RA [deg] |
| WFICENDEC  | Pointing Dec [deg] |
| WFICENPA   | Position angle [deg] |
| DETNUM     | SCA number (1-18) |
| EXPTIME    | Exposure time [s] |
| SEED       | Top-level RNG seed |
| OPTELEM    | Dispersing element (`grism` / `prism`) |
| RNDSEED0   | JAX RNG key word 0 (per-SCA) |
| RNDSEED1   | JAX RNG key word 1 (per-SCA) |
| CODEVER    | roman_disperser package version |
| GITSHA     | Git commit SHA of pipeline code; `-dirty` suffix if the working tree had uncommitted changes |
| MA_TABLE   | MA table number from APT (batch mode) |
| PLAN       | APT plan number (batch mode) |
| PASS       | APT pass number (batch mode) |
| SEGMENT    | APT segment number (batch mode) |
| OBS        | APT observation number (batch mode) |
| VISIT      | APT visit number (batch mode) |
| EXPOSURE   | APT exposure number (batch mode) |

### Source Manifest (Parquet)

Per-pointing Parquet file listing every source dispersed onto each SCA. One row per (source, SCA, order) combination.

| Column         | Type   | Description |
|----------------|--------|-------------|
| catalog_index  | int    | Row index into the unified source catalog (`metadata.parquet`) |
| sca            | int    | SCA number (1-18) |
| order          | str    | Spectral order ("0", "1", "2") |
| type           | str    | "PSF" (star) or "SER" (galaxy/Sersic) |
| xsca           | float  | Undispersed SCA x position [pixels, 1-indexed FITS] |
| ysca           | float  | Undispersed SCA y position [pixels, 1-indexed FITS] |
| ra             | float  | Source RA [deg] |
| dec            | float  | Source Dec [deg] |
| flux_scale     | float  | SED multiplier applied at dispersion. Stars: equals `F158` (the SED template is unit-normalized in F158). Galaxies: always 1.0 (per-source SED is already F158-normalized). See `data/catalogs/README.md` for the SED scaling convention. |
| F158           | float  | F158 apparent flux in maggies (linear AB units; `mag = -2.5·log10(F158)`). Same value for all rows belonging to one source. |

### PNG Quicklook Images

Per-SCA PNGs use asinh stretch (`AsinhNorm` with `linear_width=0.01`), inferno colormap, 4x block-averaged (4088->1022 pixels), flipped to origin-lower. PNGs are rendered from the noiseless MODEL for clean visualization.

### Focal-Plane Mosaic PNG

Generated when more than one SCA is processed. Shows all SCAs arranged in the WFI focal plane layout using FPA center coordinates. Each thumbnail is 8x block-averaged. Global asinh normalization across all SCAs.

### Per-Pointing Metadata YAML

Written to `grism_<name>_meta.yaml`. Fields:

| Field | Description |
|-------|-------------|
| `pointing` | `{ra, dec, pa}` in degrees |
| `exptime` | Exposure time in seconds |
| `seed` | Top-level seed |
| `pointing_key` | JAX RNG key data for this pointing (list of 2 ints) |
| `sca_keys` | Mapping of SCA number -> JAX RNG key data |
| `dlam_angstroms` | Wavelength spacing |
| `cone_radius` | Cone search radius in degrees |
| `star_batch_size` | Stars per JIT batch |
| `galaxy_batch_size` | Galaxies per JIT batch |
| `galaxy_npix` | Sersic image size in native pixels |
| `oversample` | PSF oversampling factor |
| `source_counts` | Per-SCA dict mapping order -> {stars, galaxies} counts |
| `codever` | roman_disperser package version |
| `git_sha` | Git commit SHA of pipeline code (`-dirty` suffix if uncommitted changes) |
| `pointing_file` | ECSV filename (batch mode) |
| `apt` | APT identifiers: plan, pass, segment, observation, visit, exposure, ma_table_number (batch mode) |

## RNG Reproducibility

### Batch Mode

Per-pointing RNG keys are derived deterministically from three inputs:

1. The top-level **seed** (from the config YAML)
2. The **pointing filename** (basename of the ECSV file, used as salt)
3. The **APT identifiers** (plan, pass, segment, observation, visit, exposure)

This design ensures:
- **Deterministic**: same seed + same ECSV file + same exposure → same output
- **Slice-invariant**: processing a subset of pointings does not change any key
- **File-isolated**: different ECSV files produce different keys even with the same seed

The per-pointing key is then folded with each SCA *number* to give per-SCA
keys for Poisson sampling (`pipeline.make_sca_keys`):

```
make_pointing_key(seed, filename, plan, pass, segment, obs, visit, exposure)
  -> jax.random.fold_in(key, sca_num)   # one key per SCA, by SCA number
     -> jax.random.poisson(...)         # Poisson draw for that SCA
```

Because the fold uses the SCA number rather than the position in the SCA
list, slice-invariance extends to SCAs: a `scas: [5]` run draws from the
same key as a full 18-SCA run, removing the RNG obstacle to 1-SCA
regression gates. **Before v0.13.0 the keys came from `jax.random.split`
indexed by list position (issue #20), so ISIM realisations from earlier
versions differ by construction; the noiseless MODEL extension is
unaffected.**

Note that identical keys do not imply bit-identical products on GPU:
scatter-add accumulation is non-deterministic at the float32-epsilon level
(~1e-7 relative in MODEL run-to-run, measured on a10g), which can flip a
handful of Poisson counts. Compare GPU products with `np.allclose` /
relative-sum tolerances rather than bitwise; `--xla_gpu_deterministic_ops`
restores bit-identity but at a >100x slowdown.

Per-SCA keys are stored in both FITS headers (`RNDSEED0`/`RNDSEED1`) and the metadata YAML, so individual SCAs can be reproduced by reconstructing the key:

```python
import jax
import jax.numpy as jnp

key = jax.random.wrap_key_data(jnp.array([rndseed0, rndseed1], dtype=jnp.uint32))
```

### Quick Mode

Quick mode uses `jax.random.key(seed)` directly as the pointing key. Quick-mode and batch-mode outputs for the same seed will differ.

## Configuration Reference

Batch mode takes two input files: a YAML config (`--config`) and an ECSV pointing table (`--pointings`).

### YAML Config

Simulation parameters and data paths (see `scripts/example_grism_config.yaml`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str | *(required)* | Top-level output directory |
| `seed` | int | *(required)* | RNG seed for reproducibility |
| `element` | str | `grism` | Dispersing element: `grism` or `prism`. Sets the orders, band, default data paths, STPSF filters, and which `BANDPASS` rows of the pointing table are processed. The optical model is validated against it at load time (band/orders/`optical_element` mismatch raises). |
| `scas` | `"all"` or list[int] | `"all"` | SCAs to simulate (1-18) |
| `star_batch_size` | int | 1000 | Stars per JIT batch |
| `galaxy_batch_size` | int | 100 | Galaxies per JIT batch |
| `galaxy_npix` | int | 30 | Sersic image size in native pixels |
| `cone_radius` | float | 0.6 | Initial cone search radius [deg] |
| `cache_dir` | str | `/tmp/jax-cache-grism` | JAX compilation cache directory |
| `catalog_dir` | str | `data/catalogs` | Path to unified catalog directory |
| `sensitivity_dir` | str | per-element | Path to sensitivity FITS files (`data/sensitivities` grism, `data/sensitivities_prism` prism) |
| `optical_model` | str | per-element | Optical model YAML (`Roman_grism_OpticalModel_v0.8.yaml` / `Roman_prism_OpticalModel_v0.8.yaml`) |
| `psf_cache_dir` | str | `data/psf_cache` | Path to PSF cache directory |

**Deprecated:** `batch_size` is accepted as an alias for `star_batch_size` with a warning.

### ECSV Pointing Table

An astropy ECSV file with APT-format columns. Rows whose `BANDPASS` does not match the active element (`GRISM` / `PRISM`) are filtered out automatically. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `RA` | float | Pointing RA [deg] |
| `DEC` | float | Pointing Dec [deg] |
| `PA` | float | Position angle [deg] |
| `EXPOSURE_TIME` | float | Exposure time [s] (per-pointing) |
| `PLAN` | int | APT plan number |
| `PASS` | int | APT pass number |
| `SEGMENT` | int | APT segment number |
| `OBSERVATION` | int | APT observation number |
| `VISIT` | int | APT visit number |
| `EXPOSURE` | int | APT exposure number |
| `BANDPASS` | str | Filter name (only rows matching the active element are processed) |
| `MA_TABLE_NUMBER` | int | MA table number (stored in FITS header) |

## Catalog Format

The pipeline reads a unified source catalog from `catalog_dir` containing:

- **`metadata.parquet`** -- per-source metadata (type, position, morphology, flux scaling).
- **`seds.zarr/`** -- Zarr v3 store with wavelength grid, star SEDs, and galaxy SEDs.

See `data/catalogs/README.md` for the full format specification.

## Pipeline Architecture

### Memory-Efficient Per-SCA Processing

PSF payloads, dispersers, and JIT-compiled functions are built per-SCA and released after processing, so only one SCA's compiled code lives in memory at a time (~2-3 GB vs ~18+ GB for all 18 SCAs). Galaxy SEDs are also loaded per-SCA to avoid OOM with large catalogs.

The on-disk JAX compilation cache (default `/tmp/jax-cache-grism`, configurable via `--cache-dir`) makes subsequent runs fast (~2.5s/fn vs ~10s first compile). The default cache is cleared on reboot; set a persistent path to keep it.

### `setup_pipeline(sca_list, ...)`

One-time initialization shared across all pointings:

1. Build wavelength grid from `dlam_angstroms` over 0.9-2.0 um.
2. Load unified source catalog (Parquet metadata + Zarr SEDs).
3. Load optical model.
4. Pre-load star SEDs (all templates, small memory footprint).
5. Per-SCA optical payloads and sensitivity curves are prepared but PSF loading and JIT compilation are deferred.

Returns a `pipeline` dict consumed by `process_pointing`.

### `process_pointing(pipeline, ra, dec, pa, output_dir, ...)`

Per-pointing processing:

1. **Cone search** -- Haversine filter to `cone_radius` degrees around pointing center.
2. **Sky->FPA** -- convert (RA, Dec) to FPA coordinates via `omj.get_fpa_pos`.
3. **Per-SCA loop:**
   - Load PSF payloads and build star/galaxy dispersers for this SCA.
   - JIT warmup (hits disk cache after first run).
   - `select_sources_per_order` finds sources whose dispersed traces overlap the detector.
   - Generate star spectra from pre-loaded templates; load galaxy SEDs from Zarr.
   - Generate per-source Sersic morphology images.
   - Disperse stars and galaxies per order via batched `fori_loop`.
   - Poisson sampling: `MODEL * exptime` -> `jax.random.poisson`.
   - Write FITS, PNG, and source manifest.
   - Release compiled functions and PSF payloads.
4. Write mosaic PNG (if multiple SCAs) and metadata YAML.

### JIT Compilation Strategy

Each `(SCA, order)` combination gets a `fori_loop` compiled with a fixed batch array shape. The loop body captures the disperser, sensitivity curve, wavelength array, and `dlam` in a closure. Compilation happens once per SCA (cached to disk); all subsequent pointings reuse compiled code. See `docs/jit_compilation.md` for the general closure pattern.

## Multi-GPU and Parallel Execution

The pipeline supports parallel execution across multiple GPUs on the same machine. Work is distributed at the pointing level: each worker processes every Nth pointing in round-robin order.

### Flags

| Flag | Description |
|------|-------------|
| `--gpu N` | Select GPU device (sets `CUDA_VISIBLE_DEVICES` before JAX init) |
| `--worker-index I` | This worker's index (0-based) |
| `--num-workers K` | Total number of workers |
| `--warmup-only` | Compile JIT functions and exit (no catalog or pointings needed) |
| `--cache-dir PATH` | JAX compilation cache directory |
| `--log-file PATH` | Redirect all output to a file |

`--worker-index` and `--num-workers` must be used together. In batch mode they partition pointings (`index % K == I`); in warmup mode they partition SCAs.

### Workflow

A typical multi-GPU run has two steps:

**Step 1: Warm up the JIT cache.** Each GPU compiles a subset of SCAs in parallel, writing to a shared cache directory. This avoids redundant compilation when all workers start simultaneously.

```bash
CACHE=/path/to/jax-cache

for i in 0 1 2 3; do
  pixi run -e cuda python scripts/build_grism_image.py \
    --config cfg.yaml --warmup-only \
    --gpu $i --worker-index $i --num-workers 4 \
    --cache-dir $CACHE \
    --log-file logs/warmup_gpu$i.log &
done
wait
```

**Step 2: Run the simulation.** Each GPU processes its share of pointings, reusing the warm cache.

```bash
for i in 0 1 2 3; do
  pixi run -e cuda python scripts/build_grism_image.py \
    --config cfg.yaml --pointings pointings.ecsv \
    --gpu $i --worker-index $i --num-workers 4 \
    --cache-dir $CACHE \
    --log-file logs/run_gpu$i.log &
done
wait
```

### Cache Directory

The JAX compilation cache stores compiled XLA functions on disk. Precedence:

1. CLI `--cache-dir`
2. YAML config `cache_dir`
3. Environment variable `JAX_COMPILATION_CACHE_DIR`
4. Default: `/tmp/jax-cache-grism` (cleared on reboot)

For multi-GPU runs, use a shared path (all workers on the same machine share the filesystem). For persistence across reboots, point to a non-`/tmp` directory.

### Timing Reference

Per-SCA JIT compilation (3 orders x 2 functions = 6 compilations):

| Scenario | Time per SCA | 18 SCAs |
|----------|-------------|---------|
| Cold (no cache) | ~60s | ~18 min |
| Warm (from cache) | ~15s | ~4.5 min |

With 4-GPU parallel warmup, cold compile drops to ~5 min.

### Notes

- Workers can exceed GPUs: `--num-workers 8 --gpu 0` runs 8 sequential slices on GPU 0.
- `--worker-index` / `--num-workers` are independent of `--gpu` -- you can partition work without selecting a GPU, or vice versa.
- Output directories are named by APT identifiers, so workers never collide on output files.
- RNG keys are derived from APT identifiers, so results are identical regardless of how pointings are partitioned across workers.

## Per-Element Constants and Remaining Hardcoded Assumptions

Element-dependent values live on `roman_disperser.elements.DispersingElement`
(no longer hardcoded):

- **Wavelength range:** 0.9-2.0 um (grism) / 0.75-1.85 um (prism) — must match the optical-model YAML (validated at load).
- **Orders:** `["0", "1", "2"]` (grism) / `["1"]` (prism).
- **PSF filters:** grism order 2 reuses the order-1 (`GRISM1`) PSF cache — STPSF only provides `GRISM0`/`GRISM1`; the prism uses `PRISM`.

Still hardcoded:

- **FPA->SCA conversion:** uses the order `"1"` optical payload for the undispersed position (order `"1"` exists for both elements)
- **Normalization band:** F158 (`roman, wfi, f158`)
- **Detector size:** 4088 x 4088 pixels
- **Galaxy morphology:** Sersic profiles generated at `oversample` x resolution
- **Output naming:** the `grism_` filename prefix is kept for both elements (downstream `roman_l2_job` drivers and archived-run comparisons depend on it); products are distinguished by the `OPTELEM` header and `element:` meta field

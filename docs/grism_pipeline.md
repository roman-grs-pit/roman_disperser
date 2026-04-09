# Grism Pipeline (`scripts/build_grism_image.py`)

Simulates Roman Space Telescope grism images from a unified source catalog containing both stars and galaxies. Disperses point sources (via `star_disperser`) and extended sources (via `galaxy_disperser` with per-source Sersic morphologies) through the optical model with wavelength-dependent PSFs, applies per-SCA sensitivity curves, and produces noiseless and Poisson-sampled FITS images.

## Overview

The script has four modes:

1. **Quick mode** -- single pointing, single SCA. Runs `setup_pipeline` + `process_pointing` as a convenience wrapper.
2. **Batch mode** -- YAML config specifying multiple pointings and SCAs.
3. **`--mosaic`** -- generates a focal-plane mosaic PNG from an existing pointing directory (no dispersion).
4. **`--generate-config`** -- writes a documented template YAML config and exits.

## Quick Start

```bash
# Quick mode: single SCA
pixi run -e cuda python scripts/build_grism_image.py \
    --pointing-ra 9.5 --pointing-dec 0.95 --pointing-pa 0.0 \
    --sca 5 --output my_field.fits --seed 42

# Batch mode: multiple pointings/SCAs from config
pixi run -e cuda python scripts/build_grism_image.py \
    --config scripts/example_grism_config.yaml

# Generate a template config
pixi run -e cuda python scripts/build_grism_image.py \
    --generate-config my_config.yaml

# Generate mosaic from existing pointing directory
pixi run -e cuda python scripts/build_grism_image.py \
    --mosaic /path/to/pointing_dir
```

## Output Files

### Directory Structure (Batch Mode)

```
output_dir/
  pointing_name/
    grism_pointing_name_detSCA01.fits
    grism_pointing_name_detSCA01.png
    ...
    grism_pointing_name_detSCA18.fits
    grism_pointing_name_detSCA18.png
    grism_pointing_name_mosaic.png
    grism_pointing_name_sources.parquet
    grism_pointing_name_meta.yaml
```

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
| RNDSEED0   | JAX RNG key word 0 (per-SCA) |
| RNDSEED1   | JAX RNG key word 1 (per-SCA) |

### Source Manifest (Parquet)

Per-pointing Parquet file listing every source dispersed onto each SCA. One row per (source, SCA, order) combination.

| Column         | Type   | Description |
|----------------|--------|-------------|
| catalog_index  | int    | Index into the source catalog |
| sca            | int    | SCA number (1-18) |
| order          | str    | Spectral order ("0", "1", "2") |
| type           | str    | "PSF" (star) or "SER" (galaxy/Sersic) |
| xsca           | float  | Undispersed SCA x position |
| ysca           | float  | Undispersed SCA y position |
| ra             | float  | Source RA [deg] |
| dec            | float  | Source Dec [deg] |
| flux_scale     | float  | Flux scaling factor |
| F158           | float  | F158 AB magnitude |

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

## RNG Reproducibility

The seed produces a fully deterministic chain of JAX PRNG keys:

```
seed
  -> jax.random.key(seed)
  -> jax.random.split(n_pointings)     # one key per pointing
     -> jax.random.split(n_scas)       # one key per SCA within pointing
        -> jax.random.poisson(...)     # Poisson draw for that SCA
```

Same seed always produces identical output. Per-SCA keys are stored in both FITS headers (`RNDSEED0`/`RNDSEED1`) and the metadata YAML, so individual SCAs can be reproduced by reconstructing the key:

```python
import jax
import jax.numpy as jnp

key = jax.random.wrap_key_data(jnp.array([rndseed0, rndseed1], dtype=jnp.uint32))
```

**Note:** Quick mode uses `jax.random.key(seed)` directly as the pointing key (no split over pointings), so quick-mode and batch-mode outputs for the same seed will differ.

## Configuration Reference

All fields for the batch-mode YAML config (see `scripts/example_grism_config.yaml`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str | *(required)* | Top-level output directory |
| `seed` | int | *(required)* | RNG seed for reproducibility |
| `exptime` | float | 190.22 | Exposure time in seconds |
| `scas` | `"all"` or list[int] | `"all"` | SCAs to simulate (1-18) |
| `pointings` | list | *(required)* | List of `{name, ra, dec, pa}` dicts |
| `star_batch_size` | int | 1000 | Stars per JIT batch |
| `galaxy_batch_size` | int | 100 | Galaxies per JIT batch |
| `galaxy_npix` | int | 30 | Sersic image size in native pixels |
| `cone_radius` | float | 0.6 | Initial cone search radius [deg] |
| `catalog_dir` | str | `data/catalogs` | Path to unified catalog directory |
| `sensitivity_dir` | str | `data/sensitivities` | Path to sensitivity FITS files |
| `optical_model` | str | `data/Roman_grism_OpticalModel_v0.8.yaml` | Optical model YAML |
| `psf_cache_dir` | str | `data/psf_cache` | Path to PSF cache directory |

**Deprecated:** `batch_size` is accepted as an alias for `star_batch_size` with a warning.

## Catalog Format

The pipeline reads a unified source catalog from `catalog_dir` containing:

- **`metadata.parquet`** -- per-source metadata (type, position, morphology, flux scaling).
- **`seds.zarr/`** -- Zarr v3 store with wavelength grid, star SEDs, and galaxy SEDs.

See `data/catalogs/README.md` for the full format specification.

## Pipeline Architecture

### Memory-Efficient Per-SCA Processing

PSF payloads, dispersers, and JIT-compiled functions are built per-SCA and released after processing, so only one SCA's compiled code lives in memory at a time (~2-3 GB vs ~18+ GB for all 18 SCAs). Galaxy SEDs are also loaded per-SCA to avoid OOM with large catalogs.

The on-disk JAX compilation cache (`/tmp/jax-cache-grism`) makes subsequent runs fast (~2.5s/fn vs ~10s first compile). The cache is cleared on reboot.

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

## Hardcoded Assumptions

- **Wavelength range:** 0.9-2.0 microns
- **Orders:** always `["0", "1", "2"]`
- **Order 2 PSF:** reuses order 1 PSF cache (STPSF only provides `GRISM0` and `GRISM1`)
- **FPA->SCA conversion:** uses the order `"1"` optical payload for the undispersed position
- **Normalization band:** F158 (`roman, wfi, f158`)
- **Detector size:** 4088 x 4088 pixels
- **Galaxy morphology:** Sersic profiles generated at `oversample` x resolution

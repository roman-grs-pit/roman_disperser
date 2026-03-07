# Star Grism Pipeline (`scripts/build_star_grism_image.py`)

Simulates Roman Space Telescope grism images from a stellar catalog. Disperses point sources through the optical model with wavelength-dependent PSFs, applies per-SCA sensitivity curves, and produces noiseless and Poisson-sampled FITS images.

## Overview

The script has four modes:

1. **Quick mode** — single pointing, single SCA. Runs `setup_pipeline` + `process_pointing` as a convenience wrapper.
2. **Batch mode** — YAML config specifying multiple pointings and SCAs. Compiles JIT functions once and reuses across all pointings.
3. **`--mosaic`** — generates a focal-plane mosaic PNG from an existing pointing directory (no dispersion).
4. **`--generate-config`** — writes a documented template YAML config and exits.

## Quick Start

```bash
# Quick mode: single SCA
pixi run -e cuda python scripts/build_star_grism_image.py \
    --pointing-ra 9.5 --pointing-dec 0.95 --pointing-pa 0.0 \
    --sca 5 --output my_field.fits --seed 42

# Batch mode: multiple pointings/SCAs from config
pixi run -e cuda python scripts/build_star_grism_image.py \
    --config scripts/example_star_config.yaml

# Generate a template config
pixi run -e cuda python scripts/build_star_grism_image.py \
    --generate-config my_config.yaml

# Generate mosaic from existing pointing directory
pixi run -e cuda python scripts/build_star_grism_image.py \
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
    grism_pointing_name_meta.yaml
```

Quick mode writes a single FITS and PNG at the user-specified path.

### FITS Structure

| Extension | Name    | Description |
|-----------|---------|-------------|
| 0         | PRIMARY | Empty data; metadata headers only |
| 1         | MODEL   | Noiseless count-rate image (counts/s), float32 [4088×4088] |
| 2         | ISIM    | Poisson-sampled image (counts), float32 [4088×4088] |

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

### PNG Quicklook Images

Per-SCA PNGs use asinh stretch (`AsinhNorm` with `linear_width=0.01`), inferno colormap, 4× block-averaged (4088→1022 pixels), flipped to origin-lower. PNGs are rendered from the noiseless MODEL for clean visualization.

### Focal-Plane Mosaic PNG

Generated when more than one SCA is processed. Shows all SCAs arranged in the WFI focal plane layout using FPA center coordinates. Each thumbnail is 8× block-averaged. Global asinh normalization across all SCAs.

### Per-Pointing Metadata YAML

Written to `grism_<name>_meta.yaml`. Fields:

| Field | Description |
|-------|-------------|
| `pointing` | `{ra, dec, pa}` in degrees |
| `exptime` | Exposure time in seconds |
| `seed` | Top-level seed |
| `pointing_key` | JAX RNG key data for this pointing (list of 2 ints) |
| `sca_keys` | Mapping of SCA number → JAX RNG key data |
| `dlam_angstroms` | Wavelength spacing |
| `cone_radius` | Cone search radius in degrees |
| `batch_size` | Sources per JIT batch |
| `source_counts` | Per-SCA dict mapping order string → source count |

## RNG Reproducibility

The seed produces a fully deterministic chain of JAX PRNG keys:

```
seed
  → jax.random.key(seed)
  → jax.random.split(n_pointings)     # one key per pointing
     → jax.random.split(n_scas)       # one key per SCA within pointing
        → jax.random.poisson(...)     # Poisson draw for that SCA
```

Same seed always produces identical output. Per-SCA keys are stored in both FITS headers (`RNDSEED0`/`RNDSEED1`) and the metadata YAML, so individual SCAs can be reproduced by reconstructing the key:

```python
import jax
import jax.numpy as jnp

key = jax.random.wrap_key_data(jnp.array([rndseed0, rndseed1], dtype=jnp.uint32))
```

**Note:** Quick mode uses `jax.random.key(seed)` directly as the pointing key (no split over pointings), so quick-mode and batch-mode outputs for the same seed will differ.

## Configuration Reference

All fields for the batch-mode YAML config:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str | *(required)* | Top-level output directory |
| `seed` | int | *(required)* | RNG seed for reproducibility |
| `exptime` | float | 190.22 | Exposure time in seconds |
| `scas` | `"all"` or list[int] | `"all"` | SCAs to simulate (1-18) |
| `pointings` | list | *(required)* | List of `{name, ra, dec, pa}` dicts |
| `dlam_angstroms` | float | 2.0 | Wavelength spacing in Angstroms |
| `cone_radius` | float | 0.6 | Initial cone search radius [deg] |
| `batch_size` | int | 1000 | Sources per JIT batch |
| `catalog_dir` | str | `data/stars` | Path to star catalog directory |
| `sensitivity_dir` | str | `data/sensitivities` | Path to sensitivity FITS files |
| `optical_model` | str | `data/Roman_grism_OpticalModel_v0.8.yaml` | Optical model YAML |
| `psf_cache_dir` | str | `data/psf_cache` | Path to PSF cache directory |

See `scripts/example_star_config.yaml` for a fully commented example.

## Catalog Format (Hardcoded Assumptions)

The pipeline expects the following structure under `catalog_dir`:

- **`sim_star_cat_galacticus.txt`** — whitespace-delimited, 1 header row. Columns: id (0), template_index (1), magnitude (2), ra (3), dec (4).
- **`SEDtemplates/input_spectral_STARS.lis`** — one filename per line listing 58 available stellar templates.
- **`SEDtemplates/<template>.dat`** — two-column files: wavelength [Angstroms], flux [arbitrary units].

Template index from the catalog is wrapped modulo 58 (the number of entries in `input_spectral_STARS.lis`). All spectra are normalized to F158 AB magnitude using synphot with the `roman, wfi, f158` bandpass. The magnitude column is treated as F158 AB mag; scaling is `10^(-0.4 * mag)` relative to the 0-mag normalized template.

Sensitivity curves are loaded from per-SCA FITS files discovered via `data/sensitivities/sensitivity_map.yaml`, which maps each `(SCA, order)` pair to a FITS file containing `WAVELENGTH` (Angstroms) and `SENSITIVITY` columns.

## Other Hardcoded Assumptions

- **Wavelength range:** 0.9–2.0 microns (`LAM_MIN`, `LAM_MAX`)
- **Orders:** always `["0", "1", "2"]` — not configurable
- **Order 2 PSF:** reuses order 1 PSF cache (STPSF only provides `GRISM0` and `GRISM1`)
- **FPA→SCA conversion:** uses the order `"1"` optical payload for the undispersed position
- **Normalization band:** F158 (`roman, wfi, f158`)
- **Detector size:** 4088 × 4088 pixels

## Pipeline Architecture

### `setup_pipeline(sca_list, ...)`

One-time initialization shared across all pointings:

1. Build wavelength grid from `dlam_angstroms` over 0.9–2.0 μm.
2. Load star catalog and SED template files.
3. Load F158 bandpass via stsynphot; precompute all unique templates normalized to 0 ABmag on the wavelength grid.
4. Load optical model.
5. For each SCA: create optical payloads (per order), load sensitivity curves, load PSF payloads from cache, build `star_disperser` instances, and wrap each in a `make_batched_fori` closure.
6. JIT warmup: call each `fori_fn` once with dummy data to trigger compilation (~5–10s per SCA/order).

Returns a `pipeline` dict consumed by `process_pointing`.

### `process_pointing(pipeline, ra, dec, pa, output_dir, ...)`

Per-pointing processing:

1. **Cone search** — Haversine filter to `cone_radius` degrees around pointing center.
2. **Sky→FPA** — convert (RA, Dec) to FPA coordinates via `omj.get_fpa_pos`.
3. **Per-SCA loop:**
   - `select_sources_per_order` finds sources whose dispersed traces overlap the detector.
   - `generate_spectra` scales precomputed template grids by magnitude.
   - `fpa_to_sca` converts selected sources to SCA coordinates.
   - `disperse_batched` processes sources through the compiled `fori_loop` in chunks of `batch_size`, zero-padding the last chunk.
   - Poisson sampling: `MODEL × exptime` → `jax.random.poisson` with the per-SCA key.
   - Write FITS and PNG.
4. Write mosaic PNG (if multiple SCAs) and metadata YAML.

### JIT Compilation Strategy

Each `(SCA, order)` combination gets a `fori_loop` compiled with a fixed `batch_size` array shape. The loop body captures the star disperser, sensitivity curve, wavelength array, and `dlam` in a closure. Compilation happens once during warmup; all subsequent calls reuse compiled code. See `docs/jit_compilation.md` for the general closure pattern.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Source catalog `F158` column now stores **maggies** (linear AB flux) instead of AB magnitudes, matching the romanisim catalog convention. Conversion: `maggies = 10^(-0.4 * mag)`. For stars, `flux_scale` equals `F158` numerically. The disperser math is unchanged (it reads `flux_scale`, not `F158`); detector outputs are bit-identical to prior runs. Catalog release bumped to `catalog-v2`. Updated `build_source_catalog.py`, `verify_source_catalog.py`, `data/catalogs/README.md`, `docs/grism_pipeline.md`, and `download_source_catalog.py`. The verifier reports F158 errors in mag space (`-2.5·log10(F158)`) for human-readable diagnostics.

## [0.7.0] - 2026-04-09

### Added
- Unified grism simulation pipeline (`scripts/build_grism_image.py`): disperses both stars and galaxies from a single catalog
  - Parquet+Zarr unified source catalog format (see `data/catalogs/README.md`)
  - Per-source Sérsic morphology generation via `sersic.py`
  - Galaxy dispersion with Jacobian-based shape warping + PSF convolution via `galaxy_disperser`
  - Separate `star_batch_size` and `galaxy_batch_size` configuration
  - Per-pointing source manifest (Parquet) with source type, position, flux, and F158 mag
  - Per-pointing metadata YAML with RNG keys and per-SCA/order source counts
- JAX Sérsic profile generator (`sersic.py`) for galaxy morphology pipeline
- Shared pipeline utilities (`pipeline.py`): cone search, source selection, batched dispersion, I/O
- Unified pipeline documentation (`docs/grism_pipeline.md`)
- Catalog format specification (`data/catalogs/README.md`)
- Catalog download and extraction scripts

### Changed
- Memory-efficient per-SCA processing: PSF payloads, dispersers, and JIT functions built per-SCA and released after use (~2-3 GB vs ~18+ GB for all 18 SCAs)
- Galaxy SEDs loaded per-SCA instead of per-pointing to avoid OOM with large catalogs
- On-disk JAX compilation cache (`/tmp/jax-cache-grism`) persists compiled functions across runs (~2.5s vs ~10s per function)
- `batch_size` config key renamed to `star_batch_size` (old key accepted with deprecation warning)

### Deprecated
- `scripts/build_star_grism_image.py`: use `scripts/build_grism_image.py` instead
- `scripts/example_star_config.yaml`: use `scripts/example_grism_config.yaml` instead

## [0.6.0] - 2026-03-11

### Added
- pip installability with dependency tiers: core (`pip install -e .`) and full (`pip install -e ".[full]"`)
- Bundled synphot reference data (`data/synphot/`): F158 bandpass, G0V stellar template, KC96 galaxy templates — eliminates PYSYN_CDBS dependency
- `refdata` module for loading bundled spectral data without stsynphot
- PSF cache download script (`scripts/download_psf_caches.py`): downloads pre-generated caches from GitHub Releases
- INSTALL.md with branching quickstart (pixi and pip paths), GPU setup, data file guide

### Changed
- STPSF moved from pip dependency to pixi-only (only needed for PSF cache regeneration)
- stsynphot replaced by bundled synphot reference data throughout notebooks and pipeline
- synphot import made lazy in `refdata.py` (not needed for minimal install)
- stpsf import made optional in `psf_model.py` save/load functions
- Demo notebooks find project root by walking up to `pyproject.toml` (works outside pixi)
- README.md overhauled: points to INSTALL.md, fixed API examples, updated project structure
- ipykernel and jupyterlab moved from pip to pixi-only dependencies

### Fixed
- `import roman_disperser` now works with minimal pip install (no synphot/stpsf required)
- PSF model tests pass without stpsf installed

## [0.5.0] - 2026-03-07

### Added
- Star grism image pipeline (`scripts/build_star_grism_image.py`): full-field star simulation from catalog
  - Quick mode (single SCA) and batch mode (YAML config, multiple pointings/SCAs)
  - Per-SCA sensitivity curves applied per order
  - Poisson noise sampling with deterministic JAX RNG key tree
  - FITS output: PRIMARY (metadata) + MODEL (noiseless count-rate) + ISIM (Poisson-sampled counts)
  - Per-SCA quicklook PNGs (asinh stretch, 4× block-averaged)
  - Focal-plane mosaic PNG with all SCAs in WFI layout
  - Per-pointing metadata YAML with RNG keys and per-SCA/order source counts
  - `--force` flag to overwrite existing outputs; skips by default
  - `--mosaic` mode to regenerate mosaic from existing pointing directory
  - `--generate-config` to write a documented template YAML
- Catalog module (`catalog.py`): `select_sources` for per-order detector assignment using trace overlap
- Sky-to-FPA transforms (`optical_model_jax.py`): `get_fpa_pos` and `get_pa_rotation` standalone functions
- Example batch config (`scripts/example_star_config.yaml`)
- Per-SCA sensitivity FITS files and `sensitivity_map.yaml`
- Pipeline documentation (`docs/star_grism_pipeline.md`): output format, config reference, catalog assumptions, architecture

### Performance
- 18 SCAs × 3 orders in ~5 minutes on RTX 4090 (after ~30s one-time JIT warmup)
- Per-SCA I/O optimized to ~0.6s (FITS + PNG write)
- Spectrum generation vectorized across all sources

## [0.4.0] - 2026-03-03

### Added
- Galaxy disperser (`galaxy_disperser.py`): extended source dispersion with Jacobian-based shape warping + PSF convolution
- PSF model (`psf_model.py`): STPSF-based PSF grids with trilinear interpolation, caching
- Star disperser (`star_disperser.py`): wavelength-dependent PSF deposition with memory-efficient chunking
- PSF coordinate utilities (`psf_utils.py`)
- PSF cache generation script (`scripts/generate_psf_caches.py`)
- PSF cache migration script (`scripts/migrate_psf_caches.py`)
- Stars + galaxies GPU demo notebook (`notebooks/galaxy/stars_and_galaxies_gpu_demo.ipynb`)
- Disperser performance profiling notebook (`notebooks/galaxy/profile_dispersers.ipynb`)
- Star dispersion design docs (`docs/star_dispersion.md`, `docs/psf_phase1_plan.md`, `docs/phase2_star_dispersion_plan.md`)
- Galaxy dispersion design doc (`docs/galaxy_dispersion_plan.md`)
- Multi-source `fori_loop` JIT pattern: dynamic `n_sources` argument avoids recompilation
- PSF notebooks: analysis, interpolation validation, all-SCA validation
- Star notebooks: single star demo, multi star demo, GPU run
- Galaxy notebooks: Jacobian exploration
- Grism sensitivity and G0V star spectrum notebooks

### Performance
- Star dispersion: ~3 ms/star/order on RTX 4090 (5501 wavelengths, 2A spacing)
- Galaxy dispersion: ~7 ms/galaxy/order on RTX 4090 (120x120 image, 5501 wavelengths)
- 10K sources x 3 orders: ~5 minutes total execution (excluding one-time ~30s JIT compilation)

## [0.3.3] - 2026-01-12

### Added
- GPU scaling benchmark script (`scripts/benchmark_gpu_scaling.py`) testing performance across:
  - Galaxy counts: 100, 250, 500, 1000
  - Spectral orders: +1, 0, +2
  - Wavelength chunk sizes: 50, 100, 200
- GPU scaling analysis notebook (`notebooks/demos/gpu_scaling_analysis.ipynb`) with presentation-quality visualizations
- Order efficiency scaling factors: Order +1 (100%), Order 0 (2%), Order +2 (1%) for realistic flux ratios
- Benchmark results committed to repo (`scripts/output/`) including combined detector PNG

### Performance
- 1000 galaxies across 3 orders: ~19s total on NVIDIA RTX A5000
- Per-galaxy throughput: ~52 galaxies/second (all orders)
- Peak memory: ~418 MB (well under GPU capacity)

## [0.3.2] - 2026-01-12

### Fixed
- Added `precision='highest'` to all einsum calls for GPU/CPU numerical consistency
- Removed hardcoded `JAX_PLATFORMS="cpu"` from test files to allow GPU testing

### Added
- GPU consistency tests (`test_disperser_gpu.py`) comparing CPU vs GPU results
- GPU verification checklist documentation (`docs/guides/2026-01-11-gpu-verification-checklist.md`)
- GPU support section in README

### Performance
- Verified ~50x speedup on NVIDIA RTX A5000 vs CPU for multi-galaxy dispersion
- JIT compilation provides additional 4-10x speedup on GPU (first vs cached calls)

## [0.3.1] - 2026-01-11

### Fixed
- Fixed image position bug in demo notebooks: disperser expects image box corner (pixel [0,0] center), not source center position
- Added `center_to_corner()` helper to `demo_utils.py` for converting source center to image corner position
- Updated `make_random_galaxy_positions()` to clarify it returns center positions

### Changed
- Improved disperser docstrings to clarify that x0, y0 are pixel [0,0] center positions (FITS 1-indexed), not source centers
- Demo notebooks now correctly compute image corner from galaxy center position

### Added
- Tests for `center_to_corner()` helper in `tests/test_demo_utils.py`

## [0.3.0] - 2026-01-11

### Added
- Demo notebooks for disperser module (`single_galaxy_demo.ipynb`, `multi_galaxy_demo.ipynb`)
- JIT compilation strategy documentation (`docs/jit_compilation.md`)
- JIT compilation demonstration in both demo notebooks using closure pattern
- `make_sloped_spectrum()` function in demo_utils for spectra with edge roll-off

### Changed
- Updated demo notebooks to use `model.detmod["pixel_scale"]` for correct pixel scale access
- Improved visualization layout in single_galaxy_demo with rotated zoomed images
- Demo spectra now use sloped spectrum with 20% edge taper for clearer wavelength visualization

## [0.2.0] - 2026-01-10

### Added
- Disperser module with `disperse_2d1d_sca` and `disperse_galaxies_sequential` functions
- Bilinear scatter-add for flux accumulation
- Wavelength chunking for memory efficiency

## [0.1.0] - 2026-01-09

### Added
- Initial JAX optical model implementation (`optical_model_jax.py`)
- Reference NumPy implementation (`optical_model.py`)
- Coordinate transformations (SCA, FPA, MPA)
- Trace beam functionality for grism spectral tracing

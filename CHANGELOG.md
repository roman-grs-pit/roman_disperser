# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

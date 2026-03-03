# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based optical model and disperser for Roman Space Telescope grism spectroscopy. Six main components:
- **Class-based** (`optical_model.py`): Reference implementation using NumPy
- **JAX functional** (`optical_model_jax.py`): JIT-compilable, vectorized implementation
- **Disperser** (`disperser.py`): Legacy 2D+1D galaxy disperser (replaced by `galaxy_disperser.py`)
- **Star disperser** (`star_disperser.py`): Point source dispersion with wavelength-dependent PSFs
- **Galaxy disperser** (`galaxy_disperser.py`): Extended source dispersion with Jacobian-based shape warping + PSF convolution
- **PSF model** (`psf_model.py`): STPSF-based PSF grids with trilinear interpolation

## Design Documents

 - @docs/optical_model.md : JAX optical model API reference and usage examples.
 - @docs/disperser_design.md : Design document for the disperser module, including bilinear scatter-add and 2D→1D dispersion.
 - @docs/jit_compilation.md : JIT compilation strategy for the disperser (closure pattern for non-traceable payload).
 - @docs/stpsf.md : STPSF quick reference for Roman WFI grism mode.
 - @docs/star_dispersion.md : Star dispersion design phases and PSF interpolation approach.
 - @docs/psf_phase1_plan.md : PSF data model implementation plan with validation results.
 - @docs/galaxy_dispersion_plan.md : Design for the new galaxy disperser using Jacobian-based shape warping + PSF convolution.

## Commands

Uses [Pixi](https://pixi.sh) with environments: `default` (CPU), `cuda` (NVIDIA GPU).

```bash
pixi install                    # Install dependencies
pixi run pytest -q tests        # Run all tests
pixi run pytest -v tests/test_optical_model_jax.py::TestTraceBeam  # Test class
pixi run pytest -v tests/test_optical_model_jax.py::TestTraceBeam::test_order_1_vs_class  # Single test
pixi run pytest -m "not slow"   # Skip slow tests (STPSF generation)
pixi run check-jax              # Check JAX backend/device

# PSF cache generation (run once, takes ~2 hours with -j 2)
pixi run python scripts/generate_psf_caches.py --workers 2
```

## Architecture

### Coordinate Systems
- **SCA**: Sensor Chip Assembly [pixels]
- **FPA**: Focal Plane Assembly [degrees]
- **MPA**: Mosaic Plate Assembly [mm]

The SCA coordinate system is defined in 1-indexed FITS coordinates, where pixel n has its center at n.0.

### JAX Module Pattern

Uses payload dict for JIT compatibility:
```python
payload = omj.make_sca_payload(model, sca=1, order="1")
xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wavelength)
```

Key functions: `sca_to_mpa`, `mpa_to_sca`, `sca_to_fpa`, `fpa_to_sca`, `get_mpa_coords`, `get_trace_coeffs`, `trace_beam`

All use `jnp.einsum` for polynomial evaluation.

### Disperser JIT Pattern

The disperser uses a **closure pattern** for JIT compilation because the payload contains non-traceable types (strings) and non-hashable types (dicts with JAX arrays). Capture the payload in a closure before applying `@jax.jit`:

```python
@jax.jit
def disperse_jit(image, x0, y0, dx, dy, spec, lam0, dlam, output):
    return disperser.disperse_2d1d_sca(payload, ...)  # payload captured in closure
```

See @docs/jit_compilation.md for full details.

### Multi-Source fori_loop Pattern

When looping over many sources with `jax.lax.fori_loop`, wrap the loop in `@jax.jit` with the source count as a **dynamic** argument. This compiles once and reuses for any count:

```python
def make_star_fori(order):
    sens = sensitivities[order]
    disperser = star_dispersers[order]

    @jax.jit
    def run(n_sources, output):
        def body_fn(i, output):
            counts = spectra[i] * sens * dlam
            return disperser(x[i], y[i], wavelengths, counts, output)
        return jax.lax.fori_loop(0, n_sources, body_fn, output)

    return run

# Compile once with n=1, reuse for any count
star_fori = make_star_fori('1')
_ = star_fori(1, output)          # ~5s compilation
output = star_fori(1000, output)   # no recompilation
```

Without this pattern, calling `fori_loop` outside JIT recompiles every invocation (~5s), which dominates execution for small source counts.

### PSF Model

The PSF model uses STPSF to generate wavelength- and position-dependent PSF grids, then provides fast trilinear interpolation:

```python
from roman_disperser import psf_model

# Load or generate PSF payload (cached to data/psf_cache/)
psf_payload = psf_model.get_or_make_psf_payload(
    detector='WFI05', order='1', cache_dir='data/psf_cache'
)

# Interpolate PSF at any position and wavelength (wavelength in microns)
psf = psf_model.interpolate_psf(psf_payload, xsca=2000.0, ysca=2000.0, wavelength=1.5)
```

Key functions: `make_psf_payload`, `interpolate_psf`, `interpolate_psf_spatial`, `get_or_make_psf_payload`, `save_psf_payload`, `load_psf_payload`

Default grid: 4×4 spatial × 56 wavelengths (0.9-2.0 μm), validated to <0.03% flux error across all 18 SCAs.

**Note:** All wavelength parameters in `psf_model` are in **microns** (not meters).

### Star Disperser

The star disperser module provides functions for dispersing point sources with wavelength-dependent PSFs:

```python
from roman_disperser import star_disperser

# Create a JIT-compiled star disperser
disperser = star_disperser.make_star_disperser(psf_payload, optical_payload)

# Disperse a star (wavelengths in microns)
output = disperser(xsca_star=2000.0, ysca_star=2000.0, wavelengths=wl_array, star_flux=flux_array, output=output)
```

Key functions: `make_psf_pixel_grid`, `deposit_psf`, `disperse_star_psf`, `make_star_disperser`

**Note:** All wavelength parameters in `star_disperser` are in **microns** (consistent with optical model).

### Galaxy Disperser

The galaxy disperser extends the star disperser to handle extended sources. It warps the galaxy morphology through the dispersion Jacobian, convolves with the PSF, and deposits onto the detector:

```python
from roman_disperser import galaxy_disperser

# Create a JIT-compiled galaxy disperser
disperse = galaxy_disperser.make_galaxy_disperser(psf_payload, optical_payload)

# Disperse a galaxy (image at oversample× resolution, wavelengths in microns)
output = disperse(image=galaxy_image, x0=2000.0, y0=2000.0,
                  spectrum=flux_array, wavelengths=wl_array, output=output)
```

Key functions: `trace_beam_sca`, `trace_beam_sca_with_jacobian`, `disperse_galaxy_shape`, `prepare_galaxy_images`, `disperse_galaxy`, `make_galaxy_disperser`

**Algorithm:**
1. Compute Jacobian of the SCA→SCA dispersion map at the galaxy center (per wavelength)
2. Warp galaxy morphology through the Jacobian (forward scatter, flux-conserving)
3. Convolve warped images with wavelength-dependent PSFs (FFT)
4. Interpolate convolved images to fine wavelength grid; deposit onto detector with exact positions

**Note:** Galaxy images must be at `psf_payload['oversample']`× resolution (typically 4×). Pixel spacing is derived automatically from the PSF payload.

## Coding Guidelines

- Significant changes should ALWAYS be done on a new branch. Create a descriptive branch name for the feature or bug fix.
- When merging a branch back in to `main`, finish by tagging the release with a version number. Use semantic versioning and ask if you have a question. 
    Also, make sure to update the `CHANGELOG.md` with a summary of the changes.
- This is a research code, so value simplicity and clarity over deep class hierarchies and generality. Prefer functional routines over complex object-oriented designs.

## Notes

- JAX implementation uses modern code path only (no `old_format` legacy support)
- Spectral orders are strings: "1", "0", "2", "m1"
- Model config: `data/Roman_grism_OpticalModel_v0.8.yaml`
- `demo_utils.py` provides helpers for generating synthetic galaxy profiles and spectra
- PSF caches stored in `data/psf_cache/`; generate with `scripts/generate_psf_caches.py`
- All `jnp.einsum` calls use `precision='highest'` for CPU/GPU numerical consistency
- GPU tests in `test_disperser_gpu.py` verify CPU vs GPU results match



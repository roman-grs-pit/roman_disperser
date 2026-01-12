# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based optical model and disperser for Roman Space Telescope grism spectroscopy. Three main components:
- **Class-based** (`optical_model.py`): Reference implementation using NumPy
- **JAX functional** (`optical_model_jax.py`): JIT-compilable, vectorized implementation
- **Disperser** (`disperser.py`): 2D spatial + 1D spectral → detector simulation

## Design Documents

 - @docs/optical_model.md : JAX optical model API reference and usage examples.
 - @docs/disperser_design.md : Design document for the disperser module, including bilinear scatter-add and 2D→1D dispersion.
 - @docs/jit_compilation.md : JIT compilation strategy for the disperser (closure pattern for non-traceable payload).

## Commands

Uses [Pixi](https://pixi.sh) with environments: `default` (CPU), `cuda` (NVIDIA GPU).

```bash
pixi install                    # Install dependencies
pixi run pytest -q tests        # Run all tests
pixi run pytest -v tests/test_optical_model_jax.py::TestTraceBeam  # Test class
pixi run pytest -v tests/test_optical_model_jax.py::TestTraceBeam::test_order_1_vs_class  # Single test
pixi run check-jax              # Check JAX backend/device
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
- All `jnp.einsum` calls use `precision='highest'` for CPU/GPU numerical consistency
- GPU tests in `test_disperser_gpu.py` verify CPU vs GPU results match



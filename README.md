# Roman Disperser

JAX-based optical model and disperser for Roman Space Telescope grism spectroscopy simulations.

## Overview

This project provides tools for simulating Roman grism spectroscopy:

1. **Optical Model** (`optical_model.py`, `optical_model_jax.py`): Coordinate transformations and spectral trace calculations
2. **Disperser** (`disperser.py`): Simulates grism dispersion of 2D spatial images with 1D spectra onto a detector

### Coordinate Systems

- **SCA**: Sensor Chip Assembly coordinates [pixels]
- **FPA**: Focal Plane Assembly coordinates [degrees from SCA center]
- **MPA**: Mosaic Plate Assembly coordinates [mm]

## Installation

Using [Pixi](https://pixi.sh):

```bash
pixi install
```

This installs all dependencies including JAX, NumPy, pytest, and matplotlib.

## Running Tests

Run the full test suite:

```bash
pixi run pytest -q tests
```

For more verbose output:

```bash
pixi run pytest -v tests
```

Run specific test class:

```bash
pixi run pytest -v tests/test_optical_model_jax.py::TestMPAtoSCA
```

Run a specific test:

```bash
pixi run pytest -v tests/test_optical_model_jax.py::TestTraceBeam::test_order_1_vs_class
```

### Test Coverage

The test suite includes:

- **Optical model tests** (`test_optical_model_jax.py`): SCA/FPA/MPA coordinate transformations, polynomial mappings, trace coefficients, and spectral traces
- **Disperser tests** (`test_disperser.py`): Bilinear interpolation, flux conservation, boundary handling, multi-galaxy batching

**Total: 124 tests** validating JAX implementations against reference implementations.

## Documentation

- [Optical Model API](docs/optical_model.md) - JAX optical model functions and examples
- [JIT Compilation Strategy](docs/jit_compilation.md) - How to JIT-compile the disperser
- [Disperser Design](docs/disperser_design.md) - Implementation details for the disperser module

## Disperser Module (`disperser.py`)

The disperser module simulates grism spectroscopy by dispersing 2D spatial images with 1D spectra onto a detector.

### Key Functions

- **`disperse_2d1d_sca(payload, image, x0, y0, dx, dy, spec, lam0, dlam, output, ...)`**: Disperse a single galaxy
- **`disperse_galaxies_sequential(payload, images, x0s, y0s, dx, dy, specs, lam0s, dlams, ...)`**: Disperse multiple galaxies sequentially

### JIT Compilation

The disperser uses a closure pattern for JIT compilation because the payload contains non-traceable types (strings) and non-hashable types (dicts with JAX arrays):

```python
payload = omj.make_sca_payload(model, sca=5, order="1")

@jax.jit
def disperse_jit(image, x0, y0, dx, dy, spec, lam0, dlam, output):
    return disperser.disperse_2d1d_sca(
        payload, image, x0, y0, dx, dy, spec, lam0, dlam, output,
        wavelength_chunk_size=100
    )

output = disperse_jit(image, x0, y0, dx, dy, spec, lam0, dlam, output)
```

See [docs/jit_compilation.md](docs/jit_compilation.md) for detailed documentation on the JIT strategy.

## Notebooks

**Demo notebooks** (`notebooks/demos/`):
- **`single_galaxy_demo.ipynb`**: Single galaxy dispersion with JIT compilation
- **`multi_galaxy_demo.ipynb`**: Multi-galaxy batch dispersion

**Archive** (`notebooks/archive/`):
- **`quicklook_jax.ipynb`**: Legacy visualization of spectral traces

## Project Structure

```
roman_disperser/
├── src/roman_disperser/
│   ├── __init__.py
│   ├── optical_model.py         # Class-based model (reference)
│   ├── optical_model_jax.py     # JAX functional model
│   ├── optical_model_utils.py   # Coordinate system utilities
│   ├── disperser.py             # 2D+1D dispersion module
│   └── demo_utils.py            # Utilities for demo notebooks
├── tests/
│   ├── __init__.py
│   ├── test_optical_model_jax.py  # Optical model tests
│   └── test_disperser.py          # Disperser tests
├── notebooks/
│   ├── demos/                     # Demonstration notebooks
│   │   ├── single_galaxy_demo.ipynb
│   │   └── multi_galaxy_demo.ipynb
│   └── archive/                   # Legacy/development notebooks
│       └── quicklook_jax.ipynb
├── docs/
│   ├── disperser_design.md        # Disperser implementation plan
│   ├── jit_compilation.md         # JIT compilation strategy
│   └── optical_model.md           # Optical model API and examples
├── data/
│   └── Roman_grism_OpticalModel_v0.8.yaml
├── pixi.toml                      # Environment & task configuration
└── pyproject.toml                 # Package metadata
```

## References

- Roman Space Telescope: https://roman.gsfc.nasa.gov/
- JAX documentation: https://jax.readthedocs.io/
- Pixi: https://pixi.sh

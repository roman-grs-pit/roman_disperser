# Roman Disperser: JAX-based Optical Model

A functional JAX implementation of the Roman Space Telescope grism optical model for efficient spectral tracing and coordinate transformations.

## Overview

This project provides two complementary implementations of the Roman optical model:

1. **Class-based model** (`optical_model.py`): The original `RomanOpticalModel` class with full functionality
2. **JAX functional model** (`optical_model_jax.py`): A pure functional, JIT-compilable implementation optimized for batch operations

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

The test suite (`tests/test_optical_model_jax.py`) includes:

- **SCA↔MPA conversions** (center, corners, random points)
- **SCA↔FPA conversions** (center, corners, random points)  
- **FPA→MPA polynomial mapping** (`get_mpa_coords`, ~13 tests)
- **Trace coefficient computation** (`get_trace_coeffs`, ~14 tests)
- **Full spectral trace simulation** (`trace_beam`, ~12 parametrized + 2 additional tests)

**Total: 113 tests** validating JAX implementation against the class-based reference.

## JAX Optical Model (`optical_model_jax.py`)

### Key Functions

#### Coordinate Transformations

- **`sca_to_mpa(payload, xsca, ysca)`**: SCA pixels → MPA (mm)
- **`mpa_to_sca(payload, xmpa, ympa)`**: MPA → SCA pixels
- **`fpa_to_mpa(payload, xfpa, yfpa)`**: FPA degrees → MPA (mm)
- **`mpa_to_fpa(payload, xmpa, ympa)`**: MPA → FPA degrees

#### Optical Model Functions

- **`get_mpa_coords(payload, xfpa, yfpa)`**: FPA → MPA polynomial mapping (wavelength-independent, reference wavelength)
  - Uses Einstein summation (`einsum`) for ~12x speedup over matmul operations
  
- **`get_trace_coeffs(payload, xfpa, yfpa)`**: Compute curvature and dispersion coefficients
  - Returns IDs (wavelength-dependent position shift) and CRV (wavelength-independent curvature)
  
- **`trace_beam(payload, xfpa, yfpa, wavelength)`**: Full wavelength-dependent spectral traces
  - Maps (xfpa, yfpa, wavelength) → (xmpa_trace, ympa_trace)
  - Combines reference positions with wavelength-dependent coefficients
  - Handles arbitrary wavelength arrays; supports batching with JAX transformations

#### Utilities

- **`make_sca_payload(model, sca, order)`**: Extract per-SCA/per-order data into JAX-compatible dict

### Design Notes

- **No legacy support**: The JAX implementation uses the modern code path only (not `old_format`)
- **JIT-compatible**: All functions use JAX operations for `jax.jit()` compilation
- **Batch operations**: Vectorized to handle N points × M wavelengths efficiently
- **Einstein summation**: Polynomial evaluation uses `jnp.einsum` for optimal performance

## Example Usage

### Basic Coordinate Transformation

```python
import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel

# Load model
model = RomanOpticalModel("data/Roman_grism_OpticalModel_v0.8.yaml")

# Create payload for SCA 1, order +1
payload = omj.make_sca_payload(model, sca=1, order="1")

# Transform FPA coords to MPA
xfpa, yfpa = [0.5, 1.0], [0.0, 0.5]
xmpa, ympa = omj.fpa_to_mpa(payload, xfpa, yfpa)
```

### Spectral Trace Simulation

```python
import numpy as np

# Generate traces for 10 FPA points across the wavelength grid
xfpa = np.linspace(-1.0, 1.0, 10)
yfpa = np.zeros(10)
wavelengths = model.wl_grid  # Default wavelength array

# Trace beam: (10 points) × (20 wavelengths) → 10×20 output
xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wavelengths)

# Result shape: (10, 20)
```

### Batch Processing with vmap

```python
import jax

# Vectorize over multiple SCAs
def trace_sca(sca):
    payload = omj.make_sca_payload(model, sca=sca, order="1")
    return omj.trace_beam(payload, xfpa, yfpa, wavelengths)

scas = [1, 2, 5, 10]
results = jax.vmap(trace_sca)(np.array(scas))
```

## Notebooks

- **`quicklook_jax.ipynb`**: Visualization of spectral traces for orders 0, ±1
  - Uses `compute_traces_jax` helper to evaluate traces on a grid of FPA points
  - Plots wavelength-colored traces with source positions and reference markers
  - Demonstrates single-figure approach for clean matplotlib rendering

## Project Structure

```
roman_disperser/
├── src/roman_disperser/
│   ├── __init__.py
│   ├── optical_model.py              # Class-based model (reference)
│   ├── optical_model_jax.py          # JAX functional model
│   └── optical_model_utils.py        # Coordinate system utilities
├── tests/
│   ├── __init__.py
│   └── test_optical_model_jax.py     # 113 comprehensive tests
├── notebooks/
│   └── quicklook_jax.ipynb           # Visualization example
├── data/
│   └── Roman_grism_OpticalModel_v0.8.yaml
├── pixi.toml                         # Environment & task configuration
└── pyproject.toml                    # Package metadata
```

## References

- Roman Space Telescope: https://roman.gsfc.nasa.gov/
- JAX documentation: https://jax.readthedocs.io/
- Pixi: https://pixi.sh

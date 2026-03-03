# JAX Optical Model (`optical_model_jax.py`)

This document describes the JAX functional implementation of the Roman Space Telescope grism optical model.

## Key Functions

### Coordinate Transformations

- **`sca_to_mpa(payload, xsca, ysca)`**: SCA pixels → MPA (mm)
- **`mpa_to_sca(payload, xmpa, ympa)`**: MPA → SCA pixels
- **`fpa_to_mpa(payload, xfpa, yfpa)`**: FPA degrees → MPA (mm)
- **`mpa_to_fpa(payload, xmpa, ympa)`**: MPA → FPA degrees

### Sky-to-FPA Coordinate Transforms

These are standalone functions (no payload needed) for converting between sky coordinates and FPA:

- **`get_pa_rotation(pa)`**: Return 2×2 rotation matrix for a given position angle (degrees). Converts on-sky PA to focal plane coordinate system.
- **`get_fpa_pos(ra, dec, pointing_ra, pointing_dec, pointing_pa)`**: Convert sky coordinates (RA, Dec) to FPA position (degrees) given telescope pointing and PA. Inputs are 1D arrays for (ra, dec) and scalars for pointing parameters.

### Optical Model Functions

- **`get_mpa_coords(payload, xfpa, yfpa)`**: FPA → MPA polynomial mapping (wavelength-independent, reference wavelength)
  - Uses Einstein summation (`einsum`) for ~12x speedup over matmul operations

- **`get_trace_coeffs(payload, xfpa, yfpa)`**: Compute curvature and dispersion coefficients
  - Returns IDS (wavelength-dependent position shift) and CRV (wavelength-independent curvature)

- **`trace_beam(payload, xfpa, yfpa, wavelength)`**: Full wavelength-dependent spectral traces
  - Maps (xfpa, yfpa, wavelength) → (xmpa_trace, ympa_trace)
  - Combines reference positions with wavelength-dependent coefficients
  - Handles arbitrary wavelength arrays; supports batching with JAX transformations

### Utilities

- **`make_sca_payload(model, sca, order)`**: Extract per-SCA/per-order data into JAX-compatible dict

## Design Notes

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

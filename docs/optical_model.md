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

- **`get_pa_rotation(pa)`**: Return 2×2 rotation matrix for a given position angle (degrees). Converts on-sky PA to focal plane coordinate system. The focal-plane orientation constant is `FOCAL_PA_DEG = -60.0`; see the convention note in the docstring and the derivation in `docs/reference/tangent_plane_derivation.ipynb`.
- **`sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec)`**: Gnomonic (TAN) projection of sky coordinates onto the tangent plane at the pointing, on the host in float64 NumPy — a verbatim transcription of the derivation notebook, pinned bit-exact by test. Returns (ξ East, η North) in degrees. Periodic in RA by construction (no meridian wrap handling needed). Raises `TypeError` on float32 input (the precision is already lost by then). See the precision convention in the module docstring.
- **`get_fpa_pos_from_offsets(dx, dy, pointing_pa)`**: Rotate tangent-plane offsets into FPA coordinates. The JAX, jit-compilable half of the transform.
- **`get_fpa_pos(ra, dec, pointing_ra, pointing_dec, pointing_pa)`**: Convert sky coordinates (RA, Dec) to FPA position (degrees) given telescope pointing and PA. Composes the two functions above; **not jit-compilable** (the float64 differencing is host NumPy — jit `get_fpa_pos_from_offsets` instead). Pass float64 host arrays for (ra, dec), never `jnp.array(...)`.

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
- **JIT-compatible**: All functions use JAX operations for `jax.jit()` compilation, with one deliberate exception: `get_fpa_pos` (and `sky_to_tangent_offsets`) run on the host in float64 — jit `get_fpa_pos_from_offsets` instead
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

## Differentiable sky → detector transform (small perturbations)

`sky_to_tangent_offsets` is host float64 NumPy by design (absolute RA/Dec
need float64; see the precision convention in `optical_model_jax`), so
`jax.grad` cannot see RA/Dec. For gradients w.r.t. source positions and the
pointing, evaluate the geometry once at the nominal pointing and perturb in
JAX:

```python
ref = omj.make_sky_reference(ra, dec, ra0, dec0, pa0, payload=payload)  # host f64
x, y = omj.perturbed_sky_to_sca(ref, payload, d_xi, d_eta, d_ra, d_dec, d_pa)
J = jax.jacfwd(lambda p: jnp.stack(omj.perturbed_sky_to_sca(ref, payload, *p)))(p0)
```

All perturbations are in degrees (`d_xi`, `d_eta`: source offsets in the
nominal tangent plane, per source or scalar; `d_ra`, `d_dec`, `d_pa`:
pointing). The result is the float64 reference plus a float32 change built
only from small quantities (no small-angle approximation): within ~1.3e-3 px
of a float64 evaluation for 10″ source and 0.1° pointing perturbations on
SCA 5 (the existing `get_fpa_pos` + `fpa_to_sca` chain is at 1-5e-3 px), and
Jacobians agree with float64 finite differences to ~3e-7 relative.
`perturbed_sky_to_fpa` / `perturbed_tangent_offsets` give the intermediate
coordinates, `wavelengths=` adds the dispersed trace.

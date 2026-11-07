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
import jax
import jax.numpy as jnp
import roman_disperser.optical_model_jax as omj
from roman_disperser import paths
from roman_disperser.optical_model import RomanOpticalModel

# Load the hydrated model (or pass the YAML path yourself)
model = RomanOpticalModel(str(paths.optical_model_path()))

# Create payload for SCA 1, order +1
payload = omj.make_sca_payload(model, sca=1, order="1")

# Transform FPA coords to MPA (arrays, not Python lists)
xfpa, yfpa = jnp.array([0.5, 1.0]), jnp.array([0.0, 0.5])
xmpa, ympa = omj.fpa_to_mpa(payload, xfpa, yfpa)
```

### Spectral Trace Simulation

`trace_beam` pairs positions and wavelengths element by element (shape
`[n]`); one position with `M` wavelengths broadcasts to `[M]`. For `N`
positions x `M` wavelengths, `vmap` over the positions:

```python
# Traces of 10 FPA points at 20 wavelengths (microns)
xfpa = jnp.linspace(-0.2, 0.2, 10)
yfpa = jnp.zeros(10)
wavelengths = jnp.linspace(1.0, 1.9, 20)

trace = jax.vmap(lambda x, y: omj.trace_beam(payload, x[None], y[None],
                                             wavelengths))
xmpa, ympa = trace(xfpa, yfpa)
assert xmpa.shape == (10, 20)
```

The broadcast form agrees with the call on explicitly repeated positions
to float32 rounding (~4e-6 px), not bit for bit.

### Several SCAs

A payload mixes arrays with Python ints and strings, so it is built on the
host and cannot be created inside `jax.vmap`. Loop over the SCAs; with the
payload as a jit argument (see `docs/jit_compilation.md`) one compile serves
them all:

```python
from roman_disperser import jit_payload

def trace_arrays(arrays, xfpa, yfpa, wavelengths, static):
    p = jit_payload.merge_payload(arrays, static)
    return omj.trace_beam(p, xfpa, yfpa, wavelengths)

trace_jit = jax.jit(trace_arrays, static_argnames="static")
results = {}
for sca in (1, 2, 5, 10):
    p = omj.make_sca_payload(model, sca=sca, order="1")
    arrays, static = jit_payload.split_payload(p, jit_payload.OPTICAL_STATIC_KEYS)
    results[sca] = trace_jit(arrays, xfpa, yfpa, jnp.full(10, 1.5),
                             static=static)
```

## Differentiable sky → detector transform (small perturbations)

`sky_to_tangent_offsets` is host float64 NumPy by design (absolute RA/Dec
need float64; see the precision convention in `optical_model_jax`), so
`jax.grad` cannot see RA/Dec. For gradients w.r.t. source positions and the
pointing, evaluate the geometry once at the nominal pointing and perturb in
JAX:

```python
import numpy as np

ra0, dec0, pa0 = 10.195391, 0.102452, 60.0            # pointing, degrees
ra = np.array([10.10, 10.12])                         # sources (float64)
dec = np.array([0.20, 0.21])
payload5 = omj.make_sca_payload(model, sca=5, order="1")
ref = omj.make_sky_reference(ra, dec, ra0, dec0, pa0, payload=payload5)  # host f64

d_xi = d_eta = jnp.zeros(2)                           # source offsets, degrees
d_ra = d_dec = d_pa = 0.0                             # pointing offsets, degrees
x, y = omj.perturbed_sky_to_sca(ref, payload5, d_xi, d_eta, d_ra, d_dec, d_pa)

# Jacobian of the pixel positions w.r.t. the pointing (d_ra, d_dec, d_pa)
p0 = jnp.zeros(3)
J = jax.jacfwd(lambda p: jnp.stack(
    omj.perturbed_sky_to_sca(ref, payload5, 0.0, 0.0, *p)))(p0)   # [2, 2, 3]
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

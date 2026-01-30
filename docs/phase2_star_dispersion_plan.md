# Phase 2: Single Star Dispersion Implementation Plan

## Goal
Implement `disperse_star_psf()` to disperse a single star through the Roman grism, depositing wavelength-dependent PSFs along the spectral trace.

## Design Decisions (User-Confirmed)
- **Flux input**: Array per wavelength (allows wavelength-dependent flux/spectrum)
- **Efficiency curves**: Defer to Phase 4 (keep Phase 2 focused on geometry and PSF deposition)
- **PSF lookup position**: Use undispersed position (per STPSF help desk guidance)
- **Pixel deposition**: Direct deposit (each PSF pixel lands in one detector pixel)
- **Wavelength handling**: Interpolate to user-specified wavelengths

## Algorithm
1. Take star position (xsca, ysca) in SCA coordinates
2. Get spatially-interpolated PSFs at undispersed position: `interpolate_psf_spatial(payload, x₀, y₀)` → `[N_wl_grid, PSF_y, PSF_x]`
3. Interpolate PSFs to user wavelengths → `[N_wl_user, PSF_y, PSF_x]`
4. Compute all dispersed positions at once (vectorized):
   - `xsca_disp, ysca_disp = trace_all_wavelengths(...)` → `[N_wl_user]` each
5. Scale PSFs by flux: `scaled_psfs = psfs * star_flux[:, None, None]`
6. Deposit all PSFs onto detector using pre-computed relative coordinate grid
7. Return accumulated detector image

## Key Design Decisions

### Undispersed Position for PSF Lookup
Use the star's undispersed position for STPSF PSF lookup because:
- This is what STPSF requires (confirmed via STPSF help desk)
- Much more efficient (1 bilinear interp vs N_wl trilinear interps)
- PSF field dependence is minimal based on Phase 1 validation

### Direct Pixel Deposition (Not Bilinear)
Each oversampled PSF pixel deposits into exactly one detector pixel:
- For 4× oversampling, PSF center is at the cross-hairs of 4 central pixels
- Use floor/round to map PSF pixel centers to detector pixels
- Use `jax.numpy.ndarray.at[].add()` with `mode="drop"` for out-of-bounds
- Set `wrap_negative_indices=False` to prevent negative index wrapping

### Enforce Even Oversampling
For even oversampling (2×, 4×, etc.), the PSF center lies at the intersection of 4 central pixels rather than at a single pixel center. This is the geometry STPSF produces.
- Validate that `psf_payload['oversample']` is even
- Raise `ValueError` if odd oversampling is detected
- This ensures the coordinate grid math is correct for sub-pixel positioning

### User-Specified Wavelength Grid
Two-step interpolation:
1. Spatial interpolation at undispersed position → PSFs at grid wavelengths
2. Wavelength interpolation → PSFs at user-specified wavelengths

This allows users to specify arbitrary wavelength sampling for their spectra.

## Files to Modify

| File | Changes |
|------|---------|
| `src/roman_disperser/psf_model.py` | Add `interpolate_psf_wavelength()` |
| `src/roman_disperser/disperser.py` | Add `make_psf_pixel_grid()`, `deposit_psf()`, `disperse_star_psf()`, `make_star_disperser()` |
| `tests/test_disperser.py` | Add star dispersion tests |
| `notebooks/psf/single_star_demo.ipynb` | Demo notebook |

## Implementation

### Step 1: Wavelength Interpolation (`psf_model.py`)

```python
def interpolate_psf_wavelength(
    psfs: jnp.ndarray,           # [N_wl_grid, PSF_y, PSF_x] from interpolate_psf_spatial
    wl_grid: jnp.ndarray,        # [N_wl_grid] wavelengths in payload
    wavelengths: jnp.ndarray,    # [N_wl_user] user wavelengths
) -> jnp.ndarray:                # [N_wl_user, PSF_y, PSF_x]
    """Interpolate PSFs to user-specified wavelengths."""
```

Linear interpolation along wavelength axis with edge extrapolation (clamp to grid bounds).

### Step 2: `make_psf_pixel_grid()` helper

```python
def make_psf_pixel_grid(
    psf_shape: tuple,         # (PSF_y, PSF_x) oversampled PSF size
    oversample: int,          # PSF oversampling factor (e.g., 4)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create relative coordinate grid for PSF deposition.

    For even oversampling, PSF center is at cross-hairs of 4 central pixels.
    Returns (rel_y, rel_x) grids in detector pixel units.

    Each PSF pixel center at oversampled index (i, j) maps to detector offset:
      rel_x[i,j] = (j - (PSF_x - 1) / 2) / oversample
      rel_y[i,j] = (i - (PSF_y - 1) / 2) / oversample
    """
```

Pre-compute once, then add star position offset at deposition time.

### Step 3: `deposit_psf()` helper

```python
def deposit_psf(
    output: jnp.ndarray,      # [H, W] detector accumulator
    xsca: float,              # Center position (SCA, 1-indexed)
    ysca: float,
    psf: jnp.ndarray,         # [psf_h, psf_w] oversampled PSF (already flux-scaled)
    rel_x: jnp.ndarray,       # [psf_h, psf_w] relative x offsets (from make_psf_pixel_grid)
    rel_y: jnp.ndarray,       # [psf_h, psf_w] relative y offsets
) -> jnp.ndarray:
    """
    Deposit PSF onto detector at specified position.

    Uses direct pixel assignment (not bilinear):
    - Compute absolute detector coordinates: det_x = xsca + rel_x, det_y = ysca + rel_y
    - Convert to 0-indexed array indices: idx_x = floor(det_x - 0.5), idx_y = floor(det_y - 0.5)
    - Use at[].add() with mode="drop" and wrap_negative_indices=False
    """
```

### Step 4: `disperse_star_psf()` main function

```python
def disperse_star_psf(
    psf_payload: dict,        # From psf_model.get_or_make_psf_payload()
    optical_payload: dict,    # From optical_model_jax.make_sca_payload()
    xsca_star: float,         # Star position (SCA, 1-indexed)
    ysca_star: float,
    wavelengths: jnp.ndarray, # [N_wl] user wavelengths in meters
    star_flux: jnp.ndarray,   # [N_wl] flux per wavelength
    output: jnp.ndarray,      # [4088, 4088] accumulator
) -> jnp.ndarray:
```

Algorithm:
1. Get PSF stack at undispersed position:
   ```python
   psfs_grid = interpolate_psf_spatial(psf_payload, xsca_star, ysca_star)  # [N_wl_grid, PSF_y, PSF_x]
   ```
2. Interpolate to user wavelengths:
   ```python
   psfs = interpolate_psf_wavelength(psfs_grid, psf_payload['wavelengths'], wavelengths)  # [N_wl, PSF_y, PSF_x]
   ```
3. Convert to FPA and trace all wavelengths (vectorized):
   ```python
   xfpa, yfpa = sca_to_fpa(optical_payload, xsca_star, ysca_star)
   xmpa, ympa = trace_beam(optical_payload, xfpa, yfpa, wavelengths)  # [N_wl] each
   xsca_disp, ysca_disp = mpa_to_sca(optical_payload, xmpa, ympa)
   ```
4. Scale PSFs by flux:
   ```python
   scaled_psfs = psfs * star_flux[:, None, None]  # [N_wl, PSF_y, PSF_x]
   ```
5. Pre-compute relative coordinate grid once:
   ```python
   rel_y, rel_x = make_psf_pixel_grid(psfs.shape[1:], psf_payload['oversample'])
   ```
6. Deposit all PSFs (can be parallelized with vmap or done in loop with fori_loop):
   ```python
   for i in range(len(wavelengths)):
       output = deposit_psf(output, xsca_disp[i], ysca_disp[i],
                           scaled_psfs[i], rel_x, rel_y)
   ```
7. Return output

### Step 5: Factory function with JIT compilation

```python
def make_star_disperser(
    psf_payload: dict,
    optical_payload: dict,
) -> Callable:
    """
    Create a JIT-compiled star disperser for a specific detector/order.

    Validates payload and returns a compiled function that can be called
    repeatedly for different stars without recompilation.

    Parameters
    ----------
    psf_payload : dict
        From psf_model.get_or_make_psf_payload()
    optical_payload : dict
        From optical_model_jax.make_sca_payload()

    Returns
    -------
    disperse_star : Callable
        JIT-compiled function: (xsca, ysca, wavelengths, flux, output) -> output

    Raises
    ------
    ValueError
        If psf_payload uses odd oversampling
    """
    # Validate even oversampling (required for correct PSF center geometry)
    if psf_payload['oversample'] % 2 != 0:
        raise ValueError(
            f"PSF payload must use even oversampling, got {psf_payload['oversample']}"
        )

    # Pre-compute relative coordinate grid (captured in closure)
    psf_shape = psf_payload['psf_grid'].shape[-2:]
    rel_y, rel_x = make_psf_pixel_grid(psf_shape, psf_payload['oversample'])

    @jax.jit
    def disperse_star(xsca, ysca, wavelengths, flux, output):
        return disperse_star_psf(
            psf_payload, optical_payload,
            xsca, ysca, wavelengths, flux, output,
            rel_x, rel_y  # Pre-computed grid
        )

    return disperse_star
```

Usage:
```python
# Setup once per detector/order
disperser = make_star_disperser(psf_payload, optical_payload)

# Disperse many stars efficiently
for star in stars:
    output = disperser(star.x, star.y, wavelengths, star.flux, output)
```

## Tests

1. **Position accuracy**: Single wavelength should deposit at trace_beam predicted position
2. **Flux conservation**: Total output flux should equal input flux (within detector bounds)
3. **PSF shape**: Output at each wavelength should match expected PSF shape
4. **JIT compilation**: Verify compiles and produces consistent results
5. **Multiple stars**: Sequential dispersion should accumulate correctly
6. **Edge handling**: Stars near detector edge should not crash

## Verification

1. Run `pixi run pytest -v tests/test_disperser.py::TestStarDispersion`
2. Run demo notebook to visualize:
   - Single star trace across wavelengths
   - PSF shape variation along trace
   - Flux conservation check
3. Compare output positions with `trace_beam()` predictions

## Notes

- PSF payload wavelengths: 56 wavelengths at 0.02 μm spacing (0.9-2.0 μm)
- User `wavelengths` and `star_flux` arrays must have matching lengths
- Efficiency curves will be applied in Phase 4, not here
- All PSFs stored in GPU memory enables vectorized operations

---

## Resolved Comments

### PSF Lookup Position
> NP: The undispersed position is what it looks like STPSF requires, based on my interactions with the help desk. So that's why we should do it. The efficiency gain is a nice byproduct.

**Resolution**: Confirmed. Using undispersed position for PSF lookup as STPSF requires this. Added to Design Decisions section.

### Pixel Deposition Method
> NP: I think bilinear_scatter_add is the wrong method - the flux should just get deposited into the pixel it falls into. We could be more careful and see how the PSF pixel shape overlaps with the detector pixels, but I think that is overkill.

**Resolution**: Agreed. Changed to direct pixel assignment. Each PSF pixel center maps to one detector pixel via floor. This is simpler and sufficient given the 4× oversampling already captures sub-pixel structure.

### JAX at[].add() Usage
> NP: I think here, we should just use the at[].add() functionality of JAX. It has the ability to drop out of bound pixels, and we should also enforce that negative indices don't wrap around.

**Resolution**: Incorporated. Using `output.at[idx_y, idx_x].add(psf, mode="drop", wrap_negative_indices=False)` for safe out-of-bounds handling.

### Pre-computed Coordinate Grid
> NP: One thing about this helper - if we are using the at functionality, we will need a coordinate grid. It might be worth pre-computing a relative coordinate grid and then just add the offset on the fly.

**Resolution**: Added `make_psf_pixel_grid()` helper that pre-computes relative coordinates once. At deposition time, just add the star position offset.

### Wavelength Interpolation
> NP: I think we want to interpolate the wavelengths. So we have two interpolations - the first is to do the spatial grid and then reinterpolate onto the wavelengths.

**Resolution**: Added two-step interpolation:
1. `interpolate_psf_spatial()` - bilinear spatial interpolation, returns all grid wavelengths
2. `interpolate_psf_wavelength()` - new function, interpolates to user wavelengths

### Vectorized Deposition
> NP: If we have all the PSFs stored in memory, then I think the deposition step could also be parallelized by computing the positions of all the PSFs at once, and then just depositing them. Note that an intermediate step is the scaling of the PSFs by the fluxes in the input wavelength array.

**Resolution**: Algorithm updated to:
1. Compute all dispersed positions at once (vectorized trace_beam)
2. Scale all PSFs by flux in one broadcast operation
3. Deposit can use vmap or fori_loop for parallelization

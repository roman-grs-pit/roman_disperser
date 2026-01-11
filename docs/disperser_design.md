# Disperser Module Implementation Plan

> **To resume this plan:** This document is saved at
> `/Users/npadmana/.claude/plans/serene-squishing-stardust.md`
> and can be referenced to continue implementation.

## Overview

Create a new `disperser.py` module that disperses a 2D spatial image × 1D spectrum onto a detector using the Roman optical model's `trace_beam` function.

**Confirmed requirements:**
- ~1000s of galaxies per image
- Each galaxy: Ny,Nx ~ 50-150 pixels (with 3× oversampling), Nlam ~ 1000 wavelengths
- Output: Full detector (4088×4088) - fixed size
- Flux accumulation only (no weight map)
- Uniform wavelength spacing (lam0 + ilam * dlam)
- Fractional dx/dy supported for oversampled input grids
- Must be JIT-compilable and GPU-efficient
- **Memory-efficient:** Exploit separable structure to avoid materializing full 3D grids

---

## Key Insight: Separable Structure

The `trace_beam` function has separable components:

```python
# SPATIAL-ONLY (compute once per spatial pixel):
xfpa, yfpa = sca_to_fpa(payload, xsca, ysca)           # [N_spatial]
xmpa_ref, ympa_ref = get_mpa_coords(payload, xfpa, yfpa)  # [N_spatial]
crv, ids = get_trace_coeffs(payload, xfpa, yfpa)          # [i, N_spatial]

# WAVELENGTH-DEPENDENT (broadcasts over spatial × wavelength):
dely = polynomial(wavelength, ids)   # [N_spatial, N_wavelength]
delx = polynomial(dely, crv)         # [N_spatial, N_wavelength]
```

**Memory savings:** Instead of `O(Ny × Nx × Nlam)`, we use `O(Ny × Nx) + O(chunk_size)`.

For Ny=Nx=135, Nlam=1000:
- Full 3D approach: ~2-3 GB peak
- Chunked approach (λ chunks of 100): ~100-200 MB peak

This enables **parallel galaxy processing** on GPU due to lower per-galaxy memory.

---

## Phase 1: Core Disperser with Chunking

**File:** `src/roman_disperser/disperser.py`

### Function Signature

```python
def disperse_2d1d_sca(
    payload: dict,           # From make_sca_payload()
    image: jnp.ndarray,      # [Ny, Nx] float32 spatial image
    x0: float, y0: float,    # SCA origin of image grid (pixels)
    dx: float, dy: float,    # Pixel spacing (can be fractional for oversampling)
    spec: jnp.ndarray,       # [Nlam] float32 spectrum
    lam0: float,             # Starting wavelength (microns)
    dlam: float,             # Wavelength spacing (microns)
    output: jnp.ndarray,     # [4088, 4088] accumulator
    wavelength_chunk_size: int = 100,  # Wavelengths per chunk
) -> jnp.ndarray:            # [4088, 4088] updated output
```

### Algorithm Flow (Memory-Efficient)

```
Step 1: Precompute spatial quantities (once)
────────────────────────────────────────────
   Ny, Nx = image.shape
   N_spatial = Ny * Nx

   # Build 2D spatial grid and flatten
   iy, ix = meshgrid(arange(Ny), arange(Nx), indexing='ij')
   xsca = (x0 + ix * dx).ravel()      # [N_spatial]
   ysca = (y0 + iy * dy).ravel()      # [N_spatial]
   image_flat = image.ravel()          # [N_spatial]

   # Convert to FPA (once)
   xfpa, yfpa = sca_to_fpa(payload, xsca, ysca)  # [N_spatial]

   # Get reference positions and coefficients (once)
   xmpa_ref, ympa_ref = get_mpa_coords(payload, xfpa, yfpa)  # [N_spatial]
   crv, ids = get_trace_coeffs(payload, xfpa, yfpa)  # [i, N_spatial]

Step 2: Process wavelength chunks
─────────────────────────────────
   for λ_start in range(0, Nlam, wavelength_chunk_size):
       λ_end = min(λ_start + wavelength_chunk_size, Nlam)
       chunk_size = λ_end - λ_start

       # Wavelengths for this chunk
       wl_chunk = lam0 + (λ_start + arange(chunk_size)) * dlam  # [chunk]

       # Compute dispersed positions via broadcasting
       # ids has shape [i, N_spatial], wl_chunk has shape [chunk]
       # Result: dely[N_spatial, chunk], delx[N_spatial, chunk]
       dely = eval_polynomial(wl_chunk, ids)   # broadcast
       delx = eval_polynomial(dely, crv)       # broadcast

       # Final MPA positions
       xmpa_out = xmpa_ref[:, None] + delx  # [N_spatial, chunk]
       ympa_out = ympa_ref[:, None] + dely  # [N_spatial, chunk]

       # Convert back to SCA
       xsca_out, ysca_out = mpa_to_sca(payload, xmpa_out, ympa_out)

       # Compute flux: image[spatial] × spec[wavelength]
       flux = image_flat[:, None] * spec[λ_start:λ_end]  # [N_spatial, chunk]

       # Flatten and scatter-add with bilinear weights
       output = bilinear_scatter_add(
           output, xsca_out.ravel(), ysca_out.ravel(), flux.ravel()
       )

   return output
```

### Helper: Bilinear Scatter-Add

```python
def bilinear_scatter_add(output, x, y, values):
    """Accumulate values onto output grid using bilinear interpolation."""
    x_floor = jnp.floor(x).astype(jnp.int32)
    y_floor = jnp.floor(y).astype(jnp.int32)
    fx = x - x_floor
    fy = y - y_floor

    # Four corner weights
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    # Bounds check (need x_floor+1 < 4088, so x_floor < 4087)
    valid = (x_floor >= 0) & (x_floor < 4087) & \
            (y_floor >= 0) & (y_floor < 4087)

    # Scatter-add to four corners
    output = output.at[y_floor, x_floor].add(values * w00 * valid)
    output = output.at[y_floor, x_floor + 1].add(values * w10 * valid)
    output = output.at[y_floor + 1, x_floor].add(values * w01 * valid)
    output = output.at[y_floor + 1, x_floor + 1].add(values * w11 * valid)

    return output
```

### Implementation Notes

**Polynomial evaluation with broadcasting:**
```python
def eval_ids_polynomial(wl, ids):
    """Evaluate inverse dispersion polynomial.

    Args:
        wl: wavelengths [N_wl] (already transformed: wl - wl_ref or log)
        ids: coefficients [i, N_spatial]

    Returns:
        dely: [N_spatial, N_wl]
    """
    # wl_powers[N_wl, i] = wl^i
    wl_powers = wl[:, None] ** jnp.arange(ids.shape[0])
    # einsum: [N_wl, i] × [i, N_spatial] → [N_wl, N_spatial] → transpose
    return jnp.einsum('wi,is->sw', wl_powers, ids)
```

**JIT compilation:** Use `jax.lax.fori_loop` for the wavelength chunk loop to keep everything JIT-compatible.

**Spatial chunking (optional):** For very large images (Ny×Nx > 50k pixels), add an outer spatial chunk loop.

---

## Phase 2: Multi-Galaxy Batching

**Goal:** Efficiently disperse 1000s of galaxies onto a single output image.

With the memory-efficient chunked approach (~100-200 MB per galaxy), we have options:

### Option A: Sequential with `fori_loop` (simplest)

```python
def disperse_galaxies_sequential(
    payload: dict,
    images: jnp.ndarray,     # [N_galaxies, Ny_max, Nx_max]
    x0s: jnp.ndarray,        # [N_galaxies]
    y0s: jnp.ndarray,        # [N_galaxies]
    specs: jnp.ndarray,      # [N_galaxies, Nlam_max]
    lam0s: jnp.ndarray,      # [N_galaxies]
    dlams: jnp.ndarray,      # [N_galaxies]
    # dx, dy assumed constant (or add as arrays)
) -> jnp.ndarray:
    output = jnp.zeros((4088, 4088), dtype=jnp.float32)

    def body_fn(i, output):
        return disperse_2d1d_sca(
            payload, images[i], x0s[i], y0s[i], 1.0, 1.0,
            specs[i], lam0s[i], dlams[i], output
        )

    return jax.lax.fori_loop(0, len(images), body_fn, output)
```

### Option B: Parallel batches with `vmap` + sequential accumulation

Process galaxies in parallel batches, then sum the outputs:

```python
def disperse_galaxies_parallel(
    payload, images, x0s, y0s, specs, lam0s, dlams,
    batch_size: int = 16,  # Galaxies to process in parallel
) -> jnp.ndarray:
    """Process galaxies in parallel batches."""

    # vmap over galaxy dimension (each gets its own output buffer)
    @jax.vmap
    def disperse_batch(image, x0, y0, spec, lam0, dlam):
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)
        return disperse_2d1d_sca(
            payload, image, x0, y0, 1.0, 1.0,
            spec, lam0, dlam, output
        )

    # Process in batches and sum
    n_galaxies = len(images)
    total_output = jnp.zeros((4088, 4088), dtype=jnp.float32)

    for start in range(0, n_galaxies, batch_size):
        end = min(start + batch_size, n_galaxies)
        batch_outputs = disperse_batch(
            images[start:end], x0s[start:end], y0s[start:end],
            specs[start:end], lam0s[start:end], dlams[start:end]
        )  # [batch, 4088, 4088]
        total_output += batch_outputs.sum(axis=0)

    return total_output
```

**Memory for Option B:** batch_size × 4088² × 4 bytes = batch_size × 67 MB
- batch_size=16: ~1 GB for output buffers + ~1.6-3.2 GB working memory
- Suitable for GPUs with 8+ GB memory

**Recommendation:** Start with Option A (sequential), benchmark, then try Option B if GPU utilization is low.

---

## Phase 2b: Extension to Non-Separable Data (Future)

For cases where we have a full 3D data cube `data[Ny, Nx, Nlam]` instead of separable `image × spec`:

```python
def disperse_3d_sca(
    payload: dict,
    data: jnp.ndarray,       # [Ny, Nx, Nlam] - full 3D cube
    x0: float, y0: float,
    dx: float, dy: float,
    lam0: float, dlam: float,
    output: jnp.ndarray,
    spatial_chunk_size: int = 1024,
    wavelength_chunk_size: int = 100,
) -> jnp.ndarray:
    """Disperse non-separable 3D data cube.

    The coordinate computation is still separable (same positions
    for all wavelengths at a given spatial pixel), but the flux
    values come from the full 3D cube.
    """
    # Same algorithm as disperse_2d1d_sca, but:
    # - Chunk over BOTH spatial and wavelength dimensions
    # - flux = data[spatial_chunk, wavelength_chunk] instead of outer product
```

This shares the same coordinate machinery but handles arbitrary 3D input.

---

## Phase 3: Testing

**File:** `tests/test_disperser.py`

### Test 1: Delta function dispersion
```python
def test_delta_function_dispersion():
    """Single pixel image should land at trace_beam predicted position."""
    # Create 1x1 image with value 1.0 at known SCA position (x0, y0)
    # Create flat spectrum with single wavelength
    # Verify output peak location matches trace_beam(xfpa, yfpa, wavelength)
    # Verify total flux = image * spec
```

### Test 2: Bilinear weight conservation
```python
def test_bilinear_weights_sum_to_one():
    """Ensure total flux is conserved (within detector bounds)."""
    # Place small image near detector center (all flux lands on detector)
    # Disperse with image.sum()=1.0, spec.sum()=1.0
    # Verify output.sum() ≈ 1.0 (rtol=1e-5)
```

### Test 3: Boundary handling
```python
def test_out_of_bounds_ignored():
    """Flux dispersed off-detector should be ignored, not crash."""
    # Place image near detector edge where trace goes off-detector
    # Verify no errors, output.sum() < input total (some flux lost)
```

### Test 4: Separable vs direct comparison
```python
def test_separable_matches_direct():
    """Separable image×spec should match manual outer product."""
    # Compute dispersion with separable inputs
    # Compute same with explicit 3D flux array
    # Verify outputs match
```

### Test 5: Wavelength chunking invariance
```python
def test_chunking_invariant():
    """Result should be same regardless of wavelength_chunk_size."""
    # Run with chunk_size=10, 50, 100, 1000
    # All should produce identical output (within numerical precision)
```

### Test 6: JIT compilation
```python
def test_jit_compiles():
    """Verify function compiles and produces correct output."""
    jitted = jax.jit(disperse_2d1d_sca, static_argnums=(8,))  # chunk_size static
    result1 = jitted(...)
    result2 = jitted(...)  # Should use cached compilation
    # Verify results match non-jitted version
```

---

## Phase 4: Example Notebook (Optional)

**File:** `notebooks/disperse_example.ipynb`

Simple demonstration:
1. Create Gaussian 2D profile (20×20 pixels, 3× oversampled)
2. Create emission line spectrum (Gaussian + flat continuum)
3. Disperse single galaxy onto detector
4. Disperse 100 galaxies at random positions
5. Visualize result with wavelength coloring (like quicklook_jax.ipynb)

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/roman_disperser/disperser.py` | **Create** - main disperser module |
| `src/roman_disperser/__init__.py` | **Modify** - add exports |
| `tests/test_disperser.py` | **Create** - test suite |

---

## Verification Steps

1. **Unit tests:** `pixi run pytest -v tests/test_disperser.py`
2. **JIT test:** Verify compilation succeeds and is cached on second call
3. **Flux conservation:** Centered image should conserve flux perfectly
4. **Position accuracy:** Delta function peak matches `trace_beam` prediction
5. **Chunk invariance:** Same result with different `wavelength_chunk_size`
6. **GPU test:** `pixi run check-jax` then run full test suite with GPU

---

## Implementation Order

### Step 0: Save this plan
Copy this plan to `docs/disperser_design.md` for future reference.

### Step 1: Core helpers
1. `bilinear_scatter_add(output, x, y, values)` - bilinear interpolation scatter
2. `eval_ids_polynomial(wl, ids)` - wavelength polynomial with broadcasting
3. `eval_crv_polynomial(dely, crv)` - curvature polynomial with broadcasting

### Step 2: Main disperser function
1. `disperse_2d1d_sca(...)` - single galaxy, wavelength-chunked

### Step 3: Basic tests
1. Delta function test
2. Flux conservation test
3. JIT compilation test

### Step 4: Multi-galaxy wrapper
1. `disperse_galaxies_sequential(...)` - fori_loop version

### Step 5: Extended tests
1. Boundary handling
2. Chunk invariance
3. Multi-galaxy accumulation

### Step 6 (Optional): Parallel batching
1. `disperse_galaxies_parallel(...)` - vmap version
2. Benchmarking notebook

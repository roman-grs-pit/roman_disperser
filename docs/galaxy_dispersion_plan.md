# Galaxy Dispersion Implementation Plan

## Overview

Extend the star dispersion routines to handle extended sources (galaxies). The key difference from stars is that the galaxy's spatial extent must be dispersed before PSF convolution.

**Input:**
- Galaxy image: 2D grid `[Ny, Nx]` at `psf_payload['oversample']`× oversampling
- Galaxy spectrum: 1D array `[Nlam]`
- Undispersed position `(x0, y0)` in SCA coordinates
- Pixel spacing derived from PSF payload: `dx = dy = 1.0 / oversample`

**Output:**
- Accumulated flux on detector `[4088, 4088]`

---

## Phase 0: Jacobian Exploration (Research)

**Goal:** Characterize the dispersion Jacobian to understand shear/stretch magnitude.

Create a notebook `notebooks/galaxy/jacobian_exploration.ipynb` that:

1. Define a helper function to compute the SCA→SCA Jacobian:
   ```python
   def trace_beam_sca(payload, xsca, ysca, wavelength):
       """Trace from SCA through optical model back to SCA."""
       xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)
       xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wavelength)
       return omj.mpa_to_sca(payload, xmpa, ympa)

   # Jacobian w.r.t. (xsca, ysca) at fixed wavelength
   jacobian_fn = jax.jacobian(trace_beam_sca, argnums=(1, 2))
   ```

2. Compute J at a grid of positions across the detector (e.g., 5×5 grid)

3. Compute J across wavelength range (0.9 - 2.0 μm)

4. Analyze:
   - How close is J to identity? Compute `||J - I||` (Frobenius norm)
   - Is there systematic shear? Look at off-diagonal terms
   - How does J vary with position and wavelength?
   - For a 50-pixel galaxy (~12 native pixels), what's the maximum position error if we used J=I?

5. Document findings - this informs whether optimizations (like J≈I) are viable in the future

---

## Phase 1: Core Galaxy Disperser

**File:** `src/roman_disperser/galaxy_disperser.py`

### 1.1 Helper: SCA-to-SCA trace with Jacobian

```python
def trace_beam_sca(payload, xsca, ysca, wavelength):
    """Trace position through optical model: SCA → FPA → MPA → SCA."""
    xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)
    xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wavelength)
    return omj.mpa_to_sca(payload, xmpa, ympa)

def trace_beam_sca_with_jacobian(payload, xsca, ysca, wavelength):
    """Compute dispersed position and Jacobian at that position.

    Returns:
        x_out, y_out: dispersed SCA position
        J: 2×2 Jacobian matrix [[∂x'/∂x, ∂x'/∂y], [∂y'/∂x, ∂y'/∂y]]
    """
    # Use JAX autodiff for Jacobian
    ...
```

### 1.2 Helper: Disperse galaxy shape at single wavelength

```python
def disperse_galaxy_shape(payload, image, x0, y0, dx, dy, wavelength):
    """Disperse galaxy morphology at a single wavelength.

    Algorithm:
    1. Compute dispersed center position and Jacobian at (x0, y0, wavelength)
    2. Build relative coordinate grid for input image pixels
    3. Transform relative coords through Jacobian: (Δx', Δy') = J @ (Δx, Δy)
    4. Use bilinear interpolation to redistribute flux onto output grid

    Args:
        payload: optical model payload
        image: [Ny, Nx] galaxy image (4× oversampled)
        x0, y0: undispersed center position (SCA coords)
        dx, dy: pixel spacing (0.25 for 4× oversampling)
        wavelength: wavelength in microns

    Returns:
        dispersed_image: [Ny, Nx] flux redistributed to dispersed positions
        x_center, y_center: dispersed center position
    """
```

**Key implementation detail:** The bilinear redistribution is a "reverse" operation from the scatter-add in disperser.py. Here we're mapping FROM source positions TO a regular grid, which is a standard image warping/resampling operation.

### 1.3 Main function: Create dispersed+convolved galaxy images

```python
def prepare_galaxy_images(
    optical_payload,
    psf_payload,
    image,           # [Ny, Nx] galaxy image
    x0, y0,          # undispersed center (SCA)
    dx, dy,          # pixel spacing
):
    """Prepare dispersed and PSF-convolved galaxy images at PSF wavelengths.

    Algorithm:
    1. Get PSF at undispersed center: interpolate_psf_spatial() → [N_wl, PSF_y, PSF_x]
    2. For each PSF wavelength:
       a. Disperse galaxy shape using Jacobian
       b. Convolve with PSF using FFT
    3. Return convolved images and center positions

    Returns:
        convolved_images: [N_wl, Conv_y, Conv_x] - convolved galaxy at each wavelength
        center_positions: [N_wl, 2] - (x, y) dispersed center at each wavelength
        wavelengths: [N_wl] - PSF grid wavelengths
    """
```

**Convolution details:**
- Use `jax.scipy.signal.fftconvolve` with `mode='full'` to capture all PSF flux
  - `mode='same'` would truncate ~91 PSF pixels at galaxy edges
  - `mode='full'`: output size = (Ny + PSF_y - 1, Nx + PSF_x - 1)
- **Kernel centering:** Both the PSF and dispersed galaxy image are centered in their arrays.
  With `mode='full'`, the output center is at ((Ny + PSF_y - 1) // 2, (Nx + PSF_x - 1) // 2).
  The `mode='full'` handles the kernel origin correctly - no need to fftshift the PSF to (0,0).
- **Convolution vs correlation:** We want physical PSF convolution:
  `output[x,y] = Σ image[i,j] * PSF[x-i, y-j]`
  This is exactly what `fftconvolve` computes. Use `convolve`, not `correlate`.
- Zero-padding handled automatically by FFT

### 1.4 Main disperser function (with wavelength chunking)

```python
def disperse_galaxy(
    optical_payload,
    psf_payload,
    image,              # [Ny, Nx] galaxy image (at psf_payload['oversample']× oversampling)
    x0, y0,             # undispersed center (SCA)
    spectrum,           # [Nlam] flux values
    wavelengths,        # [Nlam] wavelength array (microns)
    output,             # [4088, 4088] accumulator
    chunk_size=1000,    # wavelengths per chunk
):
    """Disperse a galaxy onto the detector.

    Algorithm:
    1. Call prepare_galaxy_images() to get convolved images at PSF wavelengths
    2. Process spectrum wavelengths in chunks (using jax.lax.scan):
       a. Interpolate convolved images to chunk wavelengths
       b. Interpolate center positions to chunk wavelengths
       c. Scale by spectrum flux
       d. Deposit onto output using direct pixel addition (4× oversampled)
    3. Return accumulated output
    """
```

**Deposition:** Since images are 4× oversampled, use direct pixel assignment like star_disperser (not bilinear scatter-add).

### 1.5 Factory function for JIT compilation

```python
def make_galaxy_disperser(psf_payload, optical_payload, chunk_size=1000):
    """Create JIT-compiled galaxy disperser with payloads captured in closure.

    Validation (before JIT, in factory):
    - Assert psf_payload['oversample'] matches expected galaxy oversampling
    - Assert even oversampling (like star_disperser)

    The disperse_fn expects image at psf_payload['oversample']× oversampling.
    Pixel spacing dx, dy should be 1.0 / oversample.

    Returns:
        disperse_fn(image, x0, y0, spectrum, wavelengths, output) -> output

    Note: dx, dy derived from psf_payload['oversample'] in factory (not passed to disperse_fn)
    """
    # Validate oversampling (outside JIT)
    oversample = psf_payload['oversample']
    if oversample % 2 != 0:
        raise ValueError(f"PSF payload must use even oversampling, got {oversample}")

    # Derive pixel spacing from oversampling
    dx = dy = 1.0 / oversample

    # Pre-compute grids, capture in closure...

    @jax.jit
    def disperse_fn(image, x0, y0, spectrum, wavelengths, output):
        # dx, dy captured from closure
        ...
```

---

## Phase 2: Testing

**File:** `tests/test_galaxy_disperser.py`

### Test cases:

1. **Jacobian computation**: Verify autodiff Jacobian matches finite differences

2. **Delta function galaxy**: Single-pixel galaxy should match star disperser output

3. **Flux conservation**: Centered galaxy (all flux on detector) should conserve total flux

4. **Shape dispersion**: Verify Jacobian transformation correctly warps galaxy shape

5. **Convolution correctness**: Known PSF × known galaxy = expected result

6. **Wavelength interpolation**: Convolved images interpolate correctly between grid points

7. **Chunk invariance**: Same result with different chunk_size values

8. **JIT compilation**: Verify function compiles and caches correctly

---

## Phase 3: Demo Notebook

**File:** `notebooks/galaxy/single_galaxy_demo.ipynb`

1. Create synthetic galaxy (Sérsic profile or Gaussian)
2. Create emission line + continuum spectrum
3. Disperse single galaxy onto detector
4. Visualize:
   - Input galaxy morphology
   - Dispersed galaxy at a few wavelengths (before/after PSF convolution)
   - Final detector image
   - Comparison with equivalent point source (star)
5. Performance benchmarks

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `notebooks/galaxy/jacobian_exploration.ipynb` | Create | Phase 0: Characterize Jacobian |
| `src/roman_disperser/galaxy_disperser.py` | Create | Main galaxy disperser module |
| `src/roman_disperser/__init__.py` | Modify | Export galaxy_disperser |
| `tests/test_galaxy_disperser.py` | Create | Test suite |
| `notebooks/galaxy/single_galaxy_demo.ipynb` | Create | Demo and validation |
| `docs/galaxy_dispersion.md` | Create | Design documentation |

---

## Memory Estimates

For a typical galaxy (150×150 pixels, 4× oversampled):
- Input galaxy: 150 × 150 × 4 bytes = 90 KB
- PSF grid (spatial interpolated): 56 × 182 × 182 × 4 bytes ≈ 7.4 MB
- Convolved images (mode='full'): 56 × 331 × 331 × 4 bytes ≈ 24.5 MB
- Per wavelength chunk (1000 wavelengths): ~44 MB for interpolated 331×331 images
- Output detector: 4088 × 4088 × 4 bytes ≈ 67 MB

**Total peak memory:** ~150 MB per galaxy - still manageable

---

## Implementation Order

1. **Phase 0**: Jacobian exploration notebook (research)
2. **Phase 1.1-1.2**: Jacobian and shape dispersion helpers
3. **Phase 1.3**: Convolution with PSF
4. **Phase 1.4-1.5**: Main disperser with chunking and factory function
5. **Phase 2**: Core tests (delta function, flux conservation, chunk invariance)
6. **Phase 3**: Demo notebook
7. **Documentation**: Update docs/galaxy_dispersion.md

---

## Verification

1. **Jacobian accuracy**: Compare autodiff vs finite differences (<1e-5 relative error)
2. **Delta function test**: Single-pixel galaxy matches star_disperser output
3. **Flux conservation**: Centered galaxy conserves flux to <0.1%
4. **Chunk invariance**: Results identical regardless of chunk_size
5. **Visual inspection**: Demo notebook shows physically reasonable dispersion

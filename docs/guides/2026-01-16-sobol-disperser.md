The code aims to disperse the flux in a box in x-y-lambda into a region x'-y' following a coded optical model. The box is divided into a set of cells defined by a set of grid points. Currently the code calculates the dispersed position of the center of each cell, and then distributes the flux in the cell by a bilinear interpolation onto the dispersed grid. This isn't correct, since the flux could fall completely into one cell - the input cell might be a lot smaller.

The first part of the fix is to replace the bilinear interpolation with a simpler grid deposition.

To be more careful with how the flux gets distributed, one needs relatively smaller grid cells, and computing the full dispersion solution would get expensive for these points. So instead we propose the following algorithm :

1. Divide the x-y-lambda into cells; these could be "relatively large cells".
2. Calculate the dispersion solution for the central point, but also calculate the Jacobian of that transformation. This is easy since the code is all in JAX.
3. Allocate points using a Sobol sequence within the cell.
4. Calculate the flux at each point using the input image and spectrum.
5. Disperse the points using the Jacobian. Put these points in a smaller subgrid, also save the position of the subgrid.
6. Repeat 3-5 for all cells.
7. Normalize all the dispersed points by the full flux in the input box.
8. Deposit each of the subgrids into the full 4088x4088 grid.


## Implementation Plan

### Step 0: Jacobian Accuracy Validation ✓ COMPLETE

**Goal:** Test that the Jacobian calculation is accurate enough for the proposed algorithm.

**Test parameters:**
- Cell size: 10 × 10 SCA pixels × 100Å (0.01 μm)
- Spatial range: -500 to 5500 SCA pixels (extends beyond detector)
- Wavelength range: 0.9 to 2.0 μm
- All 18 SCAs, orders "0", "1", "2" (54 configurations)
- 1000 random cells sampled per configuration

**Method:** For each cell, compute dispersion at the 8 corners using both:
1. Full solution: `omj.trace_sca_to_sca(payload, x, y, λ)`
2. Jacobian approximation: `center_output + J @ [dx, dy, dλ]`

Measure the maximum Euclidean error in output (x', y') pixels.

**Results:**
| Metric | Value |
|--------|-------|
| Worst max error | 0.0079 pixels |
| Best max error | 0.0020 pixels |
| Mean of max errors | 0.0047 pixels |
| Worst 99th percentile | 0.0054 pixels |

**Conclusion:** ✓ EXCELLENT - All errors < 0.01 pixel. The Jacobian approximation is validated for 10×10×100Å cells.

**Artifacts:**
- `notebooks/demos/jacobian_accuracy_test.ipynb` - exploration notebook with visualizations
- `notebooks/demos/jacobian_accuracy_results.json` - full validation results (54 configs)
- `tests/test_jacobian_accuracy.py` - regression test (54 parametrized tests)


## Implementation Notes

### Key Functions

**`omj.trace_sca_to_sca(payload, xsca, ysca, wavelength)`**
- Computes full dispersion: (xsca, ysca, λ) → (xsca', ysca')
- Chains: `sca_to_fpa` → `trace_beam` → `mpa_to_sca`
- Returns tuple `(xsca_out, ysca_out)`

**Jacobian computation:**
```python
def compute_jacobian_at_point(payload, xsca, ysca, wavelength):
    def trace_single(inputs):
        xout, yout = omj.trace_sca_to_sca(payload, inputs[0:1], inputs[1:2], inputs[2:3])
        return jnp.stack([xout, yout]).squeeze()
    return jax.jacobian(trace_single)(jnp.array([xsca, ysca, wavelength]))
```

**Typical Jacobian values (at detector center, λ=1.5μm):**
- ∂x'/∂x ≈ 0.98, ∂x'/∂y ≈ 0, ∂x'/∂λ ≈ -6 pix/μm (small cross-dispersion)
- ∂y'/∂x ≈ 0, ∂y'/∂y ≈ 1, ∂y'/∂λ ≈ 913 pix/μm (main dispersion direction)

### JIT Compilation Pattern

The payload contains non-traceable strings (`wl_transform`), so use the closure pattern:
```python
payload = omj.make_sca_payload(model, sca=5, order="1")

@jax.jit
def my_jitted_function(x, y, lam):
    return omj.trace_sca_to_sca(payload, x, y, lam)  # payload captured in closure
```

See `docs/jit_compilation.md` for full details.


## Steps 1-8: Sobol Disperser Implementation

### Overview

Replace the bilinear flux spreading approach with a Sobol quasi-random sampling method that:
1. Samples each input cell with ~8192 Sobol points
2. Uses Jacobian approximation to disperse points (validated in Step 0)
3. Deposits to small subgrids for cache efficiency, then accumulates to detector

### Algorithm Summary

```
For each cell in (x, y, λ) space:
    1. Compute center dispersion and Jacobian
    2. Generate N Sobol points in cell
    3. Sample flux at each point: flux = image(x,y) × spectrum(λ) × (cell_volume/N)
    4. Disperse points: (x', y') = center_out + J @ [Δx, Δy, Δλ]
    5. Deposit to small subgrid (32×32)
    6. Add subgrid to detector at computed offset
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sobol points per cell | 8192 (configurable) | Matches previous 150×150×1000 density |
| Sobol caching | Module-level cache | Avoid repeated scipy calls; cache keyed by (n_points, seed) |
| Subgrid size | 32×32 (configurable) | Fits ~10×20 dispersed cell with padding |
| Subgrid validation | Test suite only | One-time test (like Jacobian accuracy) validates 32×32 is sufficient |
| Cell processing | Hybrid vmap+scan | vmap over batches (~100-500 cells), scan over batches |
| Input interpolation | Bilinear | Standard for subpixel sampling |
| Normalization | Per-cell | Simplest, no global sync needed |
| Integration | Keep both methods | Allows comparison; fix existing to use deposit instead of bilinear scatter |

### Implementation Steps

#### Step 1: Add Sobol Sequence Generation (Module-Level Cache)

Cache Sobol points at module level to avoid repeated scipy calls and CPU→GPU transfers.

```python
# Module-level cache for Sobol sequences
_SOBOL_CACHE: dict[tuple[int, int], jnp.ndarray] = {}

def get_sobol_points_unit(n_points: int, seed: int = 0) -> jnp.ndarray:
    """
    Get cached Sobol points in [0, 1]^3.

    Points are generated once and cached for reuse across all cells.
    The same unit-cube points are scaled to each cell's bounds at runtime.

    Args:
        n_points: Number of points (should be power of 2 for optimal Sobol coverage)
        seed: Sobol sequence seed for reproducibility

    Returns:
        points: Array of shape [n_points, 3] with values in [0, 1]
    """
    key = (n_points, seed)
    if key not in _SOBOL_CACHE:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=3, scramble=True, seed=seed)
        _SOBOL_CACHE[key] = jnp.array(sampler.random(n_points), dtype=jnp.float32)
    return _SOBOL_CACHE[key]


def clear_sobol_cache():
    """Clear the Sobol point cache (useful for testing or memory management)."""
    _SOBOL_CACHE.clear()
```

**Usage:** Call `get_sobol_points_unit(8192)` once at start, then scale to each cell:
```python
sobol_unit = get_sobol_points_unit(n_sobol, seed)  # [N, 3] in [0,1]^3
x_pts = x_min + sobol_unit[:, 0] * (x_max - x_min)  # Scale to cell
```

#### Step 2: Add Input Flux Sampling

Sample image and spectrum at arbitrary (x, y, λ) positions using bilinear interpolation.

```python
def sample_flux_bilinear(
    image: jnp.ndarray,      # [Ny, Nx] input image
    x0: float, y0: float,    # Origin of image grid (SCA coords)
    dx: float, dy: float,    # Pixel spacing
    spectrum: jnp.ndarray,   # [Nlam] spectrum
    lam0: float, dlam: float,# Wavelength grid
    x: jnp.ndarray,          # [N] sample x positions (SCA coords)
    y: jnp.ndarray,          # [N] sample y positions (SCA coords)
    lam: jnp.ndarray,        # [N] sample wavelengths
) -> jnp.ndarray:            # [N] flux values
    """Sample image × spectrum at arbitrary positions using bilinear interpolation."""
    # Convert SCA coords to image array indices (fractional)
    ix = (x - x0) / dx
    iy = (y - y0) / dy
    ilam = (lam - lam0) / dlam

    # Bilinear interpolation for image
    image_flux = bilinear_sample_2d(image, ix, iy)

    # Linear interpolation for spectrum
    spec_flux = linear_sample_1d(spectrum, ilam)

    return image_flux * spec_flux
```

#### Step 3: Compute Jacobian for Cell Centers

Vectorized Jacobian computation for multiple cell centers.

```python
def compute_jacobians_batch(payload, x_centers, y_centers, lam_centers):
    """
    Compute Jacobians at multiple cell centers.

    Returns:
        centers_out: (x_out, y_out) arrays of shape [N]
        jacobians: Array of shape [N, 2, 3]
    """
    # Wrapper for single-point Jacobian (from Step 0 validation)
    def jacobian_at_point(xc, yc, lamc):
        def trace_fn(inputs):
            xout, yout = omj.trace_sca_to_sca(
                payload, inputs[0:1], inputs[1:2], inputs[2:3]
            )
            return jnp.stack([xout, yout]).squeeze()
        return jax.jacobian(trace_fn)(jnp.array([xc, yc, lamc]))

    # vmap over cell centers
    jacobians = jax.vmap(jacobian_at_point)(x_centers, y_centers, lam_centers)

    # Also get center output positions
    x_out, y_out = omj.trace_sca_to_sca(payload, x_centers, y_centers, lam_centers)

    return (x_out, y_out), jacobians
```

#### Step 4: Disperse Sobol Points Using Jacobian

Apply Jacobian approximation to disperse points.

```python
def disperse_points_jacobian(
    center_out: tuple,       # (x_c', y_c') center output position
    jacobian: jnp.ndarray,   # [2, 3] Jacobian matrix
    center_in: tuple,        # (x_c, y_c, λ_c) center input position
    x: jnp.ndarray,          # [N] input x positions
    y: jnp.ndarray,          # [N] input y positions
    lam: jnp.ndarray,        # [N] input wavelengths
) -> tuple:                  # (x_out, y_out) arrays of shape [N]
    """
    Disperse points using linear Jacobian approximation.

    (x', y') ≈ (x_c', y_c') + J @ [x - x_c, y - y_c, λ - λ_c]
    """
    dx = x - center_in[0]
    dy = y - center_in[1]
    dlam = lam - center_in[2]

    delta = jnp.stack([dx, dy, dlam], axis=-1)  # [N, 3]
    output_delta = delta @ jacobian.T           # [N, 2]

    x_out = center_out[0] + output_delta[:, 0]
    y_out = center_out[1] + output_delta[:, 1]

    return x_out, y_out
```

#### Step 5: Subgrid Accumulation and Deposition

**Note:** Subgrid size validation is done in the test suite (see Testing Plan), not at runtime.

```python
def deposit_to_subgrid(
    x_out: jnp.ndarray,      # [N] output x positions (SCA coords)
    y_out: jnp.ndarray,      # [N] output y positions (SCA coords)
    flux: jnp.ndarray,       # [N] flux values
    subgrid_size: int = 32,  # Subgrid dimension
) -> tuple:                  # (subgrid, x_offset, y_offset)
    """
    Deposit points into a small subgrid centered on the output region.

    Returns:
        subgrid: [subgrid_size, subgrid_size] accumulated flux
        x_offset, y_offset: Integer offsets for placing subgrid on detector
    """
    # Find bounding box of output points
    x_min, x_max = x_out.min(), x_out.max()
    y_min, y_max = y_out.min(), y_out.max()

    # Center subgrid on output region
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Integer offset for subgrid placement
    x_offset = jnp.floor(x_center - subgrid_size / 2).astype(jnp.int32)
    y_offset = jnp.floor(y_center - subgrid_size / 2).astype(jnp.int32)

    # Convert to subgrid coordinates
    x_sub = x_out - x_offset - 0.5  # FITS to 0-indexed
    y_sub = y_out - y_offset - 0.5

    # Deposit to subgrid (nearest-neighbor - point goes to single pixel)
    subgrid = jnp.zeros((subgrid_size, subgrid_size), dtype=jnp.float32)
    ix = jnp.floor(x_sub).astype(jnp.int32)
    iy = jnp.floor(y_sub).astype(jnp.int32)

    # Use scatter-add with bounds checking
    valid = (ix >= 0) & (ix < subgrid_size) & (iy >= 0) & (iy < subgrid_size)
    subgrid = subgrid.at[iy, ix].add(jnp.where(valid, flux, 0.0))

    return subgrid, x_offset, y_offset


def add_subgrid_to_detector(
    detector: jnp.ndarray,   # [4088, 4088] output detector
    subgrid: jnp.ndarray,    # [S, S] subgrid to add
    x_offset: int,           # x offset on detector (0-indexed)
    y_offset: int,           # y offset on detector (0-indexed)
) -> jnp.ndarray:
    """Add subgrid to detector at specified offset."""
    S = subgrid.shape[0]

    # Use dynamic_slice + scatter for JIT compatibility
    # Handle boundary cases where subgrid extends beyond detector
    detector = jax.lax.dynamic_update_slice(
        detector,
        detector[y_offset:y_offset+S, x_offset:x_offset+S] + subgrid,
        (y_offset, x_offset)
    )
    return detector
```

#### Step 6: Main Sobol Disperser Function

```python
def disperse_sobol_2d1d_sca(
    payload: dict,
    image: jnp.ndarray,      # [Ny, Nx] input image
    x0: float, y0: float,    # Image origin (SCA coords)
    dx: float, dy: float,    # Pixel spacing
    spectrum: jnp.ndarray,   # [Nlam] spectrum
    lam0: float, dlam: float,# Wavelength grid
    output: jnp.ndarray,     # [4088, 4088] accumulator
    cell_size_x: float = 10.0,   # Cell size in SCA pixels
    cell_size_y: float = 10.0,
    cell_size_lam: float = 0.01, # Cell size in microns (100Å)
    n_sobol: int = 8192,         # Sobol points per cell
    subgrid_size: int = 32,      # Subgrid dimension
    sobol_seed: int = 0,
) -> jnp.ndarray:
    """
    Disperse 2D image × 1D spectrum using Sobol sampling and Jacobian approximation.
    """
    Ny, Nx = image.shape
    Nlam = spectrum.shape[0]

    # Compute cell grid
    x_max = x0 + (Nx - 1) * dx
    y_max = y0 + (Ny - 1) * dy
    lam_max = lam0 + (Nlam - 1) * dlam

    n_cells_x = int(jnp.ceil((x_max - x0) / cell_size_x))
    n_cells_y = int(jnp.ceil((y_max - y0) / cell_size_y))
    n_cells_lam = int(jnp.ceil((lam_max - lam0) / cell_size_lam))

    # Pre-generate Sobol points (one set, reused for all cells)
    sobol_unit = get_sobol_points_unit(n_sobol, seed=sobol_seed)  # [N, 3] in [0,1]^3

    # Process cells (using scan for JIT compatibility)
    def process_cell(carry, cell_idx):
        output = carry

        # Unravel cell index
        ic_x = cell_idx % n_cells_x
        ic_y = (cell_idx // n_cells_x) % n_cells_y
        ic_lam = cell_idx // (n_cells_x * n_cells_y)

        # Cell bounds
        x_cell_min = x0 + ic_x * cell_size_x
        y_cell_min = y0 + ic_y * cell_size_y
        lam_cell_min = lam0 + ic_lam * cell_size_lam

        x_cell_max = jnp.minimum(x_cell_min + cell_size_x, x_max)
        y_cell_max = jnp.minimum(y_cell_min + cell_size_y, y_max)
        lam_cell_max = jnp.minimum(lam_cell_min + cell_size_lam, lam_max)

        # Cell center
        x_c = (x_cell_min + x_cell_max) / 2
        y_c = (y_cell_min + y_cell_max) / 2
        lam_c = (lam_cell_min + lam_cell_max) / 2

        # Scale Sobol points to cell
        x_pts = x_cell_min + sobol_unit[:, 0] * (x_cell_max - x_cell_min)
        y_pts = y_cell_min + sobol_unit[:, 1] * (y_cell_max - y_cell_min)
        lam_pts = lam_cell_min + sobol_unit[:, 2] * (lam_cell_max - lam_cell_min)

        # Compute center output and Jacobian
        center_out, J = compute_jacobian_single(payload, x_c, y_c, lam_c)

        # Sample flux at Sobol points
        cell_volume = (x_cell_max - x_cell_min) * (y_cell_max - y_cell_min) * (lam_cell_max - lam_cell_min)
        weight = cell_volume / n_sobol
        flux = sample_flux_bilinear(image, x0, y0, dx, dy, spectrum, lam0, dlam,
                                     x_pts, y_pts, lam_pts) * weight

        # Disperse using Jacobian
        x_out, y_out = disperse_points_jacobian(
            center_out, J, (x_c, y_c, lam_c), x_pts, y_pts, lam_pts
        )

        # Deposit to subgrid and add to detector
        subgrid, x_off, y_off = deposit_to_subgrid(x_out, y_out, flux, subgrid_size)
        output = add_subgrid_to_detector(output, subgrid, x_off, y_off)

        return output, None

    n_cells_total = n_cells_x * n_cells_y * n_cells_lam
    output, _ = jax.lax.scan(process_cell, output, jnp.arange(n_cells_total))

    return output
```

#### Step 7: Update Existing Disperser

Modify existing `disperse_2d1d_sca` to use nearest-neighbor deposit instead of bilinear scatter:

```python
def deposit_nearest(output, x, y, values):
    """Deposit values to nearest pixel (no bilinear spreading)."""
    x_idx = jnp.floor(x - 0.5).astype(jnp.int32)  # FITS to 0-indexed
    y_idx = jnp.floor(y - 0.5).astype(jnp.int32)

    valid = (x_idx >= 0) & (x_idx < 4088) & (y_idx >= 0) & (y_idx < 4088)
    return output.at[y_idx, x_idx].add(jnp.where(valid, values, 0.0))
```

Add parameter to choose method:
```python
def disperse_2d1d_sca(..., scatter_method: str = "nearest"):
    # ... existing code ...
    if scatter_method == "bilinear":
        output = bilinear_scatter_add(output, x_out, y_out, flux)
    else:  # "nearest"
        output = deposit_nearest(output, x_out, y_out, flux)
```

### Testing Plan

#### Parameter Validation Tests (One-Time, Like Jacobian Accuracy)

1. **Subgrid size validation** (`test_subgrid_size.py`):
   - For each SCA (1-18) and order ("0", "1", "2")
   - Compute Jacobian at multiple positions across detector
   - Verify max displacement < subgrid_size/2 - margin for default cell size (10×10×100Å)
   - Similar structure to `test_jacobian_accuracy.py`
   - **Formula:** `max_disp = max(|J[0,:]|, |J[1,:]|) · [Δx/2, Δy/2, Δλ/2]`
   - For 10×10×0.01μm cell with typical J, expect max_disp ≈ 10-15 pixels → 32×32 OK

#### Unit Tests

1. **Sobol point distribution**: Verify points cover cell uniformly
2. **Flux sampling**: Compare bilinear sampling vs known values
3. **Jacobian dispersion**: Verify against full trace_sca_to_sca for cell corners
4. **Subgrid bounds**: Check subgrid captures all dispersed points
5. **Flux conservation**: Total output flux = total input flux (for centered sources)

#### Integration Tests

1. **Delta function**: Single-pixel image should produce correct trace
2. **Comparison test**: Sobol vs bilinear for same input, compare patterns
3. **Large input**: 150×150×1000 input, verify correct total flux

#### Performance Benchmarks

1. Time per galaxy vs existing bilinear approach
2. GPU vs CPU scaling
3. Memory usage vs n_sobol parameter

### Files to Modify/Create

| File | Action |
|------|--------|
| `src/roman_disperser/disperser.py` | Add `disperse_sobol_2d1d_sca`, add `scatter_method` to existing |
| `tests/test_disperser.py` | Add Sobol disperser tests |
| `tests/test_subgrid_size.py` | Add subgrid validation tests |

### Open Questions / Considerations

1. **Jacobian computation cost**: Computing Jacobian per cell might be expensive. Consider:
   - Caching Jacobians for common cell positions
   - Using finite differences instead of autodiff for speed

2. **Edge cells**: Cells at boundaries may be smaller than standard size. Current plan handles this.

3. **Subgrid boundary handling**: When subgrid extends beyond detector edge, need careful handling. Current `dynamic_update_slice` may need adjustment.

4. **Sobol sequence length**: For 8192 points, need 2^13 = 8192 exactly. Consider rounding user input to nearest power of 2.

5. **vmap vs scan for cell processing**:
   - **scan (current plan)**: Processes cells sequentially. Memory bounded. Simple.
   - **vmap over all cells**: Would parallelize all ~2250 cells. Memory estimate:
     - 2250 cells × 8192 points × 20 bytes/point ≈ 350 MB (point data)
     - 2250 × 32×32 × 4 bytes ≈ 9 MB (subgrids)
     - Total ~400 MB peak - **feasible on GPU with 8+ GB**
   - **Hybrid (recommended)**: `vmap` over batches of 100-500 cells, `scan` over batches.
     - Balances parallelism with memory
     - Good GPU utilization without risk of OOM

6. **Subgrid size validation**: This is a parameter validation, not a runtime check. Should be tested once (like Jacobian accuracy) as part of the test suite, not at runtime.

### Suggested Improvements

1. **Adaptive Sobol count**: Use fewer points for cells with low flux (based on quick flux estimate)

2. **Vectorized cell processing**: Instead of scan over cells, batch cells together for better GPU utilization

3. **Precomputed Jacobian grid**: For repeated dispersions at similar positions, cache Jacobians on a coarse grid

4. **Hybrid approach**: Use full trace_sca_to_sca for cells near wavelength boundaries where Jacobian approximation may be less accurate (though Step 0 shows this is fine)


# Goal

Disperse a set of stars throughout the grism.

# Proposed Algorithm (for one star)

1. Take the star's position in SCA coordinates.
2. Loop over wavelengths from 0.9 to 2.0 microns in say 100 Angstrom steps.
3. Find the new SCA position for the dispersed position using the optical model.
4. Interpolate the PSF at this position and wavelength.
5. Deposit/accumulate the PSF position into the output image.

## Open Question: PSF Position for Interpolation

**Which position should be used for PSF lookup?**

There are two valid approaches for step 4:

### Option A: Undispersed Position
- Use the original star position (x₀, y₀) for PSF lookup at all wavelengths
- Same spatial position for all wavelengths → more efficient
- Single bilinear spatial interpolation, then slice all wavelengths
- Use `interpolate_psf_spatial(payload, x₀, y₀)` → returns `[N_wl, PSF_y, PSF_x]`

### Option B: Dispersed Position
- Use the dispersed position (xλ, yλ) for PSF lookup at each wavelength
- Different spatial position for each wavelength → requires per-wavelength interpolation
- Use `interpolate_psf(payload, xλ, yλ, λ)` for each wavelength

### Which is Correct?

Both are physically valid. The choice depends on how STPSF defines its `detector_position`:
- If `detector_position` means "where light originates in the field" → use **undispersed**
- If `detector_position` means "where light lands on detector" → use **dispersed**

**Current status:** The PSF grid uses `[N_y, N_x, N_wl, ...]` ordering which is efficient for
both approaches. Both interpolation functions are implemented. The choice of which to use
will be determined when we better understand STPSF's coordinate convention.

**Practical note:** PSF analysis showed field dependence is minimal for enclosed energy curves,
so the difference may be small in practice.

# Design Phases

## Phase 1 : Build data model for PSF interpolation ✅ COMPLETE

- Explore the enclosed energy and determine how many pixels we will need.
- Oversampling of ~4 (parameter, but pick a good default)
- Needs to be GPU friendly (in terms of memory, access patterns, and parallelism)
- Determine the PSF interpolation method (I think we want trilinear) and implement it efficiently for GPU execution.
- what spatial and wavelength grid do we need here? Define some validation tests.
- write functions to build grids for different grism orders/detectors and do the necessary interpolations.

**Phase 1 Results (2026-01-25):**
- Optimal grid: **4×4 spatial + 0.02 μm wavelength** (896 PSFs, 121 MB, ~5.5 min per SCA)
- Accuracy: **<0.03% max flux error** across all 18 SCAs × 2 orders
- Radial error: **<5% at all radii** (target <10%)
- Implementation: `psf_model.py` with trilinear interpolation, JIT-compatible
- Caching: `scripts/generate_psf_caches.py` for batch generation (~2 hours with 2 workers)
- Validation: `notebooks/psf/psf_allsca_validation.ipynb` (all 36 detector/order combinations)
- See `docs/psf_phase1_plan.md` for detailed findings

## Phase 2 : Single star ✅ COMPLETE

Implement the proposed algorithm
- For now, use GRISM0 for zeroth order, GRISM1 for all others (1,2)

**Phase 2 Results (2026-01-30):**
- Implementation: `star_disperser.py` with `disperse_star_psf()` and `make_star_disperser()` factory
- Uses **undispersed position** for PSF lookup (Option A above)
- Memory-efficient chunked approach using `jax.lax.scan`:
  - Processes wavelengths in configurable chunks (default 1000)
  - Peak memory ~620 MB vs ~1.4 GB for non-chunked approach
  - Memory independent of total wavelength count
- JIT-compatible with closure pattern for payloads
- New vectorized helper: `psf_model.interp_wavelength_chunk()` for batch wavelength interpolation
- Tests: 24 tests in `tests/test_star_disperser.py` including chunk invariance
- Demo: `notebooks/psf/single_star_demo.ipynb`

**Memory budget (8GB target):**
- PSF grid after spatial interpolation: ~8 MB
- Per-chunk with `chunk_size=1000`: ~543 MB
- Output image: ~67 MB
- Peak memory: ~620 MB

**Wavelength scaling:**
- 2Å spacing (5,500 wavelengths): 6 chunks
- 1Å spacing (11,000 wavelengths): 11 chunks
- 0.5Å spacing (22,000 wavelengths): 22 chunks
- All use same ~620 MB peak memory

## Phase 3 : Scaling tests

- See how efficiently this runs with large numbers of stars.

## Phase 4 : Incorporate grism efficiencies + spectra 

- start with a single spectrum.



# Requirements

- Use JAX, write for efficiency and ability to jit, and run efficiently on a GPU.




# References

@docs/disperser_design.md
@docs/optical_model.md
@docs/stpsf.md (quick reference; see @docs/reference/stpsf_full.md for full details)
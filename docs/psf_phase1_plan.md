# Phase 1 Plan: PSF Data Model for Star Dispersion

> **Task:** Build PSF interpolation infrastructure for dispersing stars through Roman grism
> **Focus:** Phase 1 - PSF data model and trilinear interpolation
> **Branch:** stars
> **Created:** 2026-01-24

## Executive Summary

Phase 1 creates a GPU-friendly PSF data model that enables efficient interpolation of Point Spread Functions across spatial position (x, y) and wavelength (λ). The implementation integrates STPSF-generated PSFs with the existing JAX-based disperser using trilinear interpolation.

**Key Challenge:** Coordinate system mismatch between STPSF (0-indexed, 4096×4096) and disperser (1-indexed FITS, 4088×4088) must be resolved.

---

## Current State Assessment

### ✅ Available Infrastructure

1. **STPSF Integration (Complete)**
   - Package: `stpsf >= 2.1.0` installed
   - Documentation: `docs/stpsf.md` (1976 lines, comprehensive)
   - API: `calc_datacube()` for multi-wavelength PSF generation
   - Grism support: GRISM0 (zeroth order), GRISM1 (first order)
   - Wavelength range: 1.0-1.93 μm

2. **JAX Infrastructure (Ready)**
   - Optical model: `optical_model_jax.py` with JIT-compatible payload pattern
   - Disperser: `disperser.py` with bilinear scatter-add
   - Coordinate transforms: `sca_to_fpa()`, `fpa_to_mpa()`, `trace_beam()`
   - All functions use `jnp` arrays for GPU acceleration

3. **Existing Design Patterns**
   - Payload dict approach (see `make_sca_payload()`)
   - Closure pattern for JIT compilation (see `docs/jit_compilation.md`)
   - Wavelength chunking for memory efficiency (see `disperser.py`)
   - Bilinear interpolation example (in `bilinear_scatter_add()`)

### ❌ Missing Components

1. **No PSF Data Structures**
   - No PSF payload class or dict structure
   - No PSF grid storage format
   - No PSF caching mechanisms

2. **No PSF Interpolation**
   - No trilinear interpolation routine
   - No wavelength interpolation
   - No spatial position interpolation

3. **⚠️ Coordinate System Blocker**
   - STPSF uses 0-indexed, 4096×4096 pixels
   - Disperser uses 1-indexed FITS, 4088×4088 pixels
   - Conversion offset unknown (centered? offset? where?)
   - Must resolve before PSF integration

---

## Phase 1 Goals

### Primary Objectives

1. **Resolve coordinate system mismatch** between STPSF and disperser
2. **Design PSF payload structure** for JIT-compatible storage
3. **Implement trilinear PSF interpolation** for GPU execution
4. **Determine optimal sampling grids** for wavelength and spatial position
5. **Validate PSF quality** (enclosed energy, sampling adequacy)
6. **Create PSF generation utilities** to build payloads from STPSF

### Success Criteria

**Core Functionality:**
- [x] Coordinate conversion hardcoded and JIT-compatible (open question tracked)
- [x] Coordinate functions compile under @jax.jit
- [x] Round-trip conversion exact (no rounding errors)
- [x] PSF payload generated (optimized: 4×4 spatial grid, 56 wavelengths at 0.02 μm)
- [x] **OVERDIST extension used** (4× oversampling + detector effects)
- [x] **Timing benchmarks** completed for PSF grid generation (~5.5 min for default)
- [ ] Caching implemented **only if** generation time >10 minutes (not needed - 5.5 min is fast enough)
- [x] Trilinear interpolation with **edge extrapolation** implemented and JIT-compilable

**Accuracy Validation:**
- [x] PSF interpolation achieves <1% flux error vs direct STPSF (**achieved <0.002%**)
- [x] Enclosed energy >95% in all PSFs (5" FOV captures 96-98%)
- [x] Interpolation errors documented across grid (see validation notebook)
- [x] Edge cases handled gracefully (warnings, not crashes)

**Performance:**
- [x] PSF grid generation timing measured and documented (~5.5 min for 4×4×56)
- [x] JIT compilation works with PSF payload (closure pattern)
- [x] JIT compilation works with coordinate conversion functions
- [x] PSF grid fits in GPU memory (~121 MB for 4×4×56 grid, 5" FOV, 4× oversampled)
- [x] Interpolation runs efficiently on GPU (~5-6 ms per interpolation)
- [x] Edge extrapolation handles off-grid positions correctly
- [ ] Caching added if needed based on performance data (not needed for Phase 1)

**Integration:**
- [ ] `disperse_star_psf()` function working for single star
- [ ] PSF correctly deposited along spectral trace
- [ ] Wavelength-dependent PSF changes visible in output

**Documentation:**
- [ ] Four notebooks created demonstrating each phase (1 of 4 complete)
- [x] Coordinate assumptions clearly documented with warnings
- [x] Usage examples provided in documentation
- [x] Known limitations listed for future work
- [x] Validation notebook runs successfully and produces expected visualizations

---

## Design Decisions (User-Confirmed)

### 1. Coordinate System Strategy: Placeholder + Documentation
**Decision:** Use placeholder conversion with documented uncertainty
- Assume centered 4-pixel border (4088 region in center of 4096 array)
- Add clear warnings in code and documentation about unverified assumptions
- Create validation tests to detect coordinate errors
- Defer full resolution to future work if placeholder proves adequate

### 2. PSF Field Dependence: Full Field-Dependent
**Decision:** Implement 4×4 spatial grid across single SCA (updated from 10×10)
- 4×4 is sufficient for PSF core accuracy; coarser (3×3) degrades core
- Finer (5×5, 10×10) provides no additional benefit
- Grid spans full detector range (1 to 4088); STPSF handles edge extrapolation
- Focus on single SCA (WFI05 recommended - central detector)

### 3. Accuracy Target: <1% with Good Validation
**Decision:** Science-ready quality
- Target: <1% flux conservation errors
- Implement thorough validation tests
- Document interpolation accuracy across grid

### 4. Performance: Measure First, Cache If Needed
**Decision:** Time grid generation, defer caching decision to data
- Measure actual generation time with timing benchmarks
- If <5 min, skip caching for Phase 1
- If >10 min, implement disk caching to HDF5/npz

### 5. PSF Oversampling and FOV: Always Use OVERDIST
**Decision:** CRITICAL - always use 4× oversampled PSFs with detector effects
- Stars land at sub-pixel positions after dispersion
- Need oversampled PSFs for accurate flux deposition
- Always use STPSF OVERDIST extension (oversampled + detector effects)
- OVERDIST includes geometric distortion, charge diffusion, pixel sampling
- No fast path - detector effects always included (Roman detectors always have distortion)
- FOV: 5" (not 3") for better flux conservation (95-98% vs 92-96%)
- Wavelength range: 0.9-2.0 μm (ignore STPSF warnings about reference data edges)

### 6. Documentation Requirement: Development Notebooks
**Decision:** Create 4 notebooks demonstrating each development step
- Show high-level assessment of success at each phase
- Include visualizations (PSF grids, interpolation accuracy, validation plots)
- Provide clear examples of usage patterns

---

## Empirical Findings from PSF Analysis

> **Source:** `notebooks/psf/psf_analysis.ipynb` - comprehensive PSF characterization notebook

### PSF Size Measurements (GRISM1, 4× oversampled, OVERDIST extension)

| Wavelength | EE50 Radius | EE90 Radius | EE95 Radius | FWHM (≈2×EE50) |
|------------|-------------|-------------|-------------|----------------|
| 1.0 μm     | ~0.08"      | ~0.6"       | ~1.1"       | ~0.16"         |
| 1.5 μm     | ~0.10"      | ~0.7"       | ~1.3"       | ~0.20"         |
| 1.93 μm    | ~0.125"     | ~1.0"       | ~1.8"       | ~0.25"         |

### Key Observations

1. **Field Dependence is Minimal (for EE curves)**
   - Encircled energy curves are nearly identical at detector center vs corners
   - EE curves overlap closely across all 5 positions tested
   - Note: PSF shapes may still vary across the field; EE is just one metric

2. **Performance**
   - ~0.4-0.5 seconds per PSF calculation with STPSF
   - Full 10×10×15 grid would take ~10-12 minutes (caching recommended)

3. **5" FOV is Sufficient**
   - Captures >95% encircled energy at all wavelengths
   - Max radius of 2.5" from PSF center
   - EE95 is ~1.8" at longest wavelength, well within bounds

4. **STPSF Edge Handling**
   - Corner positions (1,1), (1,4088), (4088,1), (4088,4088) trigger warnings about being outside reference data range
   - PSFs are still generated using STPSF's internal model
   - Use full grid range (1 to 4088) and let STPSF handle edge extrapolation

### Implications for Implementation

- **Spatial Grid:** 4×4 grid spans full detector range; STPSF handles edge cases
- **Wavelength Grid:** 0.02 μm spacing (56 wavelengths) for wing accuracy
- **Order Grid:** Maintain separate grids for each order (PSF shapes may differ even if EE is similar)
- **Caching:** Not needed - 5.5 min generation time is acceptable
- **FOV:** 5" provides good margin for EE95 at all wavelengths

---

## Interpolation Validation Results

> **Source:** `notebooks/psf/psf_interpolation_validation.ipynb` - comprehensive validation notebook

### Grid Configuration Comparison (128 Sobol test points, WFI02)

| Configuration | Grid Shape | Memory | Gen Time | Max Flux Error |
|---------------|------------|--------|----------|----------------|
| 5×5, 0.1 μm   | (12, 5, 5) | 41 MB | 1.8 min | 0.0122% |
| 5×5, 0.05 μm  | (23, 5, 5) | 78 MB | 3.5 min | 0.0042% |
| 10×10, 0.1 μm | (12, 10, 10) | 163 MB | 7.3 min | 0.0122% |
| 3×3, 0.02 μm  | (56, 3, 3) | 68 MB | 3.1 min | 0.0013% |
| **4×4, 0.02 μm** | **(56, 4, 4)** | **121 MB** | **5.5 min** | **0.0013%** |

### Key Findings

1. **Wavelength sampling dominates total flux accuracy**
   - 5×5 vs 10×10 spatial gives identical flux errors (0.0122% max)
   - Finer wavelength (0.02 μm) reduces errors 10× vs coarser (0.1 μm)

2. **Spatial sampling matters for PSF core**
   - 3×3 spatial: excellent total flux but >1% fractional error at small radii
   - 4×4 spatial: maintains <1% fractional error at all radii
   - 5×5 and 10×10: no additional benefit over 4×4

3. **Recommended Configuration: 4×4 spatial + 0.02 μm wavelength**
   - Best balance of core and wing accuracy
   - <0.002% max flux error (500× better than 1% target)
   - <1% fractional error at all radii
   - 896 PSFs, 121 MB memory, ~5.5 min generation

4. **All-SCA timing estimate**
   - 18 SCAs × 6.3 min/SCA = ~1.9 hours total
   - Validation time: ~50 sec per SCA (128 test points)

---

## Implementation Roadmap

### Phase 1A: Core Infrastructure
1. **Coordinate utilities** (`psf_utils.py`)
   - Implement placeholder conversion with warnings
   - Add round-trip validation tests
   - Document assumptions clearly

2. **PSF payload structure** (`psf_model.py`)
   - Define payload dict schema
   - Implement caching (save/load HDF5) - DEFERRED until timing data
   - Test with coarse grid first (5×5×5 for speed)

3. **Notebook 01:** PSF Generation (`notebooks/01_psf_generation.ipynb`)
   - Timing benchmark: Generate 1×1×1 grid and measure time
   - Extrapolate to full grid
   - Generate coarse test grid (5×5 spatial, 5 wavelengths) with timing
   - Visualize PSF grid structure
   - Check enclosed energy
   - Generate full grid (10×10×15) with detailed timing
   - **Decision:** Add caching if >10 min

### Phase 1B: Interpolation
4. **Trilinear interpolation** (`psf_model.py`)
   - Implement `interpolate_psf()` function
   - Add edge extrapolation
   - JIT compile for GPU

5. **Notebook 02:** Interpolation Validation (`notebooks/02_psf_interpolation.ipynb`)
   - Test interpolation at grid points
   - Test at mid-points
   - Compare interpolated vs direct STPSF
   - Measure errors across wavelength range
   - Visualize accuracy map

### Phase 1C: Integration
6. **Disperser integration** (`disperser.py`)
   - Implement `disperse_star_psf()` function
   - Handle PSF scattering onto detector
   - Test with single star

7. **Notebook 03:** Single Star Demo (`notebooks/03_single_star_demo.ipynb`)
   - Disperse single star at detector center
   - Show PSF along spectral trace
   - Compare with point-source dispersion
   - Visualize wavelength-dependent PSF

### Phase 1D: Validation
8. **Comprehensive tests** (`tests/test_psf_model.py`)
   - Coordinate round-trip tests
   - PSF interpolation accuracy (<1% flux error)
   - Enclosed energy validation
   - JIT compilation tests
   - Edge case handling

9. **Notebook 04:** Validation Suite (`notebooks/04_validation_suite.ipynb`)
   - Run all validation tests with visualizations
   - Create diagnostic plots
   - Document known limitations
   - Compare with design requirements

### Phase 1E: Documentation
10. **Usage documentation** (`docs/psf_integration.md`)
    - Document coordinate system assumptions
    - Provide usage examples
    - List known limitations

---

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `src/roman_disperser/psf_utils.py` | ✅ **Done** | Coordinate conversion utilities |
| `src/roman_disperser/psf_model.py` | ✅ **Done** | PSF payload, interpolation (defaults: 4×4, 0.02μm) |
| `src/roman_disperser/disperser.py` | **Modify** | Add `disperse_star_psf()` |
| `tests/test_psf_model.py` | ✅ **Done** | PSF validation tests (21 tests) |
| `docs/psf_integration.md` | **Create** | Coordinate systems and usage |
| `notebooks/psf/psf_analysis.ipynb` | ✅ **Done** | PSF characterization and EE analysis |
| `notebooks/psf/psf_interpolation_validation.ipynb` | ✅ **Done** | Grid optimization and accuracy validation |
| `notebooks/03_single_star_demo.ipynb` | **Create** | Single star dispersion |
| `notebooks/04_validation_suite.ipynb` | **Create** | Comprehensive validation (all SCAs) |

---

## References

- @docs/star_dispersion.md - Original Phase 1 requirements
- @docs/stpsf.md - STPSF integration guide (Section 18 on coordinates!)
- @docs/optical_model.md - JAX optical model API
- @docs/disperser_design.md - Disperser architecture
- @docs/jit_compilation.md - JIT patterns for payloads

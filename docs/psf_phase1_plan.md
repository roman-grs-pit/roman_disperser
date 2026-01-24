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
- [ ] Coordinate conversion hardcoded and JIT-compatible (open question tracked)
- [ ] Coordinate functions compile under @jax.jit
- [ ] Round-trip conversion exact (no rounding errors)
- [ ] PSF payload generated for WFI05, 10×10 spatial grid, 15 wavelengths
- [ ] **OVERDIST extension used** (4× oversampling + detector effects)
- [ ] **Timing benchmarks** completed for PSF grid generation
- [ ] Caching implemented **only if** generation time >10 minutes
- [ ] Trilinear interpolation with **edge extrapolation** implemented and JIT-compilable

**Accuracy Validation:**
- [ ] PSF interpolation achieves <1% flux error vs direct STPSF
- [ ] Enclosed energy >95% in all PSFs (5" FOV captures 96-98%)
- [ ] Interpolation errors documented across grid
- [ ] Edge cases handled gracefully (warnings, not crashes)

**Performance:**
- [ ] PSF grid generation timing measured and documented
- [ ] JIT compilation works with PSF payload (closure pattern)
- [ ] JIT compilation works with coordinate conversion functions
- [ ] PSF grid fits in GPU memory (~188 MB for 10×10×15 grid, 5" FOV, 4× oversampled)
- [ ] Interpolation runs efficiently on GPU
- [ ] Edge extrapolation handles off-grid positions correctly
- [ ] Caching added if needed based on performance data (optional for Phase 1)

**Integration:**
- [ ] `disperse_star_psf()` function working for single star
- [ ] PSF correctly deposited along spectral trace
- [ ] Wavelength-dependent PSF changes visible in output

**Documentation:**
- [ ] Four notebooks created demonstrating each phase
- [ ] Coordinate assumptions clearly documented with warnings
- [ ] Usage examples provided in documentation
- [ ] Known limitations listed for future work
- [ ] All notebooks run successfully and produce expected visualizations

---

## Design Decisions (User-Confirmed)

### 1. Coordinate System Strategy: Placeholder + Documentation
**Decision:** Use placeholder conversion with documented uncertainty
- Assume centered 4-pixel border (4088 region in center of 4096 array)
- Add clear warnings in code and documentation about unverified assumptions
- Create validation tests to detect coordinate errors
- Defer full resolution to future work if placeholder proves adequate

### 2. PSF Field Dependence: Full Field-Dependent
**Decision:** Implement 10×10 spatial grid across single SCA
- Captures field-dependent aberrations realistically
- Focus on single SCA (WFI05 recommended - central detector)
- Grid points: evenly spaced from pixel ~500 to ~3500 (avoiding edges)

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
| `src/roman_disperser/psf_utils.py` | **Create** | Coordinate conversion utilities |
| `src/roman_disperser/psf_model.py` | **Create** | PSF payload, interpolation, caching |
| `src/roman_disperser/disperser.py` | **Modify** | Add `disperse_star_psf()` |
| `tests/test_psf_model.py` | **Create** | PSF validation tests |
| `docs/psf_integration.md` | **Create** | Coordinate systems and usage |
| `notebooks/01_psf_generation.ipynb` | **Create** | Generate and visualize PSF grids |
| `notebooks/02_psf_interpolation.ipynb` | **Create** | Test interpolation accuracy |
| `notebooks/03_single_star_demo.ipynb` | **Create** | Single star dispersion |
| `notebooks/04_validation_suite.ipynb` | **Create** | Comprehensive validation |

---

## References

- @docs/star_dispersion.md - Original Phase 1 requirements
- @docs/stpsf.md - STPSF integration guide (Section 18 on coordinates!)
- @docs/optical_model.md - JAX optical model API
- @docs/disperser_design.md - Disperser architecture
- @docs/jit_compilation.md - JIT patterns for payloads

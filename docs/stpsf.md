# STPSF Quick Reference for Roman WFI Grism Mode

**Purpose:** Quick reference for STPSF usage with Roman WFI grism spectroscopy.

**Version Information:**
- STPSF: v2.2.0 (released Dec 23, 2025)
- Roman Optical Model: Cycle 10 (Sept 2024, GSFC)
- Last Updated: 2026-01-30

**Official Resources:**
- Documentation: https://stpsf.readthedocs.io
- Repository: https://github.com/spacetelescope/stpsf

**For full documentation:** See `docs/stpsf_full.md`

---

## Critical: Coordinate System Warning

**STPSF uses different coordinate conventions than `roman_disperser`:**

| Property | STPSF | roman_disperser & Optical Model |
|----------|-------|--------------------------------|
| **Indexing** | 0-indexed (Python) | 1-indexed (FITS) |
| **Detector Size** | 4096×4096 pixels (full array) | 4088×4088 pixels (usable science region) |
| **Valid Range** | 0 to 4095 | 1 to 4088 |

**Current Implementation:** `psf_utils.sca_to_stpsf_position()` assumes 4-pixel centered border.
This is a working placeholder - interpolation accuracy is excellent (<0.002% flux error),
but the coordinate assumption has not been independently validated.

**TODOs for Integration:**
- [ ] Confirm STPSF detector_position is 0-indexed
- [ ] Confirm STPSF uses full 4096×4096 detector array
- [ ] Determine exact coordinate transformation
- [x] Add utility function for coordinate conversion (`psf_utils.py`)

See Section 6 for detailed coordinate system discussion.

---

## 1. Overview & Quick Start

STPSF (formerly WebbPSF) simulates PSFs for space telescopes. For Roman WFI grism:
- Calculates wavelength-dependent PSF shapes
- Provides `calc_datacube()` for multi-wavelength PSF calculation
- Does **NOT** model dispersion - use `roman_disperser` for spectral traces

```python
import stpsf.roman
import numpy as np

# Create WFI instance
wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Calculate monochromatic PSF at 1.5 microns
psf = wfi.calc_psf(
    monochromatic=1.5e-6,  # wavelength in meters
    fov_arcsec=5.0,
    oversample=4
)

# Get detector-sampled PSF with detector effects
psf_array = psf['DET_DIST'].data

# Enable logging
stpsf.setup_logging('info')
```

**Installation:** `pip install stpsf` (downloads ~GB of reference data on first use)

---

## 2. Grism Quick Reference Card

| Parameter | Value |
|-----------|-------|
| **Physical Element** | G150 grism (single element) |
| **STPSF Filters** | `'GRISM0'` (zeroth order), `'GRISM1'` (first order) |
| **Wavelength Range** | 1.0 - 1.93 μm |
| **Center Wavelength** | 1.465 μm |
| **Spectral Resolution** | R ~ 600 |
| **Detector Scale** | 0.11 arcsec/pixel |
| **PSF Calculation** | `calc_psf(monochromatic=wl)` or `calc_datacube(wavelengths)` |

**Critical Distinction:**
- **GRISM0** = Zeroth order (undispersed direct image)
- **GRISM1** = First order (dispersed spectrum) - primary science mode
- These are spectral orders, not different grisms!

**What STPSF Provides:**
- PSF shape at each wavelength
- Field-dependent aberrations
- Detector effects (pixel sampling, distortion, diffusion)

**What STPSF Does NOT Provide:**
- Dispersion direction or orientation
- Spectral trace geometry
- Spectral extraction tools

---

## 3. WFI Class API

```python
import stpsf.roman

wfi = stpsf.roman.WFI()

# Filter selection
wfi.filter = 'GRISM0'  # Zeroth order (undispersed)
wfi.filter = 'GRISM1'  # First order (dispersed)

# Detector selection (WFI01 through WFI18)
wfi.detector = 'WFI05'  # Central detector

# Position within detector (0-indexed, 0-4095)
wfi.detector_position = (2048, 2048)
```

**18 WFI Detectors:** 4096×4096 pixels each, 0.11 arcsec/pixel

---

## 4. calc_psf() API

```python
psf = wfi.calc_psf(
    # Field of view (specify one)
    fov_pixels=None,        # Array size in detector pixels
    fov_arcsec=5.0,         # Array size in arcseconds

    # Sampling
    oversample=4,           # Oversampling factor (default: 4)

    # Wavelength (use monochromatic for grism)
    monochromatic=1.5e-6,   # Wavelength in meters

    # Output control
    add_distortion=True,    # Include detector effects (default: True)
)
```

**Returns:** FITS HDUList with four extensions:

| Extension | Name | Description |
|-----------|------|-------------|
| 0 | OVERSAMP | Oversampled optical model |
| 1 | DET_SAMP | Detector-sampled optical model |
| 2 | OVERDIST | Oversampled + detector effects |
| 3 | DET_DIST | Detector-sampled + effects (most realistic) |

**For grism:** Always use `monochromatic=<wavelength>` (polychromatic not recommended)

---

## 5. calc_datacube() for Multi-Wavelength PSFs

For grism spectroscopy, calculate PSFs at multiple wavelengths efficiently:

```python
import numpy as np

wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)  # 15 wavelengths in meters

# Full simulation with detector effects
datacube = wfi.calc_datacube(
    wavelengths,
    fov_arcsec=5.0,
    oversample=4,
    add_distortion=True
)

# Extract PSF cube [N_wavelength, N_pix, N_pix]
psf_cube = datacube['DET_DIST'].data  # Most realistic
psf_cube = datacube['OVERDIST'].data  # Oversampled + effects (for sub-pixel work)

# Fast version (~150× faster, no detector effects)
datacube_fast = wfi.calc_datacube_fast(wavelengths, fov_arcsec=3.0)
psf_cube_oversamp = datacube_fast['OVERSAMP'].data
```

**Performance (rough estimates):**
| Method | Time | Extensions |
|--------|------|------------|
| 20× `calc_psf()` | ~40s | 4 per file |
| `calc_datacube()` | ~40s | 4 datacubes |
| `calc_datacube_fast()` | ~0.3s | 1 datacube |

**Alternative:** Loop over `calc_psf()` if `calc_datacube` has issues:
```python
psf_list = []
for wl in wavelengths:
    psf = wfi.calc_psf(monochromatic=wl, fov_arcsec=3.0, oversample=4)
    psf_list.append(psf['DET_DIST'].data)
psf_datacube = np.stack(psf_list, axis=0)
```

---

## 6. Coordinate System Integration

### The Problem

STPSF and `roman_disperser` use different coordinate conventions:

| Aspect | STPSF | roman_disperser |
|--------|-------|-----------------|
| **Indexing** | 0-indexed | 1-indexed FITS |
| **Array Size** | 4096×4096 | 4088×4088 (usable) |
| **Valid Range** | 0 to 4095 | 1 to 4088 |

### Coordinate Conversion

**Current implementation (psf_utils.py):**
```python
from roman_disperser.psf_utils import sca_to_stpsf_position

# Convert disperser coordinates to STPSF
xsca = 2500.5  # 1-indexed FITS (range 1-4088)
x_stpsf, y_stpsf = sca_to_stpsf_position(xsca, ysca)
# Result: x_stpsf = xsca + 3.0 (assumes 4-pixel border)

wfi.detector_position = (x_stpsf, y_stpsf)
```

**Assumption:** 4088 usable region is centered in 4096 array (4-pixel border each side).
This assumption is **UNVERIFIED** but produces excellent interpolation accuracy (<0.03% flux error).

### Implications

- Coordinate transformation required when integrating STPSF PSFs with disperser
- Positions outside 4088×4088 usable region may produce edge effects
- For relative PSF studies (wavelength dependence), coordinate conversion less critical

### Outstanding TODOs

1. Verify STPSF indexing with actual PSF calculations
2. Confirm where 4088 usable region sits in 4096 array
3. Test coordinate conversion with known source positions
4. Document any restrictions or edge cases

---

## 7. Empirical PSF Characterization

From `notebooks/psf/psf_analysis.ipynb`:

| Parameter | Value |
|-----------|-------|
| **Recommended FOV** | 5 arcsec (captures >95% EE at all wavelengths) |
| **EE50 radius** | 0.08" (1.0 μm) to 0.125" (1.93 μm) |
| **EE90 radius** | 0.6" (1.0 μm) to 1.0" (1.93 μm) |
| **EE95 radius** | 1.1" (1.0 μm) to 1.8" (1.93 μm) |
| **FWHM** | ~2 × EE50 radius |
| **Calc time** | ~0.4-0.5s per PSF |

**Observations:**
- Encircled energy curves similar at detector center vs corners
- Corner positions trigger STPSF warnings about reference data range
- Use full grid range (1 to 4088); STPSF handles edge extrapolation

---

## 8. Known Limitations

**Grism-Specific:**
1. STPSF does NOT model grism dispersion - use `roman_disperser`
2. No trace information (direction, amount, geometry)
3. No spectral extraction tools
4. Sparse grism documentation vs imaging mode

**General:**
1. Reference data required (~GB download on first use)
2. No GPU acceleration (CPU-bound)
3. Incomplete detector effects (no cosmic rays, saturation, read noise)

---

## 9. Essential Examples

### Single Wavelength PSF
```python
wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

psf = wfi.calc_psf(monochromatic=1.5e-6, fov_arcsec=5.0, oversample=4)
psf_array = psf['DET_DIST'].data
```

### PSF Grid for Disperser Integration
```python
wavelengths = np.linspace(1.0e-6, 1.93e-6, 20)  # 20 wavelengths
datacube = wfi.calc_datacube(wavelengths, fov_arcsec=5.0, oversample=4)
psf_cube = datacube['OVERDIST'].data  # [N_wl, N_pix*4, N_pix*4]
```

### Field-Dependent PSFs
```python
# PSF at different detector positions
positions = [(1000, 1000), (2048, 2048), (3000, 3000)]
for x, y in positions:
    wfi.detector_position = (x, y)
    psf = wfi.calc_psf(monochromatic=1.5e-6, fov_arcsec=3.0)
```

---

## References

- **Full STPSF Reference:** `docs/stpsf_full.md` (comprehensive, 2000+ lines)
- **PSF Model Implementation:** `src/roman_disperser/psf_model.py`
- **Coordinate Utilities:** `src/roman_disperser/psf_utils.py`
- **Phase 1 Plan:** `docs/psf_phase1_plan.md`
- **STPSF Documentation:** https://stpsf.readthedocs.io

---

**End of Quick Reference**

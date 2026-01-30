# STPSF Reference for Roman WFI (Focus: Grism Mode)

**Purpose:** Comprehensive reference for STPSF (Space Telescope PSF) usage with Roman WFI, with emphasis on grism spectroscopy mode.

**Version Information:**
- STPSF: v2.2.0 (released Dec 23, 2025)
- Roman Optical Model: Cycle 10 (Sept 2024, GSFC)
- Last Updated: 2026-01-24

**Official Resources:**
- Documentation: https://stpsf.readthedocs.io
- Repository: https://github.com/spacetelescope/stpsf
- Tutorial: https://github.com/spacetelescope/stpsf/blob/develop/notebooks/STPSF-Roman_Tutorial.ipynb

**Scope:** This document focuses on Roman WFI, particularly grism spectroscopy. STPSF also supports JWST and other missions.

---

## Important: Coordinate System Differences with roman_disperser

**⚠️ CRITICAL INTEGRATION NOTE:**

STPSF uses different coordinate conventions than the `roman_disperser` optical model and most Roman grism tools:

| Property | STPSF | roman_disperser & Optical Model |
|----------|-------|--------------------------------|
| **Indexing** | 0-indexed (Python) | 1-indexed (FITS) |
| **Detector Size** | 4096×4096 pixels (full array) | 4088×4088 pixels (usable science region) |
| **Valid Range** | 0 to 4095 | 1 to 4088 |

**Implications:**
- **Coordinate shift required:** When converting between STPSF detector_position and disperser SCA coordinates
- **Size mismatch:** STPSF can compute PSFs at positions that fall outside the usable science region
- **Integration:** Care needed when using STPSF PSFs with dispersed spectra from optical model

**TODOs for Integration:**
- [ ] **TODO:** Confirm STPSF detector_position is 0-indexed (initial evidence from source code suggests yes)
- [ ] **TODO:** Confirm STPSF uses full 4096×4096 detector array (not just 4088 usable region)
- [ ] **TODO:** Determine exact coordinate transformation between STPSF (0-indexed, 4096) and optical model (1-indexed FITS, 4088 usable)
- [ ] **TODO:** Document whether 4088 usable region is centered in 4096 array or offset
- [x] Add utility function for coordinate conversion between systems (`psf_utils.py` - uses placeholder assumption)

**Current Implementation:** `psf_utils.sca_to_stpsf_position()` assumes 4-pixel centered border.
This is a working placeholder - interpolation accuracy is excellent (<0.002% flux error),
but the coordinate assumption has not been independently validated.

**Example Issue:**
```python
# Disperser optical model output (1-indexed FITS)
xsca_dispersed = 2500.5  # Valid: 1.0 to 4088.0

# Current implementation (psf_utils.py) - uses placeholder assumption:
from roman_disperser.psf_utils import sca_to_stpsf_position
x_stpsf, y_stpsf = sca_to_stpsf_position(xsca_dispersed, 2000.0)
# Assumes 4-pixel centered border: x_stpsf = xsca + 3.0

# This assumption has NOT been validated against STPSF documentation
# The coordinate offset could be different (e.g., [0:4088] or [4:4092])
```

See Section 18 (end of document) for detailed discussion of this issue.

---

## 1. Overview

STPSF (formerly WebbPSF pre-v2.0) is a Python package for simulating point spread functions (PSFs) for space telescopes. For Roman WFI, it provides:

- **Optical PSF modeling** using OPD (Optical Path Difference) maps
- **Field-dependent aberrations** across the 18 WFI detectors
- **Filter/grism/prism support** for all WFI optical elements
- **Detector effects** including pixel sampling and geometric distortion

**Key Capabilities for Grism Mode:**
- Calculates wavelength-dependent PSF shapes for grism observations
- Provides `calc_datacube()` for efficient multi-wavelength PSF calculation
- Does **NOT** model the dispersion itself (spectral spreading) - that requires a separate optical/disperser model

**Relationship to This Project:** STPSF provides the PSF component; `roman_disperser` handles the spectral trace geometry and 2D→1D dispersion.

---

## 2. Installation

```bash
pip install stpsf
```

**Data Files:** STPSF automatically downloads required reference data on first use:
- OPD maps (optical path difference)
- Filter transmission curves
- Pupil masks
- Detector parameters

Data is cached in `$HOME/stpsf-data/` by default.

**Enable Logging:**
```python
import stpsf
stpsf.setup_logging('info')  # Show calculation progress
```

---

## 3. Quick Start

```python
import stpsf
import stpsf.roman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Create WFI instance
wfi = stpsf.roman.WFI()

# Configure for grism first order
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Calculate monochromatic PSF at 1.5 microns
psf = wfi.calc_psf(
    monochromatic=1.5e-6,  # wavelength in meters
    fov_arcsec=5.0,
    oversample=4
)

# Display the detector-sampled PSF with detector effects
plt.imshow(psf['DET_DIST'].data, norm=LogNorm())
plt.colorbar(label='Intensity')
plt.title('GRISM1 PSF at 1.5 μm')
plt.show()
```

---

## 4. Grism Quick Reference Card

**🌟 Essential Facts for Roman Grism Mode:**

| Parameter | Value |
|-----------|-------|
| **Physical Element** | G150 grism (single element) |
| **STPSF Filters** | `'GRISM0'` (zeroth order), `'GRISM1'` (first order) |
| **Wavelength Range** | 1.0 - 1.93 μm |
| **Center Wavelength** | 1.465 μm |
| **Bandwidth** | 0.930 μm |
| **Spectral Resolution** | R ~ 600 (461 per 2 pixels) |
| **Pupil Mask** | `'GRISM'` (shared by both orders) |
| **Detector Scale** | 0.11 arcsec/pixel (same as imaging) |
| **PSF Calculation** | `calc_psf(monochromatic=wl)` or `calc_datacube(wavelengths)` |
| **Multi-Wavelength** | Use `calc_datacube()` for efficient batch calculation |
| **STPSF Does NOT Model** | Dispersion direction or spectral spreading |

**Critical Distinction:**
- **GRISM0** = Zeroth order (undispersed direct image)
- **GRISM1** = First order (dispersed spectrum)
- These are **spectral orders**, not different grisms!

**Typical Workflow:**
1. Calculate PSFs at 10-20 wavelengths across 1.0-1.93 μm
   - Use `calc_datacube()` for organized output (see Section 7A)
   - Or loop with `calc_psf(monochromatic=wl)` for each wavelength
2. STPSF provides PSF shape at each wavelength
3. Use separate disperser model for spectral trace geometry
4. Combine PSF + dispersion for full grism simulation

**⚠️ Integration Warning:**
- STPSF uses 0-indexed coordinates, 4096×4096 pixels
- roman_disperser uses 1-indexed FITS coordinates, 4088×4088 pixels
- Coordinate conversion required (see Section 18 for TODOs)

---

## 5. WFI Class API

### Initialization

```python
import stpsf.roman

wfi = stpsf.roman.WFI()
```

Default configuration: F062 pupil mask, WFI01 detector.

### Key Attributes

**Filter Selection:**
```python
# Imaging filters
wfi.filter = 'F062'   # 0.480 - 0.760 μm
wfi.filter = 'F087'   # 0.760 - 0.977 μm
wfi.filter = 'F106'   # 0.927 - 1.192 μm
wfi.filter = 'F129'   # 1.131 - 1.454 μm
wfi.filter = 'F146'   # 1.280 - 1.625 μm
wfi.filter = 'F158'   # 1.380 - 1.774 μm
wfi.filter = 'F184'   # 1.683 - 2.000 μm
wfi.filter = 'F213'   # 1.950 - 2.300 μm

# Grism (G150)
wfi.filter = 'GRISM0'  # Zeroth order (undispersed)
wfi.filter = 'GRISM1'  # First order (dispersed)

# Prism
wfi.filter = 'PRISM'   # 0.75 - 1.80 μm, R ~ 80-180
```

**Detector Selection:**
```python
wfi.detector = 'WFI01'  # Through 'WFI18'
```

18 Teledyne H4RG-10 detectors:
- 4096 × 4096 pixels each
- 0.11 arcsec/pixel (110 mas/pixel)
- Numbered WFI01 through WFI18

**Detector Position:**
```python
wfi.detector_position = (2048, 2048)  # (X, Y) in pixels, zero-indexed
```

Position within the selected detector for field-dependent aberrations.
- **Coordinate system:** Zero-indexed (Python convention)
- **Valid range:** 0 to 4095 for 4096×4096 WFI detector
- **Center position:** (2047, 2047) or (2048, 2048) - exact center is at 2047.5
- **First pixel:** (0, 0) (lower-left corner)
- **Last pixel:** (4095, 4095) (upper-right corner)

**⚠️ NOTE:** This differs from the optical model which uses 1-indexed FITS coordinates with a 4088×4088 usable region. See Section 18 for coordinate conversion details.

**Pupil Mask:**

Automatically managed by `WFIPupilController`. Pupil masks vary by filter:
- Imaging filters: Individual pupil masks (Cycle 10 model)
- Grism orders: Both GRISM0 and GRISM1 share `'GRISM'` pupil mask
- Prism: `'PRISM'` pupil mask

Methods:
```python
wfi.lock_pupil()              # Lock pupil to current filter
wfi.lock_pupil_mask('GRISM')  # Explicitly lock to specific mask
```

---

## 6. Roman Grism Mode (★ PRIORITY SECTION ★)

### 6.1 Grism Overview

Roman WFI has **one grism disperser** (G150) mounted in the element wheel:

| Property | Value |
|----------|-------|
| **Physical Element** | G150 grism |
| **Wavelength Range** | 1.0 - 1.93 μm |
| **Center Wavelength** | 1.465 μm |
| **Bandwidth** | 0.930 μm |
| **Spectral Resolution** | R ~ 600 (461 per 2 pixels) |
| **Element Wheel Position** | Between F213 and F062 (clockwise from top) |

The grism provides slitless spectroscopy - every source in the field produces a spectrum.

### 6.2 GRISM0 vs GRISM1 (Critical Distinction)

**⚠️ IMPORTANT:** These are NOT two different grisms. They represent **spectral orders** of the same G150 grism element.

**GRISM0 - Zeroth Order (Undispersed):**
- Direct imaging through the grism (no dispersion)
- Shows stellar/source positions without spectral spreading
- Used to identify source locations in the field
- Acts like direct imaging but with grism optical path

**GRISM1 - First Order (Dispersed):**
- Dispersed spectrum from each source
- Primary science mode for spectroscopy
- Light is spread into a spectrum along the dispersion direction
- Each point source produces a ~continuum trace across wavelengths

**STPSF Implementation:**
```python
import stpsf.roman

wfi = stpsf.roman.WFI()

# Zeroth order PSF (undispersed)
wfi.filter = 'GRISM0'
psf_0th = wfi.calc_psf(monochromatic=1.5e-6)

# First order PSF (dispersed)
wfi.filter = 'GRISM1'
psf_1st = wfi.calc_psf(monochromatic=1.5e-6)
```

**Shared Pupil Mask:**

Both orders share the same pupil mask designation: `'GRISM'`

From STPSF source code (`stpsf/roman.py`):
```python
if wfi_filter in ('GRISM0', 'GRISM1'):
    return 'GRISM'
```

Unlike imaging filters (which have individual pupil masks in Cycle 10), both grism orders share a unified pupil mask. This is managed automatically by the `WFIPupilController` class.

### 6.3 Grism PSF Calculations

**Wavelength Specification:**

STPSF documentation emphasizes: *"Prism and grism PSFs shown here are monochromatic."*

For grism mode, **always use `monochromatic=<wavelength>`**:

```python
import stpsf
import stpsf.roman

wfi = stpsf.roman.WFI()

# Monochromatic grism PSF at 1.5 microns
wfi.filter = 'GRISM1'  # First order
psf = wfi.calc_psf(
    monochromatic=1.5e-6,  # wavelength in meters
    fov_arcsec=5.0,
    oversample=4
)
```

**Why Monochromatic?**

1. **STPSF calculates PSFs at specific wavelengths** - it provides the PSF shape (Airy pattern + aberrations) at each wavelength point
2. **Grism dispersion is NOT modeled by STPSF** - the spectral spreading (creating the spectrum) is handled by separate optical/disperser models
3. **Wavelength-dependent PSFs** - The PSF shape changes with wavelength due to diffraction and chromatic aberrations
4. **Integration with disperser models** - Calculate PSFs at multiple wavelengths, then use your disperser model to place them at the correct positions along the spectral trace

**Polychromatic Grism PSFs:**

While technically possible, polychromatic grism PSFs are not demonstrated in STPSF documentation. For grism mode, monochromatic calculations are recommended and standard practice.

### 6.4 Grism vs Imaging PSF Differences

Key considerations when comparing grism to imaging mode:

- **Different optical path:** Grism PSFs may differ from imaging PSFs due to passing through the grism element
- **Shared pupil mask:** Both GRISM0 and GRISM1 share the same `'GRISM'` pupil mask, but optical effects differ between orders
- **Field-dependent aberrations:** Still apply across the 18 WFI detectors (varies by `detector` and `detector_position`)
- **Pixel scale:** 0.11 arcsec/pixel (same as imaging mode)
- **Wavelength dependence:** PSF shape changes across the grism wavelength range (1.0-1.93 μm)

### 6.5 Dispersion Direction and Orientation

**⚠️ STPSF Limitation:** Dispersion direction is **NOT** provided by STPSF.

- Roman spacecraft can roll ±15° to adjust dispersion direction on sky
- STPSF models PSF shape only, not dispersion geometry
- Actual spectral traces determined by:
  - Grism optical design
  - Spacecraft orientation/roll angle
  - Field position (distortion)
  - Requires separate optical model (e.g., `roman_disperser` in this project)

### 6.6 Grism Wavelength Range Details

| Parameter | Value |
|-----------|-------|
| **Minimum Wavelength** | 1.0 μm |
| **Maximum Wavelength** | 1.93 μm |
| **Center Wavelength** | 1.465 μm |
| **Bandwidth (FWHM)** | 0.930 μm |
| **Spectral Resolution** | R ~ 600 |
| **Resolution (per 2 pixels)** | R ~ 461 |

**Sampling Recommendation:** For PSF calculations across the grism range, sample 10-20 wavelengths:

```python
import numpy as np

# 15 wavelengths across grism range
wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)  # meters
```

### 6.7 Prism vs Grism Comparison

For completeness, WFI also has a prism (`'PRISM'` filter):

| Property | Grism (G150) | Prism |
|----------|--------------|-------|
| **Wavelength Range** | 1.0 - 1.93 μm | 0.75 - 1.80 μm |
| **Center Wavelength** | 1.465 μm | 1.275 μm |
| **Bandwidth** | 0.930 μm | 1.05 μm |
| **Spectral Resolution** | R ~ 600 | R ~ 80-180 (λ-dependent) |
| **Use Case** | Medium-res spectroscopy | Low-res, broad wavelength surveys |
| **STPSF Filter** | `'GRISM0'`, `'GRISM1'` | `'PRISM'` |

**Prism characteristics:**
- Lower resolution but broader wavelength coverage than grism
- Resolution varies with wavelength (R ~ 80 at blue end, ~180 at red end)
- Single optical element (no zeroth/first order distinction)

### 6.8 Known Limitations for Grism Mode

**STPSF does NOT provide:**

1. **No dispersion modeling:** STPSF provides PSF at single wavelength, not the dispersed spectrum
2. **No trace information:** Dispersion direction, amount, and geometry not provided
3. **No spectral extraction:** Tools for extracting 1D spectra from 2D dispersed images not included
4. **Sparse grism documentation:** Grism-specific parameters not extensively documented compared to imaging mode
5. **Monochromatic only (practical):** Polychromatic grism PSFs not demonstrated in examples

**What you need separately:**

- **Optical/disperser model** for spectral trace geometry (e.g., this project's `roman_disperser`)
- **Spectral extraction tools** for reducing grism observations
- **Wavelength calibration** from separate sources

### 6.9 Grism PSF Output Format

Returns standard FITS HDUList with four extensions (same as imaging mode):

| Extension | Name | Description |
|-----------|------|-------------|
| 0 | OVERSAMP | Oversampled optical model (ideal PSF, no detector effects) |
| 1 | DET_SAMP | Detector-sampled optical model (pixel sampling only) |
| 2 | OVERDIST | Oversampled with detector effects (distortion, charge diffusion) |
| 3 | DET_DIST | Detector-sampled with detector effects (most realistic) |

**Recommended for realism:** Extension 3 (`DET_DIST`) includes both pixel sampling and detector effects.

**Accessing data:**
```python
# Get detector-sampled PSF with effects as numpy array
psf_array = psf['DET_DIST'].data

# Or by extension number
psf_array = psf[3].data
```

### 6.10 Practical Usage for Grism

**Typical workflow for grism simulations:**

1. **Calculate monochromatic PSFs** across grism wavelength range (1.0-1.93 μm)
2. **Sample efficiently:** ~10-20 wavelengths for most applications
3. **Extract PSF arrays:** Use wavelength-dependent PSFs as input to disperser model
4. **Combine with dispersion:** Disperser handles spectral spreading; STPSF handles PSF shape at each wavelength
5. **Build detector image:** Accumulate dispersed PSFs along spectral trace

**Example workflow:**
```python
import numpy as np
import stpsf.roman

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Sample wavelengths across grism range
wavelengths = np.linspace(1.0e-6, 1.93e-6, 10)  # 10 wavelengths in meters

# Calculate PSF at each wavelength
psfs = []
for wl in wavelengths:
    psf = wfi.calc_psf(
        monochromatic=wl,
        fov_arcsec=3.0,
        oversample=4
    )
    # Extract the most realistic PSF (detector-sampled + effects)
    psf_array = psf['DET_DIST'].data
    psfs.append(psf_array)
    print(f"Calculated PSF at {wl*1e6:.3f} μm")

# Now use psfs with your disperser model
# psfs[i] corresponds to wavelengths[i]
```

**Memory considerations:**

- Each PSF with `fov_arcsec=3.0`, `oversample=4` is relatively small (~100×100 detector pixels)
- 10-20 PSFs easily fit in memory
- For higher wavelength sampling, calculate on-the-fly in disperser loop

---

## 7. General calc_psf() API Reference

Complete parameter documentation (applies to all filters including grism):

### Basic Signature

```python
psf = wfi.calc_psf(
    # Field of view (specify one)
    fov_pixels=None,        # Array size in detector pixels
    fov_arcsec=5.0,         # Array size in arcseconds (default for Roman tutorial)

    # Sampling
    oversample=4,           # Oversampling factor (default: 4)

    # Wavelength options (mutually exclusive)
    monochromatic=None,     # Single wavelength in meters (e.g., 1.2e-6)
    nlambda=10,             # Number of wavelengths for polychromatic (default: 10)
    source=None,            # Custom spectrum (synphot.SourceSpectrum or dict)

    # Source position
    source_offset_x=0.0,    # Cartesian offset in arcseconds
    source_offset_y=0.0,
    source_offset_r=0.0,    # Polar offset (radius in arcsec)
    source_offset_theta=0.0,# Polar offset (angle in degrees)

    # Output control
    add_distortion=True,    # Include detector effects (default: True)
    normalize='exit_pupil', # Normalization mode ('exit_pupil', 'last', or default)
)
```

### Parameter Details

**Field of View:**
- `fov_pixels`: Output array size in detector pixels (e.g., `fov_pixels=101` for 101×101 array)
- `fov_arcsec`: Output array size in arcseconds (e.g., `fov_arcsec=5.0` for 5″×5″ field)
- Default: `fov_arcsec=5.0` in Roman tutorial examples
- Specify only one; `fov_arcsec` takes precedence if both given

**Sampling:**
- `oversample`: Integer oversampling factor for optical model
  - Default: 4 (each detector pixel sampled with 4×4 subpixels)
  - Higher values = better sampling of PSF core, slower computation
  - Extension 0 (OVERSAMP) has size `fov_pixels × oversample`
  - Extension 1/3 (DET_SAMP/DET_DIST) have size `fov_pixels`

**Wavelength Options (mutually exclusive):**

1. **Monochromatic** (recommended for grism):
   ```python
   psf = wfi.calc_psf(monochromatic=1.5e-6)  # 1.5 μm
   ```
   - Wavelength in meters
   - Single wavelength, fastest calculation
   - **Use this for grism mode**

2. **Polychromatic with nlambda**:
   ```python
   psf = wfi.calc_psf(nlambda=10)  # 10 wavelengths across filter bandpass
   ```
   - Integrates over filter bandpass
   - Weighted by source spectrum (default: 5700K blackbody or flat)
   - Higher `nlambda` = better spectral fidelity, slower computation

3. **Custom source spectrum**:
   ```python
   import synphot
   sp = synphot.SourceSpectrum(synphot.BlackBodyNorm1D, temperature=10000)
   psf = wfi.calc_psf(source=sp, nlambda=10)
   ```
   - Requires `synphot` package
   - Can also pass dict with `'wavelengths'` and `'weights'` keys

**Source Position:**

Two modes (Cartesian or polar):

1. **Cartesian offsets:**
   ```python
   psf = wfi.calc_psf(source_offset_x=0.5, source_offset_y=0.3)  # arcsec
   ```
   - Offsets in arcseconds from detector position
   - X: along detector X-axis, Y: along detector Y-axis

2. **Polar offsets:**
   ```python
   psf = wfi.calc_psf(source_offset_r=1.0, source_offset_theta=45.0)
   ```
   - Radius in arcseconds, theta in degrees
   - Useful for radial symmetry studies

**Output Control:**

- `add_distortion`: Include detector effects (pixel grid, charge diffusion, geometric distortion)
  - Default: `True`
  - Set to `False` for idealized optical model only (faster, no Extensions 2/3)

- `normalize`: PSF normalization method
  - `'exit_pupil'`: Normalize to total flux at exit pupil
  - `'last'`: Normalize final detector-sampled PSF to 1.0
  - Default behavior if not specified

### Return Value

Returns `astropy.io.fits.HDUList` with PSF data in multiple extensions (see section 8).

---

## 7A. calc_datacube() for Multi-Wavelength PSFs

**⚠️ Note:** The `calc_datacube` methods are inherited from the parent `SpaceTelescopeInstrument` class and documented primarily for JWST IFU modes. Their use with Roman WFI grism is **not explicitly documented** but should work. Testing recommended before production use.

### Why Use calc_datacube for Grism Mode?

For grism spectroscopy, you need PSFs at **multiple wavelengths** (typically 10-100 across the 1.0-1.93 μm range). Instead of calling `calc_psf()` in a loop, `calc_datacube` provides:

- **Single function call** for all wavelengths
- **Organized output** as a 3D datacube [wavelength, y, x]
- **Potential optimization** (especially with `calc_datacube_fast`)
- **Consistent parameters** across all wavelength slices

### Two Methods Available

STPSF provides two datacube calculation methods:

| Method | Speed | Output | Detector Effects | Use Case |
|--------|-------|--------|------------------|----------|
| **calc_datacube()** | Standard | 4 extensions (full) | ✅ Yes | Complete simulation |
| **calc_datacube_fast()** | ~150× faster | 1 extension (oversamp only) | ❌ No | Quick analysis |

### calc_datacube() - Full Simulation

**Function Signature:**
```python
datacube = wfi.calc_datacube(
    wavelengths,           # Array of wavelengths in meters
    fov_pixels=None,       # Field of view in pixels
    fov_arcsec=5.0,        # Field of view in arcseconds
    oversample=4,          # Oversampling factor
    add_distortion=True,   # Include detector effects
    outfile=None,          # Optional output filename
    **kwargs               # Additional calc_psf parameters
)
```

**Parameters:**

- **wavelengths** (required): Array-like of wavelengths in meters
  ```python
  wavelengths = np.linspace(1.0e-6, 1.93e-6, 20)  # 20 wavelengths
  ```

- **fov_pixels, fov_arcsec, oversample**: Same as `calc_psf()`

- **add_distortion**: If True, includes all detector effects (pixel sampling, charge diffusion, geometric distortion)

- **outfile**: Optional filename to save datacube

**Returns:**

FITS HDUList with **four extensions**, each containing a 3D datacube [wavelength, y, x]:

| Extension | Name | Description | Shape |
|-----------|------|-------------|-------|
| 0 | OVERSAMP | Oversampled optical PSFs | [N_wavelength, N_pix×oversample, N_pix×oversample] |
| 1 | DET_SAMP | Detector-sampled optical PSFs | [N_wavelength, N_pix, N_pix] |
| 2 | OVERDIST | Oversampled + detector effects | [N_wavelength, N_pix×oversample, N_pix×oversample] |
| 3 | DET_DIST | Detector-sampled + effects (most realistic) | [N_wavelength, N_pix, N_pix] |

**Example Usage:**

```python
import numpy as np
import stpsf.roman

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Define wavelength grid across grism range
wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)  # 15 wavelengths in meters

# Calculate datacube with full detector effects
datacube = wfi.calc_datacube(
    wavelengths,
    fov_arcsec=3.0,
    oversample=4,
    add_distortion=True
)

# Access individual wavelength slices
# datacube[3] = DET_DIST extension (most realistic)
# datacube[3].data.shape = (15, N_pix, N_pix)

# Extract PSF at 5th wavelength (index 4)
psf_at_wl4 = datacube['DET_DIST'].data[4, :, :]  # 2D array

# Get wavelength array from header
wl_array = datacube['DET_DIST'].header['WAVE*']  # Wavelength keywords

print(f"Datacube shape: {datacube['DET_DIST'].data.shape}")
print(f"Wavelengths: {len(wavelengths)}")
```

### calc_datacube_fast() - Optimized Version

**Function Signature:**
```python
datacube = wfi.calc_datacube_fast(
    wavelengths,           # Array of wavelengths in meters
    fov_pixels=None,       # Field of view in pixels
    fov_arcsec=5.0,        # Field of view in arcseconds
    oversample=4,          # Oversampling factor
    add_distortion=True,   # Ignored (always False in fast version)
    compare_methods=False, # Compare fast vs standard methods
    outfile=None,          # Optional output filename
    **kwargs
)
```

**Key Differences from calc_datacube():**

1. **Much faster:** ~150× speedup (from JWST NIRSpec benchmarks)
2. **Simplified output:** Only OVERSAMP extension (no detector sampling or effects)
3. **Assumption:** Wavefront error (OPD) and amplitude are wavelength-independent
   - Calculates exit pupil once, reuses for all wavelengths
   - Reasonable assumption for Roman WFI grism mode

**Returns:**

FITS HDUList with **one extension**:
- Extension 0 (OVERSAMP): Oversampled PSF datacube [N_wavelength, N_pix×oversample, N_pix×oversample]

**When to Use:**

✅ **Good for:**
- Quick PSF shape analysis across wavelengths
- Preliminary grism simulations
- High wavelength sampling (N_wavelength > 50)
- Optical-only studies (no detector effects needed)

❌ **Not suitable for:**
- Cases requiring detector effects (use full `calc_datacube()`)
- Instruments with wavelength-dependent pupils (not applicable to Roman WFI)
- Final/publication-quality simulations (use full version)

**Example Usage:**

```python
import numpy as np
import stpsf.roman

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# High wavelength sampling for smooth PSF evolution
wavelengths = np.linspace(1.0e-6, 1.93e-6, 50)  # 50 wavelengths

# Fast calculation (no detector effects)
datacube_fast = wfi.calc_datacube_fast(
    wavelengths,
    fov_arcsec=3.0,
    oversample=4
)

# Only OVERSAMP extension available
psf_cube = datacube_fast['OVERSAMP'].data  # Shape: (50, N_pix*4, N_pix*4)

# Extract PSF at 10th wavelength
psf_wl10 = psf_cube[10, :, :]

print(f"Fast datacube shape: {psf_cube.shape}")
print(f"Calculation time: ~150x faster than full calc_datacube")
```

### Comparison Mode

`calc_datacube_fast` has a validation mode to compare results:

```python
# Calculate both fast and standard methods for validation
result = wfi.calc_datacube_fast(
    wavelengths,
    compare_methods=True,  # Returns comparison data
    fov_arcsec=3.0
)

# result contains both fast and standard calculations
# Typical difference: ~1/100th or less in oversampled PSF
```

### Performance Comparison

**Example:** 20 wavelengths, 5″ FOV, oversample=4

| Method | Approx. Time | Output Size | Extensions |
|--------|--------------|-------------|------------|
| 20× `calc_psf()` | ~40 seconds | 20 separate files | 4 per file |
| `calc_datacube()` | ~40 seconds | 1 file | 4 datacubes |
| `calc_datacube_fast()` | ~0.3 seconds | 1 file | 1 datacube |

**Note:** Times are illustrative. Actual performance depends on CPU, FOV size, and oversampling.

### Practical Workflow for Grism

**Recommended approach:**

```python
import numpy as np
import stpsf.roman

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Define wavelength grid (10-20 wavelengths usually sufficient)
wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)

# Option 1: Full simulation with detector effects (recommended for final results)
datacube_full = wfi.calc_datacube(
    wavelengths,
    fov_arcsec=3.0,
    oversample=4,
    add_distortion=True
)

# Extract realistic PSFs (DET_DIST extension)
psf_array_3d = datacube_full['DET_DIST'].data  # Shape: (15, N_pix, N_pix)

# Use in disperser: loop over wavelengths
for i, wl in enumerate(wavelengths):
    psf_2d = psf_array_3d[i, :, :]
    # Feed psf_2d to disperser at this wavelength
    # disperser.disperse_2d1d_sca(..., psf=psf_2d, wavelength=wl, ...)


# Option 2: Fast preliminary analysis
datacube_fast = wfi.calc_datacube_fast(
    wavelengths,
    fov_arcsec=3.0,
    oversample=4
)

# Oversampled PSFs only (no detector effects)
psf_oversamp_3d = datacube_fast['OVERSAMP'].data  # Shape: (15, N_pix*4, N_pix*4)
```

### Saving and Loading Datacubes

```python
# Save datacube to file
datacube = wfi.calc_datacube(wavelengths, fov_arcsec=3.0,
                              outfile='grism1_psf_datacube.fits')

# Or save after calculation
datacube.writeto('grism1_psf_datacube.fits', overwrite=True)

# Load later
from astropy.io import fits
datacube_loaded = fits.open('grism1_psf_datacube.fits')

# Extract wavelength array from header
# (Wavelengths stored in header keywords WAVE0, WAVE1, etc.)
n_wave = datacube_loaded['DET_DIST'].header['NAXIS3']
wavelengths_loaded = np.array([
    datacube_loaded['DET_DIST'].header[f'WAVE{i}']
    for i in range(n_wave)
])
```

### Known Limitations for Roman WFI

**⚠️ Important Caveats:**

1. **Not explicitly documented for Roman:** These methods are inherited from parent class, primarily documented for JWST
2. **Testing recommended:** Verify output format and accuracy for Roman WFI grism before production use
3. **Coordinate system:** Same 0-indexed, 4096×4096 issues as `calc_psf()` (see Section 18)
4. **Memory usage:** Large datacubes (many wavelengths, large FOV, high oversample) can consume significant RAM
   - Example: 50 wavelengths × 256×256 pixels × 4 extensions × 4 bytes ≈ 500 MB
5. **No wavelength interpolation:** PSFs calculated only at specified wavelengths; interpolate manually if needed

### TODO: Verify for Roman WFI

- [ ] **TODO:** Confirm `calc_datacube()` works with Roman WFI (test with GRISM1)
- [ ] **TODO:** Verify output datacube format matches expectations
- [ ] **TODO:** Check if wavelengths stored in FITS header (WAVE* keywords)
- [ ] **TODO:** Benchmark performance vs loop of `calc_psf()` calls
- [ ] **TODO:** Test `calc_datacube_fast()` accuracy for Roman grism wavelength range
- [ ] **TODO:** Determine if fast method assumption (wavelength-independent OPD) holds for Roman

### Alternative: Loop Over calc_psf()

If `calc_datacube` has issues with Roman WFI, use a simple loop:

```python
import numpy as np

wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)
psf_list = []

for wl in wavelengths:
    psf = wfi.calc_psf(monochromatic=wl, fov_arcsec=3.0, oversample=4)
    psf_list.append(psf['DET_DIST'].data)

# Stack into 3D array
psf_datacube = np.stack(psf_list, axis=0)  # Shape: (15, N_pix, N_pix)
```

This is equivalent to `calc_datacube()` but more explicit and guaranteed to work.

---

---

## 8. Output FITS Structure

All `calc_psf()` calls return an `astropy.io.fits.HDUList` with four standard extensions:

| Extension | Name | Content | Sampling | Effects | Typical Use |
|-----------|------|---------|----------|---------|-------------|
| **0** | OVERSAMP | Oversampled optical model | 4× (or `oversample` param) | None | High-fidelity PSF analysis |
| **1** | DET_SAMP | Detector-sampled optical model | 1× (detector pixels) | Pixelation only | Quick preview |
| **2** | OVERDIST | Oversampled with detector effects | 4× (or `oversample` param) | Full (distortion, diffusion) | Detailed PSF modeling |
| **3** | DET_DIST | Detector-sampled with effects | 1× (detector pixels) | Full (distortion, diffusion) | **Most realistic for observations** |

**Detector Effects Include:**
- Pixel sampling (binning to detector pixel grid)
- Charge diffusion (interpixel capacitance, IPC)
- Geometric distortion

**Accessing Data:**

```python
# By extension name
psf_oversamp = psf['OVERSAMP'].data     # numpy array, oversampled ideal PSF
psf_det_dist = psf['DET_DIST'].data     # numpy array, detector-sampled + effects

# By extension number
psf_det_dist = psf[3].data

# Get header information
header = psf['DET_DIST'].header
pixelscale = header['PIXELSCL']  # arcsec/pixel
```

**Recommended Extension:**

- **For observation comparison:** Use Extension 3 (`DET_DIST`) - includes realistic detector sampling and effects
- **For optical analysis:** Use Extension 0 (`OVERSAMP`) - ideal oversampled PSF
- **For quick preview:** Use Extension 1 (`DET_SAMP`) - detector-sampled, no effects

---

## 9. Field-Dependent Aberrations

Roman WFI PSFs vary across the focal plane due to optical aberrations.

**Implementation:**

STPSF uses Zernike polynomial coefficients (Z₁ through Z₄₅) that are:
- Interpolated by detector position (X, Y within detector)
- Interpolated by wavelength
- Based on optical modeling by GSFC (Cycle 10, Sept 2024)

**Controlling Field Position:**

```python
wfi = stpsf.roman.WFI()

# Select detector (18 detectors in focal plane)
wfi.detector = 'WFI05'  # Central detector
wfi.detector = 'WFI01'  # Corner detector

# Set position within detector (X, Y in pixels, zero-indexed)
wfi.detector_position = (2048, 2048)  # Near center of 4096×4096 detector
wfi.detector_position = (1024, 3072)  # Off-center position

# Calculate PSF with field-dependent aberrations
psf = wfi.calc_psf(monochromatic=1.5e-6)
```

**Variation Across Field:**

PSF quality varies across the focal plane:
- Central detectors typically have better PSF quality
- Corner detectors may show more aberrations
- Variation is wavelength-dependent (chromatic aberrations)

**Detector Layout:**

The 18 WFI detectors form a grid covering ~0.281 square degrees. Each detector is 4096×4096 pixels at 0.11 arcsec/pixel.

**Applies to All Modes:**

Field-dependent aberrations affect both imaging and grism modes. For grism simulations, consider sampling multiple field positions if sources span a large area.

---

## 10. WFI Element Wheel Configuration

The WFI element wheel contains 11 positions (clockwise from top):

| Position | Element | Type | Wavelength Range (μm) | Notes |
|----------|---------|------|----------------------|-------|
| 1 | Grism (G150) | Disperser | 1.0 - 1.93 | STPSF: 'GRISM0', 'GRISM1' |
| 2 | F213 | Filter | 1.95 - 2.30 | Wide red filter |
| 3 | F062 | Filter | 0.48 - 0.76 | Blue filter |
| 4 | F106 | Filter | 0.93 - 1.19 | NIR filter |
| 5 | F129 | Filter | 1.13 - 1.45 | NIR filter |
| 6 | Prism | Disperser | 0.75 - 1.80 | Low-res spectroscopy |
| 7 | F158 | Filter | 1.38 - 1.77 | NIR filter |
| 8 | F184 | Filter | 1.68 - 2.00 | NIR filter |
| 9 | F146 | Filter | 1.28 - 1.63 | NIR filter |
| 10 | F087 | Filter | 0.76 - 0.98 | NIR filter |
| 11 | Dark | Calibration | N/A | Not available for science observations |

**Note:** The "Dark" position is used for calibration only and cannot be selected for science observations.

**Filter Selection in STPSF:**
```python
wfi.filter = 'GRISM0'  # Position 1, zeroth order
wfi.filter = 'GRISM1'  # Position 1, first order
wfi.filter = 'F213'    # Position 2
wfi.filter = 'PRISM'   # Position 6
# ... etc.
```

---

## 11. Logging and Diagnostics

Enable progress tracking and debugging during PSF calculations:

```python
import stpsf

# Show calculation progress (recommended)
stpsf.setup_logging('info')

# Detailed debugging information
stpsf.setup_logging('DEBUG')

# Only show errors
stpsf.setup_logging('ERROR')

# Disable logging
stpsf.setup_logging(None)
```

**Useful for:**
- Monitoring long calculations (high `nlambda`, large `fov_pixels`)
- Debugging wavelength clipping warnings
- Understanding what STPSF is doing internally

**Example Output (info level):**
```
Calculating PSF for WFI05 with filter GRISM1
Wavelength: 1.500 um
Pupil: GRISM
OPD: Loaded from cache
Calculating oversampled PSF (oversample=4)...
Applying detector effects...
Done.
```

**Wavelength Clipping Warnings:**

You may see warnings like:
```
WARNING: Requested wavelength 2.1 um is outside the range of aberration data.
```

This occurs when requesting PSFs at wavelengths outside the reference data range. STPSF will extrapolate, but results may be less accurate.

---

## 12. Working with Output

### Accessing PSF Data

```python
# Calculate PSF
psf = wfi.calc_psf(monochromatic=1.5e-6)

# Get numpy array from specific extension
psf_array = psf['DET_DIST'].data  # Most realistic
psf_oversamp = psf['OVERSAMP'].data  # Oversampled ideal

# Get extension by number
psf_array = psf[3].data

# Get header information
header = psf['DET_DIST'].header
pixelscale = header['PIXELSCL']  # arcsec/pixel
wavelength = header['WAVELEN']   # meters
filter_name = header['FILTER']   # e.g., 'GRISM1'
```

### Display with Matplotlib

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Display with logarithmic scaling (recommended for PSFs)
plt.figure(figsize=(8, 8))
plt.imshow(psf['DET_DIST'].data, norm=LogNorm(), origin='lower')
plt.colorbar(label='Intensity')
plt.title(f'{wfi.filter} PSF at {wavelength*1e6:.2f} μm')
plt.xlabel('Detector X (pixels)')
plt.ylabel('Detector Y (pixels)')
plt.show()
```

### Using STPSF Display Function

```python
import stpsf

# Built-in display function
stpsf.display_psf(psf, ext='DET_DIST')  # Display specific extension
stpsf.display_psf(psf, ext=3)           # Same, by number
stpsf.display_psf(psf)                  # Display all extensions in grid
```

### Writing to FITS File

```python
# Save PSF to file
psf.writeto('grism_psf_1.5um.fits', overwrite=True)

# Read back later
from astropy.io import fits
psf_reloaded = fits.open('grism_psf_1.5um.fits')
```

### Extracting Metrics

```python
import numpy as np

# Get detector-sampled PSF
psf_array = psf['DET_DIST'].data

# Total flux
total_flux = np.sum(psf_array)

# Peak pixel value
peak = np.max(psf_array)

# Centroid
y, x = np.indices(psf_array.shape)
centroid_x = np.sum(x * psf_array) / total_flux
centroid_y = np.sum(y * psf_array) / total_flux

# Encircled energy in central NxN box
N = 5  # 5x5 pixel box
center = np.array(psf_array.shape) // 2
box = psf_array[center[0]-N//2:center[0]+N//2+1,
                center[1]-N//2:center[1]+N//2+1]
encircled_energy = np.sum(box) / total_flux

print(f"Centroid: ({centroid_x:.2f}, {centroid_y:.2f})")
print(f"Encircled energy in {N}×{N} box: {encircled_energy:.1%}")
```

---

## 13. Performance Tips

**Calculation Speed Optimizations:**

1. **Reduce oversample factor:**
   ```python
   psf = wfi.calc_psf(monochromatic=1.5e-6, oversample=2)  # Faster than 4
   ```
   - Trades PSF core sampling for speed
   - `oversample=2` often sufficient for preliminary work

2. **Reduce field of view:**
   ```python
   psf = wfi.calc_psf(monochromatic=1.5e-6, fov_arcsec=3.0)  # Smaller than 5.0
   ```
   - Smaller arrays compute faster
   - 3″ often sufficient for compact PSFs

3. **Skip detector effects:**
   ```python
   psf = wfi.calc_psf(monochromatic=1.5e-6, add_distortion=False)
   ```
   - Only calculates Extensions 0 and 1 (optical model only)
   - Faster for optical studies where detector effects not needed

4. **Use monochromatic over polychromatic:**
   ```python
   # Fast
   psf = wfi.calc_psf(monochromatic=1.5e-6)

   # Slower (10× slower for nlambda=10)
   psf = wfi.calc_psf(nlambda=10)
   ```
   - Polychromatic PSFs calculate multiple wavelengths and integrate
   - **For grism: monochromatic is recommended**

5. **Adjust nlambda for polychromatic:**
   ```python
   psf = wfi.calc_psf(nlambda=5)  # Faster than default 10
   ```
   - Balance between spectral fidelity and speed
   - Higher `nlambda` = better accuracy for broadband filters

**Memory Considerations:**

- Each PSF with `fov_arcsec=5.0`, `oversample=4` uses ~10-20 MB
- For batch calculations (many wavelengths), consider calculating on-the-fly rather than storing all PSFs
- Oversampled extensions (0 and 2) are larger than detector-sampled (1 and 3)

**Typical Timing (rough estimates, CPU-dependent):**

| Configuration | Time per PSF |
|---------------|--------------|
| Monochromatic, `oversample=2`, `fov_arcsec=3`, no distortion | ~0.5s |
| Monochromatic, `oversample=4`, `fov_arcsec=5`, with distortion | ~1-2s |
| Polychromatic, `nlambda=10`, `oversample=4`, `fov_arcsec=5` | ~10-20s |

**Caching:**

STPSF caches OPD maps and reference data. First calculation loads data from disk; subsequent calculations reuse cached data.

---

## 14. Code Examples

### Example 1: Grism First Order PSF (Single Wavelength)

```python
import stpsf
import stpsf.roman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Setup logging to monitor calculation
stpsf.setup_logging('info')

# Create WFI instance
wfi = stpsf.roman.WFI()

# Configure for grism first order
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'  # Central detector
wfi.detector_position = (2048, 2048)  # Near center (zero-indexed: 0-4095)

# Calculate monochromatic PSF at center of grism bandpass
psf = wfi.calc_psf(
    monochromatic=1.465e-6,  # Center wavelength (1.465 μm)
    fov_arcsec=5.0,
    oversample=4
)

# Display the detector-sampled PSF with detector effects
plt.figure(figsize=(10, 8))
plt.imshow(psf['DET_DIST'].data, norm=LogNorm(), origin='lower')
plt.colorbar(label='Intensity')
plt.title('GRISM1 PSF at 1.465 μm (WFI05 center)')
plt.xlabel('Detector X (pixels)')
plt.ylabel('Detector Y (pixels)')
plt.show()

# Print some info
print(f"PSF shape: {psf['DET_DIST'].data.shape}")
print(f"Total flux: {psf['DET_DIST'].data.sum():.3f}")
print(f"Peak value: {psf['DET_DIST'].data.max():.6f}")
```

### Example 2: Grism PSF Across Wavelength Range (Using calc_datacube)

```python
import numpy as np
import stpsf
import stpsf.roman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Sample grism wavelength range (1.0 - 1.93 μm)
wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)  # 15 wavelengths in meters

# Calculate PSF datacube (single function call)
print("Calculating PSF datacube...")
datacube = wfi.calc_datacube(
    wavelengths,
    fov_arcsec=3.0,
    oversample=4,
    add_distortion=True
)

# Extract detector-sampled PSFs with effects (most realistic)
psf_cube = datacube['DET_DIST'].data  # Shape: (15, N_pix, N_pix)
print(f"Datacube shape: {psf_cube.shape}")

# Plot a grid of PSFs
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for i in range(15):
    axes[i].imshow(psf_cube[i, :, :], norm=LogNorm(), origin='lower')
    axes[i].set_title(f'{wavelengths[i]*1e6:.2f} μm')
    axes[i].axis('off')

plt.suptitle('GRISM1 PSFs Across Wavelength Range (calc_datacube)', fontsize=14)
plt.tight_layout()
plt.show()

# Use with disperser model
# psf_cube[i, :, :] corresponds to wavelengths[i]
```

### Example 2b: Grism PSF Across Wavelength Range (Manual Loop)

```python
import numpy as np
import stpsf.roman

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Sample wavelength range
wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)

# Calculate PSF at each wavelength (alternative to calc_datacube)
psfs = []
for i, wl in enumerate(wavelengths):
    print(f"Calculating PSF {i+1}/15 at {wl*1e6:.3f} μm...")
    psf = wfi.calc_psf(
        monochromatic=wl,
        fov_arcsec=3.0,
        oversample=4
    )
    # Extract detector-sampled PSF with effects
    psf_array = psf['DET_DIST'].data
    psfs.append(psf_array)

# Stack into 3D array
psf_cube = np.stack(psfs, axis=0)  # Shape: (15, N_pix, N_pix)

# This is equivalent to calc_datacube but more explicit
```

### Example 3: Comparing Zeroth and First Order

```python
import stpsf.roman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

wfi = stpsf.roman.WFI()
wfi.detector = 'WFI05'
wavelength = 1.5e-6  # 1.5 μm

# Zeroth order (undispersed)
wfi.filter = 'GRISM0'
psf_0th = wfi.calc_psf(monochromatic=wavelength, fov_arcsec=5.0, oversample=4)

# First order (dispersed)
wfi.filter = 'GRISM1'
psf_1st = wfi.calc_psf(monochromatic=wavelength, fov_arcsec=5.0, oversample=4)

# Compare the two
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(psf_0th['DET_DIST'].data, norm=LogNorm(), origin='lower')
axes[0].set_title('GRISM0 (Zeroth Order) at 1.5 μm')
axes[0].set_xlabel('Detector X (pixels)')
axes[0].set_ylabel('Detector Y (pixels)')

axes[1].imshow(psf_1st['DET_DIST'].data, norm=LogNorm(), origin='lower')
axes[1].set_title('GRISM1 (First Order) at 1.5 μm')
axes[1].set_xlabel('Detector X (pixels)')
axes[1].set_ylabel('Detector Y (pixels)')

plt.tight_layout()
plt.show()

# Note: PSF shapes may differ due to optical path differences
```

### Example 4: Imaging Filter PSF (for Comparison)

```python
import stpsf.roman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# For comparison with grism mode
wfi = stpsf.roman.WFI()
wfi.filter = 'F158'  # Imaging filter, similar wavelength range to grism
wfi.detector = 'WFI05'
wfi.detector_position = (2048, 2048)  # Near center (zero-indexed: 0-4095)

# Polychromatic PSF (integrates over filter bandpass)
psf = wfi.calc_psf(nlambda=10, fov_arcsec=5.0, oversample=4)

# Display
plt.figure(figsize=(10, 8))
plt.imshow(psf['DET_DIST'].data, norm=LogNorm(), origin='lower')
plt.colorbar(label='Intensity')
plt.title('F158 Imaging PSF (Polychromatic)')
plt.xlabel('Detector X (pixels)')
plt.ylabel('Detector Y (pixels)')
plt.show()
```

### Example 5: Field-Dependent PSF Variation (Grism Mode)

```python
import stpsf.roman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

wavelength = 1.5e-6  # 1.5 μm

# PSF at center detector
wfi_center = stpsf.roman.WFI()
wfi_center.filter = 'GRISM1'
wfi_center.detector = 'WFI05'  # Central detector
wfi_center.detector_position = (2048, 2048)
psf_center = wfi_center.calc_psf(monochromatic=wavelength, fov_arcsec=5.0)

# PSF at corner detector
wfi_corner = stpsf.roman.WFI()
wfi_corner.filter = 'GRISM1'
wfi_corner.detector = 'WFI01'  # Corner detector
wfi_corner.detector_position = (2048, 2048)
psf_corner = wfi_corner.calc_psf(monochromatic=wavelength, fov_arcsec=5.0)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(psf_center['DET_DIST'].data, norm=LogNorm(), origin='lower')
axes[0].set_title('GRISM1 PSF - WFI05 (Center)')

axes[1].imshow(psf_corner['DET_DIST'].data, norm=LogNorm(), origin='lower')
axes[1].set_title('GRISM1 PSF - WFI01 (Corner)')

plt.suptitle('Field-Dependent PSF Variation at 1.5 μm', fontsize=14)
plt.tight_layout()
plt.show()
```

### Example 6: Batch Processing for Disperser Integration

```python
import numpy as np
import stpsf.roman

def calculate_grism_psf_grid(wavelengths, fov_arcsec=3.0, oversample=4,
                             detector='WFI05'):
    """
    Calculate PSF grid for grism disperser integration.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths in meters
    fov_arcsec : float
        Field of view in arcseconds
    oversample : int
        Oversampling factor
    detector : str
        Detector name (e.g., 'WFI05')

    Returns
    -------
    psf_grid : list of ndarray
        List of PSF arrays (DET_DIST extension)
    wavelengths : ndarray
        Wavelength array (for reference)
    """
    wfi = stpsf.roman.WFI()
    wfi.filter = 'GRISM1'
    wfi.detector = detector

    psf_grid = []
    for wl in wavelengths:
        psf = wfi.calc_psf(
            monochromatic=wl,
            fov_arcsec=fov_arcsec,
            oversample=oversample
        )
        psf_grid.append(psf['DET_DIST'].data)

    return psf_grid, np.array(wavelengths)

# Usage
wavelengths = np.linspace(1.0e-6, 1.93e-6, 20)  # 20 wavelengths
psf_grid, wl_array = calculate_grism_psf_grid(wavelengths)

print(f"Created PSF grid: {len(psf_grid)} wavelengths")
print(f"Each PSF shape: {psf_grid[0].shape}")
print(f"Total memory: ~{sum(p.nbytes for p in psf_grid) / 1e6:.1f} MB")

# Now use psf_grid with your disperser model
```

### Example 7: Fast Datacube for High Wavelength Sampling

```python
import numpy as np
import stpsf.roman
import matplotlib.pyplot as plt

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# High wavelength sampling (50 wavelengths)
wavelengths = np.linspace(1.0e-6, 1.93e-6, 50)

# Fast calculation (no detector effects, ~150x faster)
print("Calculating fast datacube...")
datacube_fast = wfi.calc_datacube_fast(
    wavelengths,
    fov_arcsec=3.0,
    oversample=4
)

# Extract oversampled PSF cube (only extension available)
psf_cube = datacube_fast['OVERSAMP'].data  # Shape: (50, N_pix*4, N_pix*4)
print(f"Fast datacube shape: {psf_cube.shape}")

# Analyze PSF FWHM evolution with wavelength
fwhm_values = []
for i in range(len(wavelengths)):
    psf_2d = psf_cube[i, :, :]
    # Calculate FWHM (simplified - just measure peak width)
    peak_row = psf_2d[psf_2d.shape[0]//2, :]
    half_max = peak_row.max() / 2
    fwhm = np.sum(peak_row > half_max)  # pixels at half maximum
    fwhm_values.append(fwhm)

# Plot FWHM vs wavelength
plt.figure(figsize=(10, 6))
plt.plot(wavelengths * 1e6, fwhm_values, 'o-')
plt.xlabel('Wavelength (μm)')
plt.ylabel('FWHM (oversampled pixels)')
plt.title('PSF FWHM Evolution Across Grism Range')
plt.grid(True)
plt.show()

# This fast method is ideal for PSF analysis across many wavelengths
```

### Example 8: Saving and Loading PSF Library

```python
import numpy as np
import stpsf.roman
from astropy.io import fits

# Create library of PSFs at standard wavelengths
wavelengths = np.linspace(1.0e-6, 1.93e-6, 10)

wfi = stpsf.roman.WFI()
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Save each PSF to file
for i, wl in enumerate(wavelengths):
    psf = wfi.calc_psf(monochromatic=wl, fov_arcsec=3.0, oversample=4)
    filename = f'psf_grism1_wfi05_{wl*1e6:.3f}um.fits'
    psf.writeto(filename, overwrite=True)
    print(f"Saved {filename}")

# Later: Load PSFs on demand
def load_grism_psf(wavelength_um, detector='WFI05'):
    """Load pre-calculated PSF from library."""
    filename = f'psf_grism1_{detector.lower()}_{wavelength_um:.3f}um.fits'
    psf = fits.open(filename)
    return psf['DET_DIST'].data

# Usage
psf_array = load_grism_psf(1.500)  # Load PSF at 1.5 μm
```

---

## 15. Known Limitations

### Grism-Specific Limitations

1. **STPSF does NOT model grism dispersion**
   - Provides PSF shape at each wavelength
   - Does NOT provide spectral spreading or trace geometry
   - Requires separate optical/disperser model for spectral traces

2. **No trace information provided**
   - Dispersion direction not specified
   - Dispersion amount (Å/pixel) not provided
   - Trace curvature not modeled
   - **Use:** Separate tools like `roman_disperser` for trace geometry

3. **Sparse grism documentation**
   - Grism-specific parameters not extensively documented
   - Limited examples compared to imaging mode
   - Most tutorial examples focus on imaging filters

4. **Monochromatic calculations recommended**
   - Polychromatic grism PSFs not demonstrated in STPSF examples
   - Standard practice: calculate monochromatic PSFs across wavelength range
   - Uncertain behavior for polychromatic grism mode

5. **No spectral extraction tools**
   - STPSF provides PSFs only
   - Spectral extraction from 2D grism images requires separate tools
   - Wavelength calibration not included

### General Limitations

1. **Reference data required**
   - Large downloads (~GB) on first use
   - Stored in `$HOME/stpsf-data/` by default
   - Requires internet connection for initial setup

2. **Wavelength range limitations**
   - Aberration data covers specific wavelength ranges
   - Extrapolation warnings when requesting PSFs outside range
   - Results may be less accurate at extreme wavelengths

3. **Computation time**
   - High-fidelity PSFs (large `fov_pixels`, high `oversample`, high `nlambda`) can be slow
   - No GPU acceleration
   - CPU-bound calculations

4. **Memory usage**
   - Oversampled PSFs can be large
   - Batch calculations (many wavelengths) require careful memory management
   - Consider calculating on-the-fly vs. storing all PSFs

5. **Incomplete detector effects**
   - Models geometric distortion and charge diffusion
   - Does NOT model: cosmic rays, saturation, read noise, dark current, flat field variations
   - For full detector simulation, combine with separate detector simulators

---

## 16. References and Resources

### STPSF Documentation and Code

- **Official Documentation:** https://stpsf.readthedocs.io
- **GitHub Repository:** https://github.com/spacetelescope/stpsf
- **Roman Tutorial Notebook:** https://github.com/spacetelescope/stpsf/blob/develop/notebooks/STPSF-Roman_Tutorial.ipynb
- **API Reference:** https://stpsf.readthedocs.io/en/latest/api.html

**Version Information:**
- STPSF Version: 2.2.0 (released Dec 23, 2025)
- Roman WFI Optical Model: Cycle 10 (Sept 2024, NASA GSFC)
- Former Name: WebbPSF (renamed to STPSF in v2.0 to reflect multi-mission support)

### Roman WFI Documentation

- **WFI Technical Details:** https://roman.gsfc.nasa.gov/science/WFI_technical.html
- **WFI Optical Elements:** https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/wfi-design/wfi-optical-elements
- **Roman Documentation Hub (STScI):** https://roman-docs.stsci.edu/
- **Grism/Prism Spectroscopy:** https://roman-docs.stsci.edu/roman-instruments-home/wfi-spectroscopy-mode-user-guide

### Technical Papers and Reports

- **Nancy Grace Roman Space Telescope Grism and Prism: Optical Design** (NASA/GSFC)
- **Roman Space Telescope Mission Overview:** Various NASA/GSFC publications
- **WFI Element Wheel Configuration:** From Roman technical documentation

### Related Tools

- **roman_disperser (this project):** JAX-based disperser for grism spectral traces
- **GalSim:** Galaxy image simulation (includes Roman WFI module)
- **synphot:** Synthetic photometry for custom spectra
- **Pandeia:** Exposure time calculator (includes Roman)

### Community Resources

- **STScI Help Desk:** https://stsci.service-now.com/roman
- **Roman User Forums:** Community discussions and Q&A
- **GitHub Issues:** Bug reports and feature requests for STPSF

---

## 17. Quick Reference Summary

### ⚠️ CRITICAL: Coordinate System Mismatch

**STPSF vs roman_disperser:**
- STPSF: 0-indexed, 4096×4096 pixels (0-4095)
- Disperser: 1-indexed FITS, 4088×4088 pixels (1-4088)
- **Action required:** Coordinate conversion before integration (see Section 18)

---

### Essential Commands

```python
import stpsf
import stpsf.roman
import numpy as np

# Create WFI instance
wfi = stpsf.roman.WFI()

# Configure for grism first order
wfi.filter = 'GRISM1'
wfi.detector = 'WFI05'

# Method 1: Single wavelength PSF
psf = wfi.calc_psf(monochromatic=1.5e-6, fov_arcsec=5.0, oversample=4)
psf_array = psf['DET_DIST'].data  # 2D array

# Method 2: Multiple wavelengths with calc_datacube (recommended for grism)
wavelengths = np.linspace(1.0e-6, 1.93e-6, 15)
datacube = wfi.calc_datacube(wavelengths, fov_arcsec=3.0, oversample=4)
psf_cube = datacube['DET_DIST'].data  # 3D array [wavelength, y, x]

# Method 3: Fast datacube (no detector effects, ~150x faster)
datacube_fast = wfi.calc_datacube_fast(wavelengths, fov_arcsec=3.0)
psf_cube_oversamp = datacube_fast['OVERSAMP'].data  # 3D oversampled only

# Enable logging
stpsf.setup_logging('info')
```

### Grism Wavelength Range

- **Minimum:** 1.0 μm
- **Maximum:** 1.93 μm
- **Center:** 1.465 μm
- **Typical sampling:** 10-20 wavelengths across range

### Key Distinctions

- **GRISM0:** Zeroth order (undispersed) - use for source identification
- **GRISM1:** First order (dispersed spectrum) - primary science mode
- **Both share:** Same `'GRISM'` pupil mask, same G150 physical element

### What STPSF Provides for Grism

✅ PSF shape at each wavelength
✅ Field-dependent aberrations
✅ Detector effects (pixel sampling, distortion, diffusion)
✅ Wavelength-dependent chromatic effects

### What STPSF Does NOT Provide for Grism

❌ Dispersion direction or orientation
❌ Spectral trace geometry
❌ Dispersion amount (Å/pixel or mm/μm)
❌ Spectral extraction tools
❌ Wavelength calibration

**Solution:** Use `roman_disperser` (this project) or similar optical models for spectral traces.

---

## 18. Integration with roman_disperser: Coordinate Systems

### 18.1 The Problem

STPSF and the `roman_disperser` optical model use **different coordinate conventions**, creating integration challenges:

| Aspect | STPSF | roman_disperser / Optical Model |
|--------|-------|--------------------------------|
| **Indexing Convention** | 0-indexed (Python/NumPy) | 1-indexed (FITS standard) |
| **Detector Array Size** | 4096×4096 pixels (full H4RG-10 array) | 4088×4088 pixels (usable science region) |
| **Valid Pixel Range** | 0 to 4095 | 1 to 4088 (FITS coordinates) |
| **Pixel Center** | Pixel N at coordinate N (0-indexed) | Pixel N at coordinate N.0 (1-indexed FITS) |
| **Reference Frame** | Python array indices | FITS world coordinates |

### 18.2 Why This Matters

When integrating STPSF PSFs with dispersed spectra from the optical model:

1. **Coordinate transformation required:** You cannot directly use optical model coordinates with STPSF
2. **Out-of-bounds issue:** STPSF can calculate PSFs at positions outside the 4088×4088 usable region
3. **Pixel registration:** Improper conversion leads to PSFs misaligned with spectral traces by ~4-8 pixels
4. **Edge cases:** Spectra near detector edges may have PSFs that extend beyond usable region

### 18.3 FITS Convention (Optical Model)

The optical model follows FITS standards:
- **1-indexed:** First pixel is pixel 1, last pixel is pixel 4088
- **Pixel centers:** Pixel N has center at coordinate N.0
- **Pixel boundaries:** Pixel N spans [N-0.5, N+0.5]
- **Valid range:** 1.0 to 4088.0 (some tools may use 0.5 to 4088.5 for edges)

**Example:**
```python
# Optical model trace output (FITS 1-indexed)
xsca_trace = 2500.5  # Center of pixel 2500.5 (midway between pixels 2500 and 2501)
ysca_trace = 1024.0  # Center of pixel 1024
```

### 18.4 STPSF Convention (0-indexed)

STPSF uses Python/NumPy conventions:
- **0-indexed:** First pixel is pixel 0, last pixel is pixel 4095
- **Full array:** Uses complete 4096×4096 H4RG-10 detector array
- **Pixel centers:** Pixel N at coordinate N (not N.0)

**Example:**
```python
# STPSF detector position (0-indexed)
wfi.detector_position = (2500, 1024)  # Pixel (2500, 1024) in 0-indexed array
```

### 18.5 Detector Size Discrepancy

**Key Question:** Where does the 4088×4088 usable region sit within the 4096×4096 full array?

**Possibilities:**
1. **Centered:** Pixels [4:4092] in 0-indexed (4-pixel border on each side)
2. **Offset:** Pixels [0:4088] (8-pixel border on one side) or [8:4096] (8-pixel border on other side)
3. **Other:** Some other configuration

**TODO: Investigate:**
- Check Roman technical documentation for reference pixel locations
- Compare STPSF pixel scale with optical model pixel scale
- Examine Roman FITS headers for CRPIX reference pixels
- Test with known source positions in both systems

### 18.6 Coordinate Conversion (Preliminary)

**⚠️ WARNING: This conversion is UNVERIFIED. TODOs must be completed first.**

```python
def sca_to_stpsf_position(xsca, ysca):
    """
    Convert optical model SCA coordinates to STPSF detector_position.

    WARNING: This is a preliminary conversion that needs verification!

    Parameters
    ----------
    xsca, ysca : float
        SCA coordinates from optical model (1-indexed FITS, range 1-4088)

    Returns
    -------
    x_stpsf, y_stpsf : int
        STPSF detector_position (0-indexed, range 0-4095)

    TODO:
    - Confirm 0-indexed assumption
    - Determine offset for 4088->4096 conversion
    - Verify with test cases
    """
    # ASSUMPTION: 4088 usable region is centered in 4096 array
    # This means 4-pixel border on each side: pixels [4:4092] in 0-indexed

    # Convert FITS 1-indexed to Python 0-indexed
    x_0indexed = xsca - 1.0
    y_0indexed = ysca - 1.0

    # Add offset for centered 4088 region within 4096 array
    # TODO: Verify this offset is correct!
    offset = 4  # Assumed border width
    x_stpsf = int(round(x_0indexed + offset))
    y_stpsf = int(round(y_0indexed + offset))

    # Validate
    if not (0 <= x_stpsf <= 4095 and 0 <= y_stpsf <= 4095):
        raise ValueError(f"Converted position ({x_stpsf}, {y_stpsf}) "
                        f"outside STPSF range [0, 4095]")

    return x_stpsf, y_stpsf


def stpsf_to_sca_position(x_stpsf, y_stpsf):
    """
    Convert STPSF detector_position to optical model SCA coordinates.

    WARNING: This is a preliminary conversion that needs verification!

    Parameters
    ----------
    x_stpsf, y_stpsf : int
        STPSF detector_position (0-indexed, range 0-4095)

    Returns
    -------
    xsca, ysca : float
        SCA coordinates for optical model (1-indexed FITS, range 1-4088)

    TODO:
    - Confirm conversion is inverse of sca_to_stpsf_position
    - Handle out-of-bounds cases (STPSF position outside 4088 usable region)
    """
    # Remove offset for centered 4088 region
    offset = 4  # TODO: Verify this offset!
    x_0indexed = x_stpsf - offset
    y_0indexed = y_stpsf - offset

    # Convert Python 0-indexed to FITS 1-indexed
    xsca = x_0indexed + 1.0
    ysca = y_0indexed + 1.0

    # Validate
    if not (1.0 <= xsca <= 4088.0 and 1.0 <= ysca <= 4088.0):
        raise ValueError(f"Converted SCA position ({xsca}, {ysca}) "
                        f"outside usable region [1, 4088]")

    return xsca, ysca
```

### 18.7 Outstanding TODOs

Before integrating STPSF PSFs with `roman_disperser`, complete these tasks:

1. **Confirm STPSF indexing:**
   - [x] Verified from source code: 0-indexed (valid range 0-4095)
   - [ ] Verify with actual PSF calculation at known positions
   - [ ] Cross-check with STPSF documentation or developers

2. **Confirm detector size:**
   - [ ] Verify STPSF uses full 4096×4096 array (not 4088)
   - [ ] Check if STPSF has any usable region restrictions
   - [ ] Look for STPSF documentation on detector geometry

3. **Determine coordinate offset:**
   - [ ] Find Roman technical docs on reference pixels (CRPIX in FITS headers)
   - [ ] Determine where 4088 usable region sits in 4096 array
   - [ ] Is it centered (4-pixel border)? Offset? Other?
   - [ ] Check Roman SCA layout diagrams

4. **Verify conversion:**
   - [ ] Test coordinate conversion with known source positions
   - [ ] Compare STPSF PSF position with optical model trace positions
   - [ ] Verify pixel alignment in test images
   - [ ] Check edge cases (near detector boundaries)

5. **Document final solution:**
   - [ ] Update conversion functions with verified offsets
   - [ ] Add unit tests for coordinate conversion
   - [ ] Document any restrictions or edge cases
   - [ ] Add examples showing correct integration

### 18.8 Practical Workaround (Until Resolved)

Until the coordinate conversion is verified, use STPSF for **relative** PSF studies only:

**Safe Use Cases:**
- Calculate PSFs at multiple wavelengths (relative shape changes)
- Study PSF variation across detectors (comparative analysis)
- Analyze PSF profiles for resolution estimates
- Generate PSF libraries indexed by wavelength

**Unsafe Use Cases (require coordinate conversion):**
- Direct integration: placing STPSF PSFs at optical model trace positions
- Pixel-level alignment of PSFs with dispersed spectra
- Creating realistic detector images with STPSF+disperser

**Recommended Approach:**
1. Calculate STPSF PSFs at wavelength grid points
2. Extract PSF shape/profile information
3. Use optical model for coordinate geometry (trace positions)
4. Apply PSF shape at trace positions (after coordinate verification)

### 18.9 Alternative: Native PSF Model

If coordinate conversion proves problematic, consider:
- Using optical model's native PSF representation (if available)
- Generating simple analytic PSFs (Gaussian, Airy disk) at trace positions
- Requesting STPSF team to add FITS-coordinate mode
- Creating a wrapper tool that handles both coordinate systems

### 18.10 References for Investigation

**Roman Technical Documentation:**
- Roman WFI Detector Layout: https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/wfi-design
- SCA Pixel Coordinate System: Check FITS header conventions in Roman documentation
- Reference Pixels: Look for CRPIX definitions in Roman FITS standards

**STPSF Resources:**
- GitHub Issues: Search for discussions on coordinate systems
- Developer Contact: May need to ask STPSF team directly
- Source Code: Examine detector geometry definitions in `stpsf/roman.py`

**Optical Model Resources:**
- `Roman_grism_OpticalModel_v0.8.yaml`: Check for detector size/offset parameters
- This project's `optical_model.py`: Compare coordinate handling with STPSF

### 18.11 Empirical PSF Characterization

The PSF analysis notebook (`notebooks/psf/psf_analysis.ipynb`) provides empirical measurements using STPSF:

**Key Findings:**

| Parameter | Value |
|-----------|-------|
| **Recommended FOV** | 5 arcsec (captures >95% EE at all wavelengths) |
| **EE50 radius** | 0.08" (1.0 μm) to 0.125" (1.93 μm) |
| **EE90 radius** | 0.6" (1.0 μm) to 1.0" (1.93 μm) |
| **EE95 radius** | 1.1" (1.0 μm) to 1.8" (1.93 μm) |
| **FWHM** | ~2 × EE50 radius |
| **Calc time** | ~0.4-0.5s per PSF |

**Observations:**
- Encircled energy curves are similar at detector center vs corners (note: PSF shapes may still vary)
- Corner positions (1,1), etc. trigger STPSF warnings about being outside reference data range
- Use full grid range (1 to 4088); STPSF handles edge extrapolation internally

See `docs/psf_phase1_plan.md` for full findings and implementation plan.

---

**End of Document**

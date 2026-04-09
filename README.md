# Roman Disperser

JAX-based optical model and disperser for Roman Space Telescope grism spectroscopy simulations.

## Overview

| Component | Module | Description |
|-----------|--------|-------------|
| **Optical Model** (reference) | `optical_model.py` | NumPy reference implementation |
| **Optical Model** (JAX) | `optical_model_jax.py` | JIT-compilable, vectorized implementation |
| **PSF Model** | `psf_model.py` | STPSF grids with trilinear interpolation |
| **Star Disperser** | `star_disperser.py` | Point sources with wavelength-dependent PSFs |
| **Galaxy Disperser** | `galaxy_disperser.py` | Extended sources with Jacobian warping + PSF convolution |
| **Catalog Pipeline** | `catalog.py` + `pipeline.py` + `scripts/build_grism_image.py` | Source selection, full-field grism simulation (stars + galaxies) |
| **Sérsic Profiles** | `sersic.py` | JAX/vmap Sérsic profile generator for galaxy morphologies |
| **Reference Data** | `refdata.py` | Bundled F158 bandpass and spectral templates |

## Installation

See [INSTALL.md](INSTALL.md) for full instructions (pixi and pip paths, GPU setup, data files).

Quick version:

```bash
git clone git@github.com:roman-grs-pit/roman_disperser.git
cd roman_disperser
pixi install && pixi shell        # or: pip install -e ".[full]"
python scripts/download_psf_caches.py
pytest -q tests -m "not slow"
```

## Quick Start

```python
import jax.numpy as jnp
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser import optical_model_jax as omj, psf_model, star_disperser

# Load optical model and create payloads
model = RomanOpticalModel("data/Roman_grism_OpticalModel_v0.8.yaml")
optical_payload = omj.make_sca_payload(model, sca=5, order="1")
psf_payload = psf_model.get_or_make_psf_payload(
    detector="WFI05", order="1", cache_dir="data/psf_cache"
)

# Create a JIT-compiled star disperser
disperse = star_disperser.make_star_disperser(psf_payload, optical_payload)

# Disperse a star (wavelengths in microns)
wavelengths = jnp.linspace(0.9, 2.0, 5500)
flux = jnp.ones_like(wavelengths)
output = jnp.zeros((4088, 4088), dtype=jnp.float32)
output = disperse(xsca_star=2000.0, ysca_star=2000.0,
                  wavelengths=wavelengths, star_flux=flux, output=output)
```

### Galaxy Dispersion

```python
from roman_disperser import galaxy_disperser

# Create a JIT-compiled galaxy disperser
disperse = galaxy_disperser.make_galaxy_disperser(psf_payload, optical_payload)

# Galaxy image at 4× oversampling (must match PSF payload)
galaxy_image = jnp.ones((120, 120), dtype=jnp.float32)  # 30×30 native × 4

# Disperse a galaxy (wavelengths in microns)
output = disperse(image=galaxy_image, x0=2000.0, y0=2000.0,
                  spectrum=flux, wavelengths=wavelengths, output=output)
```

### GPU Performance

Typical GPU throughput (RTX 4090): ~3 ms/star/order, ~7 ms/galaxy/order.
10K sources × 3 orders takes ~5 minutes (see `notebooks/galaxy/profile_dispersers.ipynb`).

## Running Tests

```bash
pytest -q tests                    # All tests
pytest -m "not slow" tests         # Skip slow tests (STPSF generation)
pytest -v tests/test_star_disperser.py      # Star disperser tests
pytest -v tests/test_galaxy_disperser.py    # Galaxy disperser tests
pytest -v tests/test_psf_model.py           # PSF model tests
```

### Test Coverage

- **Optical model** (`test_optical_model_jax.py`): SCA/FPA/MPA coordinate transformations, sky-to-FPA transforms, polynomial mappings, trace coefficients, spectral traces
- **Disperser** (`test_disperser.py`): Bilinear interpolation, flux conservation, boundary handling, multi-galaxy batching
- **PSF model** (`test_psf_model.py`): PSF interpolation, caching, trilinear accuracy
- **Star disperser** (`test_star_disperser.py`): PSF deposition, chunk invariance, flux conservation
- **Galaxy disperser** (`test_galaxy_disperser.py`): Jacobian warping, PSF convolution, delta-vs-star comparison
- **Sérsic profiles** (`test_sersic.py`): b_n accuracy, astropy comparison, normalization, PA transformation
- **GPU consistency** (`test_disperser_gpu.py`): CPU vs GPU verification (skipped if no GPU)
- **Demo utils** (`test_demo_utils.py`): Synthetic data generation helpers

## Documentation

- [Installation Guide](INSTALL.md) — Pixi/pip setup, GPU support, data files
- [Optical Model API](docs/optical_model.md) — JAX optical model functions and examples
- [Grism Pipeline](docs/grism_pipeline.md) — User guide for `build_grism_image.py` (stars + galaxies)
- [Star Grism Pipeline](docs/star_grism_pipeline.md) — Legacy star-only pipeline (deprecated)
- [Disperser Design](docs/disperser_design.md) — Legacy disperser implementation details
- [JIT Compilation Strategy](docs/jit_compilation.md) — Closure pattern for non-traceable payloads
- [STPSF Quick Reference](docs/stpsf.md) — Roman WFI grism PSF generation
- [Star Dispersion](docs/star_dispersion.md) — Star dispersion design phases and PSF interpolation
- [PSF Phase 1 Plan](docs/psf_phase1_plan.md) — PSF data model implementation and validation results
- [Galaxy Dispersion Plan](docs/galaxy_dispersion_plan.md) — Galaxy disperser design (Jacobian-based)
- [Catalog Format](data/catalogs/README.md) — Unified source catalog format specification

## Notebooks

**Demo notebooks** (`notebooks/galaxy/`):
- `stars_and_galaxies_demo.ipynb` — Full field: 100 stars + 100 galaxies × 3 orders (CPU)
- `stars_and_galaxies_gpu_demo.ipynb` — Same demo optimized for GPU
- `profile_dispersers.ipynb` — Performance profiling: per-operation timing breakdown
- `jacobian_exploration.ipynb` — Jacobian characterization for galaxy disperser design
- `sersic_profiles.ipynb` — Sérsic profile validation: astropy comparison, PA transforms, radial profiles

**PSF and star notebooks** (`notebooks/psf/`):
- `single_star_demo.ipynb` — Single star dispersal with wavelength-dependent PSF
- `multi_star_demo.ipynb` — Multiple stars on one detector
- `multi_star_demo_gpu_run.ipynb` — GPU performance benchmarks
- `psf_analysis.ipynb` — PSF characterization and enclosed energy
- `psf_interpolation_validation.ipynb` — Grid optimization and interpolation accuracy
- `psf_allsca_validation.ipynb` — All 18 SCAs × 2 orders validation

## Project Structure

```
roman_disperser/
├── src/roman_disperser/
│   ├── optical_model.py           # Reference NumPy model
│   ├── optical_model_jax.py       # JAX functional model
│   ├── optical_model_utils.py     # Coordinate system utilities
│   ├── disperser.py               # Legacy 2D+1D galaxy disperser
│   ├── star_disperser.py          # Star disperser with PSF deposition
│   ├── galaxy_disperser.py        # Galaxy disperser (Jacobian warp + PSF convolution)
│   ├── psf_model.py               # PSF grids and trilinear interpolation
│   ├── psf_utils.py               # STPSF ↔ disperser coordinate conversion
│   ├── sersic.py                  # Sérsic profile generator (JAX/vmap)
│   ├── catalog.py                 # Source selection for detector fields
│   ├── pipeline.py                # Shared pipeline utilities (I/O, batching, sensitivity)
│   ├── refdata.py                 # Bundled synphot reference data
│   └── demo_utils.py              # Synthetic galaxy/spectrum helpers
├── tests/
├── notebooks/
│   ├── galaxy/                    # Galaxy + star demo notebooks
│   ├── psf/                       # PSF and star notebooks
│   ├── demos/                     # Legacy galaxy demo notebooks
│   └── archive/                   # Legacy notebooks
├── scripts/
│   ├── build_grism_image.py       # Unified grism pipeline (stars + galaxies)
│   ├── build_star_grism_image.py  # Legacy star-only pipeline (deprecated)
│   ├── download_psf_caches.py     # Download pre-generated PSF caches
│   ├── download_source_catalog.py # Download unified source catalog
│   ├── generate_psf_caches.py     # Regenerate PSF caches from STPSF
│   ├── example_grism_config.yaml  # Documented batch pipeline config
│   └── example_star_config.yaml   # Legacy star-only config
├── data/
│   ├── Roman_grism_OpticalModel_v0.8.yaml
│   ├── catalogs/                  # Unified source catalog (Parquet + Zarr)
│   ├── sensitivities/             # Per-SCA sensitivity curves
│   ├── stars/                     # Legacy star catalog and SED templates
│   ├── synphot/                   # Bundled F158/F184 bandpass and templates
│   └── psf_cache/                 # Pre-generated PSF grids (36 files, ~4.3 GB)
├── docs/
├── pixi.toml
└── pyproject.toml
```

## References

- Roman Space Telescope: https://roman.gsfc.nasa.gov/
- STPSF (Space Telescope PSF Simulator): https://stpsf.readthedocs.io
- JAX: https://jax.readthedocs.io/
- Pixi: https://pixi.sh

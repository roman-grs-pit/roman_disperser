# Roman Disperser

JAX-based optical model and disperser for Roman Space Telescope slitless
spectroscopy simulations — both WFI dispersing elements: the G150 **grism**
(default) and the P127 **prism** (opt-in via `--element prism`).

## News

- **2026-08-05 — v0.14.0: prism support.** The package now simulates both
  dispersing elements; grism behaviour is unchanged. Coming from v0.10.x?
  Two intervening releases also **changed simulated results** (GPU placement
  and sky-projection fixes) — read
  [Migrating to v0.14 (from v0.10)](docs/migrating-v0.10-to-v0.14.md)
  before comparing old and new products.

## Overview

| Component | Module | Description |
|-----------|--------|-------------|
| **Optical Model** (reference) | `optical_model.py` | NumPy reference implementation |
| **Optical Model** (JAX) | `optical_model_jax.py` | JIT-compilable, vectorized implementation |
| **Dispersing Elements** | `elements.py` | Grism/prism registry: per-element orders, band, STPSF filters, data files |
| **PSF Model** | `psf_model.py` | STPSF grids with trilinear interpolation |
| **Star Disperser** | `star_disperser.py` | Point sources with wavelength-dependent PSFs |
| **Galaxy Disperser** | `galaxy_disperser.py` | Extended sources with Jacobian warping + PSF convolution |
| **Catalog Pipeline** | `catalog.py` + `pipeline.py` + `scripts/build_dispersed_image.py` | Source selection, full-field dispersed-image simulation (stars + galaxies, either element) |
| **Sérsic Profiles** | `sersic.py` | JAX/vmap Sérsic profile generator for galaxy morphologies |
| **Reference Data** | `refdata.py` + `paths.py` + `hydrate.py` | Vendored-data resolver and hydrator; synphot bandpasses and templates |

## Installation

See [INSTALL.md](INSTALL.md) for full instructions (pixi and pip paths, GPU setup, data files).

Quick version:

```bash
git clone git@github.com:roman-grs-pit/roman_disperser.git
cd roman_disperser
pixi install && pixi shell        # or: pip install -e ".[full]"
pixi run hydrate                  # fetch vendored reference data (~6.4 GB)
pytest -q tests -m "not slow"
```

## Quick Start

```python
import jax.numpy as jnp
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser import optical_model_jax as omj, paths, psf_model, star_disperser

# Load optical model and create payloads (paths resolve the hydrated data dir)
model = RomanOpticalModel(str(paths.optical_model_path()))
optical_payload = omj.make_sca_payload(model, sca=5, order="1")
psf_payload = psf_model.get_or_make_psf_payload(
    detector="WFI05", order="1", cache_dir=str(paths.psf_cache_dir())
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
- [Migrating to v0.14 (from v0.10)](docs/migrating-v0.10-to-v0.14.md) — What changed across 0.11–0.14, including the results-changing fixes
- [Optical Model API](docs/optical_model.md) — JAX optical model functions and examples
- [Dispersed-Image Pipeline](docs/grism_pipeline.md) — User guide for `build_dispersed_image.py` (stars + galaxies, either element)
- [Element Support](docs/element_support.md) — Grism/prism status per module and script
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
- `sensitivities.ipynb`, `g0v-star.ipynb` — Sensitivity-curve and stellar-SED explorations

The demo notebooks predate prism support and demonstrate the grism (the
default element); see `notebooks/README.md` for status notes and
`notebooks/archive/` for retired notebooks that no longer track the current
API.

## Project Structure

```
roman_disperser/
├── src/roman_disperser/
│   ├── optical_model.py           # Reference NumPy model
│   ├── optical_model_jax.py       # JAX functional model
│   ├── optical_model_utils.py     # Coordinate system utilities
│   ├── elements.py                # Dispersing-element registry (grism / prism)
│   ├── disperser.py               # Legacy 2D+1D galaxy disperser
│   ├── star_disperser.py          # Star disperser with PSF deposition
│   ├── galaxy_disperser.py        # Galaxy disperser (Jacobian warp + PSF convolution)
│   ├── psf_model.py               # PSF grids and trilinear interpolation
│   ├── psf_utils.py               # STPSF ↔ disperser coordinate conversion
│   ├── sersic.py                  # Sérsic profile generator (JAX/vmap)
│   ├── catalog.py                 # Source selection for detector fields
│   ├── pipeline.py                # Shared pipeline utilities (I/O, batching, sensitivity)
│   ├── paths.py                   # Vendored-data directory resolver
│   ├── hydrate.py                 # Reference-data hydrator (roman-disperser-hydrate)
│   ├── refdata.py                 # Bundled synphot reference data
│   └── demo_utils.py              # Synthetic galaxy/spectrum helpers
├── tests/
├── notebooks/
│   ├── galaxy/                    # Galaxy + star demo notebooks
│   ├── psf/                       # PSF and star notebooks
│   └── archive/                   # Retired notebooks (no longer track the current API)
├── scripts/
│   ├── build_dispersed_image.py   # Unified pipeline, either element (stars + galaxies)
│   ├── build_grism_image.py       # Deprecated forwarding alias for the above
│   ├── generate_psf_caches.py     # Regenerate PSF caches from STPSF (--element)
│   ├── wrap_with_romanisim.py     # Wrap products through romanisim to L2 ASDF
│   ├── example_grism_config.yaml  # Documented batch pipeline config
├── data/                          # Hydrated reference data (gitignored except stars/)
│   ├── Roman_grism_OpticalModel_v0.8.yaml
│   ├── Roman_prism_OpticalModel_v0.8.yaml
│   ├── catalogs/                  # Unified source catalog (Parquet + Zarr)
│   ├── sensitivities/             # Per-SCA grism sensitivity curves
│   ├── sensitivities_prism/       # Per-SCA prism sensitivity curves
│   ├── stars/                     # Star catalog and SED templates (in-repo)
│   ├── synphot/                   # F158/F184 bandpass and templates
│   └── psf_cache/                 # PSF grids, both elements (54 files, ~6.0 GB)
├── docs/
├── pixi.toml
└── pyproject.toml
```

## References

- Roman Space Telescope: https://roman.gsfc.nasa.gov/
- STPSF (Space Telescope PSF Simulator): https://stpsf.readthedocs.io
- JAX: https://jax.readthedocs.io/
- Pixi: https://pixi.sh

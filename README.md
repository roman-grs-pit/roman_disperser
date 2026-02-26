# Roman Disperser

JAX-based optical model and disperser for Roman Space Telescope grism spectroscopy simulations.

## Overview

| Component | Module | Status |
|-----------|--------|--------|
| **Optical Model** (reference) | `optical_model.py` | Complete |
| **Optical Model** (JAX) | `optical_model_jax.py` | Complete |
| **PSF Model** | `psf_model.py` | Complete — STPSF grids with trilinear interpolation |
| **Star Disperser** | `star_disperser.py` | Complete — point sources with wavelength-dependent PSFs |
| **Galaxy Disperser** | `galaxy_disperser.py` | Complete — extended sources with Jacobian warping + PSF convolution |
| **Galaxy Disperser** (legacy) | `disperser.py` | Replaced by `galaxy_disperser.py` |

## Installation

Using [Pixi](https://pixi.sh):

```bash
pixi install
```

### PSF Cache Generation

PSF caches are required for the star disperser. Generate them once (~2 hours with 2 workers):

```bash
pixi run python scripts/generate_psf_caches.py --workers 2
```

Caches are stored in `data/psf_cache/` (36 files: 18 SCAs x 2 grism orders).

## Quick Start

```python
import jax.numpy as jnp
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser import optical_model_jax as omj, psf_model, star_disperser

# Load optical model and create payloads
model = RomanOpticalModel("data/Roman_grism_OpticalModel_v0.8.yaml")
optical_payload = omj.make_sca_payload(model, sca=5, order="1")
psf_payload = psf_model.load_psf_payload("data/psf_cache", detector="WFI05", order="1")

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
# TODO: add galaxy dispersion example
```

## Running Tests

```bash
pixi run pytest -q tests              # All tests
pixi run pytest -m "not slow" tests   # Skip slow tests (STPSF generation)
pixi run pytest -v tests/test_star_disperser.py      # Star disperser tests
pixi run pytest -v tests/test_galaxy_disperser.py    # Galaxy disperser tests
pixi run pytest -v tests/test_psf_model.py           # PSF model tests
```

Test files:
- `test_optical_model_jax.py` — coordinate transforms, polynomial mappings, spectral traces
- `test_disperser.py` — bilinear interpolation, flux conservation, boundary handling
- `test_psf_model.py` — PSF interpolation, caching, trilinear accuracy
- `test_star_disperser.py` — PSF deposition, chunk invariance, flux conservation
- `test_galaxy_disperser.py` — Jacobian warping, PSF convolution, delta-vs-star comparison
- `test_demo_utils.py` — synthetic data generation helpers

## Documentation

- [Optical Model API](docs/optical_model.md) — JAX optical model functions and examples
- [Disperser Design](docs/disperser_design.md) — Legacy disperser implementation details
- [JIT Compilation Strategy](docs/jit_compilation.md) — Closure pattern for non-traceable payloads
- [STPSF Quick Reference](docs/stpsf.md) — Roman WFI grism PSF generation
- [Star Dispersion](docs/star_dispersion.md) — Star dispersion design phases and PSF interpolation
- [PSF Phase 1 Plan](docs/psf_phase1_plan.md) — PSF data model implementation and validation results
- [Galaxy Dispersion Plan](docs/galaxy_dispersion_plan.md) — New galaxy disperser design (Jacobian-based)

## Notebooks

**PSF analysis** (`notebooks/psf/`):
- `psf_analysis.ipynb` — PSF characterization and enclosed energy
- `psf_interpolation_validation.ipynb` — Grid optimization and interpolation accuracy
- `psf_allsca_validation.ipynb` — All 18 SCAs x 2 orders validation

**Star dispersion** (`notebooks/psf/`):
- `single_star_demo.ipynb` — Single star dispersal with wavelength-dependent PSF
- `multi_star_demo.ipynb` — Multiple stars on one detector
- `multi_star_demo_gpu_run.ipynb` — GPU performance benchmarks
- `g0v-star.ipynb` — G0V star spectrum example
- `sensitivities.ipynb` — Sensitivity analysis

**Galaxy dispersion** (`notebooks/galaxy/`):
- `jacobian_exploration.ipynb` — Jacobian characterization for galaxy disperser design

**Galaxy dispersion — legacy** (`notebooks/demos/`):
- `single_galaxy_demo.ipynb` — Single galaxy dispersion (legacy disperser)
- `multi_galaxy_demo.ipynb` — Multi-galaxy batch dispersion (legacy disperser)

## Project Structure

```
roman_disperser/
├── src/roman_disperser/
│   ├── __init__.py
│   ├── optical_model.py           # Reference NumPy model
│   ├── optical_model_jax.py       # JAX functional model
│   ├── optical_model_utils.py     # Coordinate system utilities
│   ├── disperser.py               # Legacy 2D+1D galaxy disperser
│   ├── star_disperser.py          # Star disperser with PSF deposition
│   ├── galaxy_disperser.py        # Galaxy disperser (Jacobian warp + PSF convolution)
│   ├── psf_model.py               # PSF grids and trilinear interpolation
│   ├── psf_utils.py               # STPSF ↔ disperser coordinate conversion
│   └── demo_utils.py              # Synthetic galaxy/spectrum helpers
├── tests/
│   ├── conftest.py
│   ├── test_optical_model_jax.py
│   ├── test_disperser.py
│   ├── test_psf_model.py
│   ├── test_star_disperser.py
│   ├── test_galaxy_disperser.py
│   └── test_demo_utils.py
├── notebooks/
│   ├── psf/                         # PSF and star notebooks
│   ├── galaxy/                       # Galaxy disperser notebooks
│   ├── demos/                        # Legacy galaxy demo notebooks
│   └── archive/                      # Legacy notebooks
├── docs/
│   ├── optical_model.md
│   ├── disperser_design.md
│   ├── jit_compilation.md
│   ├── stpsf.md
│   ├── star_dispersion.md
│   ├── psf_phase1_plan.md
│   ├── galaxy_dispersion_plan.md
│   └── reference/stpsf_full.md
├── scripts/
│   ├── generate_psf_caches.py       # Batch PSF cache generation
│   └── migrate_psf_caches.py        # PSF cache migration
├── data/
│   ├── Roman_grism_OpticalModel_v0.8.yaml
│   └── psf_cache/                   # Generated PSF grids (36 files)
├── workbench/                         # Cross-code comparisons & experiments
├── pixi.toml
└── pyproject.toml
```

## Data Management

Large data files (PSF caches, workbench datasets) are git-ignored and not stored in the repository. Plan is to host them in a public Backblaze B2 bucket (e.g., `roman-disperser-data`), downloadable via plain HTTPS — no credentials or special tools needed for users.

PSF caches can also be regenerated locally (~2 hours): `pixi run python scripts/generate_psf_caches.py --workers 2`

## References

- Roman Space Telescope: https://roman.gsfc.nasa.gov/
- STPSF (Space Telescope PSF Simulator): https://stpsf.readthedocs.io
- JAX: https://jax.readthedocs.io/
- Pixi: https://pixi.sh

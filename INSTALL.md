# Installation

## Quick Start (recommended)

The recommended development workflow uses [Pixi](https://pixi.sh):

```bash
pixi install          # CPU environment
pixi install -e cuda  # GPU environment (Linux + NVIDIA CUDA 12)
pixi run pytest -q tests -m "not slow"
```

Pixi manages all dependencies (Python, JAX, STPSF, etc.) automatically.

## pip install (without pixi)

For production use or custom environments:

```bash
git clone git@github.com:roman-grs-pit/roman_disperser.git
cd roman_disperser

# Minimal install — core only (optical model, dispersers, pre-cached PSF loading)
pip install -e .

# Full install — all dependencies (pipeline, testing, notebooks)
pip install -e ".[full]"
```

### Dependency tiers

| Tier | Includes | Enables |
|------|----------|---------|
| Core (`pip install -e .`) | numpy, jax, scipy, pyyaml, matplotlib, pandas | Optical model, dispersers, pre-cached PSF loading |
| Full (`pip install -e ".[full]"`) | Core + astropy, tqdm, synphot, pytest | Star grism pipeline, sensitivity calibration, testing |

## GPU support

With pixi, use the `cuda` environment: `pixi install -e cuda`. This handles everything automatically.

For pip installs, JAX GPU support is installed separately. There are two options depending on your system:

```bash
# Option 1: System with CUDA already installed (e.g., runpod, HPC clusters)
# Uses your system's CUDA/cuDNN libraries
pip install jax[cuda12-local]

# Option 2: System without CUDA (installs CUDA libraries via pip wheels)
pip install jax[cuda12]
```

**Which to use?** If `nvcc --version` or `nvidia-smi` works on your system, you already have CUDA — use `cuda12-local`. On cloud VMs with pre-installed NVIDIA drivers (runpod, Lambda, etc.), `cuda12-local` is the safer choice since the pip-bundled CUDA libraries can conflict with the system ones.

Verify your GPU is visible:

```bash
python -c "import jax; print(f'Backend: {jax.default_backend()}'); print(jax.devices())"
```

## Data files

| Data | Location | Notes |
|------|----------|-------|
| Optical model, sensitivity curves, star catalog | `data/` (in repo) | Included |
| Synphot reference spectra (F158 bandpass, templates) | `data/synphot/` (in repo) | Included (~60 KB) |
| PSF caches (~4.3 GB) | `data/psf_cache/` | `python scripts/download_psf_caches.py` |
| STPSF reference data | `~/data/stpsf-data` | Only for PSF cache regeneration (~1-2 GB) |

### PSF caches

Most users should download pre-generated PSF caches:

```bash
python scripts/download_psf_caches.py          # skip existing files
python scripts/download_psf_caches.py --force  # re-download all
```

This downloads 36 files (~4.3 GB) from a public GitHub release. No authentication required.

To regenerate caches from scratch (requires STPSF and its reference data):

```bash
pixi run python scripts/generate_psf_caches.py --workers 2  # ~2 hours
```

STPSF reference data is only needed for regeneration. STPSF looks for data in
`~/data/stpsf-data` by default, or set `STPSF_PATH` to override. See
[STPSF docs](https://stpsf.readthedocs.io) for download instructions.

## Verifying your install

```bash
# Minimal install
python -c "import roman_disperser; print('OK')"

# Full install
pytest -q tests -m "not slow"
```

## Getting started

Once installed with PSF caches downloaded:

- **Interactive demo** — `notebooks/galaxy/stars_and_galaxies_demo.ipynb` (CPU) or
  `notebooks/galaxy/stars_and_galaxies_gpu_demo.ipynb` (GPU).
  Disperses stars and galaxies onto a single detector with visualization.

- **Batch pipeline** — generate a config, edit it, and run:
  ```bash
  python scripts/build_star_grism_image.py --generate-config my_config.yaml
  # Edit my_config.yaml (pointings, SCAs, output directory, etc.)
  python scripts/build_star_grism_image.py --config my_config.yaml
  ```
  See `scripts/example_star_config.yaml` for a fully commented example and
  `docs/star_grism_pipeline.md` for details.

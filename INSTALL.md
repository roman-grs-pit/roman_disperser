# Installation

## Quick Start (recommended)

The recommended development workflow uses [Pixi](https://pixi.sh):

```bash
pixi install          # CPU environment
pixi install -e cuda  # GPU environment (Linux + NVIDIA CUDA 12)
pixi run pytest -q tests -m "not slow"
```

Pixi manages all dependencies (Python, JAX, STPSF, etc.) and sets environment variables automatically.

## pip install (without pixi)

For production use or custom environments:

```bash
git clone git@github.com:roman-grs-pit/roman_disperser.git
cd roman_disperser

# Full install — all dependencies (pipeline, PSF generation, testing, notebooks)
pip install -e ".[full]"

# Minimal install — core only (optical model, dispersers, pre-cached PSF loading)
pip install -e .
```

### Dependency tiers

| Tier | Includes | Enables |
|------|----------|---------|
| Core (`pip install -e .`) | numpy, jax, scipy, pyyaml, matplotlib, pandas | Optical model, dispersers, pre-cached PSF loading |
| Full (`pip install -e ".[full]"`) | Core + astropy, tqdm, synphot, stpsf, pytest | Star grism pipeline, PSF generation, testing |

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

## Environment variables

| Variable | Purpose | Default (pixi) |
|----------|---------|-----------------|
| `STPSF_PATH` | Path to STPSF reference data (~1-2 GB) | `$HOME/data/Roman/stpsf-data` |

Pixi sets this default automatically via `scripts/activate.sh`. To override, export the variable before running pixi:

```bash
export STPSF_PATH=/my/custom/path
pixi run pytest -q tests
```

Non-pixi users must set this manually for PSF generation.

- STPSF reference data: downloaded on first use by STPSF, or see [STPSF docs](https://stpsf.readthedocs.io)

The F158 bandpass and spectral templates used for spectrum normalization are
bundled in `data/synphot/` (see `data/synphot/README.md`). `PYSYN_CDBS` and
`stsynphot` are no longer required for normal use.

## Data files

| Data | Location | Notes |
|------|----------|-------|
| Optical model, sensitivity curves, star catalog | `data/` (in repo) | Included |
| Synphot reference spectra (F158 bandpass, templates) | `data/synphot/` (in repo) | Included (~60 KB) |
| PSF caches (~4.3 GB) | `data/psf_cache/` | Generate with `scripts/generate_psf_caches.py` |
| STPSF reference data | `$STPSF_PATH` | External, ~1-2 GB |

## Verifying your install

```bash
# Minimal install
python -c "import roman_disperser; print('OK')"

# Full install
pytest -q tests -m "not slow"
```

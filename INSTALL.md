# Installation

## Quick Start

**1. Clone the repo**

```bash
git clone git@github.com:roman-grs-pit/roman_disperser.git
cd roman_disperser
```

**2. Install dependencies** — pick one path:

- **[Pixi](https://pixi.sh)** — self-contained environment, recommended for development. Manages Python, JAX, STPSF, and all dependencies automatically.
- **pip** — use when adding to an existing Python environment (cloud VMs, HPC, existing venvs).

### Pixi path

```bash
pixi install              # CPU environment
pixi install -e cuda      # GPU environment (Linux + NVIDIA CUDA 12)
pixi shell                # activate the environment for subsequent steps
```

### pip path

We recommend installing into a virtual environment:

```bash
conda create -n roman python=3.12 && conda activate roman  # or use python -m venv
pip install -e ".[full]"  # all dependencies (pipeline, notebooks, testing)
```

For GPU support with pip, see [GPU support](#gpu-support) below.

A minimal install (`pip install -e .`) is also available — it includes only the
core library (optical model, dispersers, pre-cached PSF loading) without astropy,
synphot, or pytest.

**3. Download data assets**

```bash
python scripts/download_psf_caches.py        # PSF caches (~4.3 GB)
python scripts/download_source_catalog.py     # Source catalog (~155 MB)
```

**4. Verify**

```bash
pytest -q tests -m "not slow"
```

**5. Run a demo** — see [Getting started](#getting-started) below.

## GPU support

With pixi, use the `cuda` environment: `pixi install -e cuda`. This handles everything automatically.

For pip installs, JAX GPU support is installed separately:

```bash
# System with CUDA already installed (runpod, HPC clusters, Lambda, etc.)
pip install jax[cuda12-local]

# System without CUDA (installs CUDA libraries via pip wheels)
pip install jax[cuda12]
```

**Which to use?** If `nvcc --version` or `nvidia-smi` works, you already have CUDA — use `cuda12-local`. On cloud VMs with pre-installed NVIDIA drivers, `cuda12-local` is the safer choice since pip-bundled CUDA libraries can conflict with system ones.

Verify your GPU is visible:

```bash
python -c "import jax; print(f'Backend: {jax.default_backend()}'); print(jax.devices())"
```

## Data files

| Data | Location | Notes |
|------|----------|-------|
| Optical model, sensitivity curves, star catalog | `data/` (in repo) | Included |
| Synphot reference spectra (F158/F184 bandpass, templates) | `data/synphot/` (in repo) | Included (~90 KB) |
| Source catalog (~155 MB) | `data/catalogs/` | `python scripts/download_source_catalog.py` |
| PSF caches (~4.3 GB) | `data/psf_cache/` | `python scripts/download_psf_caches.py` |
| STPSF reference data | `~/data/stpsf-data` | Only for PSF cache regeneration (~1-2 GB) |

### Source catalog

The source catalog contains galaxy and star metadata (Parquet) and SEDs (Zarr)
for grism simulations. See `data/catalogs/README.md` for the format specification.

```bash
python scripts/download_source_catalog.py          # skip if exists
python scripts/download_source_catalog.py --force  # re-download
```

This downloads ~155 MB from a public GitHub release. No authentication required.

To rebuild the catalog from scratch (requires access to the Galacticus 4 deg²
mock at `~/data/Roman/galacticus_4deg2_mock/`):

```bash
pixi run python scripts/build_source_catalog.py --sims 1
pixi run python scripts/verify_source_catalog.py   # validate the build
```

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

  **Note:** The batch pipeline processes all 18 SCAs per pointing and benefits
  significantly from a GPU. On CPU, expect ~30 min per SCA; on GPU (e.g., RTX 4090),
  ~1 min per SCA.

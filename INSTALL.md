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

A minimal install (`pip install -e .`) omits the `[full]` extras (astropy,
synphot, tqdm, pytest) — enough for the core JAX optical model and dispersers,
but not the notebook/pipeline/synphot/testing stack. Reference data is vendored
and hydrated separately (step 3) either way.

> The commands below run inside your installed environment — either inside a
> `pixi shell` (or via `pixi run …`), or an activated venv/conda env.

**3. Hydrate reference data**

A fresh checkout ships **no** reference data. Fetch every vendored asset —
optical model, sensitivities, synphot, PSF caches, and source catalog — from the
public `roman_disperser_data` releases. The `roman-disperser-hydrate` command
lives inside your environment, so invoke it through that environment:

- **Pixi** — run through `pixi run` (data lands in `<repo>/data` automatically,
  no configuration needed):
  ```bash
  pixi run hydrate          # all reference data (~4.5 GB)
  ```
  (Equivalently, call `roman-disperser-hydrate` directly from inside `pixi shell`.)
- **pip** — with your venv/conda env activated, set a stable data dir *first* so
  the hydrator and the runtime agree regardless of your working directory:
  ```bash
  export ROMAN_DISPERSER_DATA=~/roman_disperser_data   # any path; add to ~/.bashrc
  roman-disperser-hydrate                              # all reference data (~4.5 GB)
  ```

Fetch only part of it (e.g. on a laptop) by adding flags — under pixi prefix with
`pixi run hydrate`, e.g. `pixi run hydrate --only psf --sca 1 2`:

```bash
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
roman-disperser-hydrate --only psf --sca 1 2                         # a couple of PSF SCAs
```

See [Reference data](#reference-data) below for the data directory, versions,
and reproducibility.

**4. Verify**

```bash
pytest -q tests -m "not slow"
```

**5. Run a demo** — see [Getting started](#getting-started) below.

## romanisim wrapping (spectro acceptance)

A separate `romanisim` pixi environment wraps the
[roman-grs-pit/romanisim](https://github.com/roman-grs-pit/romanisim) fork
on the `extra_counts` branch (the `--extra-counts` patch is unmerged
upstream as of 2026-05-07). Linux-only and isolated from the JAX/CUDA
disperser env so dep trees don't collide.

The CRDS and STPSF caches are kept **outside** the repo so they can be
shared with other projects. Set these in your shell (e.g. add to
`~/.bashrc`) before installing the env — the paths below are examples, adjust
them for your system:

```bash
export STPSF_PATH=/data/npadman/refdata/stpsf-data
export CRDS_PATH=/data/npadman/refdata/crds
export CRDS_SERVER_URL=https://roman-crds.stsci.edu
# CRDS_CONTEXT intentionally unset → operational context
```

Then:

```bash
pixi install -e romanisim
pixi run -e romanisim hydrate-romanisim   # CRDS sync + STPSF status
```

`hydrate-romanisim` runs `crds sync` against the operational context
(unless you've pinned `CRDS_CONTEXT`) and reports STPSF data status.
STPSF reference data (~1-2 GB) is a separate one-time manual download;
the task prints the URL and target path if `$STPSF_PATH/WFI/` is
missing.

When the upstream PR merges, replace the `git`/`branch` pin in
`[feature.romanisim.pypi-dependencies]` with a normal `romanisim` PyPI
spec.

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

## Reference data

All reference data is **vendored**: versioned independently of the code and
fetched on demand with `roman-disperser-hydrate`. Nothing is bundled in the
wheel, and the same data backs pixi and pip installs.

| Data | Path under data dir | Size | Release tag |
|------|---------------------|------|-------------|
| Optical model | `Roman_grism_OpticalModel_v0.8.yaml` | 28 KB | `optical-model-*` |
| Sensitivity curves | `sensitivities/` | 1.9 MB | `sensitivities-*` |
| Synphot reference (F158/F184 + templates) | `synphot/` | 88 KB | `synphot-*` |
| Source catalog | `catalogs/` | 155 MB | `catalog-*` |
| PSF caches | `psf_cache/` | 4.3 GB | `psf-*` |
| STPSF reference data | `~/data/stpsf-data` | 1-2 GB | (only for PSF regeneration) |

`data/stars/` (star catalog, ~12 MB) is a catalog-build input and stays in the
repo. STPSF reference data is **not** fetched by `roman-disperser-hydrate` — it
is listed only because regenerating PSF caches needs it (see
[Regenerating from scratch](#regenerating-from-scratch)).

### Where data lives

`roman-disperser-hydrate` writes into the **data directory**, resolved as:

1. `--dest DIR`,
2. `$ROMAN_DISPERSER_DATA`,
3. `$PIXI_PROJECT_ROOT/data` (set automatically inside a pixi env),
4. `./data` (default).

Pixi sets `PIXI_PROJECT_ROOT`, so pixi users get `<repo>/data` with no
configuration. **pip users should set `ROMAN_DISPERSER_DATA`** to a stable path
— otherwise the default is `./data` relative to wherever you launch, which is
brittle (e.g. notebooks run from subdirectories). The hydrator and the runtime
resolve from the same place, so once set, hydrated data is found automatically.

### Hydrating

```bash
roman-disperser-hydrate                                              # everything (~4.5 GB)
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
roman-disperser-hydrate --only psf --sca 1 2                         # a couple of PSF SCAs
roman-disperser-hydrate --dry-run                                    # show what would be fetched
```

Under pixi, invoke these as `pixi run hydrate …` (or run
`roman-disperser-hydrate` inside `pixi shell`). The legacy
`scripts/download_psf_caches.py` / `download_source_catalog.py` still work as
thin wrappers.

### Versions and reproducibility

Which version of each asset is fetched comes from a **manifest** in the
`roman_disperser_data` repo, so data can be re-versioned without a code release.
Every run also writes `<data>/data-versions.lock` recording the exact versions
installed. To reproduce a specific data state:

```bash
roman-disperser-hydrate --lock <data>/data-versions.lock   # exact versions from a saved lock
roman-disperser-hydrate --manifest <git-ref>               # use a pinned manifest revision
```

### Regenerating from scratch

Most users hydrate. To rebuild instead:

- **Source catalog** — requires the Galacticus 4 deg² mock at
  `~/data/Roman/galacticus_4deg2_mock/`:
  ```bash
  pixi run python scripts/build_source_catalog.py --sims 1
  pixi run python scripts/verify_source_catalog.py   # validate the build
  ```
- **PSF caches** — require STPSF and its reference data (in `~/data/stpsf-data`
  by default, or set `STPSF_PATH`; ~2 hours):
  ```bash
  pixi run python scripts/generate_psf_caches.py --workers 2
  ```
  See [STPSF docs](https://stpsf.readthedocs.io) for the reference-data download.

## Getting started

Once installed and reference data hydrated:

- **Interactive demo** — `notebooks/galaxy/stars_and_galaxies_demo.ipynb` (CPU) or
  `notebooks/galaxy/stars_and_galaxies_gpu_demo.ipynb` (GPU).
  Disperses stars and galaxies onto a single detector with visualization.

- **Batch pipeline** — provide a config YAML (simulation parameters) and an
  ECSV pointing table (APT format):
  ```bash
  python scripts/build_grism_image.py --generate-config my_config.yaml
  # Edit my_config.yaml (SCAs, output directory, batch sizes, etc.)
  python scripts/build_grism_image.py --config my_config.yaml --pointings pointings.ecsv
  ```
  See `scripts/example_grism_config.yaml` and `scripts/example_pointings.ecsv`
  for examples, and `docs/grism_pipeline.md` for details. The batch pipeline
  processes all 18 SCAs per pointing and is substantially faster on a GPU.


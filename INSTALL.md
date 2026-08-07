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
  pixi run hydrate          # all reference data, both elements (~6.5 GB)
  ```
  (Equivalently, call `roman-disperser-hydrate` directly from inside `pixi shell`.)
- **pip** — with your venv/conda env activated, set a stable data dir *first* so
  the hydrator and the runtime agree regardless of your working directory:
  ```bash
  export ROMAN_DISPERSER_DATA=~/roman_disperser_data   # any path; add to ~/.bashrc
  roman-disperser-hydrate                              # all reference data (~6.5 GB)
  ```

You can also fetch only part of it (e.g. on a laptop), with the same flags on
either path:

- **Pixi:**
  ```bash
  pixi run hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
  pixi run hydrate --only psf --sca 1 2                         # a couple of PSF SCAs
  ```
- **pip** (env activated):
  ```bash
  roman-disperser-hydrate --only optical_model,sensitivities,synphot
  roman-disperser-hydrate --only psf --sca 1 2
  ```

See [Reference data](#reference-data) below for the data directory, versions,
and reproducibility.

**4. Verify**

```bash
pytest -q tests -m "not slow"
```

**5. Run a demo** — see [Getting started](#getting-started), next.

## Getting started

Once installed and reference data hydrated:

- **Interactive demo** — `notebooks/galaxy/stars_and_galaxies_demo.ipynb` (CPU) or
  `notebooks/galaxy/stars_and_galaxies_gpu_demo.ipynb` (GPU).
  Disperses stars and galaxies onto a single detector with visualization.
  Note the CPU notebook is a real simulation, not a toy — expect ~30 minutes
  end-to-end (each notebook states its runtime and hardware up front).

- **Batch pipeline** — provide a config YAML (simulation parameters) and an
  ECSV pointing table (APT format):
  ```bash
  python scripts/build_dispersed_image.py --generate-config my_config.yaml
  # Edit my_config.yaml (SCAs, output directory, batch sizes, etc.)
  python scripts/build_dispersed_image.py --config my_config.yaml --pointings pointings.ecsv
  ```
  The grism is the default; add `element: prism` to the config (or
  `--element prism`) for the prism. See `scripts/example_grism_config.yaml`
  / `scripts/example_prism_config.yaml` and `scripts/example_pointings.ecsv`
  for examples, and `docs/grism_pipeline.md` for details. The batch pipeline processes all 18
  SCAs per pointing and is substantially faster on a GPU.
  (`build_grism_image.py` is the deprecated pre-v0.14 name and still works.)

The sections below are reference material: romanisim wrapping, GPU setup,
and the reference-data system in detail.

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

Sizes below are decimal (1 GB = 10⁹ bytes); the full hydrate totals ~6.5 GB.

| Data | Path under data dir | Size | Release tag |
|------|---------------------|------|-------------|
| Optical model (grism) | `Roman_grism_OpticalModel_v0.8.yaml` | 28 KB | `optical-model-*` |
| Optical model (prism) | `Roman_prism_OpticalModel_v0.8.yaml` | 11 KB | `optical-model-prism-*` |
| Sensitivity curves (grism) | `sensitivities/` | 1.9 MB | `sensitivities-*` |
| Sensitivity curves (prism) | `sensitivities_prism/` | 0.6 MB | `sensitivities-prism-*` |
| Synphot reference (F158/F184 + templates) | `synphot/` | 80 KB | `synphot-*` |
| Source catalog | `catalogs/` | 163 MB | `catalog-*` |
| PSF caches (both elements, 54 files) | `psf_cache/` | 6.4 GB | `psf-*`, `psf-prism-*` |
| STPSF reference data | `~/data/stpsf-data` | 1-2 GB | (only for PSF regeneration) |

Grism-only work can skip the prism assets:
`--only optical_model,sensitivities,synphot,psf,catalog` (~4.5 GB). Note the
vendored catalog is currently grism-only (`catalog-v2`, 9000 Å floor) — prism
runs need a 7500 Å-floor catalog; see `docs/element_support.md`.

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

A fully hydrated data directory looks like this:

```
<data dir>/
├── data-versions.lock                    # release tag per asset, written by hydrate;
│                                         #   run time resolves the optical model from it
├── Roman_grism_OpticalModel_v0.8.yaml
├── Roman_prism_OpticalModel_v0.8.yaml
├── catalogs/                             # metadata.parquet + seds.zarr/
├── sensitivities/                        # per-SCA grism FITS + sensitivity_map.yaml
├── sensitivities_prism/                  # per-SCA prism FITS + sensitivity_map.yaml
├── synphot/                              # bandpasses + spectral templates
└── psf_cache/                            # psf_WFI*_{GRISM0,GRISM1,PRISM}_*.npz (54 files)
```

### Hydrating

```bash
roman-disperser-hydrate                                              # everything (~6.5 GB)
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
roman-disperser-hydrate --only psf --sca 1 2                         # a couple of PSF SCAs
roman-disperser-hydrate --dry-run                                    # show what would be fetched
roman-disperser-hydrate --update                                     # upgrade pinned versions (see below)
roman-disperser-hydrate --force                                      # re-download even files already present
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

### Staying current

**Re-running `roman-disperser-hydrate` on an existing data dir never changes
your data.** The versions are pinned by `<data>/data-versions.lock`, and a
plain re-run reuses them: it fills in anything missing (a deleted file, a
PSF SCA you skipped, a newly published asset) but never upgrades an
installed asset, so a mistaken re-hydrate is harmless.

Upgrading to newer deliveries is an explicit choice:

```bash
roman-disperser-hydrate --update      # move the pin to the current manifest
```

`--update` re-installs only the assets whose version actually moved and
rewrites the lock to match, so the lock always describes what is on disk.
(One documented `--sca` caveat: a filtered PSF hydrate records the full
release tag while only the requested SCAs are installed — right version,
partial contents. Using `--sca` to *upgrade* an installed set is refused,
since files outside the filter would silently stay at the old delivery.) The code reads the lock at run time (the optical-model
delivery is resolved from it — see `docs/element_support.md`), and every
pipeline product records the delivery it used (`OPTMODEL` FITS card,
`optical_model` in the meta YAML), so an upgrade is always visible in your
provenance. `--force` re-downloads files that already exist (e.g. to repair
a corrupted file) without moving any version.

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



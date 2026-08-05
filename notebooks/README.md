# Notebooks

Demonstration and validation notebooks for the `roman_disperser` package.
Run them under the pixi environment (`pixi run jupyter lab`, or the `cuda`
env for the GPU variants) after hydrating reference data
(`pixi run hydrate` — see `../INSTALL.md`).

Status note: all current notebooks predate prism support (v0.14.0) and
demonstrate the **grism**, which is the package default — they remain
correct as-is. Prism simulations currently go through the batch pipeline
(`scripts/build_dispersed_image.py --element prism`); there is no prism
notebook yet.

## Galaxy + star demos (`galaxy/`)

The recommended starting point.

- **`stars_and_galaxies_demo.ipynb`** — full field on one detector: 100
  stars + 100 galaxies × 3 spectral orders (CPU).
- **`stars_and_galaxies_gpu_demo.ipynb`** — the same demo, GPU-optimized.
- **`profile_dispersers.ipynb`** — performance profiling: per-operation
  timing breakdown.
- **`jacobian_exploration.ipynb`** — Jacobian characterization behind the
  galaxy-disperser design.
- **`sersic_profiles.ipynb`** — Sérsic profile validation: astropy
  comparison, PA transforms, radial profiles.

## PSF and star notebooks (`psf/`)

- **`single_star_demo.ipynb`** — single star with wavelength-dependent PSF.
- **`multi_star_demo.ipynb`** / **`multi_star_demo_gpu_run.ipynb`** — many
  stars on one detector; GPU benchmarks.
- **`psf_analysis.ipynb`** — PSF characterization and enclosed energy.
- **`psf_interpolation_validation.ipynb`** — PSF grid optimization and
  interpolation accuracy (the 4×4×56 default grid comes from here).
- **`psf_allsca_validation.ipynb`** — all 18 SCAs × 2 grism orders
  validated to <0.03% flux error.
- **`sensitivities.ipynb`**, **`g0v-star.ipynb`** — sensitivity-curve and
  stellar-SED explorations.

## Archive (`archive/`)

Retired notebooks kept for the record; they target superseded APIs and are
not maintained:

- **`single_galaxy_demo.ipynb`**, **`multi_galaxy_demo.ipynb`**,
  **`multi_galaxy_demo_gpu.ipynb`**, **`gpu_scaling_analysis.ipynb`** —
  demos of the legacy `disperser.py` module (`disperse_2d1d_sca`,
  `disperse_galaxies_sequential`), replaced in production by
  `galaxy_disperser.py` (Jacobian warp + PSF convolution). Archived
  2026-08-05 with the v0.14 documentation refresh.

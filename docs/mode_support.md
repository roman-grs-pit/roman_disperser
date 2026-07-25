# Mode support: which scripts work with which dispersing element

`roman_disperser` supports two dispersing elements — the **grism** (G150) and
the **prism** (P127). See `src/roman_disperser/modes.py` for the mode records
and `docs/dispersers.md` for the design.

**Grism is the default everywhere.** Anything that does not ask for a mode gets
the grism, with exactly its pre-merge behaviour. Selection order:

```
explicit argument  >  --mode  >  config `mode:`  >  $ROMAN_DISPERSER_MODE  >  grism
```

Not every script was made mode-aware. That was a deliberate scoping decision
rather than an oversight: scripts nobody runs any more were left alone, and made
to **raise** in prism mode rather than silently return a grism-shaped answer.
This table is the record of which is which.

## Library (`src/roman_disperser/`)

| Module | Status |
|---|---|
| `modes.py` | The mode records themselves, plus `get_mode()` and `check_band()` |
| `paths.py` | ✅ `optical_model_path(mode=...)`, `sensitivity_dir(mode=...)`. `mode` is **keyword-optional** — no-argument callers keep resolving to the grism |
| `pipeline.py` | ✅ `resolve_paths`, `select_sources_per_order`, `load_sensitivities`, `write_mosaic_from_directory` all take `mode=` |
| `psf_model.py` | ✅ `get_cache_filename`, `make_psf_payload`, `get_or_make_psf_payload`, `generate_all_psf_caches` take `mode=`; `default_wavelengths(mode)` spans the mode's band |
| `catalog.py` | ✅ `select_sources` band defaults come from the mode |
| `hydrate.py` | ⏳ prism asset keys written, releases not yet published — see below |
| `optical_model.py`, `optical_model_jax.py` | ✅ element-independent by construction; `wl_transform: log` (prism) was already handled generically |
| `disperser.py`, `star_disperser.py`, `galaxy_disperser.py`, `sersic.py`, `psf_utils.py` | ✅ element-independent — they take payloads, not elements |
| `refdata.py`, `demo_utils.py` | ✅ element-independent (synphot bandpasses, synthetic demos) |

## Scripts (`scripts/`)

| Script | Grism | Prism | Notes |
|---|---|---|---|
| `build_grism_image.py` | ✅ | ✅ | The production pipeline. `--mode`, or `mode:` in the config. Filters the pointing table on the matching `BANDPASS`, and calls `modes.check_band` at setup |
| `build_source_catalog.py` | ✅ | ✅ | Default grid now **7500–21000 Å**, a superset for both elements. `--wl-min 9000` reproduces the grism-era `catalog-v2` grid. The Galacticus slice is *derived* from the requested range, not hardcoded |
| `verify_source_catalog.py` | ✅ | ✅ | Its Galacticus slice is likewise derived from the catalog's own grid |
| `generate_psf_caches.py` | ✅ | ✅ | `--mode`; orders default to the mode's PSF orders |
| `download_psf_caches.py` | ✅ | ⏳ | `--mode prism` maps to the `psf_prism` asset, which 404s until `psf-prism-v1` is published |
| `wrap_with_romanisim.py` | ✅ | ✅ | `--mode` sets `--bandpass` (GRISM/PRISM) unless given explicitly |
| `slurm_generate_psfs.sh` | ✅ | ✅ | Passes `--mode` through to `generate_psf_caches.py` |
| `compute_showcase_traces.py` | ✅ | ✅ | Both elements via `mode=`, resolved through the vendored data dir |
| `compute_roll_traces.py`, `make_showcase_figure.py`, `make_roll_figure.py` | ✅ | n/a | Figure scripts for specific published products |
| **`magnitude_cutoff.py`** | ✅ | ❌ **raises** | Blocker is *physics*: the SNR is per resolution element and `R_GRISM = 461` at 1.45 µm is the grism's resolving power. Needs a prism R(λ) from the optical model or the SSC delivery |
| **`debug_dispersion_nan.py`** | ✅ | ❌ | One-off debugging aid for a 2026-04 NERSC failure. Hardcodes grism orders and `BANDPASS == "GRISM"` |

## Reference data

| Asset | Grism | Prism |
|---|---|---|
| Optical model | `Roman_grism_OpticalModel_v0.8.yaml` | `Roman_prism_OpticalModel_v0.8.yaml` |
| Sensitivities | `<data>/sensitivities/` | `<data>/sensitivities_prism/` |
| PSF caches | `<data>/psf_cache/psf_WFI*_GRISM{0,1}_*.npz` | `<data>/psf_cache/psf_WFI*_PRISM_*.npz` — same directory; the filenames carry the STPSF filter so they cannot collide |
| Catalog | shared — `<data>/catalogs/` | shared, **but see below** |

⏳ **Not yet published.** The prism optical model, sensitivities and PSF caches
are currently *staged by hand* into the shared reference-data directory on the
cluster; they have no release, no manifest entry, and no `data-versions.lock`
entry. See that directory's `PROVENANCE.md`. Publication is deferred until the
Opus and Fable merge branches have been reconciled.

⚠️ **The vendored catalog is still grism-only.** `catalog-v2` is on
`linspace(9000, 21000, 6001)` Å; the prism band opens at 7500 Å, so 7500–9000 Å
has no SED support. `build_grism_image.validate_catalog` **raises** on this
rather than dispersing a truncated spectrum. Prism runs must point
`catalog_dir` at a 7500 Å-floor catalog until `catalog-v3` is built and
published — e.g. `/mnt/roman-science/grs/prism-testing-20260527/catalogs`.

## Consistency checks

`modes.check_band(mode, model)` runs at pipeline setup and raises on:

1. **mode/model mismatch** — prism mode with the grism YAML, or vice versa;
2. **extrapolation** — the mode's band extending outside the model's
   `wl_min`/`wl_max`, which would evaluate the trace polynomials beyond their
   fitted range and return plausible, wrong positions. Narrowing *inside* the
   validity range is allowed;
3. **undefined order** — an order the delivered model does not define.

`tests/test_modes.py` asserts the same pairing for every declared mode without
needing a run, so a future reference-data delivery that moves a band edge trips
a test rather than a production job.

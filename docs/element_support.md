# Element support: which scripts work with which dispersing element

`roman_disperser` supports two dispersing elements — the **grism** (G150) and
the **prism** (P127). See `src/roman_disperser/elements.py` for the element
records (`DispersingElement`, `get_element`, `validate_against_model`).

**Grism is the default everywhere.** Anything that does not ask for an element
gets the grism, with exactly its pre-merge behaviour. Selection is explicit
only, by design — there is no environment-variable element switch, so a shell's
ambient state can never flip which instrument a job simulates:

```
explicit argument  >  --element / config `element:`  >  grism
```

Not every script was made element-aware. That was a deliberate scoping decision
rather than an oversight: scripts nobody runs any more were left alone, and
made to **raise** for the prism rather than silently return a grism-shaped
answer. This table is the record of which is which.

## Library (`src/roman_disperser/`)

| Module | Status |
|---|---|
| `elements.py` | The element records themselves, plus `get_element()` and `validate_against_model()` |
| `paths.py` | element-independent resolver for the data dir; its named helpers (`sensitivity_dir()`, `optical_model_path()`) return the grism defaults. Per-element paths come from `pipeline.resolve_paths(..., element=)`, which reads the element's `optical_model_file` / `sensitivities_subdir` fields |
| `pipeline.py` | ✅ `resolve_paths(..., element=)` picks the per-element optical model and sensitivity paths; everything downstream (`select_sources_per_order`, `load_sensitivities`, the batched dispersers) consumes plain per-element values (`orders`, band) that the caller takes from the element — the element itself never enters jitted code |
| `psf_model.py` | ✅ takes `stpsf_filter=` explicitly; the per-element order→filter mapping lives on `element.stpsf_filters` |
| `catalog.py` | ✅ `select_sources` takes explicit `wl_min`/`wl_max` (grism defaults); callers pass the element's band |
| `hydrate.py` | ⏳ prism asset keys (`optical_model_prism`, `sensitivities_prism`, `psf_prism`) are wired, but no `roman_disperser_data` release carries them yet — see below |
| `optical_model.py`, `optical_model_jax.py` | ✅ element-independent by construction; `wl_transform: log` (prism) was already handled generically |
| `disperser.py`, `star_disperser.py`, `galaxy_disperser.py`, `sersic.py`, `psf_utils.py` | ✅ element-independent — they take payloads, not elements |
| `refdata.py`, `demo_utils.py` | ✅ element-independent (synphot bandpasses, synthetic demos) |

## Scripts (`scripts/`)

| Script | Grism | Prism | Notes |
|---|---|---|---|
| `build_dispersed_image.py` | ✅ | ✅ | The production pipeline. `--element`, or `element:` in the config. Filters the pointing table on the matching `BANDPASS`, validates the optical model against the element at load, and writes the element to the `OPTELEM` FITS card. `build_grism_image.py` is a deprecated forwarding alias |
| `build_source_catalog.py` | ✅ | ✅ | Default grid is **7500–21000 Å**, a superset for both elements. `--wl-min 9000` reproduces the grism-era `catalog-v2` grid. The Galacticus slice is *derived* from the requested range, not hardcoded |
| `verify_source_catalog.py` | ✅ | ✅ | Its Galacticus slice is likewise derived from the catalog's own grid |
| `generate_psf_caches.py` | ✅ | ✅ | `--element`; orders default to the element's orders with a distinct STPSF filter (0,1 grism; 1 prism) |
| `download_psf_caches.py` | ✅ | ⏳ | `--element prism` maps to the `psf_prism` asset, which fails until a release publishes it |
| `wrap_with_romanisim.py` | ✅ | ✅ | Bandpass resolved **per file** from the `OPTELEM` header card (`--bandpass` overrides; `--element` covers pre-v0.13 files without the card). Globs both `grism_*` and `prism_*` product prefixes by default; `--prefix` for custom `output_prefix` runs |
| `slurm_generate_psfs.sh` | ✅ | ✅ | Passes `--element` through to `generate_psf_caches.py` |
| `compute_showcase_traces.py` | ✅ | ✅ | Figure script with its own `--mode` flag and hardcoded paths into the 2026-05 staging checkouts; predates the merge and was left as the record of how those figures were made |
| `compute_roll_traces.py`, `make_showcase_figure.py`, `make_roll_figure.py` | ✅ | n/a | Figure scripts for specific published products |
| **`magnitude_cutoff.py`** | ✅ | ❌ **raises** | Blocker is *physics*: the SNR is per resolution element and `R_GRISM = 461` at 1.45 µm is the grism's resolving power. Needs a prism R(λ) from the optical model or the SSC delivery. `--element prism` raises `NotImplementedError` saying so |
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
entry. See that directory's `PROVENANCE.md`. The hydrate asset keys exist, so
publication is a `roman_disperser_data` release away — unblocked now that the
merge base is settled.

⚠️ **The vendored catalog is still grism-only.** `catalog-v2` is on
`linspace(9000, 21000, 6001)` Å; the prism band opens at 7500 Å, so 7500–9000 Å
has no SED support. `build_dispersed_image.validate_catalog` **raises** on this
rather than dispersing a truncated spectrum. Prism runs must point
`catalog_dir` at a 7500 Å-floor catalog until `catalog-v3` is built and
published — e.g. `/mnt/roman-science/grs/prism-testing-20260527/catalogs`.

## Consistency checks

`elements.validate_against_model(element, model)` runs at pipeline setup and
raises on:

1. **element/model mismatch** — the YAML's `meta.optical_element` differing
   from the requested element (prism run pointed at the grism YAML, or vice
   versa);
2. **band mismatch** — the element's `lam_min`/`lam_max` disagreeing with the
   model's calibrated band beyond float tolerance. Unlike a range *containment*
   check this is strict equality: a deliberately narrowed band is not currently
   supported (revisit with a `padding` argument if that becomes a real use
   case);
3. **undefined order** — an order the delivered model does not define.

`tests/test_elements.py` asserts the same pairing for every declared element
without needing a run, so a future reference-data delivery that moves a band
edge trips a test rather than a production job.

## Optical-model versioning (design note)

Each element record carries `optical_model_file`, a filename that includes
the upstream delivery version (`..._v0.8.yaml`). This is a **default**, not
a binding: the pipeline config `optical_model:` key,
`pipeline.resolve_paths(optical_model_path=...)`, and
`RomanOpticalModel(path)` all accept explicit paths, and
`validate_against_model` guards element identity and band (not version)
whichever file is loaded. So running against a different or newer model is
an explicit-path change, not a code change — but the *default* moves only by
editing `elements.py` in lockstep with the data release that ships the new
YAML. If workflows emerge that hold several model versions side by side
(e.g. delivery comparisons), the filename should probably move out of the
element record into the data manifest; that is an open design question, not
an accident.

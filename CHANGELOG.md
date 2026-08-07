# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.14.2] - 2026-08-07

### Changed
- **Optical-model delivery resolution is now data-driven** (issue #26).
  `DispersingElement.optical_model_file` and `paths.OPTICAL_MODEL_FILE` are
  **removed** (breaking for anyone reading that field), and
  `paths.optical_model_path(element=, version=)` resolves the delivery YAML
  as: explicit path (config `optical_model:`) > explicit version (new config
  key `optical_model_version:` / CLI `--optical-model-version`, also a new
  keyword on `pipeline.resolve_paths`, `setup_pipeline` and
  `build_dispersed_image`) > the delivery recorded in the data dir's
  `data-versions.lock` by hydrate > loud `FileNotFoundError` with hints.
  Resolution is declared, never inferred — directory contents are listed on
  failure but never adopted. No delivery *versions* live in code any more
  (the one naming assumption left is the upstream filename template in
  `paths.optical_model_filename` and the fixed subdir names). The element
  records now carry hardware identity only; their band edges are documented
  as *nominal* (the exact edges come from the sensitivity curves), making
  `validate_against_model` a hardware-consistency gate rather than a
  version pin — a new delivery of the same hardware needs no code change.
  Products record the resolved delivery (`OPTMODEL` FITS card,
  `optical_model` in the meta YAML).
- **Hydrate is pinned by default on an existing data dir.** A plain
  `roman-disperser-hydrate` re-run reuses the versions in
  `data-versions.lock`: it repairs or completes the installation (missing
  files, newly published assets) but never upgrades an installed asset, so
  a mistaken re-hydrate cannot silently move science data. Upgrading is
  explicit: the new `--update` flag re-pins to the current manifest and
  re-installs what changed. `--manifest`/`--lock` still select versions
  explicitly, and the version-change reinstall fix below applies to them
  too. On network use: a pinned re-run of an up-to-date *tarball* asset
  makes no network calls; non-extract assets (optical models, PSF caches)
  still make one release-listing API call per asset to enumerate files.
  Guidance in INSTALL.md ("Staying current").
- **`--mosaic` now resolves the optical model through the lock** (it loads
  the model for focal-plane geometry). A `--mosaic` run against a data dir
  with no lock entry — which worked at v0.14.1 via the filename then baked
  into code — now raises `FileNotFoundError`; the escape hatches are
  `--optical-model` on the command line or a re-hydrate.
- `psf_model.get_or_make_psf_payload` **raises** when given a non-grism
  `stpsf_filter` with default wavelengths: the default grid is the grism
  band and the band is baked into the cache filename, so the old behaviour
  silently looked up (and regenerated) a wrong-band PSF cache — which bit
  the demo notebook on 2026-08-05, shifting its prism flux by 2e-4
  relative. Prefer the new `element=` parameter (below), which makes the
  inconsistency inexpressible.

### Added
- **`element=` on `psf_model.get_or_make_psf_payload`** — the dispersing
  element derives the STPSF filter and the wavelength grid together
  (mutually exclusive with passing either explicitly), so the wrong-band
  combination guarded against above cannot be written at the element level.
  The README Quick Start and both demo notebooks now use it.
- **`paths.sensitivity_dir(element=)`** resolves the per-element
  sensitivity subdir (grism default unchanged); `pipeline.resolve_paths`
  delegates to it instead of duplicating the choice.
- **`roman_disperser.__version__`** — the installed-package version at the
  interpreter, same source as `pipeline.get_code_version()` and the
  `CODEVER` FITS card so the three cannot disagree. The `pipeline` module
  is also exported in `__all__` (docs and notebooks tell users to reach
  for it; previously `hasattr(rd, "pipeline")` was `False`).
- New public names supporting lock-based resolution:
  `paths.optical_model_filename`, `paths.LOCK_NAME`, `hydrate.read_lock`.
- `hydrate.DEFAULT_MANIFEST` gains the three prism keys
  (`optical-model-prism-v0.8`, `sensitivities-prism-v1`, `psf-prism-v1`),
  so the offline fallback covers every registered asset; a test now pins
  that invariant.
- `scripts/example_prism_config.yaml`; prism section in
  `stars_and_galaxies_demo.ipynb` (and its GPU twin, executed on an a10g)
  dispersing the same field through both elements.
- Tests: hydrate lock-vs-contents for non-extract assets,
  `DEFAULT_MANIFEST` completeness, `sensitivity_dir(element=)`,
  `element=` derivation/exclusivity/vendored-cache selection on
  `get_or_make_psf_payload`, `__version__`/`pipeline` export
  (targeted suite 106 passed).

### Fixed
- Hydrating a version that differs from the installed one now re-extracts
  tarball assets (sensitivities, synphot, catalog) instead of skipping on
  the done-marker — which only proves *some* version was extracted once —
  **while still recording the new version in `data-versions.lock`**. The
  lock could previously claim contents that were never installed, untenable
  now that it is the provenance record and drives optical-model resolution.
  The lock now always matches what is on disk; a marker with no lock entry
  (pre-lock data dir) counts as unknown and is reinstalled.
- **The same lock/contents guarantee now covers non-extract assets** (PSF
  caches, optical models): their filenames need not carry the release
  version, so present files whose lock entry disagrees with the resolved
  tag are re-downloaded rather than skipped — previously `--update` to a
  same-filename release would have rewritten the lock while downloading
  nothing. An `--sca`-filtered install at a tag the lock disagrees with is
  **refused** when files outside the filter are present (it would leave
  them at the old delivery while the lock recorded the new tag for all).
  The one remaining `--sca` caveat is completeness: a filtered hydrate
  records the full release tag while only the requested SCAs are on disk.
- README Quick Start snippets carried pre-rename disperser kwargs
  (`star_flux=`/`xsca_star=`) and raised `TypeError` if copy-pasted; both
  snippets are now executed as part of doc maintenance.
- Demo notebooks load per-SCA sensitivity curves via
  `pipeline.load_sensitivities` (they previously loaded SCA1 curves while
  dispersing SCA5) and resolve all vendored data via
  `roman_disperser.paths`.
- `scripts/example_grism_config.yaml` regenerated from `--generate-config`:
  the shipped usage line invoked the deprecated `build_grism_image.py` and
  the file predated the `element:` key.
- Documentation corrected against the code for release (pre-tag review):
  `docs/element_support.md` no longer claims the prism reference data is
  unpublished/hand-staged (it has been published since v0.14.0); the
  documented prism PSF call in README and the migration guide no longer
  raises (now element-parametric); INSTALL.md leads a fresh user to the
  demos directly after the install steps, spells out the pixi and pip
  hydrate variants separately, and documents `--force` and the data-dir
  layout; download-size figures re-measured and unified to decimal units
  (~6.5 GB total); a laptop-local path removed from
  `docs/disperser_design.md`; both notebooks state prerequisites (install
  + hydrate), runtime and hardware up front and fail fast in an early
  cell if reference data is missing; the GPU notebook's introduction now announces
  its prism section and its prism cells use the same JIT-wrapped
  `fori_loop` pattern the notebook teaches (both notebooks re-executed);
  `docs/grism_pipeline.md` documents `OPTMODEL`, the element/optical_model
  meta fields, the general CLI flags, and de-grisms two prism-blind
  passages; README's test-coverage and project-structure listings brought
  back in step with the tree, and the documentation list split into
  current guides vs historical design notes.

## [0.14.1] - 2026-08-05

Documentation release; no code or behaviour change.

### Added
- **`docs/migrating-v0.10-to-v0.14.md`** — user-facing migration guide across
  0.11–0.14: the results-changing fixes (GPU TF32 placement, gnomonic
  projection, per-SCA RNG keys), the loud API breaks
  (`pipeline.ORDERS`/`LAM_MIN`/`LAM_MAX`, float64-only `get_fpa_pos`), the
  renames and new element knobs, and a migration checklist. Linked from a new
  **News** section at the top of the README.

### Changed
- README, INSTALL.md, and CLAUDE.md refreshed for v0.14 reality: both
  dispersing elements in the overview, `build_dispersed_image.py` naming,
  hydrate-based install (the prism assets are published, full hydrate is now
  ~6.4 GB), prism rows in the reference-data table, and corrected project
  structure (elements/paths/hydrate modules, both optical-model YAMLs).
- `notebooks/README.md` rewritten as an accurate index with status notes
  (all current notebooks demonstrate the grism, the default element; prism
  runs go through the batch pipeline).

### Removed
- `notebooks/demos/` — its four notebooks (`single_galaxy_demo`,
  `multi_galaxy_demo`, `multi_galaxy_demo_gpu`, `gpu_scaling_analysis`)
  moved to `notebooks/archive/`: they demo the legacy `disperser.py` module,
  replaced in production by `galaxy_disperser.py`.

## [0.14.0] - 2026-08-05

> Upgrading from v0.10.x? See **`docs/migrating-v0.10-to-v0.14.md`** — two of
> the intervening releases (0.11.0, 0.12.0) changed simulated source
> placement, and 0.13.0 changed ISIM noise realisations.

Prism support (`feature/prism-merge-fable`): the package now simulates
**both** WFI dispersing elements — the G150 grism (default, unchanged
behaviour) and the P127 prism (opt-in) — rebuilt on current `main` rather
than merged from the frozen `prism` branch, which stays as the record of the
original attempt. Chosen over the parallel `feature/prism-merge-opus` attempt
after a head-to-head comparison (both agree at the same-code GPU noise floor;
see `workbench/20260805-prism-merge-validation/`), with that branch's four
distinguishing features grafted in. Full suite green at the release tree:
CPU 474 passed / 0 failed, GPU (a10g, `-m 'not slow'`) 472 passed / 0 failed
(SLURM 7096).

### Added
- **`roman_disperser.elements`** — the dispersing-element registry. A
  `DispersingElement` is a frozen bundle of the constants that differ
  between elements (orders, band, STPSF filter map, default data-file
  names, ECSV `BANDPASS` value); `GRISM` and `PRISM` are the two instances,
  and grism is the default wherever an element argument is optional.
  `validate_against_model()` **raises** on any element/optical-model
  mismatch (wrong `optical_element`, band disagreement beyond 0.005 um,
  undefined order) — never a warning. Host-side configuration only; nothing
  enters jit-compiled code.
- `build_dispersed_image.py` (né `build_grism_image.py`; see Deprecated):
  `element:` config key and `--element` CLI flag.
  The element drives the wavelength trim, the ECSV `BANDPASS` row filter
  (`GRISM`/`PRISM`), per-order optical payloads, sensitivity loading, and
  PSF-cache selection. New provenance: `OPTELEM` card in every FITS primary
  header and `element:` in the per-pointing meta YAML.
- Catalog band coverage is validated against the element band at setup —
  running the prism against the grism-grid `catalog-v2` (opens at 9000 Å vs
  the 7500 Å prism band) is now a hard error instead of silently dispersing
  nothing below 9000 Å.
- `generate_psf_caches.py --element`, `download_psf_caches.py --element`,
  and an element-aware `scripts/slurm_generate_psfs.sh` (ported from the
  prism branch).
- `hydrate.py`: prism asset keys (`optical_model_prism`,
  `sensitivities_prism`, `psf_prism`). **No releases are cut yet**; hydrate
  skips them until the `roman_disperser_data` manifest carries the keys.
- Tests: `tests/test_elements.py`; the shared conftest fixtures are
  parametrized over both elements (disperser/galaxy machinery runs under
  both the grism's linear and the prism's log wavelength transform); prism
  spot-checks `TestTraceBeamPrism` and `TestSelectSourcesPrism`.
- `build_source_catalog.py`: default SED grid widened to **7500–21000 Å**, a
  superset of both element bands (grafted from the parallel
  `feature/prism-merge-opus` attempt). The Galacticus slice is *derived* from
  the requested range and asserted against the grid rather than hardcoded;
  `--wl-min`/`--wl-max` are exposed (`--wl-min 9000` reproduces the grism-era
  `catalog-v2` grid exactly). Same derivation in `verify_source_catalog.py`;
  `tests/test_build_source_catalog.py` pins it.
- `wrap_with_romanisim.py`: the romanisim bandpass is now resolved **per
  file** from the `OPTELEM` header card (explicit `--bandpass` overrides;
  `--element` covers pre-v0.13 products without the card; `GRISM` remains the
  last-resort default), and the input glob accepts both `grism_*` and
  `prism_*` product prefixes (`--element` narrows, `--prefix` overrides for
  custom `output_prefix` runs).
- `magnitude_cutoff.py --element prism` **raises** `NotImplementedError`
  naming the physics blocker (its SNR is per resolution element and
  `R_GRISM = 461` is the grism's resolving power), instead of silently
  returning a grism answer; its sensitivity dir now resolves through
  `paths.sensitivity_dir()` (honors `$ROMAN_DISPERSER_DATA`).
- **`docs/element_support.md`** — status table of grism/prism support per
  module and script, including the deliberate non-ports and the
  reference-data publication state.

### Changed
- **Removed the module-level grism constants `pipeline.ORDERS`,
  `pipeline.LAM_MIN`, `pipeline.LAM_MAX`** — orders and band are
  per-element and passed explicitly. Anything still importing them fails
  with `ImportError` (intentionally loud). `scripts/debug_dispersion_nan.py`
  and `workbench/20260323-batch-tuning/bench_galaxy_batch.py` were repaired
  against the new API (both stay grism-only by declaration, taking their
  constants from `elements.GRISM`); remaining historical `notebooks/`
  entries are deliberately not ported.
- `pipeline.load_sensitivities` and `pipeline.select_sources_per_order`
  take `orders` (and band) explicitly; `resolve_paths` takes `element=`.
- `psf_model`: the three hardcoded `{'0': 'GRISM0', '1': 'GRISM1'}` maps
  are replaced by an explicit `stpsf_filter` parameter (grism defaults via
  `resolve_stpsf_filter`). `get_cache_filename` no longer silently maps
  unknown orders to `ORDER<n>`; it raises. PSF payloads and cache files
  record `stpsf_filter` (pre-existing caches load as `'unknown'`).
- Test fixtures resolve the optical model via `roman_disperser.paths`
  instead of hand-built `$PIXI_PROJECT_ROOT/data/...` paths (issue #24 —
  they failed on any clean checkout).

### Deprecated
- `scripts/build_grism_image.py` — renamed to
  **`scripts/build_dispersed_image.py`** (it simulates either element). The
  old name remains as a forwarding wrapper with identical CLI, defaults and
  outputs, emitting a `FutureWarning`; existing drivers (`roman_l2_job`)
  keep working unchanged. It will be removed in a later release.

### Unchanged (deliberately)
- Grism products keep their `grism_` filename prefix. The prefix now
  defaults to the **element name** (`prism_` for prism runs) and is
  configurable via `output_prefix:` — pin `output_prefix: grism` to mimic
  the prism branch's historical naming. `roman_l2_job` drivers that glob
  `grism_*` need that pin (or a glob update) before any prism L2 wrap.
  `--mosaic` now discovers any `*_detSCA*.fits` prefix.
- `scripts/build_star_grism_image.py` stays deleted (removed in `0881597`
  for duplicating the sky→FPA call site); it was not resurrected from the
  prism branch.
- Reference data stays out of git: the prism YAML, `sensitivities_prism/`,
  and the 18 PRISM PSF caches live in the shared vendored data directory.

## [0.13.1] - 2026-08-05

The optical-model line-centering validation harness and the SSC line-grid
campaign, plus a pre-merge cleanup pass. **No change to simulation behaviour**
— placement, flux and noise are identical to 0.13.0; the full suite is green on
CPU (401 passed) and GPU (407 passed, a10g).

### Added
- **`roman_disperser.refdata.FLAM_0AB_COEFF`** (newly public) — f_lambda of a
  0-AB-mag flat-f_nu source, stored as the product `3.631e-20 * 2.99792458e18`
  so the derivation *is* the value. Previously two hand-rounded copies
  (`0.10885`, `0.108866`, a 1.5e-4 relative spread) lived in separate scripts.
- Line-centering validation tools in `scripts/`: `build_line_test_catalog.py`
  (synthetic emission-line catalogs), `check_line_centering.py` (predict line
  positions from the optical model and centroid them in the rendered image),
  `analyze_psf_shifts.py` and `make_gaussian_psf_cache.py` (PSF-shift controls).
- `workbench/20260731-ssc-line-grid/` — waves 1b/2/3 of the SSC line-grid
  campaign rerun on 0.13.0, plus the before/after analysis and addendum-deck
  generator, and a **README recording known labelling defects in the delivered
  products** (`F158`/`flux_scale` are the flux anchor rather than synthetic
  photometry — 3.425 mag off for the two lines-only waves; `lines.ecsv` asserts
  a continuum the lines-only catalogs do not contain).

### Changed
- `scripts/magnitude_cutoff.py` imports `FLAM_0AB_COEFF` instead of its own
  rounded literal. Its cutoff moves by 4.3e-5 relative.
- `.gitignore`: vendored `data/` entries no longer carry a trailing slash, so
  they match when a worktree symlinks them at the shared cache instead of
  hydrating locally; the optical model is globbed over element and version;
  workbench data is ignored at any depth (`workbench/**/data/`).

### Removed
- `scripts/{line_test_pointing.ecsv,line_test_config.yaml,slurm_line_test.sh}` —
  the pointing file carried `MA_TABLE_NUMBER=1` against 1036 in every campaign
  from 20260724 on, and `MA_TABLE` reaches romanisim's `--ma_table_number`, so
  it selects the L2 readout pattern. No product was affected. `example_pointings
  .ecsv` already covers the same field correctly.
- `scripts/{make_report_figures.py,compare_controls.py}` and the superseded
  `20260723` / `20260724-*` line-grid campaigns. Products and mirrored recipes
  remain at `/mnt/roman-science/grs/line-tests-20260724{,-cont,-gal}/`.

### Fixed
- `make_truth_tables.py` described its output as "DOUBLE-precision evaluations
  of the optical model". It is neither that nor the float32 values the renderer
  used: `make_sca_payload` returns an all-float32 payload, and JAX keeps
  float32⊗float32 in float32 even with x64 enabled. Measured against the shipped
  `residuals.parquet` (8,570 rows): zero bit-identical, median 5.3e-4 px, max
  3.4e-3 px; against true float64, 7.2e-4 px max, ~99% of it a deterministic
  per-(SCA, order) constant.
- `analyze_psf_shifts.py` documented `psf_grid` as x-first, contradicting
  `psf_model`'s y-first convention.
- `before_after.py` no longer reconstructs `d_disp` as `x_meas - x_pred_jax`
  when the column is absent — a different statistic from the checker's
  projection onto the local dispersion tangent.

## [0.13.0] - 2026-07-31

Stage 3 of the precision/reproducibility sequence (stages 1 and 2 were
0.11.0 and 0.12.0): reproducible per-SCA noise and self-identifying
products. No placement or flux change — the noiseless MODEL extension is
bit-identical to 0.12.0.

### Changed
- **Per-SCA RNG keys are now folded from the SCA number** (issue #20).
  Previously keys came from `jax.random.split(pointing_key, len(sca_list))`
  indexed by *list position*, so a `scas: [5]` run gave SCA 5 the key an
  18-SCA run gave SCA 1, and no subset run could reproduce a full run's
  noise. Keys are now `jax.random.fold_in(pointing_key, sca_num)`
  (`pipeline.make_sca_keys`), making them independent of which other SCAs
  are in the run. **ISIM noise realisations change by construction** for
  every SCA relative to ≤0.12.0; MODEL is unaffected. Old products remain
  reconstructible from their `RNDSEED0`/`RNDSEED1` header cards via
  `jax.random.wrap_key_data`. This removes the RNG obstacle to 1-SCA
  regression gates for full 18-SCA runs — but note that GPU products are
  not bit-reproducible even at fixed keys (scatter-add nondeterminism at
  the f32-epsilon level, ~1e-7 relative in MODEL, measured a10g
  2026-07-31; flips ~tens of Poisson counts per SCA), so GPU gates must
  compare with tolerances (`allclose`, relative sums); bitwise gates
  require a deterministic backend (`--xla_gpu_deterministic_ops` measured
  >100x slower).

### Added
- **`CODEVER` and `GITSHA` provenance cards in every FITS product** (both
  quick and batch mode; previously `GITSHA` was batch-only and `CODEVER`
  did not exist), written unconditionally by `pipeline.write_fits`.
  `CODEVER` is the installed package version; `GITSHA` is the full commit
  SHA of the pipeline checkout, with a `-dirty` suffix when the working
  tree has uncommitted changes — a SHA over uncommitted changes identifies
  nothing. The per-pointing metadata YAML likewise gains `codever` and
  carries `git_sha` in both modes.
- `tests/test_pipeline.py` — first coverage of the RNG-key path
  (subset-invariance, distinctness, the exact fold-in derivation) and of
  the provenance cards round-tripping through a written FITS file.

## [0.12.0] - 2026-07-31

### Changed
- **Sky→FPA now uses a proper gnomonic (TAN) projection.**
  `sky_to_tangent_offsets` replaces the flat-sky approximation
  (`Δα·cos δ`, `Δδ`) with the exact tangent-plane projection, closing
  issues #5 and #19 together (stage 2 of the sky→FPA precision work; stage 1
  was 0.11.0). Source placement moves by the size of the flat-sky error being
  removed — third order in the offset at the equator but second order off it
  (North error `Δα² sin(2δ₀)/4`, worst at δ₀ = 45°): over a ±0.4° field,
  ~0.06 px at Dec 0 growing to ~20 px at Dec 60. The golden-value literals
  shifted by 0.064 / 1.294 / 4.509 / 19.739 px (equator → Dec 60) on
  regeneration. **Any product simulated off the equator before this release
  carries the flat-sky placement error** (in addition to the GPU TF32 error
  fixed in 0.11.0 if applicable); equatorial products (e.g. the SSC line-grid
  pointing at Dec 0.0–0.95) are affected only at the ≲0.1 px level.
- The implementation is a **verbatim transcription** of the derivation
  notebook (`docs/reference/tangent_plane_derivation.ipynb`, steps 1–3:
  rotate the pointing to the pole in 3-D, project by dividing by z), same
  NumPy float64 operations in the same order. Tests exec the `tangent_plane`
  cell straight from the committed notebook and assert **bitwise** equality,
  so the arithmetic must not be "cleaned up" independently of the notebook.

### Added
- `docs/reference/tangent_plane_derivation.ipynb` — the derivation of record
  for the projection and the rotation conventions, including the proof that
  the legacy rotation construction (`pa + 180 − 60` plus double negation)
  equals `R_NE(−(PA + focal_pa))`, and the Taylor expansion of the flat-sky
  error. Committed byte-for-byte from the executed original apart from one
  scrubbed stderr line.
- `optical_model_jax.FOCAL_PA_DEG = -60.0` — the focal-plane orientation
  constant, previously a bare literal inside `get_pa_rotation` (numerically a
  no-op), with a convention note in the docstring.
- `TestGnomonicNotebookOracle` — the exec-from-notebook oracle: bitwise
  projection equality plus rotation-convention equivalence of the full
  `get_fpa_pos` against the notebook over a PA × declination grid (worst
  case 2.6e-3 px, float32-rotation round-off). The independent astropy TAN
  oracle now asserts agreement (< 0.01 px at Dec 0/30/60/85) instead of
  characterising the flat-sky error.

### Removed
- The meridian-crossing `ValueError` in `sky_to_tangent_offsets`: the
  gnomonic projection is periodic in RA by construction, so a field
  straddling RA = 0 is now simply placed correctly (the previously-xfailed
  meridian test passes and the marker is gone). The float32 `TypeError`
  guard is **kept, permanently** — quantisation of absolute RA happens
  before the projection can help.

## [0.11.0] - 2026-07-31

### Fixed
- **Sky→FPA source placement was wrong on GPU (TF32).** The position-angle
  rotation in `optical_model_jax.get_fpa_pos` was an unannotated float32
  matmul. With `jax_enable_x64` off, XLA:GPU serves that as TF32 on Ampere and
  later — a 10-bit mantissa, eps ≈ 4.9e-4 — while the identical op on CPU is
  exact. Every source in a GPU run was placed at a perturbed focal-plane
  position: median 1.84 px, up to 7.08 px, growing with field radius. Now uses
  `jnp.matmul(..., precision='highest')`. **All grism/prism products generated
  on GPU before this release carry the error**; the catalogue and manifest
  RA/Dec are correct, only the placement was wrong, so the products remain
  internally self-consistent. A consumer who re-runs their own disperser from
  the catalogue coordinates will disagree with them.
- **Absolute RA was silently downcast to float32 before differencing.** The
  call site passed `jnp.array(ra)` from float64 pandas, quantising right
  ascension *before* `(ra - pointing_ra)` — catastrophic cancellation, with an
  error that scaled with the pointing: 0.006 px at RA 10, but 0.40 px at RA
  260, where float32 ulp is 0.11 arcsec. The differencing now happens in float64
  on the host (`sky_to_tangent_offsets`), making the accuracy independent of
  where the telescope points (9.8e-5 px at every RA tested).

### Added
- `optical_model_jax.sky_to_tangent_offsets` (host, float64) and
  `get_fpa_pos_from_offsets` (JAX, jit-compilable), the two halves of the
  sky→FPA transform. `get_fpa_pos` composes them and keeps its signature.
- `sky_to_tangent_offsets` **raises `TypeError` on float32 input** (including
  JAX arrays, which are float32 with x64 disabled). By the time absolute RA is
  float32 the quantisation has already happened and upcasting cannot undo it,
  so a call site that wraps its catalogue columns in `jnp.array()` — how the
  original defect shipped — now fails loudly instead of silently reintroducing
  the pointing-dependent error. The cost is an O(1) dtype check on the host.
- `tests/test_precision_convention.py`: AST scan asserting every matmul-class
  JAX op in the package declares `precision=`, with self-tests pinning the
  checker's own behaviour on known-good and known-bad forms. Runs on CPU in
  milliseconds, so unlike the `has_gpu()`-guarded tests it cannot skip silently
  on a GPU-less machine.
- `scripts/slurm_run_tests.sh`: submits the GPU test suite to `gpu-med`. The
  CPU/GPU tests existed but were never run anywhere with a GPU, which is why
  the TF32 defect survived.
- Sky→FPA tests now state tolerances in **pixels**. The previous cross-check
  compared positions in degrees at `ATOL = 1e-3`, which is 33 px — loose enough
  to pass a 1.84 px displacement. Adds an independent astropy TAN oracle and
  float64 golden values.
- Showcase visualization scripts (`scripts/`) and figures (`figures/`):
  - `make_showcase_figure.py` — 3-panel "same sky, three views" of the
    acceptance/roll test products at RA≈10/Dec≈0: imaging (L3 coadd) | grism |
    prism, all at the same pointing (RA=10/Dec=0/PA=0, SCA3) so the grism and
    prism show identical objects. Marks the brightest stars with order-1 trace
    boxes and an N/E compass.
  - `make_roll_figure.py` — 2×2 of the same grism field at three telescope rolls
    (PA=0/10/180), showing the dispersion rotating/flipping with roll.
  - `compute_showcase_traces.py` / `compute_roll_traces.py` — precompute the
    marker stars' order-1 traces via the optical model (run in the
    `roman_disperser`/`roman_disperser_prism` envs); the figure scripts run in
    the `roman_l2_job` env to read the L2/L3 ASDFs.

### Changed
- **`get_fpa_pos` is no longer jit-compilable**, deliberately: its float64
  differencing is host NumPy. Jit `get_fpa_pos_from_offsets` instead. In
  production the host half runs once per pointing over a few thousand sources
  and is not on a hot path.

### Known issues
- **`get_fpa_pos` has no RA wrap handling; a field crossing RA = 0 is
  refused.** `dx = ra - pointing_ra` gives +359.8 deg where the true separation
  is −0.2 deg. Because `cone_search` uses a wrap-safe haversine, such sources
  are correctly *selected*, then placed ~360 deg off the focal plane and
  silently culled by the detector bounding box — vanishing from both image and
  truth table with no error. Rather than mis-place them, `sky_to_tangent_offsets`
  now raises when any source lies more than 180 deg from the pointing in RA.
  Affects any pointing within ~0.4 deg of RA = 0; no run to date is affected.
  Fixed for free by the gnomonic projection (next release), since a
  tangent-plane transform uses sin/cos of the RA difference and is periodic.
  See issue #19.
- **The flat-sky approximation `Δα·cos δ` is still in use**, and is a larger
  error than the TF32 defect for any pointing off the equator. The dominant term
  is second order — `Δη ≈ -(tan δ₀ / 2)·u²` with `u = Δα·cos δ₀`, the curvature
  of parallels — so it grows as `u²·tan δ₀`: quadratic in field radius, and zero
  at the equator. Source displacement measured against an astropy TAN
  projection over a ±0.4 deg field:

  | pointing Dec | median | max |
  |---|---|---|
  | 0° | 0.12 px | 0.74 px |
  | 30° | 6.75 px | 26.76 px |
  | 60° | 20.25 px | 79.53 px |

  Replaced by a gnomonic projection in the next release, which also fixes the
  RA wrap above (a tangent-plane transform is periodic in the RA difference).
  See issue #5.

## [0.10.0] - 2026-06-12

> `0.9.x` is reserved for the prism line (`v0.9.0-prism`); the mainline
> continues at `0.10.0`.

All reference data is now **vendored** — fetched on demand and versioned
independently of the code, rather than tracked in this repo.

### Added
- `roman-disperser-hydrate` console command (module `roman_disperser.hydrate`):
  downloads vendored reference data (optical model, sensitivities, synphot, PSF
  caches, source catalog) from the public `roman_disperser_data` releases.
  Supports `--only`/`--sca` selection, `--dry-run`, and a manifest/lock
  versioning model (`--manifest <ref>`, `--lock`); each run writes
  `<data>/data-versions.lock`. `pixi run hydrate` is the pixi invocation.
- `roman_disperser.paths`: single resolver for the data directory
  (`--dest`/arg → `$ROMAN_DISPERSER_DATA` → `$PIXI_PROJECT_ROOT/data` →
  `./data`).

### Changed
- `pipeline.resolve_paths()` and `refdata.py` resolve all reference data through
  `roman_disperser.paths`. New neutral `$ROMAN_DISPERSER_DATA` override;
  `$PIXI_PROJECT_ROOT` retained as a fallback. `refdata` synphot lookup no
  longer walks a repo-relative `data/` (so it works from a wheel install).
- `scripts/download_psf_caches.py` / `download_source_catalog.py` are now thin
  back-compat wrappers around the hydrator.
- INSTALL.md and CLAUDE.md document the vendored-data install/hydrate flow.

### Removed
- The optical model, `data/sensitivities/`, and `data/synphot/` are no longer
  tracked in the repo — they are vendored (`optical-model-v0.8`,
  `sensitivities-v1`, `synphot-v1`). A fresh checkout has no reference data and
  must hydrate. `data/stars/` (catalog-build input) remains in the repo.

## [0.8.2] - 2026-06-11

### Removed
- Stale one-off development scripts under `notebooks/archive/` (`20260110-devel.py`, `quicklook_jax.ipynb`, `simple_optical_test_00.py`, `test_3d_grids.py`, `test_efficiency.py`, `test_multidim.py`, `test_vandermonde.py`, `test_vmap_fusion.py`). These were unmaintained, used stale APIs and hardcoded relative data paths, and broke bare `pytest` collection. Preserved in git history.

## [0.8.1] - 2026-06-03

### Fixed
- `validate_catalog` now checks that `star_seds` and every `galaxy_seds/sim_XXX` partition have a wavelength-bin count matching the shared `wavelengths` grid (GitHub issue #11). A catalog with an off-by-one wavelength grid (e.g. galaxy SEDs built with `np.arange` while the grid used `np.linspace`) previously slipped past validation and crashed mid-dispersion with a cryptic `IndexError` from `seds_full[:, wl_mask]`. It now fails fast at startup with a clear message naming the offending array and the two sizes. The check reads only zarr array `.shape` metadata, so it adds negligible time.

### Changed
- `pipeline.select_sources_per_order` now threads `wl_min`/`wl_max` (defaulting to `LAM_MIN`/`LAM_MAX`) into the trace bbox cull instead of relying on `catalog.select_sources` defaults. Behavior is unchanged for the current band (0.9–2.0 µm), but this removes a latent dependency on the defaults matching the active mode — a different band (e.g. the prism at 0.75–1.85 µm) previously could silently drop real on-detector sources from the per-order mask.

## [0.8.0] - 2026-05-08

### Added
- Romanisim L2 wrap (`scripts/wrap_with_romanisim.py`): produces L2 ASDF outputs from disperser FITS via header-driven `romanisim-make-image` invocations
  - Header-driven argument derivation: `WFICENRA/DEC` → `--radec`, `WFICENPA-60` → `--roll`, `DETNUM` → `--sca`, `MA_TABLE` → `--ma_table_number`, `RNDSEED0 ^ RNDSEED1` → `--rng_seed`
  - `ThreadPoolExecutor` pool of `--num-threads` subprocesses; deterministic round-robin partitioning via `--worker-index/--num-workers`; SHA-256 manifest hash logged per worker for cross-node drift detection
  - Idempotent: skips files whose `_l2.asdf` already exists, so resubmits cheaply fill in gaps after partial completion
- Isolated `[feature.romanisim]` pixi env (linux-64, no default feature) pinned to the `roman-grs-pit/romanisim @ extra_counts` fork
  - `scripts/hydrate_romanisim.py` syncs the operational CRDS context (~140 GB) and reports STPSF status; monkey-patches around two CRDS 13.1.16 bugs that crash bulk Roman sync
- APT pointing tables: batch mode now consumes ECSV pointing tables with full APT identifiers (plan/pass/segment/observation/visit/exposure)
  - Per-pointing output dirs named from APT IDs; identifiers stored in FITS headers and meta YAML
  - Per-pointing RNG keys derived deterministically from `(seed, ECSV basename, APT IDs)` — slice-invariant and file-isolated
- Multi-GPU + parallel-worker support
  - `--gpu N` selects a GPU device; `--worker-index I --num-workers K` partitions pointings round-robin across SLURM tasks
  - `--warmup-only` mode compiles JIT functions for a SCA subset for use as a JIT-cache prewarm pass
  - Configurable JAX compilation cache directory (`--cache-dir`, YAML `cache_dir`, or `JAX_COMPILATION_CACHE_DIR` env)
- Catalog tooling: `scripts/pad_catalog.py` for RA-periodic replication of a unified catalog (workaround for narrow source catalogs near pointing edges)
- Acceptance-test workbenches under `workbench/`:
  - `20260414-acceptance-testing/`: NERSC initial run (32 pointings × 18 SCAs on 4 GPUs)
  - `20260505-acceptance-testing-aws/`: AWS rerun after the NERSC SED-related failures
  - `20260508-romanisim-wrap/`: post-dispersion chain (romanisim wrap, RA/Dec mosaic renderer, S3 archive)
- `workbench/20260508-romanisim-wrap/render_l2_pointing_mosaic.py`: per-pointing RA/Dec mosaic PNG that reads each detector's GWCS, rebins in detector space, and reprojects pixel centers onto a tangent-plane canvas
- `workbench/20260508-romanisim-wrap/s3_archive.sh`: idempotent `aws s3 sync` of an acceptance run + pointing ECSV to the staging S3 bucket

### Changed
- Source catalog `F158` column now stores **maggies** (linear AB flux) instead of AB magnitudes, matching the romanisim catalog convention. Conversion: `maggies = 10^(-0.4 * mag)`. For stars, `flux_scale` equals `F158` numerically; for galaxies, `flux_scale` is always 1.0. The disperser math is unchanged (reads `flux_scale`, not `F158`); detector outputs are bit-identical to prior runs. Catalog release bumped to `catalog-v2`. Updated `build_source_catalog.py`, `verify_source_catalog.py`, `data/catalogs/README.md`, `docs/grism_pipeline.md`, and `download_source_catalog.py`. The verifier reports F158 errors in mag space (`-2.5·log10(F158)`) for human-readable diagnostics.
- Production SLURM driver switched from one-pointing-per-task to N-task arrays of long-running Python workers, each owning its round-robin share of pointings. Cuts process-startup overhead on large arrays and keeps the JAX in-memory state warm across pointings on the same node.

### Fixed
- Galaxy SED scrubber in `build_grism_image.py:load_galaxy_seds` discards SEDs containing NaN/Inf at load time. Three known-bad galaxies in the padded `catalogs_padded/seds.zarr` were crashing the 2026-04-30 NERSC dispersion run.
- Output writers harden against non-finite pixels.

## [0.7.0] - 2026-04-09

### Added
- Unified grism simulation pipeline (`scripts/build_grism_image.py`): disperses both stars and galaxies from a single catalog
  - Parquet+Zarr unified source catalog format (see `data/catalogs/README.md`)
  - Per-source Sérsic morphology generation via `sersic.py`
  - Galaxy dispersion with Jacobian-based shape warping + PSF convolution via `galaxy_disperser`
  - Separate `star_batch_size` and `galaxy_batch_size` configuration
  - Per-pointing source manifest (Parquet) with source type, position, flux, and F158 mag
  - Per-pointing metadata YAML with RNG keys and per-SCA/order source counts
- JAX Sérsic profile generator (`sersic.py`) for galaxy morphology pipeline
- Shared pipeline utilities (`pipeline.py`): cone search, source selection, batched dispersion, I/O
- Unified pipeline documentation (`docs/grism_pipeline.md`)
- Catalog format specification (`data/catalogs/README.md`)
- Catalog download and extraction scripts

### Changed
- Memory-efficient per-SCA processing: PSF payloads, dispersers, and JIT functions built per-SCA and released after use (~2-3 GB vs ~18+ GB for all 18 SCAs)
- Galaxy SEDs loaded per-SCA instead of per-pointing to avoid OOM with large catalogs
- On-disk JAX compilation cache (`/tmp/jax-cache-grism`) persists compiled functions across runs (~2.5s vs ~10s per function)
- `batch_size` config key renamed to `star_batch_size` (old key accepted with deprecation warning)

### Deprecated
- `scripts/build_star_grism_image.py`: use `scripts/build_grism_image.py` instead
- `scripts/example_star_config.yaml`: use `scripts/example_grism_config.yaml` instead

## [0.6.0] - 2026-03-11

### Added
- pip installability with dependency tiers: core (`pip install -e .`) and full (`pip install -e ".[full]"`)
- Bundled synphot reference data (`data/synphot/`): F158 bandpass, G0V stellar template, KC96 galaxy templates — eliminates PYSYN_CDBS dependency
- `refdata` module for loading bundled spectral data without stsynphot
- PSF cache download script (`scripts/download_psf_caches.py`): downloads pre-generated caches from GitHub Releases
- INSTALL.md with branching quickstart (pixi and pip paths), GPU setup, data file guide

### Changed
- STPSF moved from pip dependency to pixi-only (only needed for PSF cache regeneration)
- stsynphot replaced by bundled synphot reference data throughout notebooks and pipeline
- synphot import made lazy in `refdata.py` (not needed for minimal install)
- stpsf import made optional in `psf_model.py` save/load functions
- Demo notebooks find project root by walking up to `pyproject.toml` (works outside pixi)
- README.md overhauled: points to INSTALL.md, fixed API examples, updated project structure
- ipykernel and jupyterlab moved from pip to pixi-only dependencies

### Fixed
- `import roman_disperser` now works with minimal pip install (no synphot/stpsf required)
- PSF model tests pass without stpsf installed

## [0.5.0] - 2026-03-07

### Added
- Star grism image pipeline (`scripts/build_star_grism_image.py`): full-field star simulation from catalog
  - Quick mode (single SCA) and batch mode (YAML config, multiple pointings/SCAs)
  - Per-SCA sensitivity curves applied per order
  - Poisson noise sampling with deterministic JAX RNG key tree
  - FITS output: PRIMARY (metadata) + MODEL (noiseless count-rate) + ISIM (Poisson-sampled counts)
  - Per-SCA quicklook PNGs (asinh stretch, 4× block-averaged)
  - Focal-plane mosaic PNG with all SCAs in WFI layout
  - Per-pointing metadata YAML with RNG keys and per-SCA/order source counts
  - `--force` flag to overwrite existing outputs; skips by default
  - `--mosaic` mode to regenerate mosaic from existing pointing directory
  - `--generate-config` to write a documented template YAML
- Catalog module (`catalog.py`): `select_sources` for per-order detector assignment using trace overlap
- Sky-to-FPA transforms (`optical_model_jax.py`): `get_fpa_pos` and `get_pa_rotation` standalone functions
- Example batch config (`scripts/example_star_config.yaml`)
- Per-SCA sensitivity FITS files and `sensitivity_map.yaml`
- Pipeline documentation (`docs/star_grism_pipeline.md`): output format, config reference, catalog assumptions, architecture

### Performance
- 18 SCAs × 3 orders in ~5 minutes on RTX 4090 (after ~30s one-time JIT warmup)
- Per-SCA I/O optimized to ~0.6s (FITS + PNG write)
- Spectrum generation vectorized across all sources

## [0.4.0] - 2026-03-03

### Added
- Galaxy disperser (`galaxy_disperser.py`): extended source dispersion with Jacobian-based shape warping + PSF convolution
- PSF model (`psf_model.py`): STPSF-based PSF grids with trilinear interpolation, caching
- Star disperser (`star_disperser.py`): wavelength-dependent PSF deposition with memory-efficient chunking
- PSF coordinate utilities (`psf_utils.py`)
- PSF cache generation script (`scripts/generate_psf_caches.py`)
- PSF cache migration script (`scripts/migrate_psf_caches.py`)
- Stars + galaxies GPU demo notebook (`notebooks/galaxy/stars_and_galaxies_gpu_demo.ipynb`)
- Disperser performance profiling notebook (`notebooks/galaxy/profile_dispersers.ipynb`)
- Star dispersion design docs (`docs/star_dispersion.md`, `docs/psf_phase1_plan.md`, `docs/phase2_star_dispersion_plan.md`)
- Galaxy dispersion design doc (`docs/galaxy_dispersion_plan.md`)
- Multi-source `fori_loop` JIT pattern: dynamic `n_sources` argument avoids recompilation
- PSF notebooks: analysis, interpolation validation, all-SCA validation
- Star notebooks: single star demo, multi star demo, GPU run
- Galaxy notebooks: Jacobian exploration
- Grism sensitivity and G0V star spectrum notebooks

### Performance
- Star dispersion: ~3 ms/star/order on RTX 4090 (5501 wavelengths, 2A spacing)
- Galaxy dispersion: ~7 ms/galaxy/order on RTX 4090 (120x120 image, 5501 wavelengths)
- 10K sources x 3 orders: ~5 minutes total execution (excluding one-time ~30s JIT compilation)

## [0.3.3] - 2026-01-12

### Added
- GPU scaling benchmark script (`scripts/benchmark_gpu_scaling.py`) testing performance across:
  - Galaxy counts: 100, 250, 500, 1000
  - Spectral orders: +1, 0, +2
  - Wavelength chunk sizes: 50, 100, 200
- GPU scaling analysis notebook (`notebooks/demos/gpu_scaling_analysis.ipynb`) with presentation-quality visualizations
- Order efficiency scaling factors: Order +1 (100%), Order 0 (2%), Order +2 (1%) for realistic flux ratios
- Benchmark results committed to repo (`scripts/output/`) including combined detector PNG

### Performance
- 1000 galaxies across 3 orders: ~19s total on NVIDIA RTX A5000
- Per-galaxy throughput: ~52 galaxies/second (all orders)
- Peak memory: ~418 MB (well under GPU capacity)

## [0.3.2] - 2026-01-12

### Fixed
- Added `precision='highest'` to all einsum calls for GPU/CPU numerical consistency
- Removed hardcoded `JAX_PLATFORMS="cpu"` from test files to allow GPU testing

### Added
- GPU consistency tests (`test_disperser_gpu.py`) comparing CPU vs GPU results
- GPU verification checklist documentation (`docs/guides/2026-01-11-gpu-verification-checklist.md`)
- GPU support section in README

### Performance
- Verified ~50x speedup on NVIDIA RTX A5000 vs CPU for multi-galaxy dispersion
- JIT compilation provides additional 4-10x speedup on GPU (first vs cached calls)

## [0.3.1] - 2026-01-11

### Fixed
- Fixed image position bug in demo notebooks: disperser expects image box corner (pixel [0,0] center), not source center position
- Added `center_to_corner()` helper to `demo_utils.py` for converting source center to image corner position
- Updated `make_random_galaxy_positions()` to clarify it returns center positions

### Changed
- Improved disperser docstrings to clarify that x0, y0 are pixel [0,0] center positions (FITS 1-indexed), not source centers
- Demo notebooks now correctly compute image corner from galaxy center position

### Added
- Tests for `center_to_corner()` helper in `tests/test_demo_utils.py`

## [0.3.0] - 2026-01-11

### Added
- Demo notebooks for disperser module (`single_galaxy_demo.ipynb`, `multi_galaxy_demo.ipynb`)
- JIT compilation strategy documentation (`docs/jit_compilation.md`)
- JIT compilation demonstration in both demo notebooks using closure pattern
- `make_sloped_spectrum()` function in demo_utils for spectra with edge roll-off

### Changed
- Updated demo notebooks to use `model.detmod["pixel_scale"]` for correct pixel scale access
- Improved visualization layout in single_galaxy_demo with rotated zoomed images
- Demo spectra now use sloped spectrum with 20% edge taper for clearer wavelength visualization

## [0.2.0] - 2026-01-10

### Added
- Disperser module with `disperse_2d1d_sca` and `disperse_galaxies_sequential` functions
- Bilinear scatter-add for flux accumulation
- Wavelength chunking for memory efficiency

## [0.1.0] - 2026-01-09

### Added
- Initial JAX optical model implementation (`optical_model_jax.py`)
- Reference NumPy implementation (`optical_model.py`)
- Coordinate transformations (SCA, FPA, MPA)
- Trace beam functionality for grism spectral tracing

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**STPSF 2.3.0 and exact PSF caches:** PSF caches now target STPSF 2.3.0
(`psf_model.TARGET_STPSF_VERSION`), and a run needs the exact file. **The hydrated
`psf-v1` and `psf-prism-v1` caches (STPSF 2.1.0) are no longer used**: runs need
the STPSF 2.3.0 caches, made with `scripts/generate_psf_caches.py` until they are
released. (The temporary opt-in `ROMAN_DISPERSER_ALLOW_PSF_FALLBACK=1` of earlier
development builds is removed.) Caches made with 2.3.0 change results: the PSF has charge diffusion, 6 mas jitter and the Phase C optical
model.

- Cache names carry the version, `..._os4_stpsf2.3.0.npz`, plus `_ipc` for caches
  with IPC in the PSF, and the node range when it is not the full detector,
  `..._4x4x56_x16-4055_y187-4088_0.90-2.00um_...`.
- `get_or_make_psf_payload` loads the exact file or raises `PSFCacheNotFoundError`
  (a `FileNotFoundError`) with the command that makes it and the other files found.
  An exact-named file whose metadata disagrees (version, `add_ipc`, node positions,
  wavelengths, oversample, FOV, detector, filter) counts as wrong. Files marked
  non-STPSF under the bare configuration name still serve as stand-ins.
- **Removed: implicit generation.** Generation needs `allow_generate=True` or
  `force_regenerate=True`, also with `cache_dir=None`. `make_psf_payload` still
  always generates. `ROMAN_DISPERSER_REQUIRE_PSF_CACHE` is gone: this is the default.
- `--psf-grid NXxNY` and `--psf-grid-bounds XMIN,XMAX,YMIN,YMAX` on
  `generate_psf_caches.py` and `build_dispersed_image.py`, and config keys
  `psf_grid` and `psf_grid_bounds`. The driver checks all its caches exist before
  loading the catalog, and loaded node positions are checked against the request.
  `slurm_generate_psfs.sh` passes its arguments on.
- `add_ipc` on every generation path and `--add-ipc / --no-add-ipc` on
  `generate_psf_caches.py`, default off, stored in the npz and returned as a
  payload label. With `add_ipc=True` the PSF contains IPC. romanisim >= 0.16.0
  applies IPC in `make_l1`, so wrapped images would count it twice. The pinned
  v0.15.0 applies none because of a bug (romanisim #383), so with the default
  wrapped products contain no IPC until romanisim is upgraded.
- `PSFCacheVersionWarning` on load, whether or not STPSF is installed: older than
  the target, no version, filename and metadata disagree, IPC state unknown, or
  differing from the installed STPSF. One warning per file and message per process.
- Loading a cache no longer imports STPSF.
- Generation refuses an STPSF older than the target and an unset `STPSF_PATH`.
- `make_gaussian_psf_cache.py` marks its output `psf_source = gaussian`.
- Caches from parallel generation now record `stpsf_filter` (was `unknown`).

**Results change by default:** the driver now runs the interpolating deposit
with per-order count rebinning (40 / 4 / 2 Å in grism orders 0 / 1 / 2) and
the star PSF hoist (see *Changed*). `--deposit floor --rebin-dlam none
--no-hoist-star-psf` (config `deposit: floor`, `rebin_dlam: none`,
`hoist_star_psf: false`) reproduces v0.16.0 bit for bit. The golden-frame
test passes these floor settings explicitly, so golden-frames-v3 still
apply; **a new bless (golden-frames-v4) with the interp defaults is needed**
before the references can guard the default path.

### Added
- **Scene linear operator** (`roman_disperser.linear_operator.SceneOperator`,
  new module, nothing existing changed): the dispersed image of a fixed set
  of stars and Sersic galaxies on one SCA as a matrix-free linear map of
  their spectra (per-sample or a piecewise-linear basis), with `forward`
  (A), `transpose` (A^T via `jax.linear_transpose`; `transpose_vjp` via
  `jax.vjp`) and `render`
  with differentiable position offsets. Sources are looped with `lax.scan`
  (checkpointed body), so reverse mode works. Forward matches
  `make_batched_*_fori` bit for bit; the adjoint identity holds to f32
  precision. Needs `deposit="interp"` for position gradients.
- **Interpolating deposit** (`deposit="interp"` on `make_star_disperser`,
  `make_galaxy_disperser`, `disperse_star_psf`, `disperse_galaxy`,
  `deposit_stack_native`; `--deposit` / config `deposit` in
  `build_dispersed_image.py`). Two-tap interpolation between adjacent
  phase-binned stamps per axis, combined into one (n+1)² stamp per
  wavelength: the exact sub-pixel box-overlap deposit of the oversampled
  stamp. Removes the ±1/8 px per-wavelength placement quantisation, the
  0.5-2.4 % rms continuum ripple of point sources and the zero position
  gradients; flux-conserving. ~1.7x the floor deposit cost on CPU at
  the same sampling.
- **Count rebinning before the deposit** (`pipeline.rebin_counts`,
  `sed_rebin` on `make_batched_*_fori`, `--rebin-dlam` / config
  `rebin_dlam`). Sums `spectrum*sens*dlam` over groups of samples and
  deposits at the bin centre. The spacing is one value for every order or
  per order (`--rebin-dlam 0:4,1:4,2:2`, config `rebin_dlam: {"0": 4,
  "1": 4, "2": 2}`; `pipeline.sed_rebin_factors`). Interp with 4/4/2 Å costs
  about the same as floor at 2 Å. Keep order 2 at 2 Å: at ~5.4 Å/px a 4 Å
  bin is 0.73 px and blurs narrow lines by up to 5 % of their peak.
- **Star PSF hoist** (`make_star_disperser(hoist_psf=True)`,
  `--hoist-star-psf` / config `hoist_star_psf`): phase-bins the PSF grid
  nodes once per (SCA, order) (`star_disperser.bin_psf_grid`) and blends the
  binned nodes per star. Same result up to f32 rounding; +1 PSF grid of
  device memory per order.
- `star_disperser.bin_stack_phases` / `deposit_binned` (the two halves of
  `deposit_stack_native`), `psf_model.interpolate_psf_spatial(grid_key=)`,
  FITS cards `DEPOSIT` / `DEPDLAM<order>` and meta keys `deposit`,
  `deposit_dlam_angstroms` (per order), `hoist_star_psf`.
- **Reverse-mode autodiff through the production batch loops.**
  `pipeline.run_star_batch` / `run_galaxy_batch` loop over the static batch
  length with a `jax.checkpoint`-ed `lax.scan` instead of a `fori_loop`
  with the traced `n_sources` as trip count (a `while_loop`, which has no
  transpose). `jax.grad` / `jax.vjp` of the image w.r.t. spectra, positions
  (with `deposit="interp"`) and galaxy images now work through them; the
  checkpoint keeps the transpose's memory per source, not per batch. Padded
  rows take the identity branch of a `lax.cond`, so partially filled
  batches stay cheap; the trace positions (and galaxy Jacobians) are
  evaluated outside the `lax.cond`, because inside a conditional branch
  XLA:CPU rounds the trace polynomials differently (pixel differences of
  ~1e-3 of the peak with interp, flipped phases with floor). Forward images
  are bit for bit those of the `fori_loop` (stars and galaxies, floor and
  interp, production PSF); CPU throughput is unchanged and `output` is still
  donated. `disperse_star_psf` takes optional precomputed `dispersed`
  positions, `disperse_galaxy` / `prepare_galaxy_images` optional
  `geometry` (new `galaxy_disperser.galaxy_trace_geometry`) and
  `dispersed`. New `tests/test_batch_autodiff.py`.
- **Differentiable sky -> detector transform** (`optical_model_jax.
  make_sky_reference`, `perturbed_tangent_offsets`, `perturbed_sky_to_fpa`,
  `perturbed_sky_to_sca`; existing functions unchanged). A float64 host
  reference at the nominal pointing (tangent offsets, FPA and, with the
  payload, SCA position) plus a jittable float32 JAX change for source
  offsets (d_xi, d_eta) and pointing offsets (d_ra, d_dec, d_pa), assembled
  from small quantities only, so `jax.grad` / `jax.jacfwd` w.r.t. source
  position and pointing work; optional dispersed trace. Within 1.3e-3 px of
  a float64 evaluation for perturbations up to 10 arcsec and 0.1 deg (the
  existing float32 chain: 1-5e-3 px); Jacobians match float64 finite
  differences to ~3e-7. New `tests/test_sky_perturbation.py`,
  `docs/optical_model.md` section.
- **Per-order brightness cut** (`--order-mag-limits 0:24,2:24` / config
  `order_mag_limits: {"0": 24, "2": 24}`; `pipeline.parse_order_mag_limits`,
  `order_mag_limits_by_order`, `apply_order_mag_limits`, `flux_to_ab_mag`):
  in a limited order only sources with F158 AB mag <= limit are dispersed
  (catalog column `F158`, maggies, stars and galaxies). Applied per SCA after
  the on-detector selection. FITS cards `ORDMAG<order>`, meta key
  `order_mag_limits`, and `stars_mag_cut` / `galaxies_mag_cut` in the
  per-order `source_counts`. Off by default.

### Changed
- **Production deposit defaults** (`build_dispersed_image.py`:
  `DEFAULT_DEPOSIT = "interp"`, `DEFAULT_REBIN_DLAM = {"0": 40, "1": 4,
  "2": 2}`, `DEFAULT_HOIST_STAR_PSF = True`; CLI, config and
  `setup_pipeline` / `build_dispersed_image` defaults are None = these).
  `--rebin-dlam none` (config `rebin_dlam: none`, or `ORDER:none` per order)
  switches rebinning off, `--no-hoist-star-psf` the hoist. Default spacings
  round down to a multiple of the catalog spacing; explicit ones must be
  exact multiples. The library default `deposit` of `make_star_disperser`,
  `make_galaxy_disperser`, `disperse_star_psf`, `disperse_galaxy`,
  `deposit_stack_native`, `deposit_binned` and `run_*_batch` is now
  `"interp"` too; the low-level functions keep `sed_rebin=1` and
  `hoist_psf=False` (rebinning and the hoist are per-run choices of the
  driver). The per-order brightness cut stays off by default.
- **Memory across SCAs** (`build_dispersed_image.process_pointing`): the
  per-SCA MODEL is returned as a host (NumPy) array, the same object the
  mosaic uses, instead of a device array kept per SCA, and batch mode
  (`return_outputs=False`, new) keeps none. Each SCA's PSF payloads and its
  MODEL / ISIM / Poisson device buffers are released before the next SCA
  loads its PSFs. Measured on CPU (3 SCAs, `jax.live_arrays()` after each):
  503 / 566 / 630 MiB before (+64 MiB per SCA), 0.3 MiB after; outputs bit
  for bit unchanged. Quick mode's `build_dispersed_image()` now returns the
  MODEL as a NumPy array.
- **Fewer recompiles off the production path.** The factory dispersers
  (`make_star_disperser`, `make_galaxy_disperser`) cast scalar positions to
  float32 arrays, so Python float, `np.float32`, `np.float64` and
  `jnp.float32` positions share one compile (3 compiles before, 1 now).
  PSF payload `wavelengths` / `wl_grid` / `spatial_x` / `spatial_y` are
  float32 explicitly, so with `jax_enable_x64` on the payload is no longer
  mixed precision (an x64 session still compiles its own programs). The
  driver checks that every PSF payload of a run has the grid geometry of
  the first one (`check_psf_payloads`, `pipeline["psf_geometry"]`) and
  raises a ValueError naming the payload otherwise. New
  `tests/test_recompiles.py`.
- **Star and CPU benchmarks** (`benchmarks/bench_deposit.py`): next to the
  gated floor-galaxy / noscatter pair (unchanged, `gpu.json` format
  unchanged) the bench times `run_star_batch` (47² native stamps) and
  `run_galaxy_batch` per order at the production defaults (`default`) and
  the historical setting (`floor`), as ms per source per order, on CPU or
  GPU with the same command (`--kinds`, `--settings`, `--n-star`,
  `--star-batch`). `check_perf.py` keys CPU results by the CPU model and
  gates those rows against `baselines/cpu.json` (new; laptop i7-9750H) or a
  GPU entry's `ms_per_src`; new `benchmarks/compare.py` prints CPU vs GPU
  side by side. `benchmarks/README.md` documents the GPU-node run.
- Deposit chunk length is derived from the number of wavelengths
  (`ceil(N / ceil(N / chunk_size))`), so zero-flux padding is < 1 chunk
  count instead of up to a whole chunk (5501 -> 3 x 1834; the 4 Å rebinned
  2751 -> 2 x 1376 instead of 2 x 2000). Bit-identical output.
- `.payload_args` of the factory dispersers is now
  `(arrays, static, chunk_size, deposit)`.

### Fixed (cluster tests, 2026-11-07)

- Batch configs accept `segment_render`, `segment_psf_at` and `accumulate`
  (same values as the CLI flags); before, they were rejected as unknown keys.
  `--warmup-only` now checks the config keys too.
- `--segment-render auto` / `--accumulate auto` pick the fastest measured
  setting per backend: on GPU the production deposit and loop (7.7 s on an
  L40S for BDI SCA 5, against 21-24 s for the former auto choice, 56-node
  segments with windows); on CPU the 56-node segment render with the loop, as
  before. `auto` asks for speed; pass `window` or `tiled` for a GPU frame sum
  that is the same from run to run.

## [0.16.0] - 2026-10-27

**Results change: pixels differ from v0.15.x runs** (see *Changed*), so the
golden references move to `golden-frames-v3`. The release brings the HEALPix
galaxy-SED store (70 → 29 min per pointing from SED reads), payloads as jit
arguments (29 → 19.5 min from compile time), the A10G dense-SCA OOM fix,
declarative `source_cuts`, and an environment-only change to the
`romanisim` pixi env. Catalogs in the existing layout read exactly as before.

### Added
- **Declarative `source_cuts` in the driver config** (`docs/grism_pipeline.md`).
  Rules `{column, op, value, types}` select catalog rows right after loading
  (ops `lt le gt ge between isin mag_le mag_ge`; mag ops for flux columns in
  maggies). One catalog store then serves every depth, replacing hand-made
  filtered copies of `metadata.parquet`. Recorded in FITS (`SRCCUTN`,
  `SRCCUT1..n`) and the meta YAML (`catalog_rows`, `source_cuts` with per-rule
  removed counts). Reproduces the XMM-LSS `filtered.metadata.parquet` to 4
  boundary rows of 12.8M.
- **HEALPix galaxy-SED store** ([`docs/healpix_sed_store.md`](docs/healpix_sed_store.md)).
  - `scripts/healpix_repartition_seds.py` (`plan` / `fill` / `verify`) and
    the SLURM driver `scripts/slurm_healpix_repartition.sh` rewrite a
    catalog's galaxy SEDs into one single-shard array per NESTED HEALPix
    pixel (`galaxy_seds/hp_<uniq>`).
  - Pixels are split adaptively to ~0.5 GB compressed. The output is a new
    sibling catalog whose `metadata.parquet` gains `hp_uniq` / `hp_row`.
  - `verify` checks every pixel and compares random rows bitwise against the
    source. The driver's verify job then also runs `verify_source_catalog.py`
    on the new catalog.
  - New pixi dependency: `astropy-healpix` (used by the afterburner only).
- **`verify_source_catalog.py` reads either layout** and gains
  `--skip-provenance`. RA-padded catalogs such as `catalogs_padded` fail
  provenance by construction, because the pad shift is not recorded.
- **`build_dispersed_image.py` reads either layout**, detected by the
  `galaxy_seds` attribute `layout` (absent means `sim`, as before).
  - For a HEALPix store it reads each needed pixel whole.
  - On the acceptance catalog (4-roll VISIT 1, A10G) galaxy-SED reads fell
    from 2493 s to 44 s per pointing, and the pointing from 70 to 29 min.
  - `MODEL` is unchanged within the GPU golden tolerance.

### Fixed
- **GPU OOM on dense SCAs (A10G).** The driver built every galaxy's Sersic
  image for an SCA at once with `sersic.make_sersic_images`, an un-jitted
  `vmap` that materializes each intermediate at full `[N, 120, 120]` size
  (~2.1 GB per array at N = 36.5k). XMM-LSS SCAs with 34–37k galaxies OOMed
  at the final divide (`jit_true_divide`) on a 24 GB card. New
  `pipeline.disperse_batched_sersic_galaxies` generates each dispersion
  batch's images on the fly, so GPU memory scales with `galaxy_batch_size`
  (XMM-LSS 18-SCA pointing on A10G: peak 12 GiB, OOM at SCA 9 → 4.3 GiB,
  completes). The per-batch generator is jitted: images change at
  float32-rounding level (golden scenes: max |Δ| ≤ 5e-8 of frame max, sums
  unchanged), folded into the golden-v3 re-bless. The golden renderer uses
  the same path. `disperse_batched_galaxies` is unchanged.
- **String `sim` values in galaxy-SED catalogs.** Newer upstream catalogs
  (e.g. `20260917-xmm_lss`) name each partition by its Galacticus file
  (`sim = "romanUNIT-d601_625_16sqDeg.hdf5"`, array
  `galaxy_seds/sim_<name>`). The driver, `verify_source_catalog.py` and the
  HEALPix afterburner formatted keys as `sim_{int:03d}` and raised on these.
  Integer sims still map to zero-padded `sim_NNN`, so existing catalogs
  read exactly as before.

### Changed
- **Payloads are jit arguments: one compile serves every SCA. Pixels differ
  from earlier runs.** `jit_payload` splits each PSF/optical payload into
  traced arrays plus a whitelisted static tuple (`docs/jit_compilation.md`).
  New jitted entry points: `disperse_star_jit`, `disperse_galaxy_jit`,
  `run_star_batch`, `run_galaxy_batch`. The factory signatures are unchanged.
  - A10G, acceptance 4-roll VISIT 1, HEALPix store, empty JAX cache: the
    pointing went from 1741 s to 1173 s, and JIT from ~600 s to ~17 s.
    `check_perf.py` passed in a single run.
  - Why pixels move: traced arrays round differently in float32 from baked-in
    constants. Trace positions shift by ≤ 2.4e-4 px, which flips edge samples
    between neighbouring pixels. Per-SCA `MODEL` flux agrees to ≤ 6e-7. The
    per-pixel max difference is 1.3e-3 to 3.8e-2 of peak. The differences
    are local and zero-sum on golden grism-0; they are not proven to be
    boundary flips pixel by pixel everywhere. Forcing the old path was
    bit-exact.
- **`pipeline.make_batched_star_fori` / `make_batched_galaxy_fori` now
  require a factory-built disperser** (one carrying `.payload_args`) and
  raise `TypeError` otherwise. They used to fall back silently to capturing
  the disperser in a closure, which was correct but cost one compile per SCA
  (~30–70 s each on an A10G). Every in-repo and tutorial caller already
  passes factory dispersers. To loop a custom disperser, write the
  `jax.jit(fori_loop)` yourself, as `benchmarks/bench_deposit.py` now does
  for its noscatter reference.
- **jax/jaxlib now `>=0.11.2,<0.12`** (was `>=0.7,!=0.11.0`), and the
  lock moves from 0.7.2 to conda-forge 0.11.2. This is a hard floor, so
  every install matches the golden-v3 reference products. On dense
  production frames, jax 0.7.2 and 0.11.2 differ by boundary-pixel flips.
  On the coarse golden scenes (CPU) they differ only at float32 rounding
  (≤ 1.9e-7 of peak). The floor also rules out 0.10.x: conda-forge 0.10.2
  ran the galaxy deposit ~2.2x slower on A10G (8.0/5.2/2.7 ms/gal, orders
  0/1/2) and segfaulted compiling the CPU golden tests. 0.12 is capped
  until it has been tested.
- **`check_perf.py` ratio gate tightened from 3.0 to 1.5.** jax 0.10.2's
  ratio of 1.9–2.8 passed the old gate; only the absolute A10G gate caught
  it. Healthy GPU runs measure 0.73–0.77.
- **Golden references re-blessed as `golden-frames-v3`** (code 7571abd,
  conda-forge jax 0.11.2; `roman_disperser_data` release `golden-frames-v3`,
  manifest updated). Run `pixi run hydrate --update` to fetch them. Compared
  with v2:
  - total flux moves by ≤ 2.3e-8 (CPU) and ≤ 3e-9 (a10g);
  - the per-pixel changes come from the jit-argument change above. They are
    up to 3.2e-2 of peak (a10g prism coarse), and in every frame the per-row
    net changes cancel, so flux moves between rows.

  The frames are valid for jax 0.7.2 and 0.11.2. `hydrate.py`'s built-in
  fallback manifest points at v3.
- **Gate scripts.** `benchmarks/golden-jax-versions.txt` is now `0.11.2` +
  `latest`. `latest` resolves inside the `<0.12` cap, and the pip sweep no
  longer covers the conda-forge pixi build, which is benched directly.
  `scripts/slurm_run_tests.sh` now runs the suite and the golden full tier
  independently, and fails if either fails.

### Changed (`romanisim` env only)

Confined to the `romanisim` pixi env, which wraps disperser output to L2.
The disperser code and the `default`/`cuda` envs are untouched.

- **`romanisim` env: upstream romanisim 0.15.0 replaces the
  `roman-grs-pit/romanisim@extra_counts` fork.** The fork's patches are
  upstream as of romanisim 0.15.0: named-HDU `--extra-counts` and
  GRISM/PRISM bandpasses. romanisim versions independently of this package,
  so the matching 0.15 is a coincidence.
  - It is pinned to the 0.15.0 release, **not** main. romanisim PR #388
    (2026-09-11) added `epsf` to the default CRDS reference set. That
    reference is fetched whenever `--usecrds` is set, regardless of
    `--psftype`. The epsf rmap has no GRISM/PRISM entry, so every
    spectroscopic wrap fails with `CrdsLookupError`.
  - Release 0.15.0 lacks romanisim's later IPC (#384), CRDS-pedestal (#401)
    and flight `wfi_cen` (#406) changes.
- **`roman_datamodels` pinned to 1.1.x, which brings in rad 1.1.0** (was
  0.31.0). CRDS contexts from `roman_0072.pmap` on serve 2.0.0-tagged
  reference schemas: readnoise, gain, linearity, saturation, mask, …. rad
  < 1.0 cannot deserialize these (`Unknown datamodel type TaggedDict`).
  Bump rad/rdm and the CRDS context together. L2 files wrapped with this env
  are not bit-comparable to earlier wraps (fork + rdm 0.31 + CRDS 0058).

### Notes
- romanisim (0.15.0 and main alike) looks up the MA table with the CRDS
  server's *operational* context and ignores `CRDS_CONTEXT`. To honour a
  pinned non-operational context, run the wrap with
  `CRDS_SERVER_URL=https://127.0.0.1:9 CRDS_MODE=local CRDS_READONLY_CACHE=1`
  against a hydrated cache.
- First production use: the 4-roll RA 10 / Dec 0 GRISM rerun (2026-10-07),
  with CRDS `roman_0072`, 72/72 L2 files verified.

## [0.15.1] - 2026-10-05

Documentation-only follow-up to the post-merge review of the native deposit
(PR #35). No code or results change; golden references stay
`golden-frames-v2`.

### Fixed
- **`chunk_size` docstrings** (`deposit_stack_native`, `disperse_star_psf`,
  `make_star_disperser`, `disperse_galaxy`, `make_galaxy_disperser`): the
  claim that chunk 2000 was "measured best" (500/1000 being 17%/5% slower)
  came from the pre-FFT-fix prototype. On the shipped code chunk 500 and
  2000 measured identical on a10g (SLURM 7183); 2000 is kept because
  per-chunk memory at native resolution is ~4x below the old default's.
  `make_star_disperser` also still documented the old default (1000) and
  the old oversampled memory estimate.
- **Native-deposit accuracy statement** (0.15.0 entry below,
  `tests/golden_frame.py`): "lands ~1e-8 from truth" was the 20 Å figure
  only. Measured on an isolated on-detector star, prism order 1, relative
  to a float64 truth: native +9e-9 (20 Å) and −1.4e-7 (production 2 Å);
  old deposit −3.2e-7 and −1.5e-5.
- **GPU run-to-run floor** in the golden-frame docstrings: "≤6e-7 of peak"
  was a single pair; measured 2e-7–8e-7 over two a10g renders (this
  changelog already said ≲8e-7).

### Changed
- **Comments in `deposit_stack_native`** describing its two stages (bin the
  stamps once per source for all 16 phases; then per fine wavelength select
  phase, interpolate, scale, scatter-add) and noting that `bin_phase`
  operates on the closed-over `stack`, i.e. the stamps themselves.

## [0.15.0] - 2026-08-25

Results-changing at the total-flux level (for the better — see the native
deposit entry); golden references are `golden-frames-v2`. Roughly 4x faster
end-to-end on an A10G and 6-7x on CPU, flat across orders.

### Changed
- **Native-resolution deposit (16-phase pre-binning) in both dispersers**
  (`star_disperser.deposit_stack_native`, shared by `disperse_galaxy` and
  `disperse_star_psf`; issue #30). The per-subpixel oversampled scatter —
  ~92k elements per wavelength per galaxy — is replaced by pre-binning the
  stamp stack to native detector pixels in the 16 boundary-phase variants
  (once per source) and depositing ~16x fewer elements. The additions are
  identical, merely regrouped (run-length-4 blocking of the floor deposit;
  verified exactly in float64 on 2,000 random centers), so outputs are
  equivalent up to f32 summation order. Wavelength chunk defaults change
  from 500 (galaxy) / 1000 (star) to 2000 — no faster than 500 on the
  shipped code, but affordable: per-chunk memory at native resolution is
  ~4x below the old default's. The `rel_x`/`rel_y` arguments of
  `disperse_star_psf` are now unused (kept for backward compatibility).
  **This is a results-changing release at the total-flux level, for the
  better**: the old deposit's deep sequential f32 accumulation was
  systematically low by ~3e-7 (coarse λ sampling) up to ~1e-5 (order 0 at
  production 2 Å, ~59k adds/px), verified against a float64 truth
  computation — the native deposit lands ~1e-8 from truth at 20 Å and
  ~1e-7 at production (2 Å) prism sampling. Golden
  references bumped to `golden-frames-v2` accordingly; per-pixel
  differences stay within the tight gate.
- **Galaxy PSF convolution now transforms at a fast FFT size**
  (`galaxy_disperser.fft_convolve_full`, replacing the vmapped
  `jax.scipy.signal.fftconvolve` in `prepare_galaxy_images`). The full
  linear conv support is exactly X = stamp + PSF − 1, so a circular
  convolution at the next 7-smooth size ≥ X (chosen by rule in
  `_next_fast_fft_size`, not hard-coded — the optimum is
  hardware/cuFFT-version dependent), cropped to [:X, :X], is the same
  answer up to float rounding (measured max |diff| ~6e-9 vs
  `fftconvolve`; both golden-frame tiers pass unchanged). At production
  geometry X = 303 = 3·101 forced cuFFT onto its Bluestein path;
  transforming at 315 = 3²·5·7 measured ~1.7x faster for the convolution
  and ~1.6x for fused `prepare_galaxy_images` on an A10G (2026-08-19
  workbench, SLURM 7145). Applies to both grism and prism galaxy paths;
  the star path has no per-source FFT and is unaffected.

### Added
- **Golden-frame end-to-end regression test** (`tests/golden_frame.py`,
  `tests/test_golden_frame.py`). A small curated scene (5 galaxies + 2 stars
  covering detector-center, edge-clipped, off-grid sub-pixel, overlapping-pair,
  and star code paths) is rendered through the *production* dispersal path on
  real reference data (SCA 1) and compared full-frame against pinned reference
  frames. References are **per-backend** (`cpu/`, `gpu-a10g/`): the tight
  gate (per-pixel rtol 1e-5 with frame-scaled atol; total flux to 1e-7 in
  float64) applies against the matching backend's blessed frames, because
  cross-backend per-pixel comparison is ill-conditioned — measured
  CPU-vs-GPU differences reach 3.2e-2 of frame peak from benign,
  flux-conserving position-rounding boundary flips (runs of λ samples
  within float32 ULPs of a pixel boundary landing one row apart), while
  total flux agrees to ~3e-8 and GPU run-to-run repeats stay ≲8e-7 of peak
  (the issue #22 floor; measured 2e-7–8e-7 over two a10g renders). A GPU model without blessed frames falls back to
  the CPU reference at a loose gross-breakage gate (0.1·peak) plus the
  always-on flux gate. Two tiers: *coarse*
  (20 Å, grism orders 0/1/2 + prism, default suite) and *full* (production
  2 Å sampling, grism orders 1 and 0 + prism, `-m slow`) — the full tier
  exists because per-pixel accumulation depth and the deposit-collision
  regime only appear at production sampling. Reference frames are a vendored
  `roman_disperser_data` asset (`golden-frames-v2` at this release; `-v1`
  is the pre-native-deposit bless, fetched by `pixi run hydrate`), version-pinned in the test so intentional
  results-changing PRs must bump the pin and publish regenerated frames in
  the same change. This is the durable guard for refactorings of the
  dispersal internals (FFT and native-deposit work), replacing throwaway
  old-vs-new workbench comparisons.
- **Merge-gate convention recorded in CLAUDE.md**: every PR merge requires
  the usual suite on CPU and GPU, plus the full-tier golden test and the
  `benchmarks/` performance regression suite. `scripts/slurm_run_tests.sh`
  now runs the golden full tier after the suite, so it is the whole GPU
  half of the gate; its partition/GRES/log dir are env-overridable.

## [0.14.3] - 2026-08-21

### Added
- **`benchmarks/` — performance regression suite.** `bench_deposit.py` times
  the production galaxy disperser (ms/galaxy per grism order, seeded
  synthetic workload at production geometry) against a no-scatter compute
  floor; `check_perf.py` gates on the hardware-insensitive baseline/noscatter
  ratio (≤3; healthy ~1.1–1.2, the jax 0.11.0 scatter regression measured
  15–21) and, where a `baselines/gpu.json` entry exists, on absolute ms/gal
  (≤1.5× reference). `run_golden.sh` sweeps a small set of "golden" jax
  versions (`golden-jax-versions.txt`: the pixi.lock version, 0.11.1, PyPI
  latest) in throwaway venvs. Run on a GPU node before releases and after
  deposit-path or jax-floor changes; not in CI.

### Changed
- **jax 0.11.0 is now excluded** (`jax>=0.7,!=0.11.0` plus `jaxlib!=0.11.0`
  in `pyproject.toml` and `pixi.toml`). That release has a GPU scatter-add
  regression that slows the disperser's deposit step by ~16x on GPU
  (measured 157/177/125 ms/galaxy for grism orders 0/1/2 on an A10G vs the
  usual ~9 ms flat; a no-scatter variant of the same run stays at 8.6 ms,
  isolating the deposit). jax 0.11.1 on the same hardware is fast again
  (9.4/10.4/9.2 ms/galaxy). Diagnosed from a user environment pinned to
  0.11.0 via PyPI; the project's own pixi lock (jax 0.7.2, conda-forge) was
  never affected.
- **Import-time `RuntimeWarning` when running under jax 0.11.0** — the
  dependency exclusion cannot reach environments that already have 0.11.0
  installed, so the package warns (with the upgrade command) instead of
  silently running ~16x slower.

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

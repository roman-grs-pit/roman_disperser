# Migrating to v0.14 (from v0.10)

Four releases separate v0.10.0 (2026-06-12) and v0.14.0 (2026-08-05). Two of
them change **simulated results**, not just APIs — if you have products or
downstream analyses built on v0.10-era output, read the "Science changes"
section even if you never touch the Python API. The full record is in
`CHANGELOG.md`; this guide is the practical summary of what breaks, what
changes silently, and what to do about each.

Quick orientation:

| Release | One-line summary | Results change? |
|---|---|---|
| v0.11.0 | GPU TF32 + float32-RA placement fixes | **Yes — GPU products** |
| v0.12.0 | Gnomonic (TAN) sky→FPA projection | **Yes — off-equator pointings** |
| v0.13.0 | Per-SCA RNG keys; provenance header cards | ISIM noise only |
| v0.13.1 | Line-centering validation harness | No |
| v0.14.0 | Prism support (both WFI dispersing elements) | No (grism unchanged) |

## Science changes: results differ from v0.10

These are corrections, not regressions — v0.14 places sources more accurately
than v0.10 — but any comparison between old and new products must account for
them.

1. **GPU source placement (v0.11.0).** On Ampere-class and later GPUs, the
   sky→FPA position-angle rotation was silently evaluated in TF32: every
   source in a GPU-generated product was displaced by a median 1.84 px (up to
   ~7 px, growing with field radius). CPU products were unaffected. All
   matmul-class ops now pin `precision='highest'`, and an AST-scan test
   (`tests/test_precision_convention.py`) keeps it that way.
   A second fix in the same release: absolute RA was quantized to float32
   before differencing, a pointing-dependent error (0.006 px at RA 10°, 0.40
   px at RA 260°). Now float64 on the host at every RA.

2. **Flat-sky → gnomonic projection (v0.12.0).** v0.10 used the flat-sky
   approximation (Δα·cos δ, Δδ), whose dominant error grows quadratically
   with field radius and with tan δ₀. Over a ±0.4° field: ≲0.1 px at Dec 0,
   ~7 px median at Dec 30, ~20 px median at Dec 60. v0.12+ uses an exact
   tangent-plane (TAN) projection, derived and pinned bit-exact in
   `docs/reference/tangent_plane_derivation.ipynb`. Fields crossing RA = 0
   now also work (the projection is periodic in RA; v0.10–0.11 silently
   dropped such sources off the focal plane).

3. **ISIM noise realisations (v0.13.0).** Per-SCA Poisson keys are now folded
   from the SCA *number* rather than split by list position, so a
   `scas: [5]` run reproduces the same SCA-5 noise as a full 18-SCA run.
   Consequence: **every ISIM realisation differs from ≤0.12 by
   construction**; the noiseless MODEL extension is unaffected. Old products
   remain reconstructible from their `RNDSEED0`/`RNDSEED1` header cards via
   `jax.random.wrap_key_data`.

**What to do:** don't diff v0.14 products against v0.10-era products and
expect agreement. Positions move by the sizes above (dominated by items 1–2,
depending on backend and declination); ISIM noise differs everywhere. If you
must identify which era a FITS product is from, v0.13+ products carry
`CODEVER` and `GITSHA` header cards; older ones don't.

## API changes: what breaks loudly

1. **`pipeline.ORDERS`, `pipeline.LAM_MIN`, `pipeline.LAM_MAX` are gone**
   (v0.14.0). Orders and band are per-element now.

   Before:

   ```python
   from roman_disperser.pipeline import ORDERS, LAM_MIN, LAM_MAX
   ```

   After:

   ```python
   from roman_disperser.elements import GRISM

   ORDERS = GRISM.orders                        # ("0", "1", "2")
   LAM_MIN, LAM_MAX = GRISM.lam_min, GRISM.lam_max   # 0.9, 2.0 um
   ```

   `pipeline.load_sensitivities(...)` and `select_sources_per_order(...)`
   correspondingly take `orders` (and band) as explicit arguments, and
   `pipeline.resolve_paths(...)` takes `element=` to pick per-element data
   paths.

2. **`get_fpa_pos` demands float64 host input and is no longer
   jit-compilable** (v0.11–0.12). Passing float32 (including any JAX array
   with x64 disabled — i.e. `jnp.array(ra)`) raises `TypeError`, because RA
   quantized to float32 cannot be un-quantized downstream. Pass float64 NumPy
   arrays (pandas columns are already float64). If you were jitting it, jit
   `get_fpa_pos_from_offsets` (the JAX half) and call
   `sky_to_tangent_offsets` (the float64 host half) outside.

3. **`psf_model` selects PSFs by STPSF filter, not by hardcoded order maps**
   (v0.14.0). This only affects you if you *generate* PSF caches yourself or
   load them with non-grism orders — ordinary use (loading the vendored grism
   caches with `order="0"`/`"1"`/`"2"`) is unchanged, because the grism
   default mapping is applied automatically. What changed: code that relied
   on unknown orders silently mapping to `ORDER<n>` cache filenames now gets
   an exception, and prism caches are selected by passing
   `stpsf_filter=element.stpsf_filters[order]`.

4. **Meridian/float32 guards.** Code paths that used to produce silently
   wrong answers now raise: float32 RA into `sky_to_tangent_offsets`
   (`TypeError`), element/optical-model mismatches
   (`elements.validate_against_model`, `ValueError`), a prism run against a
   grism-band catalog (`validate_catalog`, `ValueError`).

## Renames and new knobs (back-compatible)

- **`scripts/build_grism_image.py` → `scripts/build_dispersed_image.py`**
  (v0.14.0). The old name still works — it forwards with a `FutureWarning`
  and will be removed in a later release. Update drivers at your leisure,
  but do update them.
- **Element selection** (v0.14.0): `element: prism` in the config YAML or
  `--element prism` on the CLI. Grism remains the default everywhere;
  selection is explicit-only (no environment variable, deliberately). See
  `docs/element_support.md` for which scripts support which element.
- **Output naming** (v0.14.0): product filename prefix defaults to the
  element name (`grism_*` / `prism_*`); pin `output_prefix:` in the config
  to override. Products self-identify via the `OPTELEM` header card, which
  `wrap_with_romanisim.py` now reads per file to choose the romanisim
  bandpass.
- **Catalog builder default grid** (v0.14.0): `build_source_catalog.py` now
  builds 7500–21000 Å (a superset serving both elements) instead of
  9000–21000 Å. `--wl-min 9000` reproduces the grism-era `catalog-v2` grid
  exactly.

## Reference data

Hydration (`pixi run hydrate` / `roman-disperser-hydrate`) works as in v0.10;
the manifest gained three prism assets (`optical_model_prism`,
`sensitivities_prism`, `psf_prism`), so a full hydrate is now ~6.4 GB (was
~4.5 GB). Grism-only work can skip them:
`--only optical_model,sensitivities,synphot,psf,catalog`.

One caveat: the vendored `catalog-v2` is still on the grism 9000 Å grid, so
**prism runs cannot use it** — the pipeline raises rather than dispersing a
spectrum with no blue-end SED support. Until a `catalog-v3` release exists,
prism runs must point `catalog_dir` at a 7500 Å-floor catalog.

Since v0.14.2 the data directory's *role* also changes: the optical-model
delivery loaded at run time is resolved from the data dir's
`data-versions.lock` (written by hydrate), not from a filename baked into the
code. Two consequences for old setups:

- A data dir with **no lock entry** for the optical model — hydrated before
  the lock existed, or assembled by hand — now fails loudly
  (`FileNotFoundError` listing the candidate files) even though the YAML is
  sitting right there: undeclared files are never adopted. Re-hydrate (which
  records the version), or declare one explicitly via the
  `optical_model_version:` config key / `--optical-model-version` flag.
- The lock **pins** your data: upstream publishing a new delivery changes
  nothing until you re-run hydrate. Re-running is cheap — up-to-date assets
  are skipped, and lock and contents are updated together.

## Checklist

For a v0.10-era driver or analysis moving to v0.14:

1. Replace `pipeline.ORDERS`/`LAM_MIN`/`LAM_MAX` imports with
   `elements.GRISM` fields (loud `ImportError` if you miss one).
2. Make sure nothing wraps RA/Dec in `jnp.array()` before `get_fpa_pos`
   (loud `TypeError` if it does).
3. Rename `build_grism_image.py` invocations to `build_dispersed_image.py`
   (or accept the `FutureWarning` for now).
4. Re-hydrate reference data; re-run your own regression baselines rather
   than comparing against pre-v0.13 products (positions and ISIM noise both
   moved, for the reasons above).
5. If you consume products from mixed code versions, key your comparisons on
   the `CODEVER`/`GITSHA` cards (present since v0.13.0).

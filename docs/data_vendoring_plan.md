# Vendored reference-data plan

**Status:** planning (branch `feature/vendored-refdata`)
**Goal:** make `roman_disperser` fully usable from a plain `pip install` (no repo
checkout) by treating **all** reference data as independently-versioned,
vendored assets fetched on demand — never bundled in the wheel, never assumed
to sit in a repo checkout.

## Motivation

Today every data path resolves relative to a repo checkout
(`PIXI_PROJECT_ROOT/data/...`), and none of `data/` ships in the wheel. So a
`pip install` from git yields the code but **no runtime data** — not even the
28 KB optical model. The optical model and sensitivity curves also evolve on
their **own cadence** (new calibrations) independent of the code, so binding
them to the wheel (package data) would be wrong: it would force a code release
for every calibration update and stop users from updating data without
updating code.

**Decision:** all reference data is *vendored* — the same treatment the PSF
caches and catalogs already get. Nothing is package data.

### Consequence (accepted)

A `pip install` alone never yields a runnable disperser; you always hydrate
first, even for tiny assets. This is already true for PSF caches and is the
price of independent versioning. Acceptable for a research tool.

## Data classes (all vendored)

| Asset | Size | Today | Release (to create) |
|-------|------|-------|---------------------|
| optical model (`Roman_grism_OpticalModel_v0.8.yaml`) | 28 KB | committed in `data/` | `optical-model-*` |
| sensitivities (`data/sensitivities/`) | 1.9 MB | committed in `data/` | `sensitivities-*` |
| synphot (`data/synphot/`) | 88 KB | committed, resolved oddly in `refdata.py` | `synphot-*` |
| PSF caches (`data/psf_cache/`) | 4.1 GB | release `psf-v1` | `psf-*` (exists) |
| catalogs (`data/catalogs/`) | 155 MB | release `catalog-v2` | `catalog-*` (exists) |

`data/stars/` (12 MB) is a **catalog-build input only** (used by
`build_source_catalog.py`), not a runtime asset — left as-is, not vendored for
runtime.

Filenames keep their versions (e.g. `…_v0.8.yaml`) — the optical model is
planned to be vendored on its own track and may change separately.

## Versioning model — manifest + lock (like `pixi.toml` / `pixi.lock`)

The crux of "data changes without a code release": version selection lives with
the **data**, not the code.

- **Remote manifest** (a file in `roman_disperser_data`) names the current
  blessed version of each asset. The data maintainer bumps it when a new
  calibration is published — no `roman_disperser` release. Git history of the
  manifest gives historical states for free.
- **Lock** (`data-versions.lock`, a small file a consumer commits) pins exact
  versions. The lock *is* the reproducibility pin; historical reproduction does
  not depend on the data repo retaining old manifests.

Hydrator resolution order:
1. `--lock <file>` if given → install exactly those versions.
2. else fetch the remote manifest (latest blessed) → install those.
3. **always** write the resolved `data-versions.lock` it installed (like
   `pixi.lock` materializing from a solve).

Tutorials commit a `data-versions.lock` so a reader reconstructs the exact data
the tutorial was authored against.

## Path resolution

Single helper `data_dir()` with layered resolution:
`explicit arg → ROMAN_DISPERSER_DATA → PIXI_PROJECT_ROOT/data (back-compat) →
./data`.

Pixi/dev is unchanged: `PIXI_PROJECT_ROOT/data` *is* the data dir, so the
runtime looks in `data/` exactly as today regardless of how the files arrived
(committed, `pixi run hydrate`, or `roman-disperser-hydrate`). Pixi and pip
workflows converge.

## Plan (phased; order matters)

### Phase 1 — Unify + neutralize resolution
- Add `data_dir()` (layered, neutral env var, `PIXI_PROJECT_ROOT` fallback).
- Rewrite `resolve_paths()` (`pipeline.py:44-71`) so optical model,
  sensitivities, catalogs, psf_cache all resolve under `data_dir()`.
- Collapse the duplicate optical-model resolver at `pipeline.py:546-548`.
- Fold `refdata.py:13` synphot resolution (`Path(__file__)/../../../data/synphot`)
  into `data_dir()`.

### Phase 2 — Packaged hydrator + console entry point
- New `src/roman_disperser/hydrate.py` consolidating
  `scripts/download_psf_caches.py` + `download_source_catalog.py`, extended to
  all vendored assets. Manifest + lock logic lives here; release tags are NOT
  hardcoded as the source of truth (the manifest is).
- `[project.scripts] roman-disperser-hydrate = "roman_disperser.hydrate:main"`
  — on `PATH` in any venv/conda, no pixi.
- Flags: `--lock`, `--manifest <ver>`, `--dest` (default `data_dir()`),
  `--force`, `--sca` subset; idempotent skip-if-present; writes
  `data-versions.lock`.
- Hydrator PSF default = **all 36**; subsetting (`--sca`) is the caller's job
  (tutorials select a subset).
- Repoint `scripts/download_*.py` to thin wrappers around the package function
  (back-compat).
- Keep a `pixi run hydrate` task as a maintainer alias only.

### Phase 3 — Publish essentials as releases (data side, `roman_disperser_data`)
- Upload current optical model, sensitivities, synphot as their own tagged
  releases.
- Create the initial remote manifest naming the current blessed versions of
  all five assets.

### Phase 4 — Remove committed data from the repo (the clean break)
- **Only after** Phases 2–3 are verified working.
- Delete the committed optical model, `sensitivities/`, `synphot/` from `data/`;
  gitignore `data/` entirely (as `psf_cache`/`catalogs` already are).
- Repo becomes **code-only**; all production data is vendored + hydrated.
- Tests run locally against an already-hydrated `data/` (no fixtures, no CI
  hydration in scope — testing is local).
- Update `conftest.py:17`, `test_optical_model_jax.py`, `test_disperser_gpu.py`,
  `test_catalog.py`, and the `test_sersic.py:284` hardcode to resolve via
  `resolve_paths()`/`data_dir()`.

### Phase 5 — Docs + release
- `INSTALL.md`: new flow — `pip install` → `roman-disperser-hydrate`; document
  `ROMAN_DISPERSER_DATA`, manifest/lock; demote `scripts/download_*`.
- `CLAUDE.md`: vendored layout, `data_dir()` resolution, entry point, env var,
  manifest/lock model. Update the `Model config: data/...` note.
- Release per project process: branch → bump version → `pixi install` →
  `CHANGELOG.md` → tag.

## Call sites to touch (from the code map)

- `pipeline.py:59,61,63,66,69` — `resolve_paths()` body.
- `pipeline.py:546-548` — duplicate optical-model resolver.
- `refdata.py:13` — synphot `_DATA_DIR`.
- `psf_model.py` `get_or_make_psf_payload()` — already param-driven (`cache_dir`).
- `scripts/download_psf_caches.py:32`, `download_source_catalog.py:34` — become
  wrappers.
- Tests: `conftest.py:17`, `test_optical_model_jax.py:24`,
  `test_disperser_gpu.py:59`, `test_catalog.py:21`, `test_sersic.py:284`.
- `pyproject.toml` — add `[project.scripts]`.

## Decisions locked

- All data vendored; no package data.
- Filenames keep version suffixes.
- Versioning: manifest (latest) + lock (pin), hydrator writes the lock.
- Env var `ROMAN_DISPERSER_DATA`, `PIXI_PROJECT_ROOT` kept as fallback.
- Hydrator default = all PSF SCAs; `--sca` to subset.
- `scripts/download_*.py` kept as back-compat wrappers.
- synphot vendored too (consistency); `refdata.get_f158_band()` etc. require
  hydration.
- `data/stars/` stays a build-only input.
- Remove committed data from the repo after the hydrator lands (Phase 4).
- Tests are local against a hydrated `data/`; no fixtures / CI hydration.

## Open / deferred

- Exact manifest format + location in `roman_disperser_data` (a release asset
  vs a tracked file) — settle when Phase 3 starts.
- Whether the "separate optical-model vendoring" project supersedes the
  `optical-model-*` release track created here.

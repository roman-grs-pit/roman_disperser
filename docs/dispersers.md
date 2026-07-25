# Dispersing elements: the mode abstraction

Roman's WFI carries two dispersing elements, and `roman_disperser` supports
both:

| | **grism** (G150) | **prism** (P127) |
|---|---|---|
| band | 0.9 – 2.0 µm | 0.75 – 1.85 µm |
| orders rendered | `("0", "1", "2")` | `("1",)` — one beam |
| wavelength transform | `linear` about 1.55 µm | `log` about 1.20 µm |
| STPSF filters | `GRISM0`, `GRISM1` | `PRISM` |
| optical model | `Roman_grism_OpticalModel_v0.8.yaml` | `Roman_prism_OpticalModel_v0.8.yaml` |
| sensitivities | `<data>/sensitivities/` | `<data>/sensitivities_prism/` |
| romanisim bandpass | `GRISM` | `PRISM` |

Read `docs/mode_support.md` for the script-by-script status. This document is
about the design and, more importantly, about why the obvious cheaper approach
was rejected.

## The physics is the same; the constants are not

Nothing in the trace polynomials, PSF interpolation, dispersion, or flux
accumulation is element-specific. The prism's `wl_transform: log` was already
handled generically in `optical_model_jax.py` before the prism existed as a
supported mode. Every genuine difference is a **constant**.

That is the whole justification for `modes.DisperserMode` being a flat frozen
record rather than a class hierarchy with a `GrismDisperser` and a
`PrismDisperser`: there is no behaviour to override. A subclass hierarchy would
be a lot of scaffolding around a table of nine values.

## Why the in-place approach was rejected

The prism was first implemented (branch `prism`, commit `50de443`, titled
"Stage A: switch single-mode-per-checkout to P127 prism") by editing the grism
constants **in place**:

```python
-ORDERS = ["0", "1", "2"]      +ORDERS = ["1"]
-LAM_MIN = 0.9                 +LAM_MIN = 0.75
-LAM_MAX = 2.0                 +LAM_MAX = 1.85
-filter_map = {'0': 'GRISM0', '1': 'GRISM1'}
+filter_map = {'1': 'PRISM'}
```

For a throwaway experiment that was the right call — it got prism images out in
days. But it has a property that only shows up later: **every one of those edits
deletes grism support.** The result is a checkout that can do prism *or* grism,
never both, and — this is the part that actually cost time — two branches whose
conflicts have no correct resolution, because the file has nowhere to put two
modes. Merging `prism` into `main` was not a matter of picking sides on ten
hunks; it required first building the abstraction the branch had declined to
build.

The generalisable lesson, and the reason this document exists rather than a
commit message: **a module-level constant is a single-mode global.** It is
invisible at the call site, so nothing tells a reader that changing it changes
the meaning of the whole program, and editing it is always locally easier than
introducing a parameter. That is exactly what makes it a trap. `pipeline.ORDERS`,
`LAM_MIN` and `LAM_MAX` were therefore **deleted** in the merge rather than
given new values — if they still existed, the next element would be added the
same way.

## The shape

```python
mode = modes.get_mode(None)        # explicit > $ROMAN_DISPERSER_MODE > grism
mode.orders                        # ("0", "1", "2")
mode.lam_min, mode.lam_max         # microns
mode.filter_for("1")               # "GRISM1"   (raises, listing valid orders)
mode.psf_order("2")                # "1"        (grism order 2 borrows order 1)
paths.optical_model_path(mode=mode)
paths.sensitivity_dir(mode=mode)
```

Three rules that the surrounding code depends on:

1. **Grism is the default, everywhere.** Selection order mirrors
   `paths.data_dir()`: explicit argument, then `--mode` / the config's `mode:`
   key, then `$ROMAN_DISPERSER_MODE`, then grism. Anything that does not ask
   gets exactly its pre-merge behaviour.
2. **`mode` is keyword-optional, never required and never positional.** Callers
   outside this repo pass no mode at all — the tutorials repo calls all five
   `paths.*` accessors with no arguments, and the `line_test` branch builds
   paths from `data_dir()` directly. A required parameter would break them; a
   positional one would break them silently and strangely.
3. **Band and order arguments take `None` sentinels resolved from the mode**,
   rather than being made required. `pipeline.select_sources_per_order` is
   called from the tutorials as
   `select_sources_per_order({"1": opt}, xfpa, yfpa, orders=["1"])` — band
   omitted. Making the band explicit at every call site was the original plan
   and would have been a `TypeError` there.

## Fail loudly on a mode/model mismatch

`modes.check_band(mode, model)` runs at pipeline setup and raises on three
things, all of which are otherwise **silent wrong answers rather than crashes**:

1. **Mismatched element** — prism mode with the grism YAML, or vice versa. The
   polynomials load and evaluate perfectly happily and give wrong traces.
2. **Extrapolation** — the mode's band extending outside the model's
   `wl_min`/`wl_max`. The trace and dispersion polynomials are fitted over the
   model's validity range; evaluated beyond it they diverge *gracefully*,
   producing plausible positions with no warning. Narrowing **inside** the
   validity range is legitimate deliberate behaviour and is allowed; only
   extrapolation is an error. A `padding` tolerance exists but defaults to 0,
   because both modes agree with their models exactly today.
3. **Undefined order** — an order the delivered model does not define. Note the
   grism YAML defines `["1", "0", "2", "m1"]` while we render only the first
   three, so `orders` is a *choice* and cannot be derived; the subset check is
   the most that can be asserted.

`tests/test_modes.py` asserts the same pairing for every declared mode without
needing a run, so a future reference-data delivery that moves a band edge trips
a test rather than a production job.

## Adding a third element

1. Add a `DisperserMode` to `modes.py` and register it in `MODES`.
2. Publish its optical model, sensitivities and PSF caches as data-repo
   releases; add asset keys to `hydrate.py`. Follow the tag-naming rule in
   `docs/data_vendoring_plan.md` — the version in the model *filename* belongs
   to IPAC/SSC and is not ours to increment.
3. Check the catalog's wavelength grid covers the new band.
   `build_source_catalog.py --wl-min` widens it; the Galacticus slice is derived
   from the requested range, so nothing else needs changing.
4. Run the suite. `tests/test_modes.py` is parametrized over `MODES`, so the
   band and order consistency checks cover the new element automatically.
5. Update `docs/mode_support.md`.

If step 1 tempts you to edit an existing mode's values instead of adding a new
record — that is the failure this document is about.

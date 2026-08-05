# 20260731 SSC line-grid campaign — v0.13.0 rerun

Waves 1b / 2 / 3 of the SSC line-grid validation package, rerun on
`roman_disperser` v0.13.0 after the sky→FPA precision, gnomonic-projection and
per-SCA RNG fixes. See `workbench/README.md` for the wave → run-store mapping.

## Known documentation issues in the delivered products

None of these affect a rendered pixel — the images, truth tables and residuals
are all self-consistent. They are **labelling and wording defects** in metadata
that ships alongside the data, found in a pre-merge review on 2026-08-04. They
are recorded rather than fixed because fixing them means regenerating a
completed campaign, and each is trivially correctable downstream.

### 1. `F158` and `flux_scale` are the flux *anchor*, not synthetic photometry

In `metadata.parquet` (and the `flux_scale` column of the source manifests),
`F158` is the **anchor magnitude applied to the SED template**, not the
broadband F158 magnitude of the emitted spectrum. The `seds.zarr` attribute on
`star_seds` reads `"normalized to 0 mag F158"`, which holds for the template as
built but not after `--no-continuum`.

Waves 1b and 3 were built with `--no-continuum`: the flat-f_nu pedestal is
subtracted, leaving only the five Gaussian emission lines, so the emitted
spectrum carries far less broadband flux than a 0-AB flat source.

| wave | `no_continuum` | template F158 | anchor (reported as `F158`) | true source F158 |
|---|---|---|---|---|
| 1b | true | +3.425 mag | 16.3071968 | **19.732** |
| 2 | false | −0.045 mag | 16.3071968 | 16.262 |
| 3 | true | +3.425 mag | 16.3071968 | **19.732** |

So for **waves 1b and 3 the catalog `F158` overstates source brightness by
3.425 mag (×23.4)**. Wave 2 is correct to 0.045 mag — the residual is the
emission lines sitting on top of the continuum, which is the "marginally
brighter" case the builder docstring describes.

To correct: `true_F158_mag = -2.5*log10(F158) + 3.425` for the two lines-only
waves. The offset is a single constant — every source in the field shares one
template.

*How the table was measured:* the template FLAM from `seds.zarr` through the
repo's own vendored bandpass, `roman_disperser.refdata.get_f158_band()`, with
`synphot.Observation(...).effstim('abmag')`. Pass the SED with explicit
`erg/s/cm^2/AA` units — synphot's `Empirical1D` defaults to PHOTLAM and will
silently return a ~29.7 mag offset otherwise. Calibration check: a synthetic
flat-f_nu 0-AB spectrum on the same grid returns −0.00007 mag.

*Why it happened:* the 30× line-flux boost was applied by moving the continuum
anchor from mag 20 to 16.3071968 (`10^(0.4·(20 − 16.3071968)) = 30.000`) to
improve S/N. `--no-continuum` then removes the pedestal while `F158` and
`flux_scale` keep carrying the anchor.

### 2. `lines.ecsv` asserts a continuum the lines-only catalogs do not contain

`build_line_test_catalog.write_sidecar` writes `meta["continuum_mag"]`
unconditionally, ignoring `--no-continuum`. In the wave 1b and wave 3 catalog
directories, `lines.ecsv` therefore carries `continuum_mag: 16.3071968` and the
description *"amp_rel = line peak / local continuum; line added on top of
continuum"*, while `provenance.json` **in the same directory** says
`"continuum_mag": null, "no_continuum": true`.

`provenance.json` is the authoritative one. `continuum_mag` in `lines.ecsv`
should be read as *the continuum level used to set the line amplitudes*
(`amp_rel × f_continuum(centre)`), which is what makes the line fluxes
well-defined — not as a pedestal present in the data.

### 3. Truth-table precision

`truth_lines.parquet` positions are evaluated with JAX x64 enabled, but
`make_sca_payload` returns an all-float32 payload, so they are neither a true
float64 evaluation nor the float32 numbers the renderer and checker used. See
the comment block at the top of `wave*/make_truth_tables.py` for measured
figures. In short: they differ from the shipped `residuals.parquet` predictions
by a median 5.3e-4 px (max 3.4e-3 px, zero rows bit-identical), and from a true
float64 evaluation by 7.2e-4 px max. The latter offset is deterministic —
approximately 99% of it is a per-(SCA, order) constant — so it can be tabulated
and removed; it is **not** an irreducible float32 noise floor.

### 4. `d_cross` sign convention is unstated

`check_line_centering.dispersion_axes` defines the cross-dispersion unit vector
as a 90° rotation in *pixel* space (`cross = [-t_y, t_x]`). Its on-sky sense
therefore depends on each SCA's parity, and `residuals.parquet` ships a
`d_cross` column without recording this. The campaign conclusions use medians
and absolute values, so nothing downstream of us depends on the sign — but do
not assume a fixed on-sky direction.

## Known overstatements in the analysis scripts

Wording only; the code is correct and the figures are what they claim to plot.

- `scripts/analyze_psf_shifts.py` — the docstring says the per-SCA residual
  *"equals"* the PSF centroid shift and calls this *"confirmation"*. The script
  reports only Pearson `r`, which is scale-invariant and cannot support
  "equals". Fitted properly: `r` = 0.974 (x) / 0.991 (y) but slope 0.764 (x) /
  0.737 (y), so roughly 25% of the per-SCA offset is **not** accounted for by
  the PSF shift. The qualitative conclusion — the offset is PSF-dominated —
  stands; "equals" does not.
- `scripts/check_line_centering.py` — *"the method validated to reach ~0.02
  px"* names no statistic and no dataset. Measured on the shipped wave-1b
  residuals (STPSF, PA 0, 5,288 boxes): `d_disp` median −0.016, MAD 0.040,
  **rms 0.076 px**. ~0.02 px is the method's noise floor under the idealised
  centred-Gaussian PSF, not the accuracy of the delivered STPSF products.
- `scripts/make_gaussian_psf_cache.py` — *"a per-SCA ~0.05 px centroid offset"*
  is the floor, not the typical value: measured across all 18 GRISM1 caches the
  offset spans 0.050–0.141 px, median ≈0.074.
- `analysis/before_after.py` figure 4 overlays the audit-reported 0.6–2.9 px
  band on a **2-D** rms, `sqrt(<rx² + ry²>)`. For isotropic residuals a 2-D rms
  is √2× the per-axis rms, and which the audit reported is not recorded
  anywhere. The measured agreement (0.70–2.69 vs 0.6–2.9) is close either way,
  but the comparison rests on an unwritten assumption.

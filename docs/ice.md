# Water-ice throughput model (`roman_disperser.ice`)

Thin films of water ice grow on the WFI detectors between decontamination
("decon") cycles and change the throughput as a function of wavelength by up
to about ±20 % — thin-film interference, so the ratio can exceed 1; it is not a
pure loss. The `ice` module folds that into a dispersed-image simulation as a
multiplicative, wavelength- and position-dependent factor on each source's
flux. It is **off by default**; nothing changes unless an ice directory is
configured.

Status (2026-09-24): first implementation against the **toy** ice model from
`RomanSpaceTelescope/ice-toy-model` (`ice_rep_sim.py`, 2026-09-21). The growth
rate map is "representative, not realistic", and only one transmission table
(SCA10, SCIPA TVAC) exists. The code is written so that better inputs drop in
through the same files.

## The model

For a detector position $(x, y)$ (1-indexed FITS SCA pixels) at time $t$ (MJD):

$$
d(x, y, t) = r\!\left(\left\lfloor \tfrac{y-1}{32} \right\rfloor,\; \left\lfloor \tfrac{x-1}{32} \right\rfloor\right)\cdot\big[(t - t_0) \bmod P\big],
\qquad
f(\lambda; x, y, t) = \frac{T(\lambda; d)}{T(\lambda; 0)}
$$

| Symbol | Meaning | Source | Units |
|---|---|---|---|
| $r$ | ice growth rate on a coarse grid of 32-px bins (128 × 128 per SCA), constant in time; **nearest bin** lookup | `ice_rate_mosaic.npz` (FPA mosaic with per-SCA offsets) | nm/day |
| $t_0$ | decon epoch | `ice_map.yaml: epoch_mjd` (default MJD 61295.0 = 2026-09-12T00:00 UTC) | MJD |
| $P$ | decon period — thickness resets to zero over the whole FPA every $P$ days | `ice_map.yaml: decon_period_days` (default 20) | days |
| $T(\lambda; d)/T(\lambda; 0)$ | transmission relative to the ice-free state, tabulated at uniform thicknesses (0…300 nm in 30 nm steps); **linear in $d$** between the two bracketing columns | `spectral_responses_SCA10.ecsv` (one file per SCA in the map; all point at SCA10 today) | — |

MJDs before $t_0$ are rejected. Thickness beyond the table's last column is
**clamped** to it; `load_ice_payload` warns at load time when
$\max r \cdot P$ exceeds the table (a warning cannot be raised inside jitted
code). The existing sensitivity curves are, by this definition, the ice-free
($d = 0$) baseline.

The bin lookup and the thickness interpolation reproduce
`ice_rep_sim.ice_relative_transmission` exactly (`tests/test_ice.py` checks
against a numpy transcription of it). Two conventions are ours: positions are
converted from 1-indexed FITS to 0-indexed before binning (the toy model's
pixel coordinates are 0-based), and the toy model's within-detector (row, col)
is taken to be our (y, x) with no flips — a convention to pin down with the
model's authors, not a correctness risk while the map is representative.

## Where the factor is applied

`star_disperser.deposit_stack_native` deposits one native-resolution stamp
per fine wavelength at a single dispersed centre, with the flux entering as a
per-wavelength scalar. So the factor is evaluated **once per fine wavelength,
at that wavelength's dispersed centre** $(x_k, y_k)$ (already computed,
vectorised, before the deposit scan), and multiplied into the flux vector:

```
flux_k  <-  flux_k * f(lambda_k; x_k, y_k, t)
```

in `disperse_star_psf` and `disperse_galaxy`. The hot loop is untouched; the
cost is two gathers and a few flops per wavelength per source, negligible next
to the deposit. The stamp-centre evaluation is within the model's own 32-px
resolution except where a stamp straddles a bin boundary — accepted for now
(decision 2026-09-24); a per-pixel factor inside the deposit is the fallback
if it ever matters.

Because the factor uses the *dispersed* position, orders 0 and 2 (grism) get
the right thickness for where their light actually lands, with no extra code.

### Static vs dynamic (JIT)

Per SCA, `ice.load_ice_payload(ice_dir, sca, wavelengths_um)` builds a payload
holding the rate map and the ratio table **pre-interpolated in wavelength onto
the run grid** (`trans[n_thick, N_wl]`, aligned by index with the
`wavelengths` the dispersers receive). It is captured in the dispersers'
closures like the optical and PSF payloads.

The thickness map `r · dt` depends on the exposure time, so it is a
**dynamic argument**: `make_star_disperser(..., ice_payload=p)` and
`make_galaxy_disperser(..., ice_payload=p)` return functions with one extra
trailing argument, `ice_thickness_nm` ([128, 128], from
`ice.thickness_map(p, mjd)`), and `pipeline.make_batched_*_fori(..., ice=True)`
/ `disperse_batched_*(..., ice_thickness_nm=...)` thread it through. A new
exposure time therefore does **not** recompile. With `ice_payload=None` the
compiled program is byte-for-byte the pre-existing one (Python-level branch at
trace time), so golden frames are unaffected.

Time handling is host-side and deliberately split out so the decon schedule
can be replaced (e.g. by a list of real decon dates):
`ice.time_since_decon(mjd, epoch_mjd, period_days)` → `ice.thickness_map`.

## Data: embargoed, hand-copied, outside the repo

The rate mosaic and the transmission table are **not** shipped with the
package or published as `roman_disperser_data` releases: the table header
restricts it to the Roman calibration and spectrophotometric working groups
("contact Eric Switzer for approval for any additional distribution"). `pixi
run hydrate` does not fetch them. Put them in a directory of your own —
`paths.ice_dir()` = `$ROMAN_DISPERSER_DATA/ice/` by default — with an
`ice_map.yaml` index:

```yaml
rate_mosaic: ice_rate_mosaic.npz
epoch_mjd: 61295.0            # optional (default shown)
decon_period_days: 20.0       # optional (default shown)
tables:
  SCA1:  spectral_responses_SCA10.ecsv
  SCA2:  spectral_responses_SCA10.ecsv
  # ... SCA3 .. SCA18, all -> the SCA10 table until per-SCA tables exist
```

File formats, exactly as in the toy-model repo:

- `ice_rate_mosaic.npz`: `rate_mosaic` [rows, cols] float (NaN between
  detectors), `det_row_offsets` / `det_col_offsets` (dict, 1-indexed SCA →
  bin offset of the detector's corner), `pix_per_bin` (32).
- `spectral_responses_*.ecsv`: `wavelength` [nm] plus one `T_ratio_d<N>nm`
  column per thickness `N` [nm]; thicknesses must start at 0 and be uniformly
  spaced. Wavelengths outside the table hold the edge values; the current
  table (480–2300 nm) covers both dispersers' bands.

The package's own tests use synthetic files with this schema
(`tests/test_ice.py::write_synthetic_ice_dir`), so the suite never needs the
real data.

## Using it in the pipeline

Config (`build_dispersed_image.py --config`):

```yaml
ice_dir: default        # <data>/ice/, or an explicit directory
```

With `ice_dir` set the pointing ECSV **must** carry an `MJD` column (exposure
start, Modified Julian Date); the run refuses to start otherwise. Quick mode:
`--ice-dir default --mjd 61300.5`.

Provenance written per product: FITS cards `ICEMJD`, `ICEDT` (days since
decon), `ICEEPOCH`, `ICEPER`, `ICERATE` (mosaic file), `ICETABLE` (this SCA's
table); an `ice:` block in the per-pointing `_meta.yaml` (directory, MJD,
schedule, per-SCA tables). Absent when the model is off.

## Open items

- Stamps straddling bin edges (accepted approximation; see above).
- Orientation of the toy model's (row, col) vs our (x, y).
- Per-SCA tables, if/when they exist — the map already supports them.
- Whether the parametric model behind the table (sqrt-diffusion growth,
  $n_\mathrm{water}$ = 1.25, …) can be shared; if so the physics could be
  implemented directly and the table kept as a private validation check.

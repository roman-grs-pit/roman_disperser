#!/usr/bin/env python
"""Check that injected emission lines land where the optical model predicts.

Companion to `build_line_test_catalog.py`. Given a grism simulation of the
line-test field (a `MODEL` image + source manifest from `build_grism_image.py`)
and the `lines.ecsv` sidecar, this:

1. For every dispersed star and every injected line wavelength, predicts the
   first-order detector pixel where that line should fall, using BOTH optical
   models:
     - JAX functional model  (`optical_model_jax`, the production path), and
     - the original class model (`RomanOpticalModel`),
   and reports their agreement (a pure model-vs-model cross-check).
2. Cuts a small box around each predicted position in the noiseless MODEL
   image, subtracts a local continuum, and measures the flux-weighted centroid.
3. Reports residuals (measured - predicted) decomposed into the along-dispersion
   and cross-dispersion directions, which respectively probe the wavelength
   solution and the trace geometry.

Conventions
-----------
- Detector coordinates are 1-indexed FITS pixels (pixel n centered at n.0), so
  array element [i, j] corresponds to FITS (x, y) = (j+1, i+1).
- The optical model / disperser take wavelengths in **microns**; `lines.ecsv`
  stores centers in Angstroms.
- What this validates: rendering with the JAX disperser and centroiding against
  the model is a self-consistency test of disperser + PSF + centroiding, plus a
  direct JAX-vs-class equivalence check. It is not an absolute check against an
  external reference (e.g. grizli/romanisim).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

import jax.numpy as jnp
import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser import paths

ORDER = "1"  # first order


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

def predict_jax(payload, xsca, ysca, lam_um):
    """Production JAX path: (SCA pos, lambda[um]) -> dispersed SCA pixel.

    Mirrors `star_disperser._compute_dispersed_positions`:
    sca_to_fpa -> trace_beam -> mpa_to_sca. `xsca, ysca` are scalars; `lam_um`
    is a 1-D array. Returns (xpix[Nlam], ypix[Nlam]).
    """
    xf, yf = omj.sca_to_fpa(payload, jnp.array([xsca]), jnp.array([ysca]))
    xm, ym = omj.trace_beam(payload, xf, yf, jnp.asarray(lam_um))
    xp, yp = omj.mpa_to_sca(payload, xm, ym)
    return np.asarray(xp).ravel(), np.asarray(yp).ravel()


def _lines_in_model_units(model, lam_A):
    """Convert Angstrom line centers to the model's native wl_grid units."""
    native_is_micron = np.nanmax(model.wl_grid) < 100.0
    return (np.asarray(lam_A) / 1e4) if native_is_micron else np.asarray(lam_A)


def predict_class(model, sca, xsca, ysca, lam_A):
    """Original class path: drive `_get_beam_trace` at the line wavelengths.

    `_get_beam_trace` evaluates the trace over `model.wl_grid`, so we swap in the
    line wavelengths (in the model's native units), run the genuine class
    workhorse for a width-1 trace, and read out the dispersed SCA positions.
    Independent of the JAX path down to the shared coordinate transforms.
    """
    xfpa, yfpa = model.coords.convert_sca_to_fpa(
        np.atleast_1d(float(xsca)), np.atleast_1d(float(ysca)), sca=sca)
    saved = model.wl_grid
    try:
        model.wl_grid = _lines_in_model_units(model, lam_A)
        coeff = model._get_beam_trace(
            xref_fpa=float(xfpa[0]), yref_fpa=float(yfpa[0]),
            sca=sca, width=1, order=ORDER)
    finally:
        model.wl_grid = saved
    return np.asarray(coeff["trace_sca_x"][0]), np.asarray(coeff["trace_sca_y"][0])


# --------------------------------------------------------------------------
# Centroiding
# --------------------------------------------------------------------------

def _centroid_1d(prof, center_idx, linehalf):
    """Centroid a 1-D line profile after subtracting a linear baseline.

    The baseline is fit to the off-line pixels (|i - center_idx| > linehalf) and
    subtracted; the centroid is the flux-weighted mean over the line window. This
    removes the smooth dispersed-continuum ridge under the line, which otherwise
    biases a flat-background flux-weighted centroid. Returns a float index into
    `prof`, or nan if unusable.
    """
    idx = np.arange(len(prof))
    off = np.abs(idx - center_idx) > linehalf
    if off.sum() < 3:
        return np.nan
    baseline = np.polyval(np.polyfit(idx[off], prof[off], 1), idx)
    resid = np.clip(prof - baseline, 0.0, None)
    sel = np.abs(idx - center_idx) <= linehalf
    w = resid[sel]
    if w.sum() <= 0:
        return np.nan
    return float((idx[sel] * w).sum() / w.sum())


def measure_line(img, xpred, ypred, u_disp, half=8, narrow=3, linehalf=4):
    """Measure a line centroid by 1-D collapse along and across dispersion.

    `img` is the [Ny, Nx] MODEL array (0-indexed); `xpred, ypred` are 1-indexed
    FITS coords; `u_disp` is the local dispersion unit vector (x, y). We treat
    the detector axis with the larger |u_disp| component as the dispersion axis
    (Roman SCAs disperse ~along one axis; the tilt across a small box is
    negligible). For each direction we collapse the box over a narrow window in
    the other direction, subtract a linear baseline fit to the off-line pixels,
    and centroid — the method validated to reach ~0.02 px, versus ~0.18 px for a
    whole-box flat-background centroid contaminated by the continuum trace.
    Returns (xmeas, ymeas) in 1-indexed FITS coords, or (nan, nan).
    """
    ny, nx = img.shape
    xc = int(round(xpred)) - 1  # 0-indexed nearest pixel
    yc = int(round(ypred)) - 1
    r0, r1, c0, c1 = yc - half, yc + half + 1, xc - half, xc + half + 1
    if r0 < 0 or c0 < 0 or r1 > ny or c1 > nx:
        return np.nan, np.nan
    box = np.asarray(img[r0:r1, c0:c1], float)
    cenx = (xpred - 1) - c0  # predicted center within box (col), 0-indexed
    ceny = (ypred - 1) - r0  # predicted center within box (row)

    def collapse(axis, keep_center):
        """Sum a narrow band (2*narrow+1 wide) centered on `keep_center` along
        `axis` (0=rows, 1=cols), returning the 1-D profile along the other axis."""
        k = int(round(keep_center))
        lo, hi = max(k - narrow, 0), k + narrow + 1
        return box[lo:hi, :].sum(0) if axis == 0 else box[:, lo:hi].sum(1)

    if abs(u_disp[1]) >= abs(u_disp[0]):        # dispersion ~ along y (rows)
        disp_meas = _centroid_1d(collapse(1, cenx), ceny, linehalf)   # vs row
        cross_meas = _centroid_1d(collapse(0, ceny), cenx, linehalf)  # vs col
        ymeas = r0 + disp_meas + 1.0
        xmeas = c0 + cross_meas + 1.0
    else:                                       # dispersion ~ along x (cols)
        disp_meas = _centroid_1d(collapse(0, ceny), cenx, linehalf)   # vs col
        cross_meas = _centroid_1d(collapse(1, cenx), ceny, linehalf)  # vs row
        xmeas = c0 + disp_meas + 1.0
        ymeas = r0 + cross_meas + 1.0
    return xmeas, ymeas


def dispersion_axes(payload, xsca, ysca, lam_um, dlam_um=1e-3):
    """Local unit vectors along/across dispersion at each line wavelength.

    Tangent = d(pixel)/d(lambda) via a small finite difference; the
    cross-dispersion unit vector is its 90-deg rotation. Returns (u_disp,
    u_cross), each [Nlam, 2] with columns (x, y).
    """
    x0, y0 = predict_jax(payload, xsca, ysca, lam_um)
    x1, y1 = predict_jax(payload, xsca, ysca, np.asarray(lam_um) + dlam_um)
    t = np.stack([x1 - x0, y1 - y0], axis=1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    cross = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return t, cross


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_inputs(fits_path, catalog_dir):
    """Load MODEL image + SCA, the source manifest, and the injected lines."""
    fits_path = Path(fits_path)
    with fits.open(fits_path) as hdul:
        model_img = np.asarray(hdul["MODEL"].data, float)
        sca = int(hdul[0].header["DETNUM"])
    manifest_hits = sorted(fits_path.parent.glob("*sources.parquet"))
    if not manifest_hits:
        raise FileNotFoundError(
            f"No *sources.parquet manifest next to {fits_path}")
    manifest = pd.read_parquet(manifest_hits[0])
    lines = Table.read(Path(catalog_dir) / "lines.ecsv", format="ascii.ecsv")
    return model_img, sca, manifest, lines


def run(fits_path, catalog_dir, optical_model=None, box_half=8, max_stars=None,
        model=None, quiet=False, source_type="PSF", narrow=3, linehalf=4):
    model_img, sca, manifest, lines = load_inputs(fits_path, catalog_dir)
    lam_A = np.asarray(lines["center_A"], float)
    lam_um = lam_A / 1e4

    if model is None:
        om_path = optical_model or (paths.data_dir() / "Roman_grism_OpticalModel_v0.8.yaml")
        model = RomanOpticalModel(str(om_path))
    payload = omj.make_sca_payload(model, sca=sca, order=ORDER)

    # Batch-mode manifests list every SCA; keep only this detector's first-order
    # sources of the requested type. For SER (Sersic galaxy) catalogs the
    # prediction is still the *point* trace at the galaxy center, so measured -
    # predicted is the morphology-induced centroid shift, which is the measurand.
    stars = manifest[(manifest["order"] == ORDER)
                     & (manifest["type"] == source_type)
                     & (manifest["sca"] == sca)]
    stars = stars.drop_duplicates("catalog_index").reset_index(drop=True)
    if max_stars:
        stars = stars.iloc[:max_stars]
    print(f"SCA {sca}: {len(stars)} first-order {source_type} sources, "
          f"{len(lam_A)} lines each")

    rows = []
    jc_max = 0.0
    for _, s in stars.iterrows():
        xsca, ysca = float(s["xsca"]), float(s["ysca"])
        xj, yj = predict_jax(payload, xsca, ysca, lam_um)
        xc, yc = predict_class(model, sca, xsca, ysca, lam_A)
        jc_max = max(jc_max, np.nanmax(np.hypot(xj - xc, yj - yc)))
        u_disp, u_cross = dispersion_axes(payload, xsca, ysca, lam_um)
        for k in range(len(lam_A)):
            xm, ym = measure_line(model_img, xj[k], yj[k], u_disp[k],
                                  half=box_half, narrow=narrow, linehalf=linehalf)
            d = np.array([xm - xj[k], ym - yj[k]])
            rows.append({
                "sca": int(sca),
                "catalog_index": int(s["catalog_index"]),
                "xsca": xsca, "ysca": ysca,
                "line_id": int(lines["line_id"][k]), "center_A": lam_A[k],
                "x_pred_jax": xj[k], "y_pred_jax": yj[k],
                "x_pred_class": xc[k], "y_pred_class": yc[k],
                "x_meas": xm, "y_meas": ym,
                "dx": d[0], "dy": d[1],
                "d_disp": float(d @ u_disp[k]) if np.isfinite(xm) else np.nan,
                "d_cross": float(d @ u_cross[k]) if np.isfinite(xm) else np.nan,
            })
    res = pd.DataFrame(rows)

    if not quiet:
        _print_summary(res, jc_max)
    res.attrs["jc_max"] = jc_max
    return res


def _print_summary(res, jc_max, label=""):
    ok = res.dropna(subset=["d_disp"])
    print(f"\n{label}JAX vs class max separation: {jc_max:.4f} px")
    print(f"{label}Centroided {len(ok)}/{len(res)} (line, star) boxes")
    for col in ("d_disp", "d_cross"):
        if not len(ok):
            break
        v = ok[col].to_numpy()
        print(f"  {col}: median {np.median(v):+.3f}  "
              f"MAD {np.median(np.abs(v - np.median(v))):.3f}  "
              f"RMS {np.sqrt(np.mean(v**2)):.3f}  max|.| {np.max(np.abs(v)):.3f} px")


def run_pointing_dir(pointing_dir, catalog_dir, optical_model=None, box_half=8,
                     source_type="PSF", narrow=3, linehalf=4):
    """Run the checker on every per-SCA FITS in a batch-mode pointing directory.

    Loads the optical model once and reuses it across SCAs. Returns the
    concatenated residuals table across all SCAs.
    """
    pointing_dir = Path(pointing_dir)
    fits_files = sorted(pointing_dir.glob("grism_*_detSCA*.fits"))
    if not fits_files:
        raise FileNotFoundError(f"No grism_*_detSCA*.fits in {pointing_dir}")
    om_path = optical_model or (paths.data_dir() / "Roman_grism_OpticalModel_v0.8.yaml")
    model = RomanOpticalModel(str(om_path))
    parts = []
    for f in fits_files:
        res = run(f, catalog_dir, box_half=box_half, model=model, quiet=True,
                  source_type=source_type, narrow=narrow, linehalf=linehalf)
        jc = res.attrs.get("jc_max", np.nan)
        _print_summary(res, jc, label=f"[{f.name}] ")
        parts.append(res)
    allres = pd.concat(parts, ignore_index=True)
    print("\n" + "=" * 60)
    _print_summary(allres, max(p.attrs.get("jc_max", 0.0) for p in parts),
                   label="ALL SCAs · ")
    return allres


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fits", help="Single grism sim FITS (MODEL image).")
    src.add_argument("--pointing-dir",
                     help="Batch-mode pointing dir; runs all per-SCA FITS and aggregates.")
    p.add_argument("--catalog-dir", required=True,
                   help="Line-test catalog dir (holds lines.ecsv).")
    p.add_argument("--optical-model", default=None, help="Optical model YAML.")
    p.add_argument("--box-half", type=int, default=8, help="Centroid box half-size (px).")
    p.add_argument("--source-type", choices=["PSF", "SER"], default="PSF",
                   help="Manifest source type to measure. SER measures Sersic "
                        "galaxies against the point prediction at the galaxy "
                        "center (residual = morphology-induced shift).")
    p.add_argument("--narrow", type=int, default=3,
                   help="Cross-band half-width for the 1-D collapse (px). "
                        "Widen for extended sources.")
    p.add_argument("--linehalf", type=int, default=4,
                   help="Line-window half-size for baseline fit + centroid (px). "
                        "Widen for morphology-broadened lines.")
    p.add_argument("--max-stars", type=int, default=None, help="Limit stars (debug).")
    p.add_argument("--out", default=None, help="Write residuals table (parquet).")
    args = p.parse_args()

    if args.pointing_dir:
        res = run_pointing_dir(args.pointing_dir, args.catalog_dir,
                               args.optical_model, args.box_half,
                               source_type=args.source_type,
                               narrow=args.narrow, linehalf=args.linehalf)
    else:
        res = run(args.fits, args.catalog_dir, args.optical_model,
                  args.box_half, args.max_stars, source_type=args.source_type,
                  narrow=args.narrow, linehalf=args.linehalf)
    if args.out:
        res.to_parquet(args.out)
        print(f"\nWrote residuals -> {args.out}")


if __name__ == "__main__":
    main()

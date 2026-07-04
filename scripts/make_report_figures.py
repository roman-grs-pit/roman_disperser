#!/usr/bin/env python
"""Figures for the line-centering validation report.

Produces, from a full-pointing sim + the residuals table + the line-test catalog:
  1. an example SCA MODEL image (full + zoom with predicted lines marked),
  2. one star's trace with the 5 line boxes drawn (predicted vs measured),
  3. centroiding detail for a few lines (cutout + 1-D collapsed profile), and
  4. extracted vs predicted line-flux comparison.

All coordinates are 1-indexed FITS (array[i,j] <-> FITS (x,y)=(j+1,i+1)).
Run from the line_test worktree; see the report doc for context.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FLAM_0AB_COEFF = 0.108866  # matches build_line_test_catalog


# --------------------------------------------------------------------------
# Shared box extraction (mirrors check_line_centering.measure_line internals)
# --------------------------------------------------------------------------

def cut_box(img, xp, yp, half):
    xc, yc = int(round(xp)) - 1, int(round(yp)) - 1
    r0, r1, c0, c1 = yc - half, yc + half + 1, xc - half, xc + half + 1
    if r0 < 0 or c0 < 0 or r1 > img.shape[0] or c1 > img.shape[1]:
        return None
    return np.asarray(img[r0:r1, c0:c1], float), r0, c0


def collapse_profile(box, cenx, ceny, narrow=3):
    """1-D dispersion profile (sum over narrow cross-dispersion band)."""
    k = int(round(cenx))
    lo, hi = max(k - narrow, 0), k + narrow + 1
    return box[:, lo:hi].sum(1)


def baseline_and_centroid(prof, center_idx, linehalf=4):
    idx = np.arange(len(prof))
    off = np.abs(idx - center_idx) > linehalf
    coef = np.polyfit(idx[off], prof[off], 1)
    baseline = np.polyval(coef, idx)
    resid = np.clip(prof - baseline, 0.0, None)
    sel = np.abs(idx - center_idx) <= linehalf
    w = resid[sel]
    cen = (idx[sel] * w).sum() / w.sum()
    return baseline, cen, resid


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load(sca_fits, residuals, catalog_dir, sca):
    with fits.open(sca_fits) as h:
        img = np.asarray(h["MODEL"].data, float)
    res = pd.read_parquet(residuals)
    res = res[res.sca == sca].copy()
    lines = Table.read(Path(catalog_dir) / "lines.ecsv", format="ascii.ecsv")
    return img, res, lines


# --------------------------------------------------------------------------
# Figure 1: SCA overview + zoom
# --------------------------------------------------------------------------

def fig_sca_overview(img, res, zoom, out):
    norm = ImageNormalize(img, vmin=0, vmax=np.percentile(img, 99.9),
                          stretch=AsinhStretch(0.02))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    ax[0].imshow(img, origin="lower", cmap="inferno", norm=norm)
    x0, x1, y0, y1 = zoom
    ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, ec="cyan", fc="none", lw=1.2))
    ax[0].set_title("SCA 5 MODEL (noiseless, counts/s)")
    ax[0].set_xlabel("x [pix]"); ax[0].set_ylabel("y [pix]")

    ax[1].imshow(img, origin="lower", cmap="inferno", norm=norm)
    inzoom = res[(res.x_pred_jax.between(x0, x1)) & (res.y_pred_jax.between(y0, y1))]
    sc = ax[1].scatter(inzoom.x_pred_jax - 1, inzoom.y_pred_jax - 1, s=26,
                       facecolors="none", edgecolors=plt.cm.viridis(
                           (inzoom.center_A - 11000) / 8000), lw=1.1)
    ax[1].set_xlim(x0, x1); ax[1].set_ylim(y0, y1)
    ax[1].set_title("zoom: predicted first-order line positions (color = wavelength)")
    ax[1].set_xlabel("x [pix]"); ax[1].set_ylabel("y [pix]")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)


# --------------------------------------------------------------------------
# Figure 2: one star's trace with the 5 line boxes
# --------------------------------------------------------------------------

def fig_line_boxes(img, res, star, half, out):
    s = res[res.catalog_index == star].sort_values("center_A")
    norm = ImageNormalize(img, vmin=0, vmax=np.percentile(img, 99.95),
                          stretch=AsinhStretch(0.02))
    xlo = int(s.x_pred_jax.min()) - 3 * half
    xhi = int(s.x_pred_jax.max()) + 3 * half
    ylo = int(s.y_pred_jax.min()) - 3 * half
    yhi = int(s.y_pred_jax.max()) + 3 * half
    fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.imshow(img, origin="lower", cmap="inferno", norm=norm)
    for _, r in s.iterrows():
        xc, yc = r.x_pred_jax - 1, r.y_pred_jax - 1
        ax.add_patch(Rectangle((xc - half, yc - half), 2 * half + 1, 2 * half + 1,
                               ec="cyan", fc="none", lw=1.0))
        ax.plot(xc, yc, "r+", ms=8, mew=1.4)                # predicted
        if np.isfinite(r.x_meas):
            ax.plot(r.x_meas - 1, r.y_meas - 1, "gx", ms=7, mew=1.4)  # measured
        ax.text(xc + half + 2, yc, f"{r.center_A:.0f} Å", color="cyan",
                fontsize=8, va="center")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_title(f"star {star}: first-order line boxes\n"
                 f"red + = predicted, green x = measured")
    ax.set_xlabel("x [pix]"); ax.set_ylabel("y [pix]")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)


# --------------------------------------------------------------------------
# Figure 3: centroiding detail
# --------------------------------------------------------------------------

def fig_centroiding(img, res, star, half, out, which=(0, 2, 4)):
    s = res[res.catalog_index == star].sort_values("center_A").reset_index(drop=True)
    fig, ax = plt.subplots(2, len(which), figsize=(4.2 * len(which), 7.5))
    for col, li in enumerate(which):
        r = s.iloc[li]
        box, r0, c0 = cut_box(img, r.x_pred_jax, r.y_pred_jax, half)
        cenx = (r.x_pred_jax - 1) - c0
        ceny = (r.y_pred_jax - 1) - r0
        # top: cutout with predicted/measured marks
        ax[0, col].imshow(box, origin="lower", cmap="inferno")
        ax[0, col].plot(cenx, ceny, "r+", ms=11, mew=1.6, label="predicted")
        ax[0, col].plot((r.x_meas - 1) - c0, (r.y_meas - 1) - r0, "gx", ms=9,
                        mew=1.6, label="measured")
        ax[0, col].set_title(f"{r.center_A:.0f} Å   d_disp={r.d_disp:+.3f} px")
        if col == 0:
            ax[0, col].legend(fontsize=7, loc="upper left")
        # bottom: 1-D collapsed dispersion profile + baseline + centroid
        prof = collapse_profile(box, cenx, ceny)
        base, cen, resid = baseline_and_centroid(prof, ceny)
        idx = np.arange(len(prof))
        ax[1, col].plot(idx, prof, "o-", ms=3, label="collapsed profile")
        ax[1, col].plot(idx, base, "--", color="gray", label="linear baseline")
        ax[1, col].axvline(ceny, color="r", lw=1, label="predicted")
        ax[1, col].axvline(cen, color="g", lw=1, ls=":", label="measured centroid")
        ax[1, col].set_xlabel("pixel along dispersion")
        if col == 0:
            ax[1, col].set_ylabel("counts/s (summed cross-disp)")
            ax[1, col].legend(fontsize=7)
    fig.suptitle(f"star {star}: centroiding (1-D collapse + baseline subtraction)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)


# --------------------------------------------------------------------------
# Figure 4: extracted vs predicted line flux
# --------------------------------------------------------------------------

def predicted_line_counts(catalog_dir, sca, lines):
    """Total predicted counts/s per line: sum(line_FLAM * sensitivity * dlam)."""
    import zarr, os, yaml
    store = zarr.open(str(Path(catalog_dir) / "seds.zarr"), mode="r")
    wl = np.array(store["wavelengths"])                       # Å
    sed = np.array(store["star_seds"][0])                     # 0-mag units
    prov = json.loads((Path(catalog_dir) / "provenance.json").read_text())
    fscale = prov["flux_scale_F158_maggies"]
    cont = FLAM_0AB_COEFF / wl**2
    line_only = np.clip(sed - (0 if prov.get("no_continuum") else cont), 0, None)
    sd = os.path.expandvars("$ROMAN_DISPERSER_DATA/sensitivities")
    m = yaml.safe_load(open(os.path.join(sd, "sensitivity_map.yaml")))
    with fits.open(os.path.join(sd, m[f"SCA{sca}"]["1"])) as h:
        S = np.interp(wl, np.array(h[1].data["WAVELENGTH"], float),
                      np.array(h[1].data["SENSITIVITY"], float), left=0, right=0)
    dlam = wl[1] - wl[0]
    out = {}
    for row in lines:
        c, fw = row["center_A"], row["fwhm_A"]
        win = np.abs(wl - c) < 5 * fw / 2.3548
        out[int(row["line_id"])] = float((line_only[win] * fscale * S[win] * dlam).sum())
    return out


def measured_line_counts(img, xp, yp, half=8, ap_disp=6, ap_cross=4):
    """Continuum-subtracted counts/s in an aperture around the line."""
    got = cut_box(img, xp, yp, half)
    if got is None:
        return np.nan
    box, r0, c0 = got
    yy, xx = np.mgrid[0:box.shape[0], 0:box.shape[1]]
    edge = np.zeros(box.shape, bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    A = np.column_stack([np.ones(edge.sum()), xx[edge], yy[edge]])
    coef, *_ = np.linalg.lstsq(A, box[edge], rcond=None)
    sub = box - (coef[0] + coef[1] * xx + coef[2] * yy)
    cy, cx = box.shape[0] // 2, box.shape[1] // 2
    ap = sub[cy - ap_disp:cy + ap_disp + 1, cx - ap_cross:cx + ap_cross + 1]
    return float(np.clip(ap, 0, None).sum())


def fig_flux(img, res, lines, catalog_dir, sca, out):
    pred = predicted_line_counts(catalog_dir, sca, lines)
    rows = []
    for _, r in res.iterrows():
        if not np.isfinite(r.x_meas):
            continue
        meas = measured_line_counts(img, r.x_pred_jax, r.y_pred_jax)
        rows.append((r.line_id, r.center_A, pred[int(r.line_id)], meas))
    t = pd.DataFrame(rows, columns=["line_id", "center_A", "pred", "meas"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax[0].scatter(t.pred, t.meas, c=t.center_A, s=12, cmap="viridis")
    lo, hi = t.pred.min(), t.pred.max()
    slope = np.sum(t.pred * t.meas) / np.sum(t.pred**2)
    ax[0].plot([lo, hi], [slope * lo, slope * hi], "k--",
               label=f"aperture fraction {slope:.2f}")
    ax[0].set_xlabel("predicted line counts/s (SED × sensitivity × dλ)")
    ax[0].set_ylabel("extracted counts/s (aperture)")
    ax[0].set_title("extracted vs predicted line flux (SCA 5)")
    ax[0].legend(); plt.colorbar(sc, ax=ax[0], label="wavelength [Å]")
    # per-line ratio measured/predicted (normalized to its mean → tests relative flux)
    t["ratio"] = t.meas / t.pred / slope
    for lid, g in t.groupby("line_id"):
        ax[1].scatter(g.center_A + np.random.uniform(-80, 80, len(g)), g.ratio, s=10)
        ax[1].scatter(g.center_A.iloc[0], g.ratio.median(), marker="_", s=700, c="k", zorder=5)
    ax[1].axhline(1, color="k", lw=.5)
    ax[1].set_ylim(0.7, 1.3); ax[1].set_xlabel("line center [Å]")
    ax[1].set_ylabel("extracted / predicted  (÷ aperture fraction)")
    ax[1].set_title("relative flux agreement per line")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)
    print("  aperture fraction (slope): %.3f" % slope)
    print("  per-line median extracted/predicted (÷slope):")
    for lid, g in t.groupby("line_id"):
        print(f"    line {lid} ({g.center_A.iloc[0]:.0f}Å): {g.ratio.median():.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sca-fits", required=True)
    p.add_argument("--residuals", required=True)
    p.add_argument("--catalog-dir", required=True)
    p.add_argument("--sca", type=int, default=5)
    p.add_argument("--star", type=int, default=10061)
    p.add_argument("--half", type=int, default=8)
    p.add_argument("--outdir", required=True)
    p.add_argument("--zoom", type=int, nargs=4, default=[2400, 2900, 600, 1550],
                   help="x0 x1 y0 y1 (0-indexed) for the SCA zoom panel.")
    args = p.parse_args()

    np.random.seed(0)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    img, res, lines = load(args.sca_fits, args.residuals, args.catalog_dir, args.sca)
    fig_sca_overview(img, res, args.zoom, outdir / "fig_sca_overview.png")
    fig_line_boxes(img, res, args.star, args.half, outdir / "fig_line_boxes.png")
    fig_centroiding(img, res, args.star, args.half, outdir / "fig_centroiding.png")
    fig_flux(img, res, lines, args.catalog_dir, args.sca, outdir / "fig_line_flux.png")


if __name__ == "__main__":
    main()

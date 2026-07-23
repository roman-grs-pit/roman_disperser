#!/usr/bin/env python
"""Measure PSF flux-centroid offsets directly from the PSF caches.

The line-centering residual has a per-SCA constant offset. This script measures,
straight from the STPSF PSF cache `.npz` files, each PSF's flux centroid relative
to its cutout centre (the "pixel shift"), per SCA and vs wavelength, and — if
given a residuals table — shows the per-SCA residual equals that shift. This is
the independent confirmation that the per-SCA offset is the PSF, not the optical
model.

PSF cache: `psf_grid` shape (Nx_spatial, Ny_spatial, Nwl, Ppix, Ppix), oversampled
by `oversample`; wavelengths in microns. Offsets are reported in **native px**
(oversampled centroid offset / oversample).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def psf_centroid_offsets(npz_path):
    """Per-wavelength PSF centroid offset (native px), averaged over the spatial
    grid. Returns (wl_um[Nwl], off_x[Nwl], off_y[Nwl])."""
    d = np.load(npz_path)
    g = d["psf_grid"]                      # (nx, ny, nwl, P, P)
    ov = int(d["oversample"])
    wl = np.array(d["wavelengths"], float)
    P = g.shape[-1]
    yy, xx = np.mgrid[0:P, 0:P]
    cen = (P - 1) / 2.0
    nx, ny, nwl = g.shape[:3]
    ox = np.zeros(nwl); oy = np.zeros(nwl)
    for w in range(nwl):
        cx = cy = 0.0
        for i in range(nx):
            for j in range(ny):
                p = g[i, j, w]; s = p.sum()
                cx += (p * xx).sum() / s - cen
                cy += (p * yy).sum() / s - cen
        ox[w] = cx / (nx * ny) / ov
        oy[w] = cy / (nx * ny) / ov
    return wl, ox, oy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--order", default="GRISM1", choices=["GRISM0", "GRISM1"])
    p.add_argument("--residuals", default=None,
                   help="Optional residuals parquet to overlay per-SCA offsets.")
    p.add_argument("--out", required=True, help="Output figure path.")
    args = p.parse_args()

    from roman_disperser.paths import data_dir
    psfdir = str(data_dir() / "psf_cache")
    recs = []
    curves = {}
    for sca in range(1, 19):
        f = glob.glob(f"{psfdir}/psf_WFI{sca:02d}_{args.order}*.npz")[0]
        wl, ox, oy = psf_centroid_offsets(f)
        curves[sca] = (wl, ox, oy)
        mid = len(wl) // 2
        recs.append((sca, ox[mid], oy[mid], ox.mean(), oy.mean()))
    T = pd.DataFrame(recs, columns=["sca", "cx_mid", "cy_mid", "cx_mean", "cy_mean"])

    npanel = 3 if args.residuals else 2
    fig, ax = plt.subplots(1, npanel, figsize=(5 * npanel, 4.6))

    # A: per-SCA offset vector (x vs y), labelled
    ax[0].axhline(0, color="k", lw=.4); ax[0].axvline(0, color="k", lw=.4)
    ax[0].scatter(T.cx_mid, T.cy_mid, c=T.sca, cmap="tab20", s=40)
    for _, r in T.iterrows():
        ax[0].annotate(int(r.sca), (r.cx_mid, r.cy_mid), fontsize=7,
                       xytext=(2, 2), textcoords="offset points")
    ax[0].set_xlabel("PSF centroid offset x [px]")
    ax[0].set_ylabel("PSF centroid offset y [px]")
    ax[0].set_title(f"{args.order} PSF centroid offset per SCA (mid-λ)")
    ax[0].set_aspect("equal")

    # B: offset vs wavelength, all SCAs
    for sca, (wl, ox, oy) in curves.items():
        ax[1].plot(wl, ox, lw=.8, alpha=.6)
        ax[1].plot(wl, oy, lw=.8, alpha=.6, ls="--")
    ax[1].set_xlabel("wavelength [µm]"); ax[1].set_ylabel("centroid offset [px]")
    ax[1].set_title("offset vs λ (solid=x, dashed=y; one line per SCA)")

    # C: PSF offset vs measured per-SCA residual
    if args.residuals:
        res = pd.read_parquet(args.residuals).dropna(subset=["d_disp"])
        g = res.groupby("sca").agg(dx=("dx", "median"), dy=("dy", "median"))
        m = T.set_index("sca").join(g)
        ax[2].plot([-.15, .15], [-.15, .15], "k--", lw=.7)
        ax[2].scatter(m.cx_mid, m.dx, s=30, label="x")
        ax[2].scatter(m.cy_mid, m.dy, s=30, marker="s", label="y")
        rx = np.corrcoef(m.cx_mid, m.dx)[0, 1]
        ry = np.corrcoef(m.cy_mid, m.dy)[0, 1]
        ax[2].set_xlabel("PSF centroid offset [px]")
        ax[2].set_ylabel("per-SCA median residual [px]")
        ax[2].set_title(f"residual vs PSF shift (r_x={rx:.2f}, r_y={ry:.2f})")
        ax[2].legend(); ax[2].set_aspect("equal")

    fig.tight_layout(); fig.savefig(args.out, dpi=120); plt.close(fig)
    print("saved", args.out)
    print(f"\n{args.order} PSF centroid offset per SCA (native px, mid-λ):")
    print(T[["sca", "cx_mid", "cy_mid"]].round(3).to_string(index=False))
    print(f"  |offset| range: {np.hypot(T.cx_mid, T.cy_mid).min():.3f} – "
          f"{np.hypot(T.cx_mid, T.cy_mid).max():.3f} px")


if __name__ == "__main__":
    main()

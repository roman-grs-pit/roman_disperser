#!/usr/bin/env python
"""Make a 3-panel showcase figure: imaging | grism | prism, zoomed at RA=10, Dec=0.

This is a *visualization* script for the acceptance/prism test products, not a
science tool. It tells one story across three views of the same patch of sky:

    Imaging  (L3 sky coadd)   : galaxies/stars as rectified blobs on the sky.
    Grism    (L2 detector)    : the same field dispersed -> long spectral traces.
    Prism    (L2 detector)    : the same field dispersed -> short/compressed traces.

Important: imaging is a rectified *sky* map; grism/prism are *detector* frames
(dispersed light cannot be mosaicked into a sky image). So the panels are NOT
pixel-aligned cutouts of an identical box. Instead:
  - the imaging panel shows a wide (~FOV_IMG) true RA/Dec cutout, and
  - a rectangle is drawn on it marking the smaller sky sub-region (~FOV_SUB)
    whose sources are what the grism/prism detector windows display.

ENVIRONMENT: run under the `roman_l2_job` pixi env (needs asdf + roman_datamodels
extensions to read the Roman ASDF coadds/L2, plus astropy/scipy/matplotlib/pandas):

    cd /data/npadman/1-Projects/roman_l2_job
    pixi run python /data/npadman/1-Projects/roman_disperser/scripts/make_showcase_figure.py

Sizes are CLI-tunable so the framing can be iterated visually.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import asdf
import numpy as np
import pandas as pd
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Default data locations (this specific acceptance/prism run).
# ---------------------------------------------------------------------------
ROOT = "/mnt/roman-science/grs"
IMG_ROOT = f"{ROOT}/acceptance-testing-20260430/imaging/2026-04-30"
GRISM_ROOT = f"{ROOT}/acceptance-testing-20260430/spectro/2026-05-05/acceptance"
PRISM_ROOT = f"{ROOT}/prism-testing-20260527/spectro/2026-05-28"

# Defaults for the (SCA, exposure) that cover RA=10/Dec=0 (found from source manifests).
# Nudged showcase point RA=10.183/Dec=-0.184 (~16' from 10/0) lands dead-center on
# both grism exp010.001 SCA05 (edge 1802px) and prism pt1 SCA03 (edge ~2030px).
GRISM_EXP = "001.001.001.001.010.001"
GRISM_SCA = 5
PRISM_DIR = "prism-single.sim_001.001.001.001.001.001"  # RA 10, Dec 0, PA 0
PRISM_SCA = 3

# Skycell index -> (RA, Dec) affine for projection cell 010p00, fit from sampled
# tiles (residuals ~1e-4 deg). RA = A.[xi,yi,1], Dec = B.[xi,yi,1].
_AFFINE_SAMPLES = [
    (65, 69, 11.7777, 1.0995),
    (35, 46, 10.0913, -1.0998),
    (36, 29, 8.8450, -1.0262),
    (48, 41, 9.7247, -0.1466),
    (50, 62, 11.2646, 0.0000),
    (47, 57, 10.8979, -0.2200),
]


def _skycell_affine():
    M = np.array([[xi, yi, 1] for xi, yi, _, _ in _AFFINE_SAMPLES])
    ra = np.array([p[2] for p in _AFFINE_SAMPLES])
    dec = np.array([p[3] for p in _AFFINE_SAMPLES])
    cra = np.linalg.lstsq(M, ra, rcond=None)[0]
    cdec = np.linalg.lstsq(M, dec, rcond=None)[0]
    return cra, cdec


# ---------------------------------------------------------------------------
# Imaging panel: resample overlapping L3 skycells onto a common tangent grid.
# ---------------------------------------------------------------------------
def make_target_wcs(ra, dec, fov_arcmin, scale_arcsec):
    n = int(round(fov_arcmin * 60.0 / scale_arcsec))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return w, n


def load_imaging_cutout(ra, dec, fov_arcmin, scale_arcsec, verbose=True):
    """Resample all overlapping 010p00 coadd skycells onto a TAN grid at (ra,dec)."""
    cra, cdec = _skycell_affine()
    # Enumerate available tiles by filename (cheap), predict centers, keep nearby.
    files = {}
    for f in glob.glob(f"{IMG_ROOT}/**/*010p00*_coadd.asdf", recursive=True):
        b = os.path.basename(f)
        files.setdefault(b, f)  # dedupe across partial dirs
    tiles = []
    reach = fov_arcmin / 60.0 / 2.0 + 0.08  # half-fov + ~one skycell half-width (deg)
    for b, f in files.items():
        m = re.search(r"x(\d+)y(\d+)", b)
        xi, yi = int(m.group(1)), int(m.group(2))
        tra = cra @ [xi, yi, 1]
        tdec = cdec @ [xi, yi, 1]
        if abs(tra - ra) < reach / max(np.cos(np.radians(dec)), 0.1) and abs(tdec - dec) < reach:
            tiles.append(f)
    if verbose:
        print(f"[imaging] {len(tiles)} overlapping skycells")

    tw, n = make_target_wcs(ra, dec, fov_arcmin, scale_arcsec)
    yy, xx = np.mgrid[0:n, 0:n]
    ra_g, dec_g = tw.all_pix2world(xx, yy, 0)
    # Background-matched averaging: subtract each skycell's median over the cutout
    # before coadding, then add back a common median. This removes the seams that
    # arise from per-skycell background offsets in overlapping tiles.
    acc = np.zeros((n, n), dtype=np.float64)
    cnt = np.zeros((n, n), dtype=np.float64)
    bgs = []
    for f in tiles:
        with asdf.open(f, lazy_load=True, memmap=True) as af:
            gw = af["roman"]["meta"]["wcs"]
            data = np.asarray(af["roman"]["data"], dtype=np.float32)
            ny, nx = data.shape
            tx, ty = gw.world_to_pixel_values(ra_g, dec_g)
            valid = np.isfinite(tx) & np.isfinite(ty) & (tx >= 0) & (tx < nx - 1) & (ty >= 0) & (ty < ny - 1)
            if not valid.any():
                continue
            samp = map_coordinates(data, [ty, tx], order=1, mode="constant", cval=np.nan, prefilter=False)
            ok = valid & np.isfinite(samp)
            if not ok.any():
                continue
            bg = float(np.median(samp[ok]))
            bgs.append(bg)
            acc[ok] += samp[ok] - bg
            cnt[ok] += 1.0
    out = np.full((n, n), np.nan, dtype=np.float32)
    have = cnt > 0
    out[have] = (acc[have] / cnt[have]) + (np.median(bgs) if bgs else 0.0)
    return out, tw


# ---------------------------------------------------------------------------
# Grism / prism detector panels.
# ---------------------------------------------------------------------------
def load_l2(path):
    with asdf.open(path, lazy_load=True, memmap=True) as af:
        return np.asarray(af["roman"]["data"], dtype=np.float32)


def source_box(parquet, sca, ra0, dec0, fov_arcmin, order="1"):
    """Undispersed (xsca,ysca) of order-`order` sources within a sky box on `sca`."""
    t = pd.read_parquet(parquet, columns=["sca", "order", "xsca", "ysca", "ra", "dec"])
    r = fov_arcmin / 60.0 / 2.0
    m = (
        (t.sca == sca)
        & (t["order"].astype(str) == order)
        & (t.ra > ra0 - r) & (t.ra < ra0 + r)
        & (t.dec > dec0 - r) & (t.dec < dec0 + r)
    )
    s = t[m]
    return s.xsca.to_numpy(), s.ysca.to_numpy()


def detector_window(data, xsca, ysca, margin, pad):
    """Cut a wide detector window around the field, plus the field-footprint box.

    Shows the field's undispersed bbox surrounded by `margin` px of detector
    context (so the dispersed traces stream through it and surrounding spectra
    fill the frame instead of whitespace). The returned box marks the 1' field's
    undispersed footprint (the "same region" highlighted on the imaging panel).

    xsca/ysca are 1-indexed FITS coords; array index is [y-1, x-1].
    Returns (cutout, (x0,x1,y0,y1) detector coords, (bx,by,bw,bh) box in cutout-local coords).
    """
    ny, nx = data.shape
    fx0, fx1 = xsca.min() - 1.0, xsca.max() - 1.0  # 1-indexed -> 0-indexed
    fy0, fy1 = ysca.min() - 1.0, ysca.max() - 1.0
    x0 = max(int(np.floor(fx0 - margin)), 0)
    x1 = min(int(np.ceil(fx1 + margin)), nx)
    y0 = max(int(np.floor(fy0 - margin)), 0)
    y1 = min(int(np.ceil(fy1 + margin)), ny)
    box = (fx0 - x0 - pad, fy0 - y0 - pad, (fx1 - fx0) + 2 * pad, (fy1 - fy0) + 2 * pad)
    return data[y0:y1, x0:x1], (x0, x1, y0, y1), box


def full_frame(data, xsca, ysca, pad):
    """Whole SCA (4088^2) with the 1' field-footprint box (the imaging-zoom region)."""
    ny, nx = data.shape
    fx0, fx1 = xsca.min() - 1.0, xsca.max() - 1.0
    fy0, fy1 = ysca.min() - 1.0, ysca.max() - 1.0
    box = (fx0 - pad, fy0 - pad, (fx1 - fx0) + 2 * pad, (fy1 - fy0) + 2 * pad)
    return data, (0, nx, 0, ny), box


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def asinh_norm(img, pct=99.5, a=0.1):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return ImageNormalize(vmin=0, vmax=1)
    lo = np.nanpercentile(finite, 30)
    hi = np.nanpercentile(finite, pct)
    return ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(a=a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", type=float, default=10.183, help="nudged off (10,0) to center the dispersed fields")
    ap.add_argument("--dec", type=float, default=-0.184)
    ap.add_argument("--fov-img", type=float, default=2.0, help="imaging panel FOV [arcmin]")
    ap.add_argument("--fov-sub", type=float, default=1.0, help="grism/prism sky sub-region [arcmin]")
    ap.add_argument("--img-scale", type=float, default=0.11, help="imaging render scale [arcsec/pix]")
    ap.add_argument("--crop", action="store_true", help="crop dispersed panels to a window (default: full SCA frame)")
    ap.add_argument("--grism-margin", type=float, default=550.0, help="--crop: detector context around grism field [pix]")
    ap.add_argument("--prism-margin", type=float, default=400.0, help="--crop: detector context around prism field [pix]")
    ap.add_argument("--pad", type=float, default=25.0, help="field-footprint box pad [pix]")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--no-star", action="store_true", help="disable the bright-star orientation marker")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_grism_prism"),
    )
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # --- imaging ---
    img, tw = load_imaging_cutout(args.ra, args.dec, args.fov_img, args.img_scale)

    # --- grism ---
    g_l2 = glob.glob(f"{GRISM_ROOT}/output_l2/*{GRISM_EXP}/*detSCA{GRISM_SCA:02d}_l2.asdf")[0]
    g_src = glob.glob(f"{GRISM_ROOT}/output/*{GRISM_EXP}/*sources.parquet")[0]
    gx, gy = source_box(g_src, GRISM_SCA, args.ra, args.dec, args.fov_sub)
    gdata = load_l2(g_l2)
    if args.crop:
        gcut, gwin, gbox = detector_window(gdata, gx, gy, args.grism_margin, args.pad)
    else:
        gcut, gwin, gbox = full_frame(gdata, gx, gy, args.pad)
    print(f"[grism] {len(gx)} sources, window {gwin}")

    # --- prism ---
    p_l2 = glob.glob(f"{PRISM_ROOT}/output_l2/{PRISM_DIR}/*detSCA{PRISM_SCA:02d}_l2.asdf")[0]
    p_src = glob.glob(f"{PRISM_ROOT}/output/{PRISM_DIR}/*sources.parquet")[0]
    px, py = source_box(p_src, PRISM_SCA, args.ra, args.dec, args.fov_sub)
    pdata = load_l2(p_l2)
    if args.crop:
        pcut, pwin, pbox = detector_window(pdata, px, py, args.prism_margin, args.pad)
    else:
        pcut, pwin, pbox = full_frame(pdata, px, py, args.pad)
    print(f"[prism] {len(px)} sources, window {pwin}")

    # --- marker stars: circle in imaging, order-1 trace overlaid on dispersed panels ---
    # Traces are precomputed by scripts/compute_showcase_traces.py (needs the optical
    # model, which lives in a different env), read from a tracked JSON.
    trace_json = os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_star_traces.json")
    stars = []
    if not args.no_star and os.path.exists(trace_json):
        with open(trace_json) as f:
            stars = json.load(f).get("stars", [])
        print(f"[stars] {len(stars)} marker stars (mag {[round(s['mag'],1) for s in stars]})")

    # --- figure ---
    cmap = matplotlib.colormaps[args.cmap].copy()
    cmap.set_bad("black")  # NaN/masked detector pixels blend with the background
    args.cmap = cmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(img, origin="lower", cmap=args.cmap, norm=asinh_norm(img), interpolation="nearest")
    axes[0].set_title(f"Imaging F158 (L3 coadd)\n{args.fov_img:.0f}' @ RA={args.ra}, Dec={args.dec}")
    # sub-region rectangle (sky box -> imaging pixels), only if imaging is wider
    if args.fov_img > args.fov_sub * 1.05:
        half = args.fov_sub / 60.0 / 2.0
        corners_ra = args.ra + np.array([-1, 1, 1, -1]) * half / np.cos(np.radians(args.dec))
        corners_dec = args.dec + np.array([-1, -1, 1, 1]) * half
        cx, cy = tw.all_world2pix(corners_ra, corners_dec, 0)
        axes[0].add_patch(
            plt.Polygon(np.c_[cx, cy], fill=False, ec="cyan", lw=1.5, ls="--")
        )

    def add_box(ax, box):
        bx, by, bw, bh = box
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="cyan", lw=1.5, ls="--"))

    axes[1].imshow(gcut, origin="lower", cmap=args.cmap, norm=asinh_norm(gcut), interpolation="nearest")
    add_box(axes[1], gbox)
    axes[1].set_title(f"Grism (L2, SCA{GRISM_SCA:02d})\nfull SCA, 7.5' / 0.11\" px")

    axes[2].imshow(pcut, origin="lower", cmap=args.cmap, norm=asinh_norm(pcut), interpolation="nearest")
    add_box(axes[2], pbox)
    axes[2].set_title(f"Prism (L2, SCA{PRISM_SCA:02d})\nfull SCA, 7.5' / 0.11\" px")

    # marker stars: numbered circles in imaging, matching order-1 traces on the
    # dispersed panels (same colour per star). Orients all three views.
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.5, foreground="black")]
    palette = ["cyan", "springgreen", "white", "deepskyblue", "magenta", "yellow"]

    def label(ax, x, y, k, c):
        ax.text(x, y, str(k + 1), color=c, fontsize=11, weight="bold",
                path_effects=halo, ha="left", va="bottom")

    for k, s in enumerate(stars):
        c = palette[k % len(palette)]
        ix, iy = tw.all_world2pix([s["ra"]], [s["dec"]], 0)
        rad = max(8.0, 4.0 / args.img_scale)  # ~4" radius
        axes[0].add_patch(plt.Circle((ix[0], iy[0]), rad, fill=False, ec=c, lw=2.0, path_effects=halo))
        label(axes[0], ix[0] + rad, iy[0] + rad, k, c)
        for ax, key, win in ((axes[1], "g_trace", gwin), (axes[2], "p_trace", pwin)):
            if key in s:
                tx = np.array(s[key]["x"]) - 1 - win[0]
                ty = np.array(s[key]["y"]) - 1 - win[2]
                ax.plot(tx, ty, "-", color=c, lw=2.0, alpha=0.95, path_effects=halo, solid_capstyle="round")
                label(ax, tx[0], ty[0], k, c)

    fig.suptitle("Roman WFI simulation — same sky, three views", fontsize=15, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
        print("wrote", f"{args.out}.{ext}")


if __name__ == "__main__":
    main()

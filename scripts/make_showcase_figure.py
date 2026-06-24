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
        return np.asarray(af["roman"]["data"], dtype=np.float32), af["roman"]["meta"]["wcs"]


def draw_compass(ax, wcs, cx, cy, length, halo, color="white"):
    """Draw an N/E compass at the panel's lower-left from a WCS (astropy or GWCS).

    Directions are evaluated at (cx, cy) full-frame pixels; arrows are drawn in
    display coordinates. The dispersed L2 GWCS rotation is reliable (validated
    against the optical model: grism N≈up, prism N≈149°), so each panel gets a
    compass showing how sky N/E fall on that frame.
    """
    d = 2.0 / 3600.0  # 2 arcsec probe
    ra, dec = [np.atleast_1d(v)[0] for v in wcs.pixel_to_world_values(cx, cy)]
    xn, yn = [np.atleast_1d(v)[0] for v in wcs.world_to_pixel_values(ra, dec + d)]
    xe, ye = [np.atleast_1d(v)[0] for v in wcs.world_to_pixel_values(ra + d / np.cos(np.radians(dec)), dec)]

    def unit(x2, y2):
        v = np.array([x2 - cx, y2 - cy])
        n = np.hypot(*v)
        return v / n * length if n else v

    vn, ve = unit(xn, yn), unit(xe, ye)
    ox = oy = length * 1.4  # origin near lower-left of the panel
    for v, lab in ((vn, "N"), (ve, "E")):
        ax.annotate("", xy=(ox + v[0], oy + v[1]), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
        ax.text(ox + v[0] * 1.28, oy + v[1] * 1.28, lab, color=color, fontsize=10,
                weight="bold", ha="center", va="center", path_effects=halo)


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


def full_frame(data, xsca, ysca, pad):
    """Whole SCA (4088^2) with the 1' field-footprint box (the imaging-zoom region)."""
    ny, nx = data.shape
    fx0, fx1 = xsca.min() - 1.0, xsca.max() - 1.0
    fy0, fy1 = ysca.min() - 1.0, ysca.max() - 1.0
    box = (fx0 - pad, fy0 - pad, (fx1 - fx0) + 2 * pad, (fy1 - fy0) + 2 * pad)
    return data, (0, nx, 0, ny), box


def disp_crop(data, xsca, ysca, fov_arcmin, pad, scale=0.11):
    """Crop the dispersed SCA to a `fov_arcmin` box centred on the field footprint.

    Centre = midpoint of the 1' field's undispersed source positions (≈ the
    nudged RA/Dec on this SCA). Returns (cut, (x0,x1,y0,y1) detector coords,
    field-footprint box in cut-local coords).
    """
    ny, nx = data.shape
    half = fov_arcmin * 60.0 / scale / 2.0
    fx0, fx1 = xsca.min() - 1.0, xsca.max() - 1.0
    fy0, fy1 = ysca.min() - 1.0, ysca.max() - 1.0
    cx, cy = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
    x0 = max(int(cx - half), 0); x1 = min(int(cx + half), nx)
    y0 = max(int(cy - half), 0); y1 = min(int(cy + half), ny)
    box = (fx0 - x0 - pad, fy0 - y0 - pad, (fx1 - fx0) + 2 * pad, (fy1 - fy0) + 2 * pad)
    return data[y0:y1, x0:x1], (x0, x1, y0, y1), box


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
    ap.add_argument("--fov-img", type=float, default=3.0, help="imaging panel FOV [arcmin]")
    ap.add_argument("--fov-sub", type=float, default=1.0, help="grism/prism sky sub-region [arcmin]")
    ap.add_argument("--img-scale", type=float, default=0.11, help="imaging render scale [arcsec/pix]")
    ap.add_argument("--fov-disp", type=float, default=3.0, help="grism/prism panel FOV [arcmin] (0 = full SCA)")
    ap.add_argument("--pad", type=float, default=25.0, help="field-footprint box pad [pix]")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--no-star", action="store_true", help="disable the bright-star orientation marker")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_grism_prism"),
    )
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # --- marker stars (read first: the panels are centred on the star group) ---
    trace_json = os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_star_traces.json")
    stars = []
    if not args.no_star and os.path.exists(trace_json):
        with open(trace_json) as f:
            stars = json.load(f).get("stars", [])
        print(f"[stars] {len(stars)} marker stars (mag {[round(s['mag'],1) for s in stars]})")

    # Centre each panel on the marker-star group (falls back to args.ra/dec).
    if stars:
        ra_c = float(np.mean([s["ra"] for s in stars]))
        dec_c = float(np.mean([s["dec"] for s in stars]))
        gmx = np.array([s["g_undisp"][0] for s in stars]); gmy = np.array([s["g_undisp"][1] for s in stars])
        pmx = np.array([s["p_undisp"][0] for s in stars]); pmy = np.array([s["p_undisp"][1] for s in stars])
    else:
        ra_c, dec_c = args.ra, args.dec

    # --- imaging ---
    img, tw = load_imaging_cutout(ra_c, dec_c, args.fov_img, args.img_scale)

    # --- grism ---
    g_l2 = glob.glob(f"{GRISM_ROOT}/output_l2/*{GRISM_EXP}/*detSCA{GRISM_SCA:02d}_l2.asdf")[0]
    g_src = glob.glob(f"{GRISM_ROOT}/output/*{GRISM_EXP}/*sources.parquet")[0]
    gdata, gwcs = load_l2(g_l2)
    if not stars:
        gmx, gmy = source_box(g_src, GRISM_SCA, args.ra, args.dec, args.fov_sub)
    gcut, gwin, _ = (disp_crop(gdata, gmx, gmy, args.fov_disp, args.pad)
                     if args.fov_disp > 0 else full_frame(gdata, gmx, gmy, args.pad))
    print(f"[grism] window {gwin}")

    # --- prism ---
    p_l2 = glob.glob(f"{PRISM_ROOT}/output_l2/{PRISM_DIR}/*detSCA{PRISM_SCA:02d}_l2.asdf")[0]
    p_src = glob.glob(f"{PRISM_ROOT}/output/{PRISM_DIR}/*sources.parquet")[0]
    pdata, pwcs = load_l2(p_l2)
    if not stars:
        pmx, pmy = source_box(p_src, PRISM_SCA, args.ra, args.dec, args.fov_sub)
    pcut, pwin, _ = disp_crop(pdata, pmx, pmy, args.fov_disp, args.pad) if args.fov_disp > 0 else full_frame(pdata, pmx, pmy, args.pad)
    print(f"[prism] window {pwin}")

    # --- figure ---
    cmap = matplotlib.colormaps[args.cmap].copy()
    cmap.set_bad("black")  # NaN/masked detector pixels blend with the background
    args.cmap = cmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(img, origin="lower", cmap=args.cmap, norm=asinh_norm(img), interpolation="nearest")
    axes[0].set_title(f"Imaging F158 (L3 coadd)\n{args.fov_img:.0f}' @ RA={ra_c:.3f}, Dec={dec_c:.3f}")

    disp_lbl = f"{args.fov_disp:.0f}' / 0.11\" px" if args.fov_disp > 0 else "full SCA, 7.5'"
    axes[1].imshow(gcut, origin="lower", cmap=args.cmap, norm=asinh_norm(gcut), interpolation="nearest")
    axes[1].set_title(f"Grism (L2, SCA{GRISM_SCA:02d})\n{disp_lbl}")

    axes[2].imshow(pcut, origin="lower", cmap=args.cmap, norm=asinh_norm(pcut), interpolation="nearest")
    axes[2].set_title(f"Prism (L2, SCA{PRISM_SCA:02d})\n{disp_lbl}")

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
        # imaging: circle at the star's sky (direct-image) position
        ix, iy = tw.all_world2pix([s["ra"]], [s["dec"]], 0)
        rad = max(8.0, 4.0 / args.img_scale)  # ~4" radius
        axes[0].add_patch(plt.Circle((ix[0], iy[0]), rad, fill=False, ec=c, lw=2.0, path_effects=halo))
        label(axes[0], ix[0] + rad, iy[0] + rad, k, c)
        # dispersed: box around the order-1 spectrum (so the dispersed light shows
        # through) + mark the direct-image (undispersed) position with the number
        bpx, bpy = 16.0, 8.0  # box pad around the trace extent [pix]
        for ax, tkey, ukey, win in (
            (axes[1], "g_trace", "g_undisp", gwin),
            (axes[2], "p_trace", "p_undisp", pwin),
        ):
            if tkey in s:
                tx = np.array(s[tkey]["x"]) - 1 - win[0]
                ty = np.array(s[tkey]["y"]) - 1 - win[2]
                x0, x1, y0, y1 = tx.min() - bpx, tx.max() + bpx, ty.min() - bpy, ty.max() + bpy
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=1.6, path_effects=halo))
            if ukey in s:
                ux, uy = s[ukey][0] - 1 - win[0], s[ukey][1] - 1 - win[2]
                ax.plot(ux, uy, "x", color=c, ms=7, mew=1.8, path_effects=halo)
                label(ax, ux + 6, uy + 6, k, c)

    # clamp each panel to its image extent so overlay patches don't expand the axes
    for ax, im in zip(axes, (img, gcut, pcut)):
        ax.set_xlim(0, im.shape[1] - 1)
        ax.set_ylim(0, im.shape[0] - 1)

    # N/E compass on every panel (sky orientation differs per frame: grism N≈up,
    # prism N rotated ~59deg). Imaging from its TAN WCS, dispersed from the L2 GWCS.
    draw_compass(axes[0], tw, img.shape[1] / 2, img.shape[0] / 2, 0.10 * img.shape[1], halo)
    draw_compass(axes[1], gwcs, 0.5 * (gwin[0] + gwin[1]), 0.5 * (gwin[2] + gwin[3]), 0.10 * gcut.shape[1], halo)
    draw_compass(axes[2], pwcs, 0.5 * (pwin[0] + pwin[1]), 0.5 * (pwin[2] + pwin[3]), 0.10 * pcut.shape[1], halo)

    fig.suptitle("Roman WFI simulation — same sky, three views", fontsize=15, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
        print("wrote", f"{args.out}.{ext}")


if __name__ == "__main__":
    main()

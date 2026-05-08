"""Render a sky-coordinate (RA/Dec) mosaic PNG of an L2 pointing.

Loads all *_l2.asdf files from a pointing directory, reads each detector's
GWCS, rebins the L2 image in detector space, projects pixel centers to a
common tangent-plane RA/Dec WCS centered at the pointing, and accumulates
into a single canvas (per-pixel mean across overlapping samples).

Usage:
    pixi run -e romanisim python render_l2_pointing_mosaic.py \\
        <pointing_dir> <output.png> [--rebin 4] [--target-arcsec 0.5]

The detector WCS handles orientation correctly (no FPA-layout assumption).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import asdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from astropy.wcs import WCS


def rebin_block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean rebin a 2D array. Trailing pixels (if shape % factor != 0) dropped."""
    H, W = arr.shape
    nH, nW = H // factor, W // factor
    arr = arr[: nH * factor, : nW * factor]
    return arr.reshape(nH, factor, nW, factor).mean(axis=(1, 3))


def build_target_wcs(ra0: float, dec0: float, pix_arcsec: float, npx: int) -> WCS:
    """TAN projection at (ra0, dec0); npx x npx pixels at given scale."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [npx / 2 + 0.5, npx / 2 + 0.5]  # 1-indexed FITS center
    cd = pix_arcsec / 3600.0
    # RA increases to the left in the standard view (E to the left).
    w.wcs.cd = [[-cd, 0.0], [0.0, cd]]
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pointing_dir", type=Path, help="Directory of *_l2.asdf for one pointing")
    ap.add_argument("output_png", type=Path)
    ap.add_argument("--rebin", type=int, default=4,
                    help="Source-detector rebin factor (default: 4 -> 1022x1022 per SCA)")
    ap.add_argument("--target-arcsec", type=float, default=0.5,
                    help="Output canvas pixel scale, arcsec (default: 0.5)")
    ap.add_argument("--target-npx", type=int, default=None,
                    help="Output canvas size in pixels (default: auto-fit)")
    ap.add_argument("--vmax-quantile", type=float, default=99.99,
                    help="Quantile of finite pixels used as vmax (default: 99.99)")
    ap.add_argument("--vmin", type=float, default=0.0,
                    help="vmin in data units (default: 0.0; clamps L2 ramp-fit outliers)")
    ap.add_argument("--linear-width", type=float, default=0.5,
                    help="matplotlib AsinhNorm linear_width in data units (default: 0.5; "
                         "tuned for L2 grism rate images: sky ~2 DN/s -> magenta, "
                         "sources stand out as orange-to-yellow streaks)")
    args = ap.parse_args()

    files = sorted(args.pointing_dir.glob("*_l2.asdf"))
    if not files:
        raise SystemExit(f"No *_l2.asdf in {args.pointing_dir}")
    print(f"loaded {len(files)} SCAs from {args.pointing_dir}")

    # First pass: get each detector's pointing-center sky position to estimate
    # the canvas extent and central RA/Dec.
    centers, corners = [], []
    for f in files:
        af = asdf.open(f, lazy_load=True)
        gwcs = af["roman"].meta.wcs
        H, W = af["roman"].data.shape
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        ra_c, dec_c = gwcs(cx, cy)
        centers.append((ra_c, dec_c))
        # Detector corners (0-indexed)
        cx4 = np.array([0, W - 1, W - 1, 0])
        cy4 = np.array([0, 0, H - 1, H - 1])
        ra4, dec4 = gwcs(cx4, cy4)
        corners.append(np.column_stack([ra4, dec4]))

    centers = np.array(centers)
    corners = np.vstack(corners)
    ra0, dec0 = centers.mean(axis=0)
    print(f"pointing center (mean of SCA centers): RA={ra0:.5f}  Dec={dec0:.5f}")

    # Estimate canvas size from corner extents in tangent-plane offsets.
    # cos(dec) factor for RA dimension.
    dra = (corners[:, 0] - ra0) * np.cos(np.deg2rad(dec0))
    ddec = corners[:, 1] - dec0
    half_extent_arcsec = max(np.abs(dra).max(), np.abs(ddec).max()) * 3600.0 * 1.05
    if args.target_npx is None:
        target_npx = int(np.ceil(2 * half_extent_arcsec / args.target_arcsec))
        target_npx = max(target_npx, 256)  # sane minimum
    else:
        target_npx = args.target_npx
    print(f"canvas: {target_npx} x {target_npx} px @ {args.target_arcsec} arcsec/px "
          f"({target_npx * args.target_arcsec / 60:.2f} arcmin per side)")

    twcs = build_target_wcs(ra0, dec0, args.target_arcsec, target_npx)

    sum_grid = np.zeros((target_npx, target_npx), dtype=np.float32)
    n_grid = np.zeros((target_npx, target_npx), dtype=np.int32)

    # Detector-pixel grid for rebinned image (centers in 0-indexed detector pixels).
    rebin = args.rebin

    for f in files:
        af = asdf.open(f, lazy_load=True)
        img = np.asarray(af["roman"].data, dtype=np.float32)
        gwcs = af["roman"].meta.wcs
        H, W = img.shape

        rebinned = rebin_block_mean(img, rebin)
        nH, nW = rebinned.shape

        # Center of each rebinned super-pixel in original detector coords.
        ix = (np.arange(nW) + 0.5) * rebin - 0.5
        iy = (np.arange(nH) + 0.5) * rebin - 0.5
        ix2d, iy2d = np.meshgrid(ix, iy)

        # GWCS forward eval (det px -> sky).
        ra, dec = gwcs(ix2d.ravel(), iy2d.ravel())

        # Sky -> target pixel coords (0-indexed).
        tx, ty = twcs.wcs_world2pix(ra, dec, 0)

        tx_int = np.rint(tx).astype(np.int32)
        ty_int = np.rint(ty).astype(np.int32)

        ok = (
            (tx_int >= 0) & (tx_int < target_npx)
            & (ty_int >= 0) & (ty_int < target_npx)
            & np.isfinite(rebinned).ravel()
        )
        np.add.at(sum_grid, (ty_int[ok], tx_int[ok]), rebinned.ravel()[ok])
        np.add.at(n_grid, (ty_int[ok], tx_int[ok]), 1)

        det = af["roman"].meta.instrument.detector
        print(f"  {det:<6s} placed {ok.sum()}/{rebinned.size} rebinned pixels")

    canvas = np.where(n_grid > 0, sum_grid / np.maximum(n_grid, 1), np.nan)
    finite_frac = np.isfinite(canvas).mean()
    print(f"canvas finite fraction: {finite_frac * 100:.1f}%")

    # Render.
    finite = np.isfinite(canvas)
    if not finite.any():
        raise SystemExit("Empty canvas; check WCS / target scale.")
    vmax = float(np.percentile(canvas[finite], args.vmax_quantile))
    vmin = args.vmin
    print(f"stretch: vmin={vmin:.3f}  vmax={vmax:.3f}  linear_width={args.linear_width}")
    norm = AsinhNorm(linear_width=args.linear_width, vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(1, 1, 1, projection=twcs)
    im = ax.imshow(canvas, origin="lower", cmap="inferno", norm=norm)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(f"{args.pointing_dir.name}\n"
                 f"{len(files)} SCAs  rebin={rebin}  scale={args.target_arcsec}\"/px")
    ax.coords[0].set_major_formatter("d.dd")
    ax.coords[1].set_major_formatter("d.dd")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L2 rate (DN/s)")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=130, bbox_inches="tight")
    print(f"wrote {args.output_png}")


if __name__ == "__main__":
    main()

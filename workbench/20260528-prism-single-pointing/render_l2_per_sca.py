"""Render per-SCA quick-look PNGs from L2 ASDF files for one pointing.

For each *_l2.asdf in a pointing directory, block-averages the L2 rate image,
applies an asinh stretch, and writes a detector-space quick-look PNG (inferno,
origin-lower). This mirrors the disperser-stage per-SCA PNGs but reads the
romanisim L2 rate images (DN/s) instead of the noiseless MODEL.

Usage:
    pixi run -e romanisim python render_l2_per_sca.py \\
        <pointing_dir> [--outdir DIR] [--rebin 4] [--linear-width 0.5]

L2 prism rate images have a sky floor ~3.5 DN/s with ramp-fit negative
outliers, so vmin is clamped to 0 and vmax is taken from a high quantile of
the rebinned finite pixels (sources stand out as bright streaks).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import asdf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def rebin_block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean rebin a 2D array. Trailing pixels (if shape % factor != 0) dropped."""
    H, W = arr.shape
    nH, nW = H // factor, W // factor
    arr = arr[: nH * factor, : nW * factor]
    return arr.reshape(nH, factor, nW, factor).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pointing_dir", type=Path, help="Directory of *_l2.asdf for one pointing")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory for PNGs (default: alongside the ASDF files)")
    ap.add_argument("--rebin", type=int, default=4,
                    help="Block-mean rebin factor (default: 4 -> 1022x1022)")
    ap.add_argument("--vmax-quantile", type=float, default=99.99,
                    help="Quantile of rebinned finite pixels used as vmax (default: 99.99)")
    ap.add_argument("--vmin", type=float, default=0.0,
                    help="vmin in data units (default: 0.0; clamps L2 ramp-fit outliers)")
    ap.add_argument("--linear-width", type=float, default=0.5,
                    help="matplotlib AsinhNorm linear_width in data units (default: 0.5; "
                         "tuned for L2 prism rate images, sky ~3.5 DN/s)")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    files = sorted(args.pointing_dir.glob("*_l2.asdf"))
    if not files:
        raise SystemExit(f"No *_l2.asdf in {args.pointing_dir}")
    outdir = args.outdir or args.pointing_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(files)} SCAs -> {outdir}")

    for f in files:
        af = asdf.open(f, lazy_load=True)
        r = af["roman"]
        img = np.asarray(r.data, dtype=np.float32)
        det = r.meta.instrument.detector
        oe = r.meta.instrument.optical_element

        rebinned = rebin_block_mean(img, args.rebin)
        finite = np.isfinite(rebinned)
        vmax = float(np.percentile(rebinned[finite], args.vmax_quantile))
        norm = AsinhNorm(linear_width=args.linear_width, vmin=args.vmin, vmax=vmax)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(rebinned, origin="lower", cmap="inferno", norm=norm)
        ax.set_title(f"{det}  {oe}  (rebin={args.rebin})  vmax={vmax:.2f} DN/s")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L2 rate (DN/s)")
        fig.tight_layout()

        out_png = outdir / (f.stem + ".png")
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  {det:<6s} vmax={vmax:7.2f}  -> {out_png.name}")

    print("done")


if __name__ == "__main__":
    main()

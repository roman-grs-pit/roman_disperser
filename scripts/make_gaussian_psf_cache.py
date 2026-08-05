#!/usr/bin/env python
"""Synthesize centred-Gaussian PSF caches to isolate the optical model.

The real STPSF PSFs are slightly off-centre in their cutouts, which shows up as a
per-SCA ~0.05 px centroid offset in the line-centering residual. Replacing them
with **perfectly centred, symmetric Gaussians** (same cache format, shape, grid)
should drive that offset to ~0, leaving only the optical model + centroiding —
and a Gaussian is small enough to fit the extraction window and be fit
analytically.

Each real cache `psf_WFI{sca}_{GRISM0|GRISM1}_...npz` is copied verbatim except
`psf_grid`, which is replaced by a centred 2-D Gaussian (fixed FWHM in native px,
normalized to unit sum) at every (spatial, wavelength) slot. Only the SCAs you
pass are built (a single-SCA sim needs that SCA's GRISM0 + GRISM1).
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np


def gaussian_grid(shape, oversample, fwhm_native):
    """(nx, ny, nwl, P, P) grid of identical centred unit-sum Gaussians."""
    P = shape[-1]
    sigma = (fwhm_native * oversample) / 2.3548   # oversampled px
    c = (P - 1) / 2.0
    y, x = np.mgrid[0:P, 0:P]
    g = np.exp(-0.5 * (((x - c) ** 2 + (y - c) ** 2) / sigma**2))
    g /= g.sum()
    grid = np.empty(shape, dtype=np.float32)
    grid[...] = g.astype(np.float32)             # broadcast into all slots
    return grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scas", type=int, nargs="+", default=[5])
    p.add_argument("--fwhm-native", type=float, default=2.5,
                   help="Gaussian FWHM in native detector px (default 2.5).")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    from roman_disperser.paths import data_dir
    src = str(data_dir() / "psf_cache")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for sca in args.scas:
        for order in ("GRISM0", "GRISM1"):
            f = glob.glob(f"{src}/psf_WFI{sca:02d}_{order}*.npz")[0]
            d = dict(np.load(f))
            ov = int(d["oversample"])
            d["psf_grid"] = gaussian_grid(d["psf_grid"].shape, ov, args.fwhm_native)
            dest = out / Path(f).name
            np.savez(dest, **d)
            print(f"wrote {dest.name}  (FWHM {args.fwhm_native} px, os {ov})")


if __name__ == "__main__":
    main()

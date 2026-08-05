#!/usr/bin/env python
"""Compare two per-SCA FITS products (MODEL and ISIM extensions).

Usage:
    python compare_gate.py A.fits B.fits [--label "branch vs main"]

Prints, per extension:
- rel_sum_diff = |sum(A) - sum(B)| / sum(B)  — the acceptance-gate statistic;
  measured GPU run-to-run noise on identical code/keys is ~1e-9 (issue #22 /
  pipeline.make_sca_keys docstring).
- max |A-B|, and max relative diff over pixels where |B| > 1e-6 * max(B).
- fraction of pixels failing np.allclose(rtol=1e-5, atol=1e-8*max(B)).
- flux-weighted centroid of each image and the shift between them, in pixels
  — separates a pure position shift (expected when comparing against
  pre-v0.11/v0.12 archives: flat-sky -> gnomonic projection moved source
  positions at the sub-pixel level) from a flux discrepancy.
- for ISIM additionally: number of pixels with differing counts (integer
  content, so exact-zero equality is meaningful only when the RNG keys and
  scatter-add order agree; a handful of ~1-count flips is the measured GPU
  nondeterminism at fixed keys, ~88/2.9M in the 2026-07-31 measurement).

No pass/fail verdict is printed — thresholds live with the writeup, not the
tool. Exit code is always 0 unless the files can't be read.
"""

import argparse

import numpy as np
from astropy.io import fits


def centroid(img):
    """Flux-weighted centroid (x, y) in 0-indexed pixels; clips negatives."""
    img = np.clip(img, 0, None).astype(np.float64)
    tot = img.sum()
    if tot == 0:
        return (np.nan, np.nan)
    ny, nx = img.shape
    y = (img.sum(axis=1) @ np.arange(ny)) / tot
    x = (img.sum(axis=0) @ np.arange(nx)) / tot
    return (x, y)


def compare_ext(a, b, name):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    sum_a, sum_b = a64.sum(), b64.sum()
    rel_sum = abs(sum_a - sum_b) / abs(sum_b) if sum_b != 0 else np.inf
    diff = a64 - b64
    max_abs = np.abs(diff).max()
    bmax = np.abs(b64).max()
    sig = np.abs(b64) > 1e-6 * bmax
    max_rel = (np.abs(diff[sig]) / np.abs(b64[sig])).max() if sig.any() else 0.0
    atol = 1e-8 * bmax
    n_fail = int((np.abs(diff) > atol + 1e-5 * np.abs(b64)).sum())
    cxa, cya = centroid(a64)
    cxb, cyb = centroid(b64)
    print(f"  [{name}]")
    print(f"    sum(A)={sum_a:.9e}  sum(B)={sum_b:.9e}  "
          f"rel_sum_diff={rel_sum:.3e}")
    print(f"    max|A-B|={max_abs:.3e}  max_rel(|B|>1e-6*max)={max_rel:.3e}")
    print(f"    allclose(rtol=1e-5, atol=1e-8*max) failing pixels: "
          f"{n_fail} / {a.size}")
    print(f"    centroid A=({cxa:.4f},{cya:.4f})  B=({cxb:.4f},{cyb:.4f})  "
          f"shift=({cxa-cxb:+.4f},{cya-cyb:+.4f}) px")
    if name == "ISIM":
        n_diff = int((a64 != b64).sum())
        max_cdiff = np.abs(diff).max()
        print(f"    differing-count pixels: {n_diff} / {a.size}  "
              f"(max count diff {max_cdiff:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label", default="A vs B")
    args = ap.parse_args()

    with fits.open(args.file_a) as ha, fits.open(args.file_b) as hb:
        print(f"== {args.label} ==")
        print(f"A: {args.file_a}")
        print(f"   GITSHA={ha[0].header.get('GITSHA', '?')[:12]} "
              f"CODEVER={ha[0].header.get('CODEVER', '?')} "
              f"OPTELEM={ha[0].header.get('OPTELEM', '?')} "
              f"RNDSEED=({ha[0].header.get('RNDSEED0','?')},"
              f"{ha[0].header.get('RNDSEED1','?')})")
        print(f"B: {args.file_b}")
        print(f"   GITSHA={hb[0].header.get('GITSHA', '?')[:12]} "
              f"CODEVER={hb[0].header.get('CODEVER', '?')} "
              f"OPTELEM={hb[0].header.get('OPTELEM', '?')} "
              f"RNDSEED=({hb[0].header.get('RNDSEED0','?')},"
              f"{hb[0].header.get('RNDSEED1','?')})")
        for name in ("MODEL", "ISIM"):
            compare_ext(ha[name].data, hb[name].data, name)


if __name__ == "__main__":
    main()

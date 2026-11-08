#!/usr/bin/env python
"""Run ``romanisim-make-image`` with a disperser rate image injected as a source.

``romanisim-make-image --extra-counts FILE ISIM`` adds the disperser's counts
*after* romanisim has applied quantum efficiency and Poisson noise, but the L2
step still divides every pixel by the flat field. The dispersed light therefore
skips the QE that real sources get and then comes out of L2 divided by the
flat: a flat-field imprint (~9 % rms) and a mean offset (T45/T48 measured
0.87-0.91x).

This worker instead adds ``MODEL * exptime`` (MODEL in electrons/s, noiseless)
to the image at the point where romanisim renders its own sources, inside
``simulate_counts_generic``. From there it gets exactly what a romanisim
source gets: Poisson noise, binomial QE thinning by the relative QE derived
from the CRDS flat, then dark, the L1 ramp (IPC, CRs, non-linearity, read
noise) and the L2 ramp fit. ``exptime`` is romanisim's own MA-table exposure
time, so the header ``EXPTIME`` no longer has to agree with it.

It also skips building the PSF when there are no romanisim catalog sources
(``--nobj 0``): romanisim builds STPSF PSFs unconditionally, ~20 s per frame
that nothing uses (T95).

Usage (romanisim env; all other arguments go to romanisim-make-image):

    python romanisim_inject_rate.py --rate-file F.fits [--rate-hdu MODEL] \\
        -- out_l2.asdf --radec ... --sca 5 --nobj 0 ...
"""
from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
import time

import numpy as np
from astropy.io import fits


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        sys.exit("usage: romanisim_inject_rate.py --rate-file F [--rate-hdu H] "
                 "[--keep-psf] -- <romanisim-make-image args>")
    split = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--rate-file", required=True)
    p.add_argument("--rate-hdu", default="MODEL")
    p.add_argument("--keep-psf", action="store_true",
                   help="build the PSF even with no catalog sources")
    a = p.parse_args(argv[:split])
    ris_args = argv[split + 1:]

    with fits.open(a.rate_file) as h:
        rate = np.asarray(h[a.rate_hdu].data, dtype=np.float64)
        bunit = h[a.rate_hdu].header.get("BUNIT", "")
    if not np.all(np.isfinite(rate)) or rate.min() < 0:
        bad = (~np.isfinite(rate)).sum() + (rate < 0).sum()
        print(f"[inject] clipping {bad} non-finite/negative rate pixels to 0")
        rate = np.where(np.isfinite(rate) & (rate > 0), rate, 0.0)
    print(f"[inject] {a.rate_file}[{a.rate_hdu}] BUNIT={bunit!r} "
          f"shape={rate.shape} total={rate.sum():.6g} e/s")

    import romanisim.image as rimage
    import romanisim.psf as rpsf

    orig_generic = rimage.simulate_counts_generic
    state = {"done": False}

    def generic(image, exptime, *args, qe=None, **kw):
        if state["done"]:
            raise RuntimeError("simulate_counts_generic called twice")
        arr = image.array
        if arr.shape != rate.shape:
            raise ValueError(f"rate shape {rate.shape} != image {arr.shape}")
        q = 1.0 if qe is None else qe
        # Must match simulate_counts_generic exactly: it takes maxqe = max(qe)
        # BEFORE clipping qe to 2, inflates rendered counts by maxqe and then
        # thins every pixel by clip(qe, 0, 2) / maxqe. With no CRDS flat (the
        # grism and prism have none) qe = gain/area renormalized, whose max is
        # far above 2, so using the clipped max loses ~99 % of the light.
        maxqe = float(np.max(q))
        arr += (rate * exptime * maxqe).astype(arr.dtype)
        state.update(done=True, exptime=float(exptime), maxqe=maxqe,
                     qe_median=float(np.median(np.clip(q, 0, 2))))
        print(f"[inject] added rate*{exptime:.3f}s*maxqe({maxqe:.4f}) before "
              f"Poisson + QE thinning (median qe {state['qe_median']:.4f})")
        before = float(np.median(arr))
        if os.environ.get("INJECT_DEBUG"):
            def st(v):
                v = getattr(v, "array", v)
                return "scalar %r" % (v,) if np.ndim(v) == 0 else "median %.4g p1 %.4g p99 %.4g" % (
                    np.median(v), *np.percentile(v, [1, 99]))
            print("[inject-debug] args", [type(x).__name__ for x in args],
                  {k: (st(v) if k in ("sky", "dark", "qe") else type(v).__name__) for k, v in kw.items()},
                  "qe", st(q), flush=True)
        res = orig_generic(image, exptime, *args, qe=qe, **kw)
        print(f"[inject] counts median: {before:.1f} after injection, "
              f"{float(np.median(image.array)):.1f} after noise/QE/dark", flush=True)
        return res

    rimage.simulate_counts_generic = generic

    nobj0 = "--nobj" in ris_args and ris_args[ris_args.index("--nobj") + 1] == "0"
    if nobj0 and not a.keep_psf:
        def no_psf(*args, **kw):
            print("[inject] --nobj 0: PSF build skipped")
            return None
        rpsf.make_psf = no_psf

    exe = shutil.which("romanisim-make-image")
    if exe is None:
        sys.exit("romanisim-make-image not on PATH (activate the romanisim env)")
    t0 = time.monotonic()
    sys.argv = [exe] + ris_args
    runpy.run_path(exe, run_name="__main__")
    if not state["done"]:
        sys.exit("[inject] rate was never injected (romanisim code path changed?)")
    print(f"[inject] romanisim done in {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()

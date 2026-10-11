"""Compare the A (sim-layout) and B (healpix-layout) renders of 4-roll VISIT 1.

Timing: parses the per-SCA "Galaxy SEDs: ... in <t>s" and "Total: <t>s" lines
from the worker logs of A, B and the original array-10117 worker 1 (REF).

Images: for each SCA, checks MODEL (noiseless count rate) of B against A and
of A against REF with the GPU golden-frame tolerance from
tests/test_golden_frame.py: ``allclose(new, ref, rtol=1e-5, atol=1e-5 *
max|ref|)``, plus max |diff| and the relative difference of the image sums.
The store change is a pure copy of the SED bytes, so the only expected
difference is GPU scatter-add ordering (last bit), which that tolerance absorbs.

ISIM (Poisson realisation) is *reported, not gated*: last-bit MODEL noise
flips a handful of Poisson draws run-to-run even with identical keys (see
``pipeline.make_sca_keys``), so ISIM can never be bitwise-stable on GPU. We
print the number of differing ISIM pixels, how many of those have a
bit-identical MODEL (expected 0), and the largest flip in units of
sqrt(expected counts).

Usage: pixi run python compare.py   (any env with astropy)
"""

import re
from pathlib import Path

import numpy as np
from astropy.io import fits

OUT = Path("/mnt/roman-science/grs/20261011_sed-store-ab")
REF = Path("/mnt/roman-science/grs/20261007_4roll-ra10dec0-v0151/spectro/2026-10-07/4roll")
POINTING = "4roll-ra10dec0.sim_001.001.001.001.001.001"
RTOL, ATOL_SCALE = 1e-5, 1e-5


def parse_log(path):
    """Per-SCA SED-read and total seconds, plus the run total, from a worker log."""
    sed, tot, sca = {}, {}, None
    run_total = None
    for line in Path(path).read_text().splitlines():
        if m := re.match(r"\s+SCA (\d+):", line):
            sca = int(m.group(1))
        elif m := re.search(r"Galaxy SEDs: .* in ([\d.]+)s", line):
            sed[sca] = float(m.group(1))
        elif m := re.match(r"\s+Total: ([\d.]+)s,", line):
            tot[sca] = float(m.group(1))
        elif m := re.match(r"\s+Total:\s+([\d.]+)s$", line):
            run_total = float(m.group(1))
    return sed, tot, run_total


def fits_path(root, sca):
    return root / POINTING / f"grism_{POINTING}_detSCA{sca:02d}.fits"


EXPTIME = 190.22287  # s, all 4-roll visits (MA table 1036)


def compare(new_path, ref_path):
    """MODEL gate (ok, max|d|, rel dsum) and ISIM flip stats (n, n_same_model, max sigma)."""
    with fits.open(new_path) as hn, fits.open(ref_path) as hr:
        mn, mr = hn["MODEL"].data, hr["MODEL"].data
        inew, iref = hn["ISIM"].data, hr["ISIM"].data
    new, ref = mn.astype(np.float64), mr.astype(np.float64)
    atol = ATOL_SCALE * np.abs(ref).max()
    model = (bool(np.allclose(new, ref, rtol=RTOL, atol=atol)),
             float(np.abs(new - ref).max()),
             float(abs(new.sum() - ref.sum()) / abs(ref.sum())))
    di = inew != iref
    lam = np.maximum(ref[di] * EXPTIME, 1.0)
    nsig = (np.abs(inew[di].astype(np.float64) - iref[di]) / np.sqrt(lam)).max() if di.any() else 0.0
    isim = (int(di.sum()), int((di & (mn == mr)).sum()), float(nsig))
    return model, isim


def main():
    logs = {"REF": REF / "logs/worker-1.log",
            "A": OUT / "A/logs/worker.log", "B": OUT / "B/logs/worker.log"}
    parsed = {k: parse_log(v) for k, v in logs.items()}

    print("Per-SCA galaxy-SED read / SCA total [s]")
    print(f"{'SCA':>3} | {'REF sed':>8} {'A sed':>8} {'B sed':>8} | "
          f"{'REF tot':>8} {'A tot':>8} {'B tot':>8}")
    for sca in range(1, 19):
        row = [parsed[k][0].get(sca, np.nan) for k in ("REF", "A", "B")]
        row += [parsed[k][1].get(sca, np.nan) for k in ("REF", "A", "B")]
        print(f"{sca:>3} | " + " ".join(f"{x:8.1f}" for x in row[:3]) + " | "
              + " ".join(f"{x:8.1f}" for x in row[3:]))
    for k in ("REF", "A", "B"):
        sed, tot, run_total = parsed[k]
        print(f"{k:>3}: sum SED {sum(sed.values()):7.0f}s, sum SCA {sum(tot.values()):7.0f}s, "
              f"run total {run_total}")

    print("\nMODEL gate (golden GPU tolerance rtol=1e-5, atol=1e-5*max|ref|); "
          "ISIM flips reported only")
    n_fail = 0
    for label, new_root, ref_root in (("B vs A", OUT / "B/output", OUT / "A/output"),
                                      ("A vs REF", OUT / "A/output", REF / "output")):
        for sca in range(1, 19):
            (ok, dmax, dsum), (nflip, nflip_same, nsig) = compare(
                fits_path(new_root, sca), fits_path(ref_root, sca))
            n_fail += not ok
            print(f"  {label} SCA{sca:02d}: MODEL {'ok' if ok else 'FAIL'} "
                  f"max|d|={dmax:.2e} dsum={dsum:.1e} | ISIM flips {nflip:3d} "
                  f"(same-MODEL {nflip_same}), max {nsig:.2f} sigma")
    print("PASS" if n_fail == 0 else f"FAIL ({n_fail} SCA comparisons)")


if __name__ == "__main__":
    main()

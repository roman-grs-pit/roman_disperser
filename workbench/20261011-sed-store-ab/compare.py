"""Compare the A (sim-layout) and B (healpix-layout) renders of 4-roll VISIT 1.

Timing: parses the per-SCA "Galaxy SEDs: ... in <t>s" and "Total: <t>s" lines
from the worker logs of A, B and the original array-10117 worker 1 (REF).

Images: for each SCA and each image extension (MODEL = noiseless count rate,
ISIM = with noise), checks B against A and A against REF with the GPU golden-
frame tolerance from tests/test_golden_frame.py:
``allclose(new, ref, rtol=1e-5, atol=1e-5 * max|ref|)``. The store change is a
pure copy of the SED bytes, so the only expected difference is GPU atomic-add
ordering (last-bit), which that tolerance absorbs. Also reports max |diff| and
the relative difference of the image sums.

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


def compare(new_path, ref_path):
    out = {}
    with fits.open(new_path) as hn, fits.open(ref_path) as hr:
        for ext in ("MODEL", "ISIM"):
            new = hn[ext].data.astype(np.float64)
            ref = hr[ext].data.astype(np.float64)
            atol = ATOL_SCALE * np.abs(ref).max()
            out[ext] = (bool(np.allclose(new, ref, rtol=RTOL, atol=atol)),
                        float(np.abs(new - ref).max()),
                        float(abs(new.sum() - ref.sum()) / abs(ref.sum())))
    return out


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

    print("\nImages (golden GPU tolerance rtol=1e-5, atol=1e-5*max|ref|)")
    n_fail = 0
    for label, new_root, ref_root in (("B vs A", OUT / "B/output", OUT / "A/output"),
                                      ("A vs REF", OUT / "A/output", REF / "output")):
        for sca in range(1, 19):
            r = compare(fits_path(new_root, sca), fits_path(ref_root, sca))
            ok = all(v[0] for v in r.values())
            n_fail += not ok
            print(f"  {label} SCA{sca:02d}: " + "  ".join(
                f"{e} {'ok' if v[0] else 'FAIL'} max|d|={v[1]:.2e} dsum={v[2]:.1e}"
                for e, v in r.items()))
    print("PASS" if n_fail == 0 else f"FAIL ({n_fail} SCA comparisons)")


if __name__ == "__main__":
    main()

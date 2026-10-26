"""Compare the OOM-fix verification runs (see submit.sh in this directory).

Runs (Keith's XMM-LSS part_03 row 0, unfiltered catalog, A10G):
  REPRO = v0.15.1 path, our env (job 12200; SCAs 1-8, OOM at SCA 9)
  BASE  = perf/jit-payload-args, no fix (job 12218; SCAs 1-4)
  FIX   = fix/sersic-batching, per-batch Sersic (job 12219; all 18 SCAs)
  JITS  = jitted per-batch Sersic (../jitsersic; SCAs 1-4) -- gated vs FIX

Timing: per SCA, JIT, galaxy-SED read, dispersion ms/galaxy per order, SCA
total, and the GPU running peak / host VmHWM from the "Memory:" line.

Images: FIX vs BASE on the SCAs both have, MODEL gated at the GPU golden-frame
tolerance (allclose rtol=1e-5, atol=1e-5*max|ref|), as in
../../20261011-sed-store-ab/compare.py. The fix is bit-identical on CPU, so
the only expected GPU difference is scatter-add ordering. ISIM flips are
reported, not gated (last-bit MODEL noise flips Poisson draws run to run).
FIX vs REPRO is also reported for SCAs 1-8 for context: those differ by
design (jit-payload-args pixel flips, pending golden v3), so not gated.

Usage: pixi run python compare.py   (any env with astropy)
"""

import re
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path("/mnt/roman-science/grs/20261012_xmm-scale-perf")
RUNS = {"REPRO": ROOT / "repro", "BASE": ROOT / "oomfix/base", "FIX": ROOT / "oomfix/fix",
        "JITS": ROOT / "jitsersic/run"}
POINTING = "part_03_001.087.001.001.021.002"
EXPTIME = 190.22287  # s (MA table 1036)
RTOL, ATOL_SCALE = 1e-5, 1e-5


def parse_log(path):
    """Per-SCA dict of timing / memory fields from a worker log."""
    out, sca = {}, None
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if m := re.match(r"\s+SCA (\d+):$", line):
            sca = int(m.group(1)); out[sca] = {"ms": {}}
        elif sca is None:
            continue
        elif m := re.search(r"JIT compile: ([\d.]+)s", line):
            out[sca]["jit"] = float(m.group(1))
        elif m := re.search(r"Selected: \d+ stars, (\d+) galaxies", line):
            out[sca]["ngal"] = int(m.group(1))
        elif m := re.search(r"Galaxy SEDs: .* in ([\d.]+)s", line):
            out[sca]["sed"] = float(m.group(1))
        elif m := re.search(r"Order (\w+): \d+ galaxies in [\d.]+s \(([\d.]+) ms/galaxy\)", line):
            out[sca]["ms"][m.group(1)] = float(m.group(2))
        elif m := re.match(r"\s+Total: ([\d.]+)s,", line):
            out[sca]["total"] = float(m.group(1))
        elif m := re.search(r"peak=([\d.]+) limit=.*host VmHWM=([\d.]+)", line):
            out[sca]["gpu_peak"], out[sca]["host"] = float(m.group(1)), float(m.group(2))
    return out


def fits_path(run, sca):
    return RUNS[run] / "output" / POINTING / f"grism_{POINTING}_detSCA{sca:02d}.fits"


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
    return model, (int(di.sum()), int((di & (mn == mr)).sum()), float(nsig))


def main():
    logs = {k: parse_log(v / "logs/worker.log") for k, v in RUNS.items()}
    print("Per SCA: N_gal | JIT s | SED s | ms/gal o0/o1/o2 | total s | GPU peak GiB | host GiB")
    for sca in range(1, 19):
        for k in RUNS:
            d = logs[k].get(sca)
            if not d or "total" not in d:
                continue
            ms = "/".join(f"{d['ms'].get(o, np.nan):.1f}" for o in ("0", "1", "2"))
            print(f"  SCA{sca:02d} {k:5s} {d.get('ngal', 0):6d} | {d.get('jit', np.nan):6.1f} | "
                  f"{d.get('sed', np.nan):6.1f} | {ms:>13s} | {d['total']:7.1f} | "
                  f"{d.get('gpu_peak', np.nan):5.2f} | {d.get('host', np.nan):5.2f}")
    for k in RUNS:
        done = [d for d in logs[k].values() if "total" in d]
        if done:
            print(f"{k:5s}: {len(done)} SCAs done, sum total {sum(d['total'] for d in done):.0f} s, "
                  f"sum JIT {sum(d.get('jit', 0) for d in done):.0f} s, "
                  f"sum SED {sum(d.get('sed', 0) for d in done):.0f} s, "
                  f"max GPU peak {max(d.get('gpu_peak', 0) for d in done):.2f} GiB")

    print("\nFIX vs BASE: MODEL gate (rtol=1e-5, atol=1e-5*max|ref|); ISIM flips reported only")
    n_fail = n_cmp = 0
    for sca in range(1, 19):
        if not (fits_path("FIX", sca).exists() and fits_path("BASE", sca).exists()):
            continue
        (ok, dmax, dsum), (nf, nfs, nsig) = compare(fits_path("FIX", sca), fits_path("BASE", sca))
        n_cmp += 1; n_fail += not ok
        print(f"  SCA{sca:02d}: MODEL {'ok' if ok else 'FAIL'} max|d|={dmax:.2e} dsum={dsum:.1e} | "
              f"ISIM flips {nf} (same-MODEL {nfs}), max {nsig:.2f} sigma")
    print("\nJITS vs FIX: MODEL gate (jitted vs un-jitted Sersic; ~1e-8 of max expected)")
    for sca in range(1, 19):
        if not (fits_path("JITS", sca).exists() and fits_path("FIX", sca).exists()):
            continue
        (ok, dmax, dsum), (nf, nfs, nsig) = compare(fits_path("JITS", sca), fits_path("FIX", sca))
        n_cmp += 1; n_fail += not ok
        print(f"  SCA{sca:02d}: MODEL {'ok' if ok else 'FAIL'} max|d|={dmax:.2e} dsum={dsum:.1e} | "
              f"ISIM flips {nf} (same-MODEL {nfs}), max {nsig:.2f} sigma")
    print("\nFIX vs REPRO (context only; jit-branch pixel flips expected)")
    for sca in range(1, 19):
        if fits_path("FIX", sca).exists() and fits_path("REPRO", sca).exists():
            (ok, dmax, dsum), _ = compare(fits_path("FIX", sca), fits_path("REPRO", sca))
            print(f"  SCA{sca:02d}: within-tol={ok} max|d|={dmax:.2e} dsum={dsum:.1e}")
    print(f"\n{'PASS' if n_cmp and n_fail == 0 else 'FAIL' if n_fail else 'NO COMPARISONS'} "
          f"({n_cmp} gated SCA comparisons, {n_fail} failed)")


if __name__ == "__main__":
    main()

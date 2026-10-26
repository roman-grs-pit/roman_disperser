"""Digest the step-6 timing runs (see submit.sh in this directory).

Runs (Keith's XMM-LSS part_03 row 0, 18 SCAs, grism, A10G):
  OLD  = fix/sersic-batching on the sim-layout store, no cuts (job 12219)
  full = feature/source-cuts on the HEALPix store, no cuts      (job 12269)
  h27  = as full, F158 <= 27 mag + r_hl > 0 for SER              (job 12270)
  h265 = as full, F158 <= 26.5 mag + r_hl > 0 for SER            (job 12271)

Timing per SCA from the worker logs (parser shared with ../oomfix/compare.py).
Images: full vs OLD, MODEL gated at the GPU golden tolerance. Same selection,
same renderer; only the SED store layout differs, so the SEDs must be the same
bytes and MODEL must agree to GPU scatter-add noise.

Usage: pixi run python compare.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oomfix"))
from compare import compare, parse_log  # noqa: E402

ROOT = Path("/mnt/roman-science/grs/20261012_xmm-scale-perf")
RUNS = {"OLD": ROOT / "oomfix/fix", "full": ROOT / "timing/full",
        "h27": ROOT / "timing/h27", "h265": ROOT / "timing/h265"}
POINTING = "part_03_001.087.001.001.021.002"


def fits_path(run, sca):
    return RUNS[run] / "output" / POINTING / f"grism_{POINTING}_detSCA{sca:02d}.fits"


def main():
    logs = {k: parse_log(v / "logs/worker.log") for k, v in RUNS.items()}
    print("Per SCA: N_gal | JIT s | SED s | ms/gal o0/o1/o2 | total s | GPU peak GiB")
    for sca in range(1, 19):
        for k in RUNS:
            d = logs[k].get(sca)
            if not d or "total" not in d:
                continue
            ms = "/".join(f"{d['ms'].get(o, np.nan):.1f}" for o in ("0", "1", "2"))
            print(f"  SCA{sca:02d} {k:4s} {d.get('ngal', 0):6d} | {d.get('jit', np.nan):5.1f} | "
                  f"{d.get('sed', np.nan):6.1f} | {ms:>13s} | {d['total']:7.1f} | "
                  f"{d.get('gpu_peak', np.nan):5.2f}")
    print("\nRun   SCAs  N_gal     total_s  JIT_s  SED_s  disp+other_s  s/1k gal  max GPU GiB")
    for k in RUNS:
        done = [d for d in logs[k].values() if "total" in d]
        if not done:
            continue
        n = sum(d.get("ngal", 0) for d in done)
        t = sum(d["total"] for d in done)
        j = sum(d.get("jit", 0) for d in done)
        s = sum(d.get("sed", 0) for d in done)
        print(f"{k:5s} {len(done):4d} {n:7d} {t:10.0f} {j:6.0f} {s:6.0f} {t - j - s:12.0f} "
              f"{1000 * t / n:9.1f} {max(d.get('gpu_peak', 0) for d in done):11.2f}")

    print("\nfull vs OLD: MODEL gate (rtol=1e-5, atol=1e-5*max|ref|); ISIM flips reported only")
    n_fail = n_cmp = 0
    for sca in range(1, 19):
        if not (fits_path("full", sca).exists() and fits_path("OLD", sca).exists()):
            continue
        (ok, dmax, dsum), (nf, nfs, nsig) = compare(fits_path("full", sca), fits_path("OLD", sca))
        n_cmp += 1; n_fail += not ok
        print(f"  SCA{sca:02d}: MODEL {'ok' if ok else 'FAIL'} max|d|={dmax:.2e} dsum={dsum:.1e} | "
              f"ISIM flips {nf} (same-MODEL {nfs}), max {nsig:.2f} sigma")
    print(f"\n{'PASS' if n_cmp and n_fail == 0 else 'FAIL'} ({n_cmp} SCA comparisons, {n_fail} failed)")


if __name__ == "__main__":
    main()

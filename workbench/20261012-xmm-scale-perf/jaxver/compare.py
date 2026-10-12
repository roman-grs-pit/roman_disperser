"""Digest step-7 job 12328 (see submit.sh in this directory).

Per env (jax 0.7.2 control / conda-forge 0.10.2 / PyPI 0.11.2), all on one A10G
node: per-SCA timing for the h27 pipeline on SCAs 1-4, and MODEL images vs the
step-6 h27 run (job 12270: same code, same config, jax 0.7.2, another node),
gated at the GPU golden tolerance (allclose rtol=1e-5, atol=1e-5*max|ref|).

Usage: pixi run python compare.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oomfix"))
from compare import compare, parse_log  # noqa: E402

ROOT = Path("/mnt/roman-science/grs/20261012_xmm-scale-perf")
REF = ROOT / "timing/h27"
ENVS = {"cuda": "0.7.2 cf", "cuda-cf010": "0.10.2 cf", "cuda-pypi011": "0.11.2 pypi"}
POINTING = "part_03_001.087.001.001.021.002"
SCAS = (1, 2, 3, 4)


def fits_path(run_dir, sca):
    return run_dir / "output" / POINTING / f"grism_{POINTING}_detSCA{sca:02d}.fits"


def main():
    ref_log = parse_log(REF / "logs/worker.log")
    print("Per SCA: N_gal | JIT s | SED s | ms/gal o0/o1/o2 | total s | GPU peak GiB")
    rows = [("12270 (0.7.2)", ref_log)] + [
        (f"{e} ({v})", parse_log(ROOT / "jaxver" / e / "logs/worker.log")) for e, v in ENVS.items()]
    for sca in SCAS:
        for name, log in rows:
            d = log.get(sca, {})
            if "total" not in d:
                print(f"  SCA{sca:02d} {name:24s} missing"); continue
            ms = "/".join(f"{d['ms'].get(o, np.nan):.1f}" for o in ("0", "1", "2"))
            print(f"  SCA{sca:02d} {name:24s} {d.get('ngal', 0):6d} | {d.get('jit', np.nan):5.1f} | "
                  f"{d.get('sed', np.nan):5.1f} | {ms:>14s} | {d['total']:6.1f} | {d.get('gpu_peak', np.nan):5.2f}")
    print("\nSum over SCAs 1-4: total s, JIT s")
    for name, log in rows:
        ds = [log[s] for s in SCAS if "total" in log.get(s, {})]
        print(f"  {name:24s} {sum(d['total'] for d in ds):7.1f} {sum(d.get('jit', 0) for d in ds):6.1f}")

    print("\nMODEL vs 12270 (rtol=1e-5, atol=1e-5*max|ref|); ISIM flips reported only")
    n_fail = n_cmp = 0
    for e in ENVS:
        for sca in SCAS:
            new, ref = fits_path(ROOT / "jaxver" / e, sca), fits_path(REF, sca)
            if not (new.exists() and ref.exists()):
                print(f"  {e} SCA{sca:02d}: missing"); continue
            (ok, dmax, dsum), (nf, nfs, nsig) = compare(new, ref)
            n_cmp += 1; n_fail += not ok
            print(f"  {e:13s} SCA{sca:02d}: MODEL {'ok' if ok else 'FAIL'} max|d|={dmax:.2e} "
                  f"dsum={dsum:.1e} | ISIM flips {nf} (same-MODEL {nfs}), max {nsig:.2f} sigma")
    print(f"\n{'PASS' if n_cmp and not n_fail else 'FAIL'} ({n_cmp} comparisons, {n_fail} failed)")


if __name__ == "__main__":
    main()

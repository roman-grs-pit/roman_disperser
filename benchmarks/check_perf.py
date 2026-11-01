"""Gate a bench_deposit.py result JSON against performance-regression limits.

Two gates, applied per grism order:

1. **Scatter ratio** (always applied, hardware-insensitive):
   min-over-repeats baseline ms/gal divided by min-over-repeats noscatter
   ms/gal must be <= --max-ratio (default 1.5). Healthy jax on GPU measures
   ~0.73-0.77 since the native-deposit port (the fused deposit kernel beats
   noscatter's separate reduction; pre-native16 it was ~1.1-1.2). The jax
   0.11.0 scatter-add regression (jax-ml/jax#39959) measured 15-21 on the
   old deposit; conda-forge jax 0.10.2 measured 1.9-2.8 (2026-10-12), which
   the old default of 3.0 let through -- only the absolute gate caught it.
   1.5 keeps ~2x headroom over healthy. The threshold was calibrated on GPU;
   CPU runs are reported but ratio behaviour there has not been
   characterised — use --max-ratio to loosen if gating CPU.

2. **Absolute time** (only when the result's device has an entry in the
   baselines file): baseline ms/gal must be <= recorded reference *
   --max-slowdown (default 1.5). Catches uniform slowdowns that leave the
   ratio unchanged (e.g. a regression in the shared interp/FFT path). The
   same limit applies to the production-throughput rows (``ms_per_src`` of
   stars and galaxies at the ``default`` and ``floor`` settings) when the
   baseline has them (``ms_per_src: {"star/default": {order: ms}, ...}``).

Devices are keyed by the ``nvidia-smi`` GPU name for GPU results
(``baselines/gpu.json``) and by the CPU model string for CPU results
(``baselines/cpu.json``); the default baselines file follows the result's
backend.

The min over repeats is used as the steady-state estimator (repeats after
warmup; min is the least-noise choice for wall-clock benchmarks).

Exit status: 0 = all gates pass, 1 = any gate fails, 2 = malformed input.

Usage:
    python benchmarks/check_perf.py result.json [result2.json ...] \
        [--baselines benchmarks/baselines/gpu.json] [--max-ratio 1.5] \
        [--max-slowdown 1.5]
"""

import argparse
import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).parent / "baselines"
DEFAULT_BASELINES = BASELINE_DIR / "gpu.json"


def device_key(meta):
    """The GPU name, or for CPU results the CPU model string."""
    return meta.get("cpu", "unknown") if meta.get("backend") == "cpu" \
        else meta.get("gpu", "unknown")


def check_file(path, baselines, max_ratio, max_slowdown):
    data = json.loads(Path(path).read_text())
    meta, results = data["meta"], data["results"]
    gpu = device_key(meta)
    if baselines is None:
        bp = BASELINE_DIR / ("cpu.json" if meta.get("backend") == "cpu"
                             else "gpu.json")
        baselines = json.loads(bp.read_text()) if bp.exists() else {}
    print(f"== {path}: jax {meta['jax']} on {gpu} "
          f"(backend {meta.get('backend')}, commit {meta.get('code_commit')})")

    by_order = {}
    for r in results:
        if "ms_per_gal" in r:
            by_order.setdefault(r["order"], {})[r["variant"]] = min(r["ms_per_gal"])

    ref = baselines.get(gpu)
    if ref is None:
        print(f"   (no absolute baseline recorded for {gpu!r}; "
              f"ratio gate only)")

    ok = True
    if not by_order:
        print("   (no gate rows: ratio gate skipped)")
    for order, v in sorted(by_order.items()):
        if "baseline" not in v or "noscatter" not in v:
            print(f"   order {order}: missing variant "
                  f"(have {sorted(v)}) — cannot gate")
            ok = False
            continue
        ratio = v["baseline"] / v["noscatter"]
        line = (f"   order {order}: baseline {v['baseline']:7.2f} ms/gal, "
                f"noscatter {v['noscatter']:7.2f}, ratio {ratio:5.2f}")
        order_ok = True
        if ratio > max_ratio:
            line += f"  FAIL ratio > {max_ratio}"
            order_ok = False
        if ref is not None:
            ref_ms = ref["ms_per_gal"][order]
            limit = ref_ms * max_slowdown
            line += f" | ref {ref_ms:.2f} ms, limit {limit:.2f}"
            if v["baseline"] > limit:
                line += f"  FAIL abs > {max_slowdown}x ref"
                order_ok = False
        print(line + ("  ok" if order_ok else ""))
        ok = ok and order_ok

    ref_src = (ref or {}).get("ms_per_src", {})
    for r in results:
        if "ms_per_src" not in r:
            continue
        key = f"{r['kind']}/{r['variant']}"
        got = min(r["ms_per_src"])
        line = f"   order {r['order']}: {key:15s} {got:8.2f} ms/src"
        row_ok = True
        ref_ms = ref_src.get(key, {}).get(r["order"])
        if ref_ms is not None:
            limit = ref_ms * max_slowdown
            line += f" | ref {ref_ms:.2f} ms, limit {limit:.2f}"
            if got > limit:
                line += f"  FAIL abs > {max_slowdown}x ref"
                row_ok = False
        print(line + ("  ok" if row_ok and ref_ms is not None else ""))
        ok = ok and row_ok
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="bench_deposit.py JSON output(s)")
    ap.add_argument("--baselines", default=None,
                    help="baseline JSON (default: baselines/gpu.json or "
                         "baselines/cpu.json, by the result's backend)")
    ap.add_argument("--max-ratio", type=float, default=1.5)
    ap.add_argument("--max-slowdown", type=float, default=1.5)
    args = ap.parse_args()

    baselines = None
    if args.baselines is not None:
        bp = Path(args.baselines)
        baselines = json.loads(bp.read_text()) if bp.exists() else {}

    try:
        all_ok = all(
            check_file(p, baselines, args.max_ratio, args.max_slowdown)
            for p in args.results
        )
    except (KeyError, json.JSONDecodeError) as e:
        print(f"malformed result file: {e}", file=sys.stderr)
        sys.exit(2)

    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

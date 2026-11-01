"""Side-by-side ms per source of two or more bench_deposit.py results.

Prints, per (kind, setting, order), the min-over-repeats ms per source of
each result and the ratio to the first one, e.g. a laptop CPU run against
a GPU node run of the same code:

    python benchmarks/compare.py results/cpu.json results/a10g.json

Rows: the production-throughput rows (star / galaxy at the ``default``
and ``floor`` settings) and the gated galaxy pair (``gate/baseline`` =
floor galaxy, ``gate/noscatter``). A baseline JSON
(``baselines/{cpu,gpu}.json``) is not a result; compare result files.
"""

import argparse
import json
from pathlib import Path


def rows(path):
    data = json.loads(Path(path).read_text())
    meta = data["meta"]
    dev = meta.get("cpu") if meta.get("backend") == "cpu" else meta.get("gpu")
    out = {}
    for r in data["results"]:
        if "ms_per_src" in r:
            out[(f"{r['kind']}/{r['variant']}", r["order"])] = min(
                r["ms_per_src"])
        elif "ms_per_gal" in r:
            out[(f"gate/{r['variant']}", r["order"])] = min(r["ms_per_gal"])
    return f"{dev} ({meta.get('backend')}, jax {meta.get('jax')})", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    args = ap.parse_args()
    labels, tables = zip(*(rows(p) for p in args.results))
    for i, (p, lab) in enumerate(zip(args.results, labels)):
        print(f"[{i}] {p}: {lab}")
    keys = sorted(set().union(*tables))
    head = f"{'row':18s} {'order':>5s} " + " ".join(
        f"{'[' + str(i) + '] ms':>11s}" for i in range(len(tables)))
    head += "".join(f" {'[' + str(i) + ']/[0]':>9s}"
                    for i in range(1, len(tables)))
    print(head)
    for key in keys:
        vals = [t.get(key) for t in tables]
        line = f"{key[0]:18s} {key[1]:>5s} " + " ".join(
            f"{v:11.2f}" if v is not None else f"{'-':>11s}" for v in vals)
        for v in vals[1:]:
            line += (f" {v / vals[0]:9.3f}" if v is not None and vals[0]
                     else f" {'-':>9s}")
        print(line)


if __name__ == "__main__":
    main()

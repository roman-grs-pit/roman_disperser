#!/usr/bin/env python
"""Summarize the line-centering control experiments side by side.

Takes several residuals tables (label + parquet) and produces a comparison figure
(per-case d_disp / d_cross median ± MAD, plus d_disp histograms) and a printed
table. Used to compare: baseline, a different roll, wavelength spacing, and a
centred-Gaussian PSF.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stat(v):
    v = v[np.isfinite(v)]
    med = np.median(v)
    return med, np.median(np.abs(v - med)), len(v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", nargs=2, action="append", metavar=("LABEL", "PARQUET"),
                   required=True, help="Repeatable: a label and its residuals parquet.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    labels = [c[0] for c in args.case]
    data = {c[0]: pd.read_parquet(c[1]) for c in args.case}
    dd = {L: stat(d.d_disp.to_numpy()) for L, d in data.items()}
    dc = {L: stat(d.d_cross.to_numpy()) for L, d in data.items()}

    print("case                 d_disp med  MAD   d_cross med  MAD    n")
    for L in labels:
        print(f"  {L:18s} {dd[L][0]:+.3f}    {dd[L][1]:.3f}  "
              f"{dc[L][0]:+.3f}     {dc[L][1]:.3f}   {dd[L][2]}")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(x - 0.2, [dd[L][0] for L in labels], 0.4,
              yerr=[dd[L][1] for L in labels], capsize=3, label="d_disp")
    ax[0].bar(x + 0.2, [dc[L][0] for L in labels], 0.4,
              yerr=[dc[L][1] for L in labels], capsize=3, label="d_cross")
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=20, ha="right")
    ax[0].set_ylabel("residual median ± MAD [px]")
    ax[0].set_title("line-centering residual by control (SCA 5, lines only)")
    ax[0].legend()

    for L in labels:
        v = data[L].d_disp.to_numpy()
        v = v[np.isfinite(v)]
        ax[1].hist(v, bins=np.linspace(-0.15, 0.15, 46), histtype="step", label=L)
    ax[1].axvline(0, color="k", lw=.5)
    ax[1].set_xlabel("d_disp [px]"); ax[1].set_title("d_disp distribution")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=120); plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()

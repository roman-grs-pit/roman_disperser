#!/usr/bin/env python
"""2x2 showcase: imaging + the same grism field at three telescope rolls.

Panel layout:
    imaging (F158 sky)   |  grism PA=0   (SCA per roll)
    grism PA=10          |  grism PA=180

The same sky field (offset from the RA=10/Dec=0 boresight) lands on a different
SCA at each roll: SCA5 at PA0/10 (a subtle 10deg roll) and SCA13 at PA180 (a full
flip). The marker stars are the same physical stars in every panel; their order-1
trace boxes show how the dispersed spectra rotate/flip with roll.

Traces precomputed by scripts/compute_roll_traces.py -> figures/showcase_roll_traces.json.

ENVIRONMENT: run under the roman_l2_job pixi env (asdf + roman_datamodels), same as
make_showcase_figure.py:
    cd /data/npadman/1-Projects/roman_l2_job
    pixi run python /data/npadman/1-Projects/roman_disperser/scripts/make_roll_figure.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

import make_showcase_figure as M  # shared helpers (same scripts/ dir)

ROLL_JSON = os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_roll_traces.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_grism_rolls")
FOV_IMG = 3.0      # imaging panel [arcmin]
FOV_DISP = 3.0     # grism panel [arcmin]
IMG_SCALE = 0.11   # imaging render [arcsec/pix]
PAD = 25.0
PALETTE = ["cyan", "springgreen", "white", "deepskyblue", "magenta", "yellow"]
HALO = [pe.withStroke(linewidth=2.5, foreground="black")]


def grism_l2(pa, sca):
    return glob.glob(f"{M.GRISM_ROOT}/output_l2/ra10_dec0_pa{pa}/*pa{pa}_detSCA{sca:02d}_l2.asdf")[0]


def label(ax, x, y, k, c):
    ax.text(x, y, str(k + 1), color=c, fontsize=11, weight="bold",
            path_effects=HALO, ha="left", va="bottom")


def main():
    data = json.load(open(ROLL_JSON))
    rolls = data["rolls"]
    stars = data["stars"]
    cra, cdec = data["center"]["ra"], data["center"]["dec"]
    print(f"[rolls] {rolls} dom_sca={data['dom_sca']} stars mag {[round(s['mag'],1) for s in stars]}")

    cmap = matplotlib.colormaps["inferno"].copy()
    cmap.set_bad("black")

    # --- imaging ---
    img, tw = M.load_imaging_cutout(cra, cdec, FOV_IMG, IMG_SCALE)

    # --- grism per roll ---
    panels = []  # (pa, cut, win)
    for pa in rolls:
        sca = data["dom_sca"][str(pa)]
        gdata, gwcs = M.load_l2(grism_l2(pa, sca))
        mx = np.array([s["by_roll"][str(pa)]["undisp"][0] for s in stars])
        my = np.array([s["by_roll"][str(pa)]["undisp"][1] for s in stars])
        cut, win, _ = M.disp_crop(gdata, mx, my, FOV_DISP, PAD)
        panels.append((pa, sca, cut, win, gwcs))
        print(f"  PA={pa} SCA{sca:02d} window {win}")

    # --- figure: 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 12.5))
    ax_img = axes[0, 0]
    ax_roll = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    ax_img.imshow(img, origin="lower", cmap=cmap, norm=M.asinh_norm(img), interpolation="nearest")
    ax_img.set_title(f"Imaging F158 (L3 coadd)\n{FOV_IMG:.0f}' @ RA={cra:.3f}, Dec={cdec:.3f}")
    for k, s in enumerate(stars):
        c = PALETTE[k % len(PALETTE)]
        ix, iy = tw.all_world2pix([s["ra"]], [s["dec"]], 0)
        rad = max(8.0, 4.0 / IMG_SCALE)
        ax_img.add_patch(plt.Circle((ix[0], iy[0]), rad, fill=False, ec=c, lw=2.0, path_effects=HALO))
        label(ax_img, ix[0] + rad, iy[0] + rad, k, c)
    M.draw_compass(ax_img, tw, img.shape[1] / 2, img.shape[0] / 2, 0.10 * img.shape[1], HALO)

    for ax, (pa, sca, cut, win, gwcs) in zip(ax_roll, panels):
        ax.imshow(cut, origin="lower", cmap=cmap, norm=M.asinh_norm(cut), interpolation="nearest")
        ax.set_title(f"Grism PA={pa}° (L2, SCA{sca:02d}) -- {FOV_DISP:.0f}'")
        for k, s in enumerate(stars):
            c = PALETTE[k % len(PALETTE)]
            br = s["by_roll"][str(pa)]
            tx = np.array(br["trace"]["x"]) - 1 - win[0]
            ty = np.array(br["trace"]["y"]) - 1 - win[2]
            x0, x1 = tx.min() - 16, tx.max() + 16
            y0, y1 = ty.min() - 8, ty.max() + 8
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=1.6, path_effects=HALO))
            ux, uy = br["undisp"][0] - 1 - win[0], br["undisp"][1] - 1 - win[2]
            ax.plot(ux, uy, "x", color=c, ms=7, mew=1.8, path_effects=HALO)
            label(ax, ux + 6, uy + 6, k, c)
        ax.set_xlim(0, cut.shape[1] - 1)
        ax.set_ylim(0, cut.shape[0] - 1)
        M.draw_compass(ax, gwcs, 0.5 * (win[0] + win[1]), 0.5 * (win[2] + win[3]), 0.10 * cut.shape[1], HALO)

    fig.suptitle("Roman WFI grism — same field, three telescope rolls", fontsize=15, y=0.99)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print("wrote", f"{OUT}.{ext}")


if __name__ == "__main__":
    main()

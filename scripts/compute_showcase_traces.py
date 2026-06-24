#!/usr/bin/env python
"""Precompute order-1 spectral traces for the showcase figure's marker stars.

The showcase figure (make_showcase_figure.py) runs in the roman_l2_job env, which
has no optical model. This helper runs in the roman_disperser optical-model envs
to compute, for the N brightest stars in the field, where their order-1 traces
land on the grism / prism detectors. Output is a small JSON the figure overlays.

Run it TWICE (it needs a different optical model + pixi env each time):

  # grism trace (also does star selection + cross-match) -- roman_disperser env:
  cd /data/npadman/1-Projects/roman_disperser
  pixi run python scripts/compute_showcase_traces.py --mode grism

  # prism trace (reads the star list written above) -- roman_disperser_prism env:
  cd /data/npadman/1-Projects/roman_disperser_prism
  pixi run python /data/npadman/1-Projects/roman_disperser/scripts/compute_showcase_traces.py --mode prism

Writes figures/showcase_star_traces.json (tracked).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel

# Must match make_showcase_figure.py (FOV_IMG = star-selection box = imaging FOV)
# Grism and prism are the SAME pointing (RA=10/Dec=0/PA=0): the nudged field lands
# on SCA3 for both, from the same catalog, so the marked stars are identical objects.
RA0, DEC0, FOV_IMG = 10.183, -0.184, 3.0
GRISM_SRC = "/mnt/roman-science/grs/20260324_roman_disperser_NERSC_native/output/ra10_dec0_pa0/*sources.parquet"
PRISM_SRC = "/mnt/roman-science/grs/prism-testing-20260527/spectro/2026-05-28/output/prism-single.sim_001.001.001.001.001.001/*sources.parquet"
GRISM_SCA, PRISM_SCA = 3, 3
GRISM_YAML = "/data/npadman/1-Projects/roman_disperser/data/Roman_grism_OpticalModel_v0.8.yaml"
PRISM_YAML = "/data/npadman/1-Projects/roman_disperser_prism/data/Roman_prism_OpticalModel_v0.8.yaml"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_star_traces.json")


def trace_sca(payload, xsca, ysca, wl):
    """SCA -> FPA -> (dispersed) MPA -> SCA: the order-1 trace for one source."""
    xf, yf = omj.sca_to_fpa(payload, np.array([float(xsca)]), np.array([float(ysca)]))
    xm, ym = omj.trace_beam(payload, xf, yf, wl)
    xo, yo = omj.mpa_to_sca(payload, xm, ym)
    return np.array(xo).ravel(), np.array(yo).ravel()


def select_stars(n, fallback_yaml):
    """Brightest n PSF stars in the imaging FOV present on both grism & prism SCAs."""
    r = FOV_IMG / 60.0 / 2.0

    def box(parq, sca):
        # Rank by flux_scale (maggies, consistent in both catalogs). NOTE: the grism
        # catalog here is from old code that stored *magnitudes* in F158, so F158 is
        # NOT maggies there -- use flux_scale for brightness.
        t = pd.read_parquet(parq, columns=["sca", "order", "type", "xsca", "ysca", "ra", "dec", "flux_scale"])
        m = (
            (t.sca == sca) & (t["order"].astype(str) == "1") & (t["type"] == "PSF")
            & (t.ra > RA0 - r) & (t.ra < RA0 + r) & (t.dec > DEC0 - r) & (t.dec < DEC0 + r)
        )
        return t[m]

    gs = box(glob.glob(GRISM_SRC)[0], GRISM_SCA).sort_values("flux_scale", ascending=False)
    ps = box(glob.glob(PRISM_SRC)[0], PRISM_SCA)
    stars = []
    for _, s in gs.iterrows():
        sep = np.hypot((ps.ra - s.ra) * np.cos(np.radians(DEC0)), ps.dec - s.dec) * 3600.0
        if len(ps) == 0 or sep.min() > 1.0:
            continue
        m = ps.loc[sep.idxmin()]
        stars.append({
            "ra": float(s.ra), "dec": float(s.dec), "mag": float(-2.5 * np.log10(s.flux_scale)),
            "g_undisp": [float(s.xsca), float(s.ysca)],
            "p_undisp": [float(m.xsca), float(m.ysca)],
        })
        if len(stars) >= n:
            break
    return stars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grism", "prism"], required=True)
    ap.add_argument("--n-stars", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "grism":
        stars = select_stars(args.n_stars, GRISM_YAML)
        model = RomanOpticalModel(GRISM_YAML)
        payload = omj.make_sca_payload(model, sca=GRISM_SCA, order="1")
        wl = np.linspace(0.95, 1.95, 80)
        for s in stars:
            x, y = trace_sca(payload, *s["g_undisp"], wl)
            s["g_trace"] = {"x": x.tolist(), "y": y.tolist()}
        data = {"ra0": RA0, "dec0": DEC0, "grism_sca": GRISM_SCA, "prism_sca": PRISM_SCA, "stars": stars}
        os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(data, f, indent=1)
        print(f"grism: wrote {len(stars)} stars (mag {[round(s['mag'],1) for s in stars]}) -> {OUT}")

    else:  # prism
        with open(OUT) as f:
            data = json.load(f)
        model = RomanOpticalModel(PRISM_YAML)
        payload = omj.make_sca_payload(model, sca=PRISM_SCA, order="1")
        wl = np.linspace(0.78, 1.84, 80)
        for s in data["stars"]:
            x, y = trace_sca(payload, *s["p_undisp"], wl)
            s["p_trace"] = {"x": x.tolist(), "y": y.tolist()}
        with open(OUT, "w") as f:
            json.dump(data, f, indent=1)
        print(f"prism: added prism traces for {len(data['stars'])} stars -> {OUT}")


if __name__ == "__main__":
    main()

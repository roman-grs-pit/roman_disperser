#!/usr/bin/env python
"""Precompute grism order-1 traces for the roll figure (same field, several rolls).

The roll run (20260324_roman_disperser_NERSC_native) dispersed RA=10/Dec=0 at
PA = 0, 10, 170, 180. A sky field offset from the boresight lands on a DIFFERENT
SCA at each roll (the focal plane rotates), and is only covered at some rolls.
This picks a field covered at the chosen rolls, selects the brightest stars there,
and computes each star's order-1 trace on the SCA it lands on at each roll.

Runs in the roman_disperser env (grism optical model). Writes
figures/showcase_roll_traces.json (tracked), consumed by make_roll_figure.py.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel

# Field centred on a bright star covered at all of rolls 0/10/180 (SCA5 at PA0/10,
# SCA13 at PA180). PA0->PA10 is a subtle 10deg roll; PA180 is a full flip.
CRA, CDEC, FOV = 9.948, -0.173, 3.0
ROLLS = [0, 10, 180]
N_STARS = 5
BASE = "/mnt/roman-science/grs/20260324_roman_disperser_NERSC_native/output"
GRISM_YAML = "/data/npadman/1-Projects/roman_disperser/data/Roman_grism_OpticalModel_v0.8.yaml"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "showcase_roll_traces.json")


def load_cat(pa):
    sp = glob.glob(f"{BASE}/ra10_dec0_pa{pa}/*sources.parquet")[0]
    t = pd.read_parquet(sp, columns=["sca", "order", "type", "xsca", "ysca", "ra", "dec", "flux_scale"])
    return t[(t["order"].astype(str) == "1") & (t["type"] == "PSF")
             & (t.xsca > 250) & (t.xsca < 3838) & (t.ysca > 250) & (t.ysca < 3838)]


def match(cat, ra, dec):
    sep = np.hypot((cat.ra - ra) * np.cos(np.radians(dec)), cat.dec - dec) * 3600.0
    return (cat.loc[sep.idxmin()], float(sep.min())) if len(cat) else (None, 1e9)


def trace_sca(payload, xsca, ysca, wl):
    xf, yf = omj.sca_to_fpa(payload, np.array([float(xsca)]), np.array([float(ysca)]))
    xm, ym = omj.trace_beam(payload, xf, yf, wl)
    xo, yo = omj.mpa_to_sca(payload, xm, ym)
    return np.array(xo).ravel(), np.array(yo).ravel()


def main():
    cats = {pa: load_cat(pa) for pa in ROLLS}
    # dominant SCA per roll for the field
    r = FOV / 60.0 / 2.0
    ref0 = cats[ROLLS[0]]
    ref0 = ref0[(ref0.ra > CRA - r) & (ref0.ra < CRA + r) & (ref0.dec > CDEC - r) & (ref0.dec < CDEC + r)]
    dom = {}
    for pa in ROLLS:
        c = cats[pa]
        sub = c[(c.ra > CRA - r) & (c.ra < CRA + r) & (c.dec > CDEC - r) & (c.dec < CDEC + r)]
        dom[pa] = int(sub.sca.value_counts().idxmax())
    print("dominant SCA per roll:", dom)

    model = RomanOpticalModel(GRISM_YAML)
    payloads = {pa: omj.make_sca_payload(model, sca=dom[pa], order="1") for pa in ROLLS}
    wl = np.linspace(0.95, 1.95, 80)

    stars = []
    for _, s in ref0.sort_values("flux_scale", ascending=False).drop_duplicates("ra").iterrows():
        by_roll = {}
        ok = True
        for pa in ROLLS:
            m, sep = match(cats[pa], s.ra, s.dec)
            if m is None or sep > 0.6 or int(m.sca) != dom[pa]:  # must be on the displayed SCA
                ok = False
                break
            x, y = trace_sca(payloads[pa], m.xsca, m.ysca, wl)
            by_roll[str(pa)] = {"sca": dom[pa], "undisp": [float(m.xsca), float(m.ysca)],
                                "trace": {"x": x.tolist(), "y": y.tolist()}}
        if not ok:
            continue
        stars.append({"ra": float(s.ra), "dec": float(s.dec),
                      "mag": float(-2.5 * np.log10(s.flux_scale)), "by_roll": by_roll})
        if len(stars) >= N_STARS:
            break

    data = {"center": {"ra": CRA, "dec": CDEC}, "rolls": ROLLS, "dom_sca": dom, "stars": stars}
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(stars)} stars (mag {[round(s['mag'],1) for s in stars]}) -> {OUT}")


if __name__ == "__main__":
    main()

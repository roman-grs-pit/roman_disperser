#!/usr/bin/env python
"""Verify the wave-2 (lines+continuum) catalog against the wave-1b catalog.

Wave 2 is designed as a pure differential against wave 1b: same field, same
selection, same line SEDs, with only the flat-fnu continuum pedestal added.
This script asserts that design holds in the shipped files:

1. metadata.parquet rows are identical (positions, provenance indices, fluxes)
   -- the closure analysis joins on row order, so this must hold row-for-row.
2. (wave-2 SED) - (analytic flat-fnu continuum) == (wave-1b SED) to float32
   rounding. The builder constructs both from the same float64 intermediates
   and stores float32, so the residual should sit at the float32 quantization
   of the continuum (~1e-16 FLAM), far below the tolerance used here
   (1e-3 x min continuum ~ 2.5e-13).

Usage: pixi run python verify_catalog.py
"""

import json

import numpy as np
import pyarrow.parquet as pq
import zarr

from roman_disperser.refdata import FLAM_0AB_COEFF

W1B = "/mnt/roman-science/grs/line-tests-20260724/catalog_lines"
W2 = "/mnt/roman-science/grs/line-tests-20260724-cont/catalog_lines"

m1 = pq.read_table(f"{W1B}/metadata.parquet").to_pandas()
m2 = pq.read_table(f"{W2}/metadata.parquet").to_pandas()

assert len(m1) == len(m2) == 15112, (len(m1), len(m2))
for col in m1.columns:
    a, b = m1[col].to_numpy(), m2[col].to_numpy()
    same = (a == b).all()
    assert same, f"column {col} differs"
print(f"metadata.parquet: all {len(m1.columns)} columns identical over {len(m1)} rows")

z1 = zarr.open(f"{W1B}/seds.zarr", mode="r")
z2 = zarr.open(f"{W2}/seds.zarr", mode="r")
wl1 = z1["wavelengths"][:]
wl2 = z2["wavelengths"][:]
assert (wl1 == wl2).all(), "wavelength grids differ"

sed1 = z1["star_seds"][0]  # wave 1b: lines only
sed2 = z2["star_seds"][0]  # wave 2: continuum + lines
continuum = (FLAM_0AB_COEFF / wl1**2).astype(np.float64)

resid = sed2.astype(np.float64) - continuum - sed1.astype(np.float64)
tol = 1e-3 * continuum.min()
print(f"max |(w2 - continuum) - w1b| = {np.abs(resid).max():.3e} FLAM "
      f"(tolerance {tol:.3e})")
assert np.abs(resid).max() < tol, "SED line content differs from wave 1b"

prov = json.loads(open(f"{W2}/provenance.json").read())
assert prov["continuum_mag"] == 16.3071968 and not prov["no_continuum"]
assert prov["flux_scale_F158_maggies"] == json.loads(
    open(f"{W1B}/provenance.json").read())["flux_scale_F158_maggies"]
print("provenance: continuum_mag 16.3071968, flux_scale matches wave 1b")
print("wave-2 catalog verified against wave-1b")

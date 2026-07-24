#!/usr/bin/env python
"""Verify the wave-3 (Sersic galaxy) catalog against the wave-1b star catalog.

Wave 3 re-emits the wave-1b field as galaxies: same positions, same lines-only
SED, same flux -- morphology (n=1, b/a=0.3, hlr=0.275", random seeded PA) is
the only change. Asserted here:

1. Positions/flux/provenance columns identical to wave-1b row-for-row
   (except flux_scale: galaxies follow the pre-scaled-SED convention with
   flux_scale = 1.0 -- REQUIRED, because load_galaxy_seds scrubs any galaxy
   SED bin with |FLAM| > 1e-12 as corruption; a 0-mag template would lose
   every line core, which silently emptied the first wave-3 run).
2. Morphology columns exactly as specified; PA reproducible from the recorded
   seed (uniform [0, 180) from np.random.default_rng(pa_seed)).
3. star_seds bit-identical to wave-1b; galaxy_seds/sim_000 equals
   star_seds * 10^(-0.4*continuum_mag) in float32, and every bin clears the
   1e-12 scrub limit, so emitted line fluxes match wave-1b exactly.
4. The pipeline's own load_catalog + validate_catalog accept the catalog.

Usage: pixi run python verify_catalog.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

W1B = "/mnt/roman-science/grs/line-tests-20260724/catalog_lines"
GAL = "/mnt/roman-science/grs/line-tests-20260724-gal/catalog_lines"
MAIN = "/data/npadman/1-Projects/roman/roman_disperser/main"

m1 = pq.read_table(f"{W1B}/metadata.parquet").to_pandas()
mg = pq.read_table(f"{GAL}/metadata.parquet").to_pandas()
prov = json.loads(open(f"{GAL}/provenance.json").read())

assert len(m1) == len(mg) == 15112, (len(m1), len(mg))
for col in ["ra", "dec", "F158", "z_obs", "z_cosmo", "sed_index",
            "sim", "src_index"]:
    assert (m1[col].to_numpy() == mg[col].to_numpy()).all(), f"{col} differs"
assert (mg["flux_scale"].to_numpy() == np.float32(1.0)).all(), \
    "galaxy flux_scale must be 1.0 (pre-scaled SED convention)"
print("shared columns identical to wave-1b over 15112 rows; flux_scale = 1.0")

g = prov["galaxy"]
assert (mg["type"] == "SER").all()
assert (mg["n"].to_numpy() == np.float32(g["sersic_n"])).all()
assert (mg["half_light_radius"].to_numpy() == np.float32(g["hlr_arcsec"])).all()
assert (mg["ba"].to_numpy() == np.float32(g["ba"])).all()
pa_expect = np.random.default_rng(g["pa_seed"]).uniform(0.0, 180.0, len(mg))
assert (mg["pa"].to_numpy() == pa_expect.astype(np.float32)).all(), \
    "PA not reproducible from recorded seed"
print(f"morphology: SER n={g['sersic_n']} ba={g['ba']} hlr={g['hlr_arcsec']}\" "
      f"PA random seed {g['pa_seed']} (reproduced)")

z1 = zarr.open(f"{W1B}/seds.zarr", mode="r")
zg = zarr.open(f"{GAL}/seds.zarr", mode="r")
assert (z1["wavelengths"][:] == zg["wavelengths"][:]).all()
assert (z1["star_seds"][:] == zg["star_seds"][:]).all(), "star_seds differ"
f158 = np.float32(10.0 ** (-0.4 * 16.3071968))
gal_sed = zg["galaxy_seds/sim_000"][:]
expect = (zg["star_seds"][:].astype(np.float64) * float(f158)).astype(np.float32)
assert (gal_sed == expect).all(), "galaxy partition != pre-scaled template"
assert np.abs(gal_sed).max() < 1e-12, \
    f"galaxy SED max {np.abs(gal_sed).max():.2e} would trip the 1e-12 scrubber"
print(f"SEDs: star_seds bitwise-equal to wave-1b; galaxy partition = template "
      f"x {float(f158):.4e}, max {np.abs(gal_sed).max():.2e} < 1e-12 scrub limit")

spec = importlib.util.spec_from_file_location(
    "bgi", Path(MAIN) / "scripts" / "build_grism_image.py")
bgi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bgi)
meta, store, wl = bgi.load_catalog(GAL)
bgi.validate_catalog(meta, store, wl)
print("pipeline load_catalog + validate_catalog: OK")
print("wave-3 galaxy catalog verified")

#!/usr/bin/env python
"""Write the per-pointing line truth table for the SSC line-grid campaign.

For every (source, SCA, order) the pipeline actually dispersed (rows of the
pointing's ``grism_*_sources.parquet`` manifest) and every injected emission
line (``lines.ecsv`` in the catalog dir), predict the detector pixel where
that line's centre lands, and record full provenance:

    src_index      original row in the *source* catalog (survives the
                   line-test magnitude cut; written by build_line_test_catalog)
    catalog_index  row in the line-test catalog (= manifest key)
    ra, dec        source sky position [deg]
    sca            SCA number 1-18
    order          spectral order string ("0", "1", "2")
    xsca, ysca     undispersed source position [1-indexed FITS px]
    line_id        row in lines.ecsv
    center_A       line centre [Angstrom]; fwhm_A, amp_rel copied alongside
    x_pred, y_pred predicted dispersed line-centre position [1-indexed FITS px]
    on_detector    True if (x_pred, y_pred) falls inside [0.5, 4088.5]^2

The prediction chain is sca_to_fpa -> trace_beam -> mpa_to_sca — the same
call sequence as ``star_disperser._compute_dispersed_positions`` (what the
simulator deposited) and ``check_line_centering.predict_jax`` (the reviewed
checker). A per-run assertion checks the first manifest row against
``predict_jax``. Be clear on what that does and does not prove: both sides run
in this process with x64 enabled, so it verifies the flatten/reshape batching
in ``predict_batch`` against the scalar path — not agreement with the float32
precision the simulator and checker actually run at.

Output: ``truth_lines.parquet`` + ``truth_lines_provenance.json`` in the
pointing directory. The parquet file also carries the provenance dict in its
own file-level metadata (key ``roman_disperser_provenance``), so the table
stays self-identifying when copied out of the pointing directory — same
motivation as the CODEVER/GITSHA FITS cards: the v0.11-v0.13 placement fixes
mean visually similar truth tables can differ by pixels. Units: FITS pixel n
has its centre at n.0; the optical model takes wavelengths in microns
(lines.ecsv stores Angstroms).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

# These truth positions are NOT double-precision evaluations of the optical
# model, despite the x64 flag below — and the flag buys less than it looks
# like it does. `make_sca_payload` returns an all-float32 payload (X_ij, Y_ij,
# C_ijk, D_ijk, crpix1/2, pixel_scale, plate_scale, xy_center, and the wl
# reference/min/max), and under JAX's promotion rules an operation with two
# float32 operands stays float32 even with x64 enabled — so parts of the chain
# (e.g. xcen * plate_scale) are genuine float32 arithmetic, not float64
# arithmetic on rounded inputs.
#
# Measured over 25,695 rows of wave 1b STPSF PA 0 (2026-08-04 audit
# reconciliation): these positions are bit-identical to the pipeline at its
# default precision (0.0 px), and differ from the same model evaluated at true
# float64 by 7.1967e-4 px max.
#
# That 7.2e-4 px offset is deterministic, NOT a float32 noise floor: ~99% of it
# sits in a per-(SCA, order) constant, with only 2.7e-5 px rms of per-source
# scatter about that constant. A downstream user can therefore tabulate and
# correct it; do not describe it to them as an irreducible precision limit. At
# matched precision the shipped reference implementation and the pipeline agree
# to 1.7e-11 px, so the two are algebraically identical to round-off — the
# offset is entirely the float32 constants above.
#
# The flag is retained because it must be set before any JAX array exists if it
# is to have any effect at all, but on the current payload its measured effect
# on these outputs is nil.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.paths import data_dir
from roman_disperser.pipeline import get_code_version, get_git_sha

# Reuse the reviewed scalar prediction path for the cross-check.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_line_centering import predict_jax  # noqa: E402

DET_LO, DET_HI = 0.5, 4088.5  # FITS-pixel bounds of the 4088^2 science array


def predict_batch(payload, xsca, ysca, lam_um):
    """Vectorized sca_to_fpa -> trace_beam -> mpa_to_sca.

    xsca, ysca: [N] undispersed positions; lam_um: [M] wavelengths (microns).
    Returns (xpix, ypix) each [N, M], 1-indexed FITS pixels.

    trace_beam is elementwise over [n]-shaped inputs, so the (source, line)
    grid is flattened to length N*M (sources repeated, lines tiled) and
    reshaped back — the same per-element math as predict_jax's scalar calls.
    """
    n, m = len(xsca), len(lam_um)
    xs = jnp.asarray(np.repeat(np.asarray(xsca), m))
    ys = jnp.asarray(np.repeat(np.asarray(ysca), m))
    lam = jnp.asarray(np.tile(np.asarray(lam_um), n))
    xf, yf = omj.sca_to_fpa(payload, xs, ys)
    xm, ym = omj.trace_beam(payload, xf, yf, lam)
    xp, yp = omj.mpa_to_sca(payload, xm, ym)
    return (np.asarray(xp).reshape(n, m), np.asarray(yp).reshape(n, m))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pointing-dir", required=True,
                   help="build_grism_image.py batch-mode pointing directory.")
    p.add_argument("--catalog-dir", required=True,
                   help="Line-test catalog dir (metadata.parquet, lines.ecsv).")
    p.add_argument("--optical-model", default=None,
                   help="Optical model YAML (default: the vendored one).")
    args = p.parse_args()

    pdir = Path(args.pointing_dir)
    cdir = Path(args.catalog_dir)
    manifest_path = next(pdir.glob("grism_*_sources.parquet"))
    manifest = pd.read_parquet(manifest_path)
    meta = pd.read_parquet(cdir / "metadata.parquet",
                           columns=["src_index"])
    lines = Table.read(cdir / "lines.ecsv", format="ascii.ecsv").to_pandas()

    model_path = args.optical_model or str(
        data_dir() / "Roman_grism_OpticalModel_v0.8.yaml")
    model = RomanOpticalModel(model_path)

    lam_um = lines["center_A"].to_numpy() / 1e4
    out = []
    for (sca, order), grp in manifest.groupby(["sca", "order"], sort=True):
        payload = omj.make_sca_payload(model, sca=int(sca), order=str(order))
        xp, yp = predict_batch(payload, grp["xsca"].to_numpy(),
                               grp["ysca"].to_numpy(), lam_um)
        # Tie to the reviewed checker path: first row, all lines, exactly.
        x1, y1 = predict_jax(payload, float(grp["xsca"].iloc[0]),
                             float(grp["ysca"].iloc[0]), lam_um)
        assert np.allclose(x1, xp[0], atol=1e-9) and \
            np.allclose(y1, yp[0], atol=1e-9), \
            f"predict_batch disagrees with predict_jax on SCA {sca} order {order}"

        n, m = len(grp), len(lines)
        out.append(pd.DataFrame({
            "src_index": np.repeat(
                meta["src_index"].to_numpy()[grp["catalog_index"]], m),
            "catalog_index": np.repeat(grp["catalog_index"].to_numpy(), m),
            "ra": np.repeat(grp["ra"].to_numpy(), m),
            "dec": np.repeat(grp["dec"].to_numpy(), m),
            "sca": np.int16(sca),
            "order": str(order),
            "xsca": np.repeat(grp["xsca"].to_numpy(), m),
            "ysca": np.repeat(grp["ysca"].to_numpy(), m),
            "line_id": np.tile(lines["line_id"].to_numpy(), n),
            "center_A": np.tile(lines["center_A"].to_numpy(), n),
            "fwhm_A": np.tile(lines["fwhm_A"].to_numpy(), n),
            "amp_rel": np.tile(lines["amp_rel"].to_numpy(), n),
            "x_pred": xp.ravel(),
            "y_pred": yp.ravel(),
        }))
    truth = pd.concat(out, ignore_index=True)
    truth["on_detector"] = (
        truth["x_pred"].between(DET_LO, DET_HI)
        & truth["y_pred"].between(DET_LO, DET_HI))

    prov = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        # Same helpers (and -dirty convention) as the CODEVER/GITSHA FITS
        # cards, so truth tables and images identify their code identically.
        "codever": get_code_version(),
        "git_commit": get_git_sha(),
        "pointing_dir": str(pdir.resolve()),
        "manifest": manifest_path.name,
        "catalog_dir": str(cdir.resolve()),
        "optical_model": model_path,
        "n_rows": int(len(truth)),
        "n_on_detector": int(truth["on_detector"].sum()),
        "orders": sorted(truth["order"].unique().tolist()),
        "coordinate_convention":
            "1-indexed FITS pixels; pixel n centred at n.0; "
            "array[i,j] <-> FITS (x,y)=(j+1,i+1)",
    }
    out_path = pdir / "truth_lines.parquet"
    arrow = pa.Table.from_pandas(truth, preserve_index=False)
    arrow = arrow.replace_schema_metadata({
        **(arrow.schema.metadata or {}),
        b"roman_disperser_provenance": json.dumps(prov).encode(),
    })
    pq.write_table(arrow, out_path)

    (pdir / "truth_lines_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    print(f"wrote {out_path}  ({len(truth)} rows, "
          f"{prov['n_on_detector']} on-detector, "
          f"codever {prov['codever']}, git {prov['git_commit'][:12]})")


if __name__ == "__main__":
    main()

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
checker). A per-run assertion cross-checks the first manifest row against
``predict_jax`` itself so this file cannot silently drift from the checker.

Output: ``truth_lines.parquet`` + ``truth_lines_provenance.json`` in the
pointing directory. Units: FITS pixel n has its centre at n.0; the optical
model takes wavelengths in microns (lines.ecsv stores Angstroms).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

import jax.numpy as jnp
import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.paths import data_dir

# Reuse the reviewed scalar prediction path for the cross-check.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_line_centering import predict_jax  # noqa: E402

DET_LO, DET_HI = 0.5, 4088.5  # FITS-pixel bounds of the 4088^2 science array


def predict_batch(payload, xsca, ysca, lam_um):
    """Vectorized sca_to_fpa -> trace_beam -> mpa_to_sca.

    xsca, ysca: [N] undispersed positions; lam_um: [M] wavelengths (microns).
    Returns (xpix, ypix) each [N, M], 1-indexed FITS pixels.
    """
    xf, yf = omj.sca_to_fpa(payload, jnp.asarray(xsca), jnp.asarray(ysca))
    xm, ym = omj.trace_beam(payload, xf, yf, jnp.asarray(lam_um))
    xp, yp = omj.mpa_to_sca(payload, xm, ym)
    return np.asarray(xp), np.asarray(yp)


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

    out_path = pdir / "truth_lines.parquet"
    truth.to_parquet(out_path, index=False)

    def _sha(repo):
        try:
            return subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "NA"

    prov = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "git_commit": _sha(str(Path(__file__).resolve().parents[2])),
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
    (pdir / "truth_lines_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    print(f"wrote {out_path}  ({len(truth)} rows, "
          f"{prov['n_on_detector']} on-detector)")


if __name__ == "__main__":
    main()

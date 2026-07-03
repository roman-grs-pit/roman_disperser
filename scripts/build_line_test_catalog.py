#!/usr/bin/env python
"""Build a synthetic point-source catalog for optical-model line-centering tests.

Purpose
-------
Every star in an existing catalog field is reassigned the *same* known spectrum:
a flat (AB) continuum at a fixed magnitude plus a handful of narrow emission
lines at precisely known wavelengths. Dispersing this field and measuring where
each line lands lets us check the optical model's wavelength -> pixel mapping
(see `check_line_centering.py`). Using the real catalog *positions* (rather than
a synthetic grid) keeps the field consistent with the imaging runs, which is
what the SSC needs; the magnitude cut selects the bright subset those runs care
about.

What it writes
--------------
A standalone unified catalog directory (`metadata.parquet` + `seds.zarr`) in the
format documented in `data/catalogs/README.md`, consumable directly by
`build_grism_image.py --catalog-dir <out>`. Plus a `lines.ecsv` sidecar listing
the injected line centers/widths/amplitudes, read back by the checker.

Spectrum construction (units, conventions)
------------------------------------------
- Wavelength grid: the catalog-standard `np.linspace(9000, 21000, 6001)` Å
  (2 Å spacing), covering the grism first order (~1.0-1.93 um) with margin.
- SEDs are stored as f_lambda (FLAM, erg/s/cm^2/Å), float32, following the
  catalog convention: the stored row is normalized to **0 AB mag in F158** and
  the per-source `flux_scale = F158` (maggies) applies the real brightness.
- Continuum: a flat-f_nu (AB-flat) source. By definition a flat-f_nu source has
  AB mag 0 in *every* band, so its FLAM is exactly

      f_lambda(lambda) = f_nu(0 AB) * c / lambda^2 = 0.10887 / lambda_Å^2

  (erg/s/cm^2/Å). This is the same relation used in `magnitude_cutoff.py`. The
  stored continuum is therefore the 0-mag anchor; `flux_scale = 10^(-0.4*mag)`
  places it at the requested continuum magnitude (default 20.0).
- Emission lines: Gaussians added on top of the continuum. Each line's *peak*
  is `amp` times the local continuum level, i.e.

      line_k(lambda) = amp_k * f_cont(center_k)
                       * exp(-0.5 * ((lambda - center_k) / sigma_k)^2)

  with sigma = FWHM / (2*sqrt(2 ln 2)). So the total flux at a line center is
  ~(1 + amp) x the continuum there. `amp` runs from 5 to 10 across the lines.

Note on the magnitude label: because the emission lines add flux, the *total*
integrated F158 of the source is marginally brighter than the continuum
magnitude. We deliberately label the catalog by the **continuum** magnitude
(the meaningful, controllable anchor) and set both `F158` and `flux_scale`
to `10^(-0.4*continuum_mag)`. The line flux above continuum is intentional and
fully described by `lines.ecsv`.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BloscCodec

# --------------------------------------------------------------------------
# Defaults describing the injected spectrum. Overridable on the command line.
# --------------------------------------------------------------------------

# Catalog-standard wavelength grid (Angstroms).
WL_MIN, WL_MAX, N_WL = 9000.0, 21000.0, 6001  # 2 Å spacing

# f_lambda of a 0-AB-mag flat-f_nu source: f_nu(0 AB)=3631 Jy, f_l = f_nu c/l^2.
# c = 2.99792458e18 Å/s, f_nu(0AB)=3.631e-20 erg/s/cm^2/Hz -> 0.108866 / l_Å^2.
FLAM_0AB_COEFF = 0.108866  # erg/s/cm^2/Å, multiply by 1/lambda_Å^2

CONTINUUM_MAG = 20.0  # AB mag of the flat continuum in F158

# Five emission lines starting at 11000 Å with irregular spacing, all within the
# region where the first-order sensitivity is >~0.5x peak (10600-19100 Å).
LINE_CENTERS_A = [11000.0, 12500.0, 15500.0, 17000.0, 19000.0]
LINE_FWHM_A = 10.0  # FWHM of each line (Å); ~1 native pixel -> near-unresolved
LINE_AMPS = [5.0, 6.25, 7.5, 8.75, 10.0]  # peak / local-continuum, per line

MAG_LIMIT = 20.0  # keep stars with F158 mag <= this (20th mag and brighter)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# --------------------------------------------------------------------------
# Spectrum construction
# --------------------------------------------------------------------------

def wavelength_grid():
    """Catalog-standard wavelength grid in Angstroms (2 Å spacing)."""
    return np.linspace(WL_MIN, WL_MAX, N_WL)


def flat_ab_continuum(wl_a):
    """FLAM of a 0-AB-mag flat-f_nu continuum on grid `wl_a` (Angstroms)."""
    return FLAM_0AB_COEFF / wl_a**2


def build_line_test_sed(wl_a, centers, fwhms, amps):
    """0-mag continuum + Gaussian emission lines, in FLAM on grid `wl_a`.

    Returns (sed, continuum). The line peak is `amp` x the *local* continuum;
    lines are added on top, so total(center) ~ (1+amp) x continuum(center).
    """
    continuum = flat_ab_continuum(wl_a)
    sed = continuum.copy()
    for center, fwhm, amp in zip(centers, fwhms, amps):
        sigma = fwhm * FWHM_TO_SIGMA
        peak = amp * flat_ab_continuum(np.array(center))  # local continuum level
        sed = sed + peak * np.exp(-0.5 * ((wl_a - center) / sigma) ** 2)
    return sed.astype(np.float32), continuum.astype(np.float32)


# --------------------------------------------------------------------------
# Source selection
# --------------------------------------------------------------------------

def select_stars(meta_path, mag_limit, mag_eps=1e-4):
    """Load `metadata.parquet`, return the point sources with mag <= mag_limit.

    F158 is stored in maggies (linear AB): mag = -2.5*log10(F158). We compute the
    magnitude in float64 and cut with a small tolerance `mag_eps`, rather than
    thresholding F158 directly against `10^(-0.4*mag_limit)`. The direct-F158
    form has a float32/float64 knife-edge: the reference catalog floors a block
    of ~2900 faint stars to exactly mag 20.00 (F158 = float32(1e-8)), and whether
    `10.0**(-0.4*20)` includes them depends on a 1-ULP `pow` rounding. Cutting in
    magnitude space with a tolerance makes "20th mag and brighter" deterministic
    and reproducible across platforms.
    """
    df = pq.read_table(meta_path).to_pandas()
    is_star = df["type"] == "PSF"
    mag = -2.5 * np.log10(df["F158"].to_numpy(np.float64))
    keep = is_star & (mag <= mag_limit + mag_eps)
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Catalog writing
# --------------------------------------------------------------------------

def write_catalog(out_dir, stars, wl_a, sed_row, continuum_mag):
    """Write metadata.parquet + seds.zarr for the line-test field.

    All stars share `sed_index=0` and carry `flux_scale = F158 =
    10^(-0.4*continuum_mag)`, so every source is the identical spectrum anchored
    to the continuum magnitude.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(stars)
    f158 = np.float32(10.0 ** (-0.4 * continuum_mag))

    # --- metadata.parquet (columns/dtypes per data/catalogs/README.md) ---
    cols = {
        "ra": stars["ra"].to_numpy(np.float64),
        "dec": stars["dec"].to_numpy(np.float64),
        "type": np.array(["PSF"] * n, dtype=object),
        "n": np.zeros(n, np.float32),
        "half_light_radius": np.zeros(n, np.float32),
        "pa": np.zeros(n, np.float32),
        "ba": np.ones(n, np.float32),
        "F158": np.full(n, f158, np.float32),
        "z_obs": np.zeros(n, np.float32),
        "z_cosmo": np.zeros(n, np.float32),
        "sed_index": np.zeros(n, np.int32),
        "flux_scale": np.full(n, f158, np.float32),
        "sim": np.zeros(n, np.int16),
        # Provenance: original row index in the source catalog, if available.
        "src_index": stars.get(
            "src_index", stars.index.to_series()
        ).to_numpy(np.int32),
    }
    schema = pa.schema([
        pa.field("ra", pa.float64(), metadata={"unit": "deg"}),
        pa.field("dec", pa.float64(), metadata={"unit": "deg"}),
        pa.field("type", pa.string()),
        pa.field("n", pa.float32()),
        pa.field("half_light_radius", pa.float32(), metadata={"unit": "arcsec"}),
        pa.field("pa", pa.float32(), metadata={"unit": "deg"}),
        pa.field("ba", pa.float32()),
        pa.field("F158", pa.float32(), metadata={"unit": "maggies"}),
        pa.field("z_obs", pa.float32()),
        pa.field("z_cosmo", pa.float32()),
        pa.field("sed_index", pa.int32()),
        pa.field("flux_scale", pa.float32()),
        pa.field("sim", pa.int16()),
        pa.field("src_index", pa.int32()),
    ])
    table = pa.table(cols, schema=schema)
    pq.write_table(table, out_dir / "metadata.parquet")

    # --- seds.zarr (Zarr v3, one star template) ---
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    store = zarr.open(str(out_dir / "seds.zarr"), mode="w")
    store.create_array(
        "wavelengths", data=wl_a.astype(np.float64), compressors=compressor,
        attributes={"units": "Angstrom",
                    "description": "Common wavelength grid for all SEDs"},
    )
    star_seds = sed_row.reshape(1, -1).astype(np.float32)
    store.create_array(
        "star_seds", data=star_seds, chunks=star_seds.shape,
        compressors=compressor,
        attributes={"units": "FLAM (erg/s/cm^2/Å, normalized to 0 mag F158)",
                    "axes": ["template_index", "wavelength"]},
    )
    # Empty galaxy group for schema completeness (no extended sources here).
    store.create_group("galaxy_seds")
    store["galaxy_seds"].attrs.update({"n_partitions": 0})
    store.attrs.update({
        "format_version": "1.0",
        "description": "Line-centering optical-model test catalog (stars only)",
    })
    return f158


def write_sidecar(out_dir, centers, fwhms, amps, continuum_mag, provenance):
    """Write lines.ecsv (line table) + provenance.json next to the catalog."""
    from astropy.table import Table

    out_dir = Path(out_dir)
    tab = Table({
        "line_id": np.arange(len(centers), dtype=int),
        "center_A": np.asarray(centers, float),
        "fwhm_A": np.asarray(fwhms, float),
        "amp_rel": np.asarray(amps, float),
    })
    tab.meta["continuum_mag"] = float(continuum_mag)
    tab.meta["description"] = (
        "Injected emission lines. amp_rel = line peak / local continuum; "
        "line added on top of continuum. center in vacuum Angstroms."
    )
    tab.write(out_dir / "lines.ecsv", format="ascii.ecsv", overwrite=True)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import os

    default_src = os.path.expandvars(
        "$ROMAN_DISPERSER_DATA/catalogs/metadata.parquet")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-catalog", default=default_src,
                   help="Source metadata.parquet to draw positions from.")
    p.add_argument("--out-dir", required=True,
                   help="Output catalog directory (metadata.parquet + seds.zarr).")
    p.add_argument("--mag-limit", type=float, default=MAG_LIMIT,
                   help="Keep stars with F158 mag <= this (default 20).")
    p.add_argument("--continuum-mag", type=float, default=CONTINUUM_MAG,
                   help="AB mag of the flat continuum in F158 (default 20).")
    p.add_argument("--line-centers", type=float, nargs="+", default=LINE_CENTERS_A,
                   help="Emission-line centers in Angstroms.")
    p.add_argument("--line-fwhm", type=float, default=LINE_FWHM_A,
                   help="Emission-line FWHM in Angstroms (default 10).")
    p.add_argument("--line-amps", type=float, nargs="+", default=LINE_AMPS,
                   help="Per-line peak amplitude relative to local continuum.")
    args = p.parse_args()

    centers = list(args.line_centers)
    amps = list(args.line_amps)
    if len(amps) != len(centers):
        p.error(f"--line-amps ({len(amps)}) must match --line-centers ({len(centers)})")
    fwhms = [args.line_fwhm] * len(centers)

    wl_a = wavelength_grid()
    sed_row, continuum = build_line_test_sed(wl_a, centers, fwhms, amps)

    stars = select_stars(args.src_catalog, args.mag_limit)
    print(f"Selected {len(stars)} point sources (mag <= {args.mag_limit}) "
          f"from {args.src_catalog}")
    print(f"  RA {stars['ra'].min():.3f}..{stars['ra'].max():.3f}, "
          f"Dec {stars['dec'].min():.3f}..{stars['dec'].max():.3f}")

    f158 = write_catalog(args.out_dir, stars, wl_a, sed_row, args.continuum_mag)

    provenance = {
        "src_catalog": str(args.src_catalog),
        "mag_limit": args.mag_limit,
        "n_stars": int(len(stars)),
        "continuum_mag": args.continuum_mag,
        "flux_scale_F158_maggies": float(f158),
        "line_centers_A": centers,
        "line_fwhm_A": args.line_fwhm,
        "line_amps_rel": amps,
        "wavelength_grid": {"min": WL_MIN, "max": WL_MAX, "n": N_WL},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_sidecar(args.out_dir, centers, fwhms, amps, args.continuum_mag, provenance)

    print(f"Wrote catalog to {args.out_dir}")
    print(f"  flux_scale = F158 = {f158:.4e} maggies (continuum at mag "
          f"{args.continuum_mag})")
    print(f"  {len(centers)} lines at {centers} Å, FWHM {args.line_fwhm} Å, "
          f"amps {amps}")


if __name__ == "__main__":
    main()

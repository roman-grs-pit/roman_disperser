#!/usr/bin/env python
"""
Look up input catalog SEDs for sources nearest to given (RA, Dec) positions.

Matches against the source manifest from a pipeline run (order 1 only),
retrieves the input SED from the galacticus catalog, computes the expected
count rate using per-SCA sensitivity curves, and writes per-source ECSV
files and spectrum plots.

Usage:
    pixi run python workbench/20260326-spectral-lookup/lookup_spectra.py \
        --targets targets.txt \
        --pointing-dir ~/data/Roman/grism-sims/output/ra10_dec0_pa0 \
        --catalog-dir ~/data/Roman/galacticus_4deg2_mock-grism

Input file format (whitespace-delimited):
    # ra dec
    10.123  0.456
    10.789  0.012
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
import zarr
from astropy.io import fits
from astropy.table import Table


# ============================================================================
# CONSTANTS
# ============================================================================

MATCH_WARN_ARCSEC = 1.0  # warn if nearest match exceeds this
DLAM_ANGSTROMS = 2.0  # wavelength bin width (must match pipeline)
SENSITIVITY_MAP_FILE = "sensitivity_map.yaml"
DEFAULT_SENSITIVITY_DIR = "data/sensitivities"


# ============================================================================
# HELPERS
# ============================================================================


def haversine_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation in arcseconds (scalar or array inputs, degrees)."""
    ra1, dec1, ra2, dec2 = (np.radians(x) for x in (ra1, dec1, ra2, dec2))
    dlat = dec2 - dec1
    dlon = ra2 - ra1
    a = np.sin(dlat / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(a))) * 3600.0


def load_targets(path):
    """Load whitespace-delimited RA, Dec file (comments start with #)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2 columns (ra, dec), got {data.shape[1]}")
    return data[:, 0], data[:, 1]


def find_nearest(target_ra, target_dec, cat_ra, cat_dec):
    """Return index and separation (arcsec) of nearest catalog source."""
    sep = haversine_arcsec(target_ra, target_dec, cat_ra, cat_dec)
    idx = np.argmin(sep)
    return idx, sep[idx]


def load_sed(catalog_dir, source_type, sed_index, sim, flux_scale):
    """Load SED from the zarr store and apply flux_scale.

    Returns wavelengths (Angstroms) and flux (FLAM: erg/s/cm^2/A).
    """
    store = zarr.open(str(catalog_dir / "seds.zarr"), mode="r")
    wavelengths = np.array(store["wavelengths"])

    if source_type == "PSF":
        sed = np.array(store["star_seds"][sed_index])
    else:
        key = f"galaxy_seds/sim_{sim:03d}"
        sed = np.array(store[key][sed_index])

    return wavelengths, sed * flux_scale


def load_sensitivity(sensitivity_dir, sca, wavelengths_angstrom):
    """Load order-1 sensitivity curve for a given SCA.

    Parameters
    ----------
    sensitivity_dir : Path
        Directory with sensitivity FITS files and sensitivity_map.yaml.
    sca : int
        SCA number (1-18).
    wavelengths_angstrom : ndarray
        Wavelength grid in Angstroms.

    Returns
    -------
    sensitivity : ndarray
        Sensitivity interpolated onto the wavelength grid.
    """
    with open(sensitivity_dir / SENSITIVITY_MAP_FILE) as f:
        sens_map = yaml.safe_load(f)

    sca_key = f"SCA{sca}"
    fname = sens_map[sca_key]["1"]  # order 1
    with fits.open(sensitivity_dir / fname) as hdul:
        wl_sens = hdul[1].data["WAVELENGTH"]  # Angstroms
        sens_vals = hdul[1].data["SENSITIVITY"]
        return np.interp(wavelengths_angstrom, wl_sens, sens_vals)


def sca_order_summary(manifest_rows):
    """Build a string summarizing which SCAs/orders a source appears on."""
    lines = []
    for _, row in manifest_rows.iterrows():
        lines.append(
            f"SCA{row['sca']:02d} order={row['order']} "
            f"x={row['xsca']:.1f} y={row['ysca']:.1f}"
        )
    return "; ".join(lines)


def write_ecsv(path, wavelengths, flux, counts_per_s, meta):
    """Write spectrum as ECSV with metadata in header."""
    t = Table()
    t["wavelength"] = wavelengths
    t["wavelength"].unit = "Angstrom"
    t["wavelength"].description = "Wavelength"

    t["flux"] = flux.astype(np.float64)
    t["flux"].unit = "erg / (s cm2 Angstrom)"
    t["flux"].description = "f_lambda (FLAM), apparent"

    t["counts_per_s"] = counts_per_s.astype(np.float64)
    t["counts_per_s"].unit = "ct / s"
    t["counts_per_s"].description = (
        f"Count rate per wavelength bin "
        f"(flux * sensitivity * dlam, SCA{meta['sensitivity_sca']:02d} order 1)"
    )

    t.meta = meta
    t.write(path, format="ascii.ecsv", overwrite=True)
    print(f"  Wrote {path}")


def plot_spectrum(spec, meta, output_path):
    """Plot one source: two panels (FLAM and counts/s)."""
    import matplotlib.pyplot as plt

    wl = spec["wavelength"]
    fl = spec["flux"]
    cts = spec["counts_per_s"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Title ---
    title_parts = [
        f"RA={meta['ra']:.5f}, Dec={meta['dec']:.5f}",
        f"F158={meta['F158']:.2f} AB, type={meta['type']}",
    ]
    if meta["type"] == "SER":
        title_parts[-1] += (
            f", z={meta['z_obs']:.4f}"
            f", n={meta['sersic_n']:.2f}"
            f", r_e={meta['half_light_radius']:.3f}\""
        )
    title_parts.append(f"match={meta['match_sep_arcsec']:.3f}\", "
                       f"sensitivity: SCA{meta['sensitivity_sca']:02d} order 1")
    fig.suptitle("\n".join(title_parts), fontsize=10, ha="left", x=0.12)

    # --- FLAM panel ---
    ax1.plot(wl, fl, linewidth=0.5, color="C0")
    ax1.set_ylabel(r"$f_\lambda$ [erg/s/cm$^2$/Å]")
    ax1.set_xlim(9000, 21000)

    # --- Counts/s panel ---
    ax2.plot(wl, cts, linewidth=0.5, color="C1")
    ax2.set_xlabel("Wavelength [Å]")
    ax2.set_ylabel("counts/s per bin")
    ax2.set_xlim(9000, 21000)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Wrote {output_path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--targets", type=str, required=True,
        help="Whitespace-delimited file with RA, Dec columns",
    )
    parser.add_argument(
        "--pointing-dir", type=str, required=True,
        help="Pipeline output directory for one pointing",
    )
    parser.add_argument(
        "--catalog-dir", type=str, required=True,
        help="Galacticus catalog directory (metadata.parquet + seds.zarr)",
    )
    parser.add_argument(
        "--sensitivity-dir", type=str, default=DEFAULT_SENSITIVITY_DIR,
        help=f"Sensitivity curves directory (default: {DEFAULT_SENSITIVITY_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as script)",
    )
    args = parser.parse_args()

    pointing_dir = Path(args.pointing_dir)
    catalog_dir = Path(args.catalog_dir)
    sensitivity_dir = Path(args.sensitivity_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    # --- Load targets ---
    target_ra, target_dec = load_targets(args.targets)
    print(f"Loaded {len(target_ra)} target(s)")

    # --- Load source manifest (order 1 only) ---
    parquet_files = list(pointing_dir.glob("*_sources.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No sources parquet found in {pointing_dir}")
    manifest = pq.read_table(parquet_files[0]).to_pandas()
    order1 = manifest[manifest["order"] == "1"].copy()
    # De-duplicate: keep one row per catalog_index (pick first SCA)
    order1_unique = order1.drop_duplicates(subset="catalog_index", keep="first")
    print(
        f"Source manifest: {len(manifest)} rows total, "
        f"{len(order1)} order-1, {len(order1_unique)} unique sources"
    )

    cat_ra = order1_unique["ra"].values
    cat_dec = order1_unique["dec"].values

    # --- Load catalog metadata for ancillary columns ---
    cat_meta = pq.read_table(catalog_dir / "metadata.parquet").to_pandas()

    # --- Cache sensitivity curves per SCA ---
    sensitivity_cache = {}

    # --- Process each target ---
    for i in range(len(target_ra)):
        tra, tdec = target_ra[i], target_dec[i]
        print(f"\nTarget {i + 1}: RA={tra:.6f}, Dec={tdec:.6f}")

        idx, sep = find_nearest(tra, tdec, cat_ra, cat_dec)
        matched = order1_unique.iloc[idx]
        cat_idx = int(matched["catalog_index"])
        sca = int(matched["sca"])

        if sep > MATCH_WARN_ARCSEC:
            print(f"  WARNING: nearest match is {sep:.3f}\" away (>{MATCH_WARN_ARCSEC}\")")
        else:
            print(f"  Matched catalog_index={cat_idx}, sep={sep:.3f}\", SCA{sca:02d}")

        # Ancillary info from catalog
        cat_row = cat_meta.iloc[cat_idx]

        # All manifest appearances for this source
        all_appearances = manifest[manifest["catalog_index"] == cat_idx]
        appearances_str = sca_order_summary(all_appearances)

        # Load SED
        wavelengths, flux = load_sed(
            catalog_dir,
            source_type=cat_row["type"],
            sed_index=int(cat_row["sed_index"]),
            sim=int(cat_row["sim"]),
            flux_scale=float(cat_row["flux_scale"]),
        )
        print(f"  type={cat_row['type']}, F158={cat_row['F158']:.2f}, "
              f"flux_scale={cat_row['flux_scale']:.4e}")

        # Load sensitivity for this SCA
        if sca not in sensitivity_cache:
            sensitivity_cache[sca] = load_sensitivity(
                sensitivity_dir, sca, wavelengths
            )
        sensitivity = sensitivity_cache[sca]
        counts_per_s = flux * sensitivity * DLAM_ANGSTROMS

        # Build metadata dict
        meta = {
            "target_ra": float(tra),
            "target_dec": float(tdec),
            "ra": float(cat_row["ra"]),
            "dec": float(cat_row["dec"]),
            "catalog_index": cat_idx,
            "match_sep_arcsec": round(float(sep), 6),
            "type": cat_row["type"],
            "F158": round(float(cat_row["F158"]), 4),
            "flux_scale": float(cat_row["flux_scale"]),
            "z_obs": round(float(cat_row["z_obs"]), 6),
            "z_cosmo": round(float(cat_row["z_cosmo"]), 6),
            "sersic_n": round(float(cat_row["n"]), 4),
            "half_light_radius": round(float(cat_row["half_light_radius"]), 4),
            "pa": round(float(cat_row["pa"]), 4),
            "ba": round(float(cat_row["ba"]), 4),
            "sed_index": int(cat_row["sed_index"]),
            "sim": int(cat_row["sim"]),
            "sensitivity_sca": sca,
            "dlam_angstroms": DLAM_ANGSTROMS,
            "appearances": appearances_str,
            "pointing_dir": str(pointing_dir),
            "catalog_dir": str(catalog_dir),
        }

        # Write ECSV
        fname = f"source_{i + 1:02d}_cat{cat_idx}.ecsv"
        write_ecsv(output_dir / fname, wavelengths, flux, counts_per_s, meta)

        # Plot
        spec = {"wavelength": wavelengths, "flux": flux, "counts_per_s": counts_per_s}
        png_fname = f"source_{i + 1:02d}_cat{cat_idx}.png"
        plot_spectrum(spec, meta, output_dir / png_fname)

    print("\nDone.")


if __name__ == "__main__":
    main()

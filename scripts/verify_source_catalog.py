#!/usr/bin/env python
"""Verify a source catalog built by build_source_catalog.py.

The catalog stores F158 in maggies (linear flux); this verifier converts to
AB magnitudes for human-readable comparisons (`mag = -2.5*log10(maggies)`).

Checks:
1. Metadata consistency (sed_index ranges, flux_scale, types, src_index, etc.)
2. Round-trip F158 magnitude (integrate SED through F158 bandpass, compare to catalog mag)
3. F184 color diagnostic (integrate through F184, compare to Galacticus FITS index)
4. Star SED shape sanity

Usage
-----
pixi run python scripts/verify_source_catalog.py --catalog-dir /tmp/test_catalog
pixi run python scripts/verify_source_catalog.py --catalog-dir /tmp/test_catalog --skip-f184
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr


def bandpass_mag(sed, weights, norm, mf0):
    """Compute AB magnitude from an SED and precomputed bandpass weights."""
    mean_flam = (sed * weights).sum() / norm
    if mean_flam <= 0:
        return np.nan
    return -2.5 * np.log10(mean_flam / mf0)


def mag_from_maggies(maggies):
    """Convert maggies (linear flux, AB) to AB magnitude."""
    return -2.5 * np.log10(maggies)


def precompute_bandpass(band, wavelengths):
    """Precompute bandpass integration weights and zero-point."""
    from astropy import units as u

    tp = band(wavelengths * u.AA).value
    dlam = np.gradient(wavelengths)
    weights = tp * wavelengths * dlam
    norm = weights.sum()

    c_ang = 2.99792458e18
    fnu_0ab = 10.0 ** (-0.4 * 48.6)
    flam_0ab = fnu_0ab * c_ang / wavelengths**2
    mf0 = (flam_0ab * weights).sum() / norm

    return weights, norm, mf0


# ---------------------------------------------------------------------------
# Check 1: Metadata consistency
# ---------------------------------------------------------------------------

def check_metadata_consistency(df, store):
    """Check metadata internal consistency."""
    print("=== Metadata Consistency ===")
    issues = []

    # Types
    valid_types = {"PSF", "SER"}
    bad_types = set(df["type"].unique()) - valid_types
    if bad_types:
        issues.append(f"Invalid types: {bad_types}")
    else:
        print(f"  Types: OK ({dict(df['type'].value_counts())})")

    # Stars
    stars = df[df["type"] == "PSF"]
    if len(stars) > 0:
        if not (stars["sim"] == 0).all():
            issues.append("Some stars have sim != 0")
        # For stars, flux_scale equals F158 (maggies): the template is normalized
        # to 0 ABmag F158 (= 1 maggie), so the per-source flux scale is the F158
        # flux in maggies.
        fs_err = np.abs(stars["flux_scale"] - stars["F158"]) / stars["F158"]
        if fs_err.max() > 1e-5:
            issues.append(f"Star flux_scale mismatch: max rel error = {fs_err.max():.2e}")
        else:
            print(f"  Star flux_scale: OK (max rel error {fs_err.max():.2e})")
        if not (stars["n"] == 0).all():
            issues.append("Some stars have n != 0")
        if not (stars["half_light_radius"] == 0).all():
            issues.append("Some stars have half_light_radius != 0")

        n_star_templates = store["star_seds"].shape[0]
        if stars["sed_index"].max() >= n_star_templates:
            issues.append(f"Star sed_index {stars['sed_index'].max()} >= {n_star_templates}")
        else:
            print(f"  Star sed_index range: OK (0-{stars['sed_index'].max()} / {n_star_templates} templates)")

    # Galaxies
    galaxies = df[df["type"] == "SER"]
    if len(galaxies) > 0:
        if not (galaxies["flux_scale"] == 1.0).all():
            issues.append("Some galaxies have flux_scale != 1.0")
        else:
            print("  Galaxy flux_scale: OK (all 1.0)")
        if (galaxies["sim"] <= 0).any():
            issues.append("Some galaxies have sim <= 0")

        for sim_num, group in galaxies.groupby("sim"):
            key = f"galaxy_seds/sim_{sim_num:03d}"
            if key not in store:
                issues.append(f"Missing Zarr array: {key}")
                continue
            n_sources_attr = store[key].attrs.get("n_sources", store[key].shape[0])
            if group["sed_index"].max() >= n_sources_attr:
                issues.append(f"sim {sim_num}: sed_index {group['sed_index'].max()} >= n_sources {n_sources_attr}")
            else:
                print(f"  sim {sim_num}: sed_index range OK (0-{group['sed_index'].max()} / {n_sources_attr})")

    # src_index
    if "src_index" in df.columns:
        print(f"  src_index: present (star range 0-{stars['src_index'].max() if len(stars) else 'N/A'}, "
              f"galaxy range 0-{galaxies['src_index'].max() if len(galaxies) else 'N/A'})")
    else:
        issues.append("Missing src_index column")

    if issues:
        for issue in issues:
            print(f"  FAIL: {issue}")
    else:
        print("  All metadata checks passed.")
    return len(issues) == 0


# ---------------------------------------------------------------------------
# Check 2: F158 round-trip
# ---------------------------------------------------------------------------

def check_f158_roundtrip(df, store, n_check=500):
    """Integrate SEDs through F158 bandpass and compare to catalog magnitude."""
    from roman_disperser.refdata import get_f158_band

    print(f"\n=== F158 Round-Trip Magnitude (n={n_check}) ===")
    wavelengths = np.array(store["wavelengths"])
    w158, n158, mf0_158 = precompute_bandpass(get_f158_band(), wavelengths)

    ok = True

    # Stars
    stars = df[df["type"] == "PSF"]
    n_stars_check = min(n_check, len(stars))
    if n_stars_check > 0:
        star_sample = stars.sample(n=n_stars_check, random_state=42)
        errors = []
        for _, row in star_sample.iterrows():
            sed = np.array(store["star_seds"][row["sed_index"]]) * row["flux_scale"]
            errors.append(bandpass_mag(sed, w158, n158, mf0_158)
                          - mag_from_maggies(row["F158"]))
        err = np.array(errors)
        print(f"  Stars:    median Δmag = {np.median(err):+.4f}, "
              f"max |Δmag| = {np.abs(err).max():.4f}, std = {np.std(err):.4f}")
        if np.abs(err).max() > 0.01:
            print("  FAIL: star F158 round-trip error > 0.01 mag")
            ok = False

    # Galaxies
    galaxies = df[df["type"] == "SER"]
    n_gal_check = min(n_check, len(galaxies))
    if n_gal_check > 0:
        gal_sample = galaxies.sample(n=n_gal_check, random_state=42)
        errors = []
        for _, row in gal_sample.iterrows():
            key = f"galaxy_seds/sim_{row['sim']:03d}"
            sed = np.array(store[key][row["sed_index"]]) * row["flux_scale"]
            errors.append(bandpass_mag(sed, w158, n158, mf0_158)
                          - mag_from_maggies(row["F158"]))
        err = np.array(errors)
        print(f"  Galaxies: median Δmag = {np.median(err):+.4f}, "
              f"max |Δmag| = {np.abs(err).max():.4f}, std = {np.std(err):.4f}")
        if np.abs(err).max() > 0.01:
            print("  FAIL: galaxy F158 round-trip error > 0.01 mag")
            ok = False

    if ok:
        print("  F158 round-trip: PASSED")
    return ok


# ---------------------------------------------------------------------------
# Check 3: F184 color diagnostic
# ---------------------------------------------------------------------------

def check_f184_color(df, store, galacticus_dir, n_check=500):
    """Compare synthetic F184 magnitude to Galacticus catalog value.

    This is a DIAGNOSTIC, not a pass/fail test. Galacticus computes magnitudes
    via an independent code path that disagrees with synphot integration at
    the ~0.02 mag (median) level, with a tail to ~0.1-0.3 mag for sources
    with extreme colors or high redshift. This is a property of the Galacticus
    mock, not our extraction.
    """
    from astropy.io import fits as afits
    from roman_disperser.refdata import get_f184_band

    print(f"\n=== F184 Color Diagnostic (n={n_check}) ===")
    galacticus_dir = Path(galacticus_dir)
    fits_path = galacticus_dir / "Euclid_Roman_4deg2_radec.fits"
    if not fits_path.exists():
        print(f"  SKIPPED: {fits_path} not found")
        return True

    wavelengths = np.array(store["wavelengths"])
    w184, n184, mf0_184 = precompute_bandpass(get_f184_band(), wavelengths)

    # Load Galacticus reference magnitudes
    with afits.open(fits_path) as hdu:
        t = hdu[1].data
        fits_ra = t["RA"]
        fits_dec = t["DEC"]
        fits_sim = t["SIM"]
        fits_mag184 = t["mag_F184_Av1.6523"]

    galaxies = df[df["type"] == "SER"]
    n_gal_check = min(n_check, len(galaxies))
    gal_sample = galaxies.sample(n=n_gal_check, random_state=42)

    color_errors = []
    for _, row in gal_sample.iterrows():
        sim = int(row["sim"])

        # Match by RA/Dec within the sim partition
        sim_mask = fits_sim == sim
        cos_dec = np.cos(np.radians(row["dec"]))
        dra = (fits_ra[sim_mask] - row["ra"]) * cos_dec
        ddec = fits_dec[sim_mask] - row["dec"]
        dist = dra**2 + ddec**2
        best = np.argmin(dist)
        if dist[best] > 1e-10:
            continue
        ref_f184 = fits_mag184[sim_mask][best]

        # Compute synthetic F184 from our catalog SED
        key = f"galaxy_seds/sim_{row['sim']:03d}"
        sed = np.array(store[key][row["sed_index"]]) * row["flux_scale"]
        mag184_synth = bandpass_mag(sed, w184, n184, mf0_184)
        if np.isnan(mag184_synth):
            continue

        cat_mag158 = mag_from_maggies(row["F158"])
        color_ref = cat_mag158 - ref_f184
        color_synth = cat_mag158 - mag184_synth
        color_errors.append(color_synth - color_ref)

    if not color_errors:
        print("  SKIPPED: no galaxies matched")
        return True

    err = np.array(color_errors)
    p50 = np.median(np.abs(err))
    p90 = np.percentile(np.abs(err), 90)
    p95 = np.percentile(np.abs(err), 95)
    print(f"  Matched {len(err)} galaxies")
    print(f"  Color (F158-F184) error vs Galacticus:")
    print(f"    median |err| = {p50:.4f} mag, 90th pct = {p90:.4f}, 95th pct = {p95:.4f}")
    print(f"    median err   = {np.median(err):+.4f} (positive = we predict bluer)")

    # Diagnostic: flag if there's a large systematic offset (would indicate
    # f_ν/f_λ confusion or bandpass mismatch)
    if np.abs(np.median(err)) > 0.1:
        print("  WARNING: large systematic color offset — check SED unit conversion")
    else:
        print("  No systematic offset detected (SED shape conversion OK)")

    return True  # diagnostic only, never fails


# ---------------------------------------------------------------------------
# Check 4: Star SED sanity
# ---------------------------------------------------------------------------

def check_star_sed_sanity(df, store, n_check=10):
    """Verify star SED templates are physically reasonable."""
    print(f"\n=== Star SED Sanity (n={n_check}) ===")
    stars = df[df["type"] == "PSF"]
    unique_templates = stars["sed_index"].unique()
    n_check = min(n_check, len(unique_templates))

    issues = []
    for template_idx in unique_templates[:n_check]:
        sed = np.array(store["star_seds"][template_idx])
        frac_positive = (sed > 0).mean()
        if frac_positive < 0.95:
            issues.append(f"template {template_idx}: only {frac_positive:.1%} positive")
        if sed.max() <= 0:
            issues.append(f"template {template_idx}: no positive values")

    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print(f"  Checked {n_check} unique templates: all >95% positive")
        print("  Star SED sanity: PASSED")
    return len(issues) == 0


# ---------------------------------------------------------------------------
# Check 5: Source data provenance (RA, Dec, mag copied correctly)
# ---------------------------------------------------------------------------

def check_source_provenance(df, galacticus_dir, star_dir):
    """Verify RA, Dec, F158 match the original source files."""
    from astropy.io import fits as afits

    print("\n=== Source Provenance ===")
    ok = True

    # --- Galaxies: check against FITS index ---
    galaxies = df[df["type"] == "SER"]
    fits_path = Path(galacticus_dir) / "Euclid_Roman_4deg2_radec.fits"
    if len(galaxies) > 0 and fits_path.exists():
        with afits.open(fits_path) as hdu:
            t = hdu[1].data
            fits_ra = t["RA"]
            fits_dec = t["DEC"]
            fits_sim = t["SIM"]
            fits_idx = t["IDX"]
            fits_mag158 = t["mag_F158_Av1.6523"]

        n_checked = 0
        max_ra_err = 0.0
        max_dec_err = 0.0
        max_mag_err = 0.0

        for sim_num, group in galaxies.groupby("sim"):
            sim_mask = fits_sim == sim_num
            ref_ra = fits_ra[sim_mask]
            ref_dec = fits_dec[sim_mask]
            ref_idx = fits_idx[sim_mask]
            ref_mag = fits_mag158[sim_mask]

            for _, row in group.iterrows():
                # Match by IDX (= src_index = HDF5 row)
                match = ref_idx == row["src_index"]
                if match.sum() != 1:
                    continue
                max_ra_err = max(max_ra_err, abs(row["ra"] - ref_ra[match][0]))
                max_dec_err = max(max_dec_err, abs(row["dec"] - ref_dec[match][0]))
                max_mag_err = max(max_mag_err,
                                  abs(mag_from_maggies(row["F158"]) - ref_mag[match][0]))
                n_checked += 1

        print(f"  Galaxies: checked {n_checked} against FITS index")
        print(f"    max |ΔRA| = {max_ra_err:.2e} deg, max |ΔDec| = {max_dec_err:.2e} deg, "
              f"max |ΔF158| = {max_mag_err:.2e} mag")
        if max_ra_err > 1e-10 or max_dec_err > 1e-10:
            print("  FAIL: RA/Dec mismatch")
            ok = False
        if max_mag_err > 1e-4:
            print("  FAIL: F158 magnitude mismatch")
            ok = False
    elif len(galaxies) > 0:
        print(f"  Galaxies: SKIPPED ({fits_path} not found)")

    # --- Stars: check against text catalog ---
    stars = df[df["type"] == "PSF"]
    star_cat_path = Path(star_dir) / "sim_star_cat_galacticus.txt"
    if len(stars) > 0 and star_cat_path.exists():
        data = np.loadtxt(star_cat_path, skiprows=1)
        cat_ra = data[:, 3]
        cat_dec = data[:, 4]
        cat_mag = data[:, 2].astype(np.float32)

        max_ra_err = 0.0
        max_dec_err = 0.0
        max_mag_err = 0.0

        for _, row in stars.iterrows():
            idx = row["src_index"]
            max_ra_err = max(max_ra_err, abs(row["ra"] - cat_ra[idx]))
            max_dec_err = max(max_dec_err, abs(row["dec"] - cat_dec[idx]))
            max_mag_err = max(max_mag_err,
                              abs(mag_from_maggies(row["F158"]) - cat_mag[idx]))

        print(f"  Stars: checked {len(stars)} against text catalog")
        print(f"    max |ΔRA| = {max_ra_err:.2e} deg, max |ΔDec| = {max_dec_err:.2e} deg, "
              f"max |ΔF158| = {max_mag_err:.2e} mag")
        if max_ra_err > 1e-10 or max_dec_err > 1e-10:
            print("  FAIL: RA/Dec mismatch")
            ok = False
        if max_mag_err > 1e-4:
            print("  FAIL: F158 magnitude mismatch")
            ok = False
    elif len(stars) > 0:
        print(f"  Stars: SKIPPED ({star_cat_path} not found)")

    if ok:
        print("  Source provenance: PASSED")
    return ok


# ---------------------------------------------------------------------------
# Check 6: Synphot integration (vectorized vs per-source)
# ---------------------------------------------------------------------------

def check_synphot_integration(df, store, galacticus_dir, n_check=10):
    """Compare catalog SEDs against per-source synphot normalization.

    This is the gold-standard test: load raw f_ν SEDs from HDF5, normalize
    via synphot per-source, and compare to what the vectorized extraction
    produced. Should match to ~1 ppm.
    """
    import h5py
    import synphot as syn
    from astropy import units as u
    from roman_disperser.refdata import get_f158_band

    print(f"\n=== Synphot Integration Check (n={n_check}) ===")
    galacticus_dir = Path(galacticus_dir)
    f158_band = get_f158_band()
    wavelengths = np.array(store["wavelengths"])
    wl_qty = wavelengths * u.AA

    # Galacticus wavelength grid (from Readme_4sqdeg.txt)
    wl_galacticus = np.linspace(2000, 40000, 19001) * u.AA
    # Slice matching our output grid. DERIVED from the catalog's own first
    # wavelength rather than hardcoded: the floor moved from 9000 A (grism-era)
    # to 7500 A (prism band edge), and a hardcoded start index silently keeps
    # reading the old window against a rebuilt catalog.
    _wl0, _step = 2000.0, 2.0
    _i0 = int(round((float(wavelengths[0]) - _wl0) / _step))
    grism_slice = slice(_i0, _i0 + len(wavelengths))
    assert np.allclose(wl_galacticus[grism_slice].to_value(u.AA), wavelengths), (
        "derived Galacticus slice does not match the catalog wavelength grid"
    )

    galaxies = df[df["type"] == "SER"]
    ok = True

    for sim_num, group in galaxies.groupby("sim"):
        hdf5_path = galacticus_dir / f"galacticus_FOV_EVERY100_sub_{sim_num}.hdf5"
        if not hdf5_path.exists():
            continue

        sample = group.sample(n=min(n_check, len(group)), random_state=42)
        max_rel_err = 0.0
        max_mag_err = 0.0

        with h5py.File(hdf5_path, "r") as f:
            raw_seds = f["Outputs"]["SED:observed:dust:Av1.6523"]

            for _, row in sample.iterrows():
                # Per-source synphot normalization (gold standard)
                raw_fnu = raw_seds[row["src_index"], grism_slice]
                wl_slice = wl_galacticus[grism_slice]
                sp = syn.SourceSpectrum(
                    syn.Empirical1D, points=wl_slice,
                    lookup_table=raw_fnu * u.Jy,
                )
                cat_mag158 = mag_from_maggies(float(row["F158"]))
                norm_sp = sp.normalize(cat_mag158 * u.ABmag, band=f158_band)
                synphot_sed = norm_sp(wl_qty, flux_unit=syn.units.FLAM).value

                # Our catalog SED
                catalog_sed = np.array(
                    store[f"galaxy_seds/sim_{sim_num:03d}"][row["sed_index"]]
                )

                # Compare SED values where signal is significant
                mask = synphot_sed > synphot_sed.max() * 1e-6
                if mask.sum() == 0:
                    continue
                rel_err = np.abs(catalog_sed[mask] - synphot_sed[mask]) / synphot_sed[mask]
                max_rel_err = max(max_rel_err, rel_err.max())

                # Also verify F158 round-trip through synphot Observation
                cat_sp = syn.SourceSpectrum(
                    syn.Empirical1D, points=wl_qty,
                    lookup_table=catalog_sed * syn.units.FLAM,
                )
                obs158 = syn.Observation(cat_sp, f158_band)
                mag158_synphot = obs158.effstim(u.ABmag).value
                mag_err = abs(mag158_synphot - cat_mag158)
                max_mag_err = max(max_mag_err, mag_err)

        print(f"  sim {sim_num}: {len(sample)} galaxies, "
              f"max SED rel error = {max_rel_err:.2e}, "
              f"max F158 Δmag = {max_mag_err:.2e}")
        if max_rel_err > 1e-4:
            print(f"  FAIL: SED relative error > 1e-4")
            ok = False
        if max_mag_err > 0.01:
            print(f"  FAIL: F158 synphot round-trip error > 0.01 mag")
            ok = False

    if ok:
        print("  Synphot integration: PASSED")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify source catalog")
    parser.add_argument("--catalog-dir", default="data/catalogs",
                        help="Directory with metadata.parquet and seds.zarr")
    parser.add_argument("--galacticus-dir",
                        default=str(Path.home() / "data/Roman/galacticus_4deg2_mock"),
                        help="Galacticus data directory (for F184 check)")
    parser.add_argument("--star-dir", default="data/stars",
                        help="Star catalog directory")
    parser.add_argument("--skip-f184", action="store_true",
                        help="Skip F184 color diagnostic")
    parser.add_argument("--n-check", type=int, default=500,
                        help="Number of sources to check per test")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir)
    parquet_path = catalog_dir / "metadata.parquet"
    zarr_path = catalog_dir / "seds.zarr"

    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found")
        sys.exit(1)
    if not zarr_path.exists():
        print(f"Error: {zarr_path} not found")
        sys.exit(1)

    print(f"Verifying catalog: {catalog_dir}")
    meta = pq.read_table(parquet_path)
    df = meta.to_pandas()
    store = zarr.open(str(zarr_path), mode="r")
    print(f"  {len(df)} sources ({(df['type']=='PSF').sum()} stars, "
          f"{(df['type']=='SER').sum()} galaxies)")
    print(f"  Wavelength grid: {store['wavelengths'].shape[0]} points, "
          f"{np.array(store['wavelengths'])[0]:.0f}-{np.array(store['wavelengths'])[-1]:.0f} Å")
    print()

    all_ok = True

    # 1. Metadata consistency
    all_ok &= check_metadata_consistency(df, store)

    # 2. F158 round-trip (pass/fail)
    all_ok &= check_f158_roundtrip(df, store, n_check=args.n_check)

    # 3. F184 color diagnostic (informational only)
    if not args.skip_f184:
        check_f184_color(df, store, args.galacticus_dir, n_check=args.n_check)
    else:
        print("\n=== F184 Color Diagnostic: SKIPPED ===")

    # 4. Star SED sanity
    all_ok &= check_star_sed_sanity(df, store)

    # 5. Source provenance (RA, Dec, mag match originals)
    all_ok &= check_source_provenance(df, args.galacticus_dir, args.star_dir)

    # 6. Synphot integration (vectorized vs per-source)
    if not args.skip_f184:  # reuses galacticus_dir
        all_ok &= check_synphot_integration(
            df, store, args.galacticus_dir, n_check=min(10, args.n_check),
        )
    else:
        print("\n=== Synphot Integration Check: SKIPPED ===")

    print("\n" + "=" * 40)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build source catalog (Parquet metadata + Zarr SEDs) from Galacticus mock + star catalog.

Usage
-----
# Single sim (quick test):
pixi run python scripts/build_source_catalog.py --sims 1

# All 100 sims:
pixi run python scripts/build_source_catalog.py --sims 1-100

# Custom paths:
pixi run python scripts/build_source_catalog.py --sims 1-5 \
    --galacticus-dir ~/data/Roman/galacticus_4deg2_mock \
    --star-dir data/stars \
    --output-dir data/catalogs
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BloscCodec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wavelength range (Angstroms)
# Covers the P127 prism band (0.75–1.85 μm) with margin; WL_MAX kept at
# 21000 Å so the catalog is a superset for grism too (grism trims to
# 9000–20000 at consumption time via pipeline.LAM_MIN/LAM_MAX).
WL_MIN = 7500.0
WL_MAX = 21000.0
WL_STEP = 2.0  # Angstroms
N_WL = int((WL_MAX - WL_MIN) / WL_STEP) + 1  # 6751

# Galacticus SED wavelength grid (from Readme_4sqdeg.txt):
# "The data array is saved with a step size of 2 Angstroms, you can get the
# wavelength by np.linspace(2000, 40000, 19001) in units of Angstroms."
# Not stored in the HDF5 files — no attributes anywhere.
GALACTICUS_WL = np.linspace(2000, 40000, 19001)  # Angstroms
GRISM_SLICE = slice(2750, 9501)  # indices for 7500-21000 Å

# Magnitude cut
MAG_CUT = 26.0  # F158 AB mag

# Galaxy morphology defaults
GALAXY_SERSIC_N = 1.0
GALAXY_HALF_LIGHT_RADIUS = 0.275  # arcsec (2.5 pixels × 0.11 arcsec/pixel)
GALAXY_PA = 0.0  # degrees
GALAXY_BA = 1.0  # axis ratio

# Zarr compression
COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

# Sharding: inner chunk size (sources per chunk)
INNER_CHUNK_SOURCES = 10


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

def make_parquet_schema():
    """Define the catalog Parquet schema with column metadata."""
    return pa.schema([
        pa.field("ra", pa.float64(),
                 metadata={"unit": "deg", "description": "Right ascension (ICRS)"}),
        pa.field("dec", pa.float64(),
                 metadata={"unit": "deg", "description": "Declination (ICRS)"}),
        pa.field("type", pa.string(),
                 metadata={"description": "Source type: PSF (star) or SER (galaxy)"}),
        pa.field("n", pa.float32(),
                 metadata={"description": "Sérsic index (0 for PSF)"}),
        pa.field("half_light_radius", pa.float32(),
                 metadata={"unit": "arcsec", "description": "Half-light radius (0 for PSF)"}),
        pa.field("pa", pa.float32(),
                 metadata={"unit": "deg", "description": "Position angle E of N (0 for PSF)"}),
        pa.field("ba", pa.float32(),
                 metadata={"unit": "", "description": "Minor-to-major axis ratio (1 for PSF)"}),
        pa.field("F158", pa.float32(),
                 metadata={"unit": "maggies",
                           "description": "F158 apparent flux (maggies; mag = -2.5*log10(F158))"}),
        pa.field("z_obs", pa.float32(),
                 metadata={"description": "Observed redshift (0 for stars)"}),
        pa.field("z_cosmo", pa.float32(),
                 metadata={"description": "Cosmological redshift (0 for stars)"}),
        pa.field("sed_index", pa.int32(),
                 metadata={"description": "Row index into SED array"}),
        pa.field("flux_scale", pa.float32(),
                 metadata={"description": "SED multiplier (1.0 for galaxies)"}),
        pa.field("sim", pa.int16(),
                 metadata={"description": "Partition number (0 for stars)"}),
        pa.field("src_index", pa.int32(),
                 metadata={"description": "Row index in original source file (for provenance)"}),
    ])


# ---------------------------------------------------------------------------
# Star processing
# ---------------------------------------------------------------------------

def load_star_catalog(star_dir):
    """Load star catalog and template file list.

    Returns
    -------
    catalog : dict with ra, dec, mag, temp_idx arrays
    template_files : list of str (58 entries, some "garbage")
    """
    star_dir = Path(star_dir)
    data = np.loadtxt(star_dir / "sim_star_cat_galacticus.txt", skiprows=1)

    raw_template_index = data[:, 1].astype(int)
    temp_inds = raw_template_index % 58

    with open(star_dir / "SEDtemplates" / "input_spectral_STARS.lis") as f:
        template_files = [line.strip() for line in f.readlines()]

    return {
        "ra": data[:, 3],
        "dec": data[:, 4],
        "mag": data[:, 2].astype(np.float32),
        "temp_idx": temp_inds.astype(int),
    }, template_files


def build_star_seds(star_dir, template_files, unique_indices, wavelengths_angstrom):
    """Normalize unique star templates to 0 ABmag F158, sample on wavelength grid.

    Returns
    -------
    template_grid : ndarray [N_unique, N_wl] float32 in FLAM at 0 ABmag
    index_map : dict mapping template_index -> row in template_grid
    """
    import synphot as syn
    from astropy import units as u
    from roman_disperser.refdata import get_f158_band

    f158_band = get_f158_band()
    wl_qty = wavelengths_angstrom * u.AA

    seds = []
    index_map = {}

    for i, idx in enumerate(sorted(unique_indices)):
        filename = template_files[idx]
        data = np.loadtxt(Path(star_dir) / "SEDtemplates" / filename)
        wl_ang, flux = data[:, 0], data[:, 1]

        sp = syn.SourceSpectrum(
            syn.Empirical1D,
            points=wl_ang * u.AA,
            lookup_table=flux * syn.units.FLAM,
        )
        norm_sp = sp.normalize(0.0 * u.ABmag, band=f158_band)
        sed = norm_sp(wl_qty, flux_unit=syn.units.FLAM).value.astype(np.float32)
        seds.append(sed)
        index_map[idx] = i

    return np.array(seds, dtype=np.float32), index_map


def process_stars(star_dir, wavelengths):
    """Process star catalog into metadata rows and SED array.

    Returns
    -------
    star_table : pyarrow.Table
    star_seds : ndarray [N_templates, N_wl] float32
    """
    print("--- Stars ---")
    catalog, template_files = load_star_catalog(star_dir)
    n_stars = len(catalog["ra"])
    print(f"  Loaded {n_stars} stars")

    unique_templates = np.unique(catalog["temp_idx"])
    print(f"  {len(unique_templates)} unique templates, normalizing to 0 ABmag F158...")

    t0 = time.time()
    star_seds, index_map = build_star_seds(
        star_dir, template_files, unique_templates, wavelengths,
    )
    print(f"  Normalized {len(star_seds)} templates in {time.time() - t0:.1f}s")

    # Map each star's template index to row in star_seds array
    sed_indices = np.array([index_map[t] for t in catalog["temp_idx"]], dtype=np.int32)
    # F158 is stored in maggies (linear flux); for stars this also serves as the
    # SED multiplier since templates are normalized to 0 ABmag F158 (1 maggie).
    f158_maggies = (10.0 ** (-0.4 * catalog["mag"])).astype(np.float32)

    # Build metadata
    star_table = pa.table(
        {
            "ra": catalog["ra"],
            "dec": catalog["dec"],
            "type": ["PSF"] * n_stars,
            "n": np.zeros(n_stars, dtype=np.float32),
            "half_light_radius": np.zeros(n_stars, dtype=np.float32),
            "pa": np.zeros(n_stars, dtype=np.float32),
            "ba": np.ones(n_stars, dtype=np.float32),
            "F158": f158_maggies,
            "z_obs": np.zeros(n_stars, dtype=np.float32),
            "z_cosmo": np.zeros(n_stars, dtype=np.float32),
            "sed_index": sed_indices,
            "flux_scale": f158_maggies,
            "sim": np.zeros(n_stars, dtype=np.int16),
            "src_index": np.arange(n_stars, dtype=np.int32),
        },
        schema=make_parquet_schema(),
    )
    print(f"  Star metadata: {n_stars} rows, SED array: {star_seds.shape}")
    return star_table, star_seds


# ---------------------------------------------------------------------------
# Galaxy processing (single partition)
# ---------------------------------------------------------------------------

def load_galacticus_index(fits_path):
    """Load FITS index file with RA, Dec, magnitudes, SIM, IDX.

    Returns
    -------
    pandas.DataFrame with columns: RA, DEC, SIM, IDX, mag_F158, z_obs, z_cosmo
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdu:
        t = hdu[1].data
        return {
            "ra": t["RA"].astype(np.float64),
            "dec": t["DEC"].astype(np.float64),
            "sim": t["SIM"].astype(np.int16),
            "idx": t["IDX"].astype(np.int32),
            "mag_F158": t["mag_F158_Av1.6523"].astype(np.float32),
            "z": t["Z"].astype(np.float32),
        }


def compute_f158_normalization(wavelengths_angstrom):
    """Precompute F158 bandpass integration weights for vectorized normalization.

    The normalization converts raw f_ν SEDs to FLAM at the catalog F158 magnitude.
    Steps:
    1. f_ν → f_λ: multiply by c/λ² (conversion factor)
    2. Synthetic F158 flux: integrate f_λ through bandpass
    3. Scale to catalog mag: multiply by 10^(-0.4 * mag) / synthetic_flux

    We precompute the combined weight array so normalization is a single dot product
    per source.

    Returns
    -------
    fnu_to_flam : ndarray [N_wl] - conversion factor from f_ν to f_λ at each wavelength
    bandpass_weights : ndarray [N_wl] - F158 throughput × λ × dλ for synthetic flux
    """
    from roman_disperser.refdata import get_f158_band

    f158_band = get_f158_band()

    # f_ν → f_λ conversion: f_λ = f_ν × c / λ²
    # With f_ν in arbitrary units, f_λ is in (same units × c / Å²)
    # The constant c cancels in the normalization ratio, so we just need 1/λ²
    # Actually we need the full factor for absolute FLAM output.
    #
    # f_λ [erg/s/cm²/Å] = f_ν [erg/s/cm²/Hz] × c [Å/s] / λ² [Å²]
    # where c = 2.99792458e18 Å/s
    c_angstrom = 2.99792458e18  # speed of light in Å/s
    fnu_to_flam = c_angstrom / wavelengths_angstrom**2

    # F158 bandpass throughput interpolated onto our wavelength grid
    from astropy import units as u
    throughput = f158_band(wavelengths_angstrom * u.AA).value  # dimensionless

    # Synthetic flux normalization:
    # For synphot, the mean flux through a bandpass is:
    #   <f_λ> = ∫ f_λ T λ dλ / ∫ T λ dλ
    # where T is throughput. AB magnitude:
    #   mag = -2.5 log10(<f_λ>) - 21.10 - 5 log10(λ_pivot/Å) + ... [complex]
    #
    # Instead, we compute the ratio: for each source, its synthetic F158 flux
    # (in our arbitrary-but-consistent units) divided by the target F158 flux
    # (from the catalog magnitude). This ratio is the same whether computed
    # in f_λ or f_ν space.
    #
    # Weight for bandpass integration: T(λ) × λ × dλ
    dlam = np.gradient(wavelengths_angstrom)
    bandpass_weights = throughput * wavelengths_angstrom * dlam

    return fnu_to_flam, bandpass_weights


def process_galaxy_partition(sim_num, galacticus_dir, fits_index,
                             fnu_to_flam, bandpass_weights, bandpass_norm):
    """Process one galaxy partition (one HDF5 sub-file).

    Uses vectorized f_ν → FLAM conversion instead of per-source synphot calls.

    Returns
    -------
    galaxy_table : pyarrow.Table (metadata rows for this partition)
    galaxy_seds : ndarray [N_kept, N_wl] float32 in FLAM
    """
    import h5py

    galacticus_dir = Path(galacticus_dir)
    hdf5_path = galacticus_dir / f"galacticus_FOV_EVERY100_sub_{sim_num}.hdf5"

    # Select sources for this sim from FITS index
    mask_sim = fits_index["sim"] == sim_num
    mask_mag = fits_index["mag_F158"] <= MAG_CUT
    mask = mask_sim & mask_mag
    n_total_sim = int(mask_sim.sum())
    n_kept = int(mask.sum())
    print(f"  sim {sim_num:3d}: {n_kept}/{n_total_sim} sources (F158 ≤ {MAG_CUT})")

    if n_kept == 0:
        return None, None

    # Get indices and metadata for kept sources
    ra = fits_index["ra"][mask]
    dec = fits_index["dec"][mask]
    mag_f158 = fits_index["mag_F158"][mask]
    z_obs = fits_index["z"][mask]
    hdf5_indices = fits_index["idx"][mask]

    # Read HDF5 SEDs for kept sources (only grism wavelength range)
    # SEDs are f_ν in unknown absolute units, on the Galacticus wavelength grid
    # (2 Å spacing). Our output grid is a subset of their grid.
    with h5py.File(hdf5_path, "r") as f:
        outputs = f["Outputs"]
        raw_seds_fnu = outputs["SED:observed:dust:Av1.6523"][hdf5_indices, GRISM_SLICE]
        z_cosmo = outputs["lightconeRedshift"][hdf5_indices].astype(np.float32)

    # Vectorized f_ν → FLAM conversion with F158 normalization:
    # 1. Convert f_ν to f_λ (arbitrary units): f_λ_raw = f_ν × c/λ²
    # 2. Compute synthetic F158 flux: <f_λ> = Σ(f_λ_raw × T × λ × dλ) / Σ(T × λ × dλ)
    # 3. Compute target flux from catalog mag: f_target = 10^(-0.4*(mag+48.6)) × c/λ_pivot²
    #    But since we need the ratio, we use:
    #    scale = 10^(-0.4*mag) × ABMAG_ZEROPOINT_FLAM / <f_λ_raw>
    #
    # Simpler: normalize so that the synthetic F158 mag equals the catalog mag.
    # The AB magnitude of the raw f_λ is:
    #    mag_raw = -2.5 log10(<f_λ_raw> / f_AB_ref)
    # We want the output to have mag = mag_catalog, so:
    #    scale = 10^(-0.4 * (mag_catalog - mag_raw))

    # Step 1: f_ν → f_λ (all sources at once)
    seds_flam_raw = raw_seds_fnu * fnu_to_flam[np.newaxis, :]  # [N, N_wl]

    # Step 2: synthetic F158 flux for each source
    # <f_λ> = Σ(f_λ × T × λ × dλ) / Σ(T × λ × dλ)
    synth_flux = seds_flam_raw @ bandpass_weights / bandpass_norm  # [N]

    # Step 3: target flux from catalog magnitude
    # AB mag definition: mag = -2.5 log10(f_ν) - 48.6 (f_ν in erg/s/cm²/Hz)
    # <f_λ>_target for the bandpass mean flux at mag_catalog
    # f_ν = 10^(-0.4*(mag+48.6)) erg/s/cm²/Hz
    # <f_λ> = f_ν × c / λ_pivot² ... but we need the bandpass-averaged version
    #
    # Actually simpler: the scale factor is just target/synthetic
    # target_flux = <f_λ> for a source at mag_catalog
    # We know f_ν [erg/s/cm²/Hz] = 10^(-0.4*(m+48.6))
    # and <f_λ> = ∫ f_λ T λ dλ / ∫ T λ dλ
    # For AB: f_ν = const, f_λ = f_ν × c/λ², so
    # <f_λ>_AB = f_ν × ∫ (c/λ²) T λ dλ / ∫ T λ dλ = f_ν × c × ∫ T/λ dλ / ∫ T λ dλ
    #
    # Let's just compute it directly for mag=0 reference:
    # f_ν(0 AB) = 10^(-0.4*48.6) = 3.6308e-20 erg/s/cm²/Hz
    fnu_0ab = 10.0**(-0.4 * 48.6)  # erg/s/cm²/Hz
    flam_0ab = fnu_0ab * fnu_to_flam  # f_λ at 0 AB, per wavelength
    synth_flux_0ab = flam_0ab @ bandpass_weights / bandpass_norm  # scalar

    # Target mean flux for each source
    target_flux = synth_flux_0ab * 10.0**(-0.4 * mag_f158)  # [N]

    # Scale each SED
    scale = (target_flux / synth_flux).astype(np.float32)  # [N]
    galaxy_seds = (seds_flam_raw * scale[:, np.newaxis]).astype(np.float32)

    # Build metadata table
    n = n_kept
    f158_maggies = (10.0 ** (-0.4 * mag_f158)).astype(np.float32)
    galaxy_table = pa.table(
        {
            "ra": ra,
            "dec": dec,
            "type": ["SER"] * n,
            "n": np.full(n, GALAXY_SERSIC_N, dtype=np.float32),
            "half_light_radius": np.full(n, GALAXY_HALF_LIGHT_RADIUS, dtype=np.float32),
            "pa": np.full(n, GALAXY_PA, dtype=np.float32),
            "ba": np.full(n, GALAXY_BA, dtype=np.float32),
            "F158": f158_maggies,
            "z_obs": z_obs,
            "z_cosmo": z_cosmo,
            "sed_index": np.arange(n, dtype=np.int32),
            "flux_scale": np.ones(n, dtype=np.float32),
            "sim": np.full(n, sim_num, dtype=np.int16),
            "src_index": hdf5_indices,
        },
        schema=make_parquet_schema(),
    )
    return galaxy_table, galaxy_seds


# ---------------------------------------------------------------------------
# Zarr writing
# ---------------------------------------------------------------------------

def init_zarr_store(output_dir, wavelengths, star_seds):
    """Initialize seds.zarr with wavelength grid and star templates.

    Parameters
    ----------
    output_dir : Path
    wavelengths : ndarray [N_wl]
    star_seds : ndarray [N_templates, N_wl]

    Returns
    -------
    store : zarr.Group (open for writing)
    """
    zarr_path = output_dir / "seds.zarr"
    print(f"\nInitializing Zarr store at {zarr_path}")

    store = zarr.open(str(zarr_path), mode="w")

    # Wavelength grid
    store.create_array(
        "wavelengths", data=wavelengths,
        compressors=COMPRESSOR,
        attributes={
            "units": "Angstrom",
            "description": "Common wavelength grid for all SEDs",
            "grid_definition": f"np.linspace({WL_MIN}, {WL_MAX}, {N_WL})",
        },
    )
    print(f"  wavelengths: {wavelengths.shape}")

    # Star templates
    store.create_array(
        "star_seds", data=star_seds,
        chunks=star_seds.shape,  # single chunk (small)
        compressors=COMPRESSOR,
        attributes={
            "units": "FLAM (erg/s/cm^2/Å, normalized to 0 mag F158)",
            "axes": ["template_index", "wavelength"],
        },
    )
    print(f"  star_seds: {star_seds.shape}")

    return store


def write_galaxy_partition(store, sim_num, galaxy_seds):
    """Write a single galaxy SED partition to an open Zarr store.

    Parameters
    ----------
    store : zarr.Group
    sim_num : int
    galaxy_seds : ndarray [N_sources, N_wl]
    """
    n_src, n_wl = galaxy_seds.shape

    # Shard size must be multiple of inner chunk
    shard_rows = ((n_src + INNER_CHUNK_SOURCES - 1) // INNER_CHUNK_SOURCES) * INNER_CHUNK_SOURCES

    # Zero-pad if needed
    if shard_rows > n_src:
        pad = np.zeros((shard_rows - n_src, n_wl), dtype=np.float32)
        seds_padded = np.concatenate([galaxy_seds, pad], axis=0)
    else:
        seds_padded = galaxy_seds

    store.create_array(
        f"galaxy_seds/sim_{sim_num:03d}",
        data=seds_padded,
        chunks=(INNER_CHUNK_SOURCES, n_wl),
        shards=(shard_rows, n_wl),
        compressors=COMPRESSOR,
        attributes={
            "units": "FLAM (erg/s/cm^2/Å, apparent)",
            "axes": ["sed_index", "wavelength"],
            "frame": "observed",
            "n_sources": n_src,  # actual count (before padding)
        },
    )
    size_mb = seds_padded.nbytes / 1e6
    print(f"  galaxy_seds/sim_{sim_num:03d}: {galaxy_seds.shape} ({size_mb:.0f} MB uncompressed)")


def finalize_zarr_store(store, n_partitions):
    """Write group-level metadata to finalize the Zarr store.

    Parameters
    ----------
    store : zarr.Group
    n_partitions : int
    """
    store["galaxy_seds"].attrs.update({"n_partitions": n_partitions})
    store.attrs.update({
        "format_version": "1.0",
        "description": "Roman grism source catalog SEDs",
        "provenance": "Galacticus 4 deg² mock + Pickles stellar atlas",
    })
    print(f"  {n_partitions} galaxy partitions written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_sims(sims_str):
    """Parse sim specification like '1', '1-5', '1,3,5', '1-100'."""
    result = []
    for part in sims_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(
        description="Build source catalog from Galacticus mock + star catalog",
    )
    parser.add_argument(
        "--sims", default="1",
        help="Sim numbers to process: '1', '1-5', '1,3,5', '1-100' (default: 1)",
    )
    parser.add_argument(
        "--galacticus-dir",
        default=str(Path.home() / "data/Roman/galacticus_4deg2_mock"),
        help="Path to Galacticus HDF5 + FITS index",
    )
    parser.add_argument(
        "--star-dir", default="data/stars",
        help="Path to star catalog directory",
    )
    parser.add_argument(
        "--output-dir", default="data/catalogs",
        help="Output directory for metadata.parquet and seds.zarr",
    )
    parser.add_argument(
        "--no-stars", action="store_true",
        help="Skip star processing (galaxies only)",
    )
    args = parser.parse_args()

    sim_numbers = parse_sims(args.sims)
    output_dir = Path(args.output_dir)
    galacticus_dir = Path(args.galacticus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building source catalog")
    print(f"  Sims: {sim_numbers}")
    print(f"  Output: {output_dir}")
    print()

    # Wavelength grid
    wavelengths = np.linspace(WL_MIN, WL_MAX, N_WL)

    # --- Stars ---
    metadata_tables = []
    if not args.no_stars:
        star_table, star_seds = process_stars(args.star_dir, wavelengths)
        metadata_tables.append(star_table)
    else:
        print("--- Skipping stars ---")
        star_seds = np.empty((0, N_WL), dtype=np.float32)

    # --- Galaxies ---
    print("\n--- Galaxies ---")
    fits_path = galacticus_dir / "Euclid_Roman_4deg2_radec.fits"
    print(f"  Loading FITS index: {fits_path}")
    t0 = time.time()
    fits_index = load_galacticus_index(fits_path)
    print(f"  Loaded {len(fits_index['ra'])} total sources in {time.time() - t0:.1f}s")

    # Precompute bandpass integration weights (once)
    fnu_to_flam, bandpass_weights = compute_f158_normalization(wavelengths)
    bandpass_norm = bandpass_weights.sum()

    # Initialize Zarr store (wavelengths + stars)
    zarr_store = init_zarr_store(output_dir, wavelengths, star_seds)

    # Process and write galaxy partitions incrementally
    n_partitions = 0
    for sim_num in sim_numbers:
        hdf5_path = galacticus_dir / f"galacticus_FOV_EVERY100_sub_{sim_num}.hdf5"
        if not hdf5_path.exists():
            print(f"  sim {sim_num:3d}: SKIPPED (file not found: {hdf5_path.name})")
            continue

        t0 = time.time()
        galaxy_table, galaxy_seds = process_galaxy_partition(
            sim_num, galacticus_dir, fits_index,
            fnu_to_flam, bandpass_weights, bandpass_norm,
        )
        dt = time.time() - t0

        if galaxy_table is not None:
            metadata_tables.append(galaxy_table)
            write_galaxy_partition(zarr_store, sim_num, galaxy_seds)
            n_partitions += 1
            print(f"           Processed in {dt:.1f}s")

    # --- Write outputs ---
    if not metadata_tables:
        print("\nNo data to write!")
        sys.exit(1)

    # Finalize Zarr store
    finalize_zarr_store(zarr_store, n_partitions)

    # Combine and write metadata
    combined = pa.concat_tables(metadata_tables)
    parquet_path = output_dir / "metadata.parquet"
    pq.write_table(combined, parquet_path)
    print(f"\nMetadata: {parquet_path} ({combined.num_rows} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()

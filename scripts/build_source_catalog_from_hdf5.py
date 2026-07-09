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

# TODO !!!!!!
#
# AGN in the hdf5? What did Ferzem do? Is the AGN flux include in the overall magnitude?
# Do we include the AGN in the disk spectrum? Or, do we treat it as it's own point-source?
#
# Cut components on even fainter mag_cut?
#
# Put half-light radii as an angular size (see romanisim docs; scale by angular diameter distance)
#
# TODO !!!!!!

import argparse
import sys, os, gc
import time
from glob import glob
from pathlib import Path
from functools import cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import synphot as syn
from roman_disperser.paths import synphot_dir
from roman_disperser.refdata import get_f158_band
import psutil
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BloscCodec

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

try:
    from galacticus_sed_calculator import SEDCalculator, read_dust_model_from_catalog
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.getenv("github_dir"), "galacticus_sed_calculator"))
    from galacticus_sed_calculator import SEDCalculator, read_dust_model_from_catalog

import multiprocessing                                                                                                                
multiprocessing.set_start_method('spawn', force=True) 

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wavelength range (Angstroms)
# Covers grism range (0.9–2.0 μm) plus margin to fully contain the F184 bandpass
WL_MIN = 7500.0
WL_MAX = 21000.0
WL_STEP = 2.0  # Angstroms
N_WL = int((WL_MAX - WL_MIN) / WL_STEP) + 1  # 6001

# Galacticus SED wavelength grid (from Readme_4sqdeg.txt):
# "The data array is saved with a step size of 2 Angstroms, you can get the
# wavelength by np.linspace(2000, 40000, 19001) in units of Angstroms."
# Not stored in the HDF5 files — no attributes anywhere.
WAVELENGTHS = np.linspace(WL_MIN, WL_MAX, N_WL)  # Angstroms

# Zarr compression
COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

# Sharding: inner chunk size (sources per chunk)
INNER_CHUNK_SOURCES = 10

MAG_CUT = 27

SED_TEMPLATE = Path(os.getenv("github_dir")) / Path("galacticus_sed_calculator/data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5")

MP_CHUNK_SIZE = 1000

UNIT_COSMO = FlatLambdaCDM(
        H0=67.74,
        Om0=0.3089
    )

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
        pa.field("disk_spheroid", pa.string(),
                 metadata={"description": "Disk/Spheroid component of galaxy (0 for PSF)"}),
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
                           "description": "F158/H apparent flux (maggies; mag = -2.5*log10(F158))"}),
        pa.field("F106", pa.float32(),
                 metadata={"unit": "maggies",
                           "description": "F106/Y apparent flux (maggies; mag = -2.5*log10(F106))"}),
        pa.field("F129", pa.float32(),
                 metadata={"unit": "maggies",
                           "description": "F129/J apparent flux (maggies; mag = -2.5*log10(F129))"}),
        pa.field("z_obs", pa.float32(),
                 metadata={"description": "Observed redshift (0 for stars)"}),
        pa.field("z_cosmo", pa.float32(),
                 metadata={"description": "Cosmological redshift (0 for stars)"}),
        pa.field("sed_index", pa.int32(),
                 metadata={"description": "Row index into SED array"}),
        pa.field("flux_scale", pa.float32(),
                 metadata={"description": "SED multiplier (1.0 for galaxies)"}),
        pa.field("sim", pa.string(),
                 metadata={"description": "Partition number (0 for stars)"}),
        pa.field("src_index", pa.int32(),
                 metadata={"description": "Row index in original source file (for provenance)"}),
        pa.field("randoms", pa.list_(pa.float32()),
                 metadata={"description": "Randoms between 0 and 1 (for reproducability of random characteristics)"}),
    ])


# ---------------------------------------------------------------------------
# Star processing
# ---------------------------------------------------------------------------

def calculate_magnitude(wavelength, fluxes, bandpass: str, return_maggies: bool = True):

    path = synphot_dir() / f"roman_wfi_{bandpass}.fits"
    bp = syn.SpectralElement.from_file(str(path))

    all_mag = []
    for flux in fluxes:
        spec = syn.SourceSpectrum(syn.models.Empirical1D, points=wavelength * u.AA, lookup_table=flux * syn.units.FLAM)
        obs = syn.Observation(spec, bp, force='taper')
        try:
            mag = obs.effstim(flux_unit=u.ABmag).value
            if return_maggies:
                all_mag.append((10.0 ** (-0.4 * mag)))
            else:
                all_mag.append(mag)
        except syn.exceptions.SynphotError:
            all_mag.append(0 if return_maggies else np.inf)
        
    return np.asarray(all_mag, dtype=np.float32)

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

    f_maggies = {}
    f_maggies["f158"] = (10.0 ** (-0.4 * np.asarray(catalog["mag"]))).astype(np.float32)

    for bp in ("f106", "f129"):
        f_maggies[bp] = calculate_magnitude(wavelengths, star_seds, bp)

    rng = np.random.default_rng(42)

    # Build metadata
    star_table = pa.table(
        {
            "ra": catalog["ra"],
            "dec": catalog["dec"],
            "type": ["PSF"] * n_stars,
            "n": np.zeros(n_stars, dtype=np.float32),
            "disk_spheroid": ["PSF"] * n_stars,
            "half_light_radius": np.zeros(n_stars, dtype=np.float32),
            "pa": np.zeros(n_stars, dtype=np.float32),
            "ba": np.ones(n_stars, dtype=np.float32),
            "F158": f_maggies["f158"],
            "F106": f_maggies["f106"][sed_indices],
            "F129": f_maggies["f129"][sed_indices],
            "z_obs": np.zeros(n_stars, dtype=np.float32),
            "z_cosmo": np.zeros(n_stars, dtype=np.float32),
            "sed_index": sed_indices,
            "flux_scale": f_maggies["f158"],
            "sim": ["PSF"] * n_stars,
            "src_index": np.arange(n_stars, dtype=np.int32),
            "randoms": list(rng.uniform(0, 1, (n_stars, 5))),
        },
        schema=make_parquet_schema(),
    )
    print(f"  Star metadata: {n_stars} rows, SED array: {star_seds.shape}")
    return star_table, star_seds


# ---------------------------------------------------------------------------
# Galaxy processing (single partition)
# ---------------------------------------------------------------------------


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

def compute_raw_seds_flam(hdf5_path, hdf5_indices, calc, sed_component='disk'):

    sed_list = []

    dust_model_specs = read_dust_model_from_catalog(hdf5_path)

    if sed_component=='total':
        for idx in hdf5_indices:
            spec = calc.evaluate_total_spectrum(
                hdf5_path,
                galIndex=idx,
                obs_wavelengths=WAVELENGTHS * u.AA, 
                include_emission_lines=True,
                use_synphot=False,  # Enable fast path
                **dust_model_specs
            )

            sed_list.append(spec(WAVELENGTHS, flux_unit='FLAM'))

    else:
        for idx in hdf5_indices:
            spec = calc.evaluate_component_spectrum(
                hdf5_path,
                galIndex=idx,
                component=sed_component,
                obs_wavelengths=WAVELENGTHS * u.AA, 
                include_emission_lines=True,
                use_synphot=False,  # Enable fast path
                **dust_model_specs
            )

            sed_list.append(spec(WAVELENGTHS, flux_unit='FLAM'))

    sed_list = np.asarray(sed_list, dtype=np.float32)

    return sed_list


def process_galaxy_partition(mock, zarr_path, sed_component='disk'):
    """Process one galaxy partition (one HDF5 sub-file).

    Uses vectorized f_ν → FLAM conversion instead of per-source synphot calls.

    Returns
    -------
    galaxy_table : pyarrow.Table (metadata rows for this partition)
    galaxy_seds : ndarray [N_kept, N_wl] float32 in FLAM
    """

    output = "/Lightcone/Output1/nodeData"
    dustNode = "/Lightcone/Output1/dustAttenuatedNodeData/"
    mock_fn = os.path.basename(mock)

    calc = SEDCalculator(SED_TEMPLATE, cosmology=UNIT_COSMO)
    zarr_store = zarr.open(str(zarr_path), mode='a')

    with h5py.File(mock) as f:
        
        total_mag = f[dustNode]["apparentMagnitudeRomanWFI:F158"][:]
        mask = total_mag <= MAG_CUT

        if sed_component=='spheroid':
            mask &= f[output]["spheroidRadius"][:] > 0
            
        elif sed_component=='disk':
            mask &= f[output]["diskRadius"][:] > 0

        # Read galaxy properties from hdf5 file
        ra = f[output]["rightAscension"][:][mask]
        dec = f[output]["declination"][:][mask]
        z_obs = f[output]["lightconeRedshiftObserved"][:][mask]
        z_cosmo = f[output]["lightconeRedshiftCosmological"][:][mask]
        randoms = f[output]["randomUniform"][:][mask]

        # assign indices
        n_src = len(ra)
        hdf5_indices = np.where(mask)[0]

        # Set positiona_angle and ba_ratio 
        # DO NOT use pa for position angle variable as it collides with pyarrow.parquet import
        # romanisim pa is in degrees
        position_angle = randoms[:, 0] * 360 

        if sed_component=='spheroid':
            sersic_idx = 4
            ba_ratio = [1] * n_src # b/a=1 for bulge
            half_light_radii = UNIT_COSMO.arcsec_per_kpc_proper(f[output]["spheroidRadius"][:][mask] / 10**3)
            
        elif sed_component=='disk':
            sersic_idx = 1
            ba_ratio = randoms[:, 2]
            sel = ba_ratio < 0.1
            ba_ratio[sel] = 0.1 # enfore disk height = 10% disk radius
            half_light_radii = UNIT_COSMO.arcsec_per_kpc_proper(f[output]["diskRadius"][:][mask] / 10**3)

        za = zarr_store.create_array(
            f"galaxy_seds/sim_{mock_fn}/{sed_component}",
            shape=(n_src, N_WL),
            chunks=(INNER_CHUNK_SOURCES, N_WL),
            dtype=np.float32,
            compressors=COMPRESSOR,
            attributes={
            "units": "FLAM (erg/s/cm^2/Å, apparent)",
            "axes": ["sed_index", "wavelength"],
            "frame": "observed",
            "n_sources": n_src,
        },
        )

        # 2. Process in blocks
        f_maggies = {"f158": [], "f106": [], "f129": []}
        for start_idx in range(0, n_src, MP_CHUNK_SIZE):
            end_idx = min(start_idx + MP_CHUNK_SIZE, n_src)
            chunk_indices = hdf5_indices[start_idx:end_idx]
            
            # This only processes 2,000 galaxies at a time
            chunk_seds = compute_raw_seds_flam(mock, chunk_indices, calc, sed_component=sed_component)
            
            # Write this block directly to its slice in the Zarr array on disk
            za[start_idx:end_idx, :] = chunk_seds
            
            for bp in ("f158", "f106", "f129"):
                f_maggies[bp].append(calculate_magnitude(WAVELENGTHS, chunk_seds, bp))

            # 3. Aggressively clear memory before the next loop iteration
            del chunk_seds
            gc.collect()  # Forces Python to dump the intermediate object bloat

        F158_arr = np.concatenate(f_maggies["f158"]) if n_src > 0 else np.empty(0, dtype=np.float32)
        F106_arr = np.concatenate(f_maggies["f106"]) if n_src > 0 else np.empty(0, dtype=np.float32)
        F129_arr = np.concatenate(f_maggies["f129"]) if n_src > 0 else np.empty(0, dtype=np.float32)

        # Build metadata table
        galaxy_table = pa.table(
            {
                "ra": ra,
                "dec": dec,
                "type": ["SER"] * n_src,
                "n": [sersic_idx] * n_src,
                "disk_spheroid": [sed_component] * n_src,
                "half_light_radius": half_light_radii,
                "pa": position_angle,
                "ba": ba_ratio,
                "F158": F158_arr,
                "F106": F106_arr,
                "F129": F129_arr,
                "z_obs": z_obs,
                "z_cosmo": z_cosmo,
                "sed_index": np.arange(n_src, dtype=np.int32),
                "flux_scale": np.ones(n_src, dtype=np.float32),
                "sim": [mock_fn] * n_src,
                "src_index": hdf5_indices,
                "randoms": list(randoms),
            },
            schema=make_parquet_schema(),
        )
    return galaxy_table


def process_sim_worker(mock, zarr_path, sed_component):
    """Worker function: process a single sim completely.
    
    Each worker process has its own HDF5 file handle (no contention).
    Returns metadata and SEDs for one sim.
    """ 

    mock_fn = os.path.basename(mock)

    t0 = time.time()
    try:
        galaxy_table = process_galaxy_partition(
            mock, zarr_path, sed_component=sed_component
        )
        dt = time.time() - t0
        
        if galaxy_table is not None:
            print(f"           Processed in {dt:.1f}s")
            return galaxy_table, mock_fn, sed_component
        
    except Exception as e:
        print(f"  sim {mock_fn}/{sed_component}: ERROR - {e}")

    return None, mock_fn, sed_component


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


def write_galaxy_partition(store, mock_fn, sed_component, galaxy_seds):
    """Write a single galaxy SED partition to an open Zarr store.

    Parameters
    ----------
    store : zarr.Group
    mock_fn : str
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
        f"galaxy_seds/sim_{mock_fn}/{sed_component}",
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
    print(f"  galaxy_seds/sim_{mock_fn}: {galaxy_seds.shape} ({size_mb:.0f} MB uncompressed)")


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

def get_galaxy_mocks(galacticus_dir, test_flag=None):
    """Gather all hdf5 files to be run. 

    Parameters
    ----------
    galacticus_dir: Path
    test_flag: str | None
    """
    if test_flag is not None:
        return glob(str(galacticus_dir / test_flag))
    
    galaxy_mocks = glob(str(galacticus_dir / '*.hdf5'))

    return sorted(galaxy_mocks)

def main():

    parser = argparse.ArgumentParser(
        description="Build source catalog from Galacticus mock + star catalog",
    )
    parser.add_argument(
        "--test", default=None,
        help="Test storing only 1 sim. Sim to test must be named specifically.",
    )
    parser.add_argument(
        "--galacticus-dir",
        default=str(Path.home() / "data/Roman/galacticus_4deg2_mock"),
        help="Path to Galacticus HDF5",
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
    parser.add_argument(
        "--sed-component", default="disk",
        help="Component of the galaxy SED to evaluate. Options: disk, spheroid, total, or both"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    galacticus_dir = Path(args.galacticus_dir)
    sed_component = args.sed_component
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building source catalog")
    print(f"  Output: {output_dir}")

    # --- Stars ---
    metadata_tables = []
    if not args.no_stars:
        star_table, star_seds = process_stars(args.star_dir, WAVELENGTHS)
        metadata_tables.append(star_table)
    else:
        print("--- Skipping stars ---")
        star_seds = np.empty((1, N_WL), dtype=np.float32)

    # --- Galaxies ---
    print("\n--- Galaxies ---")

    galaxy_mocks = get_galaxy_mocks(galacticus_dir, args.test)

    # Initialize Zarr store (wavelengths + stars)
    zarr_store = init_zarr_store(output_dir, WAVELENGTHS, star_seds)

    # Process and write galaxy partitions incrementally
    n_partitions = 0

    # Determine number of workers to use based on available memory
    total_mem = psutil.virtual_memory().total
    per_process_mem = 0.75 * 1024**3  # estimate ~0.75 GB per worker
    max_workers = int(max(1, total_mem // per_process_mem))

    # Use ProcessPoolExecutor for cleaner API
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all sim jobs
        if sed_component=="both":
            futures = {
                executor.submit(process_sim_worker, mock, output_dir/"seds.zarr", component): None
                for mock in galaxy_mocks
                for component in ('disk', 'spheroid')
            }
        else:
            futures = {
                executor.submit(process_sim_worker, mock, output_dir/"seds.zarr", sed_component): None
                for mock in galaxy_mocks
            }
        
        # Collect results as they complete (order doesn't matter)
        for future in as_completed(futures):
            galaxy_table, mock_fn, sed_component = future.result()

            if galaxy_table is not None:
                metadata_tables.append(galaxy_table)
                n_partitions += 1

                if n_partitions % 10 == 0:
                    print(n_partitions, " completed")

            del futures[future] # free cached result inside Futures

    # --- Write outputs ---
    if not metadata_tables:
        print("\nNo data to write!")
        sys.exit(1)

    # Finalize Zarr store
    zarr_store_final = zarr.open(str(output_dir / "seds.zarr"), mode='a')
    finalize_zarr_store(zarr_store_final, n_partitions)

    # Combine and write metadata
    combined = pa.concat_tables(metadata_tables)
    parquet_path = output_dir / "metadata.parquet"
    pq.write_table(combined, parquet_path)
    print(f"\nMetadata: {parquet_path} ({combined.num_rows} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
    
#!/usr/bin/env python
"""
Build simulated Roman grism images from a unified source catalog.

Disperses both stars (via star_disperser) and galaxies (via galaxy_disperser
with per-source Sersic morphologies) from the unified Parquet+Zarr catalog.

Supports two modes:

1. **Quick mode** --- single pointing, single SCA:

    pixi run -e cuda python scripts/build_grism_image.py \
        --pointing-ra 9.5 --pointing-dec 0.95 --pointing-pa 0.0 \
        --sca 5 --output my_field.fits --seed 42

2. **Batch mode** --- YAML config + ECSV pointing table (APT format):

    pixi run -e cuda python scripts/build_grism_image.py \
        --config my_config.yaml --pointings pointings.ecsv

   The config YAML contains simulation parameters and data paths.
   The ECSV file contains the pointing list in APT format (RA, Dec, PA,
   exposure time, and APT identifiers).  See ``--generate-config`` for
   a documented template config.

Can also be imported as a module:

    from scripts.build_grism_image import setup_pipeline, process_pointing

**Memory note:** PSF payloads, dispersers, and JIT-compiled functions are built
per-SCA and released after processing, so only one SCA's compiled code lives
in memory at a time (~2-3 GB vs ~18+ GB for all 18 SCAs).  The on-disk JAX
compilation cache (``/tmp/jax-cache-grism``) makes subsequent runs fast
(~2.5s/fn vs ~10s first compile).  Galaxy SEDs are also loaded per-SCA
(~200 MB vs ~7 GB for all cone galaxies).
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# -- Early CLI pre-parse for flags that must be set before JAX import --------
# --gpu sets CUDA_VISIBLE_DEVICES; --cache-dir sets JAX_COMPILATION_CACHE_DIR.
# We parse these before importing JAX so the env vars take effect.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=None)
_pre.add_argument("--cache-dir", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)
# Cache dir precedence: CLI --cache-dir > env var > default
if _pre_args.cache_dir is not None:
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _pre_args.cache_dir
else:
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "jax-cache-grism"),
    )

import hashlib
import subprocess

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import zarr

from roman_disperser import (
    galaxy_disperser, psf_model, sersic, star_disperser,
)
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.pipeline import (
    DETECTOR_SIZE, ORDERS, LAM_MIN, LAM_MAX,
    resolve_paths, cone_search, select_sources_per_order,
    load_sensitivities,
    make_batched_star_fori, disperse_batched_stars,
    make_batched_galaxy_fori, disperse_batched_galaxies,
    write_fits, write_png, write_mosaic_png, write_mosaic_from_directory,
)
import roman_disperser.optical_model_jax as omj


def get_git_sha():
    """Return the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def make_pointing_key(seed, pointing_filename, plan, pass_, segment,
                      observation, visit, exposure):
    """Derive a deterministic JAX RNG key for a pointing.

    The key is derived from the seed, pointing filename (as salt), and the
    APT identifiers.  This ensures:
    - Same seed + same pointing file + same exposure → same key
    - Slicing or reordering the ECSV does not change keys
    - Different pointing files get different keys (filename salt)

    Parameters
    ----------
    seed : int
        Top-level RNG seed.
    pointing_filename : str
        Basename of the ECSV pointing file (used as salt).
    plan, pass_, segment, observation, visit, exposure : int
        APT identifiers for this pointing.

    Returns
    -------
    key : jax.random.key
    """
    # Build a deterministic hash from filename + APT identifiers
    tag = f"{pointing_filename}:{plan}.{pass_}.{segment}.{observation}.{visit}.{exposure}"
    h = hashlib.sha256(tag.encode()).digest()
    # Use first 4 bytes as an offset, combined with the user seed
    offset = int.from_bytes(h[:4], "big")
    return jax.random.key(seed ^ offset)


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------

def load_catalog(catalog_dir):
    """Load the unified source catalog (Parquet metadata + Zarr SEDs).

    Parameters
    ----------
    catalog_dir : str or Path
        Directory containing ``metadata.parquet`` and ``seds.zarr/``.

    Returns
    -------
    meta : pandas.DataFrame
        Source metadata (one row per source).
    store : zarr.Group
        Opened Zarr v3 store with wavelengths, star_seds, galaxy_seds.
    wavelengths : ndarray [N_wl]
        Wavelength grid in Angstroms.
    """
    catalog_dir = Path(catalog_dir)
    meta = pq.read_table(catalog_dir / "metadata.parquet").to_pandas()
    store = zarr.open(str(catalog_dir / "seds.zarr"), mode="r")
    wavelengths = np.array(store["wavelengths"])
    return meta, store, wavelengths


def validate_catalog(meta, store, wavelengths):
    """Validate catalog consistency.

    Checks wavelength coverage/spacing, required columns and types,
    sed_index bounds, NaN/Inf in critical columns, morphology ranges,
    and sim partition existence.

    Raises ValueError on hard errors; prints warnings for soft issues.
    """
    # Wavelength coverage
    wl_min_ang = LAM_MIN * 1e4
    wl_max_ang = LAM_MAX * 1e4
    if wavelengths[0] > wl_min_ang or wavelengths[-1] < wl_max_ang:
        raise ValueError(
            f"Catalog wavelengths [{wavelengths[0]:.0f}, {wavelengths[-1]:.0f}] A "
            f"do not cover required range [{wl_min_ang:.0f}, {wl_max_ang:.0f}] A"
        )

    # Wavelength spacing: should be uniform
    dlam = np.diff(wavelengths)
    if not np.allclose(dlam, dlam[0], rtol=1e-4):
        raise ValueError("Catalog wavelength grid is not uniformly spaced")

    # Required columns
    required = ["ra", "dec", "type", "n", "half_light_radius", "pa", "ba",
                "sed_index", "flux_scale", "sim"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type values
    valid_types = {"PSF", "SER"}
    actual_types = set(meta["type"].unique())
    bad_types = actual_types - valid_types
    if bad_types:
        raise ValueError(f"Invalid type values: {bad_types}")

    # NaN/Inf in critical columns
    for col in ["ra", "dec", "sed_index", "flux_scale"]:
        arr = meta[col].values
        if np.any(~np.isfinite(arr.astype(float))):
            raise ValueError(f"NaN or Inf found in column '{col}'")

    # sed_index bounds for stars
    stars = meta[meta["type"] == "PSF"]
    if len(stars) > 0:
        star_seds = store["star_seds"]
        max_star_idx = int(stars["sed_index"].max())
        if max_star_idx >= star_seds.shape[0]:
            raise ValueError(
                f"Star sed_index {max_star_idx} exceeds star_seds size "
                f"{star_seds.shape[0]}"
            )

    # Galaxy sim partition existence
    galaxies = meta[meta["type"] == "SER"]
    if len(galaxies) > 0:
        for sim_val in galaxies["sim"].unique():
            key = f"galaxy_seds/sim_{sim_val:03d}"
            if key not in store:
                raise ValueError(f"Missing Zarr array for partition: {key}")

    # Morphology warnings
    if len(galaxies) > 0:
        large = galaxies["half_light_radius"] > 1.0
        if large.any():
            n_large = int(large.sum())
            print(f"  WARNING: {n_large} galaxies have half_light_radius > 1 arcsec")


def trim_wavelength_grid(wavelengths):
    """Trim the catalog wavelength grid to the prism range [7500, 18500] A.

    Parameters
    ----------
    wavelengths : ndarray [N_wl]
        Full catalog wavelength grid in Angstroms.

    Returns
    -------
    wavelengths_trimmed : ndarray [N_trim]
        Trimmed wavelengths in Angstroms.
    wl_mask : ndarray [N_wl] bool
        Boolean mask into the original wavelength array.
    dlam_angstroms : float
        Wavelength spacing in Angstroms.
    """
    wl_min_ang = LAM_MIN * 1e4  # 7500
    wl_max_ang = LAM_MAX * 1e4  # 18500
    wl_mask = (wavelengths >= wl_min_ang) & (wavelengths <= wl_max_ang)
    wavelengths_trimmed = wavelengths[wl_mask]
    dlam_angstroms = float(wavelengths_trimmed[1] - wavelengths_trimmed[0])
    return wavelengths_trimmed, wl_mask, dlam_angstroms


# Catalog SED sanity limit. Empirically, the Galacticus catalog
# `catalogs_padded` has p99.999 max(SED) ~ 9e-15, with a 31-order-of-magnitude
# gap to a handful of pathological SEDs containing isolated single-bin spikes
# (e.g. sim_071/sed[555] = 3.5e28). Those spikes overflow float32 during the
# galaxy disperser's FFT convolution and produce NaN/Inf along the source's
# spectral trace. This threshold is far above any plausible physical SED in
# the existing catalog and below the smallest known pathological value.
_GALAXY_SED_VALUE_LIMIT = 1e-12


def load_galaxy_seds(store, galaxy_meta, wl_mask):
    """Load galaxy SEDs from Zarr, grouping by sim partition for efficient I/O.

    Parameters
    ----------
    store : zarr.Group
        Opened Zarr store.
    galaxy_meta : pandas.DataFrame
        Subset of metadata for galaxies only (must have 'sim', 'sed_index',
        'flux_scale' columns). Row order is preserved.
    wl_mask : ndarray bool
        Boolean mask for trimming wavelengths.

    Returns
    -------
    spectra : ndarray [N_galaxies, N_wl_trimmed] float32
        Galaxy SEDs scaled by flux_scale, in row order matching galaxy_meta.
    """
    n_galaxies = len(galaxy_meta)
    n_wl = int(wl_mask.sum())
    spectra = np.zeros((n_galaxies, n_wl), dtype=np.float32)

    bad_total = 0
    bad_galaxies = []  # (sim, sed_index, n_bins, max_val)

    # Group by sim partition for sequential Zarr access
    for sim_val, group in galaxy_meta.groupby("sim"):
        key = f"galaxy_seds/sim_{sim_val:03d}"
        arr = store[key]
        indices = group["sed_index"].values
        scales = group["flux_scale"].values.astype(np.float32)

        # Read all needed rows from this partition
        seds_full = np.array(arr[indices])  # [N_group, N_wl_full]
        seds_trimmed = seds_full[:, wl_mask]

        # Scrub catalog SED corruption: zero any non-finite or out-of-range
        # bins. Only flagged bins are zeroed; the rest of the SED is preserved.
        bad_mask = ~np.isfinite(seds_trimmed) | (
            np.abs(seds_trimmed) > _GALAXY_SED_VALUE_LIMIT
        )
        if bad_mask.any():
            for j, idx in enumerate(indices):
                row_bad = bad_mask[j]
                if row_bad.any():
                    bad_galaxies.append((
                        int(sim_val), int(idx), int(row_bad.sum()),
                        float(np.abs(seds_trimmed[j, row_bad]).max()),
                    ))
                    bad_total += int(row_bad.sum())
            seds_trimmed = np.where(bad_mask, 0.0, seds_trimmed)

        # Scale and place into output array (preserving original row order)
        iloc_positions = [galaxy_meta.index.get_loc(idx) for idx in group.index]
        for j, pos in enumerate(iloc_positions):
            spectra[pos] = seds_trimmed[j] * scales[j]

    if bad_galaxies:
        # Deduplicate by (sim, sed_index) — same SED may appear multiple times
        # via RA padding replication.
        unique = {(s, i): (s, i, n, m) for s, i, n, m in bad_galaxies}
        print(f"  WARNING: scrubbed {bad_total} pathological SED bins in "
              f"{len(unique)} unique catalog SED template(s):")
        for s, i, n, m in sorted(unique.values()):
            print(f"    sim_{s:03d}/sed[{i}]: {n} bin(s) zeroed, "
                  f"max abs value was {m:.3e}")

    return spectra


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def setup_pipeline(
    sca_list,
    *,
    catalog_dir=None,
    sensitivity_dir=None,
    optical_model_path=None,
    psf_cache_dir=None,
    star_batch_size=1000,
    galaxy_batch_size=100,
    galaxy_npix=30,
    verbose=True,
):
    """One-time setup: load model, catalog, optical payloads, sensitivities.

    Loads lightweight data that can be shared across pointings.  PSF loading,
    disperser construction, and JIT compilation are deferred to
    ``process_pointing`` (per-SCA) to keep memory bounded.  The JAX
    compilation cache (set at module level) avoids recompilation across
    pointings.

    Parameters
    ----------
    sca_list : list of int
        SCA numbers to prepare (1-18).
    catalog_dir, sensitivity_dir, optical_model_path, psf_cache_dir : str, optional
        Override default data paths.
    star_batch_size : int
        Number of stars per JIT batch (default: 1000).
    galaxy_batch_size : int
        Number of galaxies per JIT batch (default: 100).
    galaxy_npix : int
        Sersic image size in native pixels (default: 30).
        Oversampled size is ``galaxy_npix * oversample``.
    verbose : bool
        Print progress information.

    Returns
    -------
    pipeline : dict
        Contains all shared state for ``process_pointing``.
    """
    timings = {}
    t_total = time.time()

    def log(msg):
        if verbose:
            print(msg)

    catalog_dir, sensitivity_dir, optical_model_path, psf_cache_dir = \
        resolve_paths(catalog_dir, sensitivity_dir,
                      optical_model_path, psf_cache_dir)

    # -- Load & validate catalog ---------------------------------------------
    log("Loading catalog...")
    t0 = time.time()
    meta, store, wavelengths_full = load_catalog(catalog_dir)
    timings["load_catalog"] = time.time() - t0
    n_stars = int((meta["type"] == "PSF").sum())
    n_galaxies = int((meta["type"] == "SER").sum())
    log(f"  {len(meta)} sources ({n_stars} stars, {n_galaxies} galaxies) "
        f"in {timings['load_catalog']:.2f}s")

    log("Validating catalog...")
    validate_catalog(meta, store, wavelengths_full)

    # -- Trim wavelength grid ------------------------------------------------
    wavelengths_ang, wl_mask, dlam_angstroms = trim_wavelength_grid(
        wavelengths_full
    )
    wavelengths_um = (wavelengths_ang / 1e4).astype(np.float32)
    wavelengths_jax = jnp.array(wavelengths_um)
    n_wavelength = len(wavelengths_um)
    log(f"Wavelength grid: {LAM_MIN}-{LAM_MAX} um, "
        f"{dlam_angstroms:.1f} A spacing, {n_wavelength} samples")

    # -- Load full star SED array (trimmed) ----------------------------------
    log("Loading star SED templates...")
    t0 = time.time()
    star_seds_all = np.array(store["star_seds"])[:, wl_mask].astype(np.float32)
    timings["load_star_seds"] = time.time() - t0
    log(f"  {star_seds_all.shape[0]} templates, "
        f"{star_seds_all.nbytes / 1e6:.1f} MB "
        f"in {timings['load_star_seds']:.2f}s")

    # -- Optical model -------------------------------------------------------
    log("Loading optical model...")
    model = RomanOpticalModel(config_file=str(optical_model_path))

    # -- Per-SCA setup: optical payloads and sensitivity curves ---------------
    # PSFs, dispersers, and JIT compilation are deferred to process_pointing
    # so that only one SCA's compiled functions live in memory at a time.
    log(f"Setting up {len(sca_list)} SCAs...")
    sca_data = {}

    # Determine oversample from first PSF payload
    # (all PSF payloads use the same oversample)
    first_psf = psf_model.get_or_make_psf_payload(
        detector=f"WFI{sca_list[0]:02d}", order="1",
        cache_dir=str(psf_cache_dir), verbose=False,
    )
    oversample = int(first_psf["oversample"])
    galaxy_npix_os = galaxy_npix * oversample
    log(f"  PSF oversample: {oversample}x, "
        f"galaxy image: {galaxy_npix}px native = {galaxy_npix_os}px oversampled")

    for sca_num in sca_list:
        detector_name = f"WFI{sca_num:02d}"
        log(f"\n  SCA {sca_num} ({detector_name}):")

        # Optical payloads (small — polynomial coefficients)
        optical_payloads = {
            order: omj.make_sca_payload(model, sca=sca_num, order=order)
            for order in ORDERS
        }

        # Sensitivity curves (tiny — one array per order)
        sensitivities = load_sensitivities(
            sensitivity_dir, sca_num, wavelengths_um,
        )

        sca_data[sca_num] = {
            "optical_payloads": optical_payloads,
            "sensitivities": sensitivities,
        }

    timings["setup_total"] = time.time() - t_total
    log(f"\nSetup complete in {timings['setup_total']:.1f}s")

    return {
        "model": model,
        "meta": meta,
        "store": store,
        "star_seds_all": star_seds_all,
        "wl_mask": wl_mask,
        "wavelengths_um": wavelengths_um,
        "wavelengths_ang": wavelengths_ang,
        "wavelengths_jax": wavelengths_jax,
        "dlam_angstroms": dlam_angstroms,
        "sca_list": sca_list,
        "sca_data": sca_data,
        "psf_cache_dir": str(psf_cache_dir),
        "star_batch_size": star_batch_size,
        "galaxy_batch_size": galaxy_batch_size,
        "galaxy_npix": galaxy_npix,
        "galaxy_npix_os": galaxy_npix_os,
        "oversample": oversample,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Per-pointing processing
# ---------------------------------------------------------------------------

def process_pointing(
    pipeline,
    pointing_ra,
    pointing_dec,
    pointing_pa,
    output_dir,
    *,
    cone_radius=0.6,
    exptime=190.22,
    pointing_key=None,
    seed=0,
    verbose=True,
    extra_headers=None,
    extra_meta=None,
):
    """Process a single pointing: select sources, generate spectra, disperse.

    Parameters
    ----------
    pipeline : dict
        From ``setup_pipeline``.
    pointing_ra, pointing_dec, pointing_pa : float
        Telescope pointing in degrees.
    output_dir : str or Path
        Output directory for this pointing.  Created if needed.
    cone_radius : float
        Cone search radius in degrees (default: 0.6).
    exptime : float
        Exposure time in seconds (default: 190.22).
    pointing_key : jax.random.key or None
        JAX RNG key for this pointing.  Split into per-SCA keys.
    seed : int
        Top-level seed (stored in FITS header for provenance).
    verbose : bool
        Print progress information.
    extra_headers : dict, optional
        Additional FITS header cards as ``{keyword: (value, comment)}``.
        Written to the primary HDU of each per-SCA FITS file.
    extra_meta : dict, optional
        Additional fields to include in the per-pointing metadata YAML.

    Returns
    -------
    sca_outputs : dict mapping sca (int) -> jnp.ndarray [4088, 4088]
    """
    t_total = time.time()

    def log(msg):
        if verbose:
            print(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"grism_{output_dir.name}"

    meta = pipeline["meta"]
    sca_list = pipeline["sca_list"]
    star_batch_size = pipeline["star_batch_size"]
    galaxy_batch_size = pipeline["galaxy_batch_size"]
    galaxy_npix_os = pipeline["galaxy_npix_os"]
    oversample = pipeline["oversample"]

    # Split pointing key into per-SCA keys
    sca_keys = {}
    if pointing_key is not None:
        split_keys = jax.random.split(pointing_key, len(sca_list))
        for i, sca_num in enumerate(sca_list):
            sca_keys[sca_num] = split_keys[i]

    log(f"\nPointing: RA={pointing_ra}, Dec={pointing_dec}, PA={pointing_pa}")
    log(f"Output:   {output_dir}")

    # -- Step 1: Cone search -------------------------------------------------
    log("  Cone search...")
    t0 = time.time()
    ra_all = meta["ra"].values
    dec_all = meta["dec"].values
    cone_mask = cone_search(ra_all, dec_all, pointing_ra, pointing_dec,
                            cone_radius)
    n_cone = int(cone_mask.sum())
    log(f"    {n_cone} sources within {cone_radius} deg "
        f"in {time.time() - t0:.2f}s")

    if n_cone == 0:
        log("    WARNING: No sources in cone.")

    meta_cone = meta[cone_mask].copy()
    meta_cone_reset = meta_cone.reset_index(drop=True)

    # Separate types
    is_star = (meta_cone_reset["type"] == "PSF").values
    is_galaxy = (meta_cone_reset["type"] == "SER").values
    n_stars_cone = int(is_star.sum())
    n_galaxies_cone = int(is_galaxy.sum())
    log(f"    {n_stars_cone} stars, {n_galaxies_cone} galaxies")

    # -- Step 2: Sky -> FPA --------------------------------------------------
    if n_cone > 0:
        log("  Sky -> FPA conversion...")
        t0 = time.time()
        ra_cone = meta_cone_reset["ra"].values
        dec_cone = meta_cone_reset["dec"].values
        xfpa, yfpa = omj.get_fpa_pos(
            jnp.array(ra_cone), jnp.array(dec_cone),
            pointing_ra, pointing_dec, pointing_pa,
        )
        log(f"    Done in {time.time() - t0:.2f}s")
    else:
        xfpa = jnp.array([], dtype=jnp.float32)
        yfpa = jnp.array([], dtype=jnp.float32)

    # -- Step 3: Load star SEDs into CPU memory ------------------------------
    # Galaxy SEDs are loaded per-SCA below to avoid holding all cone galaxies
    # in memory at once (~7 GB for dense fields).
    star_spectra_all = None

    if n_stars_cone > 0:
        log("  Loading star spectra...")
        t0 = time.time()
        star_meta = meta_cone_reset[is_star]
        sed_indices = star_meta["sed_index"].values
        flux_scales = star_meta["flux_scale"].values.astype(np.float32)
        star_spectra_all = (
            pipeline["star_seds_all"][sed_indices] *
            flux_scales[:, np.newaxis]
        )
        log(f"    {n_stars_cone} star spectra "
            f"({star_spectra_all.nbytes / 1e6:.1f} MB) "
            f"in {time.time() - t0:.2f}s")

    # -- Step 4: Per-SCA loop ------------------------------------------------
    sca_outputs = {}
    sca_model_np = {}
    source_counts = {}
    manifest_rows = []

    n_wavelength = len(pipeline["wavelengths_um"])
    wavelengths_jax = pipeline["wavelengths_jax"]
    dlam_angstroms = pipeline["dlam_angstroms"]
    psf_cache_dir = pipeline["psf_cache_dir"]

    for sca_num in sca_list:
        log(f"\n  SCA {sca_num}:")
        t_sca = time.time()
        sd = pipeline["sca_data"][sca_num]

        # Skip if FITS already exists
        stem = f"{prefix}_detSCA{sca_num:02d}"
        fits_path = str(output_dir / f"{stem}.fits")
        png_path = str(output_dir / f"{stem}.png")
        if Path(fits_path).exists():
            log(f"    FITS exists, skipping.")
            # Load existing model for mosaic
            from astropy.io import fits as afits
            with afits.open(fits_path) as hdul:
                sca_model_np[sca_num] = hdul["MODEL"].data.copy()
            continue

        # -- Build dispersers and JIT-compile for this SCA -------------------
        # Built fresh per-SCA and released after processing to bound memory.
        # The on-disk JAX compilation cache (/tmp/jax-cache-grism) makes
        # subsequent runs fast (~2.5s/fn vs ~10s without cache).
        t_jit = time.time()
        detector_name = f"WFI{sca_num:02d}"

        psf_payloads = {}
        for psf_order in ORDERS:
            psf_payloads[psf_order] = psf_model.get_or_make_psf_payload(
                detector=detector_name, order=psf_order,
                cache_dir=psf_cache_dir, verbose=False,
            )

        star_fori_fns = {}
        galaxy_fori_fns = {}
        for order in ORDERS:
            sd_fn = star_disperser.make_star_disperser(
                psf_payloads[order], sd["optical_payloads"][order],
            )
            star_fori_fns[order] = make_batched_star_fori(
                sd_fn, sd["sensitivities"][order],
                wavelengths_jax, dlam_angstroms,
            )
            gd_fn = galaxy_disperser.make_galaxy_disperser(
                psf_payloads[order], sd["optical_payloads"][order],
            )
            galaxy_fori_fns[order] = make_batched_galaxy_fori(
                gd_fn, sd["sensitivities"][order],
                wavelengths_jax, dlam_angstroms,
            )

        # JIT warmup (hits disk cache after first run)
        warmup_output = jnp.zeros(
            (DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32,
        )
        warmup_spec = jnp.zeros(
            (star_batch_size, n_wavelength), dtype=jnp.float32,
        )
        warmup_x = jnp.zeros(star_batch_size, dtype=jnp.float32)
        warmup_y = jnp.zeros(star_batch_size, dtype=jnp.float32)
        warmup_gspec = jnp.zeros(
            (galaxy_batch_size, n_wavelength), dtype=jnp.float32,
        )
        warmup_gx = jnp.zeros(galaxy_batch_size, dtype=jnp.float32)
        warmup_gy = jnp.zeros(galaxy_batch_size, dtype=jnp.float32)
        warmup_imgs = jnp.zeros(
            (galaxy_batch_size, galaxy_npix_os, galaxy_npix_os),
            dtype=jnp.float32,
        )
        for order in ORDERS:
            star_fori_fns[order](
                1, warmup_spec, warmup_x, warmup_y, warmup_output,
            ).block_until_ready()
            galaxy_fori_fns[order](
                1, warmup_gspec, warmup_gx, warmup_gy,
                warmup_imgs, warmup_output,
            ).block_until_ready()
        del warmup_output, warmup_spec, warmup_x, warmup_y
        del warmup_gspec, warmup_gx, warmup_gy, warmup_imgs
        log(f"    JIT compile: {time.time() - t_jit:.1f}s")

        # -- Select sources for this SCA (per order) -------------------------
        order_masks, any_mask = select_sources_per_order(
            sd["optical_payloads"], xfpa, yfpa,
        )
        n_any = int(any_mask.sum())

        sca_counts = {order: {"stars": 0, "galaxies": 0} for order in ORDERS}
        source_counts[sca_num] = sca_counts

        # Disperse sources
        output = jnp.zeros(
            (DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32,
        )

        if n_any > 0:
            # Get SCA coordinates (order "1" defines the undispersed position)
            xfpa_sel = xfpa[any_mask]
            yfpa_sel = yfpa[any_mask]
            xsca_all, ysca_all = omj.fpa_to_sca(
                sd["optical_payloads"]["1"], xfpa_sel, yfpa_sel,
            )
            xsca_all_np = np.asarray(xsca_all)
            ysca_all_np = np.asarray(ysca_all)

            # -- Index mapping overview --
            # We have three nested subsets of cone sources:
            #
            #   cone (n_cone)  →  on-detector (n_any)  →  per-order
            #          ↓                   ↓
            #   star_spectra_all    xsca_all / ysca_all
            #                       galaxy_spectra_sel (loaded per SCA)
            #                       galaxy_images      (generated per SCA)
            #
            # star_spectra_all is a dense [n_stars_cone, N_wl] array indexed
            # by rank among stars in the cone.  To look up star spectra for
            # a given order, we map:
            #
            #   per-order index  →  on-detector index  →  cone index
            #                                           →  rank within stars
            #
            # cone_indices maps on-detector → cone position.
            # searchsorted(star_cone_indices, cone_pos) gives the rank within
            # the star-only array.
            #
            # Galaxy spectra and images are both loaded/generated per SCA
            # from is_galaxy_sel, so they share the same indexing: rank
            # among selected galaxies on this SCA.

            any_mask_np = np.asarray(any_mask) if not isinstance(any_mask, np.ndarray) else any_mask
            is_star_sel = is_star[any_mask_np]
            is_galaxy_sel = is_galaxy[any_mask_np]
            n_stars_sel = int(is_star_sel.sum())
            n_galaxies_sel = int(is_galaxy_sel.sum())
            log(f"    Selected: {n_stars_sel} stars, {n_galaxies_sel} galaxies")

            cone_indices = np.where(any_mask_np)[0]
            star_cone_indices = np.where(is_star)[0]

            order_masks_sel = {
                order: order_masks[order][any_mask_np] for order in ORDERS
            }

            # --- Load galaxy SEDs and generate images per SCA ---
            galaxy_images = None
            galaxy_spectra_sel = None
            if n_galaxies_sel > 0:
                t0 = time.time()
                gal_sel_meta = meta_cone_reset.iloc[
                    cone_indices[is_galaxy_sel]
                ]
                galaxy_spectra_sel = load_galaxy_seds(
                    pipeline["store"], gal_sel_meta, pipeline["wl_mask"],
                )
                log(f"    Galaxy SEDs: {n_galaxies_sel} spectra "
                    f"({galaxy_spectra_sel.nbytes / 1e6:.1f} MB) "
                    f"in {time.time() - t0:.2f}s")
                t0 = time.time()
                r_eff_pix = sersic.catalog_r_eff_to_pixels(
                    jnp.array(gal_sel_meta["half_light_radius"].values,
                              dtype=jnp.float32),
                    oversample=oversample,
                )
                n_sersic = jnp.array(
                    gal_sel_meta["n"].values, dtype=jnp.float32
                )
                ba = jnp.array(
                    gal_sel_meta["ba"].values, dtype=jnp.float32
                )
                theta = sersic.sky_pa_to_sca_theta(
                    jnp.array(gal_sel_meta["pa"].values, dtype=jnp.float32),
                    pointing_pa,
                )
                galaxy_images = sersic.make_sersic_images(
                    r_eff_pix, n_sersic, ba, theta, galaxy_npix_os,
                )
                galaxy_images.block_until_ready()
                log(f"    Sersic images: {n_galaxies_sel} galaxies "
                    f"({galaxy_images.nbytes / 1e6:.1f} MB) "
                    f"in {time.time() - t0:.2f}s")

            # --- Disperse per order ---
            for order in ORDERS:
                omask = order_masks_sel[order]
                n_order = int(omask.sum())
                if n_order == 0:
                    continue

                # Stars in this order
                star_omask = omask & is_star_sel
                n_star_order = int(star_omask.sum())
                sca_counts[order]["stars"] = n_star_order
                if n_star_order > 0:
                    # on-detector index → cone index → rank in star_spectra_all
                    star_order_in_sel = np.where(star_omask)[0]
                    star_cone_pos = cone_indices[star_order_in_sel]
                    star_rank = np.searchsorted(star_cone_indices, star_cone_pos)

                    x_star = xsca_all_np[star_omask]
                    y_star = ysca_all_np[star_omask]
                    spec_star = star_spectra_all[star_rank]

                    t_order = time.time()
                    output = disperse_batched_stars(
                        star_fori_fns[order],
                        spec_star, x_star, y_star,
                        output, star_batch_size,
                    )
                    elapsed = time.time() - t_order
                    ms_per = elapsed / n_star_order * 1e3
                    log(f"    Order {order}: {n_star_order} stars in "
                        f"{elapsed:.2f}s ({ms_per:.1f} ms/star)")

                # Galaxies in this order
                gal_omask = omask & is_galaxy_sel
                n_gal_order = int(gal_omask.sum())
                sca_counts[order]["galaxies"] = n_gal_order
                if n_gal_order > 0:
                    # on-detector index → rank in per-SCA arrays
                    gal_sel_positions = np.where(is_galaxy_sel)[0]
                    gal_order_in_sel = np.where(gal_omask)[0]
                    gal_rank_in_sel = np.searchsorted(
                        gal_sel_positions, gal_order_in_sel
                    )

                    x_gal = xsca_all_np[gal_omask]
                    y_gal = ysca_all_np[gal_omask]
                    spec_gal = galaxy_spectra_sel[gal_rank_in_sel]
                    imgs_gal = galaxy_images[gal_rank_in_sel]

                    t_order = time.time()
                    output = disperse_batched_galaxies(
                        galaxy_fori_fns[order],
                        spec_gal, x_gal, y_gal, imgs_gal,
                        output, galaxy_batch_size,
                    )
                    elapsed = time.time() - t_order
                    ms_per = elapsed / n_gal_order * 1e3
                    log(f"    Order {order}: {n_gal_order} galaxies in "
                        f"{elapsed:.2f}s ({ms_per:.1f} ms/galaxy)")

            # Collect manifest rows
            for order in ORDERS:
                omask = order_masks_sel[order]
                if not omask.any():
                    continue
                sel_cone_idx = cone_indices[omask]
                sel_is_star = is_star_sel[omask]
                for j, ci in enumerate(sel_cone_idx):
                    row = meta_cone_reset.iloc[ci]
                    manifest_rows.append({
                        "catalog_index": int(meta_cone.index[ci]),
                        "sca": sca_num,
                        "order": order,
                        "type": "PSF" if sel_is_star[j] else "SER",
                        "xsca": float(xsca_all_np[omask][j]),
                        "ysca": float(ysca_all_np[omask][j]),
                        "ra": float(row["ra"]),
                        "dec": float(row["dec"]),
                        "flux_scale": float(row["flux_scale"]),
                        "F158": float(row["F158"]),
                    })

        else:
            log(f"    No sources on detector.")

        # Release compiled functions and PSF payloads for this SCA
        del star_fori_fns, galaxy_fori_fns, psf_payloads

        sca_outputs[sca_num] = output

        # Log per-order counts
        for order in ORDERS:
            c = sca_counts[order]
            log(f"    Order {order}: {c['stars']} stars, "
                f"{c['galaxies']} galaxies")

        # Poisson sample on GPU
        sca_key = sca_keys.get(sca_num)
        if sca_key is not None:
            expected_counts = output * exptime
            isim = jax.random.poisson(sca_key, expected_counts).astype(
                jnp.float32,
            )
            key_data = np.array(jax.random.key_data(sca_key))
        else:
            isim = output * exptime
            key_data = np.zeros(2, dtype=np.uint32)

        # Single GPU->CPU transfer
        t0 = time.time()
        output_np = np.array(output)
        isim_np = np.array(isim)
        t_transfer = time.time() - t0
        sca_model_np[sca_num] = output_np

        # Safety net: warn if any non-finite pixels slipped through. The
        # disperser should produce only finite values when fed sane SEDs;
        # the load-time scrubber in load_galaxy_seds enforces that. A non-zero
        # count here means either a new pathological input the scrubber didn't
        # catch, or a numerical regression in the disperser itself.
        n_nan = int(np.isnan(output_np).sum())
        n_inf = int(np.isinf(output_np).sum())
        if n_nan or n_inf:
            ys, xs = np.where(~np.isfinite(output_np))
            log(f"    WARNING: SCA {sca_num} MODEL has {n_nan} NaN + "
                f"{n_inf} Inf pixels in bbox "
                f"x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

        # Write per-SCA outputs
        t0 = time.time()
        write_fits(output_np, isim_np, fits_path,
                   pointing_ra, pointing_dec, pointing_pa, sca_num,
                   exptime, key_data, seed,
                   extra_headers=extra_headers)
        t_fits = time.time() - t0
        t0 = time.time()
        write_png(output_np, png_path)
        t_png = time.time() - t0

        elapsed_sca = time.time() - t_sca
        log(f"    I/O: transfer {t_transfer:.2f}s, "
            f"FITS {t_fits:.2f}s, PNG {t_png:.2f}s")
        log(f"    Total: {elapsed_sca:.2f}s, flux={output_np.sum():.4e}, "
            f"peak={output_np.max():.4e}")

    # -- Mosaic PNG ----------------------------------------------------------
    if len(sca_list) > 1:
        log("\n  Writing mosaic PNG...")
        mosaic_path = str(output_dir / f"{prefix}_mosaic.png")
        write_mosaic_png(
            sca_model_np, sca_list, pipeline["model"], mosaic_path,
        )
        log(f"    {mosaic_path}")

    # -- Source manifest Parquet ----------------------------------------------
    if manifest_rows:
        import pyarrow as pa
        manifest_path = output_dir / f"{prefix}_sources.parquet"
        manifest_table = pa.table({
            "catalog_index": [r["catalog_index"] for r in manifest_rows],
            "sca": [r["sca"] for r in manifest_rows],
            "order": [r["order"] for r in manifest_rows],
            "type": [r["type"] for r in manifest_rows],
            "xsca": [r["xsca"] for r in manifest_rows],
            "ysca": [r["ysca"] for r in manifest_rows],
            "ra": [r["ra"] for r in manifest_rows],
            "dec": [r["dec"] for r in manifest_rows],
            "flux_scale": [r["flux_scale"] for r in manifest_rows],
            "F158": [r["F158"] for r in manifest_rows],
        })
        pq.write_table(manifest_table, str(manifest_path))
        log(f"\n  Source manifest: {len(manifest_rows)} rows -> {manifest_path}")

    # -- Metadata YAML -------------------------------------------------------
    pointing_key_data = jax.random.key_data(pointing_key).tolist() \
        if pointing_key is not None else None
    meta_yaml = {
        "pointing": {
            "ra": pointing_ra,
            "dec": pointing_dec,
            "pa": pointing_pa,
        },
        "exptime": exptime,
        "seed": seed,
        "pointing_key": pointing_key_data,
        "sca_keys": {
            sca_num: jax.random.key_data(sca_keys[sca_num]).tolist()
            for sca_num in sca_keys
        },
        "dlam_angstroms": pipeline["dlam_angstroms"],
        "cone_radius": cone_radius,
        "star_batch_size": pipeline["star_batch_size"],
        "galaxy_batch_size": pipeline["galaxy_batch_size"],
        "galaxy_npix": pipeline["galaxy_npix"],
        "oversample": pipeline["oversample"],
        "source_counts": {
            f"SCA{sca_num}": counts
            for sca_num, counts in sorted(source_counts.items())
        },
    }
    if extra_meta:
        meta_yaml.update(extra_meta)
    meta_path = output_dir / f"{prefix}_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta_yaml, f, default_flow_style=False, sort_keys=False)

    log(f"\n  Pointing complete in {time.time() - t_total:.1f}s")

    return sca_outputs


# ---------------------------------------------------------------------------
# Single-SCA convenience wrapper
# ---------------------------------------------------------------------------

def build_grism_image(
    pointing_ra,
    pointing_dec,
    pointing_pa,
    sca,
    output_file,
    *,
    seed,
    exptime=190.22,
    catalog_dir=None,
    sensitivity_dir=None,
    optical_model_path=None,
    psf_cache_dir=None,
    cone_radius=0.6,
    star_batch_size=1000,
    galaxy_batch_size=100,
    galaxy_npix=30,
    verbose=True,
    force=False,
):
    """Build a simulated grism image for a single SCA.

    Convenience wrapper around setup_pipeline + process_pointing.

    Parameters
    ----------
    pointing_ra, pointing_dec, pointing_pa : float
        Telescope pointing in degrees.
    sca : int
        SCA (detector) number, 1-18.
    output_file : str
        Output FITS filename.  A PNG with the same stem is also produced.
    seed : int
        RNG seed (required).
    exptime : float
        Exposure time in seconds (default: 190.22).
    catalog_dir, sensitivity_dir, optical_model_path, psf_cache_dir : str, optional
        Override default data paths.
    cone_radius : float
        Cone search radius in degrees (default: 0.6).
    star_batch_size : int
        Stars per JIT batch (default: 1000).
    galaxy_batch_size : int
        Galaxies per JIT batch (default: 100).
    galaxy_npix : int
        Sersic image size in native pixels (default: 30).
    verbose : bool
        Print progress information (default: True).
    force : bool
        Overwrite existing output file (default: skip).

    Returns
    -------
    output : jnp.ndarray [4088, 4088], or None if skipped
    """
    output_file = Path(output_file)
    if not force and output_file.exists():
        if verbose:
            print(f"Skipping {output_file} (already exists, "
                  f"use --force to overwrite)")
        return None

    pipeline = setup_pipeline(
        [sca],
        catalog_dir=catalog_dir,
        sensitivity_dir=sensitivity_dir,
        optical_model_path=optical_model_path,
        psf_cache_dir=psf_cache_dir,
        star_batch_size=star_batch_size,
        galaxy_batch_size=galaxy_batch_size,
        galaxy_npix=galaxy_npix,
        verbose=verbose,
    )

    # Use a temp directory, then move the files to match the requested output
    tmp_dir = output_file.parent / f".tmp_sca{sca}"
    pointing_key = jax.random.key(seed)
    sca_outputs = process_pointing(
        pipeline, pointing_ra, pointing_dec, pointing_pa,
        str(tmp_dir), cone_radius=cone_radius,
        exptime=exptime, pointing_key=pointing_key, seed=seed,
        verbose=verbose,
    )

    # Move from tmp layout to single-file output, renaming to match
    # the user-specified output stem (e.g. test_sca5.fits → test_sca5_*)
    tmp_prefix = f"grism_{tmp_dir.name}"
    out_stem = output_file.stem
    out_dir = output_file.parent

    tmp_fits = tmp_dir / f"{tmp_prefix}_detSCA{sca:02d}.fits"
    tmp_png = tmp_dir / f"{tmp_prefix}_detSCA{sca:02d}.png"
    tmp_manifest = tmp_dir / f"{tmp_prefix}_sources.parquet"
    tmp_meta = tmp_dir / f"{tmp_prefix}_meta.yaml"

    if tmp_fits.exists():
        tmp_fits.rename(output_file)
    if tmp_png.exists():
        tmp_png.rename(out_dir / f"{out_stem}.png")
    if tmp_manifest.exists():
        tmp_manifest.rename(out_dir / f"{out_stem}_sources.parquet")
    if tmp_meta.exists():
        tmp_meta.rename(out_dir / f"{out_stem}_meta.yaml")

    # Clean up temp dir
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    return sca_outputs[sca]


# ---------------------------------------------------------------------------
# Batch mode: YAML config
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = """\
# Unified grism image builder configuration
#
# Disperses both stars and galaxies from the unified Parquet+Zarr catalog.
#
# Usage (batch mode):
#   pixi run -e cuda python scripts/build_grism_image.py \\
#       --config this_file.yaml --pointings pointings.ecsv
#
# The pointing list is supplied separately as an ECSV file (APT format).
# This config file contains simulation parameters and data paths only.

# -- Output -----------------------------------------------------------------
# Top-level output directory.  Each pointing creates a subdirectory named
# {ecsv_basename}_{plan}.{pass}.{segment}.{observation}.{visit}.{exposure}/
output_dir: output/grism-fields

# -- RNG seed (required) ----------------------------------------------------
# Integer seed for reproducible Poisson noise.  Combined with the pointing
# filename and APT identifiers to derive per-pointing keys, so results are
# deterministic and independent of pointing order or slicing.
seed: 42

# -- Detectors ---------------------------------------------------------------
# Which SCAs to simulate.  Use "all" for 1-18, or list specific numbers.
scas: all
# scas: [1, 5, 12]

# -- Source selection --------------------------------------------------------
# Initial cone search radius around the pointing center (degrees).
cone_radius: 0.6

# -- Batching ----------------------------------------------------------------
# Stars per JIT batch (larger = more GPU memory, less loop overhead).
star_batch_size: 1000

# Galaxies per JIT batch (smaller than stars due to larger per-source memory).
galaxy_batch_size: 100

# Sersic image size in native pixels.  Oversampled size = galaxy_npix * oversample.
# 30 native pixels = 120 oversampled pixels at 4x oversampling.
galaxy_npix: 30

# -- JAX compilation cache (optional) ----------------------------------------
# Directory for JAX's persistent compilation cache.  Defaults to
# /tmp/jax-cache-grism (cleared on reboot).  Set to a persistent path to
# keep compiled functions across reboots.  CLI --cache-dir overrides this.
# cache_dir: /tmp/jax-cache-grism

# -- Data paths (optional, defaults shown) -----------------------------------
# Uncomment to override:
# catalog_dir: data/catalogs
# sensitivity_dir: data/sensitivities
# optical_model: data/Roman_grism_OpticalModel_v0.8.yaml
# psf_cache_dir: data/psf_cache
"""


def _pointing_dir_name(prefix, row):
    """Build the output directory name for an ECSV pointing row."""
    return (f"{prefix}"
            f"_{int(row['PLAN']):03d}"
            f".{int(row['PASS']):03d}"
            f".{int(row['SEGMENT']):03d}"
            f".{int(row['OBSERVATION']):03d}"
            f".{int(row['VISIT']):03d}"
            f".{int(row['EXPOSURE']):03d}")


def run_warmup(config_path, verbose=True, worker_index=None, num_workers=None):
    """Compile JIT functions for all SCAs and exit (no catalog needed).

    Populates the JAX compilation cache so subsequent batch runs start fast.
    Can be parallelized across GPUs by partitioning SCAs with
    ``--worker-index`` / ``--num-workers``.

    Parameters
    ----------
    config_path : str
        Path to YAML config file (for batch sizes, data paths, SCA list).
    verbose : bool
        Print progress.
    worker_index : int, optional
        This worker's index for SCA partitioning.
    num_workers : int, optional
        Total number of workers.  SCAs are assigned round-robin.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Cache dir from config (CLI already wins via pre-parse)
    if _pre_args.cache_dir is None and "cache_dir" in cfg:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cfg["cache_dir"]

    def log(msg):
        if verbose:
            print(msg)

    # Parse SCA list and apply worker partitioning
    scas = cfg.get("scas", "all")
    if scas == "all":
        sca_list = list(range(1, 19))
    else:
        sca_list = [int(s) for s in scas]

    if num_workers is not None:
        sca_list = [s for i, s in enumerate(sca_list)
                    if i % num_workers == worker_index]

    if not sca_list:
        log("No SCAs assigned to this worker.")
        return

    log(f"Warmup: compiling JIT functions for SCAs {sca_list}")
    log(f"Cache dir: {os.environ.get('JAX_COMPILATION_CACHE_DIR', '(default)')}")

    # Batch sizes (affect compiled function shapes)
    star_batch_size = cfg.get("star_batch_size", 1000)
    galaxy_batch_size = cfg.get("galaxy_batch_size", 100)
    galaxy_npix = cfg.get("galaxy_npix", 30)

    # Resolve data paths (catalog_dir needed for wavelength grid)
    catalog_dir, sensitivity_dir, optical_model_path, psf_cache_dir = \
        resolve_paths(cfg.get("catalog_dir"), cfg.get("sensitivity_dir"),
                      cfg.get("optical_model"),
                      cfg.get("psf_cache_dir"))

    # Read wavelength grid from catalog (must match what setup_pipeline uses)
    store = zarr.open(str(Path(catalog_dir) / "seds.zarr"), mode="r")
    wavelengths_full = np.array(store["wavelengths"])
    wavelengths_ang, _, dlam_angstroms = trim_wavelength_grid(wavelengths_full)
    wavelengths_um = (wavelengths_ang / 1e4).astype(np.float32)
    wavelengths_jax = jnp.array(wavelengths_um)
    n_wavelength = len(wavelengths_um)
    log(f"Wavelength grid: {LAM_MIN}-{LAM_MAX} um, "
        f"{dlam_angstroms:.1f} A spacing, {n_wavelength} samples")

    # Load optical model
    model = RomanOpticalModel(config_file=str(optical_model_path))

    # Determine oversample from first PSF payload
    first_psf = psf_model.get_or_make_psf_payload(
        detector=f"WFI{sca_list[0]:02d}", order="1",
        cache_dir=str(psf_cache_dir), verbose=False,
    )
    oversample = int(first_psf["oversample"])
    galaxy_npix_os = galaxy_npix * oversample

    t_total = time.time()
    for sca_num in sca_list:
        t_sca = time.time()
        detector_name = f"WFI{sca_num:02d}"
        log(f"\n  SCA {sca_num} ({detector_name}):")

        # Load PSF payloads
        psf_payloads = {}
        for psf_order in ORDERS:
            psf_payloads[psf_order] = psf_model.get_or_make_psf_payload(
                detector=detector_name, order=psf_order,
                cache_dir=str(psf_cache_dir), verbose=False,
            )

        # Build optical payloads and sensitivities
        optical_payloads = {
            order: omj.make_sca_payload(model, sca=sca_num, order=order)
            for order in ORDERS
        }
        sensitivities = load_sensitivities(
            sensitivity_dir, sca_num, wavelengths_um,
        )

        # Build dispersers and JIT-compile
        star_fori_fns = {}
        galaxy_fori_fns = {}
        for order in ORDERS:
            sd_fn = star_disperser.make_star_disperser(
                psf_payloads[order], optical_payloads[order],
            )
            star_fori_fns[order] = make_batched_star_fori(
                sd_fn, sensitivities[order],
                wavelengths_jax, dlam_angstroms,
            )
            gd_fn = galaxy_disperser.make_galaxy_disperser(
                psf_payloads[order], optical_payloads[order],
            )
            galaxy_fori_fns[order] = make_batched_galaxy_fori(
                gd_fn, sensitivities[order],
                wavelengths_jax, dlam_angstroms,
            )

        t_jit = time.time()
        log(f"    Build dispersers: {t_jit - t_sca:.1f}s")

        # JIT warmup calls
        warmup_output = jnp.zeros(
            (DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32,
        )
        warmup_spec = jnp.zeros(
            (star_batch_size, n_wavelength), dtype=jnp.float32,
        )
        warmup_x = jnp.zeros(star_batch_size, dtype=jnp.float32)
        warmup_y = jnp.zeros(star_batch_size, dtype=jnp.float32)
        warmup_gspec = jnp.zeros(
            (galaxy_batch_size, n_wavelength), dtype=jnp.float32,
        )
        warmup_gx = jnp.zeros(galaxy_batch_size, dtype=jnp.float32)
        warmup_gy = jnp.zeros(galaxy_batch_size, dtype=jnp.float32)
        warmup_imgs = jnp.zeros(
            (galaxy_batch_size, galaxy_npix_os, galaxy_npix_os),
            dtype=jnp.float32,
        )
        for order in ORDERS:
            star_fori_fns[order](
                1, warmup_spec, warmup_x, warmup_y, warmup_output,
            ).block_until_ready()
            galaxy_fori_fns[order](
                1, warmup_gspec, warmup_gx, warmup_gy,
                warmup_imgs, warmup_output,
            ).block_until_ready()

        # Release compiled functions and PSF payloads
        del star_fori_fns, galaxy_fori_fns, psf_payloads
        del warmup_output, warmup_spec, warmup_x, warmup_y
        del warmup_gspec, warmup_gx, warmup_gy, warmup_imgs
        log(f"    JIT warmup: {time.time() - t_jit:.1f}s")

    log(f"\nWarmup complete: {len(sca_list)} SCAs in {time.time() - t_total:.1f}s")
    log(f"Cache dir: {os.environ.get('JAX_COMPILATION_CACHE_DIR')}")


def run_batch(config_path, pointings_path, verbose=True, force=False,
              worker_index=None, num_workers=None):
    """Run the pipeline from a YAML config + ECSV pointing table.

    Parameters
    ----------
    config_path : str
        Path to YAML config file (simulation parameters and data paths).
    pointings_path : str
        Path to ECSV pointing table (APT format).
    verbose : bool
        Print progress.
    force : bool
        Overwrite existing pointing directories (default: skip them).
    worker_index : int, optional
        This worker's index (0-based) for parallel runs.
    num_workers : int, optional
        Total number of parallel workers.  When set, this worker processes
        only pointings where ``index % num_workers == worker_index``.
    """
    import warnings
    from astropy.table import Table

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def log(msg):
        if verbose:
            print(msg)

    # Cache dir from config (CLI already wins via pre-parse)
    if _pre_args.cache_dir is None and "cache_dir" in cfg:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cfg["cache_dir"]

    # Deprecation: batch_size → star_batch_size
    if "batch_size" in cfg and "star_batch_size" not in cfg:
        warnings.warn(
            "Config key 'batch_size' is deprecated, use 'star_batch_size' instead.",
            FutureWarning,
            stacklevel=2,
        )
        cfg["star_batch_size"] = cfg.pop("batch_size")
    elif "batch_size" in cfg and "star_batch_size" in cfg:
        warnings.warn(
            "Config contains both 'batch_size' and 'star_batch_size'; "
            "using 'star_batch_size' (ignoring deprecated 'batch_size').",
            FutureWarning,
            stacklevel=2,
        )
        cfg.pop("batch_size")

    # Load pointing table
    ptable = Table.read(pointings_path, format="ascii.ecsv")

    # Filter to PRISM pointings only
    if "BANDPASS" in ptable.colnames:
        prism_mask = ptable["BANDPASS"] == "PRISM"
        n_filtered = len(ptable) - prism_mask.sum()
        if n_filtered > 0:
            log(f"Filtered {n_filtered} non-PRISM rows from pointing table")
        ptable = ptable[prism_mask]

    if len(ptable) == 0:
        log("No PRISM pointings found in pointing table.")
        return

    # Parse SCA list
    scas = cfg.get("scas", "all")
    if scas == "all":
        sca_list = list(range(1, 19))
    else:
        sca_list = [int(s) for s in scas]

    seed = cfg["seed"]
    git_sha = get_git_sha()

    # Directory prefix from pointing filename
    pointing_filename = Path(pointings_path).stem
    output_dir = Path(cfg["output_dir"])

    # Check which pointings need processing before expensive setup.
    pointings_todo = []
    for idx, row in enumerate(ptable):
        # Worker partitioning: round-robin over filtered pointing list
        if num_workers is not None and idx % num_workers != worker_index:
            continue
        name = _pointing_dir_name(pointing_filename, row)
        pointing_dir = output_dir / name
        if force or not pointing_dir.exists():
            pointings_todo.append(row)

    log(f"Config: {config_path}")
    log(f"Pointings: {pointings_path}")
    log(f"SCAs: {sca_list}")
    log(f"Seed: {seed}")
    log(f"Git SHA: {git_sha}")
    if num_workers is not None:
        log(f"Worker: {worker_index}/{num_workers}")
    log(f"Pointings: {len(ptable)} total, "
        f"{len(pointings_todo)} to process")

    if not pointings_todo:
        log("Nothing to do (all pointings exist, use --force to overwrite).")
        return

    # Setup pipeline (one-time)
    pipeline = setup_pipeline(
        sca_list,
        catalog_dir=cfg.get("catalog_dir"),
        sensitivity_dir=cfg.get("sensitivity_dir"),
        optical_model_path=cfg.get("optical_model"),
        psf_cache_dir=cfg.get("psf_cache_dir"),
        star_batch_size=cfg.get("star_batch_size", 1000),
        galaxy_batch_size=cfg.get("galaxy_batch_size", 100),
        galaxy_npix=cfg.get("galaxy_npix", 30),
        verbose=verbose,
    )

    # Process pointings
    cone_radius = cfg.get("cone_radius", 0.6)
    n_skipped = len(ptable) - len(pointings_todo)

    t_all = time.time()
    for i, row in enumerate(pointings_todo):
        name = _pointing_dir_name(pointing_filename, row)
        exptime = float(row["EXPOSURE_TIME"])

        # Derive deterministic RNG key
        pointing_key = make_pointing_key(
            seed, pointing_filename,
            int(row["PLAN"]), int(row["PASS"]), int(row["SEGMENT"]),
            int(row["OBSERVATION"]), int(row["VISIT"]),
            int(row["EXPOSURE"]),
        )

        log(f"\n{'='*60}")
        log(f"Pointing {i+1}/{len(pointings_todo)}: {name}")
        log(f"  RA={row['RA']:.6f}, Dec={row['DEC']:.6f}, "
            f"PA={row['PA']:.1f}, exptime={exptime:.2f}s")
        log(f"{'='*60}")

        # Extra FITS header fields for this pointing
        extra_headers = {
            "GITSHA": (git_sha, "Git commit SHA of pipeline code"),
            "MA_TABLE": (int(row["MA_TABLE_NUMBER"]),
                         "MA table number"),
            "PLAN": (int(row["PLAN"]), "APT plan number"),
            "PASS": (int(row["PASS"]), "APT pass number"),
            "SEGMENT": (int(row["SEGMENT"]), "APT segment number"),
            "OBS": (int(row["OBSERVATION"]), "APT observation number"),
            "VISIT": (int(row["VISIT"]), "APT visit number"),
            "EXPOSURE": (int(row["EXPOSURE"]), "APT exposure number"),
        }

        # Extra metadata for the YAML file
        extra_meta = {
            "git_sha": git_sha,
            "pointing_file": str(Path(pointings_path).name),
            "apt": {
                "plan": int(row["PLAN"]),
                "pass": int(row["PASS"]),
                "segment": int(row["SEGMENT"]),
                "observation": int(row["OBSERVATION"]),
                "visit": int(row["VISIT"]),
                "exposure": int(row["EXPOSURE"]),
                "ma_table_number": int(row["MA_TABLE_NUMBER"]),
            },
        }

        pointing_dir = output_dir / name
        process_pointing(
            pipeline,
            float(row["RA"]), float(row["DEC"]), float(row["PA"]),
            str(pointing_dir),
            cone_radius=cone_radius,
            exptime=exptime,
            pointing_key=pointing_key,
            seed=seed,
            verbose=verbose,
            extra_headers=extra_headers,
            extra_meta=extra_meta,
        )

    total = time.time() - t_all
    setup_time = pipeline["timings"]["setup_total"]
    log(f"\n{'='*60}")
    log(f"All pointings complete "
        f"({len(pointings_todo)} processed, {n_skipped} skipped)")
    log(f"  Setup:      {setup_time:.1f}s")
    log(f"  Processing: {total:.1f}s")
    log(f"  Total:      {setup_time + total:.1f}s")
    log(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build simulated Roman grism images from a unified "
                    "source catalog (stars + galaxies).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --generate-config to create a documented template "
               "config file.",
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=str,
                      help="YAML config file for batch mode")
    mode.add_argument("--pointing-ra", type=float,
                      help="Pointing RA in degrees (quick mode)")
    mode.add_argument("--generate-config", type=str, metavar="FILE",
                      help="Write a documented template config file and exit")
    mode.add_argument("--mosaic", type=str, metavar="DIR",
                      help="Generate mosaic PNG from a pointing directory "
                           "containing grism_*_detSCA*.fits files")

    # Batch mode: ECSV pointing table
    parser.add_argument("--pointings", type=str,
                        help="ECSV pointing table (required for batch mode)")

    # Quick mode arguments
    parser.add_argument("--pointing-dec", type=float,
                        help="Pointing Dec in degrees (quick mode)")
    parser.add_argument("--pointing-pa", type=float,
                        help="Position angle in degrees (quick mode)")
    parser.add_argument("--sca", type=int,
                        help="SCA number, 1-18 (quick mode)")
    parser.add_argument("--output", type=str,
                        help="Output FITS filename (quick mode)")

    # Shared optional arguments
    parser.add_argument("--catalog-dir", type=str, default=None,
                        help="Path to catalog directory")
    parser.add_argument("--sensitivity-dir", type=str, default=None,
                        help="Path to sensitivity FITS files")
    parser.add_argument("--optical-model", type=str, default=None,
                        help="Path to optical model YAML")
    parser.add_argument("--psf-cache-dir", type=str, default=None,
                        help="Path to PSF cache directory")
    parser.add_argument("--cone-radius", type=float, default=0.6,
                        help="Cone search radius in degrees (default: 0.6)")
    parser.add_argument("--star-batch-size", type=int, default=1000,
                        help="Stars per JIT batch (default: 1000)")
    parser.add_argument("--galaxy-batch-size", type=int, default=100,
                        help="Galaxies per JIT batch (default: 100)")
    parser.add_argument("--galaxy-npix", type=int, default=30,
                        help="Sersic image size in native pixels (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (required for quick mode)")
    parser.add_argument("--exptime", type=float, default=190.22,
                        help="Exposure time in seconds (quick mode)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directories/files")

    # Multi-GPU / parallel worker arguments
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="JAX compilation cache directory "
                             "(default: /tmp/jax-cache-grism)")
    parser.add_argument("--worker-index", type=int, default=None,
                        help="This worker's index (0-based) for parallel runs")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Total number of parallel workers")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Compile JIT functions for all SCAs and exit "
                             "(no catalog or pointings needed)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Redirect all output to this file "
                             "(useful for parallel workers)")

    args = parser.parse_args()

    # Validate worker flags
    if (args.worker_index is None) != (args.num_workers is None):
        parser.error("--worker-index and --num-workers must be used together")
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.worker_index is not None and (
        args.worker_index < 0 or args.worker_index >= args.num_workers
    ):
        parser.error("--worker-index must be in [0, num-workers)")

    # Redirect output to log file if requested
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", buffering=1)  # line-buffered
        sys.stdout = log_fh
        sys.stderr = log_fh

    # Generate config mode
    if args.generate_config:
        with open(args.generate_config, "w") as f:
            f.write(EXAMPLE_CONFIG)
        print(f"Wrote template config to {args.generate_config}")
        return

    # Mosaic mode
    if args.mosaic:
        write_mosaic_from_directory(
            args.mosaic,
            optical_model_path=args.optical_model,
        )
        return

    # Warmup-only mode (requires --config)
    if args.warmup_only:
        if args.config is None:
            parser.error("--warmup-only requires --config")
        run_warmup(args.config, verbose=not args.quiet,
                   worker_index=args.worker_index,
                   num_workers=args.num_workers)
        return

    # Batch mode
    if args.config:
        if args.pointings is None:
            parser.error("--pointings required with --config (batch mode)")
        run_batch(args.config, args.pointings,
                  verbose=not args.quiet, force=args.force,
                  worker_index=args.worker_index,
                  num_workers=args.num_workers)
        return

    # Quick mode --- validate required arguments
    if args.pointing_dec is None or args.pointing_pa is None:
        parser.error("--pointing-dec and --pointing-pa required in quick mode")
    if args.sca is None:
        parser.error("--sca required in quick mode")
    if args.output is None:
        parser.error("--output required in quick mode")
    if args.seed is None:
        parser.error("--seed required in quick mode")

    build_grism_image(
        pointing_ra=args.pointing_ra,
        pointing_dec=args.pointing_dec,
        pointing_pa=args.pointing_pa,
        sca=args.sca,
        output_file=args.output,
        seed=args.seed,
        exptime=args.exptime,
        catalog_dir=args.catalog_dir,
        sensitivity_dir=args.sensitivity_dir,
        optical_model_path=args.optical_model,
        psf_cache_dir=args.psf_cache_dir,
        cone_radius=args.cone_radius,
        star_batch_size=args.star_batch_size,
        galaxy_batch_size=args.galaxy_batch_size,
        galaxy_npix=args.galaxy_npix,
        verbose=not args.quiet,
        force=args.force,
    )


if __name__ == "__main__":
    main()

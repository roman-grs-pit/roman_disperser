#!/usr/bin/env python
"""
Build simulated Roman grism images from a stellar catalog.

Supports two modes:

1. **Quick mode** — single pointing, single SCA:

    pixi run -e cuda python scripts/build_star_grism_image.py \
        --pointing-ra 9.5 --pointing-dec 0.95 --pointing-pa 0.0 \
        --sca 5 --output my_field.fits

2. **Batch mode** — multiple pointings, multiple SCAs via YAML config:

    pixi run -e cuda python scripts/build_star_grism_image.py \
        --config my_config.yaml

The batch mode reads a YAML config file specifying pointings and SCAs,
pre-compiles all JIT functions once, then processes each pointing
efficiently.  See the --generate-config flag to create a documented
template config file.  Existing output files are overwritten.

Can also be imported as a module:

    from scripts.build_star_grism_image import setup_pipeline, process_pointing
"""

import argparse
import os
import re
import time
from pathlib import Path

import yaml
import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import synphot as syn
from astropy.io import fits
from matplotlib.colors import AsinhNorm
from PIL import Image

from roman_disperser import catalog, psf_model, refdata, star_disperser
from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DETECTOR_SIZE = 4088
ORDERS = ["0", "1", "2"]
LAM_MIN = 0.9   # microns
LAM_MAX = 2.0   # microns
SENSITIVITY_MAP_FILE = "sensitivity_map.yaml"


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------

def load_star_catalog(catalog_dir):
    """Load the stellar catalog and template filename mapping.

    Parameters
    ----------
    catalog_dir : str or Path
        Directory containing sim_star_cat_galacticus.txt and SEDtemplates/.

    Returns
    -------
    catalog : dict with keys:
        ra      : ndarray [N] - Right ascension (degrees)
        dec     : ndarray [N] - Declination (degrees)
        mag     : ndarray [N] - F158 AB magnitude
        temp_idx: ndarray [N] - Template index into template_files list
    template_files : list of str
        Filenames for each template slot (some may be "garbage").
    """
    catalog_dir = Path(catalog_dir)
    data = np.loadtxt(catalog_dir / "sim_star_cat_galacticus.txt", skiprows=1)

    raw_template_index = data[:, 1].astype(int)
    # Wrap to valid range: input_spectral_STARS.lis has 58 templates (indices 0-57)
    temp_inds = raw_template_index % 58

    with open(catalog_dir / "SEDtemplates" / "input_spectral_STARS.lis") as f:
        template_files = [line.strip() for line in f.readlines()]

    return {
        "ra": data[:, 3],
        "dec": data[:, 4],
        "mag": data[:, 2],
        "temp_idx": temp_inds.astype(int),
    }, template_files


def load_template_sed(catalog_dir, filename):
    """Load a stellar SED template from the SEDtemplates directory.

    Parameters
    ----------
    catalog_dir : str or Path
        Base catalog directory.
    filename : str
        Template filename (e.g. "uka0v.dat").

    Returns
    -------
    wavelength : ndarray - Wavelength in Angstroms
    flux : ndarray - Flux (arbitrary units, will be renormalized)
    """
    data = np.loadtxt(Path(catalog_dir) / "SEDtemplates" / filename)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Spectrum handling
# ---------------------------------------------------------------------------

def load_templates_as_synphot(catalog_dir, template_files, unique_indices):
    """Load unique SED templates as synphot SourceSpectrum objects.

    Parameters
    ----------
    catalog_dir : str or Path
        Base catalog directory.
    template_files : list of str
        Template filename list from input_spectral_STARS.lis.
    unique_indices : array-like of int
        Unique template indices to load.

    Returns
    -------
    templates : dict mapping int -> synphot.SourceSpectrum
    """
    templates = {}
    for idx in unique_indices:
        filename = template_files[idx]
        wl_ang, flux = load_template_sed(catalog_dir, filename)
        sp = syn.SourceSpectrum(
            syn.Empirical1D,
            points=wl_ang * u.AA,
            lookup_table=flux * syn.units.FLAM,
        )
        templates[idx] = sp
    return templates


def precompute_template_grid(templates_synphot, f158_band, wavelengths_angstrom):
    """Normalize all templates to 0 ABmag in F158 and sample onto wavelength grid.

    Parameters
    ----------
    templates_synphot : dict mapping int -> synphot.SourceSpectrum
    f158_band : synphot.SpectralElement
    wavelengths_angstrom : ndarray [N_wl] in Angstroms

    Returns
    -------
    template_grid : dict mapping int -> ndarray [N_wl] in FLAM at 0 ABmag
    """
    wl_qty = wavelengths_angstrom * u.AA
    template_grid = {}
    for idx, sp in templates_synphot.items():
        norm_sp = sp.normalize(0.0 * u.ABmag, band=f158_band)
        template_grid[idx] = norm_sp(wl_qty, flux_unit=syn.units.FLAM).value.astype(
            np.float32
        )
    return template_grid


def generate_spectra(template_grid, template_indices, magnitudes):
    """Generate normalized spectra for all sources.

    Uses pre-normalized templates (at 0 ABmag) and scales by magnitude.

    Parameters
    ----------
    template_grid : dict mapping int -> ndarray [N_wl] (from precompute_template_grid)
    template_indices : ndarray [N] of int
    magnitudes : ndarray [N] of float

    Returns
    -------
    spectra_flam : ndarray [N, N_wl] in FLAM units
    """
    n_sources = len(magnitudes)
    first_key = next(iter(template_grid))
    n_wl = len(template_grid[first_key])
    spectra = np.zeros((n_sources, n_wl), dtype=np.float32)

    # Scale factor: template is at 0 ABmag, so scale by 10^(-0.4 * mag)
    scale = 10.0 ** (-0.4 * magnitudes)

    for i in range(n_sources):
        spectra[i] = template_grid[int(template_indices[i])] * scale[i]

    return spectra


# ---------------------------------------------------------------------------
# Sensitivity curves
# ---------------------------------------------------------------------------

def load_sensitivities(sensitivity_dir, sca, wavelengths):
    """Load and interpolate grism sensitivity curves for each order.

    Reads the sensitivity_map.yaml in the sensitivity directory to find
    the correct file for each SCA and order.

    Parameters
    ----------
    sensitivity_dir : str or Path
        Directory containing sensitivity FITS files and sensitivity_map.yaml.
    sca : int
        SCA number (1-18).
    wavelengths : ndarray [N_wl] in microns.

    Returns
    -------
    sensitivities : dict mapping order str -> jnp.ndarray [N_wl]
    """
    sensitivity_dir = Path(sensitivity_dir)
    with open(sensitivity_dir / SENSITIVITY_MAP_FILE) as f:
        sens_map = yaml.safe_load(f)

    sca_key = f"SCA{sca}"
    if sca_key not in sens_map:
        raise ValueError(
            f"{sca_key} not found in {SENSITIVITY_MAP_FILE}. "
            f"Available: {sorted(sens_map.keys())}"
        )

    sensitivities = {}
    for order in ORDERS:
        fname = sens_map[sca_key][order]
        with fits.open(sensitivity_dir / fname) as hdul:
            wl_sens = hdul[1].data["WAVELENGTH"] / 1e4  # Angstrom -> micron
            sens_vals = hdul[1].data["SENSITIVITY"]
            sens_interp = np.interp(wavelengths, wl_sens, sens_vals)
            sensitivities[order] = jnp.array(sens_interp.astype(np.float32))
    return sensitivities


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def cone_search(ra, dec, pointing_ra, pointing_dec, radius_deg):
    """Return boolean mask for sources within radius of pointing center.

    Uses the Haversine formula for accurate spherical distances.
    """
    d2r = np.pi / 180.0
    ra1, dec1 = pointing_ra * d2r, pointing_dec * d2r
    ra2, dec2 = ra * d2r, dec * d2r
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(a)) / d2r
    return dist <= radius_deg


def select_sources_per_order(
    optical_payloads, xfpa, yfpa, orders=ORDERS,
):
    """For each order, determine which sources have traces on the detector.

    Parameters
    ----------
    optical_payloads : dict mapping order str -> payload
    xfpa, yfpa : jnp.ndarray [N]
    orders : list of str

    Returns
    -------
    masks : dict mapping order str -> bool ndarray [N]
    any_mask : bool ndarray [N] - True if source is on detector for any order
    """
    n = len(xfpa)
    masks = {}
    any_mask = np.zeros(n, dtype=bool)
    if n == 0:
        for order in orders:
            masks[order] = any_mask.copy()
        return masks, any_mask
    for order in orders:
        mask = catalog.select_sources(optical_payloads[order], xfpa, yfpa)
        masks[order] = np.asarray(mask)
        any_mask |= masks[order]
    return masks, any_mask


# ---------------------------------------------------------------------------
# Batched dispersion
# ---------------------------------------------------------------------------

def make_batched_fori(disperser_fn, sens, wavelengths_jax, dlam_angstroms):
    """Build a JIT-compiled fori_loop that processes a fixed-size batch.

    The compiled function takes padded arrays of shape [batch_size, ...]
    and a dynamic n_sources argument controlling how many are actually
    processed. JAX traces the array shapes at the first call and reuses
    the compiled code for all subsequent calls with the same shapes.

    Parameters
    ----------
    disperser_fn : callable from make_star_disperser
    sens : jnp.ndarray [N_wl]
    wavelengths_jax : jnp.ndarray [N_wl]
    dlam_angstroms : float

    Returns
    -------
    run : callable(n_sources, spectra, xsca, ysca, output) -> output
        spectra: [batch_size, N_wl], xsca/ysca: [batch_size]
    """
    @jax.jit
    def run(n_sources, spectra, xsca, ysca, output):
        def body_fn(i, output):
            counts = spectra[i] * sens * dlam_angstroms
            return disperser_fn(xsca[i], ysca[i],
                                wavelengths_jax, counts, output)
        return jax.lax.fori_loop(0, n_sources, body_fn, output)

    return run


def disperse_batched(fori_fn, spectra, xsca, ysca, output, batch_size):
    """Disperse sources in fixed-size batches, reusing compiled code.

    Parameters
    ----------
    fori_fn : compiled fori_loop from make_batched_fori
    spectra : ndarray [N, N_wl]
    xsca, ysca : ndarray [N]
    output : jnp.ndarray [4088, 4088]
    batch_size : int
    log : callable for progress messages

    Returns
    -------
    output : jnp.ndarray [4088, 4088]
    """
    n_sources = len(xsca)
    n_wl = spectra.shape[1]
    n_batches = (n_sources + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_sources)
        n_actual = end - start

        # Pad to batch_size with zeros
        spec_batch = np.zeros((batch_size, n_wl), dtype=np.float32)
        x_batch = np.zeros(batch_size, dtype=np.float32)
        y_batch = np.zeros(batch_size, dtype=np.float32)
        spec_batch[:n_actual] = spectra[start:end]
        x_batch[:n_actual] = xsca[start:end]
        y_batch[:n_actual] = ysca[start:end]

        output = fori_fn(
            n_actual,
            jnp.array(spec_batch),
            jnp.array(x_batch),
            jnp.array(y_batch),
            output,
        )

    output.block_until_ready()
    return output


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_fits(model_np, isim_np, output_file, pointing_ra, pointing_dec,
               pointing_pa, sca, exptime, rng_key_data, seed):
    """Write the grism image to a FITS file.

    Primary HDU contains pointing/simulation metadata.  MODEL extension
    contains the noiseless count-rate image (counts/s).  ISIM extension
    contains the Poisson-sampled image (counts).

    Parameters
    ----------
    model_np : ndarray
        Noiseless count-rate image (counts/s).
    isim_np : ndarray
        Poisson-sampled image (counts).
    output_file : str
    pointing_ra, pointing_dec, pointing_pa : float
    sca : int
    exptime : float
        Exposure time in seconds.
    rng_key_data : ndarray [2] of uint32
        JAX RNG key data used for this SCA's Poisson draw.
    seed : int
        Top-level seed for the full run.
    """
    primary = fits.PrimaryHDU()
    primary.header["WFICENRA"] = (pointing_ra, "Pointing RA [deg]")
    primary.header["WFICENDEC"] = (pointing_dec, "Pointing Dec [deg]")
    primary.header["WFICENPA"] = (pointing_pa, "Position angle [deg]")
    primary.header["DETNUM"] = (sca, "SCA number")
    primary.header["EXPTIME"] = (exptime, "Exposure time [s]")
    primary.header["SEED"] = (seed, "Top-level RNG seed")
    primary.header["RNDSEED0"] = (int(rng_key_data[0]), "JAX RNG key word 0")
    primary.header["RNDSEED1"] = (int(rng_key_data[1]), "JAX RNG key word 1")

    model_hdu = fits.ImageHDU(data=model_np, name="MODEL")
    isim_hdu = fits.ImageHDU(data=isim_np, name="ISIM")

    hdul = fits.HDUList([primary, model_hdu, isim_hdu])
    hdul.writeto(output_file, overwrite=True)


def write_png(output_np, png_file, linear_width=0.01):
    """Write an asinh-stretched quicklook PNG.

    Parameters
    ----------
    output_np : ndarray
        Numpy array (not JAX).  Caller converts before calling.
    """
    # Downsample for quicklook: 4088 -> 1022 pixels (4× block-average).
    bf = 4
    ny, nx = output_np.shape
    ny_trim = (ny // bf) * bf
    nx_trim = (nx // bf) * bf
    small = output_np[:ny_trim, :nx_trim].reshape(
        ny_trim // bf, bf, nx_trim // bf, bf
    ).mean(axis=(1, 3))

    # Asinh stretch + inferno colormap via matplotlib's normalizer and cmap
    vmax = small.max()
    if vmax == 0:
        vmax = 1.0
    norm = AsinhNorm(linear_width=linear_width, vmin=0, vmax=vmax)
    cmap = matplotlib.colormaps["inferno"]
    rgba = cmap(norm(small))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Flip vertically for origin="lower" convention
    rgb = rgb[::-1]

    Image.fromarray(rgb).save(png_file)


def write_mosaic_png(sca_images, sca_list, model, png_file,
                     linear_width=0.01):
    """Write a mosaic PNG showing all SCAs in their WFI focal plane layout.

    Parameters
    ----------
    sca_images : dict mapping sca (int) -> ndarray [4088, 4088]
    sca_list : list of int
    model : RomanOpticalModel
    png_file : str
    linear_width : float
    """
    # Get FPA center for each SCA
    sca_centers = {}
    for sca_num in sca_list:
        payload = omj.make_sca_payload(model, sca=sca_num, order="1")
        xfpa, yfpa = omj.sca_to_fpa(payload, 2044.5, 2044.5)
        sca_centers[sca_num] = (float(xfpa), float(yfpa))

    # Compute layout: map FPA degrees to figure coordinates
    all_x = [c[0] for c in sca_centers.values()]
    all_y = [c[1] for c in sca_centers.values()]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)

    # Each SCA thumbnail size relative to spacing
    # FPA spacing between adjacent SCAs is ~0.135 deg, detector is ~0.125 deg
    thumb_size = 0.12  # degrees, slightly smaller than actual for gaps

    fig_width = 16
    fig_height = fig_width * (y_range + 2 * thumb_size) / (x_range + 2 * thumb_size)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Global vmax across all images for consistent scaling
    global_max = max(
        (np.array(sca_images[s]).max() for s in sca_list if s in sca_images),
        default=1.0,
    )
    if global_max == 0:
        global_max = 1.0
    norm = AsinhNorm(linear_width=linear_width, vmin=0, vmax=global_max)

    x_min = min(all_x) - thumb_size
    x_max = max(all_x) + thumb_size
    y_min = min(all_y) - thumb_size
    y_max = max(all_y) + thumb_size

    for sca_num in sca_list:
        cx, cy = sca_centers[sca_num]

        # Convert FPA position to figure fraction (x-axis reversed per
        # standard Roman convention: FPA x runs ~0.4 to -0.4 left-to-right)
        left = (x_max - cx - thumb_size / 2) / (x_max - x_min)
        bottom = (cy - thumb_size / 2 - y_min) / (y_max - y_min)
        width = thumb_size / (x_max - x_min)
        height = thumb_size / (y_max - y_min)

        ax = fig.add_axes([left, bottom, width, height])

        if sca_num in sca_images:
            img = np.array(sca_images[sca_num])
            # Downsample for thumbnails: each SCA is ~200px in the figure,
            # so full 4088×4088 is wasteful.  Block-average by 8× -> 511×511.
            bf = 8
            ny, nx = img.shape
            ny_trim = (ny // bf) * bf
            nx_trim = (nx // bf) * bf
            img = img[:ny_trim, :nx_trim].reshape(
                ny_trim // bf, bf, nx_trim // bf, bf
            ).mean(axis=(1, 3))
            ax.imshow(img, origin="lower", cmap="inferno", norm=norm,
                      aspect="equal")
        else:
            ax.set_facecolor("#eeeeee")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"SCA{sca_num}", fontsize=8, pad=2, color="black")

    fig.suptitle("WFI Focal Plane Mosaic", fontsize=14, y=0.98, color="black")
    fig.savefig(png_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_mosaic_from_directory(pointing_dir, optical_model_path=None,
                                linear_width=0.01):
    """Generate a focal-plane mosaic PNG from per-SCA FITS files in a directory.

    Scans ``pointing_dir`` for files matching ``grism_*_detSCA*.fits`` (or
    the legacy ``SCA*.fits`` pattern) and reads the MODEL extension from
    each.  This can be run standalone after the main pipeline without
    reprocessing any sources.

    Parameters
    ----------
    pointing_dir : str or Path
        Directory containing per-SCA FITS files.
    optical_model_path : str or Path, optional
        Path to the optical model YAML.  Defaults to the standard location.
    linear_width : float
        Asinh normalization parameter forwarded to ``write_mosaic_png``.
    """
    pointing_dir = Path(pointing_dir)
    if optical_model_path is None:
        project_root = Path(os.environ.get("PIXI_PROJECT_ROOT", "."))
        optical_model_path = (
            project_root / "data" / "Roman_grism_OpticalModel_v0.8.yaml"
        )
    model = RomanOpticalModel(str(optical_model_path))

    # Discover SCA FITS files (try new naming first, fall back to legacy)
    sca_images = {}
    sca_list = []
    fits_files = sorted(pointing_dir.glob("grism_*_detSCA*.fits"))
    if not fits_files:
        fits_files = sorted(pointing_dir.glob("SCA*.fits"))
    for fpath in fits_files:
        # Extract SCA number from _detSCA05.fits or SCA05.fits
        m = re.search(r"SCA(\d+)", fpath.stem)
        if m is None:
            continue
        sca_num = int(m.group(1))
        with fits.open(fpath) as hdul:
            if "MODEL" in hdul:
                sca_images[sca_num] = hdul["MODEL"].data.astype(np.float32)
                sca_list.append(sca_num)

    if not sca_list:
        print(f"No per-SCA FITS files with MODEL extension found in "
              f"{pointing_dir}")
        return

    sca_list.sort()
    prefix = f"grism_{pointing_dir.name}"
    png_file = str(pointing_dir / f"{prefix}_mosaic.png")
    write_mosaic_png(sca_images, sca_list, model, png_file,
                     linear_width=linear_width)
    print(f"Mosaic written to {png_file}")



# ---------------------------------------------------------------------------
# Pipeline setup (shared across all pointings)
# ---------------------------------------------------------------------------

def resolve_paths(catalog_dir=None, sensitivity_dir=None,
                  optical_model_path=None, psf_cache_dir=None):
    """Resolve default paths relative to project root."""
    project_root = Path(os.environ.get("PIXI_PROJECT_ROOT", "."))
    if catalog_dir is None:
        catalog_dir = project_root / "data" / "stars"
    if sensitivity_dir is None:
        sensitivity_dir = project_root / "data" / "sensitivities"
    if optical_model_path is None:
        optical_model_path = (
            project_root / "data" / "Roman_grism_OpticalModel_v0.8.yaml"
        )
    if psf_cache_dir is None:
        psf_cache_dir = project_root / "data" / "psf_cache"
    return (Path(catalog_dir), Path(sensitivity_dir),
            Path(optical_model_path), Path(psf_cache_dir))


def setup_pipeline(
    sca_list,
    *,
    catalog_dir=None,
    sensitivity_dir=None,
    optical_model_path=None,
    psf_cache_dir=None,
    dlam_angstroms=2.0,
    batch_size=1000,
    verbose=True,
):
    """One-time setup: load model, catalog, PSFs, compile dispersers.

    This function performs all expensive initialization that can be shared
    across multiple pointings.  The returned ``pipeline`` dict contains
    everything needed by ``process_pointing``.

    Parameters
    ----------
    sca_list : list of int
        SCA numbers to prepare (1-18).
    catalog_dir, sensitivity_dir, optical_model_path, psf_cache_dir : str, optional
        Override default data paths.
    dlam_angstroms : float
        Wavelength spacing in Angstroms (default: 2.0).
    batch_size : int
        Number of sources per JIT batch (default: 1000).
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

    # -- Wavelength grid -----------------------------------------------------
    dlam_um = dlam_angstroms / 1e4
    n_wavelength = int((LAM_MAX - LAM_MIN) / dlam_um) + 1
    wavelengths = np.linspace(LAM_MIN, LAM_MAX, n_wavelength, dtype=np.float32)
    wavelengths_angstrom = wavelengths * 1e4
    wavelengths_jax = jnp.array(wavelengths)

    log(f"Wavelength grid: {LAM_MIN}-{LAM_MAX} um, "
        f"{dlam_angstroms} A spacing, {n_wavelength} samples")

    # -- Load catalog --------------------------------------------------------
    log("Loading star catalog...")
    t0 = time.time()
    star_catalog, template_files = load_star_catalog(catalog_dir)
    timings["load_catalog"] = time.time() - t0
    log(f"  {len(star_catalog['ra'])} sources in {timings['load_catalog']:.2f}s")

    # -- Load F158 bandpass --------------------------------------------------
    f158_band = refdata.get_f158_band()

    # -- Load all unique SED templates and precompute on wavelength grid -----
    log("Loading spectral templates...")
    t0 = time.time()
    all_unique_templates = np.unique(star_catalog["temp_idx"])
    templates_synphot = load_templates_as_synphot(
        catalog_dir, template_files, all_unique_templates,
    )
    template_grid = precompute_template_grid(
        templates_synphot, f158_band, wavelengths_angstrom,
    )
    timings["load_templates"] = time.time() - t0
    log(f"  {len(all_unique_templates)} templates in "
        f"{timings['load_templates']:.2f}s")

    # -- Optical model -------------------------------------------------------
    log("Loading optical model...")
    model = RomanOpticalModel(config_file=str(optical_model_path))

    # -- Per-SCA setup: payloads, PSFs, dispersers, sensitivity, JIT ---------
    log(f"Setting up {len(sca_list)} SCAs...")
    sca_data = {}  # sca -> {optical_payloads, sensitivities, fori_fns}

    for sca_num in sca_list:
        detector_name = f"WFI{sca_num:02d}"
        log(f"\n  SCA {sca_num} ({detector_name}):")

        # Optical payloads
        optical_payloads = {
            order: omj.make_sca_payload(model, sca=sca_num, order=order)
            for order in ORDERS
        }

        # Sensitivity curves
        sensitivities = load_sensitivities(
            sensitivity_dir, sca_num, wavelengths,
        )

        # PSF payloads
        psf_payloads = {}
        for psf_order in ["0", "1"]:
            psf_payloads[psf_order] = psf_model.get_or_make_psf_payload(
                detector=detector_name, order=psf_order,
                cache_dir=str(psf_cache_dir), verbose=False,
            )
        # STPSF only provides GRISM0 and GRISM1; reuse order 1 PSF for order 2
        psf_payloads["2"] = psf_payloads["1"]
        log(f"    PSF payloads loaded")

        # Star dispersers
        star_dispersers = {}
        for order in ORDERS:
            star_dispersers[order] = star_disperser.make_star_disperser(
                psf_payloads[order], optical_payloads[order],
            )

        # Build batched fori_loops (fixed batch_size for reuse)
        fori_fns = {}
        for order in ORDERS:
            fori_fns[order] = make_batched_fori(
                star_dispersers[order], sensitivities[order],
                wavelengths_jax, dlam_angstroms,
            )

        sca_data[sca_num] = {
            "optical_payloads": optical_payloads,
            "sensitivities": sensitivities,
            "fori_fns": fori_fns,
        }

    # -- JIT warmup: compile once per SCA/order ------------------------------
    log("\nJIT warmup (compiling all SCA/order dispersers)...")
    t0 = time.time()
    warmup_output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
    warmup_spec = jnp.zeros((batch_size, n_wavelength), dtype=jnp.float32)
    warmup_x = jnp.zeros(batch_size, dtype=jnp.float32)
    warmup_y = jnp.zeros(batch_size, dtype=jnp.float32)

    for sca_num in sca_list:
        for order in ORDERS:
            t1 = time.time()
            _ = sca_data[sca_num]["fori_fns"][order](
                1, warmup_spec, warmup_x, warmup_y, warmup_output,
            )
            _.block_until_ready()
            log(f"  SCA {sca_num} order {order}: {time.time() - t1:.1f}s")

    timings["jit_warmup"] = time.time() - t0
    log(f"  Total warmup: {timings['jit_warmup']:.1f}s")

    timings["setup_total"] = time.time() - t_total
    log(f"\nSetup complete in {timings['setup_total']:.1f}s")

    return {
        "model": model,
        "star_catalog": star_catalog,
        "template_files": template_files,
        "template_grid": template_grid,
        "wavelengths": wavelengths,
        "wavelengths_angstrom": wavelengths_angstrom,
        "wavelengths_jax": wavelengths_jax,
        "dlam_angstroms": dlam_angstroms,
        "sca_list": sca_list,
        "sca_data": sca_data,
        "batch_size": batch_size,
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
):
    """Process a single pointing: select sources, generate spectra, disperse.

    Parameters
    ----------
    pipeline : dict
        From ``setup_pipeline``.
    pointing_ra, pointing_dec, pointing_pa : float
        Telescope pointing in degrees.
    output_dir : str or Path
        Output directory for this pointing.  Created if needed;
        existing files are overwritten without warning.
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

    star_catalog = pipeline["star_catalog"]
    sca_list = pipeline["sca_list"]
    batch_size = pipeline["batch_size"]

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
    cone_mask = cone_search(
        star_catalog["ra"], star_catalog["dec"],
        pointing_ra, pointing_dec, cone_radius,
    )
    n_cone = int(cone_mask.sum())
    log(f"    {n_cone} sources within {cone_radius} deg "
        f"in {time.time() - t0:.2f}s")

    if n_cone == 0:
        log("    WARNING: No sources in cone.")

    ra_cone = star_catalog["ra"][cone_mask]
    dec_cone = star_catalog["dec"][cone_mask]
    mag_cone = star_catalog["mag"][cone_mask]
    tidx_cone = star_catalog["temp_idx"][cone_mask]

    # -- Step 2: Sky -> FPA --------------------------------------------------
    if n_cone > 0:
        log("  Sky -> FPA conversion...")
        t0 = time.time()
        xfpa, yfpa = omj.get_fpa_pos(
            jnp.array(ra_cone), jnp.array(dec_cone),
            pointing_ra, pointing_dec, pointing_pa,
        )
        log(f"    Done in {time.time() - t0:.2f}s")
    else:
        xfpa = jnp.array([], dtype=jnp.float32)
        yfpa = jnp.array([], dtype=jnp.float32)

    # -- Step 3: Process each SCA --------------------------------------------
    sca_outputs = {}
    sca_model_np = {}  # per-SCA MODEL numpy arrays for mosaic/PNG
    source_counts = {}  # per-SCA per-order source counts

    for sca_num in sca_list:
        log(f"\n  SCA {sca_num}:")
        t_sca = time.time()
        sd = pipeline["sca_data"][sca_num]

        # Select sources for this SCA (per order)
        order_masks, any_mask = select_sources_per_order(
            sd["optical_payloads"], xfpa, yfpa,
        )
        n_any = int(any_mask.sum())

        sca_counts = {}
        for order in ORDERS:
            n_ord = int(order_masks[order].sum())
            sca_counts[order] = n_ord
            log(f"    Order {order}: {n_ord} sources")
        source_counts[sca_num] = sca_counts

        # Disperse sources (or keep zeros if none on this SCA)
        output = jnp.zeros(
            (DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32,
        )

        if n_any > 0:
            # Generate spectra for sources on this SCA
            mag_sel = mag_cone[any_mask]
            tidx_sel = tidx_cone[any_mask]

            t0 = time.time()
            spectra_flam = generate_spectra(
                pipeline["template_grid"], tidx_sel, mag_sel,
            )
            log(f"    Spectra: {n_any} sources in "
                f"{time.time() - t0:.2f}s")

            # Get SCA coordinates (order "1" defines the undispersed position)
            xfpa_sel = xfpa[any_mask]
            yfpa_sel = yfpa[any_mask]
            xsca_all, ysca_all = omj.fpa_to_sca(
                sd["optical_payloads"]["1"], xfpa_sel, yfpa_sel,
            )
            xsca_all = np.asarray(xsca_all)
            ysca_all = np.asarray(ysca_all)

            # Disperse per order with batching
            order_masks_sel = {
                order: order_masks[order][any_mask] for order in ORDERS
            }

            for order in ORDERS:
                omask = order_masks_sel[order]
                n_order = int(omask.sum())
                if n_order == 0:
                    continue

                x_ord = xsca_all[omask]
                y_ord = ysca_all[omask]
                spec_ord = spectra_flam[omask]

                t_order = time.time()
                output = disperse_batched(
                    sd["fori_fns"][order], spec_ord, x_ord, y_ord,
                    output, batch_size,
                )
                elapsed = time.time() - t_order
                ms_per = elapsed / n_order * 1e3
                log(f"    Order {order}: dispersed {n_order} in {elapsed:.2f}s "
                    f"({ms_per:.1f} ms/star)")
        else:
            log(f"    No sources on detector.")

        sca_outputs[sca_num] = output

        # Poisson sample on GPU (output is counts/s, multiply by exptime)
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

        # Single GPU->CPU transfer for both arrays
        t0 = time.time()
        output_np = np.array(output)
        isim_np = np.array(isim)
        t_transfer = time.time() - t0
        sca_model_np[sca_num] = output_np

        # Write per-SCA outputs (PNG from noiseless MODEL for clean visualization)
        stem = f"{prefix}_detSCA{sca_num:02d}"
        fits_path = str(output_dir / f"{stem}.fits")
        png_path = str(output_dir / f"{stem}.png")
        t0 = time.time()
        write_fits(output_np, isim_np, fits_path,
                   pointing_ra, pointing_dec, pointing_pa, sca_num,
                   exptime, key_data, seed)
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

    # -- Metadata YAML -------------------------------------------------------
    pointing_key_data = jax.random.key_data(pointing_key).tolist() \
        if pointing_key is not None else None
    meta = {
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
        "batch_size": pipeline["batch_size"],
        "source_counts": {
            f"SCA{sca_num}": counts
            for sca_num, counts in sorted(source_counts.items())
        },
    }
    meta_path = output_dir / f"{prefix}_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    log(f"\n  Pointing complete in {time.time() - t_total:.1f}s")

    return sca_outputs


# ---------------------------------------------------------------------------
# Single-SCA convenience wrapper (backward-compatible)
# ---------------------------------------------------------------------------

def build_star_grism_image(
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
    dlam_angstroms=2.0,
    batch_size=1000,
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
    dlam_angstroms : float
        Wavelength spacing in Angstroms (default: 2.0).
    batch_size : int
        Sources per JIT batch (default: 1000).
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
        dlam_angstroms=dlam_angstroms,
        batch_size=batch_size,
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

    # Move from tmp layout to single-file output
    tmp_prefix = f"grism_{tmp_dir.name}"
    tmp_fits = tmp_dir / f"{tmp_prefix}_detSCA{sca:02d}.fits"
    tmp_png = tmp_dir / f"{tmp_prefix}_detSCA{sca:02d}.png"
    if tmp_fits.exists():
        tmp_fits.rename(output_file)
    if tmp_png.exists():
        tmp_png.rename(output_file.with_suffix(".png"))

    # Clean up temp dir
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    return sca_outputs[sca]


# ---------------------------------------------------------------------------
# Batch mode: YAML config
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = """\
# Star grism image builder configuration
#
# This file defines one or more telescope pointings and the set of
# SCAs (detectors) to simulate for each.
#
# Usage:
#   pixi run -e cuda python scripts/build_star_grism_image.py --config this_file.yaml

# ── Output ──────────────────────────────────────────────────────────────────
# Top-level output directory.  Each pointing creates a subdirectory
# containing per-SCA FITS/PNG files and a focal-plane mosaic PNG.
output_dir: /workspace/scratch/roman-star-fields

# ── RNG seed (required) ──────────────────────────────────────────────────
# Integer seed for reproducible Poisson noise.  Split deterministically
# into per-pointing and per-SCA keys.
seed: 42

# ── Exposure time ─────────────────────────────────────────────────────────
# Exposure time in seconds.  The noiseless model (counts/s) is multiplied
# by exptime before Poisson sampling.
exptime: 190.22

# ── Detectors ───────────────────────────────────────────────────────────────
# Which SCAs to simulate.  Use "all" for 1-18, or list specific numbers.
scas: all
# scas: [1, 5, 12]

# ── Pointings ──────────────────────────────────────────────────────────────
# Each entry becomes a subdirectory under output_dir.
pointings:
  - name: ra10_dec0_pa0
    ra: 10.0
    dec: 0.0
    pa: 0.0

  - name: ra10_dec0_pa10
    ra: 10.0
    dec: 0.0
    pa: 10.0

# ── Wavelength grid ────────────────────────────────────────────────────────
# Spacing in Angstroms.  2A gives ~5500 wavelength samples over 0.9-2.0 um.
dlam_angstroms: 2.0

# ── Source selection ───────────────────────────────────────────────────────
# Initial cone search radius around the pointing center (degrees).
# Sources outside this radius are excluded before per-SCA selection.
# 0.6 deg is sufficient for the full WFI field of view.
cone_radius: 0.6

# ── Batching ───────────────────────────────────────────────────────────────
# Number of sources processed per JIT-compiled batch.  The fori_loop is
# compiled once for this batch size and reused for all pointings/SCAs.
# Sources are processed in chunks of batch_size; the last chunk is
# zero-padded to maintain the compiled shape.  Larger values use more
# GPU memory (~22 KB per source); smaller values add loop overhead.
#
# Recommended: 1000 for stars, smaller for galaxies (larger per-source memory).
batch_size: 1000

# ── Data paths (optional, defaults shown) ──────────────────────────────────
# Uncomment to override:
# catalog_dir: data/stars
# sensitivity_dir: data/sensitivities
# optical_model: data/Roman_grism_OpticalModel_v0.8.yaml
# psf_cache_dir: data/psf_cache
"""


def run_batch(config_path, verbose=True, force=False):
    """Run the pipeline from a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    verbose : bool
        Print progress.
    force : bool
        Overwrite existing pointing directories (default: skip them).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def log(msg):
        if verbose:
            print(msg)

    # Parse SCA list
    scas = cfg.get("scas", "all")
    if scas == "all":
        sca_list = list(range(1, 19))
    else:
        sca_list = [int(s) for s in scas]

    # Parse seed and exposure time
    seed = cfg["seed"]
    exptime = cfg.get("exptime", 190.22)

    # Check which pointings need processing before expensive setup.
    # Track original index so key derivation is stable regardless of skips.
    output_dir = Path(cfg["output_dir"])
    base_key = jax.random.key(seed)
    all_pointing_keys = jax.random.split(base_key, len(cfg["pointings"]))

    pointings_todo = []  # list of (pointing_dict, pointing_key)
    for idx, pointing in enumerate(cfg["pointings"]):
        pointing_dir = output_dir / pointing["name"]
        if force or not pointing_dir.exists():
            pointings_todo.append((pointing, all_pointing_keys[idx]))

    log(f"Config: {config_path}")
    log(f"SCAs: {sca_list}")
    log(f"Seed: {seed}, Exptime: {exptime}s")
    log(f"Pointings: {len(cfg['pointings'])} total, "
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
        dlam_angstroms=cfg.get("dlam_angstroms", 2.0),
        batch_size=cfg.get("batch_size", 1000),
        verbose=verbose,
    )

    # Process pointings that need work
    cone_radius = cfg.get("cone_radius", 0.6)
    n_skipped = len(cfg["pointings"]) - len(pointings_todo)

    t_all = time.time()
    for i, (pointing, pointing_key) in enumerate(pointings_todo):
        name = pointing["name"]
        log(f"\n{'='*60}")
        log(f"Pointing {i+1}/{len(pointings_todo)}: {name}")
        log(f"{'='*60}")

        pointing_dir = output_dir / name
        process_pointing(
            pipeline,
            pointing["ra"], pointing["dec"], pointing["pa"],
            str(pointing_dir),
            cone_radius=cone_radius,
            exptime=exptime,
            pointing_key=pointing_key,
            seed=seed,
            verbose=verbose,
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
        description="Build simulated Roman grism images from a stellar catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --generate-config to create a documented template config file.",
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=str,
                      help="YAML config file for batch mode "
                           "(multiple pointings/SCAs)")
    mode.add_argument("--pointing-ra", type=float,
                      help="Pointing RA in degrees (quick mode)")
    mode.add_argument("--generate-config", type=str, metavar="FILE",
                      help="Write a documented template config file and exit")
    mode.add_argument("--mosaic", type=str, metavar="DIR",
                      help="Generate mosaic PNG from a pointing directory "
                           "containing grism_*_detSCA*.fits files")

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
                        help="Path to star catalog directory")
    parser.add_argument("--sensitivity-dir", type=str, default=None,
                        help="Path to sensitivity FITS files")
    parser.add_argument("--optical-model", type=str, default=None,
                        help="Path to optical model YAML")
    parser.add_argument("--psf-cache-dir", type=str, default=None,
                        help="Path to PSF cache directory")
    parser.add_argument("--cone-radius", type=float, default=0.6,
                        help="Cone search radius in degrees (default: 0.6)")
    parser.add_argument("--dlam", type=float, default=2.0,
                        help="Wavelength spacing in Angstroms (default: 2.0)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Sources per JIT batch (default: 1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (required for quick mode)")
    parser.add_argument("--exptime", type=float, default=190.22,
                        help="Exposure time in seconds (default: 190.22)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directories/files")

    args = parser.parse_args()

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

    # Batch mode
    if args.config:
        run_batch(args.config, verbose=not args.quiet, force=args.force)
        return

    # Quick mode — validate required arguments
    if args.pointing_dec is None or args.pointing_pa is None:
        parser.error("--pointing-dec and --pointing-pa required in quick mode")
    if args.sca is None:
        parser.error("--sca required in quick mode")
    if args.output is None:
        parser.error("--output required in quick mode")
    if args.seed is None:
        parser.error("--seed required in quick mode")

    build_star_grism_image(
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
        dlam_angstroms=args.dlam,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        force=args.force,
    )


if __name__ == "__main__":
    main()

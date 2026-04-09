"""Shared pipeline utilities for Roman grism simulation scripts.

Extracted from ``build_star_grism_image.py`` so that both the star-only and
unified simulation scripts share the same I/O, batching, and source-selection
helpers.

Not added to ``__init__.py`` — scripts import directly::

    from roman_disperser.pipeline import write_fits, load_sensitivities, ...
"""

import os
import re
from pathlib import Path

import yaml
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import AsinhNorm
from PIL import Image

from roman_disperser import catalog
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
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(catalog_dir=None, sensitivity_dir=None,
                  optical_model_path=None, psf_cache_dir=None):
    """Resolve default paths relative to project root.

    Parameters
    ----------
    catalog_dir : str or Path, optional
        Default: ``data/catalogs``.
    sensitivity_dir : str or Path, optional
        Default: ``data/sensitivities``.
    optical_model_path : str or Path, optional
        Default: ``data/Roman_grism_OpticalModel_v0.8.yaml``.
    psf_cache_dir : str or Path, optional
        Default: ``data/psf_cache``.
    """
    project_root = Path(os.environ.get("PIXI_PROJECT_ROOT", "."))
    if catalog_dir is None:
        catalog_dir = project_root / "data" / "catalogs"
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
# Star batching
# ---------------------------------------------------------------------------

def make_batched_star_fori(disperser_fn, sens, wavelengths_jax, dlam_angstroms):
    """Build a JIT-compiled fori_loop that processes a fixed-size batch of stars.

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


def disperse_batched_stars(fori_fn, spectra, xsca, ysca, output, batch_size):
    """Disperse sources in fixed-size batches, reusing compiled code.

    Parameters
    ----------
    fori_fn : compiled fori_loop from make_batched_star_fori
    spectra : ndarray [N, N_wl]
    xsca, ysca : ndarray [N]
    output : jnp.ndarray [4088, 4088]
    batch_size : int

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
# Galaxy batching
# ---------------------------------------------------------------------------

def make_batched_galaxy_fori(disperser_fn, sens, wavelengths_jax,
                             dlam_angstroms):
    """Build a JIT-compiled fori_loop for galaxy dispersion.

    Like the star version but the loop body also indexes into a galaxy
    images array.

    Parameters
    ----------
    disperser_fn : callable from make_galaxy_disperser
        Signature: (image, x0, y0, spectrum, wavelengths, output) -> output
    sens : jnp.ndarray [N_wl]
    wavelengths_jax : jnp.ndarray [N_wl]
    dlam_angstroms : float

    Returns
    -------
    run : callable(n_sources, spectra, xsca, ysca, images, output) -> output
        spectra: [batch_size, N_wl], xsca/ysca: [batch_size],
        images: [batch_size, npix, npix]
    """
    @jax.jit
    def run(n_sources, spectra, xsca, ysca, images, output):
        def body_fn(i, output):
            counts = spectra[i] * sens * dlam_angstroms
            return disperser_fn(images[i], xsca[i], ysca[i],
                                counts, wavelengths_jax, output)
        return jax.lax.fori_loop(0, n_sources, body_fn, output)

    return run


def disperse_batched_galaxies(fori_fn, spectra, xsca, ysca, images, output,
                              batch_size):
    """Disperse galaxies in fixed-size batches, reusing compiled code.

    Parameters
    ----------
    fori_fn : compiled fori_loop from make_batched_galaxy_fori
    spectra : ndarray [N, N_wl]
    xsca, ysca : ndarray [N]
    images : jnp.ndarray [N, npix, npix]
        Galaxy images (already on GPU).
    output : jnp.ndarray [4088, 4088]
    batch_size : int

    Returns
    -------
    output : jnp.ndarray [4088, 4088]
    """
    n_sources = len(xsca)
    n_wl = spectra.shape[1]
    npix = images.shape[1]
    n_batches = (n_sources + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_sources)
        n_actual = end - start

        # Pad spectra and positions (CPU -> GPU)
        spec_batch = np.zeros((batch_size, n_wl), dtype=np.float32)
        x_batch = np.zeros(batch_size, dtype=np.float32)
        y_batch = np.zeros(batch_size, dtype=np.float32)
        spec_batch[:n_actual] = spectra[start:end]
        x_batch[:n_actual] = xsca[start:end]
        y_batch[:n_actual] = ysca[start:end]

        # Pad images (already on GPU, slice + pad)
        img_batch = jnp.zeros((batch_size, npix, npix), dtype=jnp.float32)
        img_batch = img_batch.at[:n_actual].set(images[start:end])

        output = fori_fn(
            n_actual,
            jnp.array(spec_batch),
            jnp.array(x_batch),
            jnp.array(y_batch),
            img_batch,
            output,
        )

    output.block_until_ready()
    return output


# ---------------------------------------------------------------------------
# Output: FITS, PNG, Mosaic
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
    # Downsample for quicklook: 4088 -> 1022 pixels (4x block-average).
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
            # Downsample for thumbnails: block-average by 8x -> 511x511
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

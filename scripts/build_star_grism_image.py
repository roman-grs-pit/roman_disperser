#!/usr/bin/env python
"""
Build a simulated Roman grism image from a stellar catalog.

This script reads a stellar catalog, selects sources visible on a given detector,
loads spectral templates, normalizes them to catalog magnitudes, and disperses
all sources through the grism optical model to produce a FITS image and PNG
quicklook.

Can be used as a CLI script or imported as a module:

    # CLI usage
    pixi run python scripts/build_star_grism_image.py \
        --pointing-ra 9.5 --pointing-dec 0.95 --pointing-pa 0.0 \
        --sca 5 --output my_field.fits

    # Module usage
    from scripts.build_star_grism_image import build_star_grism_image
    build_star_grism_image(
        pointing_ra=9.5, pointing_dec=0.95, pointing_pa=0.0,
        sca=5, output_file="my_field.fits",
    )
"""

import argparse
import os
import time
from pathlib import Path

import yaml
import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import synphot as syn
import stsynphot as stsyn
from astropy.io import fits
from matplotlib.colors import AsinhNorm

from roman_disperser import catalog, psf_model, star_disperser
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
    temp_inds = raw_template_index - 58 * (raw_template_index // 58)

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


def generate_spectra(
    templates_synphot, template_indices, magnitudes,
    wavelengths_angstrom, f158_band,
):
    """Generate normalized spectra for all sources.

    Each source's template is normalized to its catalog F158 magnitude,
    then sampled onto the output wavelength grid.

    Parameters
    ----------
    templates_synphot : dict mapping int -> synphot.SourceSpectrum
    template_indices : ndarray [N] of int
    magnitudes : ndarray [N] of float
    wavelengths_angstrom : ndarray [N_wl] in Angstroms
    f158_band : synphot.SpectralElement

    Returns
    -------
    spectra_flam : ndarray [N, N_wl] in FLAM units
    """
    n_sources = len(magnitudes)
    n_wl = len(wavelengths_angstrom)
    spectra = np.zeros((n_sources, n_wl), dtype=np.float32)
    wl_qty = wavelengths_angstrom * u.AA

    for i in range(n_sources):
        sp = templates_synphot[int(template_indices[i])]
        norm_sp = sp.normalize(magnitudes[i] * u.ABmag, band=f158_band)
        spectra[i] = norm_sp(wl_qty, flux_unit=syn.units.FLAM).value

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
    masks = {}
    any_mask = np.zeros(len(xfpa), dtype=bool)
    for order in orders:
        mask = catalog.select_sources(optical_payloads[order], xfpa, yfpa)
        masks[order] = np.asarray(mask)
        any_mask |= masks[order]
    return masks, any_mask


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

def make_fori_dispatcher(disperser_fn, sensitivities_order, wavelengths_jax,
                         dlam_angstroms, spectra_jax, xsca_jax, ysca_jax):
    """Build a JIT-compiled fori_loop dispatcher for one order.

    The source count n_sources is a dynamic argument so JAX compiles
    the loop body once and reuses it for any count.

    Parameters
    ----------
    disperser_fn : callable from make_star_disperser
    sensitivities_order : jnp.ndarray [N_wl]
    wavelengths_jax : jnp.ndarray [N_wl]
    dlam_angstroms : float
    spectra_jax : jnp.ndarray [N_sources, N_wl] in FLAM
    xsca_jax, ysca_jax : jnp.ndarray [N_sources]

    Returns
    -------
    run : callable(n_sources, output) -> output
    """
    sens = sensitivities_order

    @jax.jit
    def run(n_sources, output):
        def body_fn(i, output):
            counts = spectra_jax[i] * sens * dlam_angstroms
            return disperser_fn(xsca_jax[i], ysca_jax[i],
                                wavelengths_jax, counts, output)
        return jax.lax.fori_loop(0, n_sources, body_fn, output)

    return run


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_fits(output_array, output_file, pointing_ra, pointing_dec,
               pointing_pa, sca):
    """Write the grism image to a FITS file.

    Primary HDU contains pointing metadata. MODEL extension contains the
    counts array.
    """
    primary = fits.PrimaryHDU()
    primary.header["WFICENRA"] = (pointing_ra, "Pointing RA [deg]")
    primary.header["WFICENDEC"] = (pointing_dec, "Pointing Dec [deg]")
    primary.header["WFICENPA"] = (pointing_pa, "Position angle [deg]")
    primary.header["DETNUM"] = (sca, "SCA number")

    model_hdu = fits.ImageHDU(data=np.array(output_array), name="MODEL")

    hdul = fits.HDUList([primary, model_hdu])
    hdul.writeto(output_file, overwrite=True)


def write_png(output_array, png_file, linear_width=0.01):
    """Write an asinh-stretched quicklook PNG."""
    output_np = np.array(output_array)
    fig, ax = plt.subplots(figsize=(10, 10))
    norm = AsinhNorm(linear_width=linear_width, vmin=0, vmax=output_np.max())
    ax.imshow(output_np, origin="lower", cmap="inferno", norm=norm)
    ax.set_xlabel("X (SCA pixels)")
    ax.set_ylabel("Y (SCA pixels)")
    ax.set_title(f"Grism Image (asinh stretch)")
    plt.colorbar(
        ax.images[0], ax=ax, label="Counts (asinh stretch)", shrink=0.8,
    )
    plt.tight_layout()
    fig.savefig(png_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_star_grism_image(
    pointing_ra,
    pointing_dec,
    pointing_pa,
    sca,
    output_file,
    *,
    catalog_dir=None,
    sensitivity_dir=None,
    optical_model_path=None,
    psf_cache_dir=None,
    cone_radius=0.6,
    dlam_angstroms=2.0,
    verbose=True,
):
    """Build a simulated grism image from a stellar catalog.

    Parameters
    ----------
    pointing_ra : float
        Telescope pointing RA in degrees.
    pointing_dec : float
        Telescope pointing Dec in degrees.
    pointing_pa : float
        Telescope position angle in degrees.
    sca : int
        SCA (detector) number, 1-18.
    output_file : str
        Output FITS filename. A PNG with the same stem is also produced.
    catalog_dir : str, optional
        Path to star catalog directory (default: data/stars).
    sensitivity_dir : str, optional
        Path to sensitivity files (default: data/sensitivities).
    optical_model_path : str, optional
        Path to optical model YAML (default: data/Roman_grism_OpticalModel_v0.8.yaml).
    psf_cache_dir : str, optional
        Path to PSF cache directory (default: data/psf_cache).
    cone_radius : float
        Initial cone search radius in degrees (default: 0.6).
    dlam_angstroms : float
        Wavelength spacing in Angstroms (default: 2.0).
    verbose : bool
        Print progress information (default: True).

    Returns
    -------
    output : jnp.ndarray [4088, 4088]
        The accumulated counts image.
    """
    timings = {}
    t_total = time.time()

    def log(msg):
        if verbose:
            print(msg)

    # -- Resolve default paths relative to project root ----------------------
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

    detector_name = f"WFI{sca:02d}"

    # -----------------------------------------------------------------------
    # Step 1: Read the base star catalog
    # -----------------------------------------------------------------------
    log("Step 1: Loading star catalog...")
    t0 = time.time()
    star_catalog, template_files = load_star_catalog(catalog_dir)
    n_total = len(star_catalog["ra"])
    timings["load_catalog"] = time.time() - t0
    log(f"  Loaded {n_total} stars in {timings['load_catalog']:.2f}s")

    # -----------------------------------------------------------------------
    # Step 2: Cone search to remove distant sources
    # -----------------------------------------------------------------------
    log(f"Step 2: Cone search (radius={cone_radius} deg)...")
    t0 = time.time()
    cone_mask = cone_search(
        star_catalog["ra"], star_catalog["dec"],
        pointing_ra, pointing_dec, cone_radius,
    )
    n_cone = int(cone_mask.sum())
    timings["cone_search"] = time.time() - t0
    log(f"  {n_cone} sources within {cone_radius} deg "
        f"(removed {n_total - n_cone}) in {timings['cone_search']:.2f}s")

    if n_cone == 0:
        log("  WARNING: No sources found within cone radius. "
            "Writing empty image.")
        output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
        write_fits(output, output_file, pointing_ra, pointing_dec,
                   pointing_pa, sca)
        png_file = str(Path(output_file).with_suffix(".png"))
        write_png(output, png_file)
        return output

    # Trim catalog to cone
    ra_cone = star_catalog["ra"][cone_mask]
    dec_cone = star_catalog["dec"][cone_mask]
    mag_cone = star_catalog["mag"][cone_mask]
    tidx_cone = star_catalog["temp_idx"][cone_mask]

    # -----------------------------------------------------------------------
    # Step 3: Convert sky positions to FPA coordinates
    # -----------------------------------------------------------------------
    log("Step 3: Converting to FPA coordinates...")
    t0 = time.time()
    xfpa, yfpa = omj.get_fpa_pos(
        jnp.array(ra_cone), jnp.array(dec_cone),
        pointing_ra, pointing_dec, pointing_pa,
    )
    timings["sky_to_fpa"] = time.time() - t0
    log(f"  Done in {timings['sky_to_fpa']:.2f}s")

    # -----------------------------------------------------------------------
    # Step 4: Determine which objects land on the detector (per order)
    # -----------------------------------------------------------------------
    log("Step 4: Selecting sources per order...")
    t0 = time.time()

    model = RomanOpticalModel(config_file=str(optical_model_path))
    optical_payloads = {
        order: omj.make_sca_payload(model, sca=sca, order=order)
        for order in ORDERS
    }

    order_masks, any_mask = select_sources_per_order(optical_payloads, xfpa, yfpa)
    n_any = int(any_mask.sum())
    for order in ORDERS:
        log(f"  Order {order}: {int(order_masks[order].sum())} sources on detector")
    log(f"  Total unique sources (any order): {n_any}")
    timings["select_sources"] = time.time() - t0
    log(f"  Done in {timings['select_sources']:.2f}s")

    if n_any == 0:
        log("  WARNING: No sources land on detector. Writing empty image.")
        output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
        write_fits(output, output_file, pointing_ra, pointing_dec,
                   pointing_pa, sca)
        png_file = str(Path(output_file).with_suffix(".png"))
        write_png(output, png_file)
        return output

    # Trim to sources on any order
    mag_sel = mag_cone[any_mask]
    tidx_sel = tidx_cone[any_mask]
    xfpa_sel = xfpa[any_mask]
    yfpa_sel = yfpa[any_mask]

    # Per-order masks relative to the trimmed array
    order_masks_sel = {}
    for order in ORDERS:
        order_masks_sel[order] = order_masks[order][any_mask]

    # -----------------------------------------------------------------------
    # Step 5: Generate spectra from templates
    # -----------------------------------------------------------------------
    log("Step 5: Generating spectra...")
    t0 = time.time()

    # Wavelength grid
    dlam_um = dlam_angstroms / 1e4
    n_wavelength = int((LAM_MAX - LAM_MIN) / dlam_um) + 1
    wavelengths = np.linspace(LAM_MIN, LAM_MAX, n_wavelength, dtype=np.float32)
    wavelengths_angstrom = wavelengths * 1e4
    wavelengths_jax = jnp.array(wavelengths)

    log(f"  Wavelength grid: {LAM_MIN}-{LAM_MAX} um, "
        f"{dlam_angstroms} A spacing, {n_wavelength} samples")

    # Load unique templates via synphot
    unique_template_indices = np.unique(tidx_sel)
    log(f"  Loading {len(unique_template_indices)} unique spectral templates...")
    templates_synphot = load_templates_as_synphot(
        catalog_dir, template_files, unique_template_indices,
    )

    # F158 bandpass for normalization
    f158_band = stsyn.band("roman, wfi, f158")

    # Generate all spectra
    log(f"  Normalizing {n_any} spectra to F158 magnitudes...")
    spectra_flam = generate_spectra(
        templates_synphot, tidx_sel, mag_sel, wavelengths_angstrom, f158_band,
    )
    spectra_jax = jnp.array(spectra_flam)

    # Load sensitivity curves
    sensitivities = load_sensitivities(sensitivity_dir, sca, wavelengths)

    timings["generate_spectra"] = time.time() - t0
    log(f"  Done in {timings['generate_spectra']:.2f}s")

    # -----------------------------------------------------------------------
    # Step 6: Disperse all sources per order
    # -----------------------------------------------------------------------
    log("Step 6: Dispersing sources...")
    t0 = time.time()

    # Load PSF payloads
    log("  Loading PSF payloads...")
    psf_payloads = {}
    for psf_order in ["0", "1"]:
        psf_payloads[psf_order] = psf_model.get_or_make_psf_payload(
            detector=detector_name, order=psf_order,
            cache_dir=str(psf_cache_dir), verbose=verbose,
        )
    psf_payloads["2"] = psf_payloads["1"]  # Order 2 reuses order 1 PSFs

    # Create star dispersers
    star_dispersers = {}
    for order in ORDERS:
        star_dispersers[order] = star_disperser.make_star_disperser(
            psf_payloads[order], optical_payloads[order],
        )

    # For each order, we need SCA coords and the per-order source mask.
    # Convert FPA -> SCA for the selected sources.
    xsca_all, ysca_all = omj.fpa_to_sca(
        optical_payloads["1"], xfpa_sel, yfpa_sel,
    )

    output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)

    # JIT warmup: compile the fori_loop with n=1 for each order
    log("  JIT warmup (compiling dispersers)...")
    for order in ORDERS:
        n_order = int(order_masks_sel[order].sum())
        if n_order == 0:
            continue

        # Get sources for this order
        omask = order_masks_sel[order]
        x_ord = jnp.array(np.asarray(xsca_all)[omask])
        y_ord = jnp.array(np.asarray(ysca_all)[omask])
        spec_ord = jnp.array(np.asarray(spectra_jax)[omask])

        fori_fn = make_fori_dispatcher(
            star_dispersers[order], sensitivities[order],
            wavelengths_jax, dlam_angstroms, spec_ord, x_ord, y_ord,
        )
        t_warmup = time.time()
        _ = fori_fn(1, output)
        _.block_until_ready()
        log(f"    Order {order}: compiled in {time.time() - t_warmup:.1f}s "
            f"({n_order} sources to disperse)")

        # Now disperse all sources for this order
        t_order = time.time()
        output = fori_fn(n_order, output)
        output.block_until_ready()
        elapsed = time.time() - t_order
        ms_per_star = elapsed / n_order * 1e3
        timings[f"disperse_order_{order}"] = elapsed
        log(f"    Order {order}: {n_order} sources in {elapsed:.2f}s "
            f"({ms_per_star:.1f} ms/star)")

    timings["disperse_total"] = time.time() - t0
    log(f"  Total dispersion: {timings['disperse_total']:.2f}s")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    log("Writing outputs...")
    t0 = time.time()

    write_fits(output, output_file, pointing_ra, pointing_dec, pointing_pa, sca)
    log(f"  FITS: {output_file}")

    png_file = str(Path(output_file).with_suffix(".png"))
    write_png(output, png_file)
    log(f"  PNG:  {png_file}")

    timings["write_outputs"] = time.time() - t0
    timings["total"] = time.time() - t_total

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("Timing Summary")
    log("=" * 60)
    log(f"  Load catalog:       {timings['load_catalog']:.2f}s")
    log(f"  Cone search:        {timings['cone_search']:.2f}s")
    log(f"  Sky -> FPA:         {timings['sky_to_fpa']:.2f}s")
    log(f"  Select sources:     {timings['select_sources']:.2f}s")
    log(f"  Generate spectra:   {timings['generate_spectra']:.2f}s")
    log(f"  Dispersion total:   {timings['disperse_total']:.2f}s")
    for order in ORDERS:
        key = f"disperse_order_{order}"
        if key in timings:
            log(f"    Order {order}:         {timings[key]:.2f}s")
    log(f"  Write outputs:      {timings['write_outputs']:.2f}s")
    log(f"  ─────────────────────────────")
    log(f"  TOTAL:              {timings['total']:.2f}s")
    log("=" * 60)

    output_np = np.array(output)
    log(f"\nImage statistics:")
    log(f"  Total flux:   {output_np.sum():.4e}")
    log(f"  Peak value:   {output_np.max():.4e}")
    log(f"  Non-zero px:  {(output_np > 0).sum():,} "
        f"({100 * (output_np > 0).sum() / output_np.size:.1f}%)")

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a simulated Roman grism image from a stellar catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--pointing-ra", type=float, required=True,
                        help="Pointing RA in degrees")
    parser.add_argument("--pointing-dec", type=float, required=True,
                        help="Pointing Dec in degrees")
    parser.add_argument("--pointing-pa", type=float, required=True,
                        help="Position angle in degrees")
    parser.add_argument("--sca", type=int, required=True,
                        help="SCA number (1-18)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output FITS filename")

    # Optional arguments
    parser.add_argument("--catalog-dir", type=str, default=None,
                        help="Path to star catalog directory")
    parser.add_argument("--sensitivity-dir", type=str, default=None,
                        help="Path to sensitivity FITS files")
    parser.add_argument("--optical-model", type=str, default=None,
                        help="Path to optical model YAML")
    parser.add_argument("--psf-cache-dir", type=str, default=None,
                        help="Path to PSF cache directory")
    parser.add_argument("--cone-radius", type=float, default=0.6,
                        help="Cone search radius in degrees")
    parser.add_argument("--dlam", type=float, default=2.0,
                        help="Wavelength spacing in Angstroms")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    build_star_grism_image(
        pointing_ra=args.pointing_ra,
        pointing_dec=args.pointing_dec,
        pointing_pa=args.pointing_pa,
        sca=args.sca,
        output_file=args.output,
        catalog_dir=args.catalog_dir,
        sensitivity_dir=args.sensitivity_dir,
        optical_model_path=args.optical_model,
        psf_cache_dir=args.psf_cache_dir,
        cone_radius=args.cone_radius,
        dlam_angstroms=args.dlam,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

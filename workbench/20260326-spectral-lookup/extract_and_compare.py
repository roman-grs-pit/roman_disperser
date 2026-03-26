#!/usr/bin/env python
"""
Extract spectra from simulated grism images and compare with input SEDs.

Reads the ECSV files produced by lookup_spectra.py, loads the pipeline
FITS MODEL image, extracts a 1D spectrum using our own optical model
trace, and plots: 2D spectral cutout (top) + 1D extracted vs expected
spectrum (bottom).

Usage:
    pixi run python workbench/20260326-spectral-lookup/extract_and_compare.py \
        --pointing-dir ~/data/Roman/grism-sims/output/ra10_dec0_pa0

    # Custom aperture
    pixi run python workbench/20260326-spectral-lookup/extract_and_compare.py \
        --pointing-dir ~/data/Roman/grism-sims/output/ra10_dec0_pa0 \
        --aperture 16
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.table import Table

from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj


# ============================================================================
# CONSTANTS
# ============================================================================

OPTICAL_MODEL_PATH = "data/Roman_grism_OpticalModel_v0.8.yaml"
ORDER = "1"
LAM_MIN, LAM_MAX = 0.9, 2.0  # microns


# ============================================================================
# HELPERS
# ============================================================================


def trace_y_at_wavelength(payload, xsca, ysca, wavelength):
    """Scalar trace: returns y_sca as a function of scalar wavelength."""
    xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)
    wl = jnp.atleast_1d(wavelength)
    xfpa_arr = jnp.broadcast_to(xfpa, wl.shape)
    yfpa_arr = jnp.broadcast_to(yfpa, wl.shape)
    xmpa, ympa = omj.trace_beam(payload, xfpa_arr, yfpa_arr, wl)
    _, ty = omj.mpa_to_sca(payload, xmpa, ympa)
    return ty[0]


def compute_trace(payload, xsca, ysca, wavelengths_um):
    """Compute the spectral trace positions on the detector.

    Returns trace_x, trace_y arrays in SCA coordinates (1-indexed).
    """
    xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)
    xmpa, ympa = omj.trace_beam(
        payload,
        jnp.broadcast_to(xfpa, (len(wavelengths_um),)),
        jnp.broadcast_to(yfpa, (len(wavelengths_um),)),
        jnp.array(wavelengths_um),
    )
    trace_x, trace_y = omj.mpa_to_sca(payload, xmpa, ympa)
    return np.array(trace_x).ravel(), np.array(trace_y).ravel()


def compute_dy_dlam(payload, xsca, ysca, wavelengths_um):
    """Compute dy/dlambda (pixels per micron) along the trace via autodiff."""
    dy_dlam_fn = jax.grad(
        lambda wl: trace_y_at_wavelength(payload, xsca, ysca, wl)
    )
    dy_dlam_jit = jax.jit(jax.vmap(dy_dlam_fn))
    return np.array(dy_dlam_jit(jnp.array(wavelengths_um)))


def extract_spectrum(image, payload, xsca, ysca, wavelengths_um, aperture=12):
    """Extract a 1D spectrum by summing cross-dispersion pixels along the trace.

    Dispersion is along y; extraction sums along x at each trace y position.
    Scales by |dy/dlambda| * dlam to convert from counts-per-pixel-row to
    counts-per-wavelength-bin.

    Returns
    -------
    extracted : ndarray [N_wl] — counts/s per wavelength bin
    trace_x, trace_y : ndarray [N_wl] — trace positions (SCA 1-indexed)
    dy_dlam : ndarray [N_wl] — dispersion in pixels per micron
    """
    trace_x, trace_y = compute_trace(payload, xsca, ysca, wavelengths_um)
    dy_dlam = compute_dy_dlam(payload, xsca, ysca, wavelengths_um)

    dlam_um = wavelengths_um[1] - wavelengths_um[0]

    raw = np.zeros(len(wavelengths_um))
    for i in range(len(wavelengths_um)):
        ix = int(round(trace_x[i])) - 1  # 1-indexed -> 0-indexed
        iy = int(round(trace_y[i])) - 1
        if ix < 0 or ix >= image.shape[1] or iy < 0 or iy >= image.shape[0]:
            continue
        x0 = max(0, ix - aperture)
        x1 = min(image.shape[1], ix + aperture + 1)
        raw[i] = image[iy, x0:x1].sum()

    # Scale: each pixel row spans 1/|dy/dlam| microns in wavelength.
    # Each wavelength bin spans dlam_um microns.
    # Fraction of pixel belonging to one bin = |dy/dlam| * dlam_um.
    extracted = raw * np.abs(dy_dlam) * dlam_um

    return extracted, trace_x, trace_y, dy_dlam


def make_cutout_2d(image, trace_x, trace_y, wavelengths_um, aperture=12):
    """Extract a 2D spectral cutout aligned with the trace.

    Returns a 2D array [2*aperture+1, N_wl] where each column is the
    cross-dispersion slice at that wavelength, and the wavelength array
    for pixels that are on-detector.
    """
    n_wl = len(wavelengths_um)
    width = 2 * aperture + 1
    cutout = np.zeros((width, n_wl))

    for i in range(n_wl):
        ix = int(round(trace_x[i])) - 1
        iy = int(round(trace_y[i])) - 1
        if ix < 0 or ix >= image.shape[1] or iy < 0 or iy >= image.shape[0]:
            continue
        x0_img = max(0, ix - aperture)
        x1_img = min(image.shape[1], ix + aperture + 1)
        # Map into cutout array
        x0_cut = x0_img - (ix - aperture)
        x1_cut = x0_cut + (x1_img - x0_img)
        cutout[x0_cut:x1_cut, i] = image[iy, x0_img:x1_img]

    return cutout


def plot_extraction(cutout_2d, wavelengths_ang, extracted, expected,
                    meta, aperture, output_path):
    """Plot 2D cutout (top) and 1D extracted vs expected spectrum (bottom)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    fig, (ax_img, ax_spec) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # --- Title ---
    title_parts = [
        f"RA={meta['ra']:.5f}, Dec={meta['dec']:.5f}  |  "
        f"F158={meta['F158']:.2f} AB, type={meta['type']}",
    ]
    if meta["type"] == "SER":
        title_parts[0] += f", z={meta['z_obs']:.4f}"
    title_parts.append(
        f"SCA{meta['sensitivity_sca']:02d} order {ORDER}, "
        f"aperture=±{aperture} pix"
    )
    fig.suptitle("\n".join(title_parts), fontsize=10, ha="left", x=0.12)

    # --- 2D cutout ---
    vmax = np.percentile(cutout_2d[cutout_2d > 0], 99) if (cutout_2d > 0).any() else 1.0
    norm = AsinhNorm(linear_width=vmax * 0.01, vmin=0, vmax=vmax)
    extent = [wavelengths_ang[0], wavelengths_ang[-1],
              -aperture - 0.5, aperture + 0.5]
    ax_img.imshow(cutout_2d, origin="lower", aspect="auto", cmap="inferno",
                  norm=norm, extent=extent)
    ax_img.set_ylabel("Cross-disp [pix]")

    # --- 1D spectra ---
    ax_spec.plot(wavelengths_ang, extracted, linewidth=0.7, color="C0",
                 label="Extracted (MODEL)", alpha=0.8)
    ax_spec.plot(wavelengths_ang, expected, linewidth=0.7, color="C1",
                 label="Expected (SED × sens × Δλ)", alpha=0.8)
    ax_spec.set_xlabel("Wavelength [Å]")
    ax_spec.set_ylabel("counts/s per bin")
    ax_spec.set_xlim(9000, 20000)
    ax_spec.legend(fontsize=9)

    # Set y-axis scale from the expected continuum level so bright
    # contamination and emission spikes don't dominate the scale.
    # Use median of positive values as continuum estimate.
    pos = expected[expected > 0]
    if len(pos) > 0:
        continuum = np.median(pos)
        ax_spec.set_ylim(-0.2 * continuum, 3.0 * continuum)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
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
        "--pointing-dir", type=str, required=True,
        help="Pipeline output directory for one pointing",
    )
    parser.add_argument(
        "--aperture", type=int, default=12,
        help="Cross-dispersion half-aperture in pixels (default: 12)",
    )
    parser.add_argument(
        "--ecsv-dir", type=str, default=None,
        help="Directory with ECSV files from lookup_spectra.py (default: script dir)",
    )
    args = parser.parse_args()

    pointing_dir = Path(args.pointing_dir)
    ecsv_dir = Path(args.ecsv_dir) if args.ecsv_dir else Path(__file__).parent
    aperture = args.aperture

    # --- Find ECSV files ---
    ecsv_files = sorted(ecsv_dir.glob("source_*.ecsv"))
    if not ecsv_files:
        raise FileNotFoundError(f"No source_*.ecsv files in {ecsv_dir}")
    print(f"Found {len(ecsv_files)} ECSV file(s)")

    # --- Load optical model ---
    model = RomanOpticalModel(OPTICAL_MODEL_PATH)

    # --- Process each source ---
    # Cache FITS images and payloads per SCA
    fits_cache = {}
    payload_cache = {}

    for ecsv_path in ecsv_files:
        t = Table.read(ecsv_path)
        meta = t.meta
        sca = meta["sensitivity_sca"]
        cat_idx = meta["catalog_index"]
        print(f"\n{ecsv_path.name}: catalog_index={cat_idx}, SCA{sca:02d}")

        # Load FITS MODEL for this SCA
        if sca not in fits_cache:
            fits_pattern = f"*_detSCA{sca:02d}.fits"
            fits_files = list(pointing_dir.glob(fits_pattern))
            if not fits_files:
                print(f"  WARNING: no FITS file for SCA{sca:02d}, skipping")
                continue
            with fits.open(fits_files[0]) as hdul:
                fits_cache[sca] = hdul["MODEL"].data.copy()
            print(f"  Loaded {fits_files[0].name}")

        if sca not in payload_cache:
            payload_cache[sca] = omj.make_sca_payload(model, sca=sca, order=ORDER)

        image = fits_cache[sca]
        payload = payload_cache[sca]

        # Source position from manifest (stored in ECSV appearances)
        # Parse xsca, ysca from the order-1 appearance
        xsca, ysca = None, None
        for appearance in meta["appearances"].split("; "):
            if f"order={ORDER}" in appearance and f"SCA{sca:02d}" in appearance:
                parts = appearance.split()
                for p in parts:
                    if p.startswith("x="):
                        xsca = float(p[2:])
                    elif p.startswith("y="):
                        ysca = float(p[2:])
                break

        if xsca is None or ysca is None:
            print(f"  WARNING: no order-{ORDER} SCA{sca:02d} appearance, skipping")
            continue

        print(f"  Position: x={xsca:.1f}, y={ysca:.1f}")

        # Build wavelength grid (same as pipeline)
        dlam_um = meta["dlam_angstroms"] / 1e4
        wavelengths_um = np.arange(LAM_MIN, LAM_MAX + dlam_um / 2, dlam_um)
        wavelengths_ang = wavelengths_um * 1e4

        # Extract spectrum
        extracted, trace_x, trace_y, dy_dlam = extract_spectrum(
            image, payload, xsca, ysca, wavelengths_um, aperture=aperture,
        )
        disp_mean = np.mean(np.abs(dy_dlam))
        print(f"  Dispersion: {disp_mean:.1f} pixels/μm "
              f"({disp_mean * (wavelengths_um[1] - wavelengths_um[0]):.3f} pixels/bin)")

        # Expected spectrum: input SED counts/s (from ECSV)
        ecsv_wl = np.array(t["wavelength"])  # Angstroms
        ecsv_cts = np.array(t["counts_per_s"])
        expected = np.interp(wavelengths_ang, ecsv_wl, ecsv_cts)

        # 2D cutout
        cutout_2d = make_cutout_2d(image, trace_x, trace_y, wavelengths_um,
                                   aperture=aperture)

        # Plot
        stem = ecsv_path.stem  # e.g., source_01_cat42936
        out_png = ecsv_dir / f"{stem}_extracted.png"
        plot_extraction(cutout_2d, wavelengths_ang, extracted, expected,
                        meta, aperture, out_png)

    print("\nDone.")


if __name__ == "__main__":
    main()

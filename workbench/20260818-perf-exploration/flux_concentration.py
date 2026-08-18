"""Flux concentration of the deposited (PSF-convolved) galaxy stamp.

Quantifies the grizli-report observation that the disperser "disperses even
zero pixels": for a production-geometry convolved stamp (120 px oversampled
Sérsic convolved with the 184 px oversampled PSF -> 303^2), what fraction of
the stamp's pixels carries what fraction of its flux?

For each of a few galaxy sizes and PSF wavelengths, sorts |pixel| descending
and reports the pixel count needed to reach cumulative flux fractions.
The complement (1 - fraction) is the flux error a top-k / thresholded deposit
would introduce PER SOURCE, to compare against the equivalence tolerance
(rel_sum ~1e-8 per SCA image; per-pixel rtol 1e-5).

CPU-friendly (a handful of FFT convolutions). Run:
    pixi run python workbench/20260818-perf-exploration/flux_concentration.py
"""

import numpy as np
import jax.numpy as jnp
import jax.scipy.signal

from roman_disperser import elements, psf_model, sersic
from roman_disperser.pipeline import resolve_paths

FRACTIONS = [0.9, 0.99, 0.999, 0.9999, 0.99999]


def main():
    element = elements.get_element(None)
    _, _, _, psf_cache_dir = resolve_paths()
    payload = psf_model.get_or_make_psf_payload(
        detector="WFI01", order="1",
        wavelengths=elements.psf_cache_wavelengths(element),
        stpsf_filter=element.stpsf_filters["1"],
        cache_dir=psf_cache_dir, verbose=False)
    oversample = int(payload["oversample"])
    npix_os = 30 * oversample

    psfs = psf_model.interpolate_psf_spatial(payload, 2044.0, 2044.0)
    grid_wl = np.array(payload["wavelengths"])
    wl_idx = [5, len(grid_wl) // 2, len(grid_wl) - 6]

    print(f"conv stamp: {npix_os + psfs.shape[-1] - 1}^2 oversampled px "
          f"({(npix_os + psfs.shape[-1] - 1)**2} deposits per wavelength)")
    header = "r_eff_arcsec  wl_um  " + "  ".join(f"px@{f}" for f in FRACTIONS)
    print(header)

    for r_arcsec in [0.1, 0.25, 0.5, 1.0]:
        r_pix = sersic.catalog_r_eff_to_pixels(
            jnp.array([r_arcsec], dtype=jnp.float32), oversample=oversample)
        img = sersic.make_sersic_images(
            r_pix, jnp.array([2.0]), jnp.array([0.7]), jnp.array([0.5]),
            npix_os)[0]
        for wi in wl_idx:
            conv = np.array(jax.scipy.signal.fftconvolve(
                img, psfs[wi], mode="full"))
            flat = np.abs(conv.ravel())
            order_desc = np.sort(flat)[::-1]
            csum = np.cumsum(order_desc) / flat.sum()
            counts = [int(np.searchsorted(csum, f) + 1) for f in FRACTIONS]
            print(f"{r_arcsec:12.2f}  {grid_wl[wi]:.2f}  "
                  + "  ".join(f"{c:7d}" for c in counts))


if __name__ == "__main__":
    main()

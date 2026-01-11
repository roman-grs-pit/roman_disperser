"""
Utility functions for disperser demonstration notebooks.

This module provides helper functions for:
- Generating galaxy profiles (exponential, Sersic)
- Creating synthetic spectra
- Visualization of dispersed results
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def make_exponential_galaxy(
    npix: int,
    half_light_radius_arcsec: float,
    pixel_scale_arcsec: float = 0.11,
    oversample: int = 3,
    normalize: bool = True,
) -> jnp.ndarray:
    """
    Create a 2D exponential (Sersic n=1) galaxy profile.

    The exponential profile has surface brightness:
        I(r) = I0 * exp(-r / r_scale)
    where r_scale = half_light_radius / 1.678 for n=1 Sersic.

    Args:
        npix: Number of native pixels on each side (output will be npix*oversample)
        half_light_radius_arcsec: Half-light radius in arcseconds
        pixel_scale_arcsec: Native pixel scale in arcsec/pixel (default: 0.11 for Roman)
        oversample: Oversampling factor (default: 3)
        normalize: If True, normalize to total flux = 1.0

    Returns:
        image: [npix*oversample, npix*oversample] galaxy image

    Example:
        >>> # Create 50×50 native pixel galaxy, 3× oversampled
        >>> galaxy = make_exponential_galaxy(50, half_light_radius_arcsec=0.3, oversample=3)
        >>> galaxy.shape
        (150, 150)
        >>> float(galaxy.sum())  # Should be 1.0
        1.0
    """
    # Oversampled grid size
    n_oversample = npix * oversample

    # Effective pixel scale after oversampling
    effective_pixel_scale = pixel_scale_arcsec / oversample

    # Scale radius for exponential profile (n=1 Sersic)
    # For n=1: r_eff = 1.678 * r_scale
    r_scale_arcsec = half_light_radius_arcsec / 1.678
    r_scale_pix = r_scale_arcsec / effective_pixel_scale

    # Create coordinate grid centered at the middle of the array
    # For odd npix, center is at (npix-1)/2
    # For even npix, center is at npix/2 - 0.5
    center = (n_oversample - 1) / 2.0
    y, x = jnp.mgrid[0:n_oversample, 0:n_oversample]
    r = jnp.sqrt((x - center)**2 + (y - center)**2)

    # Exponential profile
    image = jnp.exp(-r / r_scale_pix)

    # Normalize to unit total flux if requested
    if normalize:
        image = image / image.sum()

    return image


def make_flat_spectrum(
    lam_min: float,
    lam_max: float,
    n_wavelength: int = 1000,
    flux_density: float = 1.0,
    normalize: bool = True,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Create a flat spectrum with uniform flux density.

    Args:
        lam_min: Minimum wavelength in microns
        lam_max: Maximum wavelength in microns
        n_wavelength: Number of wavelength samples
        flux_density: Flux per wavelength bin (before normalization)
        normalize: If True, normalize to total flux = 1.0

    Returns:
        spectrum: [n_wavelength] flux array
        lam0: Starting wavelength (same as lam_min)
        dlam: Wavelength spacing

    Example:
        >>> spec, lam0, dlam = make_flat_spectrum(1.0, 2.0, n_wavelength=1000)
        >>> spec.shape
        (1000,)
        >>> float(spec.sum())  # Should be 1.0
        1.0
        >>> lam0, dlam
        (1.0, 0.001001001001001001)
    """
    # Create uniform spectrum
    spectrum = jnp.ones(n_wavelength, dtype=jnp.float32) * flux_density

    # Normalize to unit total flux if requested
    if normalize:
        spectrum = spectrum / spectrum.sum()

    # Compute wavelength grid parameters
    lam0 = float(lam_min)
    dlam = float((lam_max - lam_min) / n_wavelength)

    return spectrum, lam0, dlam


def make_sloped_spectrum(
    lam_min: float,
    lam_max: float,
    n_wavelength: int = 1000,
    slope_min: float = 0.5,
    slope_max: float = 1.5,
    taper_fraction: float = 0.2,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Create a spectrum with linear slope and edge roll-off.

    This function creates a more realistic spectrum than make_flat_spectrum by:
    - Adding a linear slope across wavelength (typically increasing blue to red)
    - Smoothly rolling off to zero at the edges using a cosine taper
    - NOT normalizing (uses arbitrary flux units)

    The edge roll-off makes the wavelength extent clearly visible in dispersed
    images, avoiding visual artifacts where the spectrum extent is unclear.

    Args:
        lam_min: Minimum wavelength in microns
        lam_max: Maximum wavelength in microns
        n_wavelength: Number of wavelength samples
        slope_min: Flux at blue edge (before taper)
        slope_max: Flux at red edge (before taper)
        taper_fraction: Fraction of wavelength range to taper on each edge (0-0.5)

    Returns:
        spectrum: [n_wavelength] flux array (NOT normalized)
        lam0: Starting wavelength (same as lam_min)
        dlam: Wavelength spacing

    Example:
        >>> # Create realistic increasing spectrum with edge roll-off
        >>> spec, lam0, dlam = make_sloped_spectrum(
        ...     lam_min=1.0, lam_max=2.0, n_wavelength=1000,
        ...     slope_min=0.5, slope_max=1.5, taper_fraction=0.2
        ... )
        >>> spec.shape
        (1000,)
        >>> float(spec.max())  # Should be ~1.5 at red end (before taper)
        ~1.5
    """
    # Linear slope increasing from blue to red
    slope = jnp.linspace(slope_min, slope_max, n_wavelength, dtype=jnp.float32)

    # Edge roll-off using cosine taper
    taper_width = int(taper_fraction * n_wavelength)
    taper = jnp.ones(n_wavelength, dtype=jnp.float32)

    # Left edge (blue) taper: smoothly rise from 0 to 1
    left_taper = 0.5 * (1 - jnp.cos(jnp.pi * jnp.arange(taper_width) / taper_width))
    taper = taper.at[:taper_width].set(left_taper)

    # Right edge (red) taper: smoothly fall from 1 to 0
    right_taper = 0.5 * (1 + jnp.cos(jnp.pi * jnp.arange(taper_width) / taper_width))
    taper = taper.at[-taper_width:].set(right_taper)

    # Combined spectrum: slope × taper (NOT normalized)
    spectrum = slope * taper

    # Compute wavelength grid parameters
    lam0 = float(lam_min)
    dlam = float((lam_max - lam_min) / n_wavelength)

    return spectrum, lam0, dlam


def compute_flux_conservation(
    input_image: jnp.ndarray,
    input_spectrum: jnp.ndarray,
    output_image: jnp.ndarray,
) -> dict:
    """
    Compute flux conservation metrics for dispersed galaxy.

    Args:
        input_image: [Ny, Nx] input spatial image
        input_spectrum: [Nlam] input spectrum
        output_image: [H, W] dispersed output image

    Returns:
        metrics: dict with keys:
            - input_flux: total input flux (image.sum() * spectrum.sum())
            - output_flux: total output flux (output.sum())
            - conservation_fraction: output / input
            - conservation_percent: 100 * output / input
    """
    input_flux = float(jnp.sum(input_image) * jnp.sum(input_spectrum))
    output_flux = float(jnp.sum(output_image))

    conservation_fraction = output_flux / input_flux if input_flux > 0 else 0.0

    return {
        'input_flux': input_flux,
        'output_flux': output_flux,
        'conservation_fraction': conservation_fraction,
        'conservation_percent': 100.0 * conservation_fraction,
    }


def get_dispersed_extent(
    output: jnp.ndarray,
    threshold_fraction: float = 0.001,
) -> Tuple[int, int, int, int]:
    """
    Find the bounding box containing significant dispersed flux.

    Args:
        output: [H, W] dispersed output image
        threshold_fraction: Fraction of peak flux to use as threshold

    Returns:
        y_min, y_max, x_min, x_max: Bounding box indices (inclusive)

    Example:
        >>> y_min, y_max, x_min, x_max = get_dispersed_extent(output)
        >>> zoomed = output[y_min:y_max+1, x_min:x_max+1]
    """
    # Find pixels above threshold
    peak_flux = float(jnp.max(output))
    threshold = peak_flux * threshold_fraction
    mask = output > threshold

    # Find bounding box
    y_indices, x_indices = jnp.where(mask)

    if len(y_indices) == 0:
        # No pixels above threshold - return full image
        return 0, output.shape[0] - 1, 0, output.shape[1] - 1

    y_min = int(jnp.min(y_indices))
    y_max = int(jnp.max(y_indices))
    x_min = int(jnp.min(x_indices))
    x_max = int(jnp.max(x_indices))

    return y_min, y_max, x_min, x_max


def make_random_galaxy_positions(
    n_galaxies: int,
    x_range: Tuple[float, float] = (500.0, 3500.0),
    y_range: Tuple[float, float] = (500.0, 3500.0),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random galaxy CENTER positions within SCA bounds.

    Note: These are galaxy center positions, NOT the image box corner
    positions expected by the disperser. Use center_to_corner() to
    convert to the format needed by disperse_2d1d_sca.

    Args:
        n_galaxies: Number of galaxies to generate
        x_range: (x_min, x_max) SCA x-coordinate range
        y_range: (y_min, y_max) SCA y-coordinate range
        seed: Random seed for reproducibility

    Returns:
        x_centers: [n_galaxies] x-coordinates of galaxy centers
        y_centers: [n_galaxies] y-coordinates of galaxy centers

    Example:
        >>> x_centers, y_centers = make_random_galaxy_positions(10, seed=42)
        >>> len(x_centers), len(y_centers)
        (10, 10)
    """
    rng = np.random.default_rng(seed)

    x_centers = rng.uniform(x_range[0], x_range[1], size=n_galaxies)
    y_centers = rng.uniform(y_range[0], y_range[1], size=n_galaxies)

    return x_centers, y_centers


def center_to_corner(x_center, y_center, npix_x, npix_y, dx, dy):
    """
    Convert source center position to image box corner position.

    The disperser expects the position of pixel [0,0] of the input image
    (its center), not the source center. This function computes the corner
    pixel position from the source center position.

    Note: In this coordinate system, (x0, y0) is the CENTER of pixel [0,0],
    not its edge. The formula works correctly for both even and odd image
    dimensions:
    - Even N: source center is between two pixels (fractional index)
    - Odd N: source center is exactly at the middle pixel

    Args:
        x_center, y_center: Desired source center in SCA coordinates
            (can be scalars or arrays for batch processing)
        npix_x, npix_y: Image dimensions in pixels (can be even or odd).
            Will be converted to int if passed as float.
        dx, dy: Pixel spacing (1/oversample for oversampled images)

    Returns:
        x0, y0: Position of pixel [0,0] center for disperser input

    Example:
        For a 150×150 pixel image centered at (2044, 2044) with dx=dy=1/3:

        >>> x0, y0 = center_to_corner(2044.0, 2044.0, 150, 150, 1/3, 1/3)
        >>> x0  # ≈ 2019.17 (position of first pixel's center)
        2019.1666666666665
    """
    # Convert pixel counts to int in case floats are passed
    npix_x = int(npix_x)
    npix_y = int(npix_y)

    x0 = x_center - (npix_x - 1) / 2 * dx
    y0 = y_center - (npix_y - 1) / 2 * dy
    return x0, y0

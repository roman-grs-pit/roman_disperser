"""
JAX-based Sérsic profile generator for galaxy morphology images.

Generates 2D Sérsic profiles suitable for the galaxy disperser pipeline.
All images are normalized to sum=1; flux scaling comes from the spectrum.

The position angle convention matches astropy's Sersic2D: theta is measured
counter-clockwise from the +x axis in radians.
"""

import numpy as np
from scipy.special import gammaincinv

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Precomputed b_n table (computed once at import time via scipy)
# ---------------------------------------------------------------------------

_BN_N_GRID = np.linspace(0.01, 10.0, 2000).astype(np.float32)
_BN_VALUES = gammaincinv(2.0 * _BN_N_GRID, 0.5).astype(np.float32)

# Convert to JAX arrays for use in jnp.interp
_BN_N_GRID_JNP = jnp.array(_BN_N_GRID)
_BN_VALUES_JNP = jnp.array(_BN_VALUES)


def compute_bn(n):
    """Look up Sérsic b_n via interpolation on precomputed table.

    JIT-compatible (uses jnp.interp on static arrays).
    Works for scalar or array n.

    Parameters
    ----------
    n : float or jnp.ndarray
        Sérsic index. Valid range: [0.01, 10.0].

    Returns
    -------
    b_n : same shape as n
    """
    return jnp.interp(n, _BN_N_GRID_JNP, _BN_VALUES_JNP)


def _sersic_kernel(r_eff_pix, n, ba, theta, y_grid, x_grid):
    """Core Sérsic profile computation (no grid creation).

    Parameters
    ----------
    r_eff_pix : float
        Half-light radius in (oversampled) pixels.
    n : float
        Sérsic index.
    ba : float
        Minor-to-major axis ratio (0-1). 1 = circular.
    theta : float
        Position angle in radians, CCW from +x axis.
    y_grid, x_grid : jnp.ndarray [npix, npix]
        Coordinate grids centered at (0, 0).

    Returns
    -------
    image : jnp.ndarray [npix, npix], normalized to sum=1
    """
    # Clamp to avoid numerical issues
    n = jnp.clip(n, 0.01, 10.0)
    ba = jnp.clip(ba, 0.01, 1.0)
    r_eff_pix = jnp.clip(r_eff_pix, 0.1, None)

    # Rotate coordinates (matches astropy Sersic2D convention)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_rot = x_grid * cos_t + y_grid * sin_t
    y_rot = -x_grid * sin_t + y_grid * cos_t

    # Elliptical radius
    r = jnp.sqrt(x_rot**2 + (y_rot / ba) ** 2)

    # Sérsic profile
    bn = compute_bn(n)
    profile = jnp.exp(-bn * ((r / r_eff_pix) ** (1.0 / n) - 1.0))

    # Normalize to sum=1
    return profile / profile.sum()


def make_sersic_image(r_eff_pix, n, ba, theta, npix):
    """Generate a single normalized Sérsic profile image.

    Parameters
    ----------
    r_eff_pix : float
        Half-light radius in (oversampled) pixels.
    n : float
        Sérsic index.
    ba : float
        Minor-to-major axis ratio (0-1). 1 = circular.
    theta : float
        Position angle in radians, CCW from +x axis.
    npix : int
        Image size (npix × npix). Static for JIT.

    Returns
    -------
    image : jnp.ndarray [npix, npix], normalized to sum=1
    """
    center = (npix - 1) / 2.0
    y_grid, x_grid = jnp.mgrid[0:npix, 0:npix] - center
    return _sersic_kernel(r_eff_pix, n, ba, theta, y_grid, x_grid)


def make_sersic_images(r_eff_pix, n, ba, theta, npix):
    """Batch-generate Sérsic images via jax.vmap.

    Precomputes the coordinate grid once, then vmaps over galaxy
    parameters with the grid as a broadcast (non-batched) input.

    Parameters
    ----------
    r_eff_pix : jnp.ndarray [N]
    n : jnp.ndarray [N]
    ba : jnp.ndarray [N]
    theta : jnp.ndarray [N]
    npix : int (static)

    Returns
    -------
    images : jnp.ndarray [N, npix, npix], each normalized to sum=1
    """
    center = (npix - 1) / 2.0
    y_grid, x_grid = jnp.mgrid[0:npix, 0:npix] - center

    # vmap over (r_eff_pix, n, ba, theta), broadcast (y_grid, x_grid)
    batched = jax.vmap(_sersic_kernel, in_axes=(0, 0, 0, 0, None, None))
    return batched(r_eff_pix, n, ba, theta, y_grid, x_grid)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def catalog_r_eff_to_pixels(r_eff_arcsec, pixel_scale=0.11, oversample=4):
    """Convert half-light radius from arcsec to oversampled pixels.

    Parameters
    ----------
    r_eff_arcsec : float or array
        Half-light radius in arcseconds.
    pixel_scale : float
        Native pixel scale in arcsec/pixel (default 0.11 for Roman WFI).
    oversample : int
        Oversampling factor.

    Returns
    -------
    r_eff_pix : same type as input
        Half-light radius in oversampled pixels.
    """
    return r_eff_arcsec / (pixel_scale / oversample)


def sky_pa_to_sca_theta(pa_sky_deg, pointing_pa_deg):
    """Convert sky PA (E of N, degrees) to SCA theta (CCW from +x, radians).

    Accounts for telescope pointing PA and WFI detector orientation.

    The transformation is:
        theta_sca = deg2rad(pa_sky - pointing_pa + 150)

    The +150° comes from: -(V3IdlYAngle=60°) - 90° (N-to-x conversion)
    with sign flip from the sky→SCA reflection (E-is-left on sky but
    the FPA sign conventions flip one axis).

    Parameters
    ----------
    pa_sky_deg : float or array
        Position angle on sky in degrees, East of North.
    pointing_pa_deg : float
        Telescope pointing position angle in degrees.

    Returns
    -------
    theta_sca : float or array
        Position angle in SCA pixel coordinates, in radians,
        measured CCW from +x axis.
    """
    return jnp.deg2rad(pa_sky_deg - pointing_pa_deg + 150.0)

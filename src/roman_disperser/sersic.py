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

# Kept as NumPy: creating JAX arrays here would initialise the JAX backend
# at package import (on a GPU node: preallocating GPU memory), before any
# worker processes are started. jnp.interp converts them when traced.

# Cubic spline of b_n on the same 2000 nodes (float64 fit, float32
# per-interval coefficients). Linear interpolation is exact at the nodes but
# its derivative db_n/dn jumps at every node (C0); the spline is C2, so
# gradients in the Sérsic index are smooth (for HMC/NUTS or Newton fits).
# The nodes are uniform, so the interval index is arithmetic (no search).
_BN_LO, _BN_HI, _BN_N = 0.01, 10.0, 2000
_BN_STEP = (_BN_HI - _BN_LO) / (_BN_N - 1)


def _bn_spline_coeffs():
    from scipy.interpolate import CubicSpline
    x = np.linspace(_BN_LO, _BN_HI, _BN_N)
    c = CubicSpline(x, gammaincinv(2.0 * x, 0.5)).c   # [4, N-1], highest first
    return np.ascontiguousarray(c.T).astype(np.float32)  # [N-1, 4]


_BN_SPLINE = _bn_spline_coeffs()

BN_METHODS = ("spline", "linear")


def compute_bn(n, method="spline"):
    """Sérsic b_n (``gammaincinv(2n, 0.5)``) from a precomputed table.

    JIT-compatible; scalar or array ``n`` in [0.01, 10] (clamped outside).

    Parameters
    ----------
    n : float or jnp.ndarray
        Sérsic index. Valid range: [0.01, 10.0].
    method : {"spline", "linear"}
        ``"spline"`` (default): cubic spline through 2000 nodes, C2 in n,
        max error ~1e-7 relative for n >= 0.1. ``"linear"``: the earlier
        linear interpolation between the same nodes (C0: the gradient in
        n is piecewise constant).

    Returns
    -------
    b_n : same shape as n
    """
    if method == "linear":
        return jnp.interp(n, _BN_N_GRID, _BN_VALUES)
    if method != "spline":
        raise ValueError(f"method must be one of {BN_METHODS}, got {method!r}")
    n = jnp.asarray(n)
    s = (jnp.clip(n, _BN_LO, _BN_HI) - _BN_LO) / _BN_STEP
    i = jnp.clip(jnp.floor(s), 0, _BN_N - 2)
    dt = (s - i) * _BN_STEP
    c = jnp.asarray(_BN_SPLINE, n.dtype if jnp.issubdtype(n.dtype, jnp.floating)
                    else jnp.float32)[i.astype(jnp.int32)]
    return ((c[..., 0] * dt + c[..., 1]) * dt + c[..., 2]) * dt + c[..., 3]


def _sersic_kernel(r_eff_pix, n, ba, theta, y_grid, x_grid,
                   bn_method="spline"):
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
    r2 = x_rot**2 + (y_rot / ba) ** 2
    # The r = 0 pixel (odd npix) has value exp(bn) and zero gradient; sqrt
    # and (r / r_eff)**(1/n) have infinite or NaN derivatives there, so they
    # see a dummy value at that pixel (double-where). Other pixels are
    # computed exactly as before.
    nonzero = r2 > 0
    r = jnp.sqrt(jnp.where(nonzero, r2, 1.0))

    # Sérsic profile
    bn = compute_bn(n, bn_method)
    scaled = jnp.where(nonzero, (r / r_eff_pix) ** (1.0 / n), 0.0)
    profile = jnp.exp(-bn * (scaled - 1.0))

    # Normalize to sum=1
    return profile / profile.sum()


def _grid(npix, subsample=1):
    """Coordinate grids [npix*s, npix*s] centred at (0, 0): the centres of
    an s x s subgrid in every pixel (s = 1: the pixel centres)."""
    s = int(subsample)
    if s < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}")
    c = (jnp.arange(npix * s) + 0.5) / s - 0.5 - (npix - 1) / 2.0
    y_grid, x_grid = jnp.meshgrid(c, c, indexing="ij")
    return y_grid, x_grid


def _bin(image, npix, subsample):
    s = int(subsample)
    if s == 1:
        return image
    return image.reshape(npix, s, npix, s).sum(axis=(1, 3))


def make_sersic_image(r_eff_pix, n, ba, theta, npix, subsample=1):
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
    subsample : int, optional
        Evaluate the profile on an s x s subgrid per pixel and sum
        (static; default 1 = pixel-centre sampling, the production
        behaviour). Pixel-centre sampling aliases compact profiles: for
        ``r_eff`` of a few pixels or large ``n`` the image and its gradient
        in ``r_eff`` and ``n`` follow the sampling, not the integral.
        ``subsample`` of 3-5 is advised when fitting such sources.

    Returns
    -------
    image : jnp.ndarray [npix, npix], normalized to sum=1
    """
    y_grid, x_grid = _grid(npix, subsample)
    img = _sersic_kernel(r_eff_pix, n, ba, theta, y_grid, x_grid)
    return _bin(img, npix, subsample)


def make_sersic_images(r_eff_pix, n, ba, theta, npix, subsample=1):
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
    subsample : int (static), optional
        As in :func:`make_sersic_image`.

    Returns
    -------
    images : jnp.ndarray [N, npix, npix], each normalized to sum=1
    """
    y_grid, x_grid = _grid(npix, subsample)

    # vmap over (r_eff_pix, n, ba, theta), broadcast (y_grid, x_grid)
    batched = jax.vmap(_sersic_kernel, in_axes=(0, 0, 0, 0, None, None))
    imgs = batched(r_eff_pix, n, ba, theta, y_grid, x_grid)
    return jax.vmap(lambda im: _bin(im, npix, subsample))(imgs)


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

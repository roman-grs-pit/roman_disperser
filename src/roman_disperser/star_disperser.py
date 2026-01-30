"""
Star dispersion module for Roman grism simulation.

This module provides functions to disperse point sources (stars) through the
Roman grism using wavelength-dependent PSFs. It integrates the PSF model with
the optical model to simulate realistic star spectra on the detector.

Key functions:
- make_psf_pixel_grid: Create coordinate grid for PSF deposition
- deposit_psf: Deposit a single PSF onto the detector
- disperse_star_psf: Disperse a star using wavelength-dependent PSFs
- make_star_disperser: Factory for JIT-compiled star dispersion

All wavelength parameters are in **microns** (consistent with the optical model).

See docs/star_dispersion.md for the design document.
"""

import jax
import jax.numpy as jnp

from . import optical_model_jax as omj
from . import psf_model


def make_psf_pixel_grid(psf_shape, oversample):
    """
    Create relative coordinate grid for PSF deposition.

    For even oversampling (e.g., 4×), the PSF center lies at the cross-hairs
    of the 4 central pixels, not at a pixel center. This function computes
    the relative detector pixel offsets for each PSF pixel.

    Parameters
    ----------
    psf_shape : tuple
        (PSF_y, PSF_x) oversampled PSF size
    oversample : int
        PSF oversampling factor (e.g., 4)

    Returns
    -------
    rel_y : jnp.ndarray
        Relative y-offsets in detector pixel units, shape [PSF_y, PSF_x]
    rel_x : jnp.ndarray
        Relative x-offsets in detector pixel units, shape [PSF_y, PSF_x]

    Notes
    -----
    For a 4× oversampled PSF:
    - Each detector pixel contains 4×4 = 16 PSF pixels
    - PSF pixel (i, j) maps to detector offset:
        rel_x[i,j] = (j - (PSF_x - 1) / 2) / oversample
        rel_y[i,j] = (i - (PSF_y - 1) / 2) / oversample
    - At deposition time, add star position: det_x = xsca + rel_x

    Examples
    --------
    >>> rel_y, rel_x = make_psf_pixel_grid((182, 182), 4)
    >>> rel_x.shape
    (182, 182)
    >>> # Center of PSF maps to (0, 0) relative offset
    >>> rel_x[90, 90]  # Near center
    DeviceArray(-0.125, dtype=float32)
    """
    PSF_y, PSF_x = psf_shape

    # Create index grids
    i_grid = jnp.arange(PSF_y)  # Row indices (y)
    j_grid = jnp.arange(PSF_x)  # Column indices (x)
    i, j = jnp.meshgrid(i_grid, j_grid, indexing='ij')

    # Compute relative offsets from PSF center
    # PSF center is at ((PSF_y - 1) / 2, (PSF_x - 1) / 2) in PSF pixel coords
    rel_y = (i - (PSF_y - 1) / 2) / oversample
    rel_x = (j - (PSF_x - 1) / 2) / oversample

    return rel_y.astype(jnp.float32), rel_x.astype(jnp.float32)


def deposit_psf(output, xsca, ysca, psf, rel_x, rel_y):
    """
    Deposit a PSF onto the detector at the specified position.

    Uses direct pixel assignment: each oversampled PSF pixel deposits
    into exactly one detector pixel, determined by flooring the position.

    Parameters
    ----------
    output : jnp.ndarray
        [H, W] detector accumulator array
    xsca : float
        Center x-position in SCA coordinates (1-indexed FITS)
    ysca : float
        Center y-position in SCA coordinates (1-indexed FITS)
    psf : jnp.ndarray
        [PSF_y, PSF_x] oversampled PSF (already flux-scaled)
    rel_x : jnp.ndarray
        [PSF_y, PSF_x] relative x-offsets from make_psf_pixel_grid()
    rel_y : jnp.ndarray
        [PSF_y, PSF_x] relative y-offsets from make_psf_pixel_grid()

    Returns
    -------
    output : jnp.ndarray
        [H, W] updated detector array with PSF deposited

    Notes
    -----
    - Uses JAX's at[].add() with mode="drop" for out-of-bounds handling
    - wrap_negative_indices=False prevents wraparound at array edges
    - Each PSF pixel lands in exactly one detector pixel (no bilinear)
    - FITS 1-indexed: pixel n has center at n.0, convert to 0-indexed array
    """
    # Compute absolute detector coordinates
    det_x = xsca + rel_x  # [PSF_y, PSF_x]
    det_y = ysca + rel_y  # [PSF_y, PSF_x]

    # Convert FITS 1-indexed to 0-indexed array indices
    # FITS pixel n has center at n.0, spans [n-0.5, n+0.5]
    # Array index i corresponds to FITS pixel i+1
    idx_x = jnp.floor(det_x - 0.5).astype(jnp.int32)
    idx_y = jnp.floor(det_y - 0.5).astype(jnp.int32)

    # Flatten for scatter-add
    idx_x_flat = idx_x.ravel()
    idx_y_flat = idx_y.ravel()
    psf_flat = psf.ravel()

    # Deposit with mode="drop" for out-of-bounds, no negative index wrapping
    output = output.at[idx_y_flat, idx_x_flat].add(
        psf_flat, mode="drop", wrap_negative_indices=False
    )

    return output


def disperse_star_psf(
    psf_payload,
    optical_payload,
    xsca_star,
    ysca_star,
    wavelengths,
    star_flux,
    output,
    rel_x=None,
    rel_y=None,
):
    """
    Disperse a single star through the Roman grism using wavelength-dependent PSFs.

    This function deposits PSFs along the spectral trace for each wavelength,
    simulating how a point source appears after grism dispersion.

    Parameters
    ----------
    psf_payload : dict
        PSF payload from psf_model.get_or_make_psf_payload()
    optical_payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    xsca_star : float
        Star x-position in SCA coordinates (1-indexed FITS)
    ysca_star : float
        Star y-position in SCA coordinates (1-indexed FITS)
    wavelengths : jnp.ndarray
        [N_wl] wavelengths in **microns**
    star_flux : jnp.ndarray
        [N_wl] flux per wavelength (arbitrary units)
    output : jnp.ndarray
        [H, W] detector accumulator array (typically 4088×4088)
    rel_x : jnp.ndarray, optional
        Pre-computed relative x-offsets from make_psf_pixel_grid()
        If None, computed internally (less efficient for repeated calls)
    rel_y : jnp.ndarray, optional
        Pre-computed relative y-offsets from make_psf_pixel_grid()

    Returns
    -------
    output : jnp.ndarray
        [H, W] updated detector array with dispersed star

    Notes
    -----
    Algorithm:
    1. Get spatially-interpolated PSFs at undispersed position
    2. Interpolate PSFs to user wavelengths
    3. Compute all dispersed positions via optical model (vectorized)
    4. Scale PSFs by flux
    5. Deposit each PSF along the trace

    The undispersed position is used for PSF lookup (per STPSF requirements).
    Efficiency curves should be applied to star_flux before calling this function.

    Examples
    --------
    >>> # Setup payloads
    >>> psf_payload = psf_model.get_or_make_psf_payload(
    ...     detector='WFI05', order='1', cache_dir='data/psf_cache'
    ... )
    >>> optical_payload = omj.make_sca_payload(model, sca=5, order='1')
    >>>
    >>> # Disperse a star
    >>> wavelengths = jnp.linspace(1.0, 1.8, 100)  # microns
    >>> star_flux = jnp.ones(100)  # Flat spectrum
    >>> output = jnp.zeros((4088, 4088), dtype=jnp.float32)
    >>> output = disperse_star_psf(
    ...     psf_payload, optical_payload,
    ...     xsca_star=2000.0, ysca_star=2000.0,
    ...     wavelengths=wavelengths, star_flux=star_flux,
    ...     output=output
    ... )

    See Also
    --------
    make_star_disperser : Factory function for efficient repeated dispersion
    disperser.disperse_2d1d_sca : Extended source (galaxy) dispersion
    """
    # Compute relative coordinate grid if not provided
    if rel_x is None or rel_y is None:
        psf_shape = psf_payload['psf_grid'].shape[-2:]
        rel_y, rel_x = make_psf_pixel_grid(psf_shape, psf_payload['oversample'])

    # Step 1: Get PSFs at undispersed position for all grid wavelengths
    psfs_grid = psf_model.interpolate_psf_spatial(
        psf_payload, xsca_star, ysca_star
    )  # [N_wl_grid, PSF_y, PSF_x]

    # Step 2: Interpolate PSFs to user wavelengths
    psfs = psf_model.interpolate_psf_wavelength(
        psfs_grid, psf_payload['wavelengths'], wavelengths
    )  # [N_wl, PSF_y, PSF_x]

    # Step 3: Compute dispersed positions for all wavelengths (vectorized)
    # Convert star position to FPA (optical model functions expect arrays)
    xsca_arr = jnp.atleast_1d(xsca_star)
    ysca_arr = jnp.atleast_1d(ysca_star)
    xfpa, yfpa = omj.sca_to_fpa(optical_payload, xsca_arr, ysca_arr)

    # Trace beam at all wavelengths (wavelengths already in microns)
    xmpa, ympa = omj.trace_beam(optical_payload, xfpa, yfpa, wavelengths)

    # Convert back to SCA
    xsca_disp, ysca_disp = omj.mpa_to_sca(optical_payload, xmpa, ympa)
    # Shapes: [1, N_wl] each - squeeze out the single spatial dimension
    # but keep the wavelength dimension (use squeeze with axis to preserve 1D)
    xsca_disp = xsca_disp.reshape(-1)  # Flatten to [N_wl]
    ysca_disp = ysca_disp.reshape(-1)  # Flatten to [N_wl]

    # Step 4: Scale PSFs by flux
    scaled_psfs = psfs * star_flux[:, jnp.newaxis, jnp.newaxis]
    # Shape: [N_wl, PSF_y, PSF_x]

    # Step 5: Deposit all PSFs using fori_loop for JIT compatibility
    n_wl = len(wavelengths)

    def deposit_one_wavelength(i, output):
        return deposit_psf(
            output,
            xsca_disp[i],
            ysca_disp[i],
            scaled_psfs[i],
            rel_x,
            rel_y,
        )

    output = jax.lax.fori_loop(0, n_wl, deposit_one_wavelength, output)

    return output


def make_star_disperser(psf_payload, optical_payload):
    """
    Create a JIT-compiled star disperser for a specific detector/order.

    This factory function validates the PSF payload and returns a compiled
    function that can be called repeatedly for different stars without
    recompilation overhead.

    Parameters
    ----------
    psf_payload : dict
        PSF payload from psf_model.get_or_make_psf_payload()
        Must use even oversampling (e.g., 4×)
    optical_payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()

    Returns
    -------
    disperse_star : Callable
        JIT-compiled function with signature:
        (xsca, ysca, wavelengths, flux, output) -> output
        where wavelengths are in **microns**

    Raises
    ------
    ValueError
        If psf_payload uses odd oversampling

    Examples
    --------
    >>> # Setup payloads once
    >>> psf_payload = psf_model.get_or_make_psf_payload(
    ...     detector='WFI05', order='1', cache_dir='data/psf_cache'
    ... )
    >>> optical_payload = omj.make_sca_payload(model, sca=5, order='1')
    >>>
    >>> # Create JIT-compiled disperser
    >>> disperser = make_star_disperser(psf_payload, optical_payload)
    >>>
    >>> # Disperse many stars efficiently
    >>> wavelengths = jnp.linspace(1.0, 1.8, 100)  # microns
    >>> output = jnp.zeros((4088, 4088), dtype=jnp.float32)
    >>> for star in star_catalog:
    ...     output = disperser(star.x, star.y, wavelengths, star.flux, output)

    Notes
    -----
    - The PSF and optical payloads are captured in the closure
    - First call compiles the function (may take a few seconds)
    - Subsequent calls use the cached compiled function
    - Relative coordinate grid is pre-computed once
    """
    # Validate even oversampling
    oversample = psf_payload['oversample']
    if oversample % 2 != 0:
        raise ValueError(
            f"PSF payload must use even oversampling for correct PSF center "
            f"geometry, got {oversample}. Even oversampling (e.g., 2, 4) "
            f"places the PSF center at the cross-hairs of central pixels."
        )

    # Pre-compute relative coordinate grid (captured in closure)
    psf_shape = psf_payload['psf_grid'].shape[-2:]
    rel_y, rel_x = make_psf_pixel_grid(psf_shape, oversample)

    @jax.jit
    def disperse_star(xsca, ysca, wavelengths, flux, output):
        """
        Disperse a single star.

        Parameters
        ----------
        xsca, ysca : float
            Star position in SCA coordinates (1-indexed FITS)
        wavelengths : jnp.ndarray
            [N_wl] wavelengths in **microns**
        flux : jnp.ndarray
            [N_wl] flux per wavelength
        output : jnp.ndarray
            [H, W] detector accumulator

        Returns
        -------
        output : jnp.ndarray
            Updated detector array
        """
        return disperse_star_psf(
            psf_payload,
            optical_payload,
            xsca,
            ysca,
            wavelengths,
            flux,
            output,
            rel_x,
            rel_y,
        )

    return disperse_star

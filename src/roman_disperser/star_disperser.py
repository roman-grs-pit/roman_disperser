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


def _compute_dispersed_positions(optical_payload, xsca_star, ysca_star, wavelengths):
    """
    Compute dispersed SCA positions for all wavelengths.

    This is a helper function extracted for clarity. It converts the star
    position to FPA coordinates, traces the beam through the optical model,
    and converts back to SCA coordinates.

    Parameters
    ----------
    optical_payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    xsca_star, ysca_star : float
        Star position in SCA coordinates (1-indexed FITS)
    wavelengths : jnp.ndarray
        [N_wl] wavelengths in **microns**

    Returns
    -------
    xsca_disp, ysca_disp : jnp.ndarray
        [N_wl] dispersed positions in SCA coordinates
    """
    # Convert star position to FPA (optical model functions expect arrays)
    xsca_arr = jnp.atleast_1d(xsca_star)
    ysca_arr = jnp.atleast_1d(ysca_star)
    xfpa, yfpa = omj.sca_to_fpa(optical_payload, xsca_arr, ysca_arr)

    # Trace beam at all wavelengths (wavelengths already in microns)
    xmpa, ympa = omj.trace_beam(optical_payload, xfpa, yfpa, wavelengths)

    # Convert back to SCA
    xsca_disp, ysca_disp = omj.mpa_to_sca(optical_payload, xmpa, ympa)
    # Shapes: [1, N_wl] each - flatten to [N_wl]
    xsca_disp = xsca_disp.reshape(-1)
    ysca_disp = ysca_disp.reshape(-1)

    return xsca_disp, ysca_disp


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
    chunk_size=1000,
):
    """
    Disperse a single star through the Roman grism using wavelength-dependent PSFs.

    This function deposits PSFs along the spectral trace for each wavelength,
    simulating how a point source appears after grism dispersion.

    **Memory-efficient implementation:** Uses wavelength chunking with jax.lax.scan
    to avoid materializing all interpolated PSFs at once. Memory usage is
    independent of total wavelength count - only depends on chunk_size.

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
    chunk_size : int, optional
        Number of wavelengths to process per chunk (default: 1000).
        Larger chunks use more memory but may be faster due to better
        GPU utilization. Memory per chunk is approximately:
        chunk_size × PSF_y × PSF_x × 4 bytes × 4 (for indices and values)

    Returns
    -------
    output : jnp.ndarray
        [H, W] updated detector array with dispersed star

    Notes
    -----
    **Algorithm (memory-efficient chunked approach):**

    1. Spatial interpolation (once): Get PSFs at undispersed position for all
       grid wavelengths. This produces a small array [N_grid, PSF_y, PSF_x].

    2. Compute all dispersed positions (vectorized, cheap): Get detector
       positions for all wavelengths in one vectorized operation.

    3. Process wavelengths in chunks using lax.scan:
       a. Interpolate PSFs for this chunk from the grid
       b. Scale by flux
       c. Build scatter indices for all PSF pixels × wavelengths
       d. Batched scatter-add to output

    **Memory usage:**
    - PSF grid after spatial interpolation: ~8 MB (kept in memory)
    - Per chunk: ~136 MB for PSFs + ~271 MB for indices = ~540 MB total
    - Peak memory: ~620 MB (vs ~1.4 GB for non-chunked approach)

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
    >>> # Disperse a star with fine wavelength sampling
    >>> wavelengths = jnp.linspace(0.9, 2.0, 5000)  # microns
    >>> star_flux = jnp.ones(5000)  # Flat spectrum
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

    # Step 1: Spatial interpolation (once, small)
    # Get PSFs at undispersed position for all grid wavelengths
    psfs_grid = psf_model.interpolate_psf_spatial(
        psf_payload, xsca_star, ysca_star
    )  # [N_wl_grid, PSF_y, PSF_x]

    # Step 2: Compute ALL dispersed positions (cheap, vectorized)
    xsca_disp, ysca_disp = _compute_dispersed_positions(
        optical_payload, xsca_star, ysca_star, wavelengths
    )  # [N_wl] each

    # Get grid wavelengths for interpolation
    grid_wl = psf_payload['wavelengths']

    # Step 3: Pad inputs to multiple of chunk_size (for static shapes in lax.scan)
    # Padded entries have zero flux, so they don't contribute to output
    n_wl = len(wavelengths)
    n_padded = ((n_wl + chunk_size - 1) // chunk_size) * chunk_size
    pad_size = n_padded - n_wl
    n_chunks = n_padded // chunk_size

    # Pad arrays - zero flux for padded entries means they contribute nothing
    wavelengths_padded = jnp.pad(
        wavelengths, (0, pad_size), constant_values=wavelengths[-1]
    )
    star_flux_padded = jnp.pad(
        star_flux, (0, pad_size), constant_values=0.0
    )  # Zero flux!
    xsca_disp_padded = jnp.pad(
        xsca_disp, (0, pad_size), constant_values=xsca_disp[-1]
    )
    ysca_disp_padded = jnp.pad(
        ysca_disp, (0, pad_size), constant_values=ysca_disp[-1]
    )

    # Flatten relative grids for scatter-add
    rel_x_flat = rel_x.ravel()
    rel_y_flat = rel_y.ravel()
    n_psf_pixels = len(rel_x_flat)

    def process_chunk(output, chunk_idx):
        """Process a single wavelength chunk."""
        start = chunk_idx * chunk_size

        # Slice padded arrays (always exactly chunk_size elements)
        wl_chunk = jax.lax.dynamic_slice(
            wavelengths_padded, [start], [chunk_size]
        )
        flux_chunk = jax.lax.dynamic_slice(
            star_flux_padded, [start], [chunk_size]
        )
        x_chunk = jax.lax.dynamic_slice(
            xsca_disp_padded, [start], [chunk_size]
        )
        y_chunk = jax.lax.dynamic_slice(
            ysca_disp_padded, [start], [chunk_size]
        )

        # On-the-fly wavelength interpolation (vectorized over chunk)
        psfs_chunk = psf_model.interp_wavelength_chunk(psfs_grid, grid_wl, wl_chunk)
        # Shape: [chunk_size, PSF_y, PSF_x]

        # Scale by flux
        psfs_chunk = psfs_chunk * flux_chunk[:, None, None]

        # Build scatter arrays (vectorized)
        # For each wavelength i, PSF pixel j lands at:
        #   det_x = x_chunk[i] + rel_x_flat[j]
        #   det_y = y_chunk[i] + rel_y_flat[j]
        det_x = x_chunk[:, None] + rel_x_flat[None, :]  # [chunk, n_psf_pixels]
        det_y = y_chunk[:, None] + rel_y_flat[None, :]

        # Convert FITS 1-indexed to 0-indexed array indices
        idx_x = jnp.floor(det_x - 0.5).astype(jnp.int32).ravel()
        idx_y = jnp.floor(det_y - 0.5).astype(jnp.int32).ravel()
        values = psfs_chunk.ravel()

        # Batched scatter-add (GPU parallel atomics)
        output = output.at[idx_y, idx_x].add(
            values, mode="drop", wrap_negative_indices=False
        )
        return output, None

    # Use lax.scan for JIT-compatible loop over chunks
    output, _ = jax.lax.scan(process_chunk, output, jnp.arange(n_chunks))

    return output


def make_star_disperser(psf_payload, optical_payload, chunk_size=1000):
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
    chunk_size : int, optional
        Number of wavelengths to process per chunk (default: 1000).
        Larger chunks use more memory but may be faster. Memory per chunk
        is approximately: chunk_size × PSF_y × PSF_x × 4 bytes × 4.
        For 5000 wavelengths with chunk_size=1000, peak memory is ~620 MB.

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
    >>> # Disperse many stars efficiently (even with many wavelengths)
    >>> wavelengths = jnp.linspace(0.9, 2.0, 5000)  # microns
    >>> output = jnp.zeros((4088, 4088), dtype=jnp.float32)
    >>> for star in star_catalog:
    ...     output = disperser(star.x, star.y, wavelengths, star.flux, output)

    Notes
    -----
    - The PSF and optical payloads are captured in the closure
    - First call compiles the function (may take a few seconds)
    - Subsequent calls use the cached compiled function
    - Relative coordinate grid is pre-computed once
    - Memory usage is controlled by chunk_size and is independent of
      total wavelength count, enabling fine wavelength sampling (e.g., 1Å)
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
            chunk_size,
        )

    return disperse_star

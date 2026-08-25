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


def deposit_stack_native(
    stack, grid_wl, wavelengths, flux, xsca_disp, ysca_disp, output,
    oversample, chunk_size,
):
    """Deposit an oversampled stamp stack at native detector resolution.

    Shared hot loop of both dispersers: ``stack`` is a small set of
    oversampled stamps at the PSF grid wavelengths (spatially-interpolated
    PSFs for stars; warped+convolved galaxy images), and each fine
    wavelength deposits the linear interpolation of two bracketing stamps
    at its dispersed center position. The baseline implementation scattered
    every oversampled subpixel individually; this one pre-bins the stack to
    native pixels and scatters ``oversample**2`` (=16) times fewer elements
    — the measured deposit is compute-bound per element, so this is the
    ~2.9x end-to-end win (a10g, workbench 2026-08-19, SLURM 7143/7144).

    Why binning first is exact (per axis): all subpixels of one stamp share
    a single continuous center, and the baseline deposit index of subpixel
    j is ``floor(u + j/os)`` with ``u = center - 0.5 - (S-1)/(2 os)``. As j
    increments the argument steps by exactly 1/os px, so the partition of
    subpixels into native pixels is a run-length-``os`` blocking whose
    boundary offset depends only on which quarter-pixel ``frac(u)`` falls
    in: ``os`` discrete phases per axis, 16 in 2D at os=4. Pre-binning the
    stack with the phase-appropriate boundaries and depositing one value
    per native pixel performs the *identical additions*, merely regrouped —
    differences are f32 summation-order noise inside the issue-#22
    equivalence gate. Verified exactly (float64, dictionary equality) on
    2,000 random 1D + 200 2D center positions (research log 2026-08-19).
    Note this exactness argument is specific to the plain floor deposit
    used here; it does NOT hold for bilinear-weighted deposits (the legacy
    ``disperser.py`` path).

    Because binning is linear and commutes with the wavelength
    interpolation, the stack is binned once into the ``os x os`` phase
    variants ([os, os, N_grid, ny, nx] — same memory as the oversampled
    stack), and each fine wavelength selects its phase from the fractional
    part of its center position, interpolates two native stamps, and
    deposits ~S²/os² elements. The wavelength bracketing below mirrors
    ``psf_model.interp_wavelength_chunk`` exactly (searchsorted - 1,
    clipped to [0, N_grid-2], t clipped to [0, 1] for edge extrapolation).

    Out-of-detector handling matches the baseline: a native pixel is
    dropped by ``mode="drop"`` exactly when all of its subpixels would have
    been dropped (they share the native index by construction).

    Parameters
    ----------
    stack : jnp.ndarray
        [N_grid, S_y, S_x] oversampled stamps at the grid wavelengths.
    grid_wl : jnp.ndarray
        [N_grid] grid wavelengths (microns), strictly increasing.
    wavelengths, flux : jnp.ndarray
        [N_wl] fine wavelengths (microns) and per-wavelength flux.
    xsca_disp, ysca_disp : jnp.ndarray
        [N_wl] dispersed center positions (1-indexed FITS SCA pixels).
    output : jnp.ndarray
        [H, W] detector accumulator.
    oversample : int
        Stamp oversampling factor (4 in production).
    chunk_size : int
        Wavelengths per scan chunk. 2000 measured best on a10g for the
        native-binned element count (500/1000 were 17%/5% slower — the
        per-chunk work is 16x smaller than before, so scan overhead
        matters more than it did for the oversampled deposit).

    Returns
    -------
    output : jnp.ndarray
        [H, W] with the dispersed stamp flux added.
    """
    n_grid = stack.shape[0]
    s_y, s_x = stack.shape[-2:]
    os_ = int(oversample)

    # Native stamp size: k_max = floor((S-1+p)/os) <= (S-2)//os + 1, +1 for
    # the count -> one extra boundary pixel vs S/os (75.75 native px spans
    # up to 77 whole pixels for arbitrary phase).
    n_y = (s_y - 2) // os_ + 2
    n_x = (s_x - 2) // os_ + 2
    # Offset of subpixel j=0 from the stamp center, in native pixels.
    rel0_y = -(s_y - 1) / (2.0 * os_)
    rel0_x = -(s_x - 1) / (2.0 * os_)

    # Phase (p_y, p_x) left-pads by p subpixels before the os x os
    # block-sum, reproducing the run-length-os grouping that
    # floor(u + j/os) induces when frac(u) is in [p/os, (p+1)/os).
    def bin_phase(p_y, p_x):
        padded = jnp.pad(
            stack,
            ((0, 0),
             (p_y, os_ * n_y - s_y - p_y),
             (p_x, os_ * n_x - s_x - p_x)))
        return padded.reshape(n_grid, n_y, os_, n_x, os_).sum(axis=(2, 4))

    binned = jnp.stack([
        jnp.stack([bin_phase(p_y, p_x) for p_x in range(os_)])
        for p_y in range(os_)
    ])  # [os, os, N_grid, n_y, n_x]
    k_y = jnp.arange(n_y)
    k_x = jnp.arange(n_x)

    # Pad to chunk multiples; padded entries carry zero flux.
    n_wl = len(wavelengths)
    n_padded = ((n_wl + chunk_size - 1) // chunk_size) * chunk_size
    pad_size = n_padded - n_wl
    n_chunks = n_padded // chunk_size

    wavelengths_padded = jnp.pad(wavelengths, (0, pad_size),
                                 constant_values=wavelengths[-1])
    flux_padded = jnp.pad(flux, (0, pad_size), constant_values=0.0)
    x_disp_padded = jnp.pad(xsca_disp, (0, pad_size),
                            constant_values=xsca_disp[-1])
    y_disp_padded = jnp.pad(ysca_disp, (0, pad_size),
                            constant_values=ysca_disp[-1])

    def process_chunk(output, chunk_idx):
        start = chunk_idx * chunk_size
        wl_chunk = jax.lax.dynamic_slice(
            wavelengths_padded, [start], [chunk_size])
        flux_chunk = jax.lax.dynamic_slice(
            flux_padded, [start], [chunk_size])
        x_chunk = jax.lax.dynamic_slice(x_disp_padded, [start], [chunk_size])
        y_chunk = jax.lax.dynamic_slice(y_disp_padded, [start], [chunk_size])

        # Per-wavelength base native index m and boundary phase p, per axis.
        # u = detector position of subpixel j=0 minus the 0.5 FITS
        # half-pixel; the baseline computed floor(u + j/os) for every j.
        u_x = x_chunk - 0.5 + rel0_x
        u_y = y_chunk - 0.5 + rel0_y
        m_x = jnp.floor(u_x).astype(jnp.int32)
        m_y = jnp.floor(u_y).astype(jnp.int32)
        p_x = jnp.clip(jnp.floor((u_x - m_x) * os_), 0, os_ - 1
                       ).astype(jnp.int32)
        p_y = jnp.clip(jnp.floor((u_y - m_y) * os_), 0, os_ - 1
                       ).astype(jnp.int32)

        # Wavelength interpolation on the phase-selected pre-binned stack
        # (binning and linear interpolation commute); bracketing mirrors
        # psf_model.interp_wavelength_chunk.
        i0 = jnp.clip(jnp.searchsorted(grid_wl, wl_chunk) - 1, 0, n_grid - 2)
        t = (wl_chunk - grid_wl[i0]) / (grid_wl[i0 + 1] - grid_wl[i0])
        t = jnp.clip(t, 0.0, 1.0)
        lo = binned[p_y, p_x, i0]       # [chunk, n_y, n_x]
        hi = binned[p_y, p_x, i0 + 1]
        native_chunk = lo + t[:, None, None] * (hi - lo)
        native_chunk = native_chunk * flux_chunk[:, None, None]

        idx_y = m_y[:, None, None] + k_y[None, :, None]
        idx_x = m_x[:, None, None] + k_x[None, None, :]
        idx_y = jnp.broadcast_to(idx_y, native_chunk.shape)
        idx_x = jnp.broadcast_to(idx_x, native_chunk.shape)
        output = output.at[idx_y.ravel(), idx_x.ravel()].add(
            native_chunk.ravel(), mode="drop", wrap_negative_indices=False)
        return output, None

    output, _ = jax.lax.scan(process_chunk, output, jnp.arange(n_chunks))
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
    chunk_size=2000,
):
    """
    Disperse a single star through the Roman grism using wavelength-dependent PSFs.

    This function deposits PSFs along the spectral trace for each wavelength,
    simulating how a point source appears after grism dispersion.

    **Native-resolution deposit:** the PSF stack is pre-binned to native
    detector pixels (16-phase binning, see :func:`deposit_stack_native`) and
    each wavelength deposits ~oversample² (=16x) fewer elements than the
    per-subpixel scatter this replaces — same additions, regrouped, so the
    output is equivalent up to f32 summation order. Wavelength chunking with
    jax.lax.scan keeps memory independent of total wavelength count.

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
        Unused; kept for backward compatibility. The native-resolution
        deposit derives its indices from the dispersed center directly.
    rel_y : jnp.ndarray, optional
        Unused; kept for backward compatibility.
    chunk_size : int, optional
        Number of wavelengths to process per chunk (default: 2000 —
        measured best on a10g for the native-binned element count; the
        per-chunk arrays are 16x smaller than the oversampled deposit's,
        so larger chunks amortize scan overhead without the old memory
        cost).

    Returns
    -------
    output : jnp.ndarray
        [H, W] updated detector array with dispersed star

    Notes
    -----
    **Algorithm:**

    1. Spatial interpolation (once): Get PSFs at undispersed position for all
       grid wavelengths. This produces a small array [N_grid, PSF_y, PSF_x].

    2. Compute all dispersed positions (vectorized, cheap): Get detector
       positions for all wavelengths in one vectorized operation.

    3. Native-resolution chunked deposit (:func:`deposit_stack_native`):
       pre-bin the PSF stack into the 16 boundary-phase native variants
       once, then per chunk select each wavelength's phase, interpolate two
       native stamps, scale by flux, and scatter-add ~(PSF/os)² elements.

    **Memory usage:** the 16-phase binned stack costs the same as the
    oversampled PSF stack it replaces (~8 MB); per-chunk arrays are 16x
    smaller per wavelength than the oversampled deposit's.

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
    # Step 1: Spatial interpolation (once, small)
    # Get PSFs at undispersed position for all grid wavelengths
    psfs_grid = psf_model.interpolate_psf_spatial(
        psf_payload, xsca_star, ysca_star
    )  # [N_wl_grid, PSF_y, PSF_x]

    # Step 2: Compute ALL dispersed positions (cheap, vectorized)
    xsca_disp, ysca_disp = _compute_dispersed_positions(
        optical_payload, xsca_star, ysca_star, wavelengths
    )  # [N_wl] each

    # Step 3: Native-resolution chunked deposit
    return deposit_stack_native(
        psfs_grid, psf_payload['wavelengths'], wavelengths, star_flux,
        xsca_disp, ysca_disp, output,
        oversample=psf_payload['oversample'], chunk_size=chunk_size,
    )


def make_star_disperser(psf_payload, optical_payload, chunk_size=2000):
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
            chunk_size=chunk_size,
        )

    return disperse_star

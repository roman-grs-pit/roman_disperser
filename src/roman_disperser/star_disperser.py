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
- disperse_star_jit: Jitted disperse_star_psf taking payload arrays as arguments

All wavelength parameters are in **microns** (consistent with the optical model).

See docs/star_dispersion.md for the design document.
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import jit_payload
from . import optical_model_jax as omj
from . import psf_model

# Deposit schemes of deposit_binned / deposit_stack_native. "interp" is the
# default; "floor" is the historical deposit (the pinned golden frames are
# blessed with it). See deposit_binned.
DEPOSIT_MODES = ("floor", "interp")


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
    oversample, chunk_size, deposit="interp",
):
    """Deposit an oversampled stamp stack at native detector resolution.

    ``deposit`` selects how a stamp centre between phase-aligned positions
    is handled (see :data:`DEPOSIT_MODES`): ``"interp"`` (default; two-tap
    linear interpolation between adjacent phases, see
    :func:`deposit_binned`) or ``"floor"`` (the historical deposit described
    below).
    This function is :func:`bin_stack_phases` followed by
    :func:`deposit_binned`.

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
        Wavelengths per scan chunk. Speed is insensitive to it: chunk 500
        and 2000 measured identical on a10g on the shipped code (SLURM
        7183, 2026-08-25). The callers default to 2000 because it is
        affordable, not because it is faster — per-chunk arrays are
        [chunk, n_y, n_x] at *native* resolution, so chunk 2000 costs
        ~4x less per-chunk memory than the old oversampled deposit did at
        chunk 500 (galaxy, 77x77 native stamp: ~47 MB f32 values + ~95 MB
        int32 indices, vs ~184 + ~367 MB; sizes computed from the array
        shapes, not a measured peak).
    deposit : {"floor", "interp"}, optional
        Deposit scheme (default ``"interp"``; ``"floor"`` is the historical deposit); see :func:`deposit_binned`.

    Returns
    -------
    output : jnp.ndarray
        [H, W] with the dispersed stamp flux added.
    """
    os_ = int(oversample)
    binned = bin_stack_phases(stack, os_)
    return deposit_binned(
        binned, stack.shape[-2:], grid_wl, wavelengths, flux,
        xsca_disp, ysca_disp, output,
        oversample=os_, chunk_size=chunk_size, deposit=deposit,
    )


def _native_stamp_size(s_y, s_x, os_):
    """Native stamp size (n_y, n_x) of the phase-binned [S_y, S_x] stamp.

    k_max = floor((S-1+p)/os) <= (S-2)//os + 1, +1 for the count -> one
    extra boundary pixel vs S/os (75.75 native px spans up to 77 whole
    pixels for arbitrary phase).
    """
    return (s_y - 2) // os_ + 2, (s_x - 2) // os_ + 2


def bin_stack_phases(stack, oversample):
    """Bin oversampled stamps to native pixels for all ``os x os`` phases.

    Which subpixels share a native pixel depends on the stamp centre's
    phase p = which 1/os-pixel ``frac(u)`` falls in (see
    :func:`deposit_stack_native`), so the stack is binned once for every
    phase: phase p pads p subpixels at the low edge, then sums ``os x os``
    blocks. Linear in ``stack``, so it commutes with the wavelength
    interpolation and with the bilinear spatial PSF interpolation (which is
    what lets ``make_star_disperser(hoist_psf=True)`` bin the PSF grid
    nodes once per SCA instead of once per star).

    Parameters
    ----------
    stack : jnp.ndarray
        [..., S_y, S_x] oversampled stamps (any leading dimensions).
    oversample : int
        Stamp oversampling factor.

    Returns
    -------
    binned : jnp.ndarray
        [os, os, ..., n_y, n_x] with ``n = (S-2)//os + 2`` per axis;
        ``binned[p_y, p_x]`` is the stack binned at phase (p_y, p_x).
    """
    os_ = int(oversample)
    s_y, s_x = stack.shape[-2:]
    lead = stack.shape[:-2]
    n_y, n_x = _native_stamp_size(s_y, s_x, os_)
    no_pad = [(0, 0)] * len(lead)

    def bin_phase(p_y, p_x):
        padded = jnp.pad(
            stack,
            no_pad + [(p_y, os_ * n_y - s_y - p_y),
                      (p_x, os_ * n_x - s_x - p_x)])
        return padded.reshape(lead + (n_y, os_, n_x, os_)).sum(axis=(-3, -1))

    return jnp.stack([
        jnp.stack([bin_phase(p_y, p_x) for p_x in range(os_)])
        for p_y in range(os_)
    ])  # [os, os, ..., n_y, n_x]


def _interp_taps(u, os_):
    """Two-tap phase weights for the interpolating deposit, per axis.

    ``u`` is the detector position of subpixel j=0 minus the 0.5 FITS
    half-pixel (as in the floor deposit). The continuous phase
    ``f = frac(u) * os - 0.5`` is an integer exactly when the subpixel
    boxes are aligned with the native pixel edges; there the floor
    deposit with phase ``f`` is the exact box-overlap deposit. Between two
    aligned positions the exact overlap is linear in the shift, so the
    exact deposit is ``(1 - w) * phase(p0) + w * phase(p0 + 1)`` with
    ``p0 = floor(f)``, ``w = f - p0``. ``p0 = -1`` is phase ``os - 1``
    one native pixel lower, and ``p0 + 1 = os`` is phase 0 one pixel
    higher; both are expressed as a base pixel plus a 0/1 offset ``d`` of
    tap b relative to tap a.

    Returns
    -------
    p_a, p_b : int32 phases of the two taps
    w : float weight of tap b (tap a gets 1 - w)
    base : int32 native base pixel of tap a
    d : int32 in {0, 1}, offset of tap b's stamp relative to tap a's
    """
    m = jnp.floor(u)
    f = (u - m) * os_ - 0.5
    p0 = jnp.floor(f)
    w = f - p0
    p0 = p0.astype(jnp.int32)
    m = m.astype(jnp.int32)
    p_a = jnp.mod(p0, os_)
    off_a = jnp.where(p0 < 0, -1, 0)
    p_b = jnp.mod(p0 + 1, os_)
    off_b = jnp.where(p0 + 1 >= os_, 1, 0)
    return p_a, p_b, w, m + off_a, off_b - off_a


def _place(stamps, d, axis):
    """Grow ``stamps`` [c, ny, nx] by one along ``axis`` (1 or 2), placing
    each stamp at offset ``d[c]`` in {0, 1} (zero in the vacated row)."""
    pad_lo = [(0, 0)] * 3
    pad_hi = [(0, 0)] * 3
    pad_lo[axis] = (1, 0)
    pad_hi[axis] = (0, 1)
    return jnp.where(d[:, None, None] == 1,
                     jnp.pad(stamps, pad_lo), jnp.pad(stamps, pad_hi))


def deposit_binned(
    binned, stamp_shape, grid_wl, wavelengths, flux, xsca_disp, ysca_disp,
    output, oversample, chunk_size, deposit="interp",
):
    """Chunked deposit of a phase-binned stamp stack (the hot loop).

    ``binned`` is :func:`bin_stack_phases` of the [N_grid, S_y, S_x]
    oversampled stack, ``stamp_shape = (S_y, S_x)``. Other arguments as in
    :func:`deposit_stack_native`.

    Deposit schemes
    ---------------
    ``"floor"``
        Each fine wavelength selects the single phase its centre falls in
        and deposits [n_y, n_x] native values. Identical to the
        per-subpixel floor deposit (every subpixel lands wholly in the
        native pixel containing its centre), i.e. the phase is quantised to
        1/os px: line positions are off by up to ~1/(2 os) px across and
        along dispersion, and the continuum of a point source ripples at
        the ~1 % level. Piecewise constant in position, so
        its derivative w.r.t. source position is zero.
    ``"interp"``
        Each fine wavelength combines the two adjacent phases per axis
        (four phase stamps, bilinear weights from :func:`_interp_taps`)
        into one [n_y+1, n_x+1] stamp, then scatters it once. This is the
        exact box-overlap deposit of the oversampled stamp (each subpixel a
        box of width 1/os px split between native pixels by overlap
        length), continuous and piecewise linear in the centre position,
        and flux-conserving. Cost: four (not one) gathers + interpolations
        per wavelength and ~(1 + 2/n) more scattered elements; measured
        ~1.7x the floor deposit on CPU at the same wavelength sampling.

    Returns
    -------
    output : jnp.ndarray
        [H, W] with the dispersed stamp flux added.
    """
    if deposit not in DEPOSIT_MODES:
        raise ValueError(
            f"deposit must be one of {DEPOSIT_MODES}, got {deposit!r}")
    n_grid = binned.shape[2]
    s_y, s_x = stamp_shape
    os_ = int(oversample)
    n_y, n_x = binned.shape[-2:]
    # Offset of subpixel j=0 from the stamp center, in native pixels.
    rel0_y = -(s_y - 1) / (2.0 * os_)
    rel0_x = -(s_x - 1) / (2.0 * os_)
    if deposit == "interp":
        k_y = jnp.arange(n_y + 1)
        k_x = jnp.arange(n_x + 1)
    else:
        k_y = jnp.arange(n_y)
        k_x = jnp.arange(n_x)

    # Split into n_chunks = ceil(N / chunk_size) equal chunks of at most
    # chunk_size (not chunk_size each), so the zero-flux padding is < n_chunks
    # samples: 5501 -> 3 x 1834 (not 3 x 2000), and the 4 A rebinned 2751 ->
    # 2 x 1376 (not 2 x 2000). Padded entries carry zero flux.
    n_wl = len(wavelengths)
    n_chunks = -(-n_wl // chunk_size)
    chunk_size = -(-n_wl // n_chunks)
    n_padded = n_chunks * chunk_size
    pad_size = n_padded - n_wl

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

        # u = detector position of subpixel j=0 minus the 0.5 FITS
        # half-pixel; the baseline computed floor(u + j/os) for every j.
        u_x = x_chunk - 0.5 + rel0_x
        u_y = y_chunk - 0.5 + rel0_y

        # Wavelength bracketing mirrors psf_model.interp_wavelength_chunk
        # (binning and linear interpolation commute).
        i0 = jnp.clip(jnp.searchsorted(grid_wl, wl_chunk) - 1, 0, n_grid - 2)
        t = (wl_chunk - grid_wl[i0]) / (grid_wl[i0 + 1] - grid_wl[i0])
        t = jnp.clip(t, 0.0, 1.0)[:, None, None]

        def stamp_at(p_y, p_x):  # phase-selected, wavelength-interpolated
            lo = binned[p_y, p_x, i0]       # [chunk, n_y, n_x]
            hi = binned[p_y, p_x, i0 + 1]
            return lo + t * (hi - lo)

        if deposit == "interp":
            pa_y, pb_y, w_y, m_y, d_y = _interp_taps(u_y, os_)
            pa_x, pb_x, w_x, m_x, d_x = _interp_taps(u_x, os_)
            w_x = w_x[:, None, None]
            w_y = w_y[:, None, None]
            grow_x = [(0, 0), (0, 0), (0, 1)]
            grow_y = [(0, 0), (0, 1), (0, 0)]
            row_a = ((1 - w_x) * jnp.pad(stamp_at(pa_y, pa_x), grow_x)
                     + w_x * _place(stamp_at(pa_y, pb_x), d_x, 2))
            row_b = ((1 - w_x) * jnp.pad(stamp_at(pb_y, pa_x), grow_x)
                     + w_x * _place(stamp_at(pb_y, pb_x), d_x, 2))
            native_chunk = ((1 - w_y) * jnp.pad(row_a, grow_y)
                            + w_y * _place(row_b, d_y, 1))
        else:
            # Base native index m and boundary phase p, per axis.
            m_x = jnp.floor(u_x).astype(jnp.int32)
            m_y = jnp.floor(u_y).astype(jnp.int32)
            p_x = jnp.clip(jnp.floor((u_x - m_x) * os_), 0, os_ - 1
                           ).astype(jnp.int32)
            p_y = jnp.clip(jnp.floor((u_y - m_y) * os_), 0, os_ - 1
                           ).astype(jnp.int32)
            native_chunk = stamp_at(p_y, p_x)
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
    deposit="interp",
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
        Number of wavelengths to process per chunk (default: 2000). Speed
        is insensitive to it (500 and 2000 measured identical on a10g);
        2000 is kept because the native-resolution per-chunk arrays make
        it cheap in memory. See :func:`deposit_stack_native`.
    deposit : {"floor", "interp"}, optional
        Deposit scheme (default ``"interp"``; ``"floor"`` is the historical deposit); see :func:`deposit_binned`.

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
    oversample = psf_payload['oversample']
    stamp_shape = psf_payload['psf_grid'].shape[-2:]

    # Step 1: Spatial interpolation (once, small) of the phase-binned PSF
    # stack at the undispersed position. With a hoisted payload
    # (make_star_disperser(hoist_psf=True)) the grid nodes are already
    # binned, and binning commutes with the bilinear blend, so blend the
    # binned nodes directly; otherwise blend, then bin.
    if 'psf_grid_binned' in psf_payload:
        binned = psf_model.interpolate_psf_spatial(
            psf_payload, xsca_star, ysca_star, grid_key='psf_grid_binned'
        )  # [os, os, N_wl_grid, n_y, n_x]
    else:
        psfs_grid = psf_model.interpolate_psf_spatial(
            psf_payload, xsca_star, ysca_star
        )  # [N_wl_grid, PSF_y, PSF_x]
        binned = bin_stack_phases(psfs_grid, oversample)

    # Step 2: Compute ALL dispersed positions (cheap, vectorized)
    xsca_disp, ysca_disp = _compute_dispersed_positions(
        optical_payload, xsca_star, ysca_star, wavelengths
    )  # [N_wl] each

    # Step 3: Native-resolution chunked deposit
    return deposit_binned(
        binned, stamp_shape, psf_payload['wavelengths'], wavelengths,
        star_flux, xsca_disp, ysca_disp, output,
        oversample=oversample, chunk_size=chunk_size, deposit=deposit,
    )


def bin_psf_grid(psf_grid, oversample):
    """Phase-bin every node of a PSF grid (the star-PSF hoist, once per SCA).

    Parameters
    ----------
    psf_grid : jnp.ndarray
        [N_y, N_x, N_wl, PSF_y, PSF_x] payload PSF grid.
    oversample : int

    Returns
    -------
    jnp.ndarray
        [N_y, N_x, os, os, N_wl, n_y, n_x], i.e. :func:`bin_stack_phases`
        of each node stack. Same size as ``psf_grid`` (~121 MB for the
        production 4x4x56 grid at 184x184).
    """
    return _bin_psf_grid_jit(psf_grid, oversample=int(oversample))


@partial(jax.jit, static_argnames=("oversample",))
def _bin_psf_grid_jit(psf_grid, *, oversample):
    binned = bin_stack_phases(psf_grid, oversample)  # [os, os, Ny, Nx, ...]
    return jnp.moveaxis(binned, (0, 1), (2, 3))


def make_star_disperser(psf_payload, optical_payload, chunk_size=2000,
                        deposit="interp", hoist_psf=False):
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
        Number of wavelengths to process per chunk (default: 2000). Speed
        is insensitive to it (500 and 2000 measured identical on a10g);
        it sets per-chunk memory, which at native resolution is
        chunk_size × n_y × n_x × 4 bytes for the values plus twice that
        for the int32 indices. See :func:`deposit_stack_native`.
    deposit : {"floor", "interp"}, optional
        Deposit scheme (default ``"interp"``, the exact box-overlap deposit,
        continuous in position); ``"floor"`` is the historical deposit.
        See :func:`deposit_binned`.
    hoist_psf : bool, optional
        If True, phase-bin all PSF grid nodes once here
        (:func:`bin_psf_grid`, +1x the PSF grid in device memory, ~121 MB
        per order at production size) and blend the binned nodes per star,
        instead of blending the oversampled PSF and binning it per star.
        Same result up to f32 summation order (binning is linear); ~1.35x
        faster per star on CPU. Works with either ``deposit``.

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
    - The payload *arrays* are passed to :func:`disperse_star_jit` as jit
      arguments (not captured as constants), so dispersers for different
      SCAs share one compiled program; only shapes and the few static
      ints/strings in ``jit_payload`` key the compile.
    - The returned function carries ``.payload_args = (arrays, static,
      chunk_size, deposit)``; ``pipeline.make_batched_star_fori`` uses it to keep the
      payloads as arguments in the batched loop too.
    - First call compiles (tens of seconds on GPU); later calls, including
      dispersers built for other SCAs, reuse the compiled program
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

    if deposit not in DEPOSIT_MODES:
        raise ValueError(
            f"deposit must be one of {DEPOSIT_MODES}, got {deposit!r}")

    arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
    if hoist_psf:
        psf_arrays, optical_arrays = arrays
        psf_arrays = dict(psf_arrays, psf_grid_binned=bin_psf_grid(
            psf_arrays['psf_grid'], oversample))
        arrays = (psf_arrays, optical_arrays)

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
        return disperse_star_jit(
            arrays, xsca, ysca, wavelengths, flux, output,
            static=static, chunk_size=chunk_size, deposit=deposit,
        )

    # Exposed so pipeline.make_batched_star_fori can pass the payload arrays
    # as jit arguments instead of capturing them (see jit_payload).
    disperse_star.payload_args = (arrays, static, chunk_size, deposit)
    return disperse_star


@partial(jax.jit, static_argnames=("static", "chunk_size", "deposit"))
def disperse_star_jit(arrays, xsca, ysca, wavelengths, flux, output, *,
                      static, chunk_size=2000, deposit="interp"):
    """Jitted :func:`disperse_star_psf` with the payloads as arguments.

    ``arrays, static = jit_payload.split_payloads(psf_payload,
    optical_payload)``. The compiled program depends only on the array
    shapes and ``static`` (oversample, polynomial dimensions, wavelength
    transform), which are the same for every SCA of an element, so one
    compilation serves all SCAs. Capturing the payloads in a closure instead
    bakes the PSF grid into the program as a constant: one compile and one
    ~111 MB cache entry per (SCA, order).
    """
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    return disperse_star_psf(
        psf_payload, optical_payload, xsca, ysca, wavelengths, flux, output,
        chunk_size=chunk_size, deposit=deposit,
    )

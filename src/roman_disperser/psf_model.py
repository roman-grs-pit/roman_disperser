"""
PSF model for Roman grism star dispersion.

This module provides PSF (Point Spread Function) modeling infrastructure for
dispersing stars through the Roman Space Telescope grism. It integrates STPSF
(Space Telescope PSF) calculations with the JAX-based disperser.

Key Features:
- PSF payload structure for JIT-compatible storage
- Trilinear interpolation across spatial position (x, y) and wavelength (λ)
- Field-dependent PSFs (vary across detector)
- Wavelength-dependent PSF shape
- Oversampled PSFs (4×) with detector effects for sub-pixel positioning accuracy

Coordinate Systems:
- STPSF uses 0-indexed coordinates, 4096×4096 pixels
- Disperser uses 1-indexed FITS coordinates, 4088×4088 pixels
- Conversion handled by psf_utils.py (with documented assumptions)

Usage:
    >>> import roman_disperser.psf_model as psf_model
    >>> # Generate PSF grid (this is slow, ~30-60 min for 10×10×15 grid)
    >>> payload = psf_model.make_psf_payload(detector='WFI05', order='1')
    >>> # Interpolate PSF at arbitrary position
    >>> psf = psf_model.interpolate_psf(payload, xsca=2000.0, ysca=2000.0,
    ...                                  wavelength=1.5e-6)

See Also:
    docs/stpsf.md : STPSF integration reference
    docs/star_dispersion.md : Design requirements
"""

import time
import numpy as np
import jax.numpy as jnp
import jax

from .psf_utils import sca_to_stpsf_position, stpsf_to_sca_position


# ============================================================================
# PSF PAYLOAD STRUCTURE
# ============================================================================


def make_psf_payload(
    detector='WFI05',
    order='1',
    wavelengths=None,
    spatial_grid=None,
    fov_arcsec=5.0,
    oversample=4,
    verbose=True,
):
    """
    Create PSF payload for star dispersion with timing benchmarks.

    This function generates a 4D PSF grid covering spatial position (x, y)
    and wavelength (λ) using STPSF. PSFs are calculated at a coarse grid of
    positions and wavelengths, then interpolated for intermediate values.

    ⚠️ PERFORMANCE WARNING: This function is SLOW!
    Expected time: ~30-60 minutes for default 10×10×15 grid (1500 PSF calculations)
    Consider using caching (save/load) for repeated use.

    Parameters
    ----------
    detector : str, optional
        WFI detector name (default: 'WFI05' - central detector)
        Valid: 'WFI01' through 'WFI18'

    order : str, optional
        Grism spectral order (default: '1')
        Valid: '0' (zeroth order, undispersed), '1' (first order, dispersed)
        Maps to STPSF filters: '0' -> 'GRISM0', '1' -> 'GRISM1'

    wavelengths : array_like, optional
        Wavelengths in meters for PSF calculations
        Default: 15 wavelengths from 0.9 to 2.0 μm (full grism range)
        Recommendation: 15-20 wavelengths for good interpolation

    spatial_grid : dict, optional
        Spatial grid specification: {'x': x_array, 'y': y_array}
        x_array, y_array in SCA coordinates (1-indexed FITS, range 1-4088)
        Default: 10×10 grid from pixel 1 to 4088 (full detector range)

    fov_arcsec : float, optional
        PSF field of view in arcseconds (default: 5.0)
        Larger FOV captures more flux but increases memory usage
        5" FOV ≈ 15× FWHM, captures >99% enclosed energy

    oversample : int, optional
        PSF oversampling factor (default: 4)
        CRITICAL: Must use oversampling for sub-pixel positioning accuracy
        4× oversampling is required for accurate star dispersion

    verbose : bool, optional
        Print timing and progress information (default: True)

    Returns
    -------
    payload : dict
        PSF payload with keys:
        - 'detector': str, detector name
        - 'wavelengths': jnp.ndarray [N_wl], wavelengths in meters
        - 'wl_grid': jnp.ndarray [N_wl], same as wavelengths (for consistency)
        - 'spatial_x': jnp.ndarray [N_x], SCA x-coordinates
        - 'spatial_y': jnp.ndarray [N_y], SCA y-coordinates
        - 'psf_grid': jnp.ndarray [N_wl, N_y, N_x, PSF_y, PSF_x], PSF datacube
        - 'psf_fov_pixels': int, PSF array size (pixels)
        - 'pixel_scale': float, detector pixel scale (0.11 arcsec/pixel for WFI)
        - 'oversample': int, oversampling factor used
        - 'timing': dict, generation timing information

    Examples
    --------
    >>> # Generate PSF grid with default settings (first order)
    >>> payload = make_psf_payload(detector='WFI05', order='1')
    >>> print(f"PSF grid shape: {payload['psf_grid'].shape}")
    >>> # Expected: (15, 10, 10, ~108, ~108) for 3" FOV at 4× oversample

    >>> # Zeroth order (undispersed) PSFs
    >>> payload_0th = make_psf_payload(detector='WFI05', order='0')

    >>> # Custom wavelength sampling (faster for testing)
    >>> wavelengths = np.linspace(0.9e-6, 2.0e-6, 5)  # Only 5 wavelengths
    >>> payload = make_psf_payload(order='1', wavelengths=wavelengths)

    >>> # Coarse spatial grid for quick tests
    >>> spatial_grid = {
    ...     'x': np.linspace(1000, 3000, 5),
    ...     'y': np.linspace(1000, 3000, 5)
    ... }
    >>> payload = make_psf_payload(order='1', spatial_grid=spatial_grid)

    Notes
    -----
    - PSFs calculated using STPSF's OVERDIST extension (oversampled + detector effects)
    - OVERDIST includes geometric distortion, charge diffusion, pixel sampling
    - Oversampling (4×) is REQUIRED for sub-pixel positioning accuracy
    - PSF grid is stored as JAX arrays for GPU compatibility
    - Use JIT closure pattern for efficient disperser integration

    See Also
    --------
    interpolate_psf : Interpolate PSF at arbitrary position
    save_psf_payload : Save payload to disk (caching) - NOT YET IMPLEMENTED
    load_psf_payload : Load payload from disk - NOT YET IMPLEMENTED
    """
    # Setup default wavelengths
    if wavelengths is None:
        # 15 wavelengths across full grism range (0.9 - 2.0 μm)
        wavelengths = np.linspace(0.9e-6, 2.0e-6, 15)

    # Validate wavelengths are strictly increasing (required for interpolation)
    wavelengths = np.asarray(wavelengths)
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelengths must be strictly increasing")

    # Setup default spatial grid
    if spatial_grid is None:
        # 10×10 grid across full detector range (1 to 4088)
        # STPSF handles edge extrapolation for corner positions
        x_grid = np.linspace(1, 4088, 10)
        y_grid = np.linspace(1, 4088, 10)
        spatial_grid = {'x': x_grid, 'y': y_grid}

    # Validate spatial grids are strictly increasing
    x_grid = np.asarray(spatial_grid['x'])
    y_grid = np.asarray(spatial_grid['y'])
    if not np.all(np.diff(x_grid) > 0):
        raise ValueError("Spatial x grid must be strictly increasing")
    if not np.all(np.diff(y_grid) > 0):
        raise ValueError("Spatial y grid must be strictly increasing")

    # Compute PSF grid with timing
    if verbose:
        print(f"Generating PSF grid for {detector}, order {order}...")
        print(f"  Spatial grid: {len(spatial_grid['x'])}×{len(spatial_grid['y'])} positions")
        print(f"  Wavelengths: {len(wavelengths)} samples ({wavelengths[0]*1e6:.2f}-{wavelengths[-1]*1e6:.2f} μm)")
        print(f"  Total PSFs: {len(spatial_grid['x']) * len(spatial_grid['y']) * len(wavelengths)}")
        print(f"  FOV: {fov_arcsec:.1f} arcsec, Oversample: {oversample}×")
        print(f"  This may take 30-60 minutes...")

    psf_grid, timing = _compute_psf_grid_with_timing(
        detector, order, wavelengths, spatial_grid, fov_arcsec, oversample, verbose
    )

    # Return JAX-compatible payload
    payload = {
        'detector': detector,
        'order': order,
        'wavelengths': jnp.array(wavelengths),
        'wl_grid': jnp.array(wavelengths),  # Alias for consistency with optical model
        'spatial_x': jnp.array(spatial_grid['x']),
        'spatial_y': jnp.array(spatial_grid['y']),
        'psf_grid': jnp.array(psf_grid, dtype=jnp.float32),
        'psf_fov_pixels': psf_grid.shape[-1],  # Assumes square PSF
        'pixel_scale': 0.11,  # Roman WFI pixel scale (arcsec/pixel)
        'oversample': oversample,
        'timing': timing,
    }

    if verbose:
        memory_mb = payload['psf_grid'].nbytes / 1e6
        print(f"\nPSF payload created:")
        print(f"  PSF grid shape: {payload['psf_grid'].shape}")
        print(f"  Memory usage: {memory_mb:.1f} MB")
        print(f"  Timing: {timing['total_time']:.1f}s ({timing['total_time']/60:.1f} min)")

    return payload


# ============================================================================
# PSF GRID GENERATION (STPSF Integration)
# ============================================================================


def _compute_psf_grid_with_timing(
    detector, order, wavelengths, spatial_grid, fov_arcsec, oversample, verbose
):
    """
    Compute PSF grid using STPSF with detailed timing information.

    Parameters
    ----------
    detector : str
        WFI detector name
    order : str
        Spectral order ('0' or '1')
    wavelengths : array_like
        Wavelengths in meters
    spatial_grid : dict
        {'x': x_array, 'y': y_array} in SCA coordinates
    fov_arcsec : float
        PSF field of view in arcseconds
    oversample : int
        Oversampling factor
    use_fast : bool
        Use calc_datacube_fast() method
    verbose : bool
        Print progress information

    Returns
    -------
    psf_grid : ndarray
        Shape: [N_wl, N_y, N_x, PSF_y, PSF_x]
    timing : dict
        {'total_time': float, 'per_psf_time': float, 'n_psfs': int}
    """
    import stpsf.roman

    start_time = time.time()

    # Map order to STPSF filter name
    filter_map = {
        '0': 'GRISM0',  # Zeroth order (undispersed)
        '1': 'GRISM1',  # First order (dispersed)
    }
    if order not in filter_map:
        raise ValueError(f"Invalid order '{order}'. Must be '0' or '1'.")

    wfi = stpsf.roman.WFI()
    wfi.filter = filter_map[order]
    wfi.detector = detector

    x_grid = spatial_grid['x']
    y_grid = spatial_grid['y']
    N_x, N_y = len(x_grid), len(y_grid)
    N_wl = len(wavelengths)

    psf_grid = []
    n_calculated = 0
    n_total = N_x * N_y

    for iy, ysca in enumerate(y_grid):
        for ix, xsca in enumerate(x_grid):
            # Convert SCA to STPSF position
            x_stpsf, y_stpsf = sca_to_stpsf_position(float(xsca), float(ysca))

            # STPSF expects tuple of floats (not JAX arrays)
            wfi.detector_position = (float(x_stpsf), float(y_stpsf))

            # Calculate datacube at this position
            # CRITICAL: Use OVERDIST extension for sub-pixel accuracy + detector effects
            # Note: add_distortion is no longer needed - all WFI PSFs natively include distortion
            datacube = wfi.calc_datacube(
                wavelengths, fov_arcsec=fov_arcsec, oversample=oversample
            )
            # Use OVERDIST: oversampled + detector effects (distortion, diffusion)
            psf_cube = datacube['OVERDIST'].data  # [N_wl, PSF_y, PSF_x]

            psf_grid.append(psf_cube)
            n_calculated += 1

            if verbose and n_calculated % 10 == 0:
                elapsed = time.time() - start_time
                rate = n_calculated / elapsed
                remaining = (n_total - n_calculated) / rate
                print(f"  Progress: {n_calculated}/{n_total} positions "
                      f"({100*n_calculated/n_total:.1f}%), "
                      f"ETA: {remaining/60:.1f} min")

    # Reshape to [N_wl, N_y, N_x, PSF_y, PSF_x]
    psf_grid = np.array(psf_grid)  # [N_y*N_x, N_wl, PSF_y, PSF_x]
    psf_grid = psf_grid.reshape(N_y, N_x, N_wl, *psf_grid.shape[-2:])
    psf_grid = psf_grid.transpose(2, 0, 1, 3, 4)  # [N_wl, N_y, N_x, PSF_y, PSF_x]

    elapsed = time.time() - start_time
    n_psfs = N_x * N_y * N_wl

    timing = {
        'total_time': elapsed,
        'per_psf_time': elapsed / n_psfs,
        'n_psfs': n_psfs,
    }

    if verbose:
        print(f"\nPSF grid generation complete:")
        print(f"  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Time per PSF: {timing['per_psf_time']:.2f} seconds")
        print(f"  Total PSFs calculated: {n_psfs}")

    return psf_grid, timing


# ============================================================================
# TRILINEAR INTERPOLATION
# ============================================================================


def interpolate_psf(payload, xsca, ysca, wavelength):
    """
    Interpolate PSF at arbitrary (x, y, λ) using trilinear interpolation.

    This function provides wavelength-dependent, field-dependent PSFs at
    any position and wavelength by interpolating the precomputed PSF grid.

    Uses edge extrapolation: positions outside the grid use the nearest
    edge PSF. This is appropriate for dispersed stars that may land near
    detector edges but still scatter light onto the detector.

    This function is JAX-compatible and JIT-compilable.

    Parameters
    ----------
    payload : dict
        PSF payload from make_psf_payload()
    xsca, ysca : float or jnp.ndarray
        SCA coordinates (1-indexed FITS, range 1-4088)
        Can be scalar or array for vectorized operation
    wavelength : float or jnp.ndarray
        Wavelength in meters (range 1.0e-6 to 1.93e-6 for grism)
        Can be scalar or array (must match shape of xsca/ysca if arrays)

    Returns
    -------
    psf : jnp.ndarray
        Interpolated PSF array
        Shape: [PSF_y, PSF_x] if scalar inputs
        Shape: [..., PSF_y, PSF_x] if array inputs

    Examples
    --------
    >>> # Single PSF at detector center, mid-wavelength
    >>> psf = interpolate_psf(payload, xsca=2044.0, ysca=2044.0,
    ...                        wavelength=1.5e-6)
    >>> psf.shape
    (108, 108)  # For 3" FOV at 4× oversample

    >>> # Vectorized: PSFs for multiple positions
    >>> import jax.numpy as jnp
    >>> xsca = jnp.array([1000.0, 2000.0, 3000.0])
    >>> ysca = jnp.array([1000.0, 2000.0, 3000.0])
    >>> wavelength = jnp.array([1.0e-6, 1.5e-6, 1.9e-6])
    >>> psfs = interpolate_psf(payload, xsca, ysca, wavelength)
    >>> psfs.shape
    (3, 108, 108)

    Notes
    -----
    - Interpolation is linear in all three dimensions (x, y, λ)
    - Edge extrapolation: uses nearest grid PSF for out-of-bounds positions
    - JIT-compilable: use closure pattern for efficient repeated calls
    - For many stars, consider using jax.vmap for parallelization

    See Also
    --------
    make_psf_payload : Create PSF payload
    """
    # Extract grid parameters
    wl_grid = payload['wl_grid']
    x_grid = payload['spatial_x']
    y_grid = payload['spatial_y']
    psf_grid = payload['psf_grid']  # [N_wl, N_y, N_x, PSF_y, PSF_x]

    # 1. Find wavelength bracket
    wl_idx = jnp.searchsorted(wl_grid, wavelength)  # Index of next wavelength
    wl_idx = jnp.clip(wl_idx, 1, len(wl_grid) - 1)  # Ensure in bounds

    wl_idx_lo = wl_idx - 1
    wl_idx_hi = wl_idx

    wl_lo = wl_grid[wl_idx_lo]
    wl_hi = wl_grid[wl_idx_hi]
    # Division is safe: grid values should be distinct
    wl_frac = (wavelength - wl_lo) / (wl_hi - wl_lo)
    # Clamp to [0, 1] for edge extrapolation (not linear extrapolation)
    # For PSFs, we want to use nearest edge value for off-grid points
    wl_frac = jnp.clip(wl_frac, 0.0, 1.0)

    # 2. Find spatial bracket (x dimension)
    x_idx = jnp.searchsorted(x_grid, xsca)
    x_idx = jnp.clip(x_idx, 1, len(x_grid) - 1)

    x_idx_lo = x_idx - 1
    x_idx_hi = x_idx

    x_lo = x_grid[x_idx_lo]
    x_hi = x_grid[x_idx_hi]
    # Division is safe: grid values should be distinct
    x_frac = (xsca - x_lo) / (x_hi - x_lo)
    # Clamp to [0, 1] for edge extrapolation
    x_frac = jnp.clip(x_frac, 0.0, 1.0)

    # 3. Find spatial bracket (y dimension)
    y_idx = jnp.searchsorted(y_grid, ysca)
    y_idx = jnp.clip(y_idx, 1, len(y_grid) - 1)

    y_idx_lo = y_idx - 1
    y_idx_hi = y_idx

    y_lo = y_grid[y_idx_lo]
    y_hi = y_grid[y_idx_hi]
    # Division is safe: grid values should be distinct
    y_frac = (ysca - y_lo) / (y_hi - y_lo)
    # Clamp to [0, 1] for edge extrapolation
    y_frac = jnp.clip(y_frac, 0.0, 1.0)

    # 4. Trilinear interpolation
    # Get 8 corner PSFs (indices already clamped, handles extrapolation)
    psf_000 = psf_grid[wl_idx_lo, y_idx_lo, x_idx_lo]
    psf_001 = psf_grid[wl_idx_lo, y_idx_lo, x_idx_hi]
    psf_010 = psf_grid[wl_idx_lo, y_idx_hi, x_idx_lo]
    psf_011 = psf_grid[wl_idx_lo, y_idx_hi, x_idx_hi]
    psf_100 = psf_grid[wl_idx_hi, y_idx_lo, x_idx_lo]
    psf_101 = psf_grid[wl_idx_hi, y_idx_lo, x_idx_hi]
    psf_110 = psf_grid[wl_idx_hi, y_idx_hi, x_idx_lo]
    psf_111 = psf_grid[wl_idx_hi, y_idx_hi, x_idx_hi]

    # Interpolate along x
    psf_00 = (1 - x_frac) * psf_000 + x_frac * psf_001
    psf_01 = (1 - x_frac) * psf_010 + x_frac * psf_011
    psf_10 = (1 - x_frac) * psf_100 + x_frac * psf_101
    psf_11 = (1 - x_frac) * psf_110 + x_frac * psf_111

    # Interpolate along y
    psf_0 = (1 - y_frac) * psf_00 + y_frac * psf_01
    psf_1 = (1 - y_frac) * psf_10 + y_frac * psf_11

    # Interpolate along wavelength
    psf = (1 - wl_frac) * psf_0 + wl_frac * psf_1

    return psf

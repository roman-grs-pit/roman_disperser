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
- Caching support for fast reload of precomputed PSF grids

Coordinate Systems:
- STPSF uses 0-indexed coordinates, 4096×4096 pixels
- Disperser uses 1-indexed FITS coordinates, 4088×4088 pixels
- Conversion handled by psf_utils.py (with documented assumptions)

Usage:
    >>> import roman_disperser.psf_model as psf_model
    >>> # Generate PSF grid (this is slow, ~5-6 min for default 4×4×56 grid)
    >>> payload = psf_model.make_psf_payload(detector='WFI05', order='1')
    >>> # Interpolate PSF at arbitrary position
    >>> psf = psf_model.interpolate_psf(payload, xsca=2000.0, ysca=2000.0,
    ...                                  wavelength=1.5)  # microns
    >>>
    >>> # With caching (recommended for repeated use):
    >>> payload = psf_model.get_or_make_psf_payload(
    ...     detector='WFI05', order='1', cache_dir='data/psf_cache'
    ... )

Note: All wavelength parameters are in **microns** (consistent with optical model).

See Also:
    docs/stpsf.md : STPSF quick reference (see docs/stpsf_full.md for comprehensive details)
    docs/star_dispersion.md : Design requirements
"""

import time
import warnings
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jax

from .psf_utils import sca_to_stpsf_position, stpsf_to_sca_position


# ============================================================================
# CACHING UTILITIES
# ============================================================================


def get_cache_filename(
    detector,
    order,
    wavelengths,
    spatial_grid,
    fov_arcsec,
    oversample,
):
    """
    Generate a standardized cache filename for a PSF payload.

    The filename encodes key parameters to avoid collisions between different
    configurations.

    Parameters
    ----------
    detector : str
        WFI detector name (e.g., 'WFI05')
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

    Returns
    -------
    filename : str
        Standardized filename (without directory path)

    Examples
    --------
    >>> wavelengths = np.arange(0.9, 2.01, 0.02)  # microns
    >>> spatial_grid = {'x': np.linspace(1, 4088, 4), 'y': np.linspace(1, 4088, 4)}
    >>> get_cache_filename('WFI05', '1', wavelengths, spatial_grid, 5.0, 4)
    'psf_WFI05_GRISM1_4x4x56_0.90-2.00um_fov5.0_os4.npz'
    """
    wavelengths = np.asarray(wavelengths)
    x_grid = np.asarray(spatial_grid['x'])
    y_grid = np.asarray(spatial_grid['y'])

    # Map order to filter name
    filter_map = {'0': 'GRISM0', '1': 'GRISM1'}
    filter_name = filter_map.get(order, f'ORDER{order}')

    # Grid dimensions
    n_y = len(y_grid)
    n_x = len(x_grid)
    n_wl = len(wavelengths)

    # Wavelength bounds in microns (wavelengths already in microns)
    wl_min = wavelengths.min()
    wl_max = wavelengths.max()

    filename = (
        f"psf_{detector}_{filter_name}_{n_y}x{n_x}x{n_wl}_"
        f"{wl_min:.2f}-{wl_max:.2f}um_fov{fov_arcsec:.1f}_os{oversample}.npz"
    )

    return filename


def save_psf_payload(payload, filepath, verbose=True):
    """
    Save a PSF payload to disk for caching.

    Saves the PSF grid and all metadata needed to validate the cache on load.

    Parameters
    ----------
    payload : dict
        PSF payload from make_psf_payload()
    filepath : str or Path
        Path to save the cache file (.npz format)
    verbose : bool, optional
        Print save information (default: True)

    Notes
    -----
    The saved file includes:
    - PSF grid data (as float32 for space efficiency)
    - All grid parameters (wavelengths, spatial grids, fov, oversample)
    - Metadata (detector, order, STPSF version, generation timestamp)

    See Also
    --------
    load_psf_payload : Load cached payload
    get_or_make_psf_payload : Convenience function with auto-caching
    """
    import stpsf

    filepath = Path(filepath)

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    # Convert JAX arrays to numpy for saving
    save_dict = {
        # PSF grid (convert to numpy float32)
        'psf_grid': np.asarray(payload['psf_grid'], dtype=np.float32),
        # Grid parameters
        'wavelengths': np.asarray(payload['wavelengths']),
        'spatial_x': np.asarray(payload['spatial_x']),
        'spatial_y': np.asarray(payload['spatial_y']),
        # Scalar parameters (stored as 0-d arrays)
        'fov_arcsec': np.array(payload.get('fov_arcsec', 5.0)),
        'oversample': np.array(payload['oversample']),
        'pixel_scale': np.array(payload['pixel_scale']),
        # Metadata
        'detector': np.array(payload['detector'], dtype='S'),  # byte string
        'order': np.array(payload['order'], dtype='S'),
        'stpsf_version': np.array(stpsf.__version__, dtype='S'),
        'save_timestamp': np.array(time.time()),
    }

    # Save timing info if available
    if 'timing' in payload:
        save_dict['timing_total_time'] = np.array(payload['timing'].get('total_time', 0))
        save_dict['timing_n_psfs'] = np.array(payload['timing'].get('n_psfs', 0))

    np.savez_compressed(filepath, **save_dict)

    if verbose:
        size_mb = filepath.stat().st_size / 1e6
        print(f"Saved PSF cache: {filepath}")
        print(f"  Size: {size_mb:.1f} MB (compressed)")


def load_psf_payload(filepath, verbose=True):
    """
    Load a cached PSF payload from disk.

    Validates metadata and warns if STPSF version differs from current.

    Parameters
    ----------
    filepath : str or Path
        Path to the cache file (.npz format)
    verbose : bool, optional
        Print load information (default: True)

    Returns
    -------
    payload : dict
        PSF payload compatible with interpolate_psf()

    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    ValueError
        If cache file is corrupted or incompatible

    Notes
    -----
    Warns (but does not error) if the STPSF version used to generate the cache
    differs from the currently installed version.

    See Also
    --------
    save_psf_payload : Save payload to cache
    get_or_make_psf_payload : Convenience function with auto-caching
    """
    import stpsf

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"PSF cache file not found: {filepath}")

    # Load data
    with np.load(filepath, allow_pickle=False) as data:
        # Validate required fields
        required_fields = ['psf_grid', 'wavelengths', 'spatial_x', 'spatial_y',
                           'oversample', 'detector', 'order']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Cache file missing required field: {field}")

        # Helper to decode numpy byte strings
        def decode_bytes(val):
            """Decode numpy byte string to Python str."""
            if hasattr(val, 'tobytes'):
                return val.tobytes().decode('utf-8')
            elif isinstance(val, bytes):
                return val.decode('utf-8')
            return str(val)

        # Check STPSF version
        if 'stpsf_version' in data:
            cached_version = decode_bytes(data['stpsf_version'])
            current_version = stpsf.__version__
            if cached_version != current_version:
                warnings.warn(
                    f"PSF cache was generated with STPSF {cached_version}, "
                    f"but current version is {current_version}. "
                    f"PSF shapes may differ slightly.",
                    UserWarning
                )

        # Reconstruct payload with JAX arrays
        payload = {
            'psf_grid': jnp.array(data['psf_grid'], dtype=jnp.float32),
            'wavelengths': jnp.array(data['wavelengths']),
            'wl_grid': jnp.array(data['wavelengths']),  # Alias
            'spatial_x': jnp.array(data['spatial_x']),
            'spatial_y': jnp.array(data['spatial_y']),
            'psf_fov_pixels': int(data['psf_grid'].shape[-1]),
            'pixel_scale': float(data.get('pixel_scale', 0.11)),
            'oversample': int(data['oversample']),
            'detector': decode_bytes(data['detector']),
            'order': decode_bytes(data['order']),
        }

        # Optional timing info
        if 'timing_total_time' in data:
            payload['timing'] = {
                'total_time': float(data['timing_total_time']),
                'n_psfs': int(data.get('timing_n_psfs', 0)),
                'loaded_from_cache': True,
            }

    if verbose:
        print(f"Loaded PSF cache: {filepath}")
        print(f"  Detector: {payload['detector']}, Order: {payload['order']}")
        print(f"  Grid shape: {payload['psf_grid'].shape}")
        print(f"  Memory: {payload['psf_grid'].nbytes / 1e6:.1f} MB")

    return payload


def _init_worker(counter):
    """Initialize worker process with shared counter."""
    global _progress_counter
    _progress_counter = counter


def _generate_single_cache(args):
    """
    Worker function for parallel cache generation.

    Parameters
    ----------
    args : tuple
        (detector, order, cache_path, wavelengths, spatial_grid, fov_arcsec, oversample)

    Returns
    -------
    result : tuple
        (detector, order, status, message) where status is 'generated', 'skipped', or 'failed'
    """
    import logging
    import warnings
    # Suppress noisy STPSF logging and warnings in worker processes
    logging.getLogger('stpsf').setLevel(logging.ERROR)
    logging.getLogger('poppy').setLevel(logging.ERROR)
    logging.getLogger('webbpsf').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*Attempted to get aberrations.*')
    warnings.filterwarnings('ignore', message='.*outside the range.*')
    warnings.filterwarnings('ignore', message='.*clipping to closest.*')

    detector, order, cache_path, wavelengths, spatial_grid, fov_arcsec, oversample = args

    # Get shared counter if available (set by _init_worker)
    try:
        counter = _progress_counter
    except NameError:
        counter = None

    try:
        payload = _make_psf_payload_with_progress(
            detector=detector,
            order=order,
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=fov_arcsec,
            oversample=oversample,
            progress_counter=counter,
        )
        save_psf_payload(payload, cache_path, verbose=False)
        return (detector, order, 'generated', str(cache_path))
    except Exception as e:
        return (detector, order, 'failed', str(e))


def _make_psf_payload_with_progress(
    detector, order, wavelengths, spatial_grid, fov_arcsec, oversample, progress_counter=None
):
    """
    Internal version of make_psf_payload that updates a shared progress counter.
    """
    import logging
    import stpsf.roman

    # Suppress noisy STPSF messages
    logging.getLogger('stpsf').setLevel(logging.ERROR)
    logging.getLogger('poppy').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*Attempted to get aberrations.*')
    warnings.filterwarnings('ignore', message='.*outside the range.*')

    # Map order to STPSF filter name
    filter_map = {'0': 'GRISM0', '1': 'GRISM1'}
    if order not in filter_map:
        raise ValueError(f"Invalid order '{order}'. Must be '0' or '1'.")

    wfi = stpsf.roman.WFI()
    wfi.filter = filter_map[order]
    wfi.detector = detector

    x_grid = np.asarray(spatial_grid['x'])
    y_grid = np.asarray(spatial_grid['y'])

    # Convert wavelengths from microns to meters for STPSF
    wavelengths_m = wavelengths * 1e-6

    psf_grid = []
    positions = [(ysca, xsca) for ysca in y_grid for xsca in x_grid]

    for ysca, xsca in positions:
        x_stpsf, y_stpsf = sca_to_stpsf_position(float(xsca), float(ysca))
        wfi.detector_position = (float(x_stpsf), float(y_stpsf))

        datacube = wfi.calc_datacube(
            wavelengths_m, fov_arcsec=fov_arcsec, oversample=oversample
        )
        psf_cube = datacube['OVERDIST'].data
        psf_grid.append(psf_cube)

        # Update shared counter
        if progress_counter is not None:
            with progress_counter.get_lock():
                progress_counter.value += 1

    # Reshape to [N_y, N_x, N_wl, PSF_y, PSF_x]
    N_y, N_x = len(y_grid), len(x_grid)
    psf_grid = np.array(psf_grid)
    psf_grid = psf_grid.reshape(N_y, N_x, len(wavelengths), *psf_grid.shape[-2:])

    return {
        'detector': detector,
        'order': order,
        'wavelengths': jnp.array(wavelengths),
        'wl_grid': jnp.array(wavelengths),
        'spatial_x': jnp.array(x_grid),
        'spatial_y': jnp.array(y_grid),
        'psf_grid': jnp.array(psf_grid, dtype=jnp.float32),
        'psf_fov_pixels': psf_grid.shape[-1],
        'pixel_scale': 0.11,
        'oversample': oversample,
        'timing': {},
    }


def generate_all_psf_caches(
    cache_dir,
    orders=('0', '1'),
    detectors=None,
    wavelengths=None,
    spatial_grid=None,
    fov_arcsec=5.0,
    oversample=4,
    skip_existing=True,
    n_workers=1,
    verbose=True,
):
    """
    Generate PSF caches for all detectors and orders.

    This is a convenience function for pre-generating all PSF caches needed
    for full-field validation. With default settings (4×4×56 grid), this
    takes approximately 3-4 hours for all 36 combinations (18 SCAs × 2 orders)
    with a single worker, or ~30-40 minutes with 8 workers.

    Parameters
    ----------
    cache_dir : str or Path
        Directory to store cache files (will be created if needed)
    orders : tuple, optional
        Spectral orders to generate (default: ('0', '1') for GRISM0 and GRISM1)
    detectors : list, optional
        List of detector names (default: all 18 WFI detectors)
    wavelengths : array_like, optional
        Wavelengths in **microns** (default: 0.9-2.0 μm at 0.02 μm spacing)
    spatial_grid : dict, optional
        {'x': x_array, 'y': y_array} (default: 4×4 grid)
    fov_arcsec : float, optional
        PSF field of view in arcseconds (default: 5.0)
    oversample : int, optional
        Oversampling factor (default: 4)
    skip_existing : bool, optional
        Skip detectors/orders that already have cached files (default: True)
    n_workers : int, optional
        Number of parallel workers (default: 1, sequential).
        Each worker uses ~200-300 MB memory. Set to number of CPU cores
        for maximum parallelism, or fewer if memory-constrained.
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    results : dict
        Summary of generation results:
        - 'generated': list of (detector, order) tuples that were generated
        - 'skipped': list of (detector, order) tuples that were skipped
        - 'failed': list of (detector, order, error) tuples that failed
        - 'total_time': total elapsed time in seconds

    Examples
    --------
    >>> # Generate all caches with 4 parallel workers
    >>> results = generate_all_psf_caches('data/psf_cache', n_workers=4)
    >>> print(f"Generated {len(results['generated'])} payloads")

    >>> # Generate only GRISM1 for a subset of detectors
    >>> results = generate_all_psf_caches(
    ...     'data/psf_cache',
    ...     orders=('1',),
    ...     detectors=['WFI01', 'WFI05', 'WFI18'],
    ...     n_workers=3
    ... )
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Default: all 18 WFI detectors
    if detectors is None:
        detectors = [f'WFI{i:02d}' for i in range(1, 19)]

    # Setup defaults (wavelengths in microns)
    if wavelengths is None:
        wavelengths = np.arange(0.9, 2.01, 0.02)
    wavelengths = np.asarray(wavelengths)

    if spatial_grid is None:
        x_grid = np.linspace(1, 4088, 4)
        y_grid = np.linspace(1, 4088, 4)
        spatial_grid = {'x': x_grid, 'y': y_grid}

    # Build list of tasks (detector, order, cache_path)
    tasks = []
    skipped = []

    for detector in detectors:
        for order in orders:
            filename = get_cache_filename(
                detector, order, wavelengths, spatial_grid, fov_arcsec, oversample
            )
            cache_path = cache_dir / filename

            if skip_existing and cache_path.exists():
                skipped.append((detector, order))
            else:
                tasks.append((
                    detector, order, str(cache_path),
                    wavelengths, spatial_grid, fov_arcsec, oversample
                ))

    total_combinations = len(detectors) * len(orders)
    n_to_generate = len(tasks)
    t_start = time.time()

    if verbose:
        print(f"Generating PSF caches for {len(detectors)} detectors × {len(orders)} orders")
        print(f"  Total combinations: {total_combinations}")
        print(f"  To generate: {n_to_generate} (skipping {len(skipped)} existing)")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Workers: {n_workers}")
        if n_workers == 1:
            est_hours = n_to_generate * 5.5 / 60
        else:
            est_hours = n_to_generate * 5.5 / 60 / n_workers
        print(f"  Estimated time: {est_hours:.1f} hours")
        print()

    # Track results
    generated = []
    failed = []

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        have_tqdm = True
    except ImportError:
        have_tqdm = False

    if n_to_generate == 0:
        if verbose:
            print("Nothing to generate (all caches exist)")
    elif n_workers == 1:
        # Sequential execution - show per-position progress
        for i, task in enumerate(tasks):
            detector, order, cache_path = task[0], task[1], task[2]

            if verbose:
                print(f"\n[{i+1}/{n_to_generate}] Generating {detector} order {order}...")

            try:
                # Call make_psf_payload directly with verbose=True to show inner progress
                payload = make_psf_payload(
                    detector=detector,
                    order=order,
                    wavelengths=task[3],
                    spatial_grid=task[4],
                    fov_arcsec=task[5],
                    oversample=task[6],
                    verbose=verbose,  # Show inner progress bar
                )
                save_psf_payload(payload, cache_path, verbose=False)
                generated.append((detector, order))
                if verbose:
                    print(f"  Saved: {Path(cache_path).name}")
            except Exception as e:
                failed.append((detector, order, str(e)))
                if verbose:
                    print(f"  FAILED: {e}")
    else:
        # Parallel execution with shared progress counter
        from multiprocessing import Pool, Value
        import ctypes

        # Calculate total PSF positions across all tasks
        n_positions_per_cache = len(spatial_grid['x']) * len(spatial_grid['y'])
        total_positions = n_to_generate * n_positions_per_cache

        if verbose:
            print(f"Starting {n_workers} parallel workers...")
            print(f"  Total PSF positions to compute: {total_positions}")

        # Create shared counter for progress tracking
        progress_counter = Value(ctypes.c_int, 0)

        # Progress bar thread
        if verbose and have_tqdm:
            pbar = tqdm(total=total_positions, desc="Computing PSFs", unit="pos")

        with Pool(n_workers, initializer=_init_worker, initargs=(progress_counter,)) as pool:
            # Submit all tasks asynchronously
            async_results = {
                pool.apply_async(_generate_single_cache, (task,)): task
                for task in tasks
            }

            # Poll for progress and check for completed results
            completed_tasks = set()
            last_progress = 0

            while len(completed_tasks) < len(async_results):
                # Update progress bar
                if verbose and have_tqdm:
                    current = progress_counter.value
                    if current > last_progress:
                        pbar.update(current - last_progress)
                        last_progress = current

                # Check for completed tasks
                for async_result, task in async_results.items():
                    if async_result not in completed_tasks and async_result.ready():
                        completed_tasks.add(async_result)
                        detector, order = task[0], task[1]

                        try:
                            result = async_result.get()
                            det, ord_, status, msg = result

                            if status == 'generated':
                                generated.append((det, ord_))
                            else:
                                failed.append((det, ord_, msg))
                                if verbose:
                                    if have_tqdm:
                                        tqdm.write(f"FAILED: {det} order {ord_} - {msg}")
                                    else:
                                        print(f"FAILED: {det} order {ord_} - {msg}")
                        except Exception as e:
                            failed.append((detector, order, str(e)))
                            if verbose:
                                if have_tqdm:
                                    tqdm.write(f"FAILED: {detector} order {order} - {e}")
                                else:
                                    print(f"FAILED: {detector} order {order} - {e}")

                time.sleep(0.1)  # Small delay to avoid busy-waiting

        if verbose and have_tqdm:
            # Final update to ensure bar reaches 100%
            current = progress_counter.value
            if current > last_progress:
                pbar.update(current - last_progress)
            pbar.close()

    total_time = time.time() - t_start

    if verbose:
        print()
        print(f"Cache generation complete:")
        print(f"  Generated: {len(generated)}")
        print(f"  Skipped:   {len(skipped)}")
        print(f"  Failed:    {len(failed)}")
        print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
        if n_workers > 1 and len(generated) > 0:
            print(f"  Effective speedup: {len(generated) * 5.5 / (total_time / 60):.1f}x")

    return {
        'generated': generated,
        'skipped': skipped,
        'failed': failed,
        'total_time': total_time,
    }


def get_or_make_psf_payload(
    detector='WFI05',
    order='1',
    wavelengths=None,
    spatial_grid=None,
    fov_arcsec=5.0,
    oversample=4,
    cache_dir=None,
    force_regenerate=False,
    verbose=True,
):
    """
    Get a PSF payload, loading from cache if available or generating if not.

    This is the recommended function for typical usage. It automatically
    handles caching to avoid regenerating PSF grids on every run.

    Parameters
    ----------
    detector : str, optional
        WFI detector name (default: 'WFI05')
    order : str, optional
        Spectral order (default: '1')
    wavelengths : array_like, optional
        Wavelengths in **microns** (default: 0.9-2.0 μm at 0.02 μm spacing)
    spatial_grid : dict, optional
        {'x': x_array, 'y': y_array} in SCA coordinates (default: 4×4 grid)
    fov_arcsec : float, optional
        PSF field of view in arcseconds (default: 5.0)
    oversample : int, optional
        Oversampling factor (default: 4)
    cache_dir : str or Path, optional
        Directory for cache files. If None, caching is disabled and PSF
        grid is always generated fresh.
    force_regenerate : bool, optional
        If True, regenerate even if cache exists (default: False)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    payload : dict
        PSF payload for use with interpolate_psf()

    Examples
    --------
    >>> # First call generates and caches (~5-6 min)
    >>> payload = get_or_make_psf_payload(
    ...     detector='WFI05', order='1', cache_dir='data/psf_cache'
    ... )
    >>> # Second call loads from cache (~1 sec)
    >>> payload = get_or_make_psf_payload(
    ...     detector='WFI05', order='1', cache_dir='data/psf_cache'
    ... )

    See Also
    --------
    make_psf_payload : Always generates fresh (no caching)
    load_psf_payload : Load from specific file
    save_psf_payload : Save to specific file
    """
    # Setup defaults (same as make_psf_payload) - wavelengths in microns
    if wavelengths is None:
        wavelengths = np.arange(0.9, 2.01, 0.02)
    wavelengths = np.asarray(wavelengths)

    if spatial_grid is None:
        x_grid = np.linspace(1, 4088, 4)
        y_grid = np.linspace(1, 4088, 4)
        spatial_grid = {'x': x_grid, 'y': y_grid}

    # If no cache_dir, just generate
    if cache_dir is None:
        if verbose:
            print("No cache_dir specified, generating PSF grid...")
        return make_psf_payload(
            detector=detector,
            order=order,
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=fov_arcsec,
            oversample=oversample,
            verbose=verbose,
        )

    # Generate cache filepath
    cache_dir = Path(cache_dir)
    filename = get_cache_filename(
        detector, order, wavelengths, spatial_grid, fov_arcsec, oversample
    )
    cache_path = cache_dir / filename

    # Try to load from cache
    if cache_path.exists() and not force_regenerate:
        if verbose:
            print(f"Loading PSF payload from cache: {cache_path}")
        try:
            return load_psf_payload(cache_path, verbose=verbose)
        except (ValueError, KeyError) as e:
            if verbose:
                print(f"Cache file invalid ({e}), regenerating...")

    # Generate fresh payload
    if verbose:
        if force_regenerate:
            print("Force regenerate requested, generating PSF grid...")
        else:
            print(f"Cache not found, generating PSF grid...")

    payload = make_psf_payload(
        detector=detector,
        order=order,
        wavelengths=wavelengths,
        spatial_grid=spatial_grid,
        fov_arcsec=fov_arcsec,
        oversample=oversample,
        verbose=verbose,
    )

    # Save to cache
    if verbose:
        print(f"Saving to cache: {cache_path}")
    save_psf_payload(payload, cache_path, verbose=verbose)

    return payload


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
    Expected time: ~5-6 minutes for default 4×4×56 grid (896 PSF calculations)
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
        Wavelengths in **microns** for PSF calculations
        Default: 56 wavelengths from 0.9 to 2.0 μm at 0.02 μm spacing
        Finer wavelength sampling reduces interpolation errors in PSF wings

    spatial_grid : dict, optional
        Spatial grid specification: {'x': x_array, 'y': y_array}
        x_array, y_array in SCA coordinates (1-indexed FITS, range 1-4088)
        Default: 4×4 grid from pixel 1 to 4088 (full detector range)
        4×4 is sufficient for PSF core accuracy; coarser grids degrade core

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
        - 'wavelengths': jnp.ndarray [N_wl], wavelengths in **microns**
        - 'wl_grid': jnp.ndarray [N_wl], same as wavelengths (for consistency)
        - 'spatial_x': jnp.ndarray [N_x], SCA x-coordinates
        - 'spatial_y': jnp.ndarray [N_y], SCA y-coordinates
        - 'psf_grid': jnp.ndarray [N_y, N_x, N_wl, PSF_y, PSF_x], PSF datacube
        - 'psf_fov_pixels': int, PSF array size (pixels)
        - 'pixel_scale': float, detector pixel scale (0.11 arcsec/pixel for WFI)
        - 'oversample': int, oversampling factor used
        - 'timing': dict, generation timing information

    Examples
    --------
    >>> # Generate PSF grid with default settings (first order)
    >>> payload = make_psf_payload(detector='WFI05', order='1')
    >>> print(f"PSF grid shape: {payload['psf_grid'].shape}")
    >>> # Expected: (4, 4, 56, ~182, ~182) for 5" FOV at 4× oversample

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
    # Setup default wavelengths (in microns)
    if wavelengths is None:
        # 56 wavelengths across full grism range (0.9 - 2.0 μm) at 0.02 μm spacing
        # Finer wavelength sampling reduces interpolation errors in PSF wings
        wavelengths = np.arange(0.9, 2.01, 0.02)

    # Validate wavelengths are strictly increasing (required for interpolation)
    wavelengths = np.asarray(wavelengths)
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelengths must be strictly increasing")

    # Setup default spatial grid
    if spatial_grid is None:
        # 4×4 grid across full detector range (1 to 4088)
        # 4×4 is sufficient for PSF core accuracy; STPSF handles edge extrapolation
        x_grid = np.linspace(1, 4088, 4)
        y_grid = np.linspace(1, 4088, 4)
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
        print(f"  This may take 5-6 minutes for default grid...")

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
        Shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
    timing : dict
        {'total_time': float, 'per_psf_time': float, 'n_psfs': int}
    """
    import logging
    import stpsf.roman

    # Suppress noisy STPSF messages
    logging.getLogger('stpsf').setLevel(logging.ERROR)
    logging.getLogger('poppy').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*Attempted to get aberrations.*')
    warnings.filterwarnings('ignore', message='.*outside the range.*')

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

    # Try to use tqdm for progress
    try:
        from tqdm import tqdm
        have_tqdm = True
    except ImportError:
        have_tqdm = False

    psf_grid = []

    # Build list of (ysca, xsca) positions
    positions = [(ysca, xsca) for ysca in y_grid for xsca in x_grid]

    # Wrap with progress bar if verbose
    if verbose and have_tqdm:
        positions = tqdm(
            positions,
            desc=f"{detector} order {order}",
            unit="pos",
            leave=False  # Don't leave the bar after completion
        )

    # Convert wavelengths from microns to meters for STPSF
    wavelengths_m = wavelengths * 1e-6

    for ysca, xsca in positions:
        # Convert SCA to STPSF position
        x_stpsf, y_stpsf = sca_to_stpsf_position(float(xsca), float(ysca))

        # STPSF expects tuple of floats (not JAX arrays)
        wfi.detector_position = (float(x_stpsf), float(y_stpsf))

        # Calculate datacube at this position
        # CRITICAL: Use OVERDIST extension for sub-pixel accuracy + detector effects
        # Note: add_distortion is no longer needed - all WFI PSFs natively include distortion
        datacube = wfi.calc_datacube(
            wavelengths_m, fov_arcsec=fov_arcsec, oversample=oversample
        )
        # Use OVERDIST: oversampled + detector effects (distortion, diffusion)
        psf_cube = datacube['OVERDIST'].data  # [N_wl, PSF_y, PSF_x]

        psf_grid.append(psf_cube)

    # Reshape to [N_y, N_x, N_wl, PSF_y, PSF_x]
    # This ordering puts wavelength contiguous with PSF data, which is efficient for:
    # - Undispersed case: bilinear spatial interp → slice all wavelengths at once
    # - Dispersed case: trilinear still works, no penalty
    psf_grid = np.array(psf_grid)  # [N_y*N_x, N_wl, PSF_y, PSF_x]
    psf_grid = psf_grid.reshape(N_y, N_x, N_wl, *psf_grid.shape[-2:])

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
        Wavelength in **microns** (range 1.0 to 1.93 for grism)
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
    ...                        wavelength=1.5)  # microns
    >>> psf.shape
    (108, 108)  # For 3" FOV at 4× oversample

    >>> # Vectorized: PSFs for multiple positions
    >>> import jax.numpy as jnp
    >>> xsca = jnp.array([1000.0, 2000.0, 3000.0])
    >>> ysca = jnp.array([1000.0, 2000.0, 3000.0])
    >>> wavelength = jnp.array([1.0, 1.5, 1.9])  # microns
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
    psf_grid = payload['psf_grid']  # [N_y, N_x, N_wl, PSF_y, PSF_x]

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
    # Grid shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
    # Naming: psf_YXW where Y=y_idx, X=x_idx, W=wl_idx (0=lo, 1=hi)
    psf_000 = psf_grid[y_idx_lo, x_idx_lo, wl_idx_lo]
    psf_001 = psf_grid[y_idx_lo, x_idx_lo, wl_idx_hi]
    psf_010 = psf_grid[y_idx_lo, x_idx_hi, wl_idx_lo]
    psf_011 = psf_grid[y_idx_lo, x_idx_hi, wl_idx_hi]
    psf_100 = psf_grid[y_idx_hi, x_idx_lo, wl_idx_lo]
    psf_101 = psf_grid[y_idx_hi, x_idx_lo, wl_idx_hi]
    psf_110 = psf_grid[y_idx_hi, x_idx_hi, wl_idx_lo]
    psf_111 = psf_grid[y_idx_hi, x_idx_hi, wl_idx_hi]

    # Interpolate along wavelength (W dimension, index 2)
    psf_00 = (1 - wl_frac) * psf_000 + wl_frac * psf_001  # Y=lo, X=lo
    psf_01 = (1 - wl_frac) * psf_010 + wl_frac * psf_011  # Y=lo, X=hi
    psf_10 = (1 - wl_frac) * psf_100 + wl_frac * psf_101  # Y=hi, X=lo
    psf_11 = (1 - wl_frac) * psf_110 + wl_frac * psf_111  # Y=hi, X=hi

    # Interpolate along x (X dimension, index 1)
    psf_0 = (1 - x_frac) * psf_00 + x_frac * psf_01  # Y=lo
    psf_1 = (1 - x_frac) * psf_10 + x_frac * psf_11  # Y=hi

    # Interpolate along y (Y dimension, index 0)
    psf = (1 - y_frac) * psf_0 + y_frac * psf_1

    return psf


def interpolate_psf_wavelength(psfs, wl_grid, wavelengths):
    """
    Interpolate PSFs to user-specified wavelengths using linear interpolation.

    This function takes PSFs at grid wavelengths (e.g., from interpolate_psf_spatial)
    and interpolates them to arbitrary user-specified wavelengths.

    Uses edge extrapolation: wavelengths outside the grid use the nearest edge PSF
    (i.e., clamped to grid bounds).

    This function is JAX-compatible and JIT-compilable.

    Parameters
    ----------
    psfs : jnp.ndarray
        PSF array at grid wavelengths
        Shape: [N_wl_grid, PSF_y, PSF_x]
    wl_grid : jnp.ndarray
        Wavelengths in the grid (**microns**), shape [N_wl_grid]
        Must be strictly increasing
    wavelengths : jnp.ndarray
        Target wavelengths (**microns**), shape [N_wl_user]
        Can be any order, but should be within or near grid bounds

    Returns
    -------
    psfs_interp : jnp.ndarray
        Interpolated PSFs at target wavelengths
        Shape: [N_wl_user, PSF_y, PSF_x]

    Examples
    --------
    >>> # Get PSFs at grid wavelengths
    >>> psfs_grid = interpolate_psf_spatial(payload, xsca=2000.0, ysca=2000.0)
    >>> # Interpolate to specific wavelengths (microns)
    >>> wavelengths = jnp.array([1.0, 1.25, 1.5, 1.75])  # microns
    >>> psfs_user = interpolate_psf_wavelength(psfs_grid, payload['wavelengths'], wavelengths)
    >>> psfs_user.shape
    (4, 182, 182)

    Notes
    -----
    - Linear interpolation along wavelength dimension
    - Edge extrapolation: uses nearest grid PSF for out-of-bounds wavelengths
    - For single wavelength, consider using interpolate_psf() directly
    - Designed for use in star dispersion where PSFs are needed at many wavelengths

    See Also
    --------
    interpolate_psf_spatial : Get PSFs at grid wavelengths for a spatial position
    interpolate_psf : Trilinear interpolation at arbitrary (x, y, λ)
    """
    # Find wavelength brackets
    wl_idx = jnp.searchsorted(wl_grid, wavelengths)  # Index of next wavelength
    wl_idx = jnp.clip(wl_idx, 1, len(wl_grid) - 1)  # Ensure in bounds

    wl_idx_lo = wl_idx - 1
    wl_idx_hi = wl_idx

    wl_lo = wl_grid[wl_idx_lo]
    wl_hi = wl_grid[wl_idx_hi]

    # Interpolation fraction
    wl_frac = (wavelengths - wl_lo) / (wl_hi - wl_lo)
    # Clamp to [0, 1] for edge extrapolation
    wl_frac = jnp.clip(wl_frac, 0.0, 1.0)

    # Get PSFs at bracketing wavelengths
    # psfs shape: [N_wl_grid, PSF_y, PSF_x]
    psfs_lo = psfs[wl_idx_lo]  # [N_wl_user, PSF_y, PSF_x]
    psfs_hi = psfs[wl_idx_hi]  # [N_wl_user, PSF_y, PSF_x]

    # Linear interpolation
    # wl_frac has shape [N_wl_user], need to broadcast to [N_wl_user, 1, 1]
    wl_frac = wl_frac[:, jnp.newaxis, jnp.newaxis]
    psfs_interp = (1 - wl_frac) * psfs_lo + wl_frac * psfs_hi

    return psfs_interp


def interpolate_psf_spatial(payload, xsca, ysca):
    """
    Interpolate PSF at (x, y) for ALL wavelengths using bilinear interpolation.

    This is more efficient than calling interpolate_psf() for each wavelength
    when the spatial position is fixed (undispersed star case). Returns the
    full wavelength stack at once.

    Uses edge extrapolation: positions outside the spatial grid use the nearest
    edge PSF.

    This function is JAX-compatible and JIT-compilable.

    Parameters
    ----------
    payload : dict
        PSF payload from make_psf_payload()
    xsca, ysca : float
        SCA coordinates (1-indexed FITS, range 1-4088)
        Must be scalar values (not arrays)

    Returns
    -------
    psfs : jnp.ndarray
        Interpolated PSF array for all wavelengths in the grid
        Shape: [N_wl, PSF_y, PSF_x]

    Examples
    --------
    >>> # Get PSFs at detector center for all wavelengths
    >>> psfs = interpolate_psf_spatial(payload, xsca=2044.0, ysca=2044.0)
    >>> psfs.shape
    (56, 182, 182)  # For default 56-wavelength grid, 5" FOV at 4× oversample

    >>> # Access PSF at specific wavelength index
    >>> psf_at_wl_10 = psfs[10]  # 10th wavelength in grid

    Notes
    -----
    - Interpolation is bilinear in (x, y) only - no wavelength interpolation
    - Returns PSFs at grid wavelengths (payload['wavelengths'])
    - For wavelength interpolation, use interpolate_psf() instead
    - Edge extrapolation: uses nearest grid PSF for out-of-bounds positions
    - Efficient for undispersed star dispersion where spatial position is fixed

    See Also
    --------
    interpolate_psf : Trilinear interpolation at arbitrary (x, y, λ)
    make_psf_payload : Create PSF payload
    """
    # Extract grid parameters
    x_grid = payload['spatial_x']
    y_grid = payload['spatial_y']
    psf_grid = payload['psf_grid']  # [N_y, N_x, N_wl, PSF_y, PSF_x]

    # 1. Find spatial bracket (x dimension)
    x_idx = jnp.searchsorted(x_grid, xsca)
    x_idx = jnp.clip(x_idx, 1, len(x_grid) - 1)

    x_idx_lo = x_idx - 1
    x_idx_hi = x_idx

    x_lo = x_grid[x_idx_lo]
    x_hi = x_grid[x_idx_hi]
    x_frac = (xsca - x_lo) / (x_hi - x_lo)
    x_frac = jnp.clip(x_frac, 0.0, 1.0)

    # 2. Find spatial bracket (y dimension)
    y_idx = jnp.searchsorted(y_grid, ysca)
    y_idx = jnp.clip(y_idx, 1, len(y_grid) - 1)

    y_idx_lo = y_idx - 1
    y_idx_hi = y_idx

    y_lo = y_grid[y_idx_lo]
    y_hi = y_grid[y_idx_hi]
    y_frac = (ysca - y_lo) / (y_hi - y_lo)
    y_frac = jnp.clip(y_frac, 0.0, 1.0)

    # 3. Bilinear interpolation over spatial dimensions
    # Get 4 corner PSF stacks (all wavelengths at once)
    # Grid shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
    psf_00 = psf_grid[y_idx_lo, x_idx_lo]  # [N_wl, PSF_y, PSF_x]
    psf_01 = psf_grid[y_idx_lo, x_idx_hi]  # [N_wl, PSF_y, PSF_x]
    psf_10 = psf_grid[y_idx_hi, x_idx_lo]  # [N_wl, PSF_y, PSF_x]
    psf_11 = psf_grid[y_idx_hi, x_idx_hi]  # [N_wl, PSF_y, PSF_x]

    # Interpolate along x
    psf_0 = (1 - x_frac) * psf_00 + x_frac * psf_01  # Y=lo
    psf_1 = (1 - x_frac) * psf_10 + x_frac * psf_11  # Y=hi

    # Interpolate along y
    psfs = (1 - y_frac) * psf_0 + y_frac * psf_1

    return psfs

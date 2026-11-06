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
    ... )  # exact file or PSFCacheNotFoundError; see scripts/generate_psf_caches.py

Note: All wavelength parameters are in **microns** (consistent with optical model).

See Also:
    docs/stpsf.md : STPSF quick reference (see docs/reference/stpsf_full.md for comprehensive details)
    docs/star_dispersion.md : Design requirements
"""

import importlib.metadata
import os
import re
import shlex
import time
import warnings
import zipfile
import zlib
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jax

from .psf_utils import sca_to_stpsf_position, stpsf_to_sca_position

# Default order -> STPSF filter map. The grism is the package's default
# dispersing element; other elements (prism) pass ``stpsf_filter``
# explicitly — the per-element mapping lives on
# ``roman_disperser.elements.DispersingElement.stpsf_filters``.
GRISM_FILTERS = {'0': 'GRISM0', '1': 'GRISM1'}

# The STPSF version PSF caches are made with. Generation refuses anything
# older, and lookups prefer the file carrying this version in its name.
TARGET_STPSF_VERSION = "2.3.0"

# First STPSF whose OVERDIST extension has IPC on by default. Fixed history,
# independent of the target: a file stamped older than this has no IPC in it.
IPC_FIRST_STPSF_VERSION = "2.3.0"

_REGENERATE_ADVICE = "Regenerate with scripts/generate_psf_caches.py."

# Detector range for PSF nodes, in FITS SCA coordinates. A filename without a
# position tag means a uniform grid over this range.
FULL_DETECTOR_BOUNDS = (1.0, 4088.0, 1.0, 4088.0)  # XMIN, XMAX, YMIN, YMAX

# Default PSF node grid: uniform inside the area where STPSF has aberrations.
# GRISM0, GRISM1 and PRISM take theirs from 5 field points per SCA, and outside
# those points' hull STPSF's fallback jumps from place to place. This rectangle
# is inside the hull on all 54 SCA and element pairs. A position outside it uses
# the nearest edge of the grid.
DEFAULT_PSF_GRID = (4, 4)  # (NX, NY)
DEFAULT_PSF_GRID_BOUNDS = (50.0, 4050.0, 202.0, 4088.0)

# A stored float32 position at 4088 has a spacing of 5e-4 px.
_NODE_TOLERANCE_PX = 5e-3
# A stored float32 wavelength at 2 um has a spacing of 2.4e-7 um.
_WAVELENGTH_TOLERANCE_UM = 1e-5

# TEMPORARY: remove once the 54 caches are regenerated on the cluster
ALLOW_FALLBACK_ENV = "ROMAN_DISPERSER_ALLOW_PSF_FALLBACK"


class PSFCacheVersionWarning(UserWarning):
    """A PSF cache's STPSF version or IPC state is stale, unknown or inconsistent."""


class PSFCacheNotFoundError(FileNotFoundError):
    """The exact PSF cache for a request is missing or is not what its name says."""


def parse_psf_grid(text):
    """``'5x4'`` -> ``(5, 4)``, meaning (NX, NY). Accepts a pair as well."""
    if isinstance(text, str):
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
        if m is None:
            raise ValueError(f"PSF grid {text!r} is not of the form NXxNY, e.g. 5x5")
        nx, ny = int(m.group(1)), int(m.group(2))
    else:
        nx, ny = (int(n) for n in text)
    if nx < 2 or ny < 2:
        raise ValueError(f"PSF grid needs at least 2 nodes per axis, got {nx}x{ny}")
    return nx, ny


def parse_psf_grid_bounds(text):
    """``'16,4055,187,4088'`` -> XMIN, XMAX, YMIN, YMAX; validated. Accepts a sequence."""
    parts = text.split(",") if isinstance(text, str) else text
    try:
        bounds = tuple(float(v) for v in parts)
    except (TypeError, ValueError):
        raise ValueError(f"PSF grid bounds {text!r} are not numbers") from None
    if len(bounds) != 4:
        raise ValueError(
            f"PSF grid bounds {text!r} are not XMIN,XMAX,YMIN,YMAX")
    xmin, xmax, ymin, ymax = bounds
    low, high = FULL_DETECTOR_BOUNDS[:2]
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"PSF grid bounds {text!r} are not ordered: need "
                         f"XMIN < XMAX and YMIN < YMAX")
    if min(bounds) < low or max(bounds) > high:
        raise ValueError(f"PSF grid bounds {text!r} leave the detector: FITS SCA "
                         f"coordinates run from {low:g} to {high:g}")
    return bounds


def make_spatial_grid(psf_grid=None, psf_grid_bounds=None):
    """Uniform node positions ``{'x', 'y'}`` from node counts and bounds.

    Parameters
    ----------
    psf_grid : str or pair, optional
        ``'NXxNY'`` or ``(NX, NY)``. Default: 4x4.
    psf_grid_bounds : str or sequence, optional
        XMIN, XMAX, YMIN, YMAX in FITS SCA coordinates. Default:
        ``DEFAULT_PSF_GRID_BOUNDS``.
    """
    nx, ny = parse_psf_grid(DEFAULT_PSF_GRID if psf_grid is None else psf_grid)
    xmin, xmax, ymin, ymax = parse_psf_grid_bounds(
        DEFAULT_PSF_GRID_BOUNDS if psf_grid_bounds is None else psf_grid_bounds)
    return {'x': np.linspace(xmin, xmax, nx), 'y': np.linspace(ymin, ymax, ny)}


# Messages already shown in this process. A run reloads the same files (warm-up,
# then once per SCA and pointing), so each distinct message appears once. The
# message names the file, so a different file always gets its own warning, under
# Python's default filter as well (it dedupes on the text).
_warned_cache_messages = set()


def _reset_cache_warnings():
    """Forget which PSF cache warnings were shown (for tests)."""
    _warned_cache_messages.clear()


def _warn_once(message, category=PSFCacheVersionWarning, stacklevel=3):
    if message not in _warned_cache_messages:
        warnings.warn(message, category, stacklevel=stacklevel)
        _warned_cache_messages.add(message)


def _parse_version(version):
    """``'2.3.0'`` -> ``(2, 3, 0, 0)``; None if there is no leading X.Y.

    The last element orders pre-releases (``2.3.0rc1``, ``2.3.0.dev4``) before
    the release. Local and post suffixes count as the release.
    """
    m = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?(.*)$", str(version or ""))
    if m is None:
        return None
    pre = re.match(r"^[.\-_]?(a|b|c|rc|alpha|beta|dev|pre)", m.group(4))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0),
            -1 if pre else 0)


def _installed_stpsf_version():
    """Installed STPSF version, or None, without importing it.

    Importing stpsf is slow and, with STPSF_PATH unset and no data at the
    default location, downloads its data package to ~/data/stpsf-data. Reading
    a cache must not do that.
    """
    try:
        return importlib.metadata.version("stpsf")
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_stpsf_for_generation():
    """Return the installed STPSF version, or raise if it cannot make a cache."""
    installed = _installed_stpsf_version()
    if installed is None:
        raise ModuleNotFoundError(
            f"PSF generation needs STPSF >= {TARGET_STPSF_VERSION}, which is "
            f"not installed. See INSTALL.md.")
    parsed = _parse_version(installed)
    target = _parse_version(TARGET_STPSF_VERSION)
    if parsed is None or parsed < target:
        raise RuntimeError(
            f"PSF generation needs STPSF >= {TARGET_STPSF_VERSION}, but "
            f"{installed} is installed. Upgrade STPSF and its data package "
            f"(see INSTALL.md); no cache was written.")
    if not os.environ.get("STPSF_PATH"):
        raise RuntimeError(
            "STPSF_PATH is not set. Point it at the unpacked "
            f"stpsf-data-{TARGET_STPSF_VERSION} package (see INSTALL.md). "
            "Without it STPSF falls back to ~/data/stpsf-data and downloads "
            "its latest data there.")
    if parsed > target:
        warnings.warn(
            f"Generating PSFs with STPSF {installed}, newer than the target "
            f"{TARGET_STPSF_VERSION}. The file is labelled {installed}, and "
            f"runs will not use it until psf_model.TARGET_STPSF_VERSION is "
            f"bumped to that version.",
            RuntimeWarning, stacklevel=2)
    return installed


def _filename_version():
    """Version for the name of a file generated here: never a false label."""
    installed = _parse_version(_installed_stpsf_version())
    if installed is not None and installed > _parse_version(TARGET_STPSF_VERSION):
        return _installed_stpsf_version()
    return TARGET_STPSF_VERSION


def _make_wfi(detector, order, stpsf_filter, add_ipc):
    """STPSF WFI instrument set up for one cache."""
    import stpsf.roman

    wfi = stpsf.roman.WFI()
    wfi.filter = resolve_stpsf_filter(order, stpsf_filter)
    wfi.detector = detector
    # STPSF >= 2.3.0 defaults to True. romanisim >= 0.16.0 applies IPC in
    # make_l1, so a default cache must not contain it. The pinned v0.15.0
    # applies none because of a bug (romanisim #383), so wrapped products have
    # no IPC until romanisim is upgraded.
    wfi.options['add_ipc'] = bool(add_ipc)
    _use_detector_hf_map(wfi, detector)
    return wfi


def _hf_map_name(detector):
    """Name of a detector's high-frequency wavefront map in the STPSF data."""
    return f"RST_WFI{int(str(detector)[-2:])}_Z210_high_frequency_map.fits"


def _use_detector_hf_map(wfi, detector):
    """Give STPSF the detector's own high-frequency wavefront map.

    STPSF 2.3.0 keeps WFI01's map on every detector, although its detector setter
    says it updates it (stpsf/roman.py:804). Refuse to go on if STPSF has changed
    so that this no longer gives the right map.
    """
    if hasattr(wfi, '_opd_file_dict'):
        wfi._pupilopd = wfi._opd_file_dict[detector]
    if hasattr(wfi, 'pupilopd'):
        used = os.path.basename(str(wfi.pupilopd))
        if used != _hf_map_name(detector):
            raise RuntimeError(f"STPSF would use the high-frequency map {used} for "
                               f"{detector}, not {_hf_map_name(detector)}.")


def resolve_stpsf_filter(order, stpsf_filter=None):
    """Return the STPSF filter name to use for a spectral order.

    An explicit ``stpsf_filter`` always wins (this is how non-grism elements
    reach the PSF machinery). With none given, the grism defaults apply, and
    an order outside them raises rather than guessing — a wrong filter here
    silently produces a wrong PSF cache.
    """
    if stpsf_filter is not None:
        return stpsf_filter
    if order not in GRISM_FILTERS:
        raise ValueError(
            f"No default STPSF filter for order {order!r} (grism defaults "
            f"cover {sorted(GRISM_FILTERS)}). Pass stpsf_filter explicitly, "
            f"e.g. element.stpsf_filters[order]."
        )
    return GRISM_FILTERS[order]


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
    stpsf_filter=None,
    stpsf_version=None,
    add_ipc=False,
):
    """
    Generate a standardized cache filename for a PSF payload.

    The filename encodes key parameters to avoid collisions between different
    configurations. Node positions appear whenever they are not uniform over
    the full detector, as with the default grid:
    ``..._4x4x56_x50-4050_y202-4088_0.90-2.00um_...``.

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
    stpsf_filter : str, optional
        STPSF filter name (default: the grism mapping for ``order``)
    stpsf_version : str, optional
        STPSF version for the name (default: ``TARGET_STPSF_VERSION``)
    add_ipc : bool, optional
        Whether the PSFs contain IPC; adds ``_ipc`` to the name (default: False)

    Returns
    -------
    filename : str
        Standardized filename (without directory path)

    Examples
    --------
    >>> wavelengths = np.arange(0.9, 2.01, 0.02)  # microns
    >>> spatial_grid = {'x': np.linspace(1, 4088, 4), 'y': np.linspace(1, 4088, 4)}
    >>> get_cache_filename('WFI05', '1', wavelengths, spatial_grid, 5.0, 4)
    'psf_WFI05_GRISM1_4x4x56_0.90-2.00um_fov5.0_os4_stpsf2.3.0.npz'
    """
    stem = _cache_stem(detector, order, wavelengths, spatial_grid, fov_arcsec,
                       oversample, stpsf_filter)
    version = stpsf_version or TARGET_STPSF_VERSION
    return f"{stem}_stpsf{version}{'_ipc' if add_ipc else ''}.npz"


def _cache_stem(detector, order, wavelengths, spatial_grid, fov_arcsec,
                oversample, stpsf_filter=None):
    """Filename part naming the configuration; alone, it is the legacy name."""
    wavelengths = np.asarray(wavelengths)
    x_grid = np.asarray(spatial_grid['x'])
    y_grid = np.asarray(spatial_grid['y'])

    filter_name = resolve_stpsf_filter(order, stpsf_filter)

    # Grid dimensions
    n_y = len(y_grid)
    n_x = len(x_grid)
    n_wl = len(wavelengths)

    # Wavelength bounds in microns (wavelengths already in microns)
    wl_min = wavelengths.min()
    wl_max = wavelengths.max()

    return (
        f"psf_{detector}_{filter_name}_{n_y}x{n_x}x{n_wl}_"
        f"{_grid_tag(x_grid, y_grid)}"
        f"{wl_min:.2f}-{wl_max:.2f}um_fov{fov_arcsec:.1f}_os{oversample}"
    )


def _format_position(value):
    """Position for a filename: 2 decimals, no trailing zeros."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _is_uniform(values):
    steps = np.diff(values)
    return len(steps) < 2 or bool(np.allclose(steps, steps[0], atol=0.01, rtol=0))


def _grid_tag(x_grid, y_grid):
    """Filename part for node positions; empty for uniform over the full detector.

    Bounds and count fix a uniform grid. A non-uniform one adds a checksum of its
    positions, since its bounds and count do not fix it. Positions closer than the
    rounding share a name; the check of stored positions on load catches that.
    """
    tag = (f"x{_format_position(x_grid[0])}-{_format_position(x_grid[-1])}_"
           f"y{_format_position(y_grid[0])}-{_format_position(y_grid[-1])}")
    uniform = _is_uniform(x_grid) and _is_uniform(y_grid)
    low, high = FULL_DETECTOR_BOUNDS[:2]
    default = (f"x{low:g}-{high:g}_y{low:g}-{high:g}")
    if uniform and tag == default:
        return ""
    if not uniform:
        rounded = np.round(np.concatenate([x_grid, y_grid]), 2).astype(np.float64)
        tag += f"_nu{zlib.crc32(rounded.tobytes()):08x}"
    return tag + "_"


_CACHE_NAME_RE = re.compile(
    r"^(?P<stem>psf_.+_os\d+)(?:_stpsf(?P<version>[^_]+))?(?P<ipc>_ipc)?\.npz$")


def _parse_cache_filename(name):
    """Split a cache filename into ``(stem, version, ipc)``.

    ``stem`` is None for a name outside the standard form.
    """
    m = _CACHE_NAME_RE.match(name)
    if m is None:
        return None, None, False
    return m.group("stem"), m.group("version"), m.group("ipc") is not None


def _decode_label(val):
    """Decode a numpy byte-string scalar to a Python str."""
    if hasattr(val, 'item'):
        val = val.item()
    return val.decode('utf-8') if isinstance(val, bytes) else str(val)


def _header_from_npz(data):
    """Version and IPC metadata of an open cache; None where it is absent."""
    version = None
    if 'stpsf_version' in data:
        version = _decode_label(data['stpsf_version'])
        if version in ('', 'unknown'):
            version = None
    return {
        'version': version,
        'add_ipc': bool(data['add_ipc']) if 'add_ipc' in data else None,
        'psf_source': (_decode_label(data['psf_source'])
                       if 'psf_source' in data else 'stpsf'),
        'hf_map': (_decode_label(data['stpsf_hf_map'])
                   if 'stpsf_hf_map' in data else None),
        'detector': (_decode_label(data['detector'])
                     if 'detector' in data else None),
    }


def _read_cache_header(path):
    """Metadata of a cache file without reading its PSF grid; None if unreadable."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return _header_from_npz(data)
    except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile):
        return None


def _read_cache_config(path):
    """What a cache file says about itself; None if it cannot be read."""
    required = ('psf_grid', 'wavelengths', 'spatial_x', 'spatial_y', 'oversample',
                'detector', 'order')
    try:
        with np.load(path, allow_pickle=False) as data:
            config = _header_from_npz(data)
            config['missing'] = [k for k in required if k not in data]
            if config['missing']:
                return config
            config['spatial_x'] = np.asarray(data['spatial_x'], dtype=np.float64)
            config['spatial_y'] = np.asarray(data['spatial_y'], dtype=np.float64)
            config['wavelengths'] = np.asarray(data['wavelengths'], dtype=np.float64)
            config['oversample'] = int(data['oversample'])
            config['fov_arcsec'] = (float(data['fov_arcsec'])
                                    if 'fov_arcsec' in data else None)
            config['detector'] = _decode_label(data['detector'])
            config['stpsf_filter'] = (_decode_label(data['stpsf_filter'])
                                      if 'stpsf_filter' in data else 'unknown')
            return config
    except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile):
        return None


def _same_positions(stored, requested):
    requested = np.asarray(requested, dtype=np.float64)
    return (stored.shape == requested.shape
            and bool(np.allclose(stored, requested, atol=_NODE_TOLERANCE_PX, rtol=0)))


def _config_mismatches(config, request):
    """How a file's stored configuration differs from a request."""
    problems = []
    if config['missing']:
        return [f"it lacks the fields {config['missing']}"]
    if not _same_positions(config['spatial_x'], request['spatial_grid']['x']):
        problems.append(f"its stored x nodes are {config['spatial_x'].tolist()}, "
                        f"not {np.asarray(request['spatial_grid']['x']).tolist()}")
    if not _same_positions(config['spatial_y'], request['spatial_grid']['y']):
        problems.append(f"its stored y nodes are {config['spatial_y'].tolist()}, "
                        f"not {np.asarray(request['spatial_grid']['y']).tolist()}")
    requested_wl = np.asarray(request['wavelengths'], dtype=np.float64)
    if (config['wavelengths'].shape != requested_wl.shape
            or not np.allclose(config['wavelengths'], requested_wl,
                               atol=_WAVELENGTH_TOLERANCE_UM, rtol=0)):
        problems.append("its stored wavelengths differ from the request")
    if config['oversample'] != request['oversample']:
        problems.append(f"its stored oversample is {config['oversample']}")
    if (config['fov_arcsec'] is not None
            and abs(config['fov_arcsec'] - request['fov_arcsec']) > 1e-3):
        problems.append(f"its stored fov_arcsec is {config['fov_arcsec']:g}")
    if config['detector'] != request['detector']:
        problems.append(f"its stored detector is {config['detector']}")
    if config['stpsf_filter'] not in ('unknown', request['stpsf_filter']):
        problems.append(f"its stored STPSF filter is {config['stpsf_filter']}")
    return problems


def _exact_file_problems(path, request):
    """Why an exact-named file is not the right file; empty if it is."""
    config = _read_cache_config(path)
    if config is None:
        return ["it cannot be read as a PSF cache"]
    problems = _config_mismatches(config, request)
    _, name_version, name_ipc = _parse_cache_filename(path.name)
    if config['version'] is None:
        problems.append("its metadata has no STPSF version")
    elif _parse_version(config['version']) != _parse_version(name_version):
        problems.append(f"its filename says STPSF {name_version} but its metadata "
                        f"says {config['version']}")
    if config['add_ipc'] is None:
        problems.append("it has no add_ipc metadata")
    elif config['add_ipc'] != name_ipc:
        problems.append(f"its filename says IPC {'on' if name_ipc else 'off'} but "
                        f"its metadata says add_ipc={config['add_ipc']}")
    return problems


def _generation_command(request, cache_path, exists):
    """The command that makes the requested file, or how to do it if none can."""
    from roman_disperser.elements import GRISM, PRISM, psf_cache_wavelengths

    x_grid = np.asarray(request['spatial_grid']['x'], dtype=np.float64)
    y_grid = np.asarray(request['spatial_grid']['y'], dtype=np.float64)
    wavelengths = np.asarray(request['wavelengths'], dtype=np.float64)
    for element in (GRISM, PRISM):
        orders = [o for o, f in element.stpsf_filters.items()
                  if f == request['stpsf_filter']]
        if orders:
            break
    else:
        orders, element = [], None
    if element is not None:
        band = np.asarray(psf_cache_wavelengths(element), dtype=np.float64)
    if (element is None or band.shape != wavelengths.shape
            or not np.allclose(band, wavelengths, atol=_WAVELENGTH_TOLERANCE_UM, rtol=0)
            or request['fov_arcsec'] != 5.0 or request['oversample'] != 4
            or not (_is_uniform(x_grid) and _is_uniform(y_grid))):
        return ("psf_model.generate_all_psf_caches(...) with this configuration; "
                "scripts/generate_psf_caches.py makes only uniform grids of the "
                "element's band at fov 5.0 and oversample 4")
    command = ["python", "scripts/generate_psf_caches.py",
               "--cache-dir", shlex.quote(str(cache_path.parent)),
               "--element", element.name.lower(),
               "--detectors", request['detector'], "--orders", orders[0]]
    if (len(x_grid), len(y_grid)) != DEFAULT_PSF_GRID:
        command += ["--psf-grid", f"{len(x_grid)}x{len(y_grid)}"]
    bounds = (x_grid[0], x_grid[-1], y_grid[0], y_grid[-1])
    if not np.allclose(bounds, DEFAULT_PSF_GRID_BOUNDS, atol=_NODE_TOLERANCE_PX, rtol=0):
        command += ["--psf-grid-bounds", ",".join(f"{v:.10g}" for v in bounds)]
    if request['add_ipc']:
        command.append("--add-ipc")
    if exists:
        command.append("--no-skip")
    return " ".join(command)


def _describe_other_files(cache_dir, request, exclude):
    """One line per other cache of the same detector and filter."""
    prefix = f"psf_{request['detector']}_{request['stpsf_filter']}_"
    lines = []
    for path in sorted(cache_dir.glob(f"{prefix}*.npz")) if cache_dir.is_dir() else []:
        if path.name == exclude:
            continue
        config = _read_cache_config(path)
        if config is None or config['missing']:
            lines.append(f"    {path.name} (not readable as a PSF cache)")
            continue
        version = config['version'] or 'none recorded'
        ipc = 'unknown' if config['add_ipc'] is None else config['add_ipc']
        lines.append(f"    {path.name} (STPSF {version}, add_ipc {ipc}, "
                     f"{config['spatial_x'].size}x{config['spatial_y'].size} nodes)")
    return lines


def _not_found_error(request, cache_path, problems=None):
    """Error naming the missing file, the command for it, and what is there."""
    exists = cache_path.exists()
    lines = [f"PSF cache not usable: {cache_path}"]
    if problems:
        lines.append("  The file exists, but it is not the right file: "
                     + "; ".join(problems) + ".")
    else:
        lines.append("  The file does not exist. No other file is used in its "
                     "place, and nothing is generated.")
    lines.append("  Generate it with STPSF "
                 f"{TARGET_STPSF_VERSION} and STPSF_PATH set (see INSTALL.md):")
    lines.append("    " + _generation_command(request, cache_path, exists))
    others = _describe_other_files(cache_path.parent, request, cache_path.name)
    if others:
        lines.append(f"  Other files for {request['detector']} "
                     f"{request['stpsf_filter']} in this directory, not used:")
        lines.extend(others)
        # TEMPORARY: remove once the 54 caches are regenerated on the cluster
        lines.append(f"  To use the best file of the same configuration for now, "
                     f"with a warning, set {ALLOW_FALLBACK_ENV}=1.")
    return PSFCacheNotFoundError("\n".join(lines))


def _cache_request(detector, stpsf_filter, wavelengths, spatial_grid, fov_arcsec,
                   oversample, add_ipc):
    return dict(detector=detector, stpsf_filter=stpsf_filter,
                wavelengths=np.asarray(wavelengths), spatial_grid=spatial_grid,
                fov_arcsec=fov_arcsec, oversample=oversample, add_ipc=bool(add_ipc))


def _full_detector_request(request):
    """A request on the default bounds restated over the full detector, or None."""
    x_grid, y_grid = request['spatial_grid']['x'], request['spatial_grid']['y']
    if len(x_grid) < 2 or len(y_grid) < 2:
        return None
    counts = (len(x_grid), len(y_grid))
    default = make_spatial_grid(counts)
    if not (_same_positions(np.asarray(x_grid), default['x'])
            and _same_positions(np.asarray(y_grid), default['y'])):
        return None
    return dict(request, spatial_grid=make_spatial_grid(counts, FULL_DETECTOR_BOUNDS))


def _request_stem(request):
    return _cache_stem(request['detector'], '1', request['wavelengths'],
                       request['spatial_grid'], request['fov_arcsec'],
                       request['oversample'], request['stpsf_filter'])


def _non_stpsf_stand_in(cache_dir, stem, request):
    """A deliberate non-STPSF cache (Gaussian) under the bare configuration name.

    It has no STPSF version to put in a name, so it cannot be the exact file. It
    has no IPC either, so it serves only an add_ipc=False request. A Gaussian
    does not change across the field, so one made over the full detector serves
    a request on the default bounds as well.
    """
    full = _full_detector_request(request)
    if full is not None:
        stand_in = _non_stpsf_stand_in_at(cache_dir, _request_stem(full), full)
        if stand_in is not None:
            return stand_in
    return _non_stpsf_stand_in_at(cache_dir, stem, request)


def _non_stpsf_stand_in_at(cache_dir, stem, request):
    path = cache_dir / f"{stem}.npz"
    if request['add_ipc'] or not path.exists():
        return None
    config = _read_cache_config(path)
    if (config is None or config['psf_source'] == 'stpsf'
            or _config_mismatches(config, request)):
        return None
    return path


# ---- TEMPORARY: remove once the 54 caches are regenerated on the cluster ----
# Everything down to the END marker, the ALLOW_FALLBACK_ENV constant, its use in
# _resolve_cache_files, the hint in _not_found_error, and the fallback_for
# argument of load_psf_payload and _cache_issues.
def _cache_ipc_state(name, header):
    """Is IPC in this file, for matching a request? True, False, or None if unknown.

    Metadata wins, then the filename: a versioned name without ``_ipc`` declares
    IPC off. A legacy name says nothing, so it depends on the stamped version.
    STPSF before 2.3.0 has no IPC in OVERDIST. A legacy name with no version at
    all predates 2.3.0 too, so it counts as off. A legacy name stamped >= 2.3.0
    came from the old code under the new STPSF, whose default is IPC on, unless
    the caller overrode it: unknown.
    """
    _, name_version, name_ipc = _parse_cache_filename(name)
    if header is not None and header['psf_source'] != 'stpsf':
        return None
    if header is not None and header['add_ipc'] is not None:
        return header['add_ipc']
    if name_ipc:
        return True
    if name_version is not None:
        return False
    parsed = _parse_version(header['version']) if header is not None else None
    if parsed is None or parsed < _parse_version(IPC_FIRST_STPSF_VERSION):
        return False
    return None


def _fallback_cache_files(cache_dir, stem, request):
    """Files of the same configuration, best first, if the environment allows it.

    The exact target file comes first, then the newest STPSF version; at equal
    version a known IPC state beats an unknown one and a versioned name beats a
    legacy one. Dropped: a file with the wrong IPC state, or stored nodes,
    wavelengths or other settings that differ from the request. A file with an
    unreadable header stays in, so that the loader reports what is wrong with it.
    A request on the default bounds also takes full-detector files, last, since
    those are the only ones that exist today.
    """
    if os.environ.get(ALLOW_FALLBACK_ENV) != "1" or not cache_dir.is_dir():
        return []
    ranked = _ranked_fallback_files(cache_dir, stem, request)
    full = _full_detector_request(request)
    if full is not None:
        ranked += _ranked_fallback_files(cache_dir, _request_stem(full), full)
    return ranked


def _ranked_fallback_files(cache_dir, stem, request):
    ranked = []
    for path in cache_dir.iterdir():
        stem_of_file, name_version, name_ipc = _parse_cache_filename(path.name)
        if stem_of_file != stem:
            continue
        config = _read_cache_config(path)
        if config is not None and _config_mismatches(config, request):
            continue
        state = _cache_ipc_state(path.name, config)
        if state is not None and state != request['add_ipc']:
            continue
        version = (config['version'] if config is not None else None) or name_version
        exact = (name_version == TARGET_STPSF_VERSION
                 and name_ipc == request['add_ipc'])
        key = (exact, _parse_version(version) or (), state is not None,
               name_version is not None, path.name)
        ranked.append((key, path))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in ranked]
# ---- END TEMPORARY ----------------------------------------------------------


def _resolve_cache_files(cache_dir, filename, stem, request):
    """Files that may serve a request, best first, and the exact file's problems.

    Strictly this is the exact file or nothing.
    """
    exact = cache_dir / filename
    problems = None
    if exact.exists():
        problems = _exact_file_problems(exact, request)
        if not problems:
            return [exact], None
    stand_in = _non_stpsf_stand_in(cache_dir, stem, request)
    if stand_in is not None:
        return [stand_in], None
    return _fallback_cache_files(cache_dir, stem, request), problems


def _cache_issues(path, header, fallback_for=None):
    """Everything wrong with a cache's version or IPC labels, as clauses."""
    _, name_version, name_ipc = _parse_cache_filename(path.name)
    issues = []
    if fallback_for is not None:
        issues.append(f"used in place of {fallback_for}, which is missing or "
                      f"not usable in this directory")

    meta_version = header['version']
    name_parsed = _parse_version(name_version)
    meta_parsed = _parse_version(meta_version)
    if meta_version and name_version and meta_parsed != name_parsed:
        issues.append(f"its filename says STPSF {name_version} but its metadata "
                      f"says {meta_version} (the metadata is used)")

    version = meta_version or name_version
    parsed = _parse_version(version)
    target = _parse_version(TARGET_STPSF_VERSION)
    if parsed is None:
        what = (f"an unreadable STPSF version ({version})" if version
                else "no STPSF version in its metadata or filename")
        issues.append(f"it has {what}, so the PSF model behind it is unknown")
    else:
        if parsed < target:
            issues.append(f"it was generated with STPSF {version}, older than "
                          f"the target {TARGET_STPSF_VERSION}, and its PSF "
                          f"shapes differ")
        if meta_version is None:
            issues.append(f"its metadata has no STPSF version, so the filename's "
                          f"{name_version} is unconfirmed")
        if (header['add_ipc'] is None
                and parsed >= _parse_version(IPC_FIRST_STPSF_VERSION)):
            issues.append("it has no add_ipc metadata, so whether IPC is in the "
                          "PSF is unknown (STPSF >= 2.3.0 defaults to on, and "
                          "romanisim >= 0.16.0 applies IPC again when wrapping)")
        if parsed >= _parse_version(IPC_FIRST_STPSF_VERSION):
            if header['hf_map'] is None:
                issues.append("it has no record of its high-frequency map, so it "
                              "may have WFI01's on every detector, as plain STPSF "
                              "2.3.0 does")
            elif (header['detector'] is not None
                  and header['hf_map'] != _hf_map_name(header['detector'])):
                issues.append(f"it has the high-frequency map {header['hf_map']}, "
                              f"not the one for {header['detector']}")

    if (header['add_ipc'] is not None and name_version is not None
            and header['add_ipc'] != name_ipc):
        issues.append(f"its filename says IPC {'on' if name_ipc else 'off'} but "
                      f"its metadata says add_ipc={header['add_ipc']} (the "
                      f"metadata is used)")

    installed = _installed_stpsf_version()
    if (parsed is not None and installed is not None
            and _parse_version(installed) != parsed):
        issues.append(f"the installed STPSF is {installed}, not {version}, so "
                      f"PSF shapes may differ from a fresh generation")
    return issues


def _warn_on_cache(path, header, fallback_for=None, stacklevel=4):
    """Give at most one warning for a cache, once per message per process."""
    if header['psf_source'] != 'stpsf':
        _warn_once(f"PSF cache {path} holds {header['psf_source']} PSFs, not "
                   f"STPSF ones; STPSF version checks do not apply.",
                   UserWarning, stacklevel=stacklevel)
        return
    issues = _cache_issues(path, header, fallback_for)
    if issues:
        _warn_once(f"PSF cache {path}: {'; '.join(issues)}. "
                   f"{_REGENERATE_ADVICE}", stacklevel=stacklevel)


def check_psf_cache_file(path):
    """Give the load-time version and IPC warning for a file read some other way.

    For scripts that read the npz directly instead of using `load_psf_payload`.
    """
    path = Path(path)
    header = _read_cache_header(path)
    if header is not None:
        _warn_on_cache(path, header)


def find_psf_cache_file(cache_dir, detector, stpsf_filter, add_ipc=False,
                        wavelengths=None, spatial_grid=None, fov_arcsec=5.0,
                        oversample=4, warn=True):
    """Path of the cache `get_or_make_psf_payload` would load for this request.

    For scripts that read the npz directly, and for checking before expensive
    work. The rule is the same: the exact file, or `PSFCacheNotFoundError`.
    Defaults are the default node grid and, for grism filters, the grism band.

    Parameters
    ----------
    warn : bool, optional
        Give the load-time version and IPC warning for the file found.
    """
    if wavelengths is None:
        if not str(stpsf_filter).startswith("GRISM"):
            raise ValueError(f"stpsf_filter={stpsf_filter!r} needs wavelengths: "
                             f"the default is the grism band.")
        wavelengths = np.arange(0.9, 2.01, 0.02)
    if spatial_grid is None:
        spatial_grid = make_spatial_grid()
    cache_dir = Path(cache_dir)
    request = _cache_request(detector, stpsf_filter, wavelengths, spatial_grid,
                             fov_arcsec, oversample, add_ipc)
    filename = get_cache_filename(detector, '1', wavelengths, spatial_grid,
                                  fov_arcsec, oversample, stpsf_filter=stpsf_filter,
                                  add_ipc=add_ipc)
    stem = _cache_stem(detector, '1', wavelengths, spatial_grid, fov_arcsec,
                       oversample, stpsf_filter)
    usable, problems = _resolve_cache_files(cache_dir, filename, stem, request)
    if not usable:
        raise _not_found_error(request, cache_dir / filename, problems)
    if warn:
        check_psf_cache_file(usable[0])
    return usable[0]


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
    - Metadata (detector, order, STPSF version, add_ipc, generation timestamp)

    See Also
    --------
    load_psf_payload : Load cached payload
    get_or_make_psf_payload : Convenience function with auto-caching
    """
    # The payload knows what made it; the installed STPSF is only a fallback
    # for a payload that does not carry a version.
    if 'stpsf_version' in payload:
        stpsf_version = payload['stpsf_version'] or "unknown"
    else:
        stpsf_version = _installed_stpsf_version() or "unknown"

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
        # STPSF filter used (absent in caches from before prism support)
        'stpsf_filter': np.array(
            payload.get('stpsf_filter') or 'unknown', dtype='S'),
        'stpsf_version': np.array(stpsf_version, dtype='S'),
        'save_timestamp': np.array(time.time()),
    }

    # Absent means unknown, which the loader warns about; never guess here.
    if payload.get('add_ipc') is not None:
        save_dict['add_ipc'] = np.array(bool(payload['add_ipc']))
    if payload.get('stpsf_hf_map') is not None:
        save_dict['stpsf_hf_map'] = np.array(payload['stpsf_hf_map'], dtype='S')
    if payload.get('psf_source', 'stpsf') != 'stpsf':
        save_dict['psf_source'] = np.array(payload['psf_source'], dtype='S')

    # Save timing info if available
    if 'timing' in payload:
        save_dict['timing_total_time'] = np.array(payload['timing'].get('total_time', 0))
        save_dict['timing_n_psfs'] = np.array(payload['timing'].get('n_psfs', 0))

    np.savez_compressed(filepath, **save_dict)

    if verbose:
        size_mb = filepath.stat().st_size / 1e6
        print(f"Saved PSF cache: {filepath}")
        print(f"  Size: {size_mb:.1f} MB (compressed)")


def load_psf_payload(filepath, verbose=True, fallback_for=None):
    """
    Load a cached PSF payload from disk.

    Validates metadata and warns about the STPSF version and IPC state, whether
    or not STPSF is installed.

    Parameters
    ----------
    filepath : str or Path
        Path to the cache file (.npz format)
    verbose : bool, optional
        Print load information (default: True)
    fallback_for : str, optional
        Name of the file this one stands in for; added to the warning.

    Returns
    -------
    payload : dict
        PSF payload compatible with interpolate_psf(). Besides the arrays it
        holds the labels ``stpsf_version`` (str or None), ``add_ipc`` (True,
        False, or None if unknown) and ``psf_source``. Labels never reach
        jitted code (see ``jit_payload``).

    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    ValueError
        If cache file is corrupted or incompatible

    Notes
    -----
    Gives one `PSFCacheVersionWarning`, which combines every problem found, if
    the file is older than ``TARGET_STPSF_VERSION``, has no version, has filename
    and metadata that disagree, has version >= 2.3.0 and no add_ipc metadata, or
    differs from the installed STPSF. The same message is shown once per process,
    however often the file is reloaded.

    See Also
    --------
    save_psf_payload : Save payload to cache
    get_or_make_psf_payload : Convenience function with auto-caching
    """
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

        header = _header_from_npz(data)
        _warn_on_cache(filepath, header, fallback_for)
        _, name_version, name_ipc = _parse_cache_filename(filepath.name)
        version = header['version'] or name_version
        parsed = _parse_version(version)
        # What the file can vouch for: a name alone does not settle IPC for
        # STPSF >= 2.3.0, and that case has just been warned about.
        if header['add_ipc'] is not None:
            add_ipc = header['add_ipc']
        elif parsed is None or parsed < _parse_version(IPC_FIRST_STPSF_VERSION):
            add_ipc = name_ipc
        else:
            add_ipc = None

        # Reconstruct payload with JAX arrays. Every float leaf is float32
        # explicitly: with jax_enable_x64 on, an untyped jnp.array of the
        # float64 grids would give float64 leaves next to the float32 PSF
        # and optical payload (mixed precision and a separate compile).
        payload = {
            'psf_grid': jnp.array(data['psf_grid'], dtype=jnp.float32),
            'wavelengths': jnp.array(data['wavelengths'], dtype=jnp.float32),
            'wl_grid': jnp.array(data['wavelengths'], dtype=jnp.float32),  # Alias
            'spatial_x': jnp.array(data['spatial_x'], dtype=jnp.float32),
            'spatial_y': jnp.array(data['spatial_y'], dtype=jnp.float32),
            'psf_fov_pixels': int(data['psf_grid'].shape[-1]),
            'pixel_scale': float(data.get('pixel_scale', 0.11)),
            'oversample': int(data['oversample']),
            'detector': _decode_label(data['detector']),
            'order': _decode_label(data['order']),
            # Caches from before prism support have no stpsf_filter field
            'stpsf_filter': (_decode_label(data['stpsf_filter'])
                             if 'stpsf_filter' in data else 'unknown'),
            'stpsf_version': version,
            'add_ipc': add_ipc,
            'psf_source': header['psf_source'],
            'stpsf_hf_map': header['hf_map'],
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
        print(f"  STPSF: {payload['stpsf_version']}, add_ipc: {payload['add_ipc']}")
        print(f"  Grid shape: {payload['psf_grid'].shape}")
        print(f"  Memory: {payload['psf_grid'].nbytes / 1e6:.1f} MB")

    return payload


def _init_worker(counter):
    """Initialize worker process with shared counter.

    Workers only build and save payloads, so keep their JAX on the CPU: a
    GPU backend in every worker would each preallocate GPU memory.
    """
    global _progress_counter
    _progress_counter = counter
    try:
        jax.config.update("jax_platforms", "cpu")
    except RuntimeError:  # backend already initialised in this process
        pass


def _generate_single_cache(args):
    """
    Worker function for parallel cache generation.

    Parameters
    ----------
    args : tuple
        (detector, order, cache_path, wavelengths, spatial_grid, fov_arcsec,
        oversample, stpsf_filter, add_ipc)

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

    (detector, order, cache_path, wavelengths, spatial_grid,
     fov_arcsec, oversample, stpsf_filter, add_ipc) = args

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
            stpsf_filter=stpsf_filter,
            progress_counter=counter,
            add_ipc=add_ipc,
        )
        save_psf_payload(payload, cache_path, verbose=False)
        return (detector, order, 'generated', str(cache_path))
    except Exception as e:
        return (detector, order, 'failed', str(e))


def _make_psf_payload_with_progress(
    detector, order, wavelengths, spatial_grid, fov_arcsec, oversample,
    stpsf_filter=None, progress_counter=None, add_ipc=False
):
    """
    Internal version of make_psf_payload that updates a shared progress counter.
    """
    import logging

    stpsf_version = _require_stpsf_for_generation()

    # Suppress noisy STPSF messages
    logging.getLogger('stpsf').setLevel(logging.ERROR)
    logging.getLogger('poppy').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*Attempted to get aberrations.*')
    warnings.filterwarnings('ignore', message='.*outside the range.*')

    wfi = _make_wfi(detector, order, stpsf_filter, add_ipc)

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
        'stpsf_filter': resolve_stpsf_filter(order, stpsf_filter),
        'stpsf_version': stpsf_version,
        'add_ipc': bool(add_ipc),
        'stpsf_hf_map': _hf_map_name(detector),
        'wavelengths': jnp.array(wavelengths, dtype=jnp.float32),
        'wl_grid': jnp.array(wavelengths, dtype=jnp.float32),
        'spatial_x': jnp.array(x_grid, dtype=jnp.float32),
        'spatial_y': jnp.array(y_grid, dtype=jnp.float32),
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
    stpsf_filters=None,
    verbose=True,
    add_ipc=False,
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
    stpsf_filters : dict, optional
        Mapping order -> STPSF filter name, e.g.
        ``element.stpsf_filters`` from ``roman_disperser.elements``.
        Default: the grism mapping (GRISM0/GRISM1).
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
    add_ipc : bool, optional
        Put IPC in the PSFs (default: False). romanisim >= 0.16.0 applies IPC
        in make_l1, so images it wraps would count it twice. The pinned v0.15.0
        applies none because of a bug (romanisim #383), so with the default
        wrapped products contain no IPC until romanisim is upgraded.

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
        spatial_grid = make_spatial_grid()

    # Build list of tasks (detector, order, cache_path)
    tasks = []
    skipped = []
    stpsf_version = _filename_version()

    for detector in detectors:
        for order in orders:
            stpsf_filter = resolve_stpsf_filter(
                order, (stpsf_filters or {}).get(order)
            )
            filename = get_cache_filename(
                detector, order, wavelengths, spatial_grid, fov_arcsec,
                oversample, stpsf_filter=stpsf_filter,
                stpsf_version=stpsf_version, add_ipc=add_ipc,
            )
            cache_path = cache_dir / filename

            if skip_existing and cache_path.exists():
                skipped.append((detector, order))
            else:
                tasks.append((
                    detector, order, str(cache_path),
                    wavelengths, spatial_grid, fov_arcsec, oversample,
                    stpsf_filter, add_ipc,
                ))

    # Fail once, before any work, rather than once per task. A missing STPSF
    # is still reported per task, as it always was.
    if tasks and _installed_stpsf_version() is not None:
        _require_stpsf_for_generation()

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
                    stpsf_filter=task[7],
                    add_ipc=task[8],
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
        # spawn, not fork: the parent may hold a multithreaded JAX runtime
        # (and a CUDA context), which is not fork-safe (Linux default).
        import multiprocessing
        import ctypes
        mp_context = multiprocessing.get_context("spawn")

        # Calculate total PSF positions across all tasks
        n_positions_per_cache = len(spatial_grid['x']) * len(spatial_grid['y'])
        total_positions = n_to_generate * n_positions_per_cache

        if verbose:
            print(f"Starting {n_workers} parallel workers...")
            print(f"  Total PSF positions to compute: {total_positions}")

        # Create shared counter for progress tracking
        progress_counter = mp_context.Value(ctypes.c_int, 0)

        # Progress bar thread
        if verbose and have_tqdm:
            pbar = tqdm(total=total_positions, desc="Computing PSFs", unit="pos")

        with mp_context.Pool(n_workers, initializer=_init_worker,
                             initargs=(progress_counter,)) as pool:
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


def _check_loaded_nodes(payload, spatial_grid, path):
    """Refuse a payload whose node positions are not the requested ones."""
    for axis, key in (('x', 'spatial_x'), ('y', 'spatial_y')):
        if not _same_positions(np.asarray(payload[key], dtype=np.float64),
                               spatial_grid[axis]):
            raise ValueError(
                f"PSF cache {path} has {axis} nodes "
                f"{np.asarray(payload[key]).tolist()}, but "
                f"{np.asarray(spatial_grid[axis]).tolist()} were requested.")


def get_or_make_psf_payload(
    detector='WFI05',
    order='1',
    wavelengths=None,
    spatial_grid=None,
    fov_arcsec=5.0,
    oversample=4,
    cache_dir=None,
    force_regenerate=False,
    stpsf_filter=None,
    verbose=True,
    element=None,
    add_ipc=False,
    allow_generate=False,
):
    """
    Load the exact PSF cache for a request, and generate only if told to.

    This is the recommended function for typical usage.

    The PSF cache is selected by ``(detector, stpsf_filter, wavelengths,
    grid config)``, and ``stpsf_filter`` + ``wavelengths`` must describe the
    same band — a mismatch selects (or silently generates) a wrong-band
    cache, which is the bug the guard below refuses. Rather than keeping
    them consistent by hand, pass ``element=``: the dispersing element
    derives both, so the wrong combination cannot be expressed.

    The request (detector, filter, wavelengths, node positions, FOV, oversample,
    ``TARGET_STPSF_VERSION``, ``add_ipc``) fixes one filename in ``cache_dir``. That
    file must exist and its metadata must agree with the request: version, add_ipc
    and node positions. Otherwise this raises `PSFCacheNotFoundError`, which gives
    the command that makes the file. No other file is used and nothing is generated
    unless ``allow_generate`` or ``force_regenerate`` says so.

    One exception is a file marked non-STPSF (``make_gaussian_psf_cache.py``) under
    the bare configuration name, used for add_ipc=False with a notice. A temporary
    exception: with ``ROMAN_DISPERSER_ALLOW_PSF_FALLBACK=1`` the best file of the
    same configuration is loaded with a `PSFCacheVersionWarning`; that never
    generates either.

    Parameters
    ----------
    detector : str, optional
        WFI detector name (default: 'WFI05')
    order : str, optional
        Spectral order (default: '1')
    element : str or DispersingElement, optional
        Dispersing element; derives ``stpsf_filter`` (from
        ``element.stpsf_filters[order]``) and ``wavelengths`` (the
        element's band via ``elements.psf_cache_wavelengths``). Mutually
        exclusive with passing either of those explicitly. Default None
        keeps the historical grism behavior below.
    wavelengths : array_like, optional
        Wavelengths in **microns** (default: 0.9-2.0 μm at 0.02 μm spacing,
        the grism band). Low-level knob — prefer ``element=``.
    spatial_grid : dict, optional
        {'x': x_array, 'y': y_array} in SCA coordinates (default: 4×4 grid)
    fov_arcsec : float, optional
        PSF field of view in arcseconds (default: 5.0)
    oversample : int, optional
        Oversampling factor (default: 4)
    cache_dir : str or Path, optional
        Directory for cache files. If None, there is nothing to look up: this
        needs ``allow_generate`` or ``force_regenerate``, and saves nothing.
    force_regenerate : bool, optional
        If True, generate and overwrite without looking up (default: False)
    allow_generate : bool, optional
        If True, generate and save when the exact file is not usable (default:
        False). For tests and library users; runs use scripts/generate_psf_caches.py.
    stpsf_filter : str, optional
        STPSF filter name (default: the grism mapping for ``order``).
        Low-level knob — prefer ``element=``.
    verbose : bool, optional
        Print progress information (default: True)
    add_ipc : bool, optional
        Whether the PSFs should contain IPC (default: False). romanisim >=
        0.16.0 applies IPC in make_l1, so images it wraps would count it twice.
        The pinned v0.15.0 applies none because of a bug (romanisim #383), so
        with the default wrapped products contain no IPC until romanisim is
        upgraded.

    Returns
    -------
    payload : dict
        PSF payload for use with interpolate_psf()

    Examples
    --------
    >>> # Element-parametric (works for GRISM and PRISM alike)
    >>> from roman_disperser.elements import PRISM
    >>> payload = get_or_make_psf_payload(
    ...     detector='WFI05', order='1', element=PRISM,
    ...     cache_dir='data/psf_cache'
    ... )
    >>> # Generate if missing (~5-6 min), then load (~1 sec)
    >>> payload = get_or_make_psf_payload(
    ...     detector='WFI05', order='1', cache_dir='data/psf_cache',
    ...     allow_generate=True,
    ... )

    See Also
    --------
    make_psf_payload : Always generates fresh (no caching)
    load_psf_payload : Load from specific file
    save_psf_payload : Save to specific file
    """
    if element is not None:
        # The element derives the filter AND the band; allowing an explicit
        # override alongside it would reintroduce exactly the
        # keep-four-arguments-consistent problem element= exists to remove.
        if stpsf_filter is not None or wavelengths is not None:
            raise ValueError(
                "element= is mutually exclusive with stpsf_filter=/"
                "wavelengths=: the element derives both. Pass element= "
                "alone, or the explicit low-level arguments without it."
            )
        from roman_disperser.elements import get_element, psf_cache_wavelengths
        element = get_element(element)
        if order not in element.stpsf_filters:
            raise ValueError(
                f"Order {order!r} is not defined for element "
                f"{element.name!r} (orders: {sorted(element.stpsf_filters)})."
            )
        stpsf_filter = element.stpsf_filters[order]
        wavelengths = psf_cache_wavelengths(element)

    # Setup defaults (same as make_psf_payload) - wavelengths in microns.
    # The default grid is the GRISM band. For any other filter that default
    # is a trap, not a convenience: the band is baked into the cache filename,
    # so a PRISM call without wavelengths would look up (and silently
    # regenerate!) a 0.90-2.00um cache instead of finding the vendored
    # 0.75-1.85um one -- wrong band, wasted STPSF time, and a polluted cache
    # directory. Refuse instead.
    if wavelengths is None:
        if stpsf_filter is not None and not str(stpsf_filter).startswith("GRISM"):
            raise ValueError(
                f"stpsf_filter={stpsf_filter!r} with no wavelengths: the default "
                f"wavelength grid is the grism band (0.90-2.00 um), which would "
                f"select or generate a wrong-band PSF cache for this filter. "
                f"Pass the element's band explicitly, e.g. "
                f"wavelengths=elements.psf_cache_wavelengths(element)."
            )
        wavelengths = np.arange(0.9, 2.01, 0.02)
    wavelengths = np.asarray(wavelengths)

    if spatial_grid is None:
        spatial_grid = make_spatial_grid()

    # Without a cache_dir there is nothing to look up
    if cache_dir is None:
        if not (allow_generate or force_regenerate):
            raise PSFCacheNotFoundError(
                "get_or_make_psf_payload needs a cache_dir, or allow_generate=True "
                "to generate a PSF grid without saving it.")
        if verbose:
            print("No cache_dir specified, generating PSF grid...")
        return make_psf_payload(
            detector=detector,
            order=order,
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=fov_arcsec,
            oversample=oversample,
            stpsf_filter=stpsf_filter,
            verbose=verbose,
            add_ipc=add_ipc,
        )

    # Generate cache filepath
    cache_dir = Path(cache_dir)
    filename = get_cache_filename(
        detector, order, wavelengths, spatial_grid, fov_arcsec, oversample,
        stpsf_filter=stpsf_filter, add_ipc=add_ipc,
    )
    cache_path = cache_dir / filename

    # Load the exact file. Only the temporary fallback can offer anything else.
    if not force_regenerate:
        resolved_filter = resolve_stpsf_filter(order, stpsf_filter)
        request = _cache_request(detector, resolved_filter, wavelengths,
                                 spatial_grid, fov_arcsec, oversample, add_ipc)
        stem = _cache_stem(detector, order, wavelengths, spatial_grid,
                           fov_arcsec, oversample, stpsf_filter)
        usable, problems = _resolve_cache_files(cache_dir, filename, stem, request)
        for candidate in usable:
            if verbose:
                print(f"Loading PSF payload from cache: {candidate}")
            exact = candidate.name == filename
            try:
                payload = load_psf_payload(
                    candidate, verbose=verbose,
                    fallback_for=None if exact else filename)
            except (ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
                warnings.warn(f"PSF cache {candidate} is invalid ({e}).",
                              RuntimeWarning, stacklevel=2)
                problems = [f"it is invalid ({e})"] if exact else problems
                continue
            # A stand-in or fallback on the default bounds can be a full-detector file
            full = None if exact else _full_detector_request(request)
            if (full is not None and
                    _parse_cache_filename(candidate.name)[0] == _request_stem(full)):
                _check_loaded_nodes(payload, full['spatial_grid'], candidate)
            else:
                _check_loaded_nodes(payload, spatial_grid, candidate)
            return payload
        if not allow_generate:
            raise _not_found_error(request, cache_path, problems)
        warnings.warn(
            f"PSF cache {cache_path} not usable: generating a new PSF grid with "
            f"the local STPSF, as allow_generate=True asks.",
            RuntimeWarning, stacklevel=2,
        )

    if verbose:
        if force_regenerate:
            print("Force regenerate requested, generating PSF grid...")
        else:
            print("Cache not usable, generating PSF grid...")

    payload = make_psf_payload(
        detector=detector,
        order=order,
        wavelengths=wavelengths,
        spatial_grid=spatial_grid,
        fov_arcsec=fov_arcsec,
        oversample=oversample,
        stpsf_filter=stpsf_filter,
        verbose=verbose,
        add_ipc=add_ipc,
    )

    # Name the file after the STPSF that made it, which can be newer than the
    # target
    cache_path = cache_dir / get_cache_filename(
        detector, order, wavelengths, spatial_grid, fov_arcsec, oversample,
        stpsf_filter=stpsf_filter,
        stpsf_version=payload.get('stpsf_version'), add_ipc=add_ipc,
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
    stpsf_filter=None,
    verbose=True,
    add_ipc=False,
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
        Default: ``make_spatial_grid()``, 4×4 inside the aberration hull

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

    add_ipc : bool, optional
        Put IPC in the OVERDIST PSFs (default: False). romanisim >= 0.16.0
        applies IPC in make_l1, so images it wraps would count it twice. The
        pinned v0.15.0 applies none because of a bug (romanisim #383), so with
        the default wrapped products contain no IPC until romanisim is upgraded.

    Returns
    -------
    payload : dict
        PSF payload with keys:
        - 'detector': str, detector name
        - 'stpsf_version': str, STPSF version that made the grid (label)
        - 'add_ipc': bool, whether IPC is in the PSFs (label)
        - 'stpsf_hf_map': str, high-frequency wavefront map used (label)
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
    - OVERDIST includes geometric distortion, charge diffusion, pixel sampling,
      and IPC only if ``add_ipc`` is True
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
        spatial_grid = make_spatial_grid()

    # Validate spatial grids are strictly increasing
    x_grid = np.asarray(spatial_grid['x'])
    y_grid = np.asarray(spatial_grid['y'])
    if not np.all(np.diff(x_grid) > 0):
        raise ValueError("Spatial x grid must be strictly increasing")
    if not np.all(np.diff(y_grid) > 0):
        raise ValueError("Spatial y grid must be strictly increasing")

    # Resolve the STPSF filter up front so an invalid order/filter pairing
    # fails before any (slow) STPSF work starts.
    stpsf_filter = resolve_stpsf_filter(order, stpsf_filter)
    stpsf_version = _require_stpsf_for_generation()

    # Compute PSF grid with timing
    if verbose:
        print(f"Generating PSF grid for {detector}, order {order}...")
        print(f"  Spatial grid: {len(spatial_grid['x'])}×{len(spatial_grid['y'])} positions")
        print(f"  Wavelengths: {len(wavelengths)} samples ({wavelengths[0]:.2f}-{wavelengths[-1]:.2f} μm)")
        print(f"  Total PSFs: {len(spatial_grid['x']) * len(spatial_grid['y']) * len(wavelengths)}")
        print(f"  FOV: {fov_arcsec:.1f} arcsec, Oversample: {oversample}×")
        print(f"  This may take 5-6 minutes for default grid...")

    psf_grid, timing = _compute_psf_grid_with_timing(
        detector, order, wavelengths, spatial_grid, fov_arcsec, oversample,
        verbose, stpsf_filter=stpsf_filter, add_ipc=add_ipc,
    )

    # Return JAX-compatible payload
    payload = {
        'detector': detector,
        'order': order,
        'stpsf_filter': stpsf_filter,
        'stpsf_version': stpsf_version,
        'add_ipc': bool(add_ipc),
        'stpsf_hf_map': _hf_map_name(detector),
        'wavelengths': jnp.array(wavelengths, dtype=jnp.float32),
        'wl_grid': jnp.array(wavelengths, dtype=jnp.float32),  # Alias for consistency with optical model
        'spatial_x': jnp.array(spatial_grid['x'], dtype=jnp.float32),
        'spatial_y': jnp.array(spatial_grid['y'], dtype=jnp.float32),
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
    detector, order, wavelengths, spatial_grid, fov_arcsec, oversample, verbose,
    stpsf_filter=None, add_ipc=False,
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
    add_ipc : bool
        Put IPC in the PSFs

    Returns
    -------
    psf_grid : ndarray
        Shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
    timing : dict
        {'total_time': float, 'per_psf_time': float, 'n_psfs': int}
    """
    import logging

    # Suppress noisy STPSF messages
    logging.getLogger('stpsf').setLevel(logging.ERROR)
    logging.getLogger('poppy').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*Attempted to get aberrations.*')
    warnings.filterwarnings('ignore', message='.*outside the range.*')

    start_time = time.time()

    wfi = _make_wfi(detector, order, stpsf_filter, add_ipc)

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


def interp_wavelength_chunk(psfs_grid, grid_wl, target_wl):
    """
    Interpolate PSFs to target wavelengths (vectorized).

    This function is optimized for batch processing of many wavelengths
    at once, used internally by the chunked star dispersion algorithm.

    Uses edge extrapolation: wavelengths outside the grid use the nearest
    edge PSF (i.e., clamped to grid bounds).

    This function is JAX-compatible and JIT-compilable.

    Parameters
    ----------
    psfs_grid : jnp.ndarray
        PSF array at grid wavelengths, shape [N_grid, PSF_y, PSF_x]
    grid_wl : jnp.ndarray
        Wavelengths in the grid (**microns**), shape [N_grid]
        Must be strictly increasing
    target_wl : jnp.ndarray
        Target wavelengths (**microns**), shape [N_target]

    Returns
    -------
    psfs : jnp.ndarray
        Interpolated PSFs at target wavelengths
        Shape: [N_target, PSF_y, PSF_x]

    Examples
    --------
    >>> # Get PSFs at grid wavelengths via spatial interpolation
    >>> psfs_grid = interpolate_psf_spatial(payload, xsca=2000.0, ysca=2000.0)
    >>> # Interpolate to many target wavelengths (microns)
    >>> target_wl = jnp.linspace(1.0, 1.8, 1000)  # microns
    >>> psfs = interp_wavelength_chunk(psfs_grid, payload['wavelengths'], target_wl)
    >>> psfs.shape
    (1000, 182, 182)

    Notes
    -----
    - This function is semantically equivalent to interpolate_psf_wavelength()
      but may be slightly faster for large batches due to explicit vectorization
    - Linear interpolation along wavelength dimension
    - Edge extrapolation: uses nearest grid PSF for out-of-bounds wavelengths

    See Also
    --------
    interpolate_psf_wavelength : Similar functionality with different signature
    interpolate_psf_spatial : Get PSFs at grid wavelengths for a spatial position
    """
    # Find bracketing indices
    idx = jnp.searchsorted(grid_wl, target_wl) - 1
    idx = jnp.clip(idx, 0, len(grid_wl) - 2)

    # Get bracketing wavelengths
    wl_lo = grid_wl[idx]
    wl_hi = grid_wl[idx + 1]

    # Interpolation weights
    t = (target_wl - wl_lo) / (wl_hi - wl_lo)
    t = jnp.clip(t, 0.0, 1.0)  # Clamp for edge cases

    # Gather and interpolate
    psf_lo = psfs_grid[idx]      # [N_target, PSF_y, PSF_x]
    psf_hi = psfs_grid[idx + 1]  # [N_target, PSF_y, PSF_x]

    return psf_lo + t[:, None, None] * (psf_hi - psf_lo)


def interpolate_psf_spatial(payload, xsca, ysca, grid_key='psf_grid'):
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
    grid_key : str, optional
        Payload key of the node grid to blend (default ``'psf_grid'``).
        Any [N_y, N_x, ...] array on the same spatial nodes works, e.g.
        the phase-binned grid ``'psf_grid_binned'`` that
        ``star_disperser.make_star_disperser(hoist_psf=True)`` adds.

    Returns
    -------
    psfs : jnp.ndarray
        Interpolated PSF array for all wavelengths in the grid
        Shape: [N_wl, PSF_y, PSF_x] (``payload[grid_key].shape[2:]``)

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
    psf_grid = payload[grid_key]  # [N_y, N_x, N_wl, PSF_y, PSF_x]

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

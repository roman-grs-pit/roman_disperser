"""Reference-data path resolution.

All ``roman_disperser`` reference data (optical model, sensitivities, synphot,
PSF caches, catalogs) is *vendored*: fetched on demand into a data directory
rather than bundled in the wheel or assumed to sit in a repo checkout. This
module is the single place that decides where that directory is, so the rest of
the package never re-implements the lookup.

The data directory is resolved in this order:

1. an explicit argument passed by the caller (or a CLI ``--dest`` flag),
2. ``$ROMAN_DISPERSER_DATA`` — neutral override pointing *at* the data dir,
3. ``$PIXI_PROJECT_ROOT/data`` — back-compat with the pixi dev workflow,
4. ``./data`` — default: a ``data/`` directory under the current directory.

Two of these locations — the optical model and the sensitivity curves — depend on
which dispersing element is in use, so those accessors take an optional ``mode``
(see :mod:`roman_disperser.modes`). It is **keyword-optional and defaults to
grism**, never required and never positional, so every existing no-argument call
keeps working unchanged.

See ``docs/data_vendoring_plan.md`` for the full design.
"""

import os
from pathlib import Path

from roman_disperser import modes

# Layout within the data directory. The optical-model filename and the
# sensitivities subdirectory are mode-dependent and live in `modes.py`; these
# grism values are kept only as back-compatible aliases for callers that
# imported them before modes existed.
OPTICAL_MODEL_FILE = modes.GRISM.optical_model_file
CATALOGS_SUBDIR = "catalogs"
SENSITIVITIES_SUBDIR = modes.GRISM.sensitivities_subdir
PSF_CACHE_SUBDIR = "psf_cache"
SYNPHOT_SUBDIR = "synphot"


def data_dir(explicit=None):
    """Return the reference-data directory (see module docstring for order)."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("ROMAN_DISPERSER_DATA")
    if env:
        return Path(env)
    pixi_root = os.environ.get("PIXI_PROJECT_ROOT")
    if pixi_root:
        return Path(pixi_root) / "data"
    return Path("data")


def catalog_dir(explicit=None):
    """Source-catalog directory (``metadata.parquet`` + ``seds.zarr/``)."""
    return Path(explicit) if explicit is not None else data_dir() / CATALOGS_SUBDIR


def sensitivity_dir(explicit=None, mode=None):
    """Sensitivity-curve directory (FITS files + ``sensitivity_map.yaml``).

    ``mode`` selects the dispersing element (default grism); see
    :func:`roman_disperser.modes.get_mode`.
    """
    if explicit is not None:
        return Path(explicit)
    return data_dir() / modes.get_mode(mode).sensitivities_subdir


def psf_cache_dir(explicit=None):
    """PSF cache directory (``psf_WFI*.npz``)."""
    return Path(explicit) if explicit is not None else data_dir() / PSF_CACHE_SUBDIR


def synphot_dir(explicit=None):
    """Synphot reference directory (bandpasses + spectral templates)."""
    return Path(explicit) if explicit is not None else data_dir() / SYNPHOT_SUBDIR


def optical_model_path(explicit=None, mode=None):
    """Optical-model YAML path.

    ``mode`` selects the dispersing element (default grism); see
    :func:`roman_disperser.modes.get_mode`.
    """
    if explicit is not None:
        return Path(explicit)
    return data_dir() / modes.get_mode(mode).optical_model_file

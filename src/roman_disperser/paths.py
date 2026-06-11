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

See ``docs/data_vendoring_plan.md`` for the full design.
"""

import os
from pathlib import Path

# Layout within the data directory.
OPTICAL_MODEL_FILE = "Roman_grism_OpticalModel_v0.8.yaml"
CATALOGS_SUBDIR = "catalogs"
SENSITIVITIES_SUBDIR = "sensitivities"
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


def sensitivity_dir(explicit=None):
    """Sensitivity-curve directory (FITS files + ``sensitivity_map.yaml``)."""
    return Path(explicit) if explicit is not None else data_dir() / SENSITIVITIES_SUBDIR


def psf_cache_dir(explicit=None):
    """PSF cache directory (``psf_WFI*.npz``)."""
    return Path(explicit) if explicit is not None else data_dir() / PSF_CACHE_SUBDIR


def synphot_dir(explicit=None):
    """Synphot reference directory (bandpasses + spectral templates)."""
    return Path(explicit) if explicit is not None else data_dir() / SYNPHOT_SUBDIR


def optical_model_path(explicit=None):
    """Optical-model YAML path."""
    return Path(explicit) if explicit is not None else data_dir() / OPTICAL_MODEL_FILE

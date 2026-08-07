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

import json
import os
from pathlib import Path

# Layout within the data directory. (Sensitivities are per-element —
# SENSITIVITIES_SUBDIR is the grism one; see elements.sensitivities_subdir.)
CATALOGS_SUBDIR = "catalogs"
SENSITIVITIES_SUBDIR = "sensitivities"
PSF_CACHE_SUBDIR = "psf_cache"
SYNPHOT_SUBDIR = "synphot"

# Written by roman-disperser-hydrate (see hydrate.LOCK_NAME — keep in sync).
LOCK_NAME = "data-versions.lock"


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
    """Source-catalog directory (``metadata.parquet`` + ``seds.zarr/``).

    ``explicit`` is a *directory* path, returned as-is.
    """
    return Path(explicit) if explicit is not None else data_dir() / CATALOGS_SUBDIR


def sensitivity_dir(explicit=None, element=None):
    """Sensitivity-curve directory (FITS files + ``sensitivity_map.yaml``).

    ``explicit`` is a *directory* path, returned as-is. Otherwise the
    element's subdir under the data dir — ``sensitivities/`` for the grism
    (the default, as everywhere), ``sensitivities_prism/`` for the prism.
    The two deliveries cannot share one directory: each ships its own
    ``sensitivity_map.yaml`` index, and the grism filenames carry no
    element marker.
    """
    if explicit is not None:
        return Path(explicit)
    from roman_disperser.elements import get_element
    return data_dir() / get_element(element).sensitivities_subdir


def psf_cache_dir(explicit=None):
    """PSF cache directory (``psf_WFI*.npz``).

    ``explicit`` is a *directory* path, returned as-is. Both elements share
    this one directory by design — the cache filenames carry the STPSF
    filter (``GRISM0``/``GRISM1``/``PRISM``), so they cannot collide.
    """
    return Path(explicit) if explicit is not None else data_dir() / PSF_CACHE_SUBDIR


def synphot_dir(explicit=None):
    """Synphot reference directory (bandpasses + spectral templates).

    ``explicit`` is a *directory* path, returned as-is.
    """
    return Path(explicit) if explicit is not None else data_dir() / SYNPHOT_SUBDIR


def optical_model_filename(element, version):
    """Delivery filename for an element's optical model at ``version``.

    The upstream (IPAC/SSC) deliveries have so far followed
    ``Roman_<element>_OpticalModel_v<X>.yaml``; this template is the one
    naming assumption the resolver makes. ``version`` may be given with or
    without the leading ``v`` (``"v0.8"`` or ``"0.8"``).
    """
    v = str(version)
    if not v.startswith("v"):
        v = f"v{v}"
    return f"Roman_{element.name}_OpticalModel_{v}.yaml"


def _optical_model_asset_key(element):
    """Manifest/lock key for an element's optical model (see hydrate.ASSETS)."""
    return ("optical_model" if element.name == "grism"
            else f"optical_model_{element.name}")


def _version_from_lock(base, key):
    """Delivery version recorded in the data dir's lock, or None.

    Lock values are our own release tags (``optical-model-v0.8``,
    ``optical-model-prism-v0.8``); the version is everything after the last
    ``-v``. Unreadable lock or unparseable tag both resolve to None — the
    caller fails loudly rather than guessing.
    """
    try:
        lock = json.loads((Path(base) / LOCK_NAME).read_text())
        tag = lock[key]
        return "v" + tag.rsplit("-v", 1)[1] if "-v" in tag else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def optical_model_path(explicit=None, element=None, version=None):
    """Resolve the optical-model YAML path for a dispersing element.

    Resolution is *declared, never inferred*: the file used is always the
    consequence of an explicit path, an explicit version, or what hydrate
    recorded in the data dir's ``data-versions.lock``. Directory contents
    are never scanned to pick a model (a stray file must not silently
    become the calibration), only listed as a hint on failure.

    1. ``explicit`` — a full *file* path to the YAML itself, returned as-is
       (config ``optical_model:``). Unlike the sibling resolvers, whose
       ``explicit`` names a directory, this one resolves a single file —
       so its ``explicit`` is that file.
    2. ``version`` — delivery version string, e.g. ``"v0.8"`` (config
       ``optical_model_version:``); filename built by
       :func:`optical_model_filename`.
    3. The ``data-versions.lock`` written by ``roman-disperser-hydrate`` —
       the default: you get exactly the delivery you hydrated.
    4. Otherwise ``FileNotFoundError``, listing any model files present in
       the data dir (not used) and the three ways to declare one.

    ``element`` defaults to the grism, as everywhere.
    """
    if explicit is not None:
        return Path(explicit)

    from roman_disperser.elements import get_element
    element = get_element(element)
    base = data_dir()

    if version is not None:
        path = base / optical_model_filename(element, version)
        if not path.exists():
            raise FileNotFoundError(
                f"Optical model {path.name!r} (element {element.name!r}, "
                f"version {version!r}) not found in {base}. Hydrate that "
                f"delivery (`roman-disperser-hydrate --only "
                f"{_optical_model_asset_key(element)} --manifest <ref>`) or "
                f"check the version string."
            )
        return path

    lock_version = _version_from_lock(base, _optical_model_asset_key(element))
    if lock_version is not None:
        path = base / optical_model_filename(element, lock_version)
        if not path.exists():
            raise FileNotFoundError(
                f"{base / LOCK_NAME} records {lock_version!r} for element "
                f"{element.name!r} but {path.name!r} is missing from {base}. "
                f"Re-run `pixi run hydrate`."
            )
        return path

    candidates = sorted(
        p.name for p in base.glob(f"Roman_{element.name}_OpticalModel_*.yaml"))
    found = ("Found in the data dir (not used): " + ", ".join(candidates)
             if candidates else "No model files found in the data dir either.")
    raise FileNotFoundError(
        f"Cannot resolve the optical model for element {element.name!r}: no "
        f"usable {_optical_model_asset_key(element)!r} entry in "
        f"{base / LOCK_NAME}. {found} Either run `pixi run hydrate` (records "
        f"the delivery in the lock), pass version=... (config "
        f"`optical_model_version:`), or pass an explicit path (config "
        f"`optical_model:`)."
    )

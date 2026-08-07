"""Roman Disperser: JAX-based optical model and disperser for Roman Space
Telescope slitless spectroscopy — both WFI dispersing elements, the G150
grism (default) and the P127 prism."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .optical_model import RomanOpticalModel
from . import optical_model_jax
from . import disperser
from . import elements
from . import star_disperser
from . import galaxy_disperser
from . import demo_utils
from . import psf_utils
from . import psf_model
from . import catalog
from . import hydrate
from . import paths
from . import pipeline
from . import refdata
from . import sersic

# Installed-package version (same source as pipeline.get_code_version and the
# CODEVER FITS card, so the three cannot disagree). Caveat: in an editable
# install this is the version at the last `pixi install`, not what
# pyproject.toml says — the release process verifies it after bumping.
try:
    __version__ = _pkg_version("roman_disperser")
except PackageNotFoundError:  # source tree without an install
    __version__ = "unknown"

__all__ = [
    "RomanOpticalModel",
    "__version__",
    "optical_model_jax",
    "disperser",
    "elements",
    "star_disperser",
    "galaxy_disperser",
    "demo_utils",
    "psf_utils",
    "psf_model",
    "catalog",
    "hydrate",
    "paths",
    "pipeline",
    "refdata",
    "sersic",
]

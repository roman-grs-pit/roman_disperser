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

# jax 0.11.0 has a GPU scatter-add regression (jax-ml/jax#39959) that makes
# the deposit step ~16x slower. New installs are protected by the dependency
# exclusion (jax!=0.11.0), but package metadata cannot reach environments
# that already have 0.11.0 installed — hence this runtime warning.
import jax as _jax

if _jax.__version__ == "0.11.0":
    import warnings

    warnings.warn(
        "jax 0.11.0 has a GPU scatter-add performance regression "
        "(jax-ml/jax#39959) that makes roman_disperser's deposit step "
        "~16x slower. Upgrade jax (>=0.11.1), e.g. "
        "pip install -U 'jax[cuda12]'.",
        RuntimeWarning,
        stacklevel=2,
    )

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

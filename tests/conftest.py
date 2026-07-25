"""
Shared pytest fixtures for roman_disperser tests.

Optical models are located through :mod:`roman_disperser.paths`, i.e. the same
vendored-data resolution the package itself uses (``$ROMAN_DISPERSER_DATA`` ->
``$PIXI_PROJECT_ROOT/data`` -> ``./data``). Constructing the path by hand from
``$PIXI_PROJECT_ROOT`` — as these fixtures used to — silently required a
checkout-local ``data/`` directory, which stopped existing when reference data
was vendored in v0.10.0.
"""

import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import modes, paths
from roman_disperser.optical_model import RomanOpticalModel


def load_model(mode):
    """Load the optical model for a dispersing mode from the vendored data dir."""
    return RomanOpticalModel(str(paths.optical_model_path(mode=mode)))


@pytest.fixture(scope="module")
def optical_model():
    """Grism optical model — the default element."""
    return load_model(modes.GRISM)


@pytest.fixture(scope="module")
def prism_optical_model():
    """Prism optical model."""
    return load_model(modes.PRISM)


@pytest.fixture(scope="module", params=[modes.GRISM, modes.PRISM],
                ids=lambda m: m.name)
def mode(request):
    """Each dispersing mode in turn, for tests that must cover both."""
    return request.param


@pytest.fixture(scope="module")
def mode_model(mode):
    """The optical model paired with the parametrized ``mode``."""
    return mode, load_model(mode)


@pytest.fixture(scope="module")
def payload(optical_model):
    """Create payload for SCA 5, order 1."""
    return omj.make_sca_payload(optical_model, sca=5, order="1")

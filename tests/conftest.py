"""
Shared pytest fixtures for roman_disperser tests.
"""

import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import elements, paths
from roman_disperser.optical_model import RomanOpticalModel


@pytest.fixture(scope="module", params=["grism", "prism"])
def optical_model(request):
    """Each dispersing element's optical model, in turn.

    Tests taking this fixture (or ``payload``) run once per element, so the
    dispersion machinery is exercised with both the grism's linear and the
    prism's log wavelength transform. Tests that are inherently grism-specific
    (fixed order lists, values pinned against the grism model) define their
    own single-element fixture instead.
    """
    element = elements.get_element(request.param)
    return RomanOpticalModel(str(paths.data_dir() / element.optical_model_file))


@pytest.fixture(scope="module")
def payload(optical_model):
    """Payload for SCA 5, order 1 (order "1" exists for both elements)."""
    return omj.make_sca_payload(optical_model, sca=5, order="1")

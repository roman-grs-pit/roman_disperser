"""
Shared pytest fixtures for roman_disperser tests.
"""

import os

import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel


@pytest.fixture(scope="module")
def optical_model():
    """Load optical model once for all tests."""
    pixi_root_path = os.environ.get("PIXI_PROJECT_ROOT", ".")
    fn = os.path.join(pixi_root_path, "data/Roman_prism_OpticalModel_v0.8.yaml")
    return RomanOpticalModel(fn)


@pytest.fixture(scope="module")
def payload(optical_model):
    """Create payload for SCA 5, order 1."""
    return omj.make_sca_payload(optical_model, sca=5, order="1")

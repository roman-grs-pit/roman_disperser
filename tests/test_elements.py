"""Tests for the dispersing-element registry and model-consistency checks."""

import dataclasses

import pytest

from roman_disperser import elements, paths
from roman_disperser.optical_model import RomanOpticalModel


@pytest.fixture(scope="module", params=["grism", "prism"])
def element_and_model(request):
    """Each element paired with its own optical model."""
    element = elements.get_element(request.param)
    model = RomanOpticalModel(
        str(paths.data_dir() / element.optical_model_file)
    )
    return element, model


class TestRegistry:
    def test_default_is_grism(self):
        assert elements.get_element() is elements.GRISM
        assert elements.get_element(None) is elements.GRISM

    def test_lookup_by_name(self):
        assert elements.get_element("grism") is elements.GRISM
        assert elements.get_element("prism") is elements.PRISM
        assert elements.get_element("PRISM") is elements.PRISM

    def test_instance_passthrough(self):
        assert elements.get_element(elements.PRISM) is elements.PRISM

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown dispersing element"):
            elements.get_element("g150")
        with pytest.raises(ValueError, match="Unknown dispersing element"):
            elements.get_element(3)

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            elements.GRISM.lam_min = 0.5

    def test_every_order_has_a_psf_filter(self):
        for element in elements.ELEMENTS.values():
            for order in element.orders:
                assert order in element.stpsf_filters, (element.name, order)

    def test_grism_order2_reuses_order1_psf(self):
        # STPSF has no order-2 grism filter; the pipeline reuses GRISM1.
        assert elements.GRISM.stpsf_filters["2"] == "GRISM1"


class TestValidateAgainstModel:
    def test_matching_pair_passes(self, element_and_model):
        element, model = element_and_model
        assert elements.validate_against_model(element, model) is element

    def test_accepts_element_name(self, element_and_model):
        element, model = element_and_model
        elements.validate_against_model(element.name, model)

    def test_cross_pairing_raises(self, element_and_model):
        element, model = element_and_model
        other = elements.PRISM if element is elements.GRISM else elements.GRISM
        with pytest.raises(ValueError, match="inconsistent"):
            elements.validate_against_model(other, model)

    def test_band_mismatch_raises(self, element_and_model):
        element, model = element_and_model
        shifted = dataclasses.replace(element, lam_max=element.lam_max + 0.1)
        with pytest.raises(ValueError, match="band"):
            elements.validate_against_model(shifted, model)

    def test_band_within_tolerance_passes(self, element_and_model):
        element, model = element_and_model
        nudged = dataclasses.replace(
            element, lam_min=element.lam_min + 0.5 * elements.BAND_TOL_UM
        )
        elements.validate_against_model(nudged, model)

    def test_undefined_order_raises(self, element_and_model):
        element, model = element_and_model
        bad = dataclasses.replace(element, orders=element.orders + ("9",))
        with pytest.raises(ValueError, match="orders"):
            elements.validate_against_model(bad, model)

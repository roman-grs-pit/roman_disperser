"""
Tests for catalog utilities (source selection).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser.catalog import select_sources
from roman_disperser import paths
from roman_disperser.optical_model import RomanOpticalModel


@pytest.fixture(scope="module")
def optical_model():
    """Load optical model once for all tests."""
    return RomanOpticalModel(str(paths.optical_model_path()))


def brute_force_select(payload, xfpa, yfpa, wl_min=0.9, wl_max=2.0,
                       detector_size=4088, padding=300, n_wl=100):
    """Brute-force source selection by tracing at many wavelengths.

    Returns True for a source if at least one wavelength sample lands
    on the padded detector.
    """
    n = len(xfpa)
    wavelengths = np.linspace(wl_min, wl_max, n_wl)

    lo = -padding
    hi = detector_size + padding

    on_detector = np.zeros(n, dtype=bool)
    for wl in wavelengths:
        wl_arr = np.full(n, wl)
        xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wl_arr)
        xsca, ysca = omj.mpa_to_sca(payload, xmpa, ympa)
        xsca, ysca = np.asarray(xsca), np.asarray(ysca)
        hit = (xsca >= lo) & (xsca <= hi) & (ysca >= lo) & (ysca <= hi)
        on_detector |= hit

    return on_detector


class TestSelectSources:
    """Test select_sources against brute-force validation."""

    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_center_sources_selected(self, optical_model, order):
        """Sources near detector center should be selected for all orders."""
        payload = omj.make_sca_payload(optical_model, sca=5, order=order)
        xsca = jnp.array([payload["det"]["crpix1"]])
        ysca = jnp.array([payload["det"]["crpix2"]])
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)

        mask = select_sources(payload, xfpa, yfpa)
        assert mask[0]

    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_far_away_sources_rejected(self, optical_model, order):
        """Sources far from detector should be rejected for all orders."""
        payload = omj.make_sca_payload(optical_model, sca=5, order=order)
        xfpa = jnp.array([10.0, -10.0, 5.0])
        yfpa = jnp.array([10.0, -10.0, 5.0])

        mask = select_sources(payload, xfpa, yfpa)
        assert not mask.any()

    def test_empty_input_raises(self, optical_model):
        """Empty input arrays should raise ValueError."""
        payload = omj.make_sca_payload(optical_model, sca=5, order="1")
        with pytest.raises(ValueError, match="non-empty"):
            select_sources(payload, jnp.array([]), jnp.array([]))

    @pytest.mark.parametrize("sca", [1, 5, 10, 18])
    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_vs_brute_force(self, optical_model, sca, order):
        """select_sources should be conservative: never miss a source that
        the brute-force check finds on-detector.

        The bounding-box check may include a few extra sources (false positives)
        but should never have false negatives.
        """
        payload = omj.make_sca_payload(optical_model, sca=sca, order=order)

        # Generate sources across a wide area around the detector
        np.random.seed(42)
        n_sources = 200
        xsca = np.random.uniform(-500, 5000, size=n_sources)
        ysca = np.random.uniform(-500, 5000, size=n_sources)
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)

        mask_fast = np.asarray(select_sources(payload, xfpa, yfpa))
        mask_brute = brute_force_select(payload, xfpa, yfpa)

        # No false negatives: every source found by brute force must be in fast mask
        false_negatives = mask_brute & ~mask_fast
        assert not false_negatives.any(), (
            f"SCA {sca}, order {order}: {false_negatives.sum()} false negatives "
            f"out of {mask_brute.sum()} brute-force positives"
        )

    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_multiple_scas(self, optical_model, order):
        """Sources on one SCA should not be selected on a distant SCA."""
        payload_5 = omj.make_sca_payload(optical_model, sca=5, order=order)
        payload_1 = omj.make_sca_payload(optical_model, sca=1, order=order)

        xsca = jnp.array([payload_5["det"]["crpix1"]])
        ysca = jnp.array([payload_5["det"]["crpix2"]])
        xfpa, yfpa = omj.sca_to_fpa(payload_5, xsca, ysca)

        mask_5 = select_sources(payload_5, xfpa, yfpa)
        mask_1 = select_sources(payload_1, xfpa, yfpa)

        assert mask_5[0]
        assert not mask_1[0]

    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_padding_effect(self, optical_model, order):
        """Larger padding should select at least as many sources."""
        payload = omj.make_sca_payload(optical_model, sca=5, order=order)

        np.random.seed(123)
        n_sources = 100
        xsca = np.random.uniform(-200, 4500, size=n_sources)
        ysca = np.random.uniform(-200, 4500, size=n_sources)
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)

        mask_small = select_sources(payload, xfpa, yfpa, padding=0)
        mask_large = select_sources(payload, xfpa, yfpa, padding=500)

        assert mask_large.sum() >= mask_small.sum()

    def test_returns_jax_bool_array(self, optical_model):
        """Output should be a JAX boolean array."""
        payload = omj.make_sca_payload(optical_model, sca=5, order="1")
        xfpa = jnp.array([0.0])
        yfpa = jnp.array([0.0])

        mask = select_sources(payload, xfpa, yfpa)
        assert isinstance(mask, jnp.ndarray)
        assert mask.dtype == jnp.bool_

    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_jit_compilation(self, optical_model, order):
        """Verify select_sources is JIT-compilable via closure."""
        payload = omj.make_sca_payload(optical_model, sca=5, order=order)

        @jax.jit
        def jitted_select(xfpa, yfpa):
            return select_sources(payload, xfpa, yfpa)

        xsca = jnp.array([2000.0, 3000.0])
        ysca = jnp.array([2000.0, 3000.0])
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)

        mask = jitted_select(xfpa, yfpa)
        assert mask.shape == (2,)
        assert mask.all()


class TestSelectSourcesPrism:
    """Prism spot-check of the brute-force comparison above.

    The main test classes keep their grism fixture (they parametrize over the
    grism's three orders); the prism has one order and a different band, so
    it gets its own no-false-negatives check with the prism band explicit.
    """

    @pytest.mark.parametrize("sca", [1, 5, 18])
    def test_vs_brute_force(self, sca):
        from roman_disperser import elements, paths
        element = elements.get_element("prism")
        model = RomanOpticalModel(
            str(paths.data_dir() / element.optical_model_file)
        )
        payload = omj.make_sca_payload(model, sca=sca, order="1")

        np.random.seed(42)
        n_sources = 200
        xsca = np.random.uniform(-500, 5000, size=n_sources)
        ysca = np.random.uniform(-500, 5000, size=n_sources)
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)

        mask_fast = np.asarray(select_sources(
            payload, xfpa, yfpa,
            wl_min=element.lam_min, wl_max=element.lam_max,
        ))
        mask_brute = brute_force_select(
            payload, xfpa, yfpa,
            wl_min=element.lam_min, wl_max=element.lam_max,
        )

        false_negatives = mask_brute & ~mask_fast
        assert not false_negatives.any(), (
            f"SCA {sca}: {false_negatives.sum()} false negatives "
            f"out of {mask_brute.sum()} brute-force positives"
        )

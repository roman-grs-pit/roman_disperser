"""
Tests for functional optical model (optical_model_jax).
Compares JAX-based transforms against class-based implementation.
"""

import os
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel

# Tolerances for float32 precision (YAML has limited decimal places)
RTOL = 1e-5
ATOL = 1e-3


@pytest.fixture(scope="module")
def optical_model():
    """Load optical model once for all tests."""
    pixi_root_path = os.environ.get("PIXI_PROJECT_ROOT", ".")
    fn = os.path.join(pixi_root_path, "data/Roman_grism_OpticalModel_v0.8.yaml")
    return RomanOpticalModel(fn)

class TestSCAtoMPA:
    """Test SCA to MPA coordinate transformation."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_center_pixel(self, optical_model, sca):
        """Test transformation at detector center."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        xsca = np.array([payload["det"]["crpix1"]])  # [2044.5]
        ysca = np.array([payload["det"]["crpix2"]])  # [2044.5]

        xmpa_jax, ympa_jax = omj.sca_to_mpa(payload, xsca, ysca)
        xmpa_class, ympa_class = optical_model.coords.convert_sca_to_mpa(
            xsca=xsca, ysca=ysca, sca=sca
        )
        
        # Compare against class-based implementation
        np.testing.assert_allclose(xmpa_jax, xmpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ympa_jax, ympa_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_corner_pixels(self, optical_model, sca):
        """Test transformation at detector corners."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        # TODO: Verify SCA coordinate range [0.5, naxis+0.5]
        test_xsca = np.array([0.5, 4088.5, 0.5, 4088.5])
        test_ysca = np.array([0.5, 0.5, 4088.5, 4088.5])

        xmpa_jax, ympa_jax = omj.sca_to_mpa(payload, test_xsca, test_ysca)
        xmpa_class, ympa_class = optical_model.coords.convert_sca_to_mpa(
            xsca=test_xsca, ysca=test_ysca, sca=sca
        )
        
        np.testing.assert_allclose(xmpa_jax, xmpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ympa_jax, ympa_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_random_points(self, optical_model, sca):
        """Test transformation at random points."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        np.random.seed(42)
        test_xsca = np.random.uniform(0.5, 4088.5, size=10)
        test_ysca = np.random.uniform(0.5, 4088.5, size=10)

        xmpa_jax, ympa_jax = omj.sca_to_mpa(payload, test_xsca, test_ysca)
        xmpa_class, ympa_class = optical_model.coords.convert_sca_to_mpa(
            xsca=test_xsca, ysca=test_ysca, sca=sca
        )
        
        np.testing.assert_allclose(xmpa_jax, xmpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ympa_jax, ympa_class, rtol=RTOL, atol=ATOL)


class TestMPAtoSCA:
    """Test MPA to SCA coordinate transformation."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_center_position(self, optical_model, sca):
        """Test transformation at MPA center (SCA center)."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        xcen, ycen = optical_model.coords.get_sca_center(sca=sca)

        xsca_jax, ysca_jax = omj.mpa_to_sca(payload, xcen, ycen)
        xsca_class, ysca_class = optical_model.coords.convert_mpa_to_sca(
            xmpa=xcen, ympa=ycen, sca=sca
        )

        np.testing.assert_allclose(xsca_jax, xsca_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ysca_jax, ysca_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_random_points(self, optical_model, sca):
        """Test transformation at random MPA points."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        xcen, ycen = optical_model.coords.get_sca_center(sca=sca)

        np.random.seed(42)
        offset_x = np.random.uniform(-5, 5, size=10)  # mm offset
        offset_y = np.random.uniform(-5, 5, size=10)  # mm offset
        test_xmpa = xcen + offset_x
        test_ympa = ycen + offset_y

        xsca_jax, ysca_jax = omj.mpa_to_sca(payload, test_xmpa, test_ympa)
        xsca_class, ysca_class = optical_model.coords.convert_mpa_to_sca(
            xmpa=test_xmpa, ympa=test_ympa, sca=sca
        )

        np.testing.assert_allclose(xsca_jax, xsca_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ysca_jax, ysca_class, rtol=RTOL, atol=ATOL)


class TestRoundTrip:
    """Test round-trip conversions SCA -> MPA -> SCA and vice versa."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_sca_mpa_sca_roundtrip(self, optical_model, sca):
        """SCA -> MPA -> SCA should recover original coordinates."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        # TODO: Verify SCA coordinate range [0.5, naxis+0.5]
        test_xsca_orig = np.array([0.5, 2044.5, 4088.5, 1000.0, 3000.0])
        test_ysca_orig = np.array([0.5, 2044.5, 4088.5, 1500.0, 2500.0])

        # Forward
        xmpa, ympa = omj.sca_to_mpa(payload, test_xsca_orig, test_ysca_orig)

        # Backward
        xsca_recovered, ysca_recovered = omj.mpa_to_sca(payload, xmpa, ympa)

        np.testing.assert_allclose(
            xsca_recovered, test_xsca_orig, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            ysca_recovered, test_ysca_orig, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_mpa_sca_mpa_roundtrip(self, optical_model, sca):
        """MPA -> SCA -> MPA should recover original coordinates."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        xcen, ycen = optical_model.coords.get_sca_center(sca=sca)

        np.random.seed(42)
        offset_x = np.random.uniform(-5, 5, size=10)
        offset_y = np.random.uniform(-5, 5, size=10)
        test_xmpa_orig = xcen + offset_x
        test_ympa_orig = ycen + offset_y

        # Forward
        xsca, ysca = omj.mpa_to_sca(payload, test_xmpa_orig, test_ympa_orig)

        # Backward
        xmpa_recovered, ympa_recovered = omj.sca_to_mpa(payload, xsca, ysca)

        np.testing.assert_allclose(
            xmpa_recovered, test_xmpa_orig, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            ympa_recovered, test_ympa_orig, rtol=RTOL, atol=ATOL
        )


class TestMultipleSCAs:
    """Test that payloads are independent for different SCAs."""

    def test_different_sca_payloads(self, optical_model):
        """Different SCAs should have different centers."""
        payload_sca1 = omj.make_sca_payload(optical_model, sca=1, order="1")
        payload_sca2 = omj.make_sca_payload(optical_model, sca=2, order="1")

        xy1 = payload_sca1["det"]["xy_center"]
        xy2 = payload_sca2["det"]["xy_center"]

        # SCA centers should be different
        assert not np.allclose(xy1, xy2)

    def test_sca_transforms_independent(self, optical_model):
        """Verify transforms give different results for different SCA centers."""
        payload_sca1 = omj.make_sca_payload(optical_model, sca=1, order="1")
        payload_sca2 = omj.make_sca_payload(optical_model, sca=2, order="1")

        # Same pixel in different SCAs -> different MPA coords
        xmpa1, ympa1 = omj.sca_to_mpa(payload_sca1, 2044.5, 2044.5)
        xmpa2, ympa2 = omj.sca_to_mpa(payload_sca2, 2044.5, 2044.5)

        assert not np.allclose(xmpa1, xmpa2)
        assert not np.allclose(ympa1, ympa2)


class TestSCAtoFPA:
    """Test SCA to FPA coordinate transformation."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_center_pixel(self, optical_model, sca):
        """Test transformation at detector center."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        xsca = np.array([payload["det"]["crpix1"]])
        ysca = np.array([payload["det"]["crpix2"]])

        xfpa_jax, yfpa_jax = omj.sca_to_fpa(payload, xsca, ysca)
        xfpa_class, yfpa_class = optical_model.coords.convert_sca_to_fpa(
            xsca=xsca, ysca=ysca, sca=sca
        )

        np.testing.assert_allclose(xfpa_jax, xfpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(yfpa_jax, yfpa_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_corner_pixels(self, optical_model, sca):
        """Test transformation at detector corners."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        test_xsca = np.array([0.5, 4088.5, 0.5, 4088.5])
        test_ysca = np.array([0.5, 0.5, 4088.5, 4088.5])

        xfpa_jax, yfpa_jax = omj.sca_to_fpa(payload, test_xsca, test_ysca)
        xfpa_class, yfpa_class = optical_model.coords.convert_sca_to_fpa(
            xsca=test_xsca, ysca=test_ysca, sca=sca
        )

        np.testing.assert_allclose(xfpa_jax, xfpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(yfpa_jax, yfpa_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_random_points(self, optical_model, sca):
        """Test transformation at random points."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        np.random.seed(42)
        test_xsca = np.random.uniform(0.5, 4088.5, size=10)
        test_ysca = np.random.uniform(0.5, 4088.5, size=10)

        xfpa_jax, yfpa_jax = omj.sca_to_fpa(payload, test_xsca, test_ysca)
        xfpa_class, yfpa_class = optical_model.coords.convert_sca_to_fpa(
            xsca=test_xsca, ysca=test_ysca, sca=sca
        )

        np.testing.assert_allclose(xfpa_jax, xfpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(yfpa_jax, yfpa_class, rtol=RTOL, atol=ATOL)


class TestFPAtoSCA:
    """Test FPA to SCA coordinate transformation."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_center_pixel(self, optical_model, sca):
        """Test transformation at FPA center (maps to SCA center pixel)."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        
        # Get FPA coords of SCA center
        xsca_center = payload["det"]["crpix1"]
        ysca_center = payload["det"]["crpix2"]
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca_center, ysca_center)

        # Convert back
        xsca_jax, ysca_jax = omj.fpa_to_sca(payload, xfpa, yfpa)
        xsca_class, ysca_class = optical_model.coords.convert_fpa_to_sca(
            xfpa=xfpa, yfpa=yfpa, sca=sca
        )

        np.testing.assert_allclose(xsca_jax, xsca_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ysca_jax, ysca_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_random_points(self, optical_model, sca):
        """Test transformation at random FPA points."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        np.random.seed(42)
        # Random FPA points (within ~ 0.1 degree, roughly detector field)
        test_xfpa = np.random.uniform(-0.1, 0.1, size=10)
        test_yfpa = np.random.uniform(-0.1, 0.1, size=10)

        xsca_jax, ysca_jax = omj.fpa_to_sca(payload, test_xfpa, test_yfpa)
        xsca_class, ysca_class = optical_model.coords.convert_fpa_to_sca(
            xfpa=test_xfpa, yfpa=test_yfpa, sca=sca
        )

        np.testing.assert_allclose(xsca_jax, xsca_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ysca_jax, ysca_class, rtol=RTOL, atol=ATOL)


class TestRoundTripFPA:
    """Test round-trip conversions SCA <-> FPA."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_sca_fpa_sca_roundtrip(self, optical_model, sca):
        """SCA -> FPA -> SCA should recover original coordinates."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        test_xsca_orig = np.array([0.5, 2044.5, 4088.5, 1000.0, 3000.0])
        test_ysca_orig = np.array([0.5, 2044.5, 4088.5, 1500.0, 2500.0])

        # Forward
        xfpa, yfpa = omj.sca_to_fpa(payload, test_xsca_orig, test_ysca_orig)

        # Backward
        xsca_recovered, ysca_recovered = omj.fpa_to_sca(payload, xfpa, yfpa)

        np.testing.assert_allclose(
            xsca_recovered, test_xsca_orig, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            ysca_recovered, test_ysca_orig, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_fpa_sca_fpa_roundtrip(self, optical_model, sca):
        """FPA -> SCA -> FPA should recover original coordinates."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        np.random.seed(42)
        test_xfpa_orig = np.random.uniform(-0.1, 0.1, size=10)
        test_yfpa_orig = np.random.uniform(-0.1, 0.1, size=10)

        # Forward
        xsca, ysca = omj.fpa_to_sca(payload, test_xfpa_orig, test_yfpa_orig)

        # Backward
        xfpa_recovered, yfpa_recovered = omj.sca_to_fpa(payload, xsca, ysca)

        np.testing.assert_allclose(
            xfpa_recovered, test_xfpa_orig, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            yfpa_recovered, test_yfpa_orig, rtol=RTOL, atol=ATOL
        )

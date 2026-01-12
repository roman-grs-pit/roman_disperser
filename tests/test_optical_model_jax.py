"""
Tests for functional optical model (optical_model_jax).
Compares JAX-based transforms against class-based implementation.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


class TestMPAtoFPA:
    """Test MPA to FPA coordinate transformation (unit conversion)."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_roundtrip_mpa_fpa_mpa(self, optical_model, sca):
        """MPA -> FPA -> MPA should recover original coordinates."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        np.random.seed(42)
        test_xmpa_orig = np.random.uniform(-50, 50, size=10)  # mm
        test_ympa_orig = np.random.uniform(-50, 50, size=10)  # mm

        # Forward
        xfpa, yfpa = omj.mpa_to_fpa(payload, test_xmpa_orig, test_ympa_orig)

        # Backward
        xmpa_recovered, ympa_recovered = omj.fpa_to_mpa(payload, xfpa, yfpa)

        np.testing.assert_allclose(
            xmpa_recovered, test_xmpa_orig, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            ympa_recovered, test_ympa_orig, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_roundtrip_fpa_mpa_fpa(self, optical_model, sca):
        """FPA -> MPA -> FPA should recover original coordinates."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        np.random.seed(42)
        test_xfpa_orig = np.random.uniform(-0.1, 0.1, size=10)  # degrees
        test_yfpa_orig = np.random.uniform(-0.1, 0.1, size=10)  # degrees

        # Forward
        xmpa, ympa = omj.fpa_to_mpa(payload, test_xfpa_orig, test_yfpa_orig)

        # Backward
        xfpa_recovered, yfpa_recovered = omj.mpa_to_fpa(payload, xmpa, ympa)

        np.testing.assert_allclose(
            xfpa_recovered, test_xfpa_orig, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            yfpa_recovered, test_yfpa_orig, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    def test_compare_to_class(self, optical_model, sca):
        """Compare against class-based implementation."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order="1")
        
        np.random.seed(42)
        test_xmpa = np.random.uniform(-50, 50, size=5)
        test_ympa = np.random.uniform(-50, 50, size=5)

        xfpa_jax, yfpa_jax = omj.mpa_to_fpa(payload, test_xmpa, test_ympa)
        xfpa_class, yfpa_class = optical_model.coords.convert_mpa_to_fpa(
            xmpa=test_xmpa, ympa=test_ympa
        )

        np.testing.assert_allclose(xfpa_jax, xfpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(yfpa_jax, yfpa_class, rtol=RTOL, atol=ATOL)


class TestJAXCompatibility:
    """Verify functions work with JAX arrays and are JIT-compilable."""

    def test_jax_array_input(self, optical_model):
        """Functions should accept JAX arrays as input."""
        import jax.numpy as jnp
        
        payload = omj.make_sca_payload(optical_model, sca=1, order="1")
        
        # Use JAX arrays as input
        xsca = jnp.array([1000.0, 2000.0, 3000.0])
        ysca = jnp.array([1500.0, 2500.0, 3500.0])
        
        # Test multiple transforms
        xmpa, ympa = omj.sca_to_mpa(payload, xsca, ysca)
        xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)
        
        # Output should be JAX arrays
        assert isinstance(xmpa, jnp.ndarray)
        assert isinstance(ympa, jnp.ndarray)
        assert isinstance(xfpa, jnp.ndarray)
        assert isinstance(yfpa, jnp.ndarray)

    def test_jit_compilation(self, optical_model):
        """Functions should be JIT-compilable."""
        import jax
        import jax.numpy as jnp
        
        payload = omj.make_sca_payload(optical_model, sca=1, order="1")
        
        # JIT compile individual transforms
        @jax.jit
        def jitted_sca_to_mpa(xsca, ysca):
            return omj.sca_to_mpa(payload, xsca, ysca)
        
        @jax.jit
        def jitted_roundtrip(xsca, ysca):
            xmpa, ympa = omj.sca_to_mpa(payload, xsca, ysca)
            return omj.mpa_to_sca(payload, xmpa, ympa)
        
        xsca = jnp.array([1000.0, 2000.0])
        ysca = jnp.array([1500.0, 2500.0])
        
        # Should compile and run without error
        xmpa, ympa = jitted_sca_to_mpa(xsca, ysca)
        xsca_rt, ysca_rt = jitted_roundtrip(xsca, ysca)
        
        # Verify outputs
        assert xmpa.shape == (2,)
        assert ympa.shape == (2,)
        np.testing.assert_allclose(xsca_rt, xsca, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ysca_rt, ysca, rtol=RTOL, atol=ATOL)


class TestMapCoords:
    """Test polynomial coordinate mapping (get_mpa_coords)."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_compare_to_class(self, optical_model, sca, order):
        """Compare functional get_mpa_coords to class implementation."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order=order)
        
        # Generate test points in SCA coordinates and convert to FPA
        xsca = np.array([500.0, 1500.0, 2500.0, 3500.0])
        ysca = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        
        # Convert to FPA using class method (returns numpy arrays)
        xfpa, yfpa = optical_model.coords.convert_sca_to_fpa(
            xsca=xsca, ysca=ysca, sca=sca
        )
        
        # Test functional implementation
        xmpa_jax, ympa_jax = omj.get_mpa_coords(payload, xfpa, yfpa)
        
        # Compare to class implementation
        xmpa_class, ympa_class = optical_model.get_map_coords(
            xfpa=xfpa, yfpa=yfpa, order=order
        )
        
        np.testing.assert_allclose(xmpa_jax, xmpa_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ympa_jax, ympa_class, rtol=RTOL, atol=ATOL)

    def test_jit_compilation(self, optical_model):
        """Verify get_mpa_coords is JIT-compilable."""
        import jax
        import jax.numpy as jnp
        
        payload = omj.make_sca_payload(optical_model, sca=5, order="2")
        
        @jax.jit
        def jitted_get_mpa_coords(xfpa, yfpa):
            return omj.get_mpa_coords(payload, xfpa, yfpa)
        
        xfpa = jnp.array([0.001, 0.002, -0.001])
        yfpa = jnp.array([-0.002, 0.003, 0.001])
        
        # Should compile and run
        xmpa, ympa = jitted_get_mpa_coords(xfpa, yfpa)
        
        assert xmpa.shape == (3,)
        assert ympa.shape == (3,)
        assert isinstance(xmpa, jnp.ndarray)
        assert isinstance(ympa, jnp.ndarray)


class TestTraceCoeffs:
    """Test polynomial trace coefficient computation (get_trace_coeffs)."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_compare_to_class(self, optical_model, sca, order):
        """Compare functional get_trace_coeffs to class implementation."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order=order)
        
        # Generate test points in SCA coordinates and convert to FPA
        xsca = np.array([500.0, 1500.0, 2500.0, 3500.0])
        ysca = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        
        # Convert to FPA using class method
        xfpa, yfpa = optical_model.coords.convert_sca_to_fpa(
            xsca=xsca, ysca=ysca, sca=sca
        )
        
        # Test functional implementation
        crv_jax, ids_jax = omj.get_trace_coeffs(payload, xfpa, yfpa)
        
        # Compare to class implementation
        crv_class, ids_class = optical_model.get_trace_coeffs(
            xfpa=xfpa, yfpa=yfpa, order=order
        )
        
        np.testing.assert_allclose(crv_jax, crv_class, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ids_jax, ids_class, rtol=RTOL, atol=ATOL)

    def test_output_shape(self, optical_model):
        """Test that output has correct shape [i, n]."""
        payload = omj.make_sca_payload(optical_model, sca=1, order="1")
        
        # Test points
        xfpa = np.array([0.001, 0.002, -0.001, 0.003])
        yfpa = np.array([-0.002, 0.003, 0.001, -0.001])
        
        crv, ids = omj.get_trace_coeffs(payload, xfpa, yfpa)
        
        # Shape should be [i, n]
        n = len(xfpa)
        crv_i = payload["poly"]["crv_i"]
        
        assert crv.shape == (crv_i, n)
        assert ids.shape == (crv_i, n)  # ids should have same i dimension

    def test_jit_compilation(self, optical_model):
        """Verify get_trace_coeffs is JIT-compilable."""
        import jax
        import jax.numpy as jnp
        
        payload = omj.make_sca_payload(optical_model, sca=5, order="2")
        
        @jax.jit
        def jitted_get_trace_coeffs(xfpa, yfpa):
            return omj.get_trace_coeffs(payload, xfpa, yfpa)
        
        xfpa = jnp.array([0.001, 0.002, -0.001])
        yfpa = jnp.array([-0.002, 0.003, 0.001])
        
        # Should compile and run
        crv, ids = jitted_get_trace_coeffs(xfpa, yfpa)
        
        assert crv.shape[1] == 3  # [i, n] shape, n=3
        assert ids.shape[1] == 3
        assert isinstance(crv, jnp.ndarray)
        assert isinstance(ids, jnp.ndarray)


class TestTraceBeam:
    """Test beam tracing (trace_beam)."""

    @pytest.mark.parametrize("sca", [1, 2, 5, 10])
    @pytest.mark.parametrize("order", ["1", "0", "2"])
    def test_compare_to_class(self, optical_model, sca, order):
        """Compare functional trace_beam to class implementation."""
        payload = omj.make_sca_payload(optical_model, sca=sca, order=order)
        
        # Generate ~1000 random SCA points and convert to FPA
        np.random.seed(42)
        n_points = 1000
        naxis1 = optical_model.detmod["naxis1"]
        naxis2 = optical_model.detmod["naxis2"]
        xsca = np.random.uniform(0.5, naxis1 + 0.5, size=n_points)
        ysca = np.random.uniform(0.5, naxis2 + 0.5, size=n_points)
        xfpa, yfpa = optical_model.coords.convert_sca_to_fpa(
            xsca=xsca, ysca=ysca, sca=sca
        )
        
        # Extract the wavelength grid used by the class
        wl_array = optical_model.wl_grid  # shape [n_wavelengths]
        n_wl = len(wl_array)
        
        # Compute class traces for each FPA point (width=1 per point)
        trace_mpa_x = np.empty((n_points, n_wl))
        trace_mpa_y = np.empty((n_points, n_wl))
        for i in range(n_points):
            coeff = optical_model._get_beam_trace(
                xref_fpa=xfpa[i], yref_fpa=yfpa[i], sca=sca, width=1, order=order
            )
            trace_mpa_x[i, :] = coeff["trace_mpa_x"].reshape(-1)
            trace_mpa_y[i, :] = coeff["trace_mpa_y"].reshape(-1)
        
        # Compute JAX trace_beam for all (xfpa, yfpa, wl) combinations at once
        xfpa_full = np.repeat(xfpa, n_wl)
        yfpa_full = np.repeat(yfpa, n_wl)
        wl_full = np.tile(wl_array, n_points)
        xmpa_jax, ympa_jax = omj.trace_beam(payload, xfpa_full, yfpa_full, wl_full)
        xmpa_jax = xmpa_jax.reshape(n_points, n_wl)
        ympa_jax = ympa_jax.reshape(n_points, n_wl)
        
        # Compare JAX vs class
        np.testing.assert_allclose(xmpa_jax, trace_mpa_x, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(ympa_jax, trace_mpa_y, rtol=RTOL, atol=ATOL)

    def test_single_wavelength(self, optical_model):
        """Test trace_beam with a single wavelength per FPA point."""
        payload = omj.make_sca_payload(optical_model, sca=5, order="1")
        
        xfpa = np.array([0.001, 0.002, -0.001])
        yfpa = np.array([-0.002, 0.003, 0.001])
        wavelength = np.array([1.5, 1.6, 1.7])  # Different wavelength for each point
        
        xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wavelength)
        
        assert xmpa.shape == (3,)
        assert ympa.shape == (3,)
        assert isinstance(xmpa, jnp.ndarray)
        assert isinstance(ympa, jnp.ndarray)

    def test_jit_compilation(self, optical_model):
        """Verify trace_beam is JIT-compilable."""
        import jax
        import jax.numpy as jnp
        
        payload = omj.make_sca_payload(optical_model, sca=5, order="2")
        
        @jax.jit
        def jitted_trace_beam(xfpa, yfpa, wavelength):
            return omj.trace_beam(payload, xfpa, yfpa, wavelength)
        
        xfpa = jnp.array([0.001, 0.002, -0.001])
        yfpa = jnp.array([-0.002, 0.003, 0.001])
        wavelength = jnp.array([1.5, 1.6, 1.7])
        
        # Should compile and run
        xmpa, ympa = jitted_trace_beam(xfpa, yfpa, wavelength)
        
        assert xmpa.shape == (3,)
        assert ympa.shape == (3,)
        assert isinstance(xmpa, jnp.ndarray)
        assert isinstance(ympa, jnp.ndarray)


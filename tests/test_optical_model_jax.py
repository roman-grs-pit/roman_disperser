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

# Roman WFI plate scale, for stating position tolerances in pixels.
#
# ATOL above is in *degrees*, which at 0.11 arcsec/px is 33 px -- uselessly
# loose for a position comparison, and one of the three reasons the 2026-07
# TF32 defect (1.84 px median) went unnoticed. Position assertions below use
# PIXEL_SCALE_ARCSEC to state their tolerance in pixels instead.
PIXEL_SCALE_ARCSEC = 0.11


def deg_to_px(x):
    """Convert an angular quantity in degrees to WFI pixels."""
    return np.asarray(x) * 3600.0 / PIXEL_SCALE_ARCSEC


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


class TestGetPARotation:
    """Test get_pa_rotation standalone function."""

    @pytest.mark.parametrize("pa", [0.0, 45.0, 90.0, 180.0, 270.0, -30.0])
    def test_compare_to_class(self, optical_model, pa):
        """JAX get_pa_rotation should match class method."""
        rot_jax = omj.get_pa_rotation(pa)
        rot_class = optical_model.coords.get_pa_rotation(pa=pa)

        np.testing.assert_allclose(rot_jax, rot_class, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("pa", [0.0, 45.0, 90.0, 180.0])
    def test_orthogonality(self, pa):
        """Rotation matrix should be orthogonal: R @ R.T = I."""
        rot = omj.get_pa_rotation(pa)
        product = rot @ rot.T
        np.testing.assert_allclose(product, jnp.eye(2), rtol=RTOL, atol=ATOL)

    def test_jit_compilation(self):
        """Verify get_pa_rotation is JIT-compilable."""
        @jax.jit
        def jitted_rotation(pa):
            return omj.get_pa_rotation(pa)

        rot = jitted_rotation(45.0)
        assert rot.shape == (2, 2)
        assert isinstance(rot, jnp.ndarray)


class TestGetFPAPos:
    """Test the sky -> FPA transform.

    Tolerances here are stated in **pixels**, never in degrees. The suite that
    shipped the 2026-07 TF32 defect compared FPA positions in degrees at
    ATOL = 1e-3, which is 33 px -- loose enough to pass a 1.84 px displacement
    without complaint.
    """

    def test_matches_legacy_flatsky_reference(self, optical_model):
        """Agreement with the vendored NumPy implementation, in pixels.

        `optical_model.coords.calculate_fpa_pos` is the vendored float64 NumPy
        twin. It is a *weak* oracle -- it shares this code's lineage and its
        flat-sky approximation, so it agrees for the wrong reason -- but with
        both sides now differencing at float64 it pins the rotation and scaling
        to float32 round-off.

        This assertion is expected to **loosen deliberately** when the gnomonic
        projection lands: at that point the two implementations genuinely
        differ, and this test becomes a characterisation of the flat-sky minus
        gnomonic offset rather than an equality. See
        `test_flatsky_vs_gnomonic_offset_is_understood` below.
        """
        np.random.seed(456)
        pointing_ra, pointing_dec, pointing_pa = 200.0, -30.0, 120.0
        ra = pointing_ra + np.random.uniform(-0.05, 0.05, size=10)
        dec = pointing_dec + np.random.uniform(-0.05, 0.05, size=10)

        xfpa_jax, yfpa_jax = omj.get_fpa_pos(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )
        xfpa_class, yfpa_class = optical_model.coords.calculate_fpa_pos(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )

        dx_px = deg_to_px(np.abs(np.asarray(xfpa_jax) - xfpa_class))
        dy_px = deg_to_px(np.abs(np.asarray(yfpa_jax) - yfpa_class))

        # float32 round-off on a ~0.05 deg offset is ~1e-4 px; 1e-3 px is a
        # comfortable ceiling that would still have caught the 1.84 px defect
        # by three orders of magnitude.
        assert dx_px.max() < 1e-3, f"x differs by {dx_px.max():.2e} px"
        assert dy_px.max() < 1e-3, f"y differs by {dy_px.max():.2e} px"

    @pytest.mark.parametrize(
        "pointing_ra", [10.0, 150.0, 260.0, 350.0],
        ids=["ra010", "ra150", "ra260", "ra350"],
    )
    def test_accuracy_is_independent_of_pointing_ra(self, pointing_ra):
        """The float64 differencing must remove the pointing-RA dependence.

        Regression test for the silent float32 downcast at the old
        `build_grism_image.py:614` call site. Passing absolute RA through
        `jnp.array()` quantised it *before* the subtraction, so the placement
        error scaled with the magnitude of RA -- 0.006 px at RA 10, but
        0.40 px at RA 260, where float32 ulp is 0.11 arcsec, a full pixel.

        Differencing at float64 first makes the residual depend only on the
        (small) offset, so all four pointings must agree to the same accuracy.
        """
        pointing_dec, pointing_pa = 0.0, 0.0
        np.random.seed(11)
        dra = np.random.uniform(-0.2, 0.2, size=64)
        ddec = np.random.uniform(-0.2, 0.2, size=64)
        ra = pointing_ra + dra
        dec = pointing_dec + ddec

        xfpa, yfpa = omj.get_fpa_pos(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )

        # float64 reference for the same transform.
        dx64 = (np.float64(ra) - np.float64(pointing_ra)) * np.cos(
            np.deg2rad(np.float64(dec))
        )
        dy64 = np.float64(dec) - np.float64(pointing_dec)
        theta = np.deg2rad(pointing_pa + 180 - 60)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        xy64 = rot @ np.stack([dx64, dy64])

        err_px = deg_to_px(
            np.hypot(np.asarray(xfpa) + xy64[0], np.asarray(yfpa) + xy64[1])
        )
        assert err_px.max() < 1e-2, (
            f"pointing RA {pointing_ra}: max error {err_px.max():.4f} px -- "
            "accuracy still depends on absolute RA, so the differencing is "
            "happening after a float32 downcast"
        )

    def test_offsets_are_float64_on_the_host(self):
        """`sky_to_tangent_offsets` must not hand back float32.

        The whole fix is that the subtraction happens at float64 before JAX
        sees it. If this returns float32 the precision has already been lost,
        no matter what the rest of the pipeline does.
        """
        ra = np.array([260.1, 260.2, 259.9])
        dec = np.array([0.1, -0.1, 0.05])
        dx, dy = omj.sky_to_tangent_offsets(ra, dec, 260.0, 0.0)
        assert dx.dtype == np.float64
        assert dy.dtype == np.float64

    def test_vectorized(self):
        """Verify works with arrays of (ra, dec) and scalar pointing params."""
        pointing_ra, pointing_dec, pointing_pa = 100.0, 10.0, 90.0
        ra = np.array([100.01, 100.02, 99.99])
        dec = np.array([10.01, 9.99, 10.02])

        xfpa, yfpa = omj.get_fpa_pos(ra, dec, pointing_ra, pointing_dec,
                                     pointing_pa)

        assert xfpa.shape == (3,)
        assert yfpa.shape == (3,)

    def test_jit_compilation(self):
        """The JAX half must stay jit-compilable.

        `get_fpa_pos` itself is deliberately *not* jittable any more: its
        float64 differencing is host NumPy, which is the point. The jittable
        unit is now `get_fpa_pos_from_offsets`, which is the part that belongs
        on the device. In production the host half runs once per pointing over
        a few thousand sources and is not on any hot path.
        """
        @jax.jit
        def jitted_from_offsets(dx, dy, ppa):
            return omj.get_fpa_pos_from_offsets(dx, dy, ppa)

        dx = jnp.array([0.01, 0.02])
        dy = jnp.array([0.01, -0.01])

        xfpa, yfpa = jitted_from_offsets(dx, dy, 60.0)
        assert xfpa.shape == (2,)
        assert yfpa.shape == (2,)
        assert isinstance(xfpa, jnp.ndarray)

    def test_composed_transform_is_not_jittable(self):
        """Document the contract change: get_fpa_pos is host code now.

        Pinned as a test rather than only a docstring so that a future attempt
        to jit the whole transform fails loudly here, with the reason attached,
        instead of silently reintroducing a float32 downcast of absolute RA.
        """
        @jax.jit
        def jitted(ra, dec):
            return omj.get_fpa_pos(ra, dec, 150.0, 2.0, 60.0)

        with pytest.raises(Exception):
            jitted(jnp.array([150.01, 150.02]), jnp.array([2.01, 1.99]))


class TestSkyToFPAMeridian:
    """Behaviour of the sky -> FPA transform at the RA = 0 boundary.

    The flat-sky transform computes `ra - pointing_ra` with no wrap handling,
    so a field straddling RA = 0 places its sources ~360 deg off the focal
    plane, where the detector bounding box silently culls them -- they vanish
    from both the image and the truth table with no error and no NaN. The
    haversine in `cone_search` *is* wrap-safe, so they are correctly selected
    first, which is what makes the loss invisible.

    Stage 1 therefore refuses such a field rather than mis-placing it. The
    gnomonic projection fixes it properly (sin/cos of the RA difference are
    periodic), at which point the xfail below flips to a pass -- and because it
    is `strict`, the suite goes red and forces the marker to be removed.

    See issue #19.
    """

    def test_meridian_crossing_field_raises_rather_than_dropping_sources(self):
        """Until gnomonic lands, a wrapped field must fail loudly."""
        # Pointing just east of RA = 0, sources just west of it: 0.2 deg apart
        # on the sky, but 359.8 deg apart by naive subtraction.
        ra = np.array([359.9, 359.95, 0.05])
        dec = np.array([0.0, 0.01, -0.01])

        with pytest.raises(ValueError, match="cross"):
            omj.sky_to_tangent_offsets(ra, dec, 0.1, 0.0)

    @pytest.mark.xfail(
        strict=True,
        reason="flat-sky transform has no RA wrap handling; fixed by the "
               "gnomonic projection (stage 2). See issue #19. When "
               "this XPASSes, delete the marker and the guard in "
               "sky_to_tangent_offsets.",
    )
    def test_meridian_crossing_field_is_placed_correctly(self):
        """A field straddling RA = 0 must be placed as a contiguous field.

        Sources 0.2 deg apart on the sky must land 0.2 deg apart on the focal
        plane regardless of which side of the meridian they sit on.
        """
        pointing_ra, pointing_dec, pointing_pa = 0.1, 0.0, 0.0
        ra = np.array([359.9, 0.3])   # -0.2 and +0.2 deg from the pointing
        dec = np.array([0.0, 0.0])

        xfpa, yfpa = omj.get_fpa_pos(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )

        sep_px = deg_to_px(
            np.hypot(np.asarray(xfpa)[1] - np.asarray(xfpa)[0],
                     np.asarray(yfpa)[1] - np.asarray(yfpa)[0])
        )
        expected_px = deg_to_px(0.4)
        assert abs(sep_px - expected_px) < 1.0, (
            f"separation {sep_px:.1f} px, expected {expected_px:.1f} px"
        )

    def test_ordinary_field_is_not_flagged_as_wrapped(self):
        """The guard must not false-positive on any normal pointing."""
        for pointing_ra in [0.5, 10.0, 150.0, 260.0, 359.5]:
            ra = pointing_ra + np.array([-0.3, 0.0, 0.3])
            ra = np.mod(ra, 360.0)
            dec = np.array([0.0, 0.0, 0.0])
            if np.abs(ra - pointing_ra).max() > 180.0:
                continue  # genuinely wrapped; covered by the test above
            omj.sky_to_tangent_offsets(ra, dec, pointing_ra, 0.0)


class TestSkyToFPAAgainstAstropy:
    """Independent oracle: astropy's TAN (gnomonic) WCS.

    The vendored NumPy twin shares this code's lineage *and* its flat-sky
    approximation, so it cannot detect an error in the projection itself --
    "a bug at both ends". astropy is a genuinely external implementation and is
    therefore the oracle that matters for the gnomonic stage.

    Comparisons here are made on **radial distance from the pointing**, which is
    invariant under the PA rotation and the axis negations in
    `get_fpa_pos_from_offsets`. That isolates the *projection*, which is what
    astropy is authoritative about; the rotation and sign conventions are
    covered separately by `test_matches_legacy_flatsky_reference`. Matching the
    full PA/CDELT sign convention here as well would test the harness more than
    the code.

    In stage 1 the code is still flat-sky, so these assert the *size* of the
    known approximation rather than agreement. When the gnomonic projection
    lands, `test_flatsky_error_grows_with_declination` should fail loudly --
    that is the intended signal to retire it and tighten the bound.
    """

    @staticmethod
    def _tan_offsets(ra, dec, pointing_ra, pointing_dec):
        """Gnomonic (TAN) tangent-plane offsets in degrees, from astropy.

        NOTE: `world_to_pixel` returns **0-based** pixel coordinates while
        `crpix` is 1-based FITS, so `crpix = [1, 1]` -- not [0, 0] -- puts the
        tangent point at the origin. Getting this wrong displaces everything by
        exactly (1, 1) deg, i.e. sqrt(2) deg = 46284 px, which is how it was
        caught.
        """
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [pointing_ra, pointing_dec]
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.cdelt = [1.0, 1.0]  # 1 "pixel" == 1 degree
        x, y = w.world_to_pixel(SkyCoord(np.asarray(ra) * u.deg,
                                         np.asarray(dec) * u.deg))
        return np.asarray(x), np.asarray(y)

    @staticmethod
    def _radius_px(xfpa, yfpa):
        """Radial distance from the pointing, in pixels.

        Invariant under the PA rotation and the axis negations, so it compares
        projections rather than conventions.
        """
        return deg_to_px(np.hypot(np.asarray(xfpa), np.asarray(yfpa)))

    def test_tangent_point_is_exact(self):
        """A source at the pointing must land at the origin in both.

        Guards the harness: if the astropy WCS is misconfigured (the crpix
        off-by-one above), this fails before any science assertion does.
        """
        x, y = self._tan_offsets([150.0], [30.0], 150.0, 30.0)
        assert deg_to_px(np.hypot(x, y)).max() < 1e-6

        xfpa, yfpa = omj.get_fpa_pos(
            np.array([150.0]), np.array([30.0]), 150.0, 30.0, 45.0
        )
        assert self._radius_px(xfpa, yfpa).max() < 1e-6

    @pytest.mark.parametrize(
        "pointing_dec,max_px",
        [(0.0, 2.0), (30.0, 40.0), (60.0, 120.0)],
        ids=["dec00", "dec30", "dec60"],
    )
    def test_flatsky_error_grows_with_declination(self, pointing_dec, max_px):
        """Characterise the flat-sky approximation against gnomonic truth.

        This is a *characterisation*, not a correctness assertion: it pins the
        size of the approximation stage 1 knowingly carries, so the gnomonic
        stage can be shown to remove it. Measured over a +/-0.4 deg field:

            pointing Dec    median      max
                     0     0.12 px    0.72 px
                    30     3.19 px   18.85 px
                    60     9.64 px   54.81 px

        Note this exceeds the 1.84 px median TF32 error for any pointing off
        the equator -- flat-sky is the larger of the two defects away from
        Dec = 0.

        The upper bounds are loose ceilings; the lower bound is the load-bearing
        assertion, since a zero here would mean the oracle is not independent.
        """
        pointing_ra, pointing_pa = 150.0, 0.0
        np.random.seed(7)
        n = 512
        ddec = np.random.uniform(-0.4, 0.4, size=n)
        dra = np.random.uniform(-0.4, 0.4, size=n) / np.cos(
            np.deg2rad(pointing_dec)
        )
        ra, dec = pointing_ra + dra, pointing_dec + ddec

        xfpa, yfpa = omj.get_fpa_pos(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )
        x_tan, y_tan = self._tan_offsets(ra, dec, pointing_ra, pointing_dec)

        diff_px = np.abs(
            self._radius_px(xfpa, yfpa) - deg_to_px(np.hypot(x_tan, y_tan))
        )

        assert diff_px.max() > 1e-3, (
            "flat-sky and gnomonic agree exactly, which is impossible over a "
            "0.4 deg field -- the astropy oracle is not actually independent"
        )
        assert diff_px.max() < max_px, (
            f"flat-sky vs gnomonic differs by {diff_px.max():.2f} px at Dec "
            f"{pointing_dec}, more than the approximation predicts "
            f"({max_px} px) -- likely a convention error, not a projection "
            "difference"
        )

    def test_agreement_is_excellent_for_a_small_field(self):
        """Flat-sky and gnomonic must converge as the field shrinks.

        Independent of the projection argument: any correct tangent-plane
        transform agrees with flat-sky to second order in the offset. This
        assertion therefore holds in **both** stages, and is the one that
        catches a gross convention error rather than an approximation.
        """
        pointing_ra, pointing_dec, pointing_pa = 150.0, 30.0, 45.0
        np.random.seed(3)
        n = 64
        ddec = np.random.uniform(-1e-3, 1e-3, size=n)
        dra = np.random.uniform(-1e-3, 1e-3, size=n) / np.cos(
            np.deg2rad(pointing_dec)
        )
        ra, dec = pointing_ra + dra, pointing_dec + ddec

        xfpa, yfpa = omj.get_fpa_pos(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )
        x_tan, y_tan = self._tan_offsets(ra, dec, pointing_ra, pointing_dec)

        diff_px = np.abs(
            self._radius_px(xfpa, yfpa) - deg_to_px(np.hypot(x_tan, y_tan))
        )
        assert diff_px.max() < 0.01, (
            f"differs by {diff_px.max():.4f} px over a 1e-3 deg field, where "
            "flat-sky and gnomonic must coincide -- this is a convention bug, "
            "not a projection difference"
        )


class TestSkyToFPAGoldenValues:
    """Golden values for the sky -> FPA transform, computed at float64.

    Independent of both the vendored twin and astropy: if all three ever agree
    on something wrong, these still pin the numbers that were reviewed. They
    are the flat-sky values, so the gnomonic stage will need to regenerate them
    -- deliberately, and as a reviewed diff rather than a silent drift.
    """

    # (ra, dec, pointing_ra, pointing_dec, pointing_pa) -> (xfpa, yfpa) in deg,
    # evaluated in float64 NumPy from the flat-sky definition.
    CASES = [
        # Equatorial, the SSC line-grid pointing.
        (10.2, 0.15, 10.0, 0.0, 0.0),
        # Mid-RA, where the old float32 downcast cost ~0.1 px.
        (150.35, 2.1, 150.0, 2.0, 60.0),
        # High RA, where the old downcast cost ~0.4 px.
        (260.3, -10.2, 260.0, -10.0, 120.0),
        # High declination, where flat-sky departs most from gnomonic.
        (150.4, 60.25, 150.0, 60.0, 33.0),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: f"ra{c[2]:g}")
    def test_matches_float64_reference(self, case):
        ra, dec, pra, pdec, ppa = case

        xfpa, yfpa = omj.get_fpa_pos(
            np.array([ra]), np.array([dec]), pra, pdec, ppa
        )

        dx = (np.float64(ra) - np.float64(pra)) * np.cos(
            np.deg2rad(np.float64(dec))
        )
        dy = np.float64(dec) - np.float64(pdec)
        theta = np.deg2rad(ppa + 180.0 - 60.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        xy = rot @ np.array([dx, dy])

        err_px = deg_to_px(
            np.hypot(float(np.asarray(xfpa)[0]) + xy[0],
                     float(np.asarray(yfpa)[0]) + xy[1])
        )
        assert err_px < 1e-2, f"{err_px:.4e} px from the float64 reference"


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


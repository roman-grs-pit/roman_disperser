"""
Tests for the sersic module.

Validates:
1. b_n lookup accuracy vs scipy
2. Circular and elliptical profiles vs astropy Sersic2D
3. Normalization, symmetry, PA rotation
4. Batch consistency
5. sky_pa_to_sca_theta conversion
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaincinv

from roman_disperser.sersic import (
    compute_bn,
    make_sersic_image,
    make_sersic_images,
    catalog_r_eff_to_pixels,
    sky_pa_to_sca_theta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _astropy_sersic2d(npix, amplitude, r_eff, n, x_0, y_0, ellip, theta):
    """Generate a Sérsic profile using astropy for comparison."""
    from astropy.modeling.models import Sersic2D

    model = Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff,
        n=n,
        x_0=x_0,
        y_0=y_0,
        ellip=ellip,
        theta=theta,
    )
    y, x = np.mgrid[0:npix, 0:npix]
    return model(x, y)


# ---------------------------------------------------------------------------
# Test b_n accuracy
# ---------------------------------------------------------------------------


class TestComputeBn:
    """Test b_n lookup against scipy."""

    @pytest.mark.parametrize("n", [0.5, 1.0, 2.0, 4.0, 10.0])
    def test_bn_vs_scipy(self, n):
        """b_n should match scipy.special.gammaincinv to <0.01%."""
        expected = gammaincinv(2.0 * n, 0.5)
        result = float(compute_bn(jnp.float32(n)))
        rtol = 1e-4  # 0.01%
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0.5, 0.6931472),   # Gaussian: b_n = ln(2)
            (1.0, 1.6783470),   # Exponential
            (4.0, 7.6692494),   # de Vaucouleurs
        ],
    )
    def test_bn_known_values(self, n, expected):
        """b_n should match well-known analytic/tabulated values."""
        result = float(compute_bn(jnp.float32(n)))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_bn_vectorized(self):
        """compute_bn should work on arrays."""
        ns = jnp.array([0.5, 1.0, 2.0, 4.0])
        result = compute_bn(ns)
        assert result.shape == (4,)
        # Verify against scipy for each element
        expected = gammaincinv(2.0 * np.array([0.5, 1.0, 2.0, 4.0]), 0.5)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test circular profiles
# ---------------------------------------------------------------------------


class TestCircularProfile:
    """Test round profiles (ba=1) against astropy."""

    @pytest.mark.parametrize("n", [1.0, 4.0])
    def test_vs_astropy(self, n):
        """Circular Sérsic should match astropy Sersic2D pixel-by-pixel."""
        npix = 101
        r_eff = 15.0
        center = (npix - 1) / 2.0

        # Our implementation
        img_jax = np.array(make_sersic_image(r_eff, n, 1.0, 0.0, npix))

        # Astropy reference (ellip=0 for circular)
        bn = gammaincinv(2.0 * n, 0.5)
        amplitude = np.exp(bn)  # I(r=0) for normalized profile
        img_astropy = _astropy_sersic2d(
            npix, amplitude, r_eff, n, center, center, ellip=0.0, theta=0.0
        )
        img_astropy /= img_astropy.sum()

        np.testing.assert_allclose(img_jax, img_astropy, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Test elliptical profiles
# ---------------------------------------------------------------------------


class TestEllipticalProfile:
    """Test elliptical profiles against astropy."""

    def test_elliptical_vs_astropy(self):
        """Elliptical Sérsic (ba=0.5, theta=pi/4) should match astropy."""
        npix = 101
        r_eff = 15.0
        n = 2.0
        ba = 0.5
        theta = np.pi / 4.0
        center = (npix - 1) / 2.0

        img_jax = np.array(make_sersic_image(r_eff, n, ba, theta, npix))

        bn = gammaincinv(2.0 * n, 0.5)
        amplitude = np.exp(bn)
        ellip = 1.0 - ba
        img_astropy = _astropy_sersic2d(
            npix, amplitude, r_eff, n, center, center, ellip=ellip, theta=theta
        )
        img_astropy /= img_astropy.sum()

        np.testing.assert_allclose(img_jax, img_astropy, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Test normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    """Verify images are normalized to sum=1."""

    @pytest.mark.parametrize(
        "n,ba",
        [(0.5, 1.0), (1.0, 0.5), (4.0, 0.3), (2.0, 1.0)],
    )
    def test_sum_is_one(self, n, ba):
        img = make_sersic_image(10.0, n, ba, 0.0, 101)
        np.testing.assert_allclose(float(img.sum()), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test symmetry
# ---------------------------------------------------------------------------


class TestSymmetry:
    """Round profile should be symmetric under 90° rotation."""

    def test_90_degree_rotation_symmetry(self):
        npix = 101
        img = np.array(make_sersic_image(15.0, 2.0, 1.0, 0.0, npix))
        img_rot90 = np.rot90(img)
        np.testing.assert_allclose(img, img_rot90, atol=1e-6)


# ---------------------------------------------------------------------------
# Test PA rotation
# ---------------------------------------------------------------------------


class TestPARotation:
    """Position angle should orient the major axis correctly."""

    def test_major_axis_along_x_at_theta_0(self):
        """theta=0 → major axis along x → Ixx > Iyy."""
        npix = 101
        img = np.array(make_sersic_image(15.0, 2.0, 0.3, 0.0, npix))
        center = (npix - 1) / 2.0
        y, x = np.mgrid[0:npix, 0:npix] - center
        Ixx = np.sum(img * x**2)
        Iyy = np.sum(img * y**2)
        assert Ixx > Iyy, f"Expected Ixx > Iyy, got {Ixx:.4f} vs {Iyy:.4f}"

    def test_major_axis_along_y_at_theta_pi2(self):
        """theta=π/2 → major axis along y → Iyy > Ixx."""
        npix = 101
        img = np.array(make_sersic_image(15.0, 2.0, 0.3, np.pi / 2, npix))
        center = (npix - 1) / 2.0
        y, x = np.mgrid[0:npix, 0:npix] - center
        Ixx = np.sum(img * x**2)
        Iyy = np.sum(img * y**2)
        assert Iyy > Ixx, f"Expected Iyy > Ixx, got {Iyy:.4f} vs {Ixx:.4f}"


# ---------------------------------------------------------------------------
# Test batch consistency
# ---------------------------------------------------------------------------


class TestBatch:
    """make_sersic_images should match individual make_sersic_image calls."""

    def test_batch_matches_individual(self):
        npix = 51
        N = 10
        rng = np.random.default_rng(42)
        r_effs = rng.uniform(5.0, 20.0, N).astype(np.float32)
        ns = rng.uniform(0.5, 4.0, N).astype(np.float32)
        bas = rng.uniform(0.3, 1.0, N).astype(np.float32)
        thetas = rng.uniform(0.0, np.pi, N).astype(np.float32)

        # Batch
        batch_imgs = np.array(
            make_sersic_images(
                jnp.array(r_effs),
                jnp.array(ns),
                jnp.array(bas),
                jnp.array(thetas),
                npix,
            )
        )

        # Individual
        for i in range(N):
            single = np.array(
                make_sersic_image(r_effs[i], ns[i], bas[i], thetas[i], npix)
            )
            np.testing.assert_array_equal(
                batch_imgs[i], single,
                err_msg=f"Mismatch at galaxy {i}",
            )


# ---------------------------------------------------------------------------
# Test conversion helpers
# ---------------------------------------------------------------------------


class TestConversionHelpers:
    """Test catalog_r_eff_to_pixels and sky_pa_to_sca_theta."""

    def test_r_eff_conversion(self):
        """0.11 arcsec at native scale, 4× oversample → 4 oversampled pixels."""
        result = catalog_r_eff_to_pixels(0.11, pixel_scale=0.11, oversample=4)
        np.testing.assert_allclose(result, 4.0)

    def test_r_eff_conversion_half_arcsec(self):
        """0.5 arcsec → 0.5 / (0.11/4) ≈ 18.18 oversampled pixels."""
        result = catalog_r_eff_to_pixels(0.5)
        expected = 0.5 / (0.11 / 4)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestSkyPAToSCATheta:
    """Test sky PA → SCA theta conversion.

    Uses numerical Jacobian of the sky→SCA transform to verify the
    PA transformation formula.
    """

    def test_numerical_jacobian(self):
        """Verify PA conversion by computing Jacobian of sky→SCA.

        For dec≈0 where the small-angle approximation is accurate,
        the Jacobian of the sky→FPA→SCA transform tells us how
        sky directions map to pixel directions.
        """
        import roman_disperser.optical_model_jax as omj
        from roman_disperser import paths
        from roman_disperser.optical_model import RomanOpticalModel

        model = RomanOpticalModel(str(paths.optical_model_path()))
        payload = omj.make_sca_payload(model, sca=5, order="1")

        pointing_ra = 10.0
        pointing_dec = 0.5
        pointing_pa = 30.0

        # Source near pointing center (dec ≈ 0 for accuracy)
        src_ra = pointing_ra + 0.01
        src_dec = pointing_dec + 0.01

        # Compute FPA position. Float64 host arrays, not jnp.array(): the
        # eps = 1e-5 deg finite differences below are only ~10 float32 ulp at
        # RA = 10, so a float32 downcast here quantises the very step the
        # Jacobian is built from (and get_fpa_pos now refuses float32).
        xfpa, yfpa = omj.get_fpa_pos(
            np.array([src_ra]),
            np.array([src_dec]),
            pointing_ra,
            pointing_dec,
            pointing_pa,
        )

        # Compute SCA position
        xsca, ysca = omj.fpa_to_sca(payload, xfpa, yfpa)
        xsca, ysca = float(xsca[0]), float(ysca[0])

        # Numerical Jacobian of sky→SCA at this position
        eps = 1e-5  # degrees

        def sky_to_sca(ra, dec):
            xf, yf = omj.get_fpa_pos(
                np.array([ra]), np.array([dec]),
                pointing_ra, pointing_dec, pointing_pa,
            )
            xs, ys = omj.fpa_to_sca(payload, xf, yf)
            return float(xs[0]), float(ys[0])

        x_ra, y_ra = sky_to_sca(src_ra + eps, src_dec)
        x_dec, y_dec = sky_to_sca(src_ra, src_dec + eps)

        # Jacobian columns: d(sca)/d(ra), d(sca)/d(dec)
        dxdra = (x_ra - xsca) / eps
        dydra = (y_ra - ysca) / eps
        dxddec = (x_dec - xsca) / eps
        dyddec = (y_dec - ysca) / eps

        # A sky PA of α (E of N) means direction:
        # (d_ra, d_dec) = (sin α / cos(dec), cos α)
        # In SCA, this maps to:
        # dx_sca = dxdra * sin(α)/cos(dec) + dxddec * cos(α)
        # dy_sca = dydra * sin(α)/cos(dec) + dyddec * cos(α)
        # SCA theta = atan2(dy_sca, dx_sca)

        cos_dec = np.cos(np.deg2rad(src_dec))

        for pa_sky in [0.0, 45.0, 90.0, 135.0]:
            alpha = np.deg2rad(pa_sky)
            dx = dxdra * np.sin(alpha) / cos_dec + dxddec * np.cos(alpha)
            dy = dydra * np.sin(alpha) / cos_dec + dyddec * np.cos(alpha)
            theta_numerical = np.arctan2(dy, dx)

            theta_formula = float(sky_pa_to_sca_theta(pa_sky, pointing_pa))

            # Normalize both to [0, 2π] for comparison
            theta_numerical = theta_numerical % (2 * np.pi)
            theta_formula = theta_formula % (2 * np.pi)

            np.testing.assert_allclose(
                theta_formula, theta_numerical,
                atol=0.03,  # ~1.7 degrees tolerance for small-angle approximation
                err_msg=f"PA mismatch at sky_pa={pa_sky}°",
            )

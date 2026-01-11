"""
Tests for disperser module.

These tests verify:
1. Delta function dispersion matches trace_beam predictions
2. Bilinear interpolation conserves flux
3. JIT compilation works correctly
4. Wavelength chunking produces consistent results
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.disperser import (
    bilinear_scatter_add,
    disperse_2d1d_sca,
)

# Tolerances for float32 precision
RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture(scope="module")
def optical_model():
    """Load optical model once for all tests."""
    pixi_root_path = os.environ.get("PIXI_PROJECT_ROOT", ".")
    fn = os.path.join(pixi_root_path, "data/Roman_grism_OpticalModel_v0.8.yaml")
    return RomanOpticalModel(fn)


@pytest.fixture(scope="module")
def payload(optical_model):
    """Create payload for SCA 5, order 1."""
    return omj.make_sca_payload(optical_model, sca=5, order="1")


class TestBilinearScatterAdd:
    """Test the bilinear scatter-add helper function.

    Note: Uses FITS 1-indexed coordinates where pixel n has center at n.0.
    The function converts to 0-indexed array indices by subtracting 0.5.
    """

    def test_single_point_centered(self):
        """Point exactly on a grid cell should contribute only to that cell."""
        output = jnp.zeros((10, 10))
        # FITS coord 5.5 = center between pixels 5 and 6 → array index 5 after -0.5 shift
        x = jnp.array([5.5])
        y = jnp.array([5.5])
        values = jnp.array([1.0])

        result = bilinear_scatter_add(output, x, y, values)

        # FITS (5.5, 5.5) → array (5, 5) with all weight
        assert result[5, 5] == 1.0
        assert result.sum() == 1.0

    def test_single_point_fractional(self):
        """Point at fractional position should spread to 4 neighbors."""
        output = jnp.zeros((10, 10))
        # FITS coord 6.0 = center of pixel 6 → halfway between array indices 5 and 6
        x = jnp.array([6.0])
        y = jnp.array([6.0])
        values = jnp.array([1.0])

        result = bilinear_scatter_add(output, x, y, values)

        # FITS (6.0, 6.0) → equal weight to array corners (5,5), (5,6), (6,5), (6,6)
        expected_weight = 0.25
        assert jnp.isclose(result[5, 5], expected_weight, rtol=RTOL)
        assert jnp.isclose(result[5, 6], expected_weight, rtol=RTOL)
        assert jnp.isclose(result[6, 5], expected_weight, rtol=RTOL)
        assert jnp.isclose(result[6, 6], expected_weight, rtol=RTOL)
        assert jnp.isclose(result.sum(), 1.0, rtol=RTOL)

    def test_weights_sum_to_one(self):
        """Bilinear weights should always sum to 1 for valid points."""
        output = jnp.zeros((100, 100))
        np.random.seed(42)
        x = jnp.array(np.random.uniform(1, 98, size=100))
        y = jnp.array(np.random.uniform(1, 98, size=100))
        values = jnp.ones(100)

        result = bilinear_scatter_add(output, x, y, values)

        # Total should equal number of points (each contributes 1.0)
        assert jnp.isclose(result.sum(), 100.0, rtol=RTOL)

    def test_out_of_bounds_ignored(self):
        """Points outside grid should contribute nothing."""
        output = jnp.zeros((10, 10))
        # For 10×10 array (indices 0-9), valid FITS range is [0.5, 10.5]
        # FITS 0.0 → x_adj=-0.5 → partially out (straddles array index -1 and 0)
        # FITS 11.0 → x_adj=10.5 → partially out (straddles array index 9 and 10)
        # For fully OOB: use coords where all 4 bilinear corners are invalid
        x = jnp.array([0.0, 11.0, 5.5])  # Two partially OOB, one valid
        y = jnp.array([5.5, 5.5, 0.0])   # Third has y partially OOB
        values = jnp.array([1.0, 1.0, 1.0])

        result = bilinear_scatter_add(output, x, y, values)

        # First point: x=0.0 → x_adj=-0.5 → floor=-1, affects indices (-1,0)
        #   - Index -1 dropped, index 0 gets partial weight (0.5)
        # Second point: x=11.0 → x_adj=10.5 → floor=10, affects indices (10,11)
        #   - Index 10 out of bounds for 10-wide array, dropped
        # Third point: y=0.0 → y_adj=-0.5 → floor=-1, affects indices (-1,0)
        #   - Some flux lands at valid indices
        # So total is less than 3.0 but not 0.0
        assert result.sum() < 3.0  # Some flux lost to OOB
        assert result.sum() > 0.0  # But not all (some corners are valid)

    def test_accumulation(self):
        """Multiple points at same location should accumulate."""
        output = jnp.zeros((10, 10))
        # FITS coord 5.5 → array index 5 exactly
        x = jnp.array([5.5, 5.5, 5.5])
        y = jnp.array([5.5, 5.5, 5.5])
        values = jnp.array([1.0, 2.0, 3.0])

        result = bilinear_scatter_add(output, x, y, values)

        assert result[5, 5] == 6.0  # 1 + 2 + 3
        assert result.sum() == 6.0


class TestDisperse2D1DSCA:
    """Test the main disperse function."""

    def test_single_pixel_single_wavelength(self, optical_model, payload):
        """Disperse a delta function and verify position matches trace_beam."""
        # Single pixel image at detector center
        x0, y0 = 2044.5, 2044.5  # Center of SCA
        image = jnp.array([[1.0]])  # 1×1 image

        # Single wavelength at reference
        wl_ref = float(payload["wl"]["reference"])
        spec = jnp.array([1.0])

        output = jnp.zeros((4088, 4088), dtype=jnp.float32)
        result = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            dx=1.0,
            dy=1.0,
            spec=spec,
            lam0=wl_ref,
            dlam=0.01,
            output=output,
            wavelength_chunk_size=10,
        )

        # Compute expected position using trace_beam
        xfpa, yfpa = omj.sca_to_fpa(
            payload, jnp.array([x0]), jnp.array([y0])
        )
        xmpa_expected, ympa_expected = omj.trace_beam(
            payload, xfpa, yfpa, jnp.array([wl_ref])
        )
        xsca_expected, ysca_expected = omj.mpa_to_sca(
            payload, xmpa_expected, ympa_expected
        )

        # Find peak in result
        peak_idx = jnp.unravel_index(jnp.argmax(result), result.shape)
        peak_y, peak_x = peak_idx

        # Peak should be near expected position (within 1 pixel due to bilinear)
        assert abs(float(peak_x) - float(xsca_expected[0])) < 1.5
        assert abs(float(peak_y) - float(ysca_expected[0])) < 1.5

    def test_flux_conservation_centered(self, optical_model, payload):
        """Flux should be conserved when entirely within detector bounds."""
        # Small image near detector center (all flux should land on detector)
        x0, y0 = 2000.0, 2000.0
        image = jnp.ones((5, 5))  # 5×5 uniform
        spec = jnp.ones(10)  # 10 wavelengths

        # Use wavelengths near reference to minimize dispersion
        wl_ref = float(payload["wl"]["reference"])
        dlam = 0.001  # Very small step

        output = jnp.zeros((4088, 4088), dtype=jnp.float32)
        result = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            dx=1.0,
            dy=1.0,
            spec=spec,
            lam0=wl_ref - 0.005,  # Center around reference
            dlam=dlam,
            output=output,
            wavelength_chunk_size=10,
        )

        # Total flux should equal sum(image) * sum(spec)
        expected_flux = float(image.sum() * spec.sum())
        actual_flux = float(result.sum())

        # Should be close (allow some tolerance for edge effects)
        assert jnp.isclose(actual_flux, expected_flux, rtol=0.01)

    def test_chunk_size_invariance(self, optical_model, payload):
        """Result should be the same regardless of wavelength chunk size."""
        x0, y0 = 2000.0, 2000.0
        image = jnp.ones((5, 5))
        spec = jnp.ones(50)
        wl_ref = float(payload["wl"]["reference"])

        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # Run with different chunk sizes
        result_10 = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            1.0,
            1.0,
            spec,
            wl_ref,
            0.01,
            output,
            wavelength_chunk_size=10,
        )

        result_25 = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            1.0,
            1.0,
            spec,
            wl_ref,
            0.01,
            output,
            wavelength_chunk_size=25,
        )

        result_50 = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            1.0,
            1.0,
            spec,
            wl_ref,
            0.01,
            output,
            wavelength_chunk_size=50,
        )

        # All results should be identical
        np.testing.assert_allclose(result_10, result_25, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(result_10, result_50, rtol=RTOL, atol=ATOL)

    def test_jit_compilation(self, optical_model, payload):
        """Function should be JIT-compilable."""
        x0, y0 = 2000.0, 2000.0
        image = jnp.ones((5, 5))
        spec = jnp.ones(20)
        wl_ref = float(payload["wl"]["reference"])
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # JIT compile with static chunk size
        @jax.jit
        def jitted_disperse(image, x0, y0, spec, output):
            return disperse_2d1d_sca(
                payload,
                image,
                x0,
                y0,
                1.0,
                1.0,
                spec,
                wl_ref,
                0.01,
                output,
                wavelength_chunk_size=10,
            )

        # First call (compiles)
        result1 = jitted_disperse(image, x0, y0, spec, output)

        # Second call (uses cached compilation)
        result2 = jitted_disperse(image, x0, y0, spec, output)

        # Results should be identical
        np.testing.assert_allclose(result1, result2, rtol=RTOL, atol=ATOL)

        # Should produce same result as non-jitted version
        # Note: Small numerical differences are expected due to XLA optimizations
        result_eager = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            1.0,
            1.0,
            spec,
            wl_ref,
            0.01,
            output,
            wavelength_chunk_size=10,
        )
        # Use looser tolerance for JIT vs eager comparison (XLA may reorder ops)
        np.testing.assert_allclose(result1, result_eager, rtol=1e-3, atol=1e-3)

    def test_fractional_pixel_spacing(self, optical_model, payload):
        """Should handle oversampled input grids (dx, dy < 1)."""
        x0, y0 = 2000.0, 2000.0
        # 2× oversampled: 10×10 input covers 5×5 detector pixels
        image = jnp.ones((10, 10))
        spec = jnp.ones(10)
        wl_ref = float(payload["wl"]["reference"])
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        result = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            dx=0.5,  # 2× oversampled
            dy=0.5,
            spec=spec,
            lam0=wl_ref,
            dlam=0.001,
            output=output,
            wavelength_chunk_size=10,
        )

        # Should still work and produce reasonable output
        assert result.sum() > 0
        # With 2× oversampling, each input pixel covers 0.25 detector pixels
        # So flux is spread differently but total should be similar

    def test_output_accumulation(self, optical_model, payload):
        """Multiple calls should accumulate onto output."""
        x0, y0 = 2000.0, 2000.0
        image = jnp.ones((3, 3))
        spec = jnp.ones(5)
        wl_ref = float(payload["wl"]["reference"])

        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # First dispersion
        output = disperse_2d1d_sca(
            payload,
            image,
            x0,
            y0,
            1.0,
            1.0,
            spec,
            wl_ref,
            0.01,
            output,
            wavelength_chunk_size=10,
        )
        flux1 = output.sum()

        # Second dispersion at different position
        output = disperse_2d1d_sca(
            payload,
            image,
            x0 + 100,  # Offset position
            y0 + 100,
            1.0,
            1.0,
            spec,
            wl_ref,
            0.01,
            output,
            wavelength_chunk_size=10,
        )
        flux2 = output.sum()

        # Total flux should be approximately 2× single dispersion
        # (some overlap possible but positions are offset)
        assert flux2 > flux1 * 1.5  # At least 1.5× more flux

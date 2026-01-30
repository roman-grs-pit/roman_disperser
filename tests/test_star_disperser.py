"""
Tests for star_disperser module.

These tests verify:
1. PSF pixel grid creation
2. PSF deposition
3. Star dispersion with wavelength-dependent PSFs
4. JIT-compiled star disperser factory

Note: All wavelengths are in **microns** (consistent with optical model).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser.star_disperser import (
    make_psf_pixel_grid,
    deposit_psf,
    disperse_star_psf,
    make_star_disperser,
)

# Tolerances for float32 precision
RTOL = 1e-4
ATOL = 1e-4


class TestMakePsfPixelGrid:
    """Test the PSF pixel grid helper function."""

    def test_grid_shape(self):
        """Grid should have same shape as PSF."""
        rel_y, rel_x = make_psf_pixel_grid((182, 182), 4)
        assert rel_y.shape == (182, 182)
        assert rel_x.shape == (182, 182)

    def test_center_near_zero(self):
        """Grid center should be near (0, 0) offset."""
        rel_y, rel_x = make_psf_pixel_grid((182, 182), 4)
        # Center indices for 182×182 are around 90.5
        center_y, center_x = 90, 90
        # Offset should be small (sub-pixel due to even size)
        assert abs(rel_x[center_y, center_x]) < 0.5
        assert abs(rel_y[center_y, center_x]) < 0.5

    def test_offset_scaling(self):
        """Offsets should scale with 1/oversample."""
        rel_y_4, rel_x_4 = make_psf_pixel_grid((100, 100), 4)
        rel_y_2, rel_x_2 = make_psf_pixel_grid((100, 100), 2)

        # For same index offset from center, 2× oversample should have 2× the offset
        # Edge pixel at (0, 0):
        # For 4×: offset = (0 - 49.5) / 4 = -12.375
        # For 2×: offset = (0 - 49.5) / 2 = -24.75
        assert jnp.isclose(rel_x_2[0, 0], 2 * rel_x_4[0, 0], rtol=1e-5)

    def test_symmetric(self):
        """Grid should be symmetric about center."""
        rel_y, rel_x = make_psf_pixel_grid((100, 100), 4)
        # Symmetric: corner offsets should be opposite
        # For 100×100, center is at 49.5
        # Corners: (0,0) and (99,99)
        assert jnp.isclose(rel_x[0, 0], -rel_x[99, 99], rtol=1e-5)
        assert jnp.isclose(rel_y[0, 0], -rel_y[99, 99], rtol=1e-5)


class TestDepositPsf:
    """Test the PSF deposition function."""

    def test_single_pixel_psf(self):
        """Single pixel PSF should land in one detector pixel."""
        output = jnp.zeros((100, 100), dtype=jnp.float32)
        psf = jnp.array([[1.0]])  # 1×1 PSF
        rel_y, rel_x = jnp.array([[0.0]]), jnp.array([[0.0]])

        result = deposit_psf(output, 50.5, 50.5, psf, rel_x, rel_y)

        # FITS (50.5, 50.5) maps to array index 50
        assert result[50, 50] == 1.0
        assert result.sum() == 1.0

    def test_flux_conservation(self):
        """Total deposited flux should equal PSF sum."""
        output = jnp.zeros((100, 100), dtype=jnp.float32)
        # Create a small Gaussian-like PSF (9×9)
        psf = jnp.ones((9, 9), dtype=jnp.float32)
        rel_y, rel_x = make_psf_pixel_grid((9, 9), 1)  # 1× oversample for simplicity

        # Deposit at center of detector
        result = deposit_psf(output, 50.0, 50.0, psf, rel_x, rel_y)

        # All flux should land on detector
        assert jnp.isclose(result.sum(), psf.sum(), rtol=RTOL)

    def test_out_of_bounds_dropped(self):
        """Flux going off detector should be dropped silently."""
        output = jnp.zeros((100, 100), dtype=jnp.float32)
        psf = jnp.ones((20, 20), dtype=jnp.float32)
        rel_y, rel_x = make_psf_pixel_grid((20, 20), 1)

        # Deposit at corner - some flux will go off edge
        result = deposit_psf(output, 5.0, 5.0, psf, rel_x, rel_y)

        # Some flux should be lost
        assert result.sum() < psf.sum()
        assert result.sum() > 0  # But not all

    def test_accumulation(self):
        """Multiple deposits should accumulate."""
        output = jnp.zeros((100, 100), dtype=jnp.float32)
        psf = jnp.array([[1.0]])
        rel_y, rel_x = jnp.array([[0.0]]), jnp.array([[0.0]])

        # Deposit at same location twice
        output = deposit_psf(output, 50.5, 50.5, psf, rel_x, rel_y)
        output = deposit_psf(output, 50.5, 50.5, 2 * psf, rel_x, rel_y)

        assert output[50, 50] == 3.0


class TestStarDispersion:
    """Test the star dispersion functions.

    Note: These tests use a mock PSF payload to avoid slow STPSF generation.
    For full integration tests, use the demo notebook.
    """

    @pytest.fixture
    def mock_psf_payload(self):
        """Create a mock PSF payload for testing.

        Note: wavelengths are in microns (not meters).
        """
        # Simple 4×4 spatial, 10 wavelength grid with 20×20 PSFs
        # Each PSF is a normalized Gaussian-like pattern
        n_y, n_x, n_wl = 4, 4, 10
        psf_size = 20
        oversample = 4

        # Create PSF grid with simple pattern
        psf_grid = jnp.zeros((n_y, n_x, n_wl, psf_size, psf_size), dtype=jnp.float32)

        # Fill with normalized delta-like PSFs (sum to 1)
        center = psf_size // 2
        for iy in range(n_y):
            for ix in range(n_x):
                for iw in range(n_wl):
                    # Put all flux at center
                    psf_grid = psf_grid.at[iy, ix, iw, center, center].set(1.0)

        return {
            'psf_grid': psf_grid,
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl),  # microns
            'wl_grid': jnp.linspace(1.0, 1.8, n_wl),      # microns
            'spatial_x': jnp.linspace(1, 4088, n_x),
            'spatial_y': jnp.linspace(1, 4088, n_y),
            'oversample': oversample,
            'detector': 'WFI05',
            'order': '1',
        }

    def test_disperse_star_basic(self, payload, mock_psf_payload):
        """Basic star dispersion should produce output along trace."""
        wavelengths = jnp.array([1.2, 1.4, 1.6])  # 3 wavelengths in microns
        star_flux = jnp.ones(3)  # Unit flux at each wavelength
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        result = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=star_flux,
            output=output,
        )

        # Should have non-zero flux
        assert result.sum() > 0

        # Total flux should equal sum of star_flux (each PSF sums to 1.0)
        # Using tightened tolerance (0.1% instead of 1%)
        expected_flux = float(star_flux.sum())
        assert jnp.isclose(result.sum(), expected_flux, rtol=0.001)

    def test_disperse_star_flux_scaling(self, payload, mock_psf_payload):
        """Flux should scale linearly with star_flux."""
        wavelengths = jnp.array([1.4])  # Single wavelength in microns
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # Disperse with flux = 1
        result1 = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=jnp.array([1.0]),
            output=output,
        )

        # Disperse with flux = 3
        result3 = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=jnp.array([3.0]),
            output=output,
        )

        # Total flux should scale by 3
        assert jnp.isclose(result3.sum(), 3 * result1.sum(), rtol=RTOL)

    def test_disperse_star_position_matches_trace(
        self, payload, mock_psf_payload
    ):
        """Star should land at trace_beam predicted positions."""
        # Single wavelength dispersion (in microns)
        wl = jnp.array([1.5])  # 1.5 microns
        star_flux = jnp.array([1.0])
        xsca_star, ysca_star = 2000.0, 2000.0
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        result = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=xsca_star,
            ysca_star=ysca_star,
            wavelengths=wl,
            star_flux=star_flux,
            output=output,
        )

        # Compute expected position using trace_beam (functions expect arrays)
        xfpa, yfpa = omj.sca_to_fpa(
            payload, jnp.array([xsca_star]), jnp.array([ysca_star])
        )
        # trace_beam expects wavelengths in microns (which we already have)
        xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wl)
        xsca_expected, ysca_expected = omj.mpa_to_sca(payload, xmpa, ympa)

        # Find peak in result
        peak_idx = jnp.unravel_index(jnp.argmax(result), result.shape)
        peak_y, peak_x = peak_idx

        # Peak should be within 1 pixel of expected (due to discrete placement)
        assert abs(float(peak_x) - float(xsca_expected[0])) < 1.5
        assert abs(float(peak_y) - float(ysca_expected[0])) < 1.5

    def test_multiple_wavelengths_spread(
        self, payload, mock_psf_payload
    ):
        """Multiple wavelengths should spread along the trace."""
        # Widely spaced wavelengths (in microns)
        wavelengths = jnp.array([1.1, 1.5, 1.8])
        star_flux = jnp.ones(3)
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        result = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=star_flux,
            output=output,
        )

        # Find non-zero pixels
        nonzero_y, nonzero_x = jnp.where(result > 0)

        # Should have multiple non-zero pixels spread out
        # (dispersion causes y-spread for Roman grism)
        y_spread = nonzero_y.max() - nonzero_y.min()
        assert y_spread > 10  # Should be spread by more than 10 pixels


class TestMakeStarDisperser:
    """Test the star disperser factory function."""

    @pytest.fixture
    def mock_psf_payload(self):
        """Create a mock PSF payload for testing.

        Note: wavelengths are in microns (not meters).
        """
        n_y, n_x, n_wl = 2, 2, 5
        psf_size = 20
        oversample = 4

        psf_grid = jnp.zeros((n_y, n_x, n_wl, psf_size, psf_size), dtype=jnp.float32)
        center = psf_size // 2
        for iy in range(n_y):
            for ix in range(n_x):
                for iw in range(n_wl):
                    psf_grid = psf_grid.at[iy, ix, iw, center, center].set(1.0)

        return {
            'psf_grid': psf_grid,
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl),  # microns
            'wl_grid': jnp.linspace(1.0, 1.8, n_wl),      # microns
            'spatial_x': jnp.linspace(1, 4088, n_x),
            'spatial_y': jnp.linspace(1, 4088, n_y),
            'oversample': oversample,
            'detector': 'WFI05',
            'order': '1',
        }

    def test_validates_even_oversample(self, payload, mock_psf_payload):
        """Should reject odd oversampling."""
        mock_psf_payload['oversample'] = 3  # Odd

        with pytest.raises(ValueError, match="even oversampling"):
            make_star_disperser(mock_psf_payload, payload)

    def test_accepts_even_oversample(self, payload, mock_psf_payload):
        """Should accept even oversampling."""
        mock_psf_payload['oversample'] = 4

        disperser = make_star_disperser(mock_psf_payload, payload)
        assert callable(disperser)

    def test_jit_compilation(self, payload, mock_psf_payload):
        """Factory should return JIT-compiled function."""
        disperser = make_star_disperser(mock_psf_payload, payload)

        wavelengths = jnp.array([1.4])  # microns
        star_flux = jnp.array([1.0])
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # First call compiles
        result1 = disperser(2000.0, 2000.0, wavelengths, star_flux, output)

        # Second call should use cached compilation
        result2 = disperser(2000.0, 2000.0, wavelengths, star_flux, output)

        # Results should be identical
        np.testing.assert_allclose(result1, result2, rtol=RTOL, atol=ATOL)

    def test_repeated_calls_accumulate(self, payload, mock_psf_payload):
        """Multiple calls should accumulate onto output."""
        disperser = make_star_disperser(mock_psf_payload, payload)

        wavelengths = jnp.array([1.4])  # microns
        star_flux = jnp.array([1.0])
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # Disperse first star
        output = disperser(2000.0, 2000.0, wavelengths, star_flux, output)
        flux1 = output.sum()

        # Disperse second star at different position
        output = disperser(2500.0, 2500.0, wavelengths, star_flux, output)
        flux2 = output.sum()

        # Total should be 2× single star flux
        assert jnp.isclose(flux2, 2 * flux1, rtol=RTOL)

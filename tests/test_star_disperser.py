"""
Tests for star_disperser module.

These tests verify:
1. PSF pixel grid creation
2. PSF deposition
3. Star dispersion with wavelength-dependent PSFs
4. JIT-compiled star disperser factory

Note: All wavelengths are in **microns** (consistent with optical model).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import star_disperser
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


class TestChunkedDispersion:
    """Test the chunked (memory-efficient) dispersion implementation."""

    @pytest.fixture
    def mock_psf_payload(self):
        """Create a mock PSF payload for testing.

        Note: wavelengths are in microns (not meters).
        """
        n_y, n_x, n_wl = 4, 4, 10
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

    def test_chunk_size_invariance(self, payload, mock_psf_payload):
        """Result should be same regardless of chunk_size."""
        wavelengths = jnp.linspace(1.1, 1.7, 25)  # 25 wavelengths
        star_flux = jnp.ones(25)
        xsca, ysca = 2000.0, 2000.0

        # Run with different chunk sizes
        results = []
        for chunk_size in [5, 10, 25, 100]:
            output = jnp.zeros((4088, 4088), dtype=jnp.float32)
            result = disperse_star_psf(
                mock_psf_payload,
                payload,
                xsca_star=xsca,
                ysca_star=ysca,
                wavelengths=wavelengths,
                star_flux=star_flux,
                output=output,
                chunk_size=chunk_size,
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i],
                rtol=1e-5, atol=1e-6,
                err_msg=f"Chunk size {[5, 10, 25, 100][i]} differs from chunk size 5"
            )

    def test_chunk_size_with_padding(self, payload, mock_psf_payload):
        """Test chunking with wavelength counts that don't divide evenly."""
        # 17 wavelengths doesn't divide evenly by common chunk sizes
        wavelengths = jnp.linspace(1.1, 1.7, 17)
        star_flux = jnp.ones(17)

        # Run with chunk_size=5 (17 / 5 = 3.4, needs padding)
        output_5 = jnp.zeros((4088, 4088), dtype=jnp.float32)
        result_5 = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=star_flux,
            output=output_5,
            chunk_size=5,
        )

        # Run with chunk_size=20 (larger than n_wl)
        output_20 = jnp.zeros((4088, 4088), dtype=jnp.float32)
        result_20 = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=star_flux,
            output=output_20,
            chunk_size=20,
        )

        # Results should be identical
        np.testing.assert_allclose(result_5, result_20, rtol=1e-5, atol=1e-6)

    def test_many_wavelengths(self, payload, mock_psf_payload):
        """Test dispersion with many wavelengths (memory stress test)."""
        # 500 wavelengths - should work fine with chunking
        wavelengths = jnp.linspace(1.0, 1.8, 500)
        star_flux = jnp.ones(500)
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        result = disperse_star_psf(
            mock_psf_payload,
            payload,
            xsca_star=2000.0,
            ysca_star=2000.0,
            wavelengths=wavelengths,
            star_flux=star_flux,
            output=output,
            chunk_size=100,
        )

        # Should have non-zero flux
        assert result.sum() > 0

        # Total flux should equal sum of star_flux (each PSF sums to 1.0)
        expected_flux = float(star_flux.sum())
        assert jnp.isclose(result.sum(), expected_flux, rtol=0.001)

    def test_factory_with_chunk_size(self, payload, mock_psf_payload):
        """Factory function should respect chunk_size parameter."""
        wavelengths = jnp.linspace(1.1, 1.7, 15)
        star_flux = jnp.ones(15)

        # Create dispersers with different chunk sizes
        disperser_10 = make_star_disperser(mock_psf_payload, payload, chunk_size=10)
        disperser_5 = make_star_disperser(mock_psf_payload, payload, chunk_size=5)

        output_10 = jnp.zeros((4088, 4088), dtype=jnp.float32)
        output_5 = jnp.zeros((4088, 4088), dtype=jnp.float32)

        result_10 = disperser_10(2000.0, 2000.0, wavelengths, star_flux, output_10)
        result_5 = disperser_5(2000.0, 2000.0, wavelengths, star_flux, output_5)

        # Results should be identical
        np.testing.assert_allclose(result_10, result_5, rtol=1e-5, atol=1e-6)


class TestInterpWavelengthChunk:
    """Test the vectorized wavelength interpolation function."""

    def test_grid_points_exact(self):
        """Interpolation at grid points should return exact values."""
        from roman_disperser.psf_model import interp_wavelength_chunk

        # Create simple PSF grid
        n_wl = 5
        psf_size = 10
        psfs_grid = jnp.arange(n_wl * psf_size * psf_size, dtype=jnp.float32).reshape(
            n_wl, psf_size, psf_size
        )
        grid_wl = jnp.linspace(1.0, 1.8, n_wl)

        # Interpolate at grid points
        target_wl = grid_wl
        result = interp_wavelength_chunk(psfs_grid, grid_wl, target_wl)

        np.testing.assert_allclose(result, psfs_grid, rtol=1e-5)

    def test_midpoint_interpolation(self):
        """Midpoint should be average of neighbors."""
        from roman_disperser.psf_model import interp_wavelength_chunk

        # Create simple PSF grid with known values
        n_wl = 3
        psf_size = 4
        psfs_grid = jnp.zeros((n_wl, psf_size, psf_size), dtype=jnp.float32)
        psfs_grid = psfs_grid.at[0].set(0.0)
        psfs_grid = psfs_grid.at[1].set(2.0)
        psfs_grid = psfs_grid.at[2].set(4.0)
        grid_wl = jnp.array([1.0, 1.5, 2.0])

        # Interpolate at midpoints
        target_wl = jnp.array([1.25, 1.75])
        result = interp_wavelength_chunk(psfs_grid, grid_wl, target_wl)

        # At 1.25: (1.25 - 1.0) / (1.5 - 1.0) = 0.5, so 0.5 * 0 + 0.5 * 2 = 1.0
        expected_125 = 1.0
        # At 1.75: (1.75 - 1.5) / (2.0 - 1.5) = 0.5, so 0.5 * 2 + 0.5 * 4 = 3.0
        expected_175 = 3.0

        np.testing.assert_allclose(result[0], jnp.full((psf_size, psf_size), expected_125), rtol=1e-5)
        np.testing.assert_allclose(result[1], jnp.full((psf_size, psf_size), expected_175), rtol=1e-5)

    def test_edge_extrapolation(self):
        """Wavelengths outside grid should clamp to edge values."""
        from roman_disperser.psf_model import interp_wavelength_chunk

        # Create simple PSF grid
        n_wl = 3
        psf_size = 4
        psfs_grid = jnp.zeros((n_wl, psf_size, psf_size), dtype=jnp.float32)
        psfs_grid = psfs_grid.at[0].set(1.0)
        psfs_grid = psfs_grid.at[1].set(2.0)
        psfs_grid = psfs_grid.at[2].set(3.0)
        grid_wl = jnp.array([1.0, 1.5, 2.0])

        # Interpolate outside grid bounds
        target_wl = jnp.array([0.5, 2.5])  # Below and above grid
        result = interp_wavelength_chunk(psfs_grid, grid_wl, target_wl)

        # Should clamp to edge values (not extrapolate)
        # For 0.5: below grid, should use first PSF
        # For 2.5: above grid, should use last PSF
        np.testing.assert_allclose(result[0], jnp.full((psf_size, psf_size), 1.0), rtol=1e-5)
        np.testing.assert_allclose(result[1], jnp.full((psf_size, psf_size), 3.0), rtol=1e-5)

    def test_matches_interpolate_psf_wavelength(self):
        """Should match the existing interpolate_psf_wavelength function."""
        from roman_disperser.psf_model import interp_wavelength_chunk, interpolate_psf_wavelength

        # Create a realistic-ish PSF grid
        n_wl = 10
        psf_size = 20
        psfs_grid = jnp.arange(n_wl * psf_size * psf_size, dtype=jnp.float32).reshape(
            n_wl, psf_size, psf_size
        )
        grid_wl = jnp.linspace(1.0, 1.8, n_wl)

        # Test on various target wavelengths
        target_wl = jnp.array([1.1, 1.35, 1.5, 1.65, 1.75])

        result_new = interp_wavelength_chunk(psfs_grid, grid_wl, target_wl)
        result_old = interpolate_psf_wavelength(psfs_grid, grid_wl, target_wl)

        np.testing.assert_allclose(result_new, result_old, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: deposit_stack_native — stamp-size sweep against a float64 reference
# ---------------------------------------------------------------------------

def _reference_deposit_f64(stack, grid_wl, wavelengths, flux,
                           x_disp, y_disp, shape, oversample):
    """The baseline per-subpixel deposit, in float64 numpy.

    Mirrors the pre-native16 algorithm exactly: per wavelength, interpolate
    the oversampled stamp between bracketing grid wavelengths, scale by
    flux, and floor-deposit every subpixel j at
    floor(center - 0.5 + (j - (S-1)/2)/os), dropping out-of-bounds
    indices. deposit_stack_native performs the identical additions
    regrouped, so it must agree for every stamp size and phase.
    """
    out = np.zeros(shape, dtype=np.float64)
    stack = np.asarray(stack, dtype=np.float64)
    grid_wl = np.asarray(grid_wl, dtype=np.float64)
    n_grid = stack.shape[0]
    s_y, s_x = stack.shape[-2:]
    os_ = oversample

    for wl, fx, xc, yc in zip(np.asarray(wavelengths, dtype=np.float64),
                              np.asarray(flux, dtype=np.float64),
                              np.asarray(x_disp, dtype=np.float64),
                              np.asarray(y_disp, dtype=np.float64)):
        i0 = int(np.clip(np.searchsorted(grid_wl, wl) - 1, 0, n_grid - 2))
        t = np.clip((wl - grid_wl[i0]) / (grid_wl[i0 + 1] - grid_wl[i0]),
                    0.0, 1.0)
        stamp = (stack[i0] + t * (stack[i0 + 1] - stack[i0])) * fx
        idx_y = np.floor(yc - 0.5
                         + (np.arange(s_y) - (s_y - 1) / 2.0) / os_
                         ).astype(np.int64)
        idx_x = np.floor(xc - 0.5
                         + (np.arange(s_x) - (s_x - 1) / 2.0) / os_
                         ).astype(np.int64)
        ok_y = (idx_y >= 0) & (idx_y < shape[0])
        ok_x = (idx_x >= 0) & (idx_x < shape[1])
        np.add.at(out,
                  (idx_y[ok_y][:, None], idx_x[ok_x][None, :]),
                  stamp[np.ix_(ok_y, ok_x)])
    return out


class TestDepositStackNativeSizes:
    """Stamp-size sweep for the 16-phase native deposit.

    The binning arithmetic (native size (S-2)//os + 2, per-phase padding,
    the extra boundary pixel) depends on S mod os, but the production paths
    only ever exercise one residue (stamps 120, 184, conv 303+pad). This
    sweep checks every residue, non-square stamps, and a second oversample
    factor against the float64 per-subpixel reference above.

    Positions are exact multiples of 1/8 (all representable in f32), so
    the phase and floor computations agree bit-for-bit between the f32
    code and the f64 reference even AT the quarter-pixel boundaries — all
    16 phases are hit deterministically, including the degenerate
    frac == p/4 cases. One generic non-representable position is included
    per config, placed away from boundaries.
    """

    SHAPE = (64, 64)

    def _run(self, s_y, s_x, oversample, chunk_size=4):
        rng = np.random.default_rng(1000 * s_y + s_x)
        n_grid = 3
        stack = jnp.array(rng.uniform(0.1, 1.0, (n_grid, s_y, s_x)),
                          dtype=jnp.float32)
        grid_wl = jnp.array([1.0, 1.5, 2.0], dtype=jnp.float32)

        # 16 phase combinations via exact eighths, + boundary and generic
        # extras; n_wl=18 also exercises the pad-to-chunk-multiple path.
        fracs = jnp.array([0.0, 0.25, 0.5, 0.75], dtype=jnp.float32)
        base = 30.0
        x_disp = jnp.concatenate([
            base + jnp.repeat(fracs, 4),
            jnp.array([base + 0.125, 27.13], dtype=jnp.float32)])
        y_disp = jnp.concatenate([
            base + jnp.tile(fracs, 4),
            jnp.array([base + 0.625, 33.87], dtype=jnp.float32)])
        n_wl = len(x_disp)
        wavelengths = jnp.linspace(0.9, 2.1, n_wl).astype(jnp.float32)
        flux = jnp.array(rng.uniform(0.5, 2.0, n_wl), dtype=jnp.float32)

        out = jnp.zeros(self.SHAPE, dtype=jnp.float32)
        got = star_disperser.deposit_stack_native(
            stack, grid_wl, wavelengths, flux, x_disp, y_disp, out,
            oversample=oversample, chunk_size=chunk_size)
        want = _reference_deposit_f64(
            stack, grid_wl, wavelengths, flux, x_disp, y_disp,
            self.SHAPE, oversample)

        # Identical additions regrouped: agreement to f32 rounding. A
        # grouping/size bug shifts whole stamp rows -> order-unity errors.
        atol = 1e-5 * float(np.abs(want).max())
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want,
                                   rtol=1e-5, atol=atol)
        # Flux: everything is on-detector in this configuration.
        np.testing.assert_allclose(
            float(np.asarray(got, np.float64).sum()), want.sum(), rtol=1e-6)

    @pytest.mark.parametrize("s_y,s_x", [
        (12, 12),   # S % 4 == 0 (the production residue)
        (9, 9),     # S % 4 == 1
        (10, 10),   # S % 4 == 2
        (11, 11),   # S % 4 == 3
        (10, 13),   # non-square, mixed residues
        (11, 8),    # non-square, mixed residues
    ])
    def test_sizes_match_subpixel_reference(self, s_y, s_x):
        self._run(s_y, s_x, oversample=4)

    def test_oversample_2(self):
        self._run(7, 10, oversample=2)

    def test_edge_drop_matches_reference(self):
        """Stamps straddling the detector edge: mode='drop' must drop
        exactly the native pixels whose subpixels the baseline dropped."""
        rng = np.random.default_rng(7)
        stack = jnp.array(rng.uniform(0.1, 1.0, (2, 11, 10)),
                          dtype=jnp.float32)
        grid_wl = jnp.array([1.0, 2.0], dtype=jnp.float32)
        # Centers near (0,0) and beyond the far corner: partial overlap.
        x_disp = jnp.array([1.25, 63.5, -0.375], dtype=jnp.float32)
        y_disp = jnp.array([0.75, 62.875, 64.25], dtype=jnp.float32)
        wavelengths = jnp.array([1.1, 1.5, 1.9], dtype=jnp.float32)
        flux = jnp.ones(3, dtype=jnp.float32)

        out = jnp.zeros(self.SHAPE, dtype=jnp.float32)
        got = star_disperser.deposit_stack_native(
            stack, grid_wl, wavelengths, flux, x_disp, y_disp, out,
            oversample=2, chunk_size=2)
        want = _reference_deposit_f64(
            stack, grid_wl, wavelengths, flux, x_disp, y_disp,
            self.SHAPE, 2)
        atol = 1e-5 * float(np.abs(want).max())
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want,
                                   rtol=1e-5, atol=atol)

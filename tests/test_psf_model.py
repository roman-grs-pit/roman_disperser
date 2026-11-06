"""
Tests for PSF model and coordinate utilities.

Tests cover:
- Coordinate conversion (SCA <-> STPSF)
- PSF payload generation
- Trilinear interpolation
- JIT compilation
- Enclosed energy validation

Note: All wavelength parameters are in **microns** (not meters).
"""

import csv
import os
import sys
import warnings
from pathlib import Path

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from roman_disperser import psf_utils
from roman_disperser import psf_model


# ============================================================================
# COORDINATE CONVERSION TESTS
# ============================================================================


class TestCoordinateConversion:
    """Test SCA <-> STPSF coordinate conversion."""

    def test_roundtrip_center(self):
        """Round-trip conversion at detector center should be exact."""
        xsca, ysca = 2044.0, 2044.0  # Approximate detector center

        # Convert to STPSF and back
        x_stpsf, y_stpsf = psf_utils.sca_to_stpsf_position(xsca, ysca)
        xsca_back, ysca_back = psf_utils.stpsf_to_sca_position(x_stpsf, y_stpsf)

        # Should be exact (simple arithmetic, no rounding)
        assert abs(xsca - xsca_back) < 1e-10
        assert abs(ysca - ysca_back) < 1e-10

    def test_roundtrip_corners(self):
        """Round-trip conversion at grid corners should be exact."""
        corners = [
            (500.0, 500.0),
            (500.0, 3500.0),
            (3500.0, 500.0),
            (3500.0, 3500.0),
        ]

        for xsca, ysca in corners:
            x_stpsf, y_stpsf = psf_utils.sca_to_stpsf_position(xsca, ysca)
            xsca_back, ysca_back = psf_utils.stpsf_to_sca_position(x_stpsf, y_stpsf)

            assert abs(xsca - xsca_back) < 1e-10, f"Failed at ({xsca}, {ysca})"
            assert abs(ysca - ysca_back) < 1e-10, f"Failed at ({xsca}, {ysca})"

    def test_jit_compilation(self):
        """Coordinate functions should be JIT-compilable."""
        # Wrap in JIT
        @jax.jit
        def convert_forward(xsca, ysca):
            return psf_utils.sca_to_stpsf_position(xsca, ysca)

        @jax.jit
        def convert_back(x_stpsf, y_stpsf):
            return psf_utils.stpsf_to_sca_position(x_stpsf, y_stpsf)

        # Should compile and run without errors
        xsca, ysca = 2000.0, 2000.0
        x_stpsf, y_stpsf = convert_forward(xsca, ysca)
        xsca_back, ysca_back = convert_back(x_stpsf, y_stpsf)

        assert abs(xsca - xsca_back) < 1e-10

    def test_vectorization(self):
        """Coordinate functions should work with array inputs."""
        # Create array of positions
        xsca = jnp.array([1000.0, 2000.0, 3000.0])
        ysca = jnp.array([1000.0, 2000.0, 3000.0])

        # Convert to STPSF
        x_stpsf, y_stpsf = psf_utils.sca_to_stpsf_position(xsca, ysca)

        # Should return arrays of same shape
        assert x_stpsf.shape == xsca.shape
        assert y_stpsf.shape == ysca.shape

        # Round-trip should be exact for all elements
        xsca_back, ysca_back = psf_utils.stpsf_to_sca_position(x_stpsf, y_stpsf)
        assert jnp.allclose(xsca, xsca_back, atol=1e-10)
        assert jnp.allclose(ysca, ysca_back, atol=1e-10)

    def test_expected_offset(self):
        """Test that conversion matches expected 4-pixel offset assumption."""
        # At SCA pixel 1, we expect STPSF pixel 4 (if assumption is correct)
        xsca, ysca = 1.0, 1.0
        x_stpsf, y_stpsf = psf_utils.sca_to_stpsf_position(xsca, ysca)

        # Based on hardcoded offset of 4
        assert abs(x_stpsf - 4.0) < 1e-10
        assert abs(y_stpsf - 4.0) < 1e-10


# ============================================================================
# PSF PAYLOAD GENERATION TESTS (SLOW - marked for optional execution)
# ============================================================================


class TestPSFPayload:
    """Test PSF payload generation.

    Note: These tests are SLOW (minutes to tens of minutes) because they
    call STPSF to generate PSF grids. They are marked with @pytest.mark.slow
    and can be skipped with: pytest -m "not slow"
    """

    @pytest.mark.slow
    def test_minimal_payload_generation(self):
        """Generate minimal PSF payload for quick validation."""
        # Minimal configuration: 2×2 spatial, 3 wavelengths (in microns)
        wavelengths = np.linspace(0.9, 2.0, 3)
        spatial_grid = {
            'x': np.array([1000.0, 3000.0]),
            'y': np.array([1000.0, 3000.0])
        }

        payload = psf_model.make_psf_payload(
            detector='WFI05',
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=5.0,
            oversample=4,
            verbose=True
        )

        # Check payload structure
        assert 'psf_grid' in payload
        assert 'wavelengths' in payload
        assert 'spatial_x' in payload
        assert 'spatial_y' in payload
        assert 'timing' in payload

        # Check PSF grid shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
        psf_grid = payload['psf_grid']
        assert psf_grid.shape[0] == 2  # 2 y positions
        assert psf_grid.shape[1] == 2  # 2 x positions
        assert psf_grid.shape[2] == 3  # 3 wavelengths
        assert psf_grid.shape[3] > 0  # PSF has size
        assert psf_grid.shape[4] > 0

        # Check PSFs are normalized (approximately)
        for iy in range(2):
            for ix in range(2):
                for iwl in range(3):
                    psf = psf_grid[iy, ix, iwl]
                    total_flux = float(psf.sum())
                    # PSFs lose some flux outside FOV (extended wings)
                    # Expect 95-100% flux within 5" FOV
                    assert 0.95 < total_flux < 1.001, \
                        f"PSF flux {total_flux} outside [0.95, 1.001]"

    @pytest.mark.slow
    def test_payload_timing_reported(self):
        """Check that timing information is captured."""
        wavelengths = np.linspace(0.9, 2.0, 2)  # microns
        spatial_grid = {
            'x': np.array([2000.0]),
            'y': np.array([2000.0])
        }

        payload = psf_model.make_psf_payload(
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            verbose=False
        )

        # Check timing dict
        timing = payload['timing']
        assert 'total_time' in timing
        assert 'per_psf_time' in timing
        assert 'n_psfs' in timing
        assert timing['n_psfs'] == 2  # 1 spatial × 2 wavelengths
        assert timing['total_time'] > 0

    @pytest.mark.slow
    def test_zeroth_order_psf_generation(self):
        """Test that zeroth order PSFs can be generated."""
        wavelengths = np.linspace(0.9, 2.0, 2)  # microns
        spatial_grid = {
            'x': np.array([2000.0]),
            'y': np.array([2000.0])
        }

        # Generate zeroth order PSFs
        payload = psf_model.make_psf_payload(
            detector='WFI05',
            order='0',  # Zeroth order
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            verbose=False
        )

        # Check order is stored
        assert payload['order'] == '0'

        # Check PSF grid exists and has correct shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
        assert payload['psf_grid'].shape[0] == 1  # 1 y position
        assert payload['psf_grid'].shape[1] == 1  # 1 x position
        assert payload['psf_grid'].shape[2] == 2  # 2 wavelengths

    def test_invalid_order_raises_error(self):
        """Test that invalid order raises ValueError."""
        with pytest.raises(ValueError, match="No default STPSF filter"):
            psf_model.make_psf_payload(
                order='99',  # Invalid
                wavelengths=np.array([1.5]),  # microns
                spatial_grid={'x': np.array([2000.0]), 'y': np.array([2000.0])},
                verbose=False
            )

    def test_non_increasing_wavelengths_raises_error(self):
        """Test that non-strictly-increasing wavelengths raise ValueError."""
        # Duplicate wavelengths
        with pytest.raises(ValueError, match="strictly increasing"):
            psf_model.make_psf_payload(
                order='1',
                wavelengths=np.array([1.0, 1.5, 1.5, 2.0]),  # Duplicate (microns)
                spatial_grid={'x': np.array([2000.0]), 'y': np.array([2000.0])},
                verbose=False
            )

        # Decreasing wavelengths
        with pytest.raises(ValueError, match="strictly increasing"):
            psf_model.make_psf_payload(
                order='1',
                wavelengths=np.array([2.0, 1.5, 1.0]),  # Decreasing (microns)
                spatial_grid={'x': np.array([2000.0]), 'y': np.array([2000.0])},
                verbose=False
            )

    def test_non_increasing_spatial_grid_raises_error(self):
        """Test that non-strictly-increasing spatial grids raise ValueError."""
        # Duplicate x values
        with pytest.raises(ValueError, match="Spatial x grid"):
            psf_model.make_psf_payload(
                order='1',
                wavelengths=np.array([1.5]),  # microns
                spatial_grid={'x': np.array([1000.0, 2000.0, 2000.0]), 'y': np.array([2000.0])},
                verbose=False
            )

        # Duplicate y values
        with pytest.raises(ValueError, match="Spatial y grid"):
            psf_model.make_psf_payload(
                order='1',
                wavelengths=np.array([1.5]),  # microns
                spatial_grid={'x': np.array([2000.0]), 'y': np.array([1000.0, 2000.0, 2000.0])},
                verbose=False
            )

    @pytest.mark.slow
    def test_corner_detector_generation(self):
        """Test PSF generation works for corner detector (not just central WFI05)."""
        # Use WFI01 (corner detector) to test edge handling
        wavelengths = np.linspace(0.9, 2.0, 2)  # microns
        spatial_grid = {
            'x': np.array([2000.0]),
            'y': np.array([2000.0])
        }

        payload = psf_model.make_psf_payload(
            detector='WFI01',  # Corner detector
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            verbose=False
        )

        # Verify payload is valid
        assert payload['detector'] == 'WFI01'
        # Shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
        assert payload['psf_grid'].shape[2] == 2  # 2 wavelengths

        # Check PSF normalization
        for iwl in range(2):
            psf = payload['psf_grid'][0, 0, iwl]
            total_flux = float(psf.sum())
            assert 0.95 < total_flux < 1.001, \
                f"PSF flux {total_flux} outside [0.95, 1.001]"


# ============================================================================
# PSF INTERPOLATION TESTS
# ============================================================================


class TestPSFInterpolation:
    """Test trilinear PSF interpolation."""

    @pytest.fixture
    def simple_payload(self):
        """Create a simple test payload with analytical PSFs."""
        # Create a minimal payload with known PSF values for testing
        # Wavelengths in microns
        wavelengths = np.array([1.0, 1.5, 1.9])
        spatial_x = np.array([1000.0, 2000.0, 3000.0])
        spatial_y = np.array([1000.0, 2000.0, 3000.0])

        # Create simple PSF grid (just constant values for testing)
        # Shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
        psf_size = 10
        psf_grid = np.zeros((3, 3, 3, psf_size, psf_size))

        # Fill with simple patterns (different for each position/wavelength)
        # Value = iy + ix + iwl (same formula, but indexing matches new array order)
        for iy in range(3):
            for ix in range(3):
                for iwl in range(3):
                    value = float(iy + ix + iwl)
                    psf_grid[iy, ix, iwl, :, :] = value

        payload = {
            'detector': 'WFI05',
            'wavelengths': jnp.array(wavelengths),
            'wl_grid': jnp.array(wavelengths),
            'spatial_x': jnp.array(spatial_x),
            'spatial_y': jnp.array(spatial_y),
            'psf_grid': jnp.array(psf_grid, dtype=jnp.float32),
            'psf_fov_pixels': psf_size,
            'pixel_scale': 0.11,
            'oversample': 4,
        }

        return payload

    def test_interpolation_at_grid_points(self, simple_payload):
        """Interpolation at exact grid points should match grid values."""
        # Test at grid point: (2000.0, 2000.0, 1.5)
        # This is index [1, 1, 1] in the grid
        xsca, ysca, wavelength = 2000.0, 2000.0, 1.5  # microns

        psf = psf_model.interpolate_psf(simple_payload, xsca, ysca, wavelength)

        # Should match grid value at [1, 1, 1]
        # value = iwl + iy + ix = 1 + 1 + 1 = 3.0
        # Should be at machine precision for float32 (~1e-7 relative error)
        expected_value = 3.0
        assert jnp.allclose(psf, expected_value, rtol=1e-6, atol=1e-6)

    def test_interpolation_shape(self, simple_payload):
        """Interpolated PSF should have correct shape."""
        xsca, ysca, wavelength = 2000.0, 2000.0, 1.5  # microns

        psf = psf_model.interpolate_psf(simple_payload, xsca, ysca, wavelength)

        # Should match PSF size from payload
        assert psf.shape == (10, 10)

    def test_jit_compilation_interpolation(self, simple_payload):
        """PSF interpolation should be JIT-compilable with closure pattern."""
        payload = simple_payload

        @jax.jit
        def interp_jit(xsca, ysca, wl):
            return psf_model.interpolate_psf(payload, xsca, ysca, wl)

        # First call (compile + run)
        psf1 = interp_jit(2000.0, 2000.0, 1.5)  # microns

        # Second call (cached)
        psf2 = interp_jit(2100.0, 2100.0, 1.6)  # microns

        # Both should have same shape
        assert psf1.shape == psf2.shape

    def test_edge_extrapolation(self, simple_payload):
        """Interpolation should use edge values for out-of-bounds positions (not linear extrapolation)."""
        # simple_payload has grid: x=[1000, 2000, 3000], y=[1000, 2000, 3000], wl=[1.0, 1.5, 1.9] microns
        # PSF values: value = iwl + iy + ix

        # Test position outside grid (x < x_min, but y and wl on grid)
        xsca, ysca, wavelength = 500.0, 2000.0, 1.5  # x off-grid low, wl in microns

        # Should use edge PSF: x_idx=[0,1], y_idx=[1,1], wl_idx=[1,1]
        # With x_frac clamped to 0, should get psf[wl=1, y=1, x=0] = 1+1+0 = 2.0
        psf_low = psf_model.interpolate_psf(simple_payload, xsca, ysca, wavelength)
        assert psf_low.shape == (10, 10)
        assert jnp.allclose(psf_low, 2.0, rtol=1e-6)

        # Test high edge (x > x_max)
        xsca_high = 4000.0  # x off-grid high
        # Should use edge PSF: x_idx=[1,2], y_idx=[1,1], wl_idx=[1,1]
        # With x_frac clamped to 1, should get psf[wl=1, y=1, x=2] = 1+1+2 = 4.0
        psf_high = psf_model.interpolate_psf(simple_payload, xsca_high, ysca, wavelength)
        assert psf_high.shape == (10, 10)
        assert jnp.allclose(psf_high, 4.0, rtol=1e-6)

        # Test wavelength off-grid low
        wavelength_low = 0.5  # Below wl_grid[0] = 1.0 microns
        # Should use edge PSF: wl_idx=[0,1], x_idx=[1,1], y_idx=[1,1]
        # With wl_frac clamped to 0, should get psf[wl=0, y=1, x=1] = 0+1+1 = 2.0
        psf_wl_low = psf_model.interpolate_psf(simple_payload, 2000.0, 2000.0, wavelength_low)
        assert jnp.allclose(psf_wl_low, 2.0, rtol=1e-6)

    def test_vectorized_interpolation(self, simple_payload):
        """Interpolation should work with array inputs via vmap."""
        # Create arrays of positions (wavelengths in microns)
        xsca = jnp.array([1000.0, 2000.0, 3000.0])
        ysca = jnp.array([1000.0, 2000.0, 3000.0])
        wavelength = jnp.array([1.0, 1.5, 1.9])  # microns

        # Vectorize interpolation
        interp_vmap = jax.vmap(
            lambda x, y, wl: psf_model.interpolate_psf(simple_payload, x, y, wl)
        )

        psfs = interp_vmap(xsca, ysca, wavelength)

        # Should have shape [3, 10, 10]
        assert psfs.shape == (3, 10, 10)

    def test_interpolation_flux_conservation(self, simple_payload):
        """Interpolated PSFs should produce values between grid corners."""
        # Interpolate at midpoint between grid positions
        # Grid: x=[1000, 2000, 3000], y=[1000, 2000, 3000], wl=[1.0, 1.5, 1.9] microns
        xsca_mid = 1500.0  # Between 1000 and 2000
        ysca_mid = 1500.0
        wavelength_mid = 1.25  # Between 1.0 and 1.5 microns

        psf = psf_model.interpolate_psf(simple_payload, xsca_mid, ysca_mid, wavelength_mid)

        # For our simple test payload with constant values per PSF,
        # the interpolated value should be the weighted average of corners
        # Corners involved: [wl=0,y=0,x=0], [wl=0,y=0,x=1], [wl=0,y=1,x=0], [wl=0,y=1,x=1],
        #                   [wl=1,y=0,x=0], [wl=1,y=0,x=1], [wl=1,y=1,x=0], [wl=1,y=1,x=1]
        # Values: 0, 1, 1, 2, 1, 2, 2, 3
        # With all fractions at 0.5: average = (0+1+1+2+1+2+2+3)/8 = 12/8 = 1.5
        assert psf.shape == (10, 10)
        assert jnp.allclose(psf, 1.5, rtol=1e-6)


# ============================================================================
# SPATIAL INTERPOLATION TESTS (interpolate_psf_spatial)
# ============================================================================


class TestPSFSpatialInterpolation:
    """Test bilinear spatial interpolation (all wavelengths at once)."""

    @pytest.fixture
    def simple_payload(self):
        """Create a simple test payload with analytical PSFs."""
        # Same as TestPSFInterpolation fixture
        # Wavelengths in microns
        wavelengths = np.array([1.0, 1.5, 1.9])
        spatial_x = np.array([1000.0, 2000.0, 3000.0])
        spatial_y = np.array([1000.0, 2000.0, 3000.0])

        # Shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
        psf_size = 10
        psf_grid = np.zeros((3, 3, 3, psf_size, psf_size))

        for iy in range(3):
            for ix in range(3):
                for iwl in range(3):
                    value = float(iy + ix + iwl)
                    psf_grid[iy, ix, iwl, :, :] = value

        payload = {
            'detector': 'WFI05',
            'wavelengths': jnp.array(wavelengths),
            'wl_grid': jnp.array(wavelengths),
            'spatial_x': jnp.array(spatial_x),
            'spatial_y': jnp.array(spatial_y),
            'psf_grid': jnp.array(psf_grid, dtype=jnp.float32),
            'psf_fov_pixels': psf_size,
            'pixel_scale': 0.11,
            'oversample': 4,
        }

        return payload

    def test_spatial_interpolation_shape(self, simple_payload):
        """Spatial interpolation should return all wavelengths."""
        psfs = psf_model.interpolate_psf_spatial(simple_payload, 2000.0, 2000.0)

        # Should return [N_wl, PSF_y, PSF_x]
        assert psfs.shape == (3, 10, 10)

    def test_spatial_interpolation_at_grid_point(self, simple_payload):
        """Interpolation at exact grid point should match grid values."""
        # At grid point (2000.0, 2000.0), which is index [1, 1] in spatial grid
        psfs = psf_model.interpolate_psf_spatial(simple_payload, 2000.0, 2000.0)

        # Values should be: value = iy + ix + iwl = 1 + 1 + iwl
        # For iwl=0,1,2: values should be 2, 3, 4
        assert jnp.allclose(psfs[0], 2.0, rtol=1e-6)
        assert jnp.allclose(psfs[1], 3.0, rtol=1e-6)
        assert jnp.allclose(psfs[2], 4.0, rtol=1e-6)

    def test_spatial_interpolation_midpoint(self, simple_payload):
        """Interpolation at spatial midpoint should average 4 corners."""
        # At midpoint (1500.0, 1500.0), which is between [0,0], [0,1], [1,0], [1,1]
        psfs = psf_model.interpolate_psf_spatial(simple_payload, 1500.0, 1500.0)

        # For each wavelength, average of 4 corners:
        # iwl=0: (0+1+1+2)/4 = 1.0
        # iwl=1: (1+2+2+3)/4 = 2.0
        # iwl=2: (2+3+3+4)/4 = 3.0
        assert jnp.allclose(psfs[0], 1.0, rtol=1e-6)
        assert jnp.allclose(psfs[1], 2.0, rtol=1e-6)
        assert jnp.allclose(psfs[2], 3.0, rtol=1e-6)

    def test_spatial_matches_trilinear_at_grid_wavelengths(self, simple_payload):
        """Spatial interpolation should match trilinear at grid wavelengths."""
        xsca, ysca = 1500.0, 2500.0

        # Get all wavelengths via spatial interpolation
        psfs_spatial = psf_model.interpolate_psf_spatial(simple_payload, xsca, ysca)

        # Compare to trilinear at each grid wavelength
        wavelengths = simple_payload['wavelengths']
        for i, wl in enumerate(wavelengths):
            psf_trilinear = psf_model.interpolate_psf(simple_payload, xsca, ysca, float(wl))
            assert jnp.allclose(psfs_spatial[i], psf_trilinear, rtol=1e-6)

    def test_spatial_jit_compilation(self, simple_payload):
        """Spatial interpolation should be JIT-compilable."""
        payload = simple_payload

        @jax.jit
        def interp_spatial_jit(xsca, ysca):
            return psf_model.interpolate_psf_spatial(payload, xsca, ysca)

        # First call (compile + run)
        psfs1 = interp_spatial_jit(2000.0, 2000.0)

        # Second call (cached)
        psfs2 = interp_spatial_jit(2500.0, 2500.0)

        # Both should have same shape
        assert psfs1.shape == psfs2.shape == (3, 10, 10)

    def test_spatial_edge_extrapolation(self, simple_payload):
        """Spatial interpolation should use edge values for out-of-bounds."""
        # x off-grid low (500 < 1000)
        psfs_low = psf_model.interpolate_psf_spatial(simple_payload, 500.0, 2000.0)

        # Should clamp to x_idx=0, y_idx=1
        # Values: iy=1, ix=0, iwl=0,1,2 → 1, 2, 3
        assert jnp.allclose(psfs_low[0], 1.0, rtol=1e-6)
        assert jnp.allclose(psfs_low[1], 2.0, rtol=1e-6)
        assert jnp.allclose(psfs_low[2], 3.0, rtol=1e-6)


# ============================================================================
# WAVELENGTH INTERPOLATION TESTS (interpolate_psf_wavelength)
# ============================================================================


class TestPSFWavelengthInterpolation:
    """Test wavelength interpolation of PSFs from spatial interpolation."""

    @pytest.fixture
    def simple_psf_stack(self):
        """Create a simple PSF stack at grid wavelengths."""
        # 5 wavelengths (in microns), 10x10 PSFs
        wl_grid = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
        psfs = np.zeros((5, 10, 10), dtype=np.float32)

        # PSF value equals wavelength index (0, 1, 2, 3, 4)
        for i in range(5):
            psfs[i, :, :] = float(i)

        return jnp.array(psfs), jnp.array(wl_grid)

    def test_wavelength_interpolation_shape(self, simple_psf_stack):
        """Output should match number of target wavelengths."""
        psfs_grid, wl_grid = simple_psf_stack
        wavelengths = jnp.array([1.1, 1.3, 1.5])  # microns

        result = psf_model.interpolate_psf_wavelength(psfs_grid, wl_grid, wavelengths)

        assert result.shape == (3, 10, 10)

    def test_wavelength_interpolation_at_grid_points(self, simple_psf_stack):
        """Interpolation at grid wavelengths should return exact PSFs."""
        psfs_grid, wl_grid = simple_psf_stack

        # Interpolate at exact grid wavelengths
        result = psf_model.interpolate_psf_wavelength(psfs_grid, wl_grid, wl_grid)

        # Should match exactly
        for i in range(5):
            assert jnp.allclose(result[i], psfs_grid[i], rtol=1e-6)

    def test_wavelength_interpolation_midpoint(self, simple_psf_stack):
        """Midpoint between two wavelengths should give average PSF."""
        psfs_grid, wl_grid = simple_psf_stack

        # Wavelength exactly between grid points 0 and 1 (1.0 and 1.2 μm)
        midpoint = jnp.array([1.1])  # microns

        result = psf_model.interpolate_psf_wavelength(psfs_grid, wl_grid, midpoint)

        # Should be average of PSF 0 (value 0) and PSF 1 (value 1) = 0.5
        assert jnp.allclose(result[0], 0.5, rtol=1e-6)

    def test_wavelength_interpolation_edge_extrapolation(self, simple_psf_stack):
        """Out-of-range wavelengths should clamp to edge values."""
        psfs_grid, wl_grid = simple_psf_stack

        # Below minimum wavelength
        below = jnp.array([0.8])  # microns
        result_below = psf_model.interpolate_psf_wavelength(psfs_grid, wl_grid, below)
        assert jnp.allclose(result_below[0], 0.0, rtol=1e-6)  # Clamped to first PSF

        # Above maximum wavelength
        above = jnp.array([2.0])  # microns
        result_above = psf_model.interpolate_psf_wavelength(psfs_grid, wl_grid, above)
        assert jnp.allclose(result_above[0], 4.0, rtol=1e-6)  # Clamped to last PSF

    def test_wavelength_interpolation_jit_compilation(self, simple_psf_stack):
        """Function should be JIT-compilable."""
        psfs_grid, wl_grid = simple_psf_stack
        wavelengths = jnp.array([1.1, 1.5])  # microns

        @jax.jit
        def interp_jit(psfs, wl_grid, wavelengths):
            return psf_model.interpolate_psf_wavelength(psfs, wl_grid, wavelengths)

        # First call compiles
        result1 = interp_jit(psfs_grid, wl_grid, wavelengths)

        # Second call uses cached
        result2 = interp_jit(psfs_grid, wl_grid, wavelengths)

        assert jnp.allclose(result1, result2)

    def test_wavelength_interpolation_single_wavelength(self, simple_psf_stack):
        """Single wavelength should work correctly."""
        psfs_grid, wl_grid = simple_psf_stack
        wavelength = jnp.array([1.4])  # Grid point 2 (microns)

        result = psf_model.interpolate_psf_wavelength(psfs_grid, wl_grid, wavelength)

        assert result.shape == (1, 10, 10)
        assert jnp.allclose(result[0], 2.0, rtol=1e-6)


# ============================================================================
# INTEGRATION TESTS (SLOW - require STPSF)
# ============================================================================


class TestPSFIntegration:
    """Integration tests with real STPSF PSFs.

    These tests are SLOW and require STPSF to generate PSFs.
    """

    @pytest.mark.slow
    @pytest.mark.stpsf
    def test_interpolation_accuracy_vs_stpsf(self):
        """Compare interpolated PSF to direct STPSF calculation."""
        pytest.importorskip("stpsf")  # Skip if STPSF not available

        # Generate coarse payload
        # Use wavelengths within STPSF aberration reference range [1.0, 1.9] microns
        wavelengths = np.array([1.0, 1.5, 1.9])  # microns
        spatial_grid = {
            'x': np.array([1500.0, 2500.0]),
            'y': np.array([1500.0, 2500.0])
        }

        payload = psf_model.make_psf_payload(
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=5.0,
            verbose=False
        )

        # Test interpolation at mid-point between grid points
        xsca_test = 2000.0  # Midway between 1500 and 2500
        ysca_test = 2000.0
        wavelength_test = 1.5  # Exact grid wavelength (microns)

        # Get interpolated PSF
        psf_interp = psf_model.interpolate_psf(
            payload, xsca_test, ysca_test, wavelength_test
        )

        # Get direct STPSF PSF
        import stpsf.roman

        wfi = stpsf.roman.WFI()
        wfi.filter = 'GRISM1'
        wfi.detector = 'WFI05'

        x_stpsf, y_stpsf = psf_utils.sca_to_stpsf_position(xsca_test, ysca_test)
        wfi.detector_position = (float(x_stpsf), float(y_stpsf))

        # STPSF expects wavelength in meters
        datacube = wfi.calc_datacube(
            np.array([wavelength_test * 1e-6]), fov_arcsec=5.0, oversample=4
        )
        psf_direct = datacube['OVERDIST'].data[0]  # First wavelength

        # Compare total flux (should be within 1%)
        flux_interp = float(psf_interp.sum())
        flux_direct = float(psf_direct.sum())
        flux_error = abs(flux_interp - flux_direct) / flux_direct

        # Allow 1% error due to spatial interpolation
        assert flux_error < 0.01, \
            f"Flux error {flux_error:.2%} exceeds 1% threshold"

    @pytest.mark.slow
    @pytest.mark.stpsf
    def test_enclosed_energy_all_psfs(self):
        """Validate enclosed energy for all PSFs in grid."""
        pytest.importorskip("stpsf")

        # Generate minimal payload (wavelengths in microns)
        wavelengths = np.linspace(0.9, 2.0, 3)
        spatial_grid = {
            'x': np.array([2000.0]),
            'y': np.array([2000.0])
        }

        payload = psf_model.make_psf_payload(
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=5.0,
            verbose=False
        )

        # Check all PSFs have flux ~1.0 (normalized)
        # Shape: [N_y, N_x, N_wl, PSF_y, PSF_x]
        psf_grid = payload['psf_grid']
        for iy in range(psf_grid.shape[0]):
            for ix in range(psf_grid.shape[1]):
                for iwl in range(psf_grid.shape[2]):
                    psf = psf_grid[iy, ix, iwl]
                    total_flux = float(psf.sum())

                    # PSFs lose some flux outside FOV (extended wings)
                    # Expect 95-100% flux within 5" FOV
                    assert 0.95 < total_flux < 1.001, \
                        f"PSF[{iy},{ix},{iwl}] flux {total_flux} outside [0.95, 1.001]"

    @pytest.mark.slow
    @pytest.mark.stpsf
    def test_interpolation_at_spatial_midpoint(self):
        """Interpolated PSF at spatial midpoint should be close to direct STPSF calculation."""
        pytest.importorskip("stpsf")

        import stpsf.roman

        # Generate payload with 2x2 spatial grid
        # Need at least 2 wavelengths for interpolation to work
        # Use wavelengths within STPSF aberration reference range [1.0, 1.9] microns
        wavelengths = np.array([1.4, 1.6])  # microns
        spatial_grid = {
            'x': np.array([1500.0, 2500.0]),
            'y': np.array([1500.0, 2500.0])
        }

        payload = psf_model.make_psf_payload(
            detector='WFI05',
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=5.0,
            verbose=False
        )

        # Test at exact center (midpoint of all 8 corners)
        xsca_test = 2000.0  # Midway between 1500 and 2500
        ysca_test = 2000.0
        wavelength_test = 1.5  # Midway between 1.4 and 1.6 microns

        # Get interpolated PSF
        psf_interp = psf_model.interpolate_psf(
            payload, xsca_test, ysca_test, wavelength_test
        )

        # Get direct STPSF PSF at same position
        wfi = stpsf.roman.WFI()
        wfi.filter = 'GRISM1'
        wfi.detector = 'WFI05'

        x_stpsf, y_stpsf = psf_utils.sca_to_stpsf_position(xsca_test, ysca_test)
        wfi.detector_position = (float(x_stpsf), float(y_stpsf))

        # STPSF expects wavelength in meters
        datacube = wfi.calc_datacube(
            np.array([wavelength_test * 1e-6]), fov_arcsec=5.0, oversample=4
        )
        psf_direct = datacube['OVERDIST'].data[0]

        # Compare total flux (should be within 1%)
        flux_interp = float(psf_interp.sum())
        flux_direct = float(psf_direct.sum())
        flux_error = abs(flux_interp - flux_direct) / flux_direct

        # Allow 1% error due to spatial interpolation
        assert flux_error < 0.01, \
            f"Flux error {flux_error:.2%} exceeds 1% threshold"


# ============================================================================
# PSF CACHING TESTS
# ============================================================================


class TestPSFCaching:
    """Test PSF payload caching (save/load)."""

    @pytest.fixture
    def simple_payload(self):
        """Create a simple test payload with analytical PSFs."""
        # Same as TestPSFInterpolation fixture
        # Wavelengths in microns
        wavelengths = np.array([1.0, 1.5, 1.9])
        spatial_x = np.array([1000.0, 2000.0, 3000.0])
        spatial_y = np.array([1000.0, 2000.0, 3000.0])

        psf_size = 10
        psf_grid = np.zeros((3, 3, 3, psf_size, psf_size))

        for iy in range(3):
            for ix in range(3):
                for iwl in range(3):
                    value = float(iy + ix + iwl)
                    psf_grid[iy, ix, iwl, :, :] = value

        payload = {
            'detector': 'WFI05',
            'order': '1',
            'wavelengths': jnp.array(wavelengths),
            'wl_grid': jnp.array(wavelengths),
            'spatial_x': jnp.array(spatial_x),
            'spatial_y': jnp.array(spatial_y),
            'psf_grid': jnp.array(psf_grid, dtype=jnp.float32),
            'psf_fov_pixels': psf_size,
            'pixel_scale': 0.11,
            'oversample': 4,
            'timing': {'total_time': 10.0, 'n_psfs': 27},
        }

        return payload

    def test_get_cache_filename(self):
        """Test cache filename generation."""
        # Wavelengths in microns
        wavelengths = np.arange(0.9, 2.01, 0.02)
        spatial_grid = {
            'x': np.linspace(1, 4088, 4),
            'y': np.linspace(1, 4088, 4)
        }

        filename = psf_model.get_cache_filename(
            detector='WFI05',
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=5.0,
            oversample=4,
        )

        # Check filename structure
        assert filename.startswith('psf_WFI05_GRISM1_')
        assert '4x4x56' in filename
        assert '0.90-2.00um' in filename
        assert 'fov5.0' in filename
        assert 'os4' in filename
        assert filename.endswith('.npz')

    def test_get_cache_filename_different_orders(self):
        """Test that different orders produce different filenames."""
        # Wavelengths in microns
        wavelengths = np.array([1.0, 1.5])
        spatial_grid = {'x': np.array([1000.0, 3000.0]), 'y': np.array([1000.0, 3000.0])}

        filename_0 = psf_model.get_cache_filename(
            'WFI05', '0', wavelengths, spatial_grid, 5.0, 4
        )
        filename_1 = psf_model.get_cache_filename(
            'WFI05', '1', wavelengths, spatial_grid, 5.0, 4
        )

        assert 'GRISM0' in filename_0
        assert 'GRISM1' in filename_1
        assert filename_0 != filename_1

    def test_save_and_load_roundtrip(self, simple_payload, tmp_path):
        """Test that save/load roundtrip preserves payload."""
        cache_file = tmp_path / "test_payload.npz"

        # Save
        psf_model.save_psf_payload(simple_payload, cache_file, verbose=False)
        assert cache_file.exists()

        # Load
        loaded = psf_model.load_psf_payload(cache_file, verbose=False)

        # Verify key fields match
        assert loaded['detector'] == simple_payload['detector']
        assert loaded['order'] == simple_payload['order']
        assert loaded['oversample'] == simple_payload['oversample']
        assert jnp.allclose(loaded['wavelengths'], simple_payload['wavelengths'])
        assert jnp.allclose(loaded['spatial_x'], simple_payload['spatial_x'])
        assert jnp.allclose(loaded['spatial_y'], simple_payload['spatial_y'])
        assert jnp.allclose(loaded['psf_grid'], simple_payload['psf_grid'])

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            psf_model.load_psf_payload(tmp_path / "nonexistent.npz", verbose=False)

    def test_interpolation_works_after_load(self, simple_payload, tmp_path):
        """Test that loaded payload works with interpolation."""
        cache_file = tmp_path / "test_payload.npz"

        # Save and reload
        psf_model.save_psf_payload(simple_payload, cache_file, verbose=False)
        loaded = psf_model.load_psf_payload(cache_file, verbose=False)

        # Interpolate with loaded payload (wavelength in microns)
        psf = psf_model.interpolate_psf(loaded, 2000.0, 2000.0, 1.5)

        # Should match expected value (same as test_interpolation_at_grid_points)
        expected_value = 3.0  # iwl=1, iy=1, ix=1
        assert jnp.allclose(psf, expected_value, rtol=1e-6)

    def test_get_or_make_without_cache_dir(self, simple_payload, monkeypatch):
        """Test get_or_make_psf_payload without cache_dir always generates."""
        # This test would be slow with real STPSF, so we mock make_psf_payload
        call_count = [0]

        def mock_make_psf_payload(**kwargs):
            call_count[0] += 1
            return simple_payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        # Without cache_dir, should call make_psf_payload
        result = psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=None, verbose=False,
            allow_generate=True,
        )

        assert call_count[0] == 1
        assert result is simple_payload

    def test_get_or_make_non_grism_filter_requires_wavelengths(self):
        """A non-grism stpsf_filter with default wavelengths must raise.

        The default wavelength grid is the grism band, and the band is baked
        into the cache filename — a PRISM call without wavelengths would look
        up (and silently regenerate) a 0.90-2.00um cache instead of finding
        the vendored 0.75-1.85um one. This bit the demo notebook on
        2026-08-05: it generated a wrong-band prism cache into the shared
        reference-data directory.
        """
        with pytest.raises(ValueError, match="wrong-band"):
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', stpsf_filter='PRISM',
                verbose=False,
            )

        # Passing the element band explicitly is the fix — this must NOT
        # raise at the wavelength-default stage (no cache_dir would proceed
        # to slow generation, so only check the guard via a fake cache dir
        # miss being the *next* failure mode is out of scope here).
        from roman_disperser import elements
        wl = elements.psf_cache_wavelengths(elements.PRISM)
        assert wl[0] == pytest.approx(0.75)
        assert wl[-1] == pytest.approx(1.85)

    def test_get_or_make_element_derives_filter_and_band(
            self, simple_payload, monkeypatch):
        """element= derives stpsf_filter and wavelengths together."""
        from roman_disperser import elements
        captured = {}

        def mock_make_psf_payload(**kwargs):
            captured.update(kwargs)
            return simple_payload

        monkeypatch.setattr(psf_model, 'make_psf_payload',
                            mock_make_psf_payload)
        psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', element='prism',
            cache_dir=None, verbose=False, allow_generate=True,
        )
        assert captured['stpsf_filter'] == 'PRISM'
        np.testing.assert_allclose(
            captured['wavelengths'],
            elements.psf_cache_wavelengths(elements.PRISM))

    def test_get_or_make_element_selects_vendored_cache_name(
            self, tmp_path, monkeypatch):
        """element=PRISM must resolve to the vendored prism cache filename.

        This is the end-to-end property the element= parameter exists for:
        the derived (filter, wavelengths) pair lands on the published cache
        file, not on a fresh wrong-band generation. The vendored file has the
        legacy name, so it takes the temporary fallback.
        """
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, "1")
        grid = psf_model.make_spatial_grid("4x4", psf_model.FULL_DETECTOR_BOUNDS)
        monkeypatch.setattr(
            psf_model, 'load_psf_payload',
            lambda path, verbose=True, **kw: {
                'name': Path(path).name,
                'spatial_x': grid['x'], 'spatial_y': grid['y']})
        expected = "psf_WFI05_PRISM_4x4x56_0.75-1.85um_fov5.0_os4.npz"
        (tmp_path / expected).write_bytes(b"placeholder")
        result = psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', element='prism',
            cache_dir=tmp_path, verbose=False,
        )
        assert result['name'] == expected

    def test_get_or_make_element_excludes_explicit_args(self):
        """element= with stpsf_filter= or wavelengths= must raise."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', element='prism',
                stpsf_filter='PRISM', verbose=False,
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', element='grism',
                wavelengths=np.arange(0.9, 2.01, 0.02), verbose=False,
            )

    def test_get_or_make_element_undefined_order_raises(self):
        """An order the element does not define fails loudly."""
        with pytest.raises(ValueError, match="not defined for element"):
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='2', element='prism', verbose=False,
            )

    @staticmethod
    def _labelled(payload):
        """The mock payload with the labels a real generation gives it."""
        return {**payload, 'stpsf_version': psf_model.TARGET_STPSF_VERSION,
                'add_ipc': False}

    @staticmethod
    def _request(payload):
        return dict(
            detector='WFI05', order='1', verbose=False,
            wavelengths=np.asarray(payload['wavelengths']),
            spatial_grid={'x': np.asarray(payload['spatial_x']),
                          'y': np.asarray(payload['spatial_y'])})

    def test_get_or_make_uses_cache(self, simple_payload, tmp_path, monkeypatch):
        """Test that get_or_make_psf_payload uses cache on second call."""
        call_count = [0]
        payload = self._labelled(simple_payload)

        def mock_make_psf_payload(**kwargs):
            call_count[0] += 1
            return payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        cache_dir = tmp_path / "cache"
        request = self._request(payload)

        # First call: generates and saves, because it says so
        with pytest.warns(RuntimeWarning, match="allow_generate"):
            result1 = psf_model.get_or_make_psf_payload(
                cache_dir=cache_dir, allow_generate=True, **request)
        assert call_count[0] == 1

        # Second call: should load from cache, without allow_generate
        result2 = psf_model.get_or_make_psf_payload(cache_dir=cache_dir, **request)
        assert call_count[0] == 1  # Still 1, not 2

        # Both should be equivalent
        assert jnp.allclose(result1['psf_grid'], result2['psf_grid'])

    def test_get_or_make_force_regenerate(self, simple_payload, tmp_path, monkeypatch):
        """Test that force_regenerate bypasses cache."""
        call_count = [0]
        payload = self._labelled(simple_payload)

        def mock_make_psf_payload(**kwargs):
            call_count[0] += 1
            return payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        cache_dir = tmp_path / "cache"
        request = self._request(payload)

        psf_model.get_or_make_psf_payload(
            cache_dir=cache_dir, force_regenerate=True, **request)
        assert call_count[0] == 1

        psf_model.get_or_make_psf_payload(
            cache_dir=cache_dir, force_regenerate=True, **request)
        assert call_count[0] == 2  # Should regenerate

    def test_cache_creates_directory(self, simple_payload, tmp_path, monkeypatch):
        """Test that caching creates cache directory if it doesn't exist."""
        def mock_make_psf_payload(**kwargs):
            return simple_payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        # Use a nested directory that doesn't exist
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        assert not cache_dir.exists()

        with pytest.warns(RuntimeWarning, match="allow_generate"):
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', cache_dir=cache_dir, verbose=False,
                allow_generate=True,
            )

        # Directory should have been created
        assert cache_dir.exists()
        # Should have one .npz file
        npz_files = list(cache_dir.glob("*.npz"))
        assert len(npz_files) == 1

    @pytest.mark.slow
    @pytest.mark.stpsf
    def test_stpsf_payload_cache_roundtrip(self, tmp_path):
        """Test full roundtrip: generate real STPSF payload, save, load, verify."""
        pytest.importorskip("stpsf")

        # Generate minimal real STPSF payload
        # Note: need at least 2 points per dimension for interpolation to work
        # Wavelengths in microns
        wavelengths = np.array([1.4, 1.6])
        spatial_grid = {
            'x': np.array([1500.0, 2500.0]),
            'y': np.array([1500.0, 2500.0])
        }

        original = psf_model.make_psf_payload(
            detector='WFI05',
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=5.0,
            verbose=False,
        )

        # Save to cache
        cache_file = tmp_path / "test_stpsf_payload.npz"
        psf_model.save_psf_payload(original, cache_file, verbose=False)
        assert cache_file.exists()

        # Load from cache
        loaded = psf_model.load_psf_payload(cache_file, verbose=False)

        # Verify metadata matches
        assert loaded['detector'] == original['detector']
        assert loaded['order'] == original['order']
        assert loaded['oversample'] == original['oversample']

        # Verify arrays match
        assert jnp.allclose(loaded['wavelengths'], original['wavelengths'])
        assert jnp.allclose(loaded['spatial_x'], original['spatial_x'])
        assert jnp.allclose(loaded['spatial_y'], original['spatial_y'])
        assert jnp.allclose(loaded['psf_grid'], original['psf_grid'])

        # Verify interpolation works and produces same results (wavelength in microns)
        test_wl = 1.5
        psf_original = psf_model.interpolate_psf(original, 2000.0, 2000.0, test_wl)
        psf_loaded = psf_model.interpolate_psf(loaded, 2000.0, 2000.0, test_wl)

        assert jnp.allclose(psf_original, psf_loaded, rtol=1e-6)

        # Verify flux is reasonable (should be ~1.0 for normalized PSF)
        flux = float(psf_loaded.sum())
        assert 0.95 < flux < 1.001, f"Unexpected flux: {flux}"


# ---------------------------------------------------------------------------
# F11: no JAX backend at import; cache workers are spawned, not forked
# ---------------------------------------------------------------------------

def test_import_does_not_initialise_jax_backend():
    """Importing the package must not start a JAX backend (on a GPU node
    that preallocates GPU memory before cache workers are started)."""
    import subprocess
    import sys
    code = ("import roman_disperser, roman_disperser.psf_model\n"
            "from jax._src import xla_bridge as xb\n"
            "print(xb.backends_are_initialized())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_parallel_cache_generation_spawns_workers(tmp_path, monkeypatch):
    """The worker pool uses the spawn start method and returns one result per
    task (without STPSF each task fails, which still exercises the pool)."""
    import multiprocessing
    try:
        import stpsf  # noqa: F401
        pytest.skip("STPSF installed: real generation would take hours")
    except ImportError:
        pass
    methods = []
    real = multiprocessing.get_context

    def spy(method=None):
        methods.append(method)
        return real(method)

    monkeypatch.setattr(multiprocessing, "get_context", spy)
    res = psf_model.generate_all_psf_caches(
        tmp_path, orders=("1",), detectors=["WFI01", "WFI02"],
        wavelengths=np.array([1.0, 1.5]),
        spatial_grid={"x": np.array([1.0]), "y": np.array([1.0])},
        n_workers=2, verbose=False)
    assert methods == ["spawn"]
    assert len(res["generated"]) + len(res["failed"]) == 2


# ---------------------------------------------------------------------------
# STPSF version and IPC handling of PSF caches (fake caches; no STPSF needed)
# ---------------------------------------------------------------------------

_WL = np.array([1.0, 1.5])
_GRID = {'x': np.array([1.0, 4088.0]), 'y': np.array([1.0, 4088.0])}
_STEM = "psf_WFI05_GRISM1_2x2x2_1.00-1.50um_fov5.0_os4"
_ABSENT = object()


def _fake_cache(directory, name, version=_ABSENT, add_ipc=_ABSENT, marker=1.0,
                **extra):
    """Write a tiny cache; ``marker`` fills psf_grid to tell files apart."""
    fields = dict(
        psf_grid=np.full((2, 2, 2, 8, 8), marker, dtype=np.float32),
        wavelengths=_WL, spatial_x=_GRID['x'], spatial_y=_GRID['y'],
        oversample=np.array(4), pixel_scale=np.array(0.11),
        fov_arcsec=np.array(5.0), detector=np.array('WFI05', dtype='S'),
        order=np.array('1', dtype='S'),
        stpsf_filter=np.array('GRISM1', dtype='S'))
    if version is not _ABSENT:
        fields['stpsf_version'] = np.array(version, dtype='S')
        fields['stpsf_hf_map'] = np.array(
            "RST_WFI5_Z210_high_frequency_map.fits", dtype='S')
    if add_ipc is not _ABSENT:
        fields['add_ipc'] = np.array(add_ipc)
    fields.update(extra)
    path = Path(directory) / name
    np.savez(path, **fields)
    return path


def _lookup(directory, **kw):
    return psf_model.get_or_make_psf_payload(
        detector='WFI05', order='1', wavelengths=_WL, spatial_grid=_GRID,
        cache_dir=directory, verbose=False, **kw)


def _lookup_grid(directory, grid):
    return psf_model.get_or_make_psf_payload(
        detector='WFI05', order='1', wavelengths=_WL, spatial_grid=grid,
        cache_dir=directory, verbose=False)


def _fake_generation(**kwargs):
    """Stand-in for make_psf_payload, labelled as a real generation."""
    return {'detector': kwargs['detector'], 'order': kwargs['order'],
            'stpsf_filter': 'GRISM1',
            'psf_grid': np.ones((2, 2, 2, 8, 8)), 'wavelengths': kwargs['wavelengths'],
            'spatial_x': kwargs['spatial_grid']['x'],
            'spatial_y': kwargs['spatial_grid']['y'],
            'oversample': 4, 'pixel_scale': 0.11, 'stpsf_version': '2.3.0',
            'add_ipc': kwargs['add_ipc']}


def _marker(payload):
    return float(payload['psf_grid'][0, 0, 0, 0, 0])


def _version_warnings(record):
    return [w for w in record
            if issubclass(w.category, psf_model.PSFCacheVersionWarning)]


def _load_warnings(path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        payload = psf_model.load_psf_payload(path, verbose=False)
    return payload, _version_warnings(record)


@pytest.fixture
def no_stpsf(monkeypatch):
    """Behave as if STPSF is not installed; any import of it fails."""
    monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: None)
    monkeypatch.setitem(sys.modules, 'stpsf', None)


@pytest.fixture(autouse=True)
def _fresh_cache_warnings():
    psf_model._reset_cache_warnings()
    yield
    psf_model._reset_cache_warnings()


class TestCacheFilename:
    def test_defaults_to_target_version(self):
        name = psf_model.get_cache_filename('WFI05', '1', _WL, _GRID, 5.0, 4)
        assert name == f"{_STEM}_stpsf{psf_model.TARGET_STPSF_VERSION}.npz"

    def test_documented_form(self):
        wl = np.arange(0.9, 2.01, 0.02)
        g = {'x': np.linspace(1, 4088, 4), 'y': np.linspace(1, 4088, 4)}
        assert psf_model.TARGET_STPSF_VERSION == "2.3.0"
        assert psf_model.get_cache_filename('WFI05', '1', wl, g, 5.0, 4) == (
            "psf_WFI05_GRISM1_4x4x56_0.90-2.00um_fov5.0_os4_stpsf2.3.0.npz")

    def test_ipc_name_cannot_collide(self):
        off = psf_model.get_cache_filename('WFI05', '1', _WL, _GRID, 5.0, 4)
        on = psf_model.get_cache_filename('WFI05', '1', _WL, _GRID, 5.0, 4,
                                          add_ipc=True)
        assert on == f"{_STEM}_stpsf2.3.0_ipc.npz" and on != off

    def test_explicit_version(self):
        name = psf_model.get_cache_filename(
            'WFI05', '1', _WL, _GRID, 5.0, 4, stpsf_version="2.4.1")
        assert name == f"{_STEM}_stpsf2.4.1.npz"

    @pytest.mark.parametrize("name,expected", [
        (f"{_STEM}.npz", (_STEM, None, False)),
        (f"{_STEM}_stpsf2.3.0.npz", (_STEM, "2.3.0", False)),
        (f"{_STEM}_stpsf2.3.0_ipc.npz", (_STEM, "2.3.0", True)),
        (f"{_STEM}_stpsf2.3.1.dev4+gabc.npz", (_STEM, "2.3.1.dev4+gabc", False)),
        ("test_payload.npz", (None, None, False)),
    ])
    def test_parse(self, name, expected):
        assert psf_model._parse_cache_filename(name) == expected

    @pytest.mark.parametrize("older,newer", [
        ("2.1.0", "2.3.0"), ("2.3.0rc1", "2.3.0"), ("2.3.0.dev5+gabc", "2.3.0"),
        ("2.3.0", "2.3.1.dev2"), ("2.9.0", "2.10.0"), ("2.3", "2.3.1"),
    ])
    def test_version_order(self, older, newer):
        assert psf_model._parse_version(older) < psf_model._parse_version(newer)


def _good_file(directory, add_ipc=False, marker=1.0):
    suffix = "_ipc" if add_ipc else ""
    return _fake_cache(directory, f"{_STEM}_stpsf2.3.0{suffix}.npz", "2.3.0",
                       add_ipc, marker=marker)


class TestStrictLookup:
    """Without the temporary fallback: the exact file or an error."""

    @pytest.fixture(autouse=True)
    def _no_fallback(self, monkeypatch):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)

    @pytest.fixture
    def no_generation(self, monkeypatch):
        def refuse(**kwargs):
            raise AssertionError("must not generate")
        monkeypatch.setattr(psf_model, 'make_psf_payload', refuse)

    def test_error_type(self):
        assert issubclass(psf_model.PSFCacheNotFoundError, FileNotFoundError)

    def test_exact_file_loads_silently(self, tmp_path, no_stpsf, no_generation):
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0", marker=9.0)
        _good_file(tmp_path, marker=3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            payload = _lookup(tmp_path)
        assert _marker(payload) == 3.0
        assert payload['stpsf_version'] == "2.3.0" and payload['add_ipc'] is False

    def test_missing_names_file_command_and_others(self, tmp_path, no_stpsf,
                                                   no_generation):
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0_ipc.npz", "2.3.0", True)
        _fake_cache(tmp_path, _STEM.replace("WFI05", "WFI06") + ".npz", "2.1.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            _lookup(tmp_path)
        message = str(info.value)
        assert f"{_STEM}_stpsf2.3.0.npz" in message
        assert f"{_STEM}.npz (STPSF 2.1.0, add_ipc unknown, 2x2 nodes)" in message
        assert f"{_STEM}_stpsf2.3.0_ipc.npz (STPSF 2.3.0, add_ipc True" in message
        assert "WFI06" not in message
        assert psf_model.ALLOW_FALLBACK_ENV in message

    def test_empty_directory_has_no_fallback_hint(self, tmp_path, no_stpsf):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            _lookup(tmp_path / "absent")
        assert psf_model.ALLOW_FALLBACK_ENV not in str(info.value)

    @staticmethod
    def _command(error):
        lines = str(error.value).splitlines()
        return next(line.strip() for line in lines if "generate_psf_caches.py" in line)

    def _default_request(self, directory, **kw):
        return psf_model.get_or_make_psf_payload(
            detector='WFI07', order='0', cache_dir=directory, verbose=False, **kw)

    def test_command_default(self, tmp_path, no_stpsf):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            self._default_request(tmp_path)
        assert self._command(info) == (
            f"python scripts/generate_psf_caches.py --cache-dir {tmp_path} "
            "--element grism --detectors WFI07 --orders 0")

    def test_command_ipc_and_grid_flags(self, tmp_path, no_stpsf):
        grid = psf_model.make_spatial_grid("5x3", "16,4055.5,187,4088")
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            self._default_request(tmp_path, spatial_grid=grid, add_ipc=True)
        assert self._command(info).endswith(
            "--element grism --detectors WFI07 --orders 0 --psf-grid 5x3 "
            "--psf-grid-bounds 16,4055.5,187,4088 --add-ipc")

    def test_command_quotes_a_path_with_a_space(self, tmp_path, no_stpsf):
        directory = tmp_path / "a b"
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            self._default_request(directory)
        assert f"--cache-dir '{directory}'" in self._command(info)

    def test_command_prism(self, tmp_path, no_stpsf):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', element='prism', cache_dir=tmp_path,
                verbose=False)
        assert "--element prism --detectors WFI05 --orders 1" in self._command(info)

    def test_command_runs_with_the_requested_filename(self, tmp_path, no_stpsf,
                                                      monkeypatch, capsys):
        """The command in the error makes exactly the missing file."""
        import shlex
        grid = psf_model.make_spatial_grid("5x3", "16,4055.5,187,4088")
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            self._default_request(tmp_path, spatial_grid=grid, add_ipc=True)
        script = _load_script("generate_psf_caches")
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return {'failed': []}
        monkeypatch.setattr(script.psf_model, 'generate_all_psf_caches', fake)
        argv = shlex.split(self._command(info))[1:]
        monkeypatch.setattr(sys, 'argv', argv)
        with pytest.raises(SystemExit):
            script.main()
        name = psf_model.get_cache_filename(
            'WFI07', '0', seen['wavelengths'], seen['spatial_grid'], 5.0, 4,
            stpsf_filter=seen['stpsf_filters']['0'], add_ipc=seen['add_ipc'])
        assert name in str(info.value)

    def test_non_default_settings_point_to_the_library(self, tmp_path, no_stpsf):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            _lookup(tmp_path)  # two wavelengths: the CLI cannot make this
        assert "generate_all_psf_caches" in str(info.value)

    @pytest.mark.parametrize("name,kwargs,needle", [
        (f"{_STEM}_stpsf2.3.0.npz", dict(version=_ABSENT, add_ipc=False),
         "no STPSF version"),
        (f"{_STEM}_stpsf2.3.0.npz", dict(version="unknown", add_ipc=False),
         "no STPSF version"),
        (f"{_STEM}_stpsf2.3.0.npz", dict(version="2.1.0", add_ipc=False),
         "metadata says 2.1.0"),
        (f"{_STEM}_stpsf2.3.0.npz", dict(version="2.3.0"), "no add_ipc metadata"),
        (f"{_STEM}_stpsf2.3.0.npz", dict(version="2.3.0", add_ipc=True),
         "add_ipc=True"),
        (f"{_STEM}_stpsf2.3.0.npz",
         dict(version="2.3.0", add_ipc=False, spatial_x=np.array([16.0, 4055.0])),
         "stored x nodes"),
        (f"{_STEM}_stpsf2.3.0.npz",
         dict(version="2.3.0", add_ipc=False, spatial_y=np.array([187.0, 4088.0])),
         "stored y nodes"),
        (f"{_STEM}_stpsf2.3.0.npz",
         dict(version="2.3.0", add_ipc=False, wavelengths=np.array([1.0, 1.4])),
         "wavelengths"),
        (f"{_STEM}_stpsf2.3.0.npz",
         dict(version="2.3.0", add_ipc=False, oversample=np.array(2)), "oversample"),
        (f"{_STEM}_stpsf2.3.0.npz",
         dict(version="2.3.0", add_ipc=False, detector=np.array('WFI06', dtype='S')),
         "detector"),
    ])
    def test_exact_name_with_wrong_metadata_is_refused(
            self, tmp_path, no_stpsf, no_generation, name, kwargs, needle):
        _fake_cache(tmp_path, name, **kwargs)
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            _lookup(tmp_path)
        message = str(info.value)
        assert "not the right file" in message and needle in message
        assert name in message

    def test_exact_name_wrong_metadata_command_overwrites(self, tmp_path, no_stpsf):
        wl = np.arange(0.9, 2.01, 0.02)
        grid = psf_model.make_spatial_grid()
        name = psf_model.get_cache_filename('WFI05', '1', wl, grid, 5.0, 4)
        _fake_cache(tmp_path, name, "2.3.0", wavelengths=wl,
                    spatial_x=grid['x'], spatial_y=grid['y'])
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', cache_dir=tmp_path, verbose=False)
        assert "--no-skip" in str(info.value)

    def test_exact_name_unreadable_is_refused(self, tmp_path, no_stpsf, no_generation):
        (tmp_path / f"{_STEM}_stpsf2.3.0.npz").write_bytes(b"not an npz")
        with pytest.raises(psf_model.PSFCacheNotFoundError, match="cannot be read"):
            _lookup(tmp_path)

    def test_differently_placed_nodes_never_return_the_default_file(
            self, tmp_path, no_stpsf, no_generation):
        """The bug the pupil study measured."""
        _good_file(tmp_path)
        moved = {'x': np.array([16.0, 4055.0]), 'y': np.array([187.0, 4088.0])}
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup_grid(tmp_path, moved)

    def test_no_generation_without_cache_dir(self, no_stpsf, no_generation):
        with pytest.raises(psf_model.PSFCacheNotFoundError, match="allow_generate"):
            psf_model.get_or_make_psf_payload(cache_dir=None, verbose=False)

    def test_require_cache_env_is_gone(self, tmp_path, no_stpsf, no_generation,
                                       monkeypatch):
        """It is now the default behaviour; setting it changes nothing."""
        monkeypatch.setenv("ROMAN_DISPERSER_REQUIRE_PSF_CACHE", "1")
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    def test_allow_generate_generates_and_saves(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.setattr(psf_model, 'make_psf_payload', _fake_generation)
        with pytest.warns(RuntimeWarning, match="allow_generate"):
            payload = _lookup(tmp_path, allow_generate=True, add_ipc=True)
        assert payload['add_ipc'] is True
        assert [p.name for p in tmp_path.iterdir()] == [f"{_STEM}_stpsf2.3.0_ipc.npz"]
        assert _lookup(tmp_path, add_ipc=True)['add_ipc'] is True

    def test_allow_generate_replaces_a_wrong_file(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.setattr(psf_model, 'make_psf_payload', _fake_generation)
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.1.0", False)
        with pytest.warns(RuntimeWarning, match="allow_generate"):
            _lookup(tmp_path, allow_generate=True)
        assert _lookup(tmp_path)['stpsf_version'] == "2.3.0"

    def test_force_regenerate_ignores_files(self, tmp_path, no_stpsf, monkeypatch):
        calls = []

        def counting(**kwargs):
            calls.append(1)
            return _fake_generation(**kwargs)
        monkeypatch.setattr(psf_model, 'make_psf_payload', counting)
        _good_file(tmp_path)
        _lookup(tmp_path, force_regenerate=True)
        assert calls == [1]

    def test_loaded_nodes_are_checked(self, tmp_path, no_stpsf, monkeypatch):
        """Even if a file passed the checks before loading."""
        _good_file(tmp_path)
        real = psf_model.load_psf_payload

        def shifted(path, **kw):
            return {**real(path, **kw), 'spatial_x': np.array([16.0, 4055.0])}
        monkeypatch.setattr(psf_model, 'load_psf_payload', shifted)
        with pytest.raises(ValueError, match="x nodes"):
            _lookup(tmp_path)

    def test_loaded_nodes_tolerate_float32(self, tmp_path, no_stpsf):
        grid = psf_model.make_spatial_grid("4x4", "16.3,4055.7,187.1,4087.9")
        name = psf_model.get_cache_filename('WFI05', '1', _WL, grid, 5.0, 4)
        _fake_cache(tmp_path, name, "2.3.0", False,
                    psf_grid=np.ones((4, 4, 2, 8, 8), np.float32),
                    spatial_x=grid['x'].astype(np.float32),
                    spatial_y=grid['y'].astype(np.float32))
        assert _lookup_grid(tmp_path, grid)['psf_grid'].shape[0] == 4


class TestHighFrequencyMap:
    """STPSF 2.3.0 leaves WFI01's map on every detector; caches must not."""

    class _WFI:
        """Just the parts of STPSF 2.3.0 that matter here."""
        _opd_file_dict = {'WFI01': '/d/RST_WFI1_Z210_high_frequency_map.fits',
                          'WFI18': '/d/RST_WFI18_Z210_high_frequency_map.fits'}
        _pupilopd = '/d/RST_WFI1_Z210_high_frequency_map.fits'

        @property
        def pupilopd(self):
            return self._pupilopd

    def test_generator_takes_the_detector_map(self):
        wfi = self._WFI()
        psf_model._use_detector_hf_map(wfi, 'WFI18')
        assert wfi.pupilopd.endswith('RST_WFI18_Z210_high_frequency_map.fits')

    def test_refuses_if_stpsf_no_longer_gives_it(self):
        changed = type('Changed', (), {'pupilopd': '/d/RST_WFI1_Z210_high_frequency_map.fits'})()
        with pytest.raises(RuntimeError, match="RST_WFI1_"):
            psf_model._use_detector_hf_map(changed, 'WFI18')

    def test_save_and_load_keep_the_label(self, tmp_path, no_stpsf):
        path = _good_file(tmp_path)
        payload = psf_model.load_psf_payload(path, verbose=False)
        assert payload['stpsf_hf_map'] == "RST_WFI5_Z210_high_frequency_map.fits"
        out = tmp_path / "copy.npz"
        psf_model.save_psf_payload(payload, out, verbose=False)
        with np.load(out) as data:
            assert data['stpsf_hf_map'].item() == b"RST_WFI5_Z210_high_frequency_map.fits"

    def _rewrite(self, path, value):
        fields = dict(np.load(path))
        if value is None:
            del fields['stpsf_hf_map']
        else:
            fields['stpsf_hf_map'] = np.array(value, dtype='S')
        with open(path, 'wb') as handle:
            np.savez(handle, **fields)

    def test_warns_if_no_record(self, tmp_path, no_stpsf):
        path = _good_file(tmp_path)
        self._rewrite(path, None)
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="no record of its high"):
            psf_model.load_psf_payload(path, verbose=False)

    def test_warns_if_wrong_map(self, tmp_path, no_stpsf):
        path = _good_file(tmp_path)
        self._rewrite(path, "RST_WFI1_Z210_high_frequency_map.fits")
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="RST_WFI1_"):
            psf_model.load_psf_payload(path, verbose=False)

    def test_old_version_is_not_asked_for_a_record(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        self._rewrite(path, None)
        with pytest.warns(psf_model.PSFCacheVersionWarning) as record:
            psf_model.load_psf_payload(path, verbose=False)
        assert "high-frequency" not in str(record[0].message)


class TestNodeBoundsLookup:
    """The rule for the default bounds without the temporary fallback."""

    @pytest.fixture(autouse=True)
    def _no_fallback(self, monkeypatch):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)

    def _write_full(self, directory, name, version=_ABSENT, **kw):
        grid = psf_model.make_spatial_grid("2x2", psf_model.FULL_DETECTOR_BOUNDS)
        return _fake_cache(directory, name, version,
                           spatial_x=grid['x'].astype(np.float32),
                           spatial_y=grid['y'].astype(np.float32), **kw)

    def test_full_detector_stpsf_file_does_not_serve_default(self, tmp_path, no_stpsf):
        self._write_full(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0", add_ipc=False)
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            _lookup_grid(tmp_path, psf_model.make_spatial_grid("2x2"))
        assert "x50-4050_y202-4088" in str(info.value)

    def test_full_detector_gaussian_serves_default(self, tmp_path, no_stpsf):
        self._write_full(tmp_path, f"{_STEM}.npz",
                         psf_source=np.array("gaussian", dtype="S"))
        with pytest.warns(UserWarning, match="gaussian"):
            payload = _lookup_grid(tmp_path, psf_model.make_spatial_grid("2x2"))
        assert payload['psf_source'] == 'gaussian'
        np.testing.assert_allclose(payload['spatial_x'], [1, 4088])

    def test_command_for_full_detector_names_its_bounds(self, tmp_path, no_stpsf):
        grid = psf_model.make_spatial_grid("4x4", psf_model.FULL_DETECTOR_BOUNDS)
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            psf_model.find_psf_cache_file(tmp_path, 'WFI05', 'GRISM1', spatial_grid=grid)
        assert "--psf-grid-bounds 1,4088,1,4088" in str(info.value)

    def test_command_for_default_has_no_bounds(self, tmp_path, no_stpsf):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            psf_model.find_psf_cache_file(tmp_path, 'WFI05', 'GRISM1')
        assert "--psf-grid-bounds" not in str(info.value)


class TestFallbackLookup:
    """TEMPORARY: remove once the 54 caches are regenerated on the cluster."""

    @pytest.fixture(autouse=True)
    def _fallback(self, monkeypatch):
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, "1")

    @pytest.mark.parametrize("value", ["0", "", "true"])
    def test_only_the_value_1_enables_it(self, tmp_path, no_stpsf, monkeypatch, value):
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, value)
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    def test_exact_target_beats_everything(self, tmp_path, no_stpsf):
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0", marker=1.0)
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.9.0.npz", "2.9.0", False, marker=2.0)
        _good_file(tmp_path, marker=3.0)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            payload = _lookup(tmp_path)
        assert _marker(payload) == 3.0
        assert _version_warnings(record) == []

    def test_takes_newest_and_warns(self, tmp_path, no_stpsf):
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0", marker=1.0)
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.2.0.npz", "2.2.0", marker=2.0)
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="stpsf2.2.0"):
            assert _marker(_lookup(tmp_path)) == 2.0

    def test_exact_name_without_ipc_metadata_loads_with_warning(self, tmp_path, no_stpsf):
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0")
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="no add_ipc"):
            assert _lookup(tmp_path)['add_ipc'] is None

    def test_legacy_name_used_and_nothing_generated(self, tmp_path, no_stpsf, monkeypatch):
        def no_generation(**kwargs):
            raise AssertionError("must not generate")
        monkeypatch.setattr(psf_model, 'make_psf_payload', no_generation)
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        with pytest.warns(psf_model.PSFCacheVersionWarning):
            payload = _lookup(tmp_path)
        assert payload['stpsf_version'] == "2.1.0"
        assert payload['add_ipc'] is False

    def test_nothing_matching_raises_and_never_generates(self, tmp_path, no_stpsf,
                                                         monkeypatch):
        def no_generation(**kwargs):
            raise AssertionError("must not generate")
        monkeypatch.setattr(psf_model, 'make_psf_payload', no_generation)
        _fake_cache(tmp_path, _STEM.replace("os4", "os44") + ".npz", "2.1.0")
        _fake_cache(tmp_path, _STEM.replace("WFI05", "WFI06") + ".npz", "2.1.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    def test_legacy_file_with_other_nodes_is_not_used(self, tmp_path, no_stpsf):
        """An old name says nothing about node placement; the stored nodes do."""
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0",
                    spatial_x=np.array([16.0, 4055.0]))
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    def _full_detector_legacy(self, directory, **kw):
        grid = psf_model.make_spatial_grid("2x2", psf_model.FULL_DETECTOR_BOUNDS)
        return _fake_cache(directory, f"{_STEM}.npz", "2.1.0",
                           spatial_x=grid['x'].astype(np.float32),
                           spatial_y=grid['y'].astype(np.float32), **kw)

    def test_default_bounds_use_a_full_detector_file(self, tmp_path, no_stpsf):
        """Today's files have the old nodes; they keep them, with a warning."""
        self._full_detector_legacy(tmp_path)
        grid = psf_model.make_spatial_grid("2x2")
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="used in place of"):
            payload = _lookup_grid(tmp_path, grid)
        np.testing.assert_allclose(payload['spatial_x'], [1, 4088])

    def test_default_bounds_prefer_their_own_file(self, tmp_path, no_stpsf):
        self._full_detector_legacy(tmp_path, marker=1.0)
        grid = psf_model.make_spatial_grid("2x2")
        name = psf_model.get_cache_filename(
            'WFI05', '1', _WL, grid, 5.0, 4, stpsf_filter='GRISM1')
        _fake_cache(tmp_path, name, "2.3.0", False, marker=2.0,
                    spatial_x=grid['x'].astype(np.float32),
                    spatial_y=grid['y'].astype(np.float32))
        assert _marker(_lookup_grid(tmp_path, grid)) == 2.0

    def test_other_bounds_do_not_use_a_full_detector_file(self, tmp_path, no_stpsf):
        self._full_detector_legacy(tmp_path)
        grid = psf_model.make_spatial_grid("2x2", "16,4055,187,4088")
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup_grid(tmp_path, grid)

    def test_invalid_candidate_falls_through(self, tmp_path, no_stpsf):
        (tmp_path / f"{_STEM}_stpsf2.2.0.npz").write_bytes(b"not an npz")
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0", marker=5.0)
        with pytest.warns(RuntimeWarning, match="invalid"):
            assert _marker(_lookup(tmp_path)) == 5.0


class TestIPCSeparation:
    @pytest.fixture(autouse=True)
    def _fallback(self, monkeypatch):
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, "1")

    def test_off_request_skips_ipc_file(self, tmp_path, no_stpsf):
        _good_file(tmp_path, add_ipc=True)
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    def test_on_request_skips_off_file(self, tmp_path, no_stpsf):
        _good_file(tmp_path)
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path, add_ipc=True)

    def test_each_request_gets_its_file(self, tmp_path, no_stpsf, monkeypatch):
        _good_file(tmp_path, marker=1.0)
        _good_file(tmp_path, add_ipc=True, marker=2.0)
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV)
        assert _marker(_lookup(tmp_path)) == 1.0
        assert _marker(_lookup(tmp_path, add_ipc=True)) == 2.0

    def test_metadata_overrides_a_misleading_name(self, tmp_path, no_stpsf):
        """A renamed IPC-on file must not satisfy an IPC-off request."""
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0", True)
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    @pytest.mark.parametrize("version", ["2.1.0", "2.2.0", _ABSENT, "unknown"])
    def test_legacy_without_ipc_serves_only_off(self, tmp_path, no_stpsf, version):
        _fake_cache(tmp_path, f"{_STEM}.npz", version)
        with pytest.warns(psf_model.PSFCacheVersionWarning):
            assert _lookup(tmp_path)['add_ipc'] is False
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path, add_ipc=True)

    @pytest.mark.parametrize("add_ipc", [False, True])
    def test_legacy_stamped_230_is_unknown_and_warns(self, tmp_path, no_stpsf,
                                                     add_ipc):
        """Old code under STPSF 2.3.0 wrote this; IPC may be in it either way."""
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.3.0")
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="IPC"):
            assert _lookup(tmp_path, add_ipc=add_ipc)['add_ipc'] is None

    def test_known_state_beats_unknown_at_same_version(self, tmp_path, no_stpsf):
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.3.0", marker=1.0)
        _good_file(tmp_path, add_ipc=True, marker=2.0)
        assert _marker(_lookup(tmp_path, add_ipc=True)) == 2.0


class TestCacheWarnings:
    def test_is_userwarning(self):
        assert issubclass(psf_model.PSFCacheVersionWarning, UserWarning)

    def _assert_one(self, record, path, *needles):
        assert len(record) == 1, [str(w.message) for w in record]
        message = str(record[0].message)
        assert str(path) in message
        assert "scripts/generate_psf_caches.py" in message
        for needle in needles:
            assert needle in message, message

    def test_older_than_target(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.2.0.npz", "2.2.0")
        self._assert_one(_load_warnings(path)[1], path, "2.2.0", "older")

    def test_older_from_filename_alone(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.2.0.npz")
        self._assert_one(_load_warnings(path)[1], path, "older", "unconfirmed")

    @pytest.mark.parametrize("version", [_ABSENT, "unknown"])
    def test_no_version_anywhere(self, tmp_path, no_stpsf, version):
        path = _fake_cache(tmp_path, f"{_STEM}.npz", version)
        payload, record = _load_warnings(path)
        self._assert_one(record, path, "no STPSF version")
        assert payload['stpsf_version'] is None

    def test_filename_and_metadata_disagree(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.1.0")
        payload, record = _load_warnings(path)
        self._assert_one(record, path, "metadata says 2.1.0")
        assert payload['stpsf_version'] == "2.1.0"

    def test_new_version_no_ipc_metadata(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0")
        payload, record = _load_warnings(path)
        self._assert_one(record, path, "no add_ipc metadata")
        assert payload['add_ipc'] is None

    def test_ipc_filename_and_metadata_disagree(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0_ipc.npz", "2.3.0", False)
        self._assert_one(_load_warnings(path)[1], path, "add_ipc=False")

    def test_installed_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: "2.4.0")
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0", False)
        self._assert_one(_load_warnings(path)[1], path, "installed STPSF is 2.4.0")

    def test_clean_file_is_quiet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: "2.3.0")
        path = _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0", False)
        payload, record = _load_warnings(path)
        assert record == []
        assert payload['stpsf_version'] == "2.3.0" and payload['add_ipc'] is False

    def test_no_stpsf_installed_path(self, tmp_path, no_stpsf):
        """Warnings fire without STPSF, and loading never imports it."""
        path = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        payload, record = _load_warnings(path)
        self._assert_one(record, path, "2.1.0", "older")
        assert payload['detector'] == 'WFI05'

    def test_load_does_not_import_stpsf(self, tmp_path, monkeypatch):
        """Importing STPSF with STPSF_PATH unset can trigger a data download."""
        monkeypatch.delitem(sys.modules, 'stpsf', raising=False)
        path = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        _load_warnings(path)
        assert 'stpsf' not in sys.modules

    def test_legacy_stamped_210_gets_one_warning(self, tmp_path, monkeypatch):
        """Also with STPSF 2.3.0 installed, through the lookup."""
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: "2.3.0")
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, "1")
        path = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _lookup(tmp_path)
        record = _version_warnings(record)
        self._assert_one(record, path, "2.1.0", "older", "used in place of",
                         "_stpsf2.3.0.npz")

    def test_repeated_loads_warn_once_per_file(self, tmp_path, no_stpsf):
        """As in warm-up plus per-SCA loops; 'always' would otherwise spam."""
        a = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        b = _fake_cache(tmp_path, _STEM.replace("WFI05", "WFI06") + ".npz", "2.1.0")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(50):
                psf_model.load_psf_payload(a, verbose=False)
                psf_model.load_psf_payload(b, verbose=False)
        messages = [str(w.message) for w in _version_warnings(record)]
        assert len(messages) == 2
        assert str(a) in messages[0] and str(b) in messages[1]

    def test_default_filter_shows_each_file(self, tmp_path, no_stpsf):
        """Python's default filter must not merge warnings of different files."""
        paths = [_fake_cache(tmp_path, _STEM.replace("WFI05", f"WFI{i:02d}") + ".npz",
                             "2.1.0") for i in range(1, 5)]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("default")
            for path in paths:
                psf_model.load_psf_payload(path, verbose=False)
        assert len(_version_warnings(record)) == 4

    def test_gaussian_cache_has_no_version_warning(self, tmp_path, no_stpsf):
        path = _fake_cache(tmp_path, f"{_STEM}.npz",
                           psf_source=np.array("gaussian", dtype="S"))
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            payload = psf_model.load_psf_payload(path, verbose=False)
        assert _version_warnings(record) == []
        assert [str(path) in str(w.message) for w in record] == [True]
        assert payload['psf_source'] == 'gaussian'

    def test_labels_do_not_enter_jit_payloads(self, tmp_path, no_stpsf):
        from roman_disperser import jit_payload
        on = _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0_ipc.npz", "2.3.0", True)
        off = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        results = [jit_payload.split_payload(
            psf_model.load_psf_payload(p, verbose=False),
            jit_payload.PSF_STATIC_KEYS) for p in (on, off)]
        for arrays, _ in results:
            assert not {'add_ipc', 'stpsf_version', 'psf_source'} & set(arrays)
        assert results[0][1] == results[1][1]
        assert (jax.tree.structure(results[0][0])
                == jax.tree.structure(results[1][0]))

    def test_generated_metadata_round_trip(self, tmp_path, no_stpsf):
        payload = psf_model.load_psf_payload(
            _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0_ipc.npz", "2.3.0", True),
            verbose=False)
        assert payload['add_ipc'] is True
        path = tmp_path / "again" / f"{_STEM}_stpsf2.3.0_ipc.npz"
        psf_model.save_psf_payload(payload, path, verbose=False)
        with np.load(path) as data:
            assert bool(data['add_ipc']) is True
            assert data['stpsf_version'].item() == b"2.3.0"

    def test_save_does_not_invent_ipc_state(self, tmp_path, no_stpsf):
        payload = psf_model.load_psf_payload(
            _fake_cache(tmp_path, f"{_STEM}.npz", "2.3.0"), verbose=False)
        path = tmp_path / "again.npz"
        psf_model.save_psf_payload(payload, path, verbose=False)
        with np.load(path) as data:
            assert 'add_ipc' not in data


class TestFindCacheFile:
    KW = dict(wavelengths=_WL, spatial_grid=_GRID)

    def test_exact_or_error(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError, match="2.1.0"):
            psf_model.find_psf_cache_file(tmp_path, "WFI05", "GRISM1", **self.KW)
        target = _good_file(tmp_path)
        assert psf_model.find_psf_cache_file(
            tmp_path, "WFI05", "GRISM1", **self.KW) == target

    def test_wrong_metadata_is_refused(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError, match="add_ipc"):
            psf_model.find_psf_cache_file(tmp_path, "WFI05", "GRISM1", **self.KW)

    def test_fallback_when_enabled(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, "1")
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        newer = _fake_cache(tmp_path, f"{_STEM}_stpsf2.2.0.npz", "2.2.0")
        with pytest.warns(psf_model.PSFCacheVersionWarning):
            assert psf_model.find_psf_cache_file(
                tmp_path, "WFI05", "GRISM1", **self.KW) == newer

    def test_respects_ipc(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.setenv(psf_model.ALLOW_FALLBACK_ENV, "1")
        _good_file(tmp_path, add_ipc=True)
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            psf_model.find_psf_cache_file(tmp_path, "WFI05", "GRISM1", **self.KW)


class TestGridOption:
    @pytest.mark.parametrize("text,expected", [
        ("5x5", (5, 5)), ("3x7", (3, 7)), ("4X4", (4, 4)), ((6, 2), (6, 2)),
        ([6, 2], (6, 2)),
    ])
    def test_parse_grid(self, text, expected):
        assert psf_model.parse_psf_grid(text) == expected

    @pytest.mark.parametrize("text", ["5", "5x", "x5", "5x5x5", "axb", "1x5", "5x0", ""])
    def test_parse_grid_rejects(self, text):
        with pytest.raises(ValueError):
            psf_model.parse_psf_grid(text)

    def test_parse_bounds(self):
        assert psf_model.parse_psf_grid_bounds("16,4055,187,4088") == (
            16.0, 4055.0, 187.0, 4088.0)
        assert psf_model.parse_psf_grid_bounds([1, 4088, 1, 4088]) == (
            1.0, 4088.0, 1.0, 4088.0)

    @pytest.mark.parametrize("text", [
        "16,4055,187", "16,4055,187,4088,5", "a,b,c,d", "4055,16,187,4088",
        "16,4055,4088,187", "16,16,187,4088", "0,4055,187,4088", "16,4089,187,4088",
        "16,4055,0.5,4088", "16,4055,187,5000",
    ])
    def test_parse_bounds_rejects(self, text):
        with pytest.raises(ValueError):
            psf_model.parse_psf_grid_bounds(text)

    def test_default_grid_is_inside_the_aberration_hull(self):
        grid = psf_model.make_spatial_grid()
        np.testing.assert_array_equal(grid['x'], np.linspace(50, 4050, 4))
        np.testing.assert_array_equal(grid['y'], np.linspace(202, 4088, 4))

    @pytest.mark.parametrize("element", ["GRISM0", "GRISM1", "PRISM"])
    def test_default_bounds_inside_every_field_point_hull(self, element):
        """No default node may be in STPSF's fallback, on any SCA."""
        interpolate = pytest.importorskip("scipy.interpolate")
        table = Path(os.environ.get("STPSF_PATH", "")) / (
            f"WFI/aberrations/{element}_5fields_zernike_Z45_multiwavelength.csv")
        if not table.is_file():
            pytest.skip("needs the STPSF 2.3.0 data package")
        with open(table, encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        xmin, xmax, ymin, ymax = psf_model.DEFAULT_PSF_GRID_BOUNDS
        # STPSF science coordinates are FITS SCA coordinates + 3
        corners = [(x + 3, y + 3) for x in (xmin, xmax) for y in (ymin, ymax)]
        for sca in range(1, 19):
            points = sorted({(float(r['SCI_X']), float(r['SCI_Y']))
                             for r in rows if int(float(r['sca'])) == sca})
            assert len(points) == 5
            values = interpolate.griddata(points, np.arange(5.0), corners,
                                          method='linear')
            assert np.all(np.isfinite(values)), f"SCA {sca}"

    def test_uniform_spacing_and_axes(self):
        grid = psf_model.make_spatial_grid("3x5", "100,300,1000,2000")
        np.testing.assert_array_equal(grid['x'], [100, 200, 300])
        np.testing.assert_array_equal(grid['y'], [1000, 1250, 1500, 1750, 2000])

    WL = np.arange(0.9, 2.01, 0.02)

    def _name(self, grid, **kw):
        return psf_model.get_cache_filename('WFI05', '1', self.WL, grid, 5.0, 4, **kw)

    def test_default_range_in_name(self):
        assert self._name(psf_model.make_spatial_grid()) == (
            "psf_WFI05_GRISM1_4x4x56_x50-4050_y202-4088_0.90-2.00um_fov5.0_os4"
            "_stpsf2.3.0.npz")
        assert self._name(psf_model.make_spatial_grid("5x5")) == (
            "psf_WFI05_GRISM1_5x5x56_x50-4050_y202-4088_0.90-2.00um_fov5.0_os4"
            "_stpsf2.3.0.npz")

    def test_default_range_in_float32_name_unchanged(self):
        grid = psf_model.make_spatial_grid()
        as_float32 = {k: v.astype(np.float32) for k, v in grid.items()}
        assert self._name(as_float32) == self._name(grid)

    def test_full_detector_has_no_tag(self):
        grid = psf_model.make_spatial_grid("4x4", psf_model.FULL_DETECTOR_BOUNDS)
        assert self._name(grid) == (
            "psf_WFI05_GRISM1_4x4x56_0.90-2.00um_fov5.0_os4_stpsf2.3.0.npz")
        as_float32 = {k: v.astype(np.float32) for k, v in grid.items()}
        assert "_4x4x56_0.90" in self._name(as_float32)

    def test_bounds_in_name(self):
        grid = psf_model.make_spatial_grid("4x4", "16,4055,187,4088")
        assert self._name(grid) == (
            "psf_WFI05_GRISM1_4x4x56_x16-4055_y187-4088_0.90-2.00um_fov5.0_os4"
            "_stpsf2.3.0.npz")

    def test_axes_differ_in_name(self):
        """Counts keep the NYxNXxNWL order of the name."""
        grid = psf_model.make_spatial_grid("3x5", "1,4088,187,4088")
        assert "_5x3x56_x1-4088_y187-4088_" in self._name(grid)

    def test_one_side_moved_gets_a_tag(self):
        grid = psf_model.make_spatial_grid("4x4", "1,4087,1,4088")
        assert "_x1-4087_y1-4088_" in self._name(grid)

    def test_fractional_bounds_rounded_in_name(self):
        grid = psf_model.make_spatial_grid("4x4", "16.5,4055.25,187.123,4088")
        assert "_x16.5-4055.25_y187.12-4088_" in self._name(grid)

    def test_non_uniform_grids_with_same_ends_differ(self):
        a = {'x': np.array([1.0, 100.0, 4088.0]), 'y': np.array([1.0, 2000.0, 4088.0])}
        b = {'x': np.array([1.0, 200.0, 4088.0]), 'y': np.array([1.0, 2000.0, 4088.0])}
        assert self._name(a) != self._name(b)
        assert "_nu" in self._name(a)
        assert self._name(psf_model.make_spatial_grid("3x3")) != self._name(a)

    def test_tag_with_ipc_parses(self):
        grid = psf_model.make_spatial_grid("4x4", "16,4055,187,4088")
        stem, version, ipc = psf_model._parse_cache_filename(
            self._name(grid, add_ipc=True))
        assert stem.endswith("_x16-4055_y187-4088_0.90-2.00um_fov5.0_os4")
        assert (version, ipc) == ("2.3.0", True)

    def test_two_placements_coexist(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)
        monkeypatch.setattr(psf_model, 'make_psf_payload', _fake_generation)
        grids = [psf_model.make_spatial_grid("2x2", b)
                 for b in ("1,4088,1,4088", "16,4055,187,4088")]
        for grid in grids:
            with pytest.warns(RuntimeWarning, match="allow_generate"):
                _lookup_grid_generate(tmp_path, grid)
        assert len(list(tmp_path.glob("*.npz"))) == 2
        for grid in grids:
            payload = _lookup_grid(tmp_path, grid)
            np.testing.assert_allclose(payload['spatial_x'], grid['x'], atol=5e-3)


def _lookup_grid_generate(directory, grid):
    return psf_model.get_or_make_psf_payload(
        detector='WFI05', order='1', wavelengths=_WL, spatial_grid=grid,
        cache_dir=directory, verbose=False, allow_generate=True)


class _FakeWFI:
    instances = []

    def __init__(self):
        self.options = {'add_ipc': True}  # STPSF 2.3.0's default
        _FakeWFI.instances.append(self)

    def calc_datacube(self, wavelengths, fov_arcsec, oversample):
        import types
        return {'OVERDIST': types.SimpleNamespace(
            data=np.ones((len(wavelengths), 8, 8)))}


@pytest.fixture
def fake_stpsf(monkeypatch):
    """A fake stpsf.roman at the target version, with STPSF_PATH set."""
    import types
    _FakeWFI.instances = []
    roman = types.ModuleType('stpsf.roman')
    roman.WFI = _FakeWFI
    module = types.ModuleType('stpsf')
    module.roman = roman
    monkeypatch.setitem(sys.modules, 'stpsf', module)
    monkeypatch.setitem(sys.modules, 'stpsf.roman', roman)
    monkeypatch.setattr(psf_model, '_installed_stpsf_version',
                        lambda: psf_model.TARGET_STPSF_VERSION)
    monkeypatch.setenv('STPSF_PATH', '/fake/stpsf-data')
    return _FakeWFI


class TestGeneration:
    @pytest.mark.parametrize("add_ipc", [False, True])
    def test_make_psf_payload(self, fake_stpsf, add_ipc):
        payload = psf_model.make_psf_payload(
            wavelengths=_WL, spatial_grid=_GRID, verbose=False, add_ipc=add_ipc)
        assert [w.options['add_ipc'] for w in fake_stpsf.instances] == [add_ipc]
        assert payload['add_ipc'] is add_ipc
        assert payload['stpsf_version'] == psf_model.TARGET_STPSF_VERSION

    def test_defaults_off_everywhere(self, fake_stpsf, tmp_path):
        psf_model.make_psf_payload(wavelengths=_WL, spatial_grid=_GRID, verbose=False)
        psf_model._make_psf_payload_with_progress(
            'WFI05', '1', _WL, _GRID, 5.0, 4)
        psf_model.generate_all_psf_caches(
            tmp_path / "a", orders=('1',), detectors=['WFI05'], wavelengths=_WL,
            spatial_grid=_GRID, verbose=False)
        with pytest.warns(RuntimeWarning, match="generating"):
            _lookup(tmp_path / "b", allow_generate=True)
        assert [w.options['add_ipc'] for w in fake_stpsf.instances] == [False] * 4

    @pytest.mark.parametrize("add_ipc", [False, True])
    def test_progress_variant(self, fake_stpsf, add_ipc):
        payload = psf_model._make_psf_payload_with_progress(
            'WFI05', '1', _WL, _GRID, 5.0, 4, stpsf_filter='GRISM1', add_ipc=add_ipc)
        assert fake_stpsf.instances[0].options['add_ipc'] is add_ipc
        assert payload['add_ipc'] is add_ipc
        assert payload['stpsf_filter'] == 'GRISM1'

    @pytest.mark.parametrize("add_ipc", [False, True])
    def test_worker_args_tuple(self, fake_stpsf, tmp_path, add_ipc):
        path = tmp_path / "out.npz"
        result = psf_model._generate_single_cache(
            ('WFI05', '1', str(path), _WL, _GRID, 5.0, 4, 'GRISM1', add_ipc))
        assert result[2] == 'generated', result
        assert fake_stpsf.instances[0].options['add_ipc'] is add_ipc
        with np.load(path) as data:
            assert bool(data['add_ipc']) is add_ipc

    @pytest.mark.parametrize("add_ipc,suffix", [(False, ""), (True, "_ipc")])
    def test_generate_all_sequential(self, fake_stpsf, tmp_path, add_ipc, suffix):
        res = psf_model.generate_all_psf_caches(
            tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
            spatial_grid=_GRID, verbose=False, add_ipc=add_ipc)
        assert res['failed'] == []
        path = tmp_path / f"{_STEM}_stpsf2.3.0{suffix}.npz"
        with np.load(path) as data:
            assert bool(data['add_ipc']) is add_ipc
            assert data['stpsf_version'].item() == b"2.3.0"
            assert data['stpsf_filter'].item() == b"GRISM1"
        assert fake_stpsf.instances[0].options['add_ipc'] is add_ipc

    def test_generate_all_builds_worker_tuples_with_ipc(self, fake_stpsf, tmp_path,
                                                        monkeypatch):
        """The parallel branch hands the same tuple to its workers."""
        import multiprocessing
        sent = []

        class Result:
            def __init__(self, value):
                self.value = value

            def ready(self):
                return True

            def get(self):
                return self.value

        class Pool:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def apply_async(self, function, args):
                sent.append(args[0])
                return Result(function(args[0]))

        class Context:
            Value = staticmethod(multiprocessing.Value)

            def Pool(self, *args, **kwargs):
                return Pool()

        monkeypatch.setattr(multiprocessing, 'get_context', lambda m: Context())
        res = psf_model.generate_all_psf_caches(
            tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
            spatial_grid=_GRID, n_workers=2, verbose=False, add_ipc=True)
        assert res['failed'] == []
        assert sent[0][-1] is True
        assert (tmp_path / f"{_STEM}_stpsf2.3.0_ipc.npz").exists()

    def test_skip_existing_is_per_ipc_state(self, fake_stpsf, tmp_path):
        kw = dict(orders=('1',), detectors=['WFI05'], wavelengths=_WL,
                  spatial_grid=_GRID, verbose=False)
        assert psf_model.generate_all_psf_caches(tmp_path, **kw)['generated']
        assert psf_model.generate_all_psf_caches(tmp_path, **kw)['skipped']
        assert psf_model.generate_all_psf_caches(
            tmp_path, add_ipc=True, **kw)['generated']

    def test_generate_ignores_legacy_file_for_skip(self, fake_stpsf, tmp_path):
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        res = psf_model.generate_all_psf_caches(
            tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
            spatial_grid=_GRID, verbose=False)
        assert res['generated'] == [('WFI05', '1')]

    def test_get_or_make_threads_ipc(self, fake_stpsf, tmp_path):
        with pytest.warns(RuntimeWarning, match="generating"):
            payload = _lookup(tmp_path, add_ipc=True, allow_generate=True)
        assert fake_stpsf.instances[0].options['add_ipc'] is True
        assert payload['add_ipc'] is True
        assert (tmp_path / f"{_STEM}_stpsf2.3.0_ipc.npz").exists()
        fake_stpsf.instances.clear()
        assert _lookup(tmp_path, add_ipc=True)['add_ipc'] is True
        assert fake_stpsf.instances == []

    def test_no_cache_dir_threads_ipc(self, fake_stpsf):
        _lookup(None, add_ipc=True, allow_generate=True)
        assert fake_stpsf.instances[0].options['add_ipc'] is True

    @pytest.mark.parametrize("version", ["2.1.0", "2.2.0", "2.3.0rc1"])
    def test_refuses_old_stpsf_everywhere(self, fake_stpsf, tmp_path, monkeypatch,
                                          version):
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: version)
        with pytest.raises(RuntimeError, match=version):
            psf_model.make_psf_payload(
                wavelengths=_WL, spatial_grid=_GRID, verbose=False)
        with pytest.raises(RuntimeError, match=version):
            psf_model._make_psf_payload_with_progress(
                'WFI05', '1', _WL, _GRID, 5.0, 4)
        with pytest.raises(RuntimeError, match=version):
            psf_model.generate_all_psf_caches(
                tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
                spatial_grid=_GRID, verbose=False)
        with pytest.warns(RuntimeWarning, match="generating"):
            with pytest.raises(RuntimeError, match=version):
                _lookup(tmp_path, allow_generate=True)
        result = psf_model._generate_single_cache(
            ('WFI05', '1', str(tmp_path / "x.npz"), _WL, _GRID, 5.0, 4, 'GRISM1',
             False))
        assert result[2] == 'failed' and version in result[3]
        assert fake_stpsf.instances == []
        assert list(tmp_path.glob("*.npz")) == []

    def test_old_stpsf_can_still_load_and_skip(self, fake_stpsf, tmp_path, monkeypatch):
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: "2.1.0")
        _fake_cache(tmp_path, f"{_STEM}_stpsf2.3.0.npz", "2.3.0", False)
        res = psf_model.generate_all_psf_caches(
            tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
            spatial_grid=_GRID, verbose=False)
        assert res['skipped'] == [('WFI05', '1')]

    def test_refuses_without_stpsf_path(self, fake_stpsf, monkeypatch):
        monkeypatch.delenv('STPSF_PATH')
        with pytest.raises(RuntimeError, match="STPSF_PATH"):
            psf_model.make_psf_payload(
                wavelengths=_WL, spatial_grid=_GRID, verbose=False)
        assert fake_stpsf.instances == []

    def test_refuses_without_stpsf(self, tmp_path, no_stpsf):
        with pytest.raises(ModuleNotFoundError, match="not installed"):
            psf_model.make_psf_payload(
                wavelengths=_WL, spatial_grid=_GRID, verbose=False)

    def test_newer_stpsf_is_labelled_truthfully(self, fake_stpsf, tmp_path, monkeypatch):
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: "2.4.0")
        with pytest.warns(RuntimeWarning, match="newer than the target"):
            psf_model.generate_all_psf_caches(
                tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
                spatial_grid=_GRID, verbose=False)
        path = tmp_path / f"{_STEM}_stpsf2.4.0.npz"
        with np.load(path) as data:
            assert data['stpsf_version'].item() == b"2.4.0"

    def test_real_generation_paths_store_the_nodes(self, fake_stpsf, tmp_path):
        grid = psf_model.make_spatial_grid("3x2", "16,4055,187,4088")
        res = psf_model.generate_all_psf_caches(
            tmp_path, orders=('1',), detectors=['WFI05'], wavelengths=_WL,
            spatial_grid=grid, verbose=False)
        assert res['failed'] == []
        path = tmp_path / ("psf_WFI05_GRISM1_2x3x2_x16-4055_y187-4088_1.00-1.50um"
                           "_fov5.0_os4_stpsf2.3.0.npz")
        payload = psf_model.load_psf_payload(path, verbose=False)
        assert payload['psf_grid'].shape[:2] == (2, 3)
        assert _lookup_grid(tmp_path, grid)['psf_grid'].shape[:2] == (2, 3)


def _load_script(name):
    import importlib.util
    path = Path(__file__).parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScripts:
    @pytest.mark.parametrize("flags,expected,warned", [
        ([], False, False), (["--no-add-ipc"], False, False),
        (["--add-ipc"], True, True),
    ])
    def test_generate_cli_flag(self, monkeypatch, capsys, tmp_path, flags, expected,
                               warned):
        script = _load_script("generate_psf_caches")
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return {'failed': []}
        monkeypatch.setattr(script.psf_model, 'generate_all_psf_caches', fake)
        monkeypatch.setattr(sys, 'argv', ['x', '--cache-dir', str(tmp_path), *flags])
        with pytest.raises(SystemExit) as stop:
            script.main()
        assert stop.value.code == 0
        assert seen['add_ipc'] is expected
        assert ("WARNING: --add-ipc" in capsys.readouterr().out) is warned

    def test_generate_cli_old_stpsf_exits_clearly(self, monkeypatch, tmp_path, fake_stpsf):
        script = _load_script("generate_psf_caches")
        monkeypatch.setattr(psf_model, '_installed_stpsf_version', lambda: "2.1.0")
        monkeypatch.setattr(sys, 'argv', ['x', '--cache-dir', str(tmp_path),
                                          '--detectors', 'WFI05', '--orders', '1'])
        with pytest.raises(SystemExit) as stop:
            script.main()
        assert "2.1.0" in str(stop.value.code)
        assert list(tmp_path.glob("*.npz")) == []

    def test_generate_cli_grid_options(self, monkeypatch, tmp_path):
        script = _load_script("generate_psf_caches")
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return {'failed': []}
        monkeypatch.setattr(script.psf_model, 'generate_all_psf_caches', fake)
        monkeypatch.setattr(sys, 'argv', [
            'x', '--cache-dir', str(tmp_path), '--psf-grid', '3x5',
            '--psf-grid-bounds', '16,4055,187,4088'])
        with pytest.raises(SystemExit):
            script.main()
        np.testing.assert_allclose(seen['spatial_grid']['x'], np.linspace(16, 4055, 3))
        np.testing.assert_allclose(seen['spatial_grid']['y'], np.linspace(187, 4088, 5))

    def test_generate_cli_default_grid(self, monkeypatch, tmp_path):
        script = _load_script("generate_psf_caches")
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return {'failed': []}
        monkeypatch.setattr(script.psf_model, 'generate_all_psf_caches', fake)
        monkeypatch.setattr(sys, 'argv', ['x', '--cache-dir', str(tmp_path)])
        with pytest.raises(SystemExit):
            script.main()
        np.testing.assert_array_equal(seen['spatial_grid']['x'], np.linspace(50, 4050, 4))
        np.testing.assert_array_equal(seen['spatial_grid']['y'], np.linspace(202, 4088, 4))

    @pytest.mark.parametrize("flags", [
        ['--psf-grid', '5'], ['--psf-grid', '1x5'],
        ['--psf-grid-bounds', '4055,16,187,4088'],
        ['--psf-grid-bounds', '16,5000,187,4088'], ['--psf-grid-bounds', '16,4055'],
    ])
    def test_generate_cli_rejects_bad_grid(self, monkeypatch, tmp_path, capsys, flags):
        script = _load_script("generate_psf_caches")
        monkeypatch.setattr(sys, 'argv', ['x', '--cache-dir', str(tmp_path), *flags])
        with pytest.raises(SystemExit) as stop:
            script.main()
        assert stop.value.code == 2

    def _default_cache(self, directory, order, version="2.3.0", add_ipc=False):
        wl = np.arange(0.9, 2.01, 0.02)
        grid = psf_model.make_spatial_grid()
        name = psf_model.get_cache_filename(
            'WFI05', '1', wl, grid, 5.0, 4, stpsf_filter=order)
        return _fake_cache(directory, name, version, add_ipc,
                           stpsf_filter=np.array(order, dtype='S'), wavelengths=wl,
                           spatial_x=grid['x'], spatial_y=grid['y'])

    def _gaussian_setup(self, monkeypatch, tmp_path):
        script = _load_script("make_gaussian_psf_cache")
        cache = tmp_path / "data" / "psf_cache"
        cache.mkdir(parents=True)
        monkeypatch.setenv("ROMAN_DISPERSER_DATA", str(tmp_path / "data"))
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)
        out = tmp_path / "gauss"
        monkeypatch.setattr(sys, 'argv', ['x', '--out-dir', str(out)])
        return script, cache, out

    def test_gaussian_script_marks_output(self, monkeypatch, tmp_path, no_stpsf):
        script, cache, out = self._gaussian_setup(monkeypatch, tmp_path)
        for order in ("GRISM0", "GRISM1"):
            self._default_cache(cache, order)
        script.main()
        path = out / ("psf_WFI05_GRISM1_4x4x56_x50-4050_y202-4088_0.90-2.00um"
                      "_fov5.0_os4.npz")
        with np.load(path) as data:
            assert data['psf_source'].item() == b"gaussian"
            assert 'stpsf_version' not in data and 'add_ipc' not in data
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            payload = psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', cache_dir=out, verbose=False)
        assert _version_warnings(record) == []
        assert [str(path) in str(w.message) for w in record] == [True]
        assert payload['psf_source'] == 'gaussian'

    def test_gaussian_stand_in_serves_only_ipc_off(self, monkeypatch, tmp_path, no_stpsf):
        script, cache, out = self._gaussian_setup(monkeypatch, tmp_path)
        for order in ("GRISM0", "GRISM1"):
            self._default_cache(cache, order)
        script.main()
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            psf_model.get_or_make_psf_payload(
                detector='WFI05', order='1', cache_dir=out, verbose=False,
                add_ipc=True)

    def test_exact_file_beats_gaussian_stand_in(self, monkeypatch, tmp_path, no_stpsf):
        script, cache, out = self._gaussian_setup(monkeypatch, tmp_path)
        for order in ("GRISM0", "GRISM1"):
            self._default_cache(cache, order)
        script.main()
        self._default_cache(out, "GRISM1")
        payload = psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=out, verbose=False)
        assert payload['psf_source'] == 'stpsf'

    def test_unmarked_legacy_is_not_a_stand_in(self, tmp_path, no_stpsf, monkeypatch):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)
        _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            _lookup(tmp_path)

    def test_gaussian_script_needs_exact_template(self, monkeypatch, tmp_path, no_stpsf):
        script, cache, out = self._gaussian_setup(monkeypatch, tmp_path)
        _fake_cache(cache, "psf_WFI05_GRISM0_4x4x56_0.90-2.00um_fov5.0_os4.npz", "2.1.0")
        with pytest.raises(psf_model.PSFCacheNotFoundError):
            script.main()

    def test_analyze_script_uses_strict_lookup(self):
        text = (Path(__file__).parent.parent / "scripts" / "analyze_psf_shifts.py").read_text()
        assert "find_psf_cache_file" in text and "glob.glob" not in text

    def test_migrate_script_warns(self, tmp_path, no_stpsf):
        script = _load_script("migrate_psf_caches")
        path = _fake_cache(tmp_path, f"{_STEM}.npz", "2.1.0")
        with pytest.warns(psf_model.PSFCacheVersionWarning, match="older"):
            status, _ = script.migrate_cache_file(path, dry_run=True)
        assert status == 'skipped'

    def test_hydrate_sca_filter_accepts_new_names(self):
        from roman_disperser import hydrate
        files = [(f"{_STEM}_stpsf2.3.0.npz", "u"),
                 (f"{_STEM}_stpsf2.3.0_ipc.npz", "u"), (f"{_STEM}.npz", "u"),
                 (_STEM.replace("WFI05", "WFI06") + "_stpsf2.3.0.npz", "u")]
        assert len(hydrate._filter_sca(files, [5])) == 3


@pytest.fixture(scope="module")
def bdi():
    """The driver script as a module (a script, not part of the package)."""
    import importlib.util
    path = Path(__file__).parent.parent / "scripts" / "build_dispersed_image.py"
    spec = importlib.util.spec_from_file_location("_bdi_psf_grid", path)
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["build_dispersed_image.py"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


class TestDriverGrid:
    """The driver asks for the requested grid at all its PSF-loading sites."""

    @pytest.fixture(autouse=True)
    def _strict(self, monkeypatch, no_stpsf):
        monkeypatch.delenv(psf_model.ALLOW_FALLBACK_ENV, raising=False)

    @pytest.fixture
    def paths_only(self, bdi, monkeypatch, tmp_path):
        """resolve_paths without hydrated data; the catalog must not be reached."""
        monkeypatch.setattr(bdi, "resolve_paths",
                            lambda *a, **k: ("c", "s", "m", tmp_path))

        def no_catalog(*a, **k):
            raise AssertionError("expensive work reached before the PSF check")
        monkeypatch.setattr(bdi, "load_catalog", no_catalog)
        return tmp_path

    def test_setup_fails_before_catalog(self, bdi, paths_only):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            bdi.setup_pipeline([5], psf_cache_dir=paths_only)
        message = str(info.value)
        assert ("psf_WFI05_GRISM0_4x4x56_x50-4050_y202-4088_0.90-2.00um_fov5.0_os4"
                "_stpsf2.3.0.npz") in message
        assert "--detectors WFI05 --orders 0" in message

    def test_setup_passes_grid_to_the_check(self, bdi, paths_only):
        with pytest.raises(psf_model.PSFCacheNotFoundError) as info:
            bdi.setup_pipeline([5], psf_cache_dir=paths_only, psf_grid="5x3",
                               psf_grid_bounds="16,4055,187,4088")
        message = str(info.value)
        assert "_3x5x56_x16-4055_y187-4088_" in message
        assert "--psf-grid 5x3 --psf-grid-bounds 16,4055,187,4088" in message

    def test_setup_rejects_bad_grid(self, bdi, paths_only):
        with pytest.raises(ValueError, match="not ordered"):
            bdi.setup_pipeline([5], psf_cache_dir=paths_only,
                               psf_grid_bounds="4055,16,187,4088")

    def test_check_covers_every_sca_and_filter(self, bdi, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(bdi.psf_model, "find_psf_cache_file",
                            lambda d, det, f, **k: seen.append((det, f)))
        bdi.check_psf_caches_exist([1, 18], bdi.elements.get_element("grism"), tmp_path,
                                   psf_model.make_spatial_grid())
        assert seen == [("WFI01", "GRISM0"), ("WFI01", "GRISM1"),
                        ("WFI18", "GRISM0"), ("WFI18", "GRISM1")]

    def test_quick_mode_passes_grid(self, bdi, monkeypatch, tmp_path):
        class Stop(Exception):
            pass
        seen = {}

        def fake_setup(sca_list, **kwargs):
            seen.update(kwargs)
            raise Stop
        monkeypatch.setattr(bdi, "setup_pipeline", fake_setup)
        with pytest.raises(Stop):
            bdi.build_dispersed_image(
                10.0, 0.0, 60.0, 5, tmp_path / "o.fits", seed=1,
                psf_grid="5x5", psf_grid_bounds="16,4055,187,4088")
        assert seen['psf_grid'] == "5x5"
        assert seen['psf_grid_bounds'] == "16,4055,187,4088"

    @pytest.mark.parametrize("flags", [
        ["--psf-grid", "1x5"], ["--psf-grid-bounds", "4055,16,187,4088"],
    ])
    def test_cli_rejects_bad_grid(self, bdi, monkeypatch, flags):
        monkeypatch.setattr(sys, "argv", ["x", *flags])
        with pytest.raises(SystemExit) as stop:
            bdi.main()
        assert stop.value.code == 2

    def _config(self, tmp_path, **extra):
        import yaml
        cfg = {"output_dir": str(tmp_path / "out"), "seed": 1, "scas": [5],
               "psf_grid": "5x3", "psf_grid_bounds": [16, 4055, 187, 4088], **extra}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_batch_accepts_and_passes_grid_keys(self, bdi, monkeypatch, tmp_path):
        class Stop(Exception):
            pass
        seen = {}

        def fake_setup(sca_list, **kwargs):
            seen.update(kwargs)
            raise Stop
        monkeypatch.setattr(bdi, "setup_pipeline", fake_setup)
        pointings = tmp_path / "p.ecsv"
        from astropy.table import Table
        Table({"PLAN": [1], "PASS": [1], "SEGMENT": [1], "OBSERVATION": [1],
               "VISIT": [1], "EXPOSURE": [1], "RA": [10.0], "DEC": [0.0], "PA": [60.0]}
              ).write(pointings, format="ascii.ecsv")
        with pytest.raises(Stop):
            bdi.run_batch(self._config(tmp_path), pointings, verbose=False)
        assert seen['psf_grid'] == "5x3"
        assert seen['psf_grid_bounds'] == [16, 4055, 187, 4088]

    def test_warmup_fails_before_catalog_with_grid(self, bdi, monkeypatch, tmp_path):
        monkeypatch.setattr(bdi, "resolve_paths",
                            lambda *a, **k: ("c", "s", "m", tmp_path))
        with pytest.raises(psf_model.PSFCacheNotFoundError, match="_3x5x56_x16-4055"):
            bdi.run_warmup(self._config(tmp_path), verbose=False)

    def test_process_sites_pass_the_grid(self):
        """Every get_or_make call in the driver names the grid."""
        import re
        text = (Path(__file__).parent.parent / "scripts"
                / "build_dispersed_image.py").read_text()
        calls = re.findall(r"get_or_make_psf_payload\((.*?)\n\s*\)", text, re.S)
        assert len(calls) == 4
        assert all("spatial_grid=" in call for call in calls)


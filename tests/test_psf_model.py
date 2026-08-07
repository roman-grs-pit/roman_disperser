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
            detector='WFI05', order='1', cache_dir=None, verbose=False
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
            cache_dir=None, verbose=False,
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
        file, not on a fresh wrong-band generation.
        """
        monkeypatch.setattr(
            psf_model, 'load_psf_payload',
            lambda path, verbose=True: ('loaded', Path(path).name))
        expected = "psf_WFI05_PRISM_4x4x56_0.75-1.85um_fov5.0_os4.npz"
        (tmp_path / expected).write_bytes(b"placeholder")
        result = psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', element='prism',
            cache_dir=tmp_path, verbose=False,
        )
        assert result == ('loaded', expected)

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

    def test_get_or_make_uses_cache(self, simple_payload, tmp_path, monkeypatch):
        """Test that get_or_make_psf_payload uses cache on second call."""
        call_count = [0]

        def mock_make_psf_payload(**kwargs):
            call_count[0] += 1
            return simple_payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        cache_dir = tmp_path / "cache"

        # First call: should generate and save
        result1 = psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=cache_dir, verbose=False
        )
        assert call_count[0] == 1

        # Second call: should load from cache (no new generation)
        result2 = psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=cache_dir, verbose=False
        )
        assert call_count[0] == 1  # Still 1, not 2

        # Both should be equivalent
        assert jnp.allclose(result1['psf_grid'], result2['psf_grid'])

    def test_get_or_make_force_regenerate(self, simple_payload, tmp_path, monkeypatch):
        """Test that force_regenerate bypasses cache."""
        call_count = [0]

        def mock_make_psf_payload(**kwargs):
            call_count[0] += 1
            return simple_payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        cache_dir = tmp_path / "cache"

        # First call
        psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=cache_dir, verbose=False
        )
        assert call_count[0] == 1

        # Second call with force_regenerate
        psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=cache_dir,
            force_regenerate=True, verbose=False
        )
        assert call_count[0] == 2  # Should regenerate

    def test_cache_creates_directory(self, simple_payload, tmp_path, monkeypatch):
        """Test that caching creates cache directory if it doesn't exist."""
        def mock_make_psf_payload(**kwargs):
            return simple_payload

        monkeypatch.setattr(psf_model, 'make_psf_payload', mock_make_psf_payload)

        # Use a nested directory that doesn't exist
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        assert not cache_dir.exists()

        psf_model.get_or_make_psf_payload(
            detector='WFI05', order='1', cache_dir=cache_dir, verbose=False
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

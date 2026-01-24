"""
Tests for PSF model and coordinate utilities.

Tests cover:
- Coordinate conversion (SCA <-> STPSF)
- PSF payload generation
- Trilinear interpolation
- JIT compilation
- Enclosed energy validation
"""

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
        # Minimal configuration: 2×2 spatial, 3 wavelengths
        wavelengths = np.linspace(0.9e-6, 2.0e-6, 3)
        spatial_grid = {
            'x': np.array([1000.0, 3000.0]),
            'y': np.array([1000.0, 3000.0])
        }

        payload = psf_model.make_psf_payload(
            detector='WFI05',
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=3.0,
            oversample=4,
            use_fast=True,  # Use fast method for testing
            verbose=True
        )

        # Check payload structure
        assert 'psf_grid' in payload
        assert 'wavelengths' in payload
        assert 'spatial_x' in payload
        assert 'spatial_y' in payload
        assert 'timing' in payload

        # Check PSF grid shape: [N_wl, N_y, N_x, PSF_y, PSF_x]
        psf_grid = payload['psf_grid']
        assert psf_grid.shape[0] == 3  # 3 wavelengths
        assert psf_grid.shape[1] == 2  # 2 y positions
        assert psf_grid.shape[2] == 2  # 2 x positions
        assert psf_grid.shape[3] > 0  # PSF has size
        assert psf_grid.shape[4] > 0

        # Check PSFs are normalized (approximately)
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    psf = psf_grid[i, j, k]
                    total_flux = float(psf.sum())
                    # OVERSAMP PSFs should be normalized to ~1.0
                    assert 0.95 < total_flux < 1.05, \
                        f"PSF flux {total_flux} outside [0.95, 1.05]"

    @pytest.mark.slow
    def test_payload_timing_reported(self):
        """Check that timing information is captured."""
        wavelengths = np.linspace(0.9e-6, 2.0e-6, 2)
        spatial_grid = {
            'x': np.array([2000.0]),
            'y': np.array([2000.0])
        }

        payload = psf_model.make_psf_payload(
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            use_fast=True,
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
        wavelengths = np.linspace(0.9e-6, 2.0e-6, 2)
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
            use_fast=True,
            verbose=False
        )

        # Check order is stored
        assert payload['order'] == '0'

        # Check PSF grid exists and has correct shape
        assert payload['psf_grid'].shape[0] == 2  # 2 wavelengths
        assert payload['psf_grid'].shape[1] == 1  # 1 y position
        assert payload['psf_grid'].shape[2] == 1  # 1 x position

    def test_invalid_order_raises_error(self):
        """Test that invalid order raises ValueError."""
        with pytest.raises(ValueError, match="Invalid order"):
            psf_model.make_psf_payload(
                order='99',  # Invalid
                wavelengths=np.array([1.5e-6]),
                spatial_grid={'x': np.array([2000.0]), 'y': np.array([2000.0])},
                use_fast=True,
                verbose=False
            )


# ============================================================================
# PSF INTERPOLATION TESTS
# ============================================================================


class TestPSFInterpolation:
    """Test trilinear PSF interpolation."""

    @pytest.fixture
    def simple_payload(self):
        """Create a simple test payload with analytical PSFs."""
        # Create a minimal payload with known PSF values for testing
        wavelengths = np.array([1.0e-6, 1.5e-6, 1.9e-6])
        spatial_x = np.array([1000.0, 2000.0, 3000.0])
        spatial_y = np.array([1000.0, 2000.0, 3000.0])

        # Create simple PSF grid (just identity matrices for testing)
        psf_size = 10
        psf_grid = np.zeros((3, 3, 3, psf_size, psf_size))

        # Fill with simple patterns (different for each position/wavelength)
        for iwl in range(3):
            for iy in range(3):
                for ix in range(3):
                    # Create a simple pattern: value = iwl + iy + ix
                    value = float(iwl + iy + ix)
                    psf_grid[iwl, iy, ix, :, :] = value

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
        # Test at grid point: (2000.0, 2000.0, 1.5e-6)
        # This is index [1, 1, 1] in the grid
        xsca, ysca, wavelength = 2000.0, 2000.0, 1.5e-6

        psf = psf_model.interpolate_psf(simple_payload, xsca, ysca, wavelength)

        # Should match grid value at [1, 1, 1]
        # value = iwl + iy + ix = 1 + 1 + 1 = 3.0
        # Allow some numerical error from trilinear interpolation
        expected_value = 3.0
        assert jnp.allclose(psf, expected_value, rtol=1e-3, atol=1e-3)

    def test_interpolation_shape(self, simple_payload):
        """Interpolated PSF should have correct shape."""
        xsca, ysca, wavelength = 2000.0, 2000.0, 1.5e-6

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
        psf1 = interp_jit(2000.0, 2000.0, 1.5e-6)

        # Second call (cached)
        psf2 = interp_jit(2100.0, 2100.0, 1.6e-6)

        # Both should have same shape
        assert psf1.shape == psf2.shape

    def test_edge_extrapolation(self, simple_payload):
        """Interpolation should use edge values for out-of-bounds positions."""
        # Test position outside grid (x < x_min)
        xsca, ysca, wavelength = 500.0, 2000.0, 1.5e-6

        # Should not crash, should use nearest edge value
        psf = psf_model.interpolate_psf(simple_payload, xsca, ysca, wavelength)

        # Should have valid shape
        assert psf.shape == (10, 10)

        # Test high edge (x > x_max)
        xsca_high = 4000.0
        psf_high = psf_model.interpolate_psf(simple_payload, xsca_high, ysca, wavelength)
        assert psf_high.shape == (10, 10)

    def test_vectorized_interpolation(self, simple_payload):
        """Interpolation should work with array inputs via vmap."""
        # Create arrays of positions
        xsca = jnp.array([1000.0, 2000.0, 3000.0])
        ysca = jnp.array([1000.0, 2000.0, 3000.0])
        wavelength = jnp.array([1.0e-6, 1.5e-6, 1.9e-6])

        # Vectorize interpolation
        interp_vmap = jax.vmap(
            lambda x, y, wl: psf_model.interpolate_psf(simple_payload, x, y, wl)
        )

        psfs = interp_vmap(xsca, ysca, wavelength)

        # Should have shape [3, 10, 10]
        assert psfs.shape == (3, 10, 10)


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
        wavelengths = np.array([0.9e-6, 1.5e-6, 2.0e-6])
        spatial_grid = {
            'x': np.array([1500.0, 2500.0]),
            'y': np.array([1500.0, 2500.0])
        }

        payload = psf_model.make_psf_payload(
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=3.0,
            use_fast=True,
            verbose=False
        )

        # Test interpolation at mid-point between grid points
        xsca_test = 2000.0  # Midway between 1500 and 2500
        ysca_test = 2000.0
        wavelength_test = 1.5e-6  # Exact grid wavelength

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

        datacube = wfi.calc_datacube_fast(
            np.array([wavelength_test]), fov_arcsec=3.0, oversample=4
        )
        psf_direct = datacube['OVERSAMP'].data[0]  # First wavelength

        # Compare total flux (should be within a few percent)
        flux_interp = float(psf_interp.sum())
        flux_direct = float(psf_direct.sum())
        flux_error = abs(flux_interp - flux_direct) / flux_direct

        # Allow 5% error due to spatial interpolation
        assert flux_error < 0.05, \
            f"Flux error {flux_error:.2%} exceeds 5% threshold"

    @pytest.mark.slow
    @pytest.mark.stpsf
    def test_enclosed_energy_all_psfs(self):
        """Validate enclosed energy for all PSFs in grid."""
        pytest.importorskip("stpsf")

        # Generate minimal payload
        wavelengths = np.linspace(0.9e-6, 2.0e-6, 3)
        spatial_grid = {
            'x': np.array([2000.0]),
            'y': np.array([2000.0])
        }

        payload = psf_model.make_psf_payload(
            order='1',
            wavelengths=wavelengths,
            spatial_grid=spatial_grid,
            fov_arcsec=3.0,
            use_fast=True,
            verbose=False
        )

        # Check all PSFs have flux ~1.0 (normalized)
        psf_grid = payload['psf_grid']
        for iwl in range(psf_grid.shape[0]):
            for iy in range(psf_grid.shape[1]):
                for ix in range(psf_grid.shape[2]):
                    psf = psf_grid[iwl, iy, ix]
                    total_flux = float(psf.sum())

                    # OVERSAMP PSFs should be normalized
                    assert 0.95 < total_flux < 1.05, \
                        f"PSF[{iwl},{iy},{ix}] flux {total_flux} outside [0.95, 1.05]"

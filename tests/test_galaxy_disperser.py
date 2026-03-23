"""
Tests for galaxy_disperser module.

These tests verify:
1. SCA-to-SCA trace and Jacobian computation
2. Galaxy shape warping through Jacobian
3. Convolution with PSFs
4. Full galaxy dispersion pipeline
5. JIT-compiled galaxy disperser factory

Note: All wavelengths are in **microns** (consistent with optical model).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roman_disperser.galaxy_disperser import (
    trace_beam_sca,
    trace_beam_sca_with_jacobian,
    disperse_galaxy_shape,
    prepare_galaxy_images,
    disperse_galaxy,
    make_galaxy_disperser,
)

RTOL = 1e-4
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_psf_payload():
    """Create a mock PSF payload with small delta PSFs for testing."""
    n_y, n_x, n_wl = 4, 4, 10
    psf_size = 20
    oversample = 4

    psf_grid = jnp.zeros(
        (n_y, n_x, n_wl, psf_size, psf_size), dtype=jnp.float32
    )
    center = psf_size // 2
    for iy in range(n_y):
        for ix in range(n_x):
            for iw in range(n_wl):
                psf_grid = psf_grid.at[iy, ix, iw, center, center].set(1.0)

    return {
        'psf_grid': psf_grid,
        'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
        'wl_grid': jnp.linspace(1.0, 1.8, n_wl),
        'spatial_x': jnp.linspace(1, 4088, n_x),
        'spatial_y': jnp.linspace(1, 4088, n_y),
        'oversample': oversample,
        'detector': 'WFI05',
        'order': '1',
    }


@pytest.fixture
def mock_psf_payload_small():
    """Smaller PSF payload for faster factory/JIT tests."""
    n_y, n_x, n_wl = 2, 2, 5
    psf_size = 12
    oversample = 4

    psf_grid = jnp.zeros(
        (n_y, n_x, n_wl, psf_size, psf_size), dtype=jnp.float32
    )
    center = psf_size // 2
    for iy in range(n_y):
        for ix in range(n_x):
            for iw in range(n_wl):
                psf_grid = psf_grid.at[iy, ix, iw, center, center].set(1.0)

    return {
        'psf_grid': psf_grid,
        'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
        'wl_grid': jnp.linspace(1.0, 1.8, n_wl),
        'spatial_x': jnp.linspace(1, 4088, n_x),
        'spatial_y': jnp.linspace(1, 4088, n_y),
        'oversample': oversample,
        'detector': 'WFI05',
        'order': '1',
    }


# ---------------------------------------------------------------------------
# Test: trace_beam_sca
# ---------------------------------------------------------------------------

class TestTraceBeamSca:
    """Test the SCA-to-SCA trace function."""

    def test_returns_scalars(self, payload):
        """Output should be scalar values."""
        x_out, y_out = trace_beam_sca(
            payload, jnp.float32(2000.0), jnp.float32(2000.0),
            jnp.float32(1.5),
        )
        assert x_out.shape == ()
        assert y_out.shape == ()

    def test_reference_wavelength_near_input(self, payload):
        """At reference wavelength, output should be near input position."""
        wl_ref = float(payload['wl']['reference'])
        x_out, y_out = trace_beam_sca(
            payload, jnp.float32(2000.0), jnp.float32(2000.0),
            jnp.float32(wl_ref),
        )
        # At reference wavelength, dispersion offset is zero but coordinate
        # transforms still apply — just check result is reasonable
        assert 1.0 < float(x_out) < 4088.0
        assert 1.0 < float(y_out) < 4088.0

    def test_matches_vectorized_trace(self, payload):
        """Should match the vectorized trace from star_disperser."""
        from roman_disperser.star_disperser import _compute_dispersed_positions

        xsca, ysca, wl = 2000.0, 2000.0, 1.3
        x_scalar, y_scalar = trace_beam_sca(
            payload, jnp.float32(xsca), jnp.float32(ysca), jnp.float32(wl),
        )
        x_vec, y_vec = _compute_dispersed_positions(
            payload, xsca, ysca, jnp.array([wl]),
        )
        np.testing.assert_allclose(float(x_scalar), float(x_vec[0]), rtol=1e-5)
        np.testing.assert_allclose(float(y_scalar), float(y_vec[0]), rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: Jacobian computation
# ---------------------------------------------------------------------------

class TestJacobian:
    """Test the Jacobian computation against finite differences."""

    def test_jacobian_shape(self, payload):
        """Jacobian should be 2x2."""
        x_out, y_out, J = trace_beam_sca_with_jacobian(
            payload, jnp.float32(2000.0), jnp.float32(2000.0),
            jnp.float32(1.5),
        )
        assert J.shape == (2, 2)

    def test_jacobian_near_identity(self, payload):
        """Jacobian should be close to identity (||J-I|| ~ 0.02-0.04)."""
        _, _, J = trace_beam_sca_with_jacobian(
            payload, jnp.float32(2000.0), jnp.float32(2000.0),
            jnp.float32(1.5),
        )
        frob = float(jnp.linalg.norm(J - jnp.eye(2)))
        assert frob < 0.1  # Loose upper bound
        assert frob > 0.001  # Should not be exactly identity

    def test_jacobian_vs_finite_differences(self, payload):
        """Autodiff Jacobian should match finite-difference Jacobian."""
        jax.config.update("jax_enable_x64", True)
        try:
            x0 = jnp.float64(2044.0)
            y0 = jnp.float64(2044.0)
            wl = jnp.float64(1.2)
            eps = 1e-4

            *_, J_auto = trace_beam_sca_with_jacobian(payload, x0, y0, wl)

            # Finite differences
            xp, yp = trace_beam_sca(payload, x0 + eps, y0, wl)
            xm, ym = trace_beam_sca(payload, x0 - eps, y0, wl)
            dxdx = (xp - xm) / (2 * eps)
            dydx = (yp - ym) / (2 * eps)

            xp, yp = trace_beam_sca(payload, x0, y0 + eps, wl)
            xm, ym = trace_beam_sca(payload, x0, y0 - eps, wl)
            dxdy = (xp - xm) / (2 * eps)
            dydy = (yp - ym) / (2 * eps)

            J_fd = np.array([[float(dxdx), float(dxdy)],
                             [float(dydx), float(dydy)]])

            np.testing.assert_allclose(
                np.array(J_auto), J_fd, rtol=1e-4, atol=1e-6
            )
        finally:
            jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Test: disperse_galaxy_shape
# ---------------------------------------------------------------------------

class TestDisperseGalaxyShape:
    """Test the galaxy shape warping function."""

    def test_identity_jacobian_preserves_image(self):
        """Identity Jacobian should return the original image."""
        image = jnp.ones((20, 20), dtype=jnp.float32) * 5.0
        J = jnp.eye(2)
        warped = disperse_galaxy_shape(image, J, dx=0.25, dy=0.25)

        # Should preserve all flux
        np.testing.assert_allclose(
            float(warped.sum()), float(image.sum()), rtol=1e-5
        )
        # Image should be essentially unchanged
        np.testing.assert_allclose(warped, image, atol=1e-5)

    def test_flux_conservation(self):
        """Total flux should be conserved for a well-padded Gaussian."""
        # Gaussian with sigma=3 oversampled pixels in 40x40 image.
        # Extends to ~6.5 sigma from center — negligible flux at edges.
        y, x = jnp.mgrid[:40, :40]
        image = jnp.exp(-((x - 19.5)**2 + (y - 19.5)**2) / (2 * 3.0**2))
        image = image.astype(jnp.float32)

        # Realistic Jacobian perturbation (||J-I|| ~ 0.03)
        J = jnp.array([[0.99, -0.01], [-0.01, 1.02]])
        warped = disperse_galaxy_shape(image, J, dx=0.25, dy=0.25)

        # Flux loss is negligible — only affects pixels at >6 sigma
        np.testing.assert_allclose(
            float(warped.sum()), float(image.sum()), rtol=1e-5
        )

    def test_known_shift(self):
        """Delta at center should stay at center regardless of J."""
        image = jnp.zeros((21, 21), dtype=jnp.float32)
        image = image.at[10, 10].set(1.0)  # Delta at center

        # Center pixel has zero relative offset, so J @ (0,0) = (0,0).
        # The delta should stay at pixel (10,10) for any J.
        J = jnp.eye(2)
        warped = disperse_galaxy_shape(image, J, dx=1.0, dy=1.0)
        assert float(warped[10, 10]) == pytest.approx(1.0, abs=1e-5)

    def test_nonidentity_jacobian_moves_off_center_pixel(self):
        """Non-identity J should displace an off-center delta function."""
        # Place delta at pixel (10, 15) in a 21x21 image (center at 10,10).
        # Relative offset in pixel coords: (5, 0). With dx=1.0:
        #   rel_x = 5.0, rel_y = 0.0
        # Apply J = [[1, 0], [0, 2]]:
        #   warped_x = 5.0, warped_y = 0.0
        # So out_j = 5.0/1.0 + 10 = 15, out_i = 0.0/1.0 + 10 = 10
        # => pixel (10, 15) still (J only scales y, and rel_y=0 here)
        #
        # Now try J = [[2, 0], [0, 1]] which doubles x:
        #   warped_x = 10.0 => out_j = 10/1 + 10 = 20
        # => flux moves from pixel (10,15) to pixel (10,20)
        image = jnp.zeros((21, 21), dtype=jnp.float32)
        image = image.at[10, 15].set(1.0)

        J = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        warped = disperse_galaxy_shape(image, J, dx=1.0, dy=1.0)

        # Flux should move to pixel (10, 20)
        assert float(warped[10, 20]) == pytest.approx(1.0, abs=1e-5)
        # Original location should be empty
        assert float(warped[10, 15]) == pytest.approx(0.0, abs=1e-5)
        # Total flux conserved
        np.testing.assert_allclose(float(warped.sum()), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: prepare_galaxy_images
# ---------------------------------------------------------------------------

class TestPrepareGalaxyImages:
    """Test the convolution preparation step."""

    def test_output_shapes(self, payload, mock_psf_payload):
        """Output shapes should match expected dimensions."""
        image = jnp.ones((20, 20), dtype=jnp.float32)
        dx = dy = 1.0 / mock_psf_payload['oversample']

        convolved, cx, cy, wl = prepare_galaxy_images(
            payload, mock_psf_payload, image, 2000.0, 2000.0, dx, dy,
        )

        n_wl = len(mock_psf_payload['wavelengths'])
        psf_y, psf_x = mock_psf_payload['psf_grid'].shape[-2:]
        expected_conv_y = 20 + psf_y - 1
        expected_conv_x = 20 + psf_x - 1

        assert convolved.shape == (n_wl, expected_conv_y, expected_conv_x)
        assert cx.shape == (n_wl,)
        assert cy.shape == (n_wl,)
        assert wl.shape == (n_wl,)

    def test_delta_image_flux(self, payload, mock_psf_payload):
        """Delta image convolved with delta PSF should preserve flux exactly."""
        image = jnp.zeros((20, 20), dtype=jnp.float32)
        image = image.at[10, 10].set(1.0)
        dx = dy = 1.0 / mock_psf_payload['oversample']

        convolved, _, _, _ = prepare_galaxy_images(
            payload, mock_psf_payload, image, 2000.0, 2000.0, dx, dy,
        )

        # Bilinear scatter preserves total flux exactly (weights sum to 1),
        # and convolution with a unit-sum PSF preserves total flux.
        for i in range(convolved.shape[0]):
            np.testing.assert_allclose(
                float(convolved[i].sum()), 1.0, rtol=1e-5
            )

    def test_extended_psf_broadens_image(self, payload):
        """Convolution with an extended PSF should broaden the galaxy image."""
        # Create a mock PSF payload with small Gaussian PSFs
        n_y, n_x, n_wl = 2, 2, 3
        psf_size = 20
        oversample = 4

        # Build normalized Gaussian PSFs (sigma=2 oversampled pixels)
        y, x = jnp.mgrid[:psf_size, :psf_size]
        psf_center = (psf_size - 1) / 2.0
        gaussian_psf = jnp.exp(
            -((x - psf_center)**2 + (y - psf_center)**2) / (2 * 2.0**2)
        ).astype(jnp.float32)
        gaussian_psf = gaussian_psf / gaussian_psf.sum()

        psf_grid = jnp.broadcast_to(
            gaussian_psf, (n_y, n_x, n_wl, psf_size, psf_size)
        ).copy()

        extended_psf_payload = {
            'psf_grid': psf_grid,
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
            'wl_grid': jnp.linspace(1.0, 1.8, n_wl),
            'spatial_x': jnp.linspace(1, 4088, n_x),
            'spatial_y': jnp.linspace(1, 4088, n_y),
            'oversample': oversample,
            'detector': 'WFI05',
            'order': '1',
        }

        # Delta galaxy image
        image = jnp.zeros((12, 12), dtype=jnp.float32)
        image = image.at[6, 6].set(1.0)
        dx = dy = 1.0 / oversample

        convolved, _, _, _ = prepare_galaxy_images(
            payload, extended_psf_payload, image, 2000.0, 2000.0, dx, dy,
        )

        # Convolved image should be broader than a delta —
        # multiple pixels should have significant flux
        conv_slice = convolved[1]  # middle wavelength
        n_significant = int((jnp.abs(conv_slice) > 0.001 * jnp.abs(conv_slice).max()).sum())
        assert n_significant > 10, (
            f"Expected broadened image, got only {n_significant} significant pixels"
        )

        # Total flux should still be conserved
        np.testing.assert_allclose(float(conv_slice.sum()), 1.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test: Delta function galaxy vs star disperser
# ---------------------------------------------------------------------------

class TestDeltaFunctionGalaxy:
    """Single-pixel galaxy should match star disperser pixel-by-pixel."""

    def _run_star_and_galaxy(self, payload, mock_psf_payload, wavelengths):
        """Helper: run both dispersers and return output images."""
        from roman_disperser.star_disperser import disperse_star_psf

        xsca, ysca = 2000.0, 2000.0
        flux = jnp.ones(len(wavelengths))

        # Star dispersion
        output_star = jnp.zeros((4088, 4088), dtype=jnp.float32)
        output_star = disperse_star_psf(
            mock_psf_payload, payload,
            xsca_star=xsca, ysca_star=ysca,
            wavelengths=wavelengths, star_flux=flux,
            output=output_star, chunk_size=len(wavelengths),
        )

        # Galaxy dispersion with single-pixel image
        image = jnp.array([[1.0]], dtype=jnp.float32)
        output_gal = jnp.zeros((4088, 4088), dtype=jnp.float32)
        output_gal = disperse_galaxy(
            payload, mock_psf_payload,
            image, xsca, ysca,
            flux, wavelengths, output_gal,
            chunk_size=len(wavelengths),
        )

        return output_star, output_gal

    def test_delta_galaxy_matches_star_on_grid(self, payload, mock_psf_payload):
        """At PSF grid wavelengths, images should match pixel-by-pixel.

        Using grid wavelengths eliminates position interpolation differences
        between the two code paths, so the outputs should be nearly identical.
        """
        # Use a subset of exact PSF grid wavelengths
        grid_wl = mock_psf_payload['wavelengths']
        wavelengths = grid_wl[1:-1]  # skip edges to stay well within grid

        output_star, output_gal = self._run_star_and_galaxy(
            payload, mock_psf_payload, wavelengths,
        )

        # Both should have nonzero output
        assert float(output_star.sum()) > 0
        assert float(output_gal.sum()) > 0

        # Extract bounding box around nonzero region for clearer diagnostics
        mask = (output_star != 0) | (output_gal != 0)
        if mask.any():
            rows = jnp.where(mask.any(axis=1))[0]
            cols = jnp.where(mask.any(axis=0))[0]
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            star_crop = output_star[r0:r1, c0:c1]
            gal_crop = output_gal[r0:r1, c0:c1]
        else:
            star_crop = output_star
            gal_crop = output_gal

        # Pixel-level comparison
        np.testing.assert_allclose(gal_crop, star_crop, rtol=1e-4, atol=1e-7)

    def test_delta_galaxy_matches_star_off_grid(self, payload, mock_psf_payload):
        """At off-grid wavelengths, images should still match pixel-by-pixel.

        Both dispersers now compute exact center positions at every wavelength.
        The only difference is that the galaxy disperser interpolates convolved
        images from the PSF grid. With delta PSFs, the convolved images are
        themselves deltas, so interpolation is exact. Hence the outputs should
        match at float32 precision.
        """
        wavelengths = jnp.linspace(1.15, 1.65, 8)  # intentionally off-grid

        output_star, output_gal = self._run_star_and_galaxy(
            payload, mock_psf_payload, wavelengths,
        )

        # Total flux should match at float32 precision
        np.testing.assert_allclose(
            float(output_gal.sum()), float(output_star.sum()), rtol=1e-5
        )

        # Pixel-level comparison (same approach as on-grid test)
        mask = (output_star != 0) | (output_gal != 0)
        if mask.any():
            rows = jnp.where(mask.any(axis=1))[0]
            cols = jnp.where(mask.any(axis=0))[0]
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            star_crop = output_star[r0:r1, c0:c1]
            gal_crop = output_gal[r0:r1, c0:c1]
        else:
            star_crop = output_star
            gal_crop = output_gal

        np.testing.assert_allclose(gal_crop, star_crop, rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# Test: Flux conservation
# ---------------------------------------------------------------------------

class TestFluxConservation:
    """Test that flux is conserved through the disperser."""

    def test_centered_galaxy_conserves_flux(self, payload, mock_psf_payload):
        """Galaxy at detector center should conserve total flux."""
        image_size = 20

        # Small Gaussian galaxy (well within detector)
        y, x = jnp.mgrid[:image_size, :image_size]
        center = (image_size - 1) / 2.0
        image = jnp.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.0**2))
        image = image.astype(jnp.float32)

        wavelengths = jnp.array([1.3, 1.5])
        spectrum = jnp.array([1.0, 1.0])

        output = jnp.zeros((4088, 4088), dtype=jnp.float32)
        output = disperse_galaxy(
            payload, mock_psf_payload,
            image, 2000.0, 2000.0,
            spectrum, wavelengths, output,
            chunk_size=10,
        )

        expected_flux = float(image.sum()) * float(spectrum.sum())
        actual_flux = float(output.sum())

        # Gaussian sigma=2 in 20x20 image: flux at edges is negligible.
        # Delta PSFs sum to 1.0. No mechanism for significant flux loss.
        np.testing.assert_allclose(actual_flux, expected_flux, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test: Chunk invariance
# ---------------------------------------------------------------------------

class TestChunkInvariance:
    """Test that results are independent of chunk_size."""

    def test_chunk_size_invariance(self, payload, mock_psf_payload_small):
        """Result should be same regardless of chunk_size."""
        image = jnp.ones((8, 8), dtype=jnp.float32) * 0.1
        wavelengths = jnp.linspace(1.1, 1.7, 15)
        spectrum = jnp.ones(15)

        results = []
        for cs in [5, 8, 15]:
            output = jnp.zeros((4088, 4088), dtype=jnp.float32)
            result = disperse_galaxy(
                payload, mock_psf_payload_small,
                image, 2000.0, 2000.0,
                spectrum, wavelengths, output,
                chunk_size=cs,
            )
            results.append(result)

        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i], rtol=1e-4, atol=1e-6,
                err_msg=f"chunk_size={[5, 8, 15][i]} differs from chunk_size=5"
            )


# ---------------------------------------------------------------------------
# Test: JIT compilation via factory
# ---------------------------------------------------------------------------

class TestMakeGalaxyDisperser:
    """Test the galaxy disperser factory function."""

    def test_validates_even_oversample(self, payload, mock_psf_payload_small):
        """Should reject odd oversampling."""
        mock_psf_payload_small['oversample'] = 3
        with pytest.raises(ValueError, match="even oversampling"):
            make_galaxy_disperser(mock_psf_payload_small, payload)

    def test_returns_callable(self, payload, mock_psf_payload_small):
        """Factory should return a callable."""
        disperser = make_galaxy_disperser(mock_psf_payload_small, payload)
        assert callable(disperser)

    def test_jit_compilation(self, payload, mock_psf_payload_small):
        """Factory should return JIT-compiled function that works."""
        disperser = make_galaxy_disperser(
            mock_psf_payload_small, payload, chunk_size=10
        )

        image = jnp.ones((8, 8), dtype=jnp.float32)
        wavelengths = jnp.array([1.3, 1.5])
        spectrum = jnp.array([1.0, 1.0])
        output = jnp.zeros((4088, 4088), dtype=jnp.float32)

        # First call compiles
        result1 = disperser(image, 2000.0, 2000.0, spectrum, wavelengths, output)
        # Second call uses cache
        result2 = disperser(image, 2000.0, 2000.0, spectrum, wavelengths, output)

        np.testing.assert_allclose(result1, result2, rtol=RTOL, atol=ATOL)
        assert float(result1.sum()) > 0


# ---------------------------------------------------------------------------
# Test: Wavelength interpolation
# ---------------------------------------------------------------------------

class TestWavelengthInterpolation:
    """Test that convolved images interpolate correctly between grid points."""

    def test_grid_wavelength_exact(self, payload, mock_psf_payload):
        """Dispersion at a grid wavelength should use exact convolved image."""
        dx = dy = 1.0 / mock_psf_payload['oversample']
        image = jnp.ones((12, 12), dtype=jnp.float32)

        convolved, _, _, grid_wl = prepare_galaxy_images(
            payload, mock_psf_payload, image, 2000.0, 2000.0, dx, dy,
        )

        # Interpolate at exact grid wavelengths
        from roman_disperser.psf_model import interp_wavelength_chunk
        interp_at_grid = interp_wavelength_chunk(convolved, grid_wl, grid_wl)

        np.testing.assert_allclose(interp_at_grid, convolved, rtol=1e-4, atol=1e-12)

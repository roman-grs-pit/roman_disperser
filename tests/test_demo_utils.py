"""Tests for demo_utils module."""

import numpy as np
import pytest

from roman_disperser.demo_utils import center_to_corner


class TestCenterToCorner:
    """Test the center_to_corner helper function."""

    def test_basic_conversion_even(self):
        """Test basic center to corner conversion with even pixel count."""
        # 10x10 image with dx=dy=1, centered at (100, 100)
        # Center of image is at pixel index (10-1)/2 = 4.5
        # Offset = 4.5 * 1 = 4.5
        # Corner should be at (100 - 4.5, 100 - 4.5) = (95.5, 95.5)
        x0, y0 = center_to_corner(100.0, 100.0, 10, 10, 1.0, 1.0)
        assert x0 == 95.5
        assert y0 == 95.5

    def test_odd_pixel_count(self):
        """Test with odd pixel count - center should be at exact pixel."""
        # 11x11 image with dx=dy=1, centered at (100, 100)
        # Center of image is at pixel index (11-1)/2 = 5.0
        # Offset = 5.0 * 1 = 5.0
        # Corner should be at (100 - 5, 100 - 5) = (95, 95)
        x0, y0 = center_to_corner(100.0, 100.0, 11, 11, 1.0, 1.0)
        assert x0 == 95.0
        assert y0 == 95.0
        # Center pixel (index 5) should be at exactly 100.0
        center_pixel_pos = x0 + 5 * 1.0
        assert center_pixel_pos == 100.0

    def test_oversampled_image_even(self):
        """Test with oversampled image (dx, dy < 1), even pixel count."""
        # 150x150 oversampled image (50 native × 3), dx=dy=1/3
        # Centered at (2044, 2044)
        # Center of image is at pixel index (150-1)/2 = 74.5
        # Offset = 74.5 * (1/3) = 74.5/3 ≈ 24.833
        # Corner = 2044 - 24.833 ≈ 2019.167
        x0, y0 = center_to_corner(2044.0, 2044.0, 150, 150, 1 / 3, 1 / 3)
        expected = 2044.0 - 74.5 / 3
        np.testing.assert_allclose(x0, expected, rtol=1e-10)
        np.testing.assert_allclose(y0, expected, rtol=1e-10)

    def test_asymmetric_image(self):
        """Test with non-square image."""
        # 20x10 image with dx=0.5, dy=1.0
        # x offset = (20-1)/2 * 0.5 = 9.5 * 0.5 = 4.75
        # y offset = (10-1)/2 * 1.0 = 4.5 * 1.0 = 4.5
        x0, y0 = center_to_corner(500.0, 500.0, 20, 10, 0.5, 1.0)
        np.testing.assert_allclose(x0, 495.25)
        np.testing.assert_allclose(y0, 495.5)

    def test_vectorized_centers(self):
        """Test with array inputs (for multi-galaxy case)."""
        x_centers = np.array([1000.0, 2000.0, 3000.0])
        y_centers = np.array([1500.0, 2500.0, 3500.0])

        x0s, y0s = center_to_corner(x_centers, y_centers, 10, 10, 1.0, 1.0)

        expected_x0s = x_centers - 4.5
        expected_y0s = y_centers - 4.5
        np.testing.assert_allclose(x0s, expected_x0s)
        np.testing.assert_allclose(y0s, expected_y0s)

    def test_float_npix_converted_to_int(self):
        """Test that float pixel counts are handled correctly."""
        # Pass npix as floats - should work the same as ints
        x0_float, y0_float = center_to_corner(100.0, 100.0, 10.0, 10.0, 1.0, 1.0)
        x0_int, y0_int = center_to_corner(100.0, 100.0, 10, 10, 1.0, 1.0)
        assert x0_float == x0_int
        assert y0_float == y0_int

    def test_roundtrip_consistency(self):
        """Verify that center computed from corner matches original center."""
        # Given a center, compute corner, then verify center is correct
        x_center, y_center = 2044.0, 2044.0
        npix_x, npix_y = 150, 150
        dx, dy = 1 / 3, 1 / 3

        x0, y0 = center_to_corner(x_center, y_center, npix_x, npix_y, dx, dy)

        # The center of the image should be at:
        # x_center = x0 + (npix_x - 1) / 2 * dx
        computed_x_center = x0 + (npix_x - 1) / 2 * dx
        computed_y_center = y0 + (npix_y - 1) / 2 * dy

        np.testing.assert_allclose(computed_x_center, x_center, rtol=1e-10)
        np.testing.assert_allclose(computed_y_center, y_center, rtol=1e-10)

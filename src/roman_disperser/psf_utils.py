"""
PSF coordinate utilities for STPSF integration.

This module provides coordinate conversion between the disperser's SCA coordinate
system (1-indexed FITS) and STPSF's detector_position coordinate system (0-indexed).

⚠️ CRITICAL COORDINATE SYSTEM ASSUMPTIONS ⚠️
============================================

This module contains HARDCODED assumptions about the relationship between:
- STPSF detector coordinates (0-indexed, 4096×4096 pixels)
- Roman disperser SCA coordinates (1-indexed FITS, 4088×4088 usable region)

ASSUMPTION: The 4088×4088 usable science region is CENTERED within the 4096×4096
full H4RG-10 detector array, with a 4-pixel border on each side.

This means:
- STPSF pixels [4:4092] (0-indexed) correspond to SCA pixels [1:4088] (1-indexed)
- The conversion offset is 4 pixels

⚠️ THIS ASSUMPTION IS UNVERIFIED AND MAY BE INCORRECT ⚠️

TODO: Verify this assumption against:
- Roman technical documentation (reference pixel locations, CRPIX)
- STPSF source code (detector geometry definitions)
- Cross-validation with known source positions
- FITS headers from Roman data

If this assumption is wrong, the coordinate conversion functions below will
produce systematic offsets in PSF positioning, leading to incorrect dispersed
spectra. The error will be a fixed offset (likely 4-8 pixels) in x and/or y.

See docs/stpsf.md Section 18 for detailed discussion of this issue.
"""

import jax.numpy as jnp

# ============================================================================
# HARDCODED CONSTANTS (for JIT compatibility)
# ============================================================================

# Assumed offset for converting between SCA and STPSF coordinates
# Based on assumption that 4088 usable region is centered in 4096 array
_SCA_TO_STPSF_OFFSET = 4  # pixels

# ============================================================================
# COORDINATE CONVERSION FUNCTIONS (JAX-compatible, JIT-compilable)
# ============================================================================


def sca_to_stpsf_position(xsca, ysca):
    """
    Convert disperser SCA coordinates to STPSF detector_position.

    ⚠️ HARDCODED ASSUMPTIONS (for JIT compatibility):
    - 4088×4088 usable region is CENTERED in 4096×4096 array
    - 4-pixel border on each side (pixels [4:4092] in 0-indexed array)
    - This conversion is UNVERIFIED - tracked as open question

    This function is JAX-compatible and JIT-compilable:
    - Uses only jnp operations
    - No Python conditionals or type conversions
    - Returns floats (STPSF accepts float detector_position)

    Parameters
    ----------
    xsca, ysca : float or jnp.ndarray
        SCA coordinates from disperser optical model
        - Convention: 1-indexed FITS coordinates
        - Valid range: 1.0 to 4088.0 (usable science region)
        - Pixel centers at integer coordinates (e.g., pixel 1 at x=1.0)

    Returns
    -------
    x_stpsf, y_stpsf : float or jnp.ndarray
        STPSF detector_position coordinates
        - Convention: 0-indexed (Python/NumPy)
        - Approximate range: 4.0 to 4092.0 (if assumption correct)
        - Returns float, not int, for JAX compatibility

    Examples
    --------
    >>> # Detector center (approximate)
    >>> x_stpsf, y_stpsf = sca_to_stpsf_position(2044.0, 2044.0)
    >>> # Expected: ~(2048.0, 2048.0) if assumption correct

    >>> # Array input (vectorized)
    >>> import jax.numpy as jnp
    >>> xsca = jnp.array([1000.0, 2000.0, 3000.0])
    >>> ysca = jnp.array([1000.0, 2000.0, 3000.0])
    >>> x_stpsf, y_stpsf = sca_to_stpsf_position(xsca, ysca)

    Notes
    -----
    Round-trip conversion (SCA → STPSF → SCA) should be exact due to
    simple arithmetic (no rounding operations).

    See Also
    --------
    stpsf_to_sca_position : Inverse conversion
    """
    # Convert 1-indexed FITS to 0-indexed, then add assumed border offset
    # All operations are JAX-compatible (no int(), round(), or conditionals)
    x_stpsf = xsca - 1.0 + _SCA_TO_STPSF_OFFSET
    y_stpsf = ysca - 1.0 + _SCA_TO_STPSF_OFFSET

    return x_stpsf, y_stpsf


def stpsf_to_sca_position(x_stpsf, y_stpsf):
    """
    Convert STPSF detector_position to disperser SCA coordinates.

    ⚠️ HARDCODED ASSUMPTIONS (for JIT compatibility):
    Inverse of sca_to_stpsf_position(). Same assumptions apply.

    This function is JAX-compatible and JIT-compilable.

    Parameters
    ----------
    x_stpsf, y_stpsf : float or jnp.ndarray
        STPSF detector_position coordinates
        - Convention: 0-indexed (Python/NumPy)
        - Typical range: 0.0 to 4095.0 (full detector)

    Returns
    -------
    xsca, ysca : float or jnp.ndarray
        SCA coordinates for disperser optical model
        - Convention: 1-indexed FITS coordinates
        - Expected range: 1.0 to 4088.0 (usable region)
        - May be outside valid range if input is near detector edges

    Examples
    --------
    >>> # Convert STPSF detector center to SCA
    >>> xsca, ysca = stpsf_to_sca_position(2048.0, 2048.0)
    >>> # Expected: ~(2044.0, 2044.0) if assumption correct

    >>> # Round-trip test
    >>> x_orig, y_orig = 2044.0, 2044.0
    >>> x_stpsf, y_stpsf = sca_to_stpsf_position(x_orig, y_orig)
    >>> x_back, y_back = stpsf_to_sca_position(x_stpsf, y_stpsf)
    >>> assert abs(x_back - x_orig) < 1e-10  # Should be exact

    See Also
    --------
    sca_to_stpsf_position : Forward conversion
    """
    # Remove assumed border offset, then convert 0-indexed to 1-indexed FITS
    xsca = x_stpsf - _SCA_TO_STPSF_OFFSET + 1.0
    ysca = y_stpsf - _SCA_TO_STPSF_OFFSET + 1.0

    return xsca, ysca

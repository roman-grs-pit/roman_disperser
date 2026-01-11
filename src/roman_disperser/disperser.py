"""
Disperser module for Roman grism simulation.

This module provides functions to disperse 2D spatial images combined with
1D spectra onto a detector, simulating grism spectroscopy observations.

The key insight is that the optical model has separable structure:
- Spatial-only quantities (FPA coords, reference positions, trace coefficients)
  are computed once per spatial pixel
- Wavelength-dependent quantities (dispersion offsets) are computed via
  broadcasting over the wavelength dimension

This allows memory-efficient processing by chunking over wavelengths rather
than materializing the full 3D (Ny × Nx × Nlam) grid.

See docs/disperser_design.md for the full design document.
"""

import jax
import jax.numpy as jnp

from . import optical_model_jax as omj


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def bilinear_scatter_add(output, x, y, values):
    """
    Accumulate values onto output grid using bilinear interpolation.

    For each (x, y) position, distributes the corresponding value to the
    four nearest grid points with weights based on distance.

    Args:
        output: [H, W] array to accumulate onto
        x: [N] x-coordinates (column indices, can be fractional)
        y: [N] y-coordinates (row indices, can be fractional)
        values: [N] values to scatter

    Returns:
        output: [H, W] updated array with accumulated values

    Notes:
        - Out-of-bounds points are silently ignored (no contribution)
        - Uses JAX's .at[].add() for functional scatter-add
        - The four corner weights always sum to 1.0 for valid points
    """
    # Get integer indices (floor)
    x_floor = jnp.floor(x).astype(jnp.int32)
    y_floor = jnp.floor(y).astype(jnp.int32)

    # Fractional parts for interpolation weights
    fx = x - x_floor
    fy = y - y_floor

    # Bilinear weights for four corners
    # (x_floor, y_floor) is bottom-left in image coordinates
    w00 = (1 - fx) * (1 - fy)  # bottom-left
    w10 = fx * (1 - fy)  # bottom-right
    w01 = (1 - fx) * fy  # top-left
    w11 = fx * fy  # top-right

    # Bounds check: need x_floor+1 < width and y_floor+1 < height
    # For 4088×4088 detector, valid range is [0, 4086] for floor indices
    H, W = output.shape
    valid = (x_floor >= 0) & (x_floor < W - 1) & (y_floor >= 0) & (y_floor < H - 1)

    # Zero out contributions from invalid points
    v00 = values * w00 * valid
    v10 = values * w10 * valid
    v01 = values * w01 * valid
    v11 = values * w11 * valid

    # Scatter-add to four corners (JAX functional update)
    output = output.at[y_floor, x_floor].add(v00)
    output = output.at[y_floor, x_floor + 1].add(v10)
    output = output.at[y_floor + 1, x_floor].add(v01)
    output = output.at[y_floor + 1, x_floor + 1].add(v11)

    return output


def _transform_wavelength(wl, payload):
    """
    Apply wavelength transformation for polynomial evaluation.

    Args:
        wl: wavelengths in microns [N_wl]
        payload: dict from make_sca_payload containing wl transform info

    Returns:
        transformed wavelengths (linear: wl - wl_ref, log: log10(wl/wl_ref))
    """
    wl_ref = payload["wl"]["reference"]
    wl_transform = payload["wl"]["transform"]

    if wl_transform == "linear":
        return wl - wl_ref
    elif wl_transform == "log":
        return jnp.log10(wl / wl_ref)
    else:
        raise ValueError(f"Invalid wavelength transform: {wl_transform}")


def _eval_ids_polynomial(wl_transformed, ids):
    """
    Evaluate inverse dispersion polynomial with broadcasting.

    Computes dely = sum_i(ids[i] * wl^i) for each spatial point and wavelength.

    Args:
        wl_transformed: transformed wavelengths [N_wl]
        ids: inverse dispersion coefficients [i, N_spatial]

    Returns:
        dely: y-displacement in MPA coords [N_spatial, N_wl]

    Notes:
        Uses einsum for efficient batched polynomial evaluation:
        - wl_powers[w, i] = wl[w]^i for i in range(n_coeffs)
        - einsum('wi,is->sw') computes the polynomial for all (spatial, wl) pairs
    """
    n_coeffs = ids.shape[0]

    # Build Vandermonde-like matrix: wl_powers[N_wl, n_coeffs]
    wl_powers = wl_transformed[:, jnp.newaxis] ** jnp.arange(n_coeffs)

    # Polynomial evaluation via einsum
    # [N_wl, i] × [i, N_spatial] → [N_spatial, N_wl]
    return jnp.einsum("wi,is->sw", wl_powers, ids)


def _eval_crv_polynomial(dely, crv):
    """
    Evaluate curvature polynomial with broadcasting.

    Computes delx = sum_i(crv[i] * dely^i) for each spatial point and wavelength.

    Args:
        dely: y-displacement values [N_spatial, N_wl]
        crv: curvature coefficients [i, N_spatial]

    Returns:
        delx: x-displacement in MPA coords [N_spatial, N_wl]

    Notes:
        Similar to _eval_ids_polynomial but operates on dely values
        which already have shape [N_spatial, N_wl].
    """
    n_coeffs = crv.shape[0]

    # dely_powers[N_spatial, N_wl, i] = dely^i
    dely_powers = dely[:, :, jnp.newaxis] ** jnp.arange(n_coeffs)

    # Polynomial evaluation via einsum
    # [N_spatial, N_wl, i] × [i, N_spatial] → [N_spatial, N_wl]
    # Need to be careful with indices - crv[i, s] applies to spatial point s
    return jnp.einsum("swi,is->sw", dely_powers, crv)


# -----------------------------------------------------------------------------
# Main Disperser Functions
# -----------------------------------------------------------------------------


def disperse_2d1d_sca(
    payload,
    image,
    x0,
    y0,
    dx,
    dy,
    spec,
    lam0,
    dlam,
    output,
    wavelength_chunk_size=100,
):
    """
    Disperse a 2D spatial image × 1D spectrum onto a detector.

    This function simulates grism spectroscopy by tracing each spatial pixel
    through the optical model at each wavelength, and accumulating the flux
    onto the output detector using bilinear interpolation.

    The implementation exploits separability: spatial-only quantities are
    computed once, then wavelength-dependent offsets are computed in chunks
    to limit memory usage.

    Args:
        payload: dict from make_sca_payload() containing optical model params
        image: [Ny, Nx] float32 spatial image (flux per pixel)
        x0, y0: SCA origin of image grid (pixels, can be fractional)
        dx, dy: Pixel spacing (1.0 for native, <1.0 for oversampled input)
        spec: [Nlam] float32 spectrum (flux per wavelength bin)
        lam0: Starting wavelength (microns)
        dlam: Wavelength spacing (microns)
        output: [4088, 4088] accumulator array (modified functionally)
        wavelength_chunk_size: Number of wavelengths to process at once
            (controls memory usage vs. loop overhead tradeoff)

    Returns:
        output: [4088, 4088] updated array with dispersed flux

    Notes:
        - The total flux deposited equals sum(image) * sum(spec) for points
          that land within the detector bounds
        - Out-of-bounds flux is silently discarded
        - For JIT compilation, use static_argnums for wavelength_chunk_size

    Example:
        >>> payload = omj.make_sca_payload(model, sca=5, order="1")
        >>> image = jnp.ones((10, 10))  # 10×10 uniform source
        >>> spec = jnp.ones(100)  # Flat spectrum, 100 wavelengths
        >>> output = jnp.zeros((4088, 4088))
        >>> output = disperse_2d1d_sca(
        ...     payload, image, x0=2000, y0=2000, dx=1.0, dy=1.0,
        ...     spec=spec, lam0=1.0, dlam=0.01, output=output
        ... )
    """
    Ny, Nx = image.shape
    Nlam = spec.shape[0]
    N_spatial = Ny * Nx

    # Ensure chunk size doesn't exceed Nlam (for dynamic_slice compatibility)
    actual_chunk_size = min(wavelength_chunk_size, Nlam)

    # Pad spectrum to multiple of chunk size for clean slicing
    # This avoids issues with dynamic_slice when Nlam < chunk_size
    n_chunks = (Nlam + actual_chunk_size - 1) // actual_chunk_size
    padded_len = n_chunks * actual_chunk_size
    spec_padded = jnp.pad(spec, (0, padded_len - Nlam), mode="constant")

    # -------------------------------------------------------------------------
    # Step 1: Precompute spatial-only quantities (done once)
    # -------------------------------------------------------------------------

    # Build 2D spatial grid and flatten
    iy, ix = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx), indexing="ij")
    xsca = (x0 + ix * dx).ravel()  # [N_spatial]
    ysca = (y0 + iy * dy).ravel()  # [N_spatial]
    image_flat = image.ravel()  # [N_spatial]

    # Convert SCA → FPA (once per spatial pixel)
    xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)  # [N_spatial]

    # Get reference MPA positions (at wl_reference)
    xmpa_ref, ympa_ref = omj.get_mpa_coords(payload, xfpa, yfpa)  # [N_spatial]

    # Get trace coefficients for dispersion calculation
    crv, ids = omj.get_trace_coeffs(payload, xfpa, yfpa)  # [i, N_spatial]

    # -------------------------------------------------------------------------
    # Step 2: Process wavelength chunks
    # -------------------------------------------------------------------------

    def process_wavelength_chunk(output, chunk_idx):
        """Process a single wavelength chunk."""
        # Compute wavelength range for this chunk
        lam_start = chunk_idx * actual_chunk_size

        # Wavelengths for this chunk (in microns)
        wl_indices = jnp.arange(actual_chunk_size)
        wl_chunk = lam0 + (lam_start + wl_indices) * dlam  # [chunk_size]

        # Transform wavelengths for polynomial evaluation
        wl_transformed = _transform_wavelength(wl_chunk, payload)

        # Compute y-displacement (inverse dispersion)
        dely = _eval_ids_polynomial(wl_transformed, ids)  # [N_spatial, chunk_size]

        # Compute x-displacement (curvature)
        delx = _eval_crv_polynomial(dely, crv)  # [N_spatial, chunk_size]

        # Final MPA positions
        xmpa_out = xmpa_ref[:, jnp.newaxis] + delx  # [N_spatial, chunk_size]
        ympa_out = ympa_ref[:, jnp.newaxis] + dely  # [N_spatial, chunk_size]

        # Convert MPA → SCA for output detector coordinates
        xsca_out, ysca_out = omj.mpa_to_sca(payload, xmpa_out, ympa_out)

        # Get spectrum chunk (from padded spectrum)
        spec_chunk = jax.lax.dynamic_slice(
            spec_padded, (lam_start,), (actual_chunk_size,)
        )
        flux = image_flat[:, jnp.newaxis] * spec_chunk  # [N_spatial, chunk_size]

        # Padded values are already zero, so no additional masking needed

        # Flatten and scatter-add with bilinear interpolation
        output = bilinear_scatter_add(
            output, xsca_out.ravel(), ysca_out.ravel(), flux.ravel()
        )

        return output, None

    # Process all chunks using scan (JIT-compatible loop)
    output, _ = jax.lax.scan(process_wavelength_chunk, output, jnp.arange(n_chunks))

    return output

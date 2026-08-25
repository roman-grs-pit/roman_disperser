"""
Galaxy dispersion module for Roman grism simulation.

This module extends the star dispersion routines to handle extended sources
(galaxies). The key difference from stars is that the galaxy's spatial extent
must be warped through the dispersion Jacobian before PSF convolution.

Key functions:
- trace_beam_sca: SCA -> FPA -> trace_beam -> MPA -> SCA (scalar inputs)
- trace_beam_sca_with_jacobian: Position + 2x2 Jacobian via autodiff
- disperse_galaxy_shape: Warp galaxy image through Jacobian (forward scatter)
- prepare_galaxy_images: Warp + PSF convolve at all PSF grid wavelengths
- disperse_galaxy: Full galaxy dispersion with wavelength chunking
- make_galaxy_disperser: Factory for JIT-compiled galaxy dispersion

All wavelength parameters are in **microns** (consistent with the optical model).

Approximations and what is exact:
- **Center positions** are computed exactly at every fine wavelength via
  ``trace_beam`` (same as the star disperser). No interpolation.
- **Convolved galaxy images** are computed at PSF grid wavelengths (~56 points)
  and linearly interpolated to the fine wavelength grid. This is the only
  interpolation approximation, and it is accurate because the PSF varies
  smoothly with wavelength.
- **Jacobian warping** uses a single Jacobian evaluated at the galaxy center.
  The linear approximation error is <0.02 oversampled pixels for galaxies
  up to 25 native pixels in radius (see notebooks/galaxy/jacobian_exploration).

See docs/galaxy_dispersion_plan.md for the design document.
"""

import jax
import jax.numpy as jnp
import jax.scipy.signal

from . import optical_model_jax as omj
from . import psf_model
from . import star_disperser


def trace_beam_sca(payload, xsca, ysca, wavelength):
    """Trace position through optical model: SCA -> FPA -> MPA -> SCA.

    Accepts scalar inputs (wraps to 1D arrays for optical model functions).

    Parameters
    ----------
    payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    xsca, ysca : float
        Position in SCA coordinates (1-indexed FITS)
    wavelength : float
        Wavelength in microns

    Returns
    -------
    x_out, y_out : float
        Dispersed SCA position (scalars)
    """
    xsca_a = jnp.atleast_1d(xsca)
    ysca_a = jnp.atleast_1d(ysca)
    wl_a = jnp.atleast_1d(wavelength)
    xfpa, yfpa = omj.sca_to_fpa(payload, xsca_a, ysca_a)
    xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wl_a)
    x_out, y_out = omj.mpa_to_sca(payload, xmpa, ympa)
    return x_out[0], y_out[0]


def trace_beam_sca_with_jacobian(payload, xsca, ysca, wavelength):
    """Compute dispersed position AND 2x2 Jacobian.

    Uses JAX autodiff to compute the Jacobian of the SCA->SCA trace
    with respect to (xsca, ysca) at fixed wavelength.

    Parameters
    ----------
    payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    xsca, ysca : float
        Position in SCA coordinates (1-indexed FITS)
    wavelength : float
        Wavelength in microns

    Returns
    -------
    x_out, y_out : float
        Dispersed SCA position (scalars)
    J : jnp.ndarray
        2x2 Jacobian matrix [[dx'/dx, dx'/dy], [dy'/dx, dy'/dy]]
    """
    # Forward pass for position
    x_out, y_out = trace_beam_sca(payload, xsca, ysca, wavelength)

    # Jacobian via autodiff
    # Note: this recomputes the forward pass internally. jax.jacobian doesn't
    # expose the primal output, so the explicit call above is needed.
    # The trace is cheap (~0.4 ms) so the redundancy is acceptable.
    jac_raw = jax.jacobian(trace_beam_sca, argnums=(1, 2))(
        payload, xsca, ysca, wavelength
    )
    # jac_raw = ((dx'/dx, dx'/dy), (dy'/dx, dy'/dy))
    J = jnp.array([[jac_raw[0][0], jac_raw[0][1]],
                    [jac_raw[1][0], jac_raw[1][1]]])

    return x_out, y_out, J


def disperse_galaxy_shape(image, jacobian, dx, dy):
    """Warp galaxy image through Jacobian using bilinear forward scatter.

    Forward scatter is flux-conserving by construction: each input pixel's
    flux is distributed to the 4 nearest output pixels via bilinear weights.

    Parameters
    ----------
    image : jnp.ndarray
        [Ny, Nx] galaxy image (oversampled)
    jacobian : jnp.ndarray
        [2, 2] Jacobian matrix
    dx, dy : float
        Pixel spacing in SCA coords (1/oversample)

    Returns
    -------
    warped : jnp.ndarray
        [Ny, Nx] warped image (same shape as input)

    Notes
    -----
    The output array has the same shape as the input. Since J ~ I
    (||J-I|| ~ 0.03), a few edge pixels may map outside the output and
    their flux is dropped. For a 150-pixel galaxy with J[1,1]=1.02, this
    affects ~1-2 edge rows. For realistic galaxy profiles (flux concentrated
    toward center), the loss is negligible. Callers should ensure the input
    image has a few pixels of padding beyond the region of significant flux.
    """
    Ny, Nx = image.shape

    # Build relative coordinate grid in SCA units
    j_idx = jnp.arange(Nx)
    i_idx = jnp.arange(Ny)
    ii, jj = jnp.meshgrid(i_idx, j_idx, indexing='ij')

    # Relative coords from image center
    rel_x = (jj - (Nx - 1) / 2.0) * dx  # SCA units
    rel_y = (ii - (Ny - 1) / 2.0) * dy

    # Apply Jacobian: [warped_x, warped_y] = J @ [rel_x, rel_y]
    warped_x = jacobian[0, 0] * rel_x + jacobian[0, 1] * rel_y
    warped_y = jacobian[1, 0] * rel_x + jacobian[1, 1] * rel_y

    # Convert back to fractional pixel indices
    out_j = warped_x / dx + (Nx - 1) / 2.0
    out_i = warped_y / dy + (Ny - 1) / 2.0

    # Bilinear forward scatter
    out_j_floor = jnp.floor(out_j).astype(jnp.int32)
    out_i_floor = jnp.floor(out_i).astype(jnp.int32)
    fj = out_j - out_j_floor
    fi = out_i - out_i_floor

    w00 = (1 - fj) * (1 - fi)
    w10 = fj * (1 - fi)
    w01 = (1 - fj) * fi
    w11 = fj * fi

    flux = image.ravel()
    idx_j = out_j_floor.ravel()
    idx_i = out_i_floor.ravel()

    kw = dict(mode="drop", wrap_negative_indices=False)
    warped = jnp.zeros_like(image)
    warped = warped.at[idx_i, idx_j].add(flux * w00.ravel(), **kw)
    warped = warped.at[idx_i, idx_j + 1].add(flux * w10.ravel(), **kw)
    warped = warped.at[idx_i + 1, idx_j].add(flux * w01.ravel(), **kw)
    warped = warped.at[idx_i + 1, idx_j + 1].add(flux * w11.ravel(), **kw)

    return warped


def _next_fast_fft_size(n):
    """Smallest integer >= n whose prime factors are all in {2, 3, 5, 7}.

    FFT backends (cuFFT on GPU, pocketfft on CPU) only have fast kernels
    for small prime radices; a size with a large prime factor falls back to
    the much slower Bluestein algorithm. Chosen by rule rather than
    hard-coding a size because the best padding is hardware/cuFFT-version
    dependent (315 beat 320 by ~4% on an A10G but that margin is not
    portable). Host-side Python ints — runs at trace time, so the choice is
    baked into the compiled function.
    """
    m = int(n)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def fft_convolve_full(images, kernels):
    """Full 2D linear convolution via FFT at a fast transform size.

    Equivalent to ``jax.scipy.signal.fftconvolve(images, kernels,
    mode='full')`` applied over the last two axes (leading axes broadcast),
    up to float rounding: a full linear convolution of sizes (Ny, Py) has
    support exactly X = Ny + Py - 1, so a circular convolution at any FFT
    size S >= X, cropped to [:X, :X], is the same answer.

    We transform at the next fast FFT size rather than at X itself: at
    production geometry X = 120 + 184 - 1 = 303 = 3*101, and the prime
    factor 101 forces cuFFT onto its slow Bluestein path — transforming at
    S = 315 = 3^2*5*7 measured ~1.7x faster for the whole convolution on an
    A10G, with max |diff| ~6e-9 vs fftconvolve (workbench 2026-08-19,
    SLURM 7145). ``jax.scipy.signal.fftconvolve`` (jax 0.7.2) transforms at
    the literal full shape (its source carries a TODO(jakevdp) about
    next_fast_len), so we do the padding by hand.

    Parameters
    ----------
    images, kernels : jnp.ndarray
        [..., Ny, Nx] and [..., Py, Px]; leading axes must broadcast.

    Returns
    -------
    jnp.ndarray
        [..., Ny + Py - 1, Nx + Px - 1] full linear convolution.
    """
    ny, nx = images.shape[-2:]
    py, px = kernels.shape[-2:]
    conv_y, conv_x = ny + py - 1, nx + px - 1
    s_y, s_x = _next_fast_fft_size(conv_y), _next_fast_fft_size(conv_x)
    spectrum = (jnp.fft.rfft2(images, s=(s_y, s_x))
                * jnp.fft.rfft2(kernels, s=(s_y, s_x)))
    return jnp.fft.irfft2(spectrum, s=(s_y, s_x))[..., :conv_y, :conv_x]


def prepare_galaxy_images(optical_payload, psf_payload, image, x0, y0, dx, dy):
    """Prepare dispersed, PSF-convolved galaxy images at PSF grid wavelengths.

    For each PSF grid wavelength:
    1. Compute Jacobian and dispersed center position
    2. Warp galaxy shape through Jacobian
    3. Convolve with wavelength-dependent PSF

    Parameters
    ----------
    optical_payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    psf_payload : dict
        PSF payload from psf_model.get_or_make_psf_payload()
    image : jnp.ndarray
        [Ny, Nx] galaxy image (at psf_payload['oversample']x oversampling)
    x0, y0 : float
        Undispersed center position in SCA coordinates (1-indexed FITS)
    dx, dy : float
        Pixel spacing in SCA coords (1/oversample)

    Returns
    -------
    convolved : jnp.ndarray
        [N_wl, Conv_y, Conv_x] convolved galaxy images
        where Conv = image_dim + PSF_dim - 1
    centers_x, centers_y : jnp.ndarray
        [N_wl] dispersed center positions in SCA coordinates
    wavelengths : jnp.ndarray
        [N_wl] PSF grid wavelengths in microns
    """
    # Get PSFs at undispersed position for all grid wavelengths
    psfs = psf_model.interpolate_psf_spatial(
        psf_payload, x0, y0
    )  # [N_wl, PSF_y, PSF_x]

    grid_wl = psf_payload['wavelengths']

    # Compute Jacobians and dispersed positions at all grid wavelengths
    def compute_jac_and_pos(wavelength):
        x_out, y_out, J = trace_beam_sca_with_jacobian(
            optical_payload, x0, y0, wavelength
        )
        return x_out, y_out, J

    # vmap over wavelengths
    centers_x, centers_y, jacobians = jax.vmap(
        compute_jac_and_pos
    )(grid_wl)  # [N_wl], [N_wl], [N_wl, 2, 2]

    # Warp galaxy at each wavelength
    def warp_at_wl(jacobian):
        return disperse_galaxy_shape(image, jacobian, dx, dy)

    warped_images = jax.vmap(warp_at_wl)(jacobians)  # [N_wl, Ny, Nx]

    # Convolve each warped image with its PSF (batched over wavelength).
    convolved = fft_convolve_full(warped_images, psfs)
    # [N_wl, Ny + PSF_y - 1, Nx + PSF_x - 1]

    return convolved, centers_x, centers_y, grid_wl


def disperse_galaxy(
    optical_payload,
    psf_payload,
    image,
    x0,
    y0,
    spectrum,
    wavelengths,
    output,
    chunk_size=500,
):
    """Disperse a galaxy onto the detector.

    This function:
    1. Warps the galaxy shape through the dispersion Jacobian at each PSF
       grid wavelength
    2. Convolves with wavelength-dependent PSFs
    3. Interpolates convolved images to the fine wavelength grid
    4. Deposits flux onto the detector using wavelength chunking

    Parameters
    ----------
    optical_payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    psf_payload : dict
        PSF payload from psf_model.get_or_make_psf_payload()
    image : jnp.ndarray
        [Ny, Nx] galaxy image at psf_payload['oversample']x oversampling
    x0, y0 : float
        Undispersed center position in SCA coordinates (1-indexed FITS)
    spectrum : jnp.ndarray
        [Nlam] flux values at each wavelength
    wavelengths : jnp.ndarray
        [Nlam] wavelength array in microns
    output : jnp.ndarray
        [H, W] detector accumulator (typically 4088x4088)
    chunk_size : int, optional
        Wavelengths per chunk (default: 500)

    Returns
    -------
    output : jnp.ndarray
        [H, W] updated detector array with dispersed galaxy
    """
    oversample = psf_payload['oversample']
    dx = dy = 1.0 / oversample

    # Step 1: Prepare convolved images at PSF grid wavelengths
    convolved, _, _, grid_wl = prepare_galaxy_images(
        optical_payload, psf_payload, image, x0, y0, dx, dy
    )  # [N_grid_wl, Conv_y, Conv_x]

    # Step 2: Build relative coordinate grid for convolved image
    conv_shape = convolved.shape[1:]  # (Conv_y, Conv_x)
    rel_y, rel_x = star_disperser.make_psf_pixel_grid(conv_shape, oversample)
    rel_x_flat = rel_x.ravel()
    rel_y_flat = rel_y.ravel()

    # Step 3: Compute exact dispersed positions at every fine wavelength
    # (cheap vectorized trace_beam call — no interpolation needed here)
    xsca_disp, ysca_disp = star_disperser._compute_dispersed_positions(
        optical_payload, x0, y0, wavelengths
    )

    # Step 4: Pad inputs to chunk_size multiples
    n_wl = len(wavelengths)
    n_padded = ((n_wl + chunk_size - 1) // chunk_size) * chunk_size
    pad_size = n_padded - n_wl
    n_chunks = n_padded // chunk_size

    wavelengths_padded = jnp.pad(
        wavelengths, (0, pad_size), constant_values=wavelengths[-1]
    )
    spectrum_padded = jnp.pad(
        spectrum, (0, pad_size), constant_values=0.0
    )
    x_disp_padded = jnp.pad(
        xsca_disp, (0, pad_size), constant_values=xsca_disp[-1]
    )
    y_disp_padded = jnp.pad(
        ysca_disp, (0, pad_size), constant_values=ysca_disp[-1]
    )

    # Step 5: Process wavelengths in chunks using lax.scan
    def process_chunk(output, chunk_idx):
        start = chunk_idx * chunk_size

        wl_chunk = jax.lax.dynamic_slice(
            wavelengths_padded, [start], [chunk_size]
        )
        flux_chunk = jax.lax.dynamic_slice(
            spectrum_padded, [start], [chunk_size]
        )
        x_chunk = jax.lax.dynamic_slice(
            x_disp_padded, [start], [chunk_size]
        )
        y_chunk = jax.lax.dynamic_slice(
            y_disp_padded, [start], [chunk_size]
        )

        # Interpolate convolved images to chunk wavelengths
        conv_chunk = psf_model.interp_wavelength_chunk(
            convolved, grid_wl, wl_chunk
        )  # [chunk_size, Conv_y, Conv_x]

        # Scale by spectrum flux
        conv_chunk = conv_chunk * flux_chunk[:, None, None]

        # Build detector coordinates for all pixels in all chunk wavelengths
        det_x = x_chunk[:, None] + rel_x_flat[None, :]  # [chunk, n_conv]
        det_y = y_chunk[:, None] + rel_y_flat[None, :]

        # Convert FITS 1-indexed to 0-indexed array indices
        idx_x = jnp.floor(det_x - 0.5).astype(jnp.int32).ravel()
        idx_y = jnp.floor(det_y - 0.5).astype(jnp.int32).ravel()
        values = conv_chunk.ravel()

        # Scatter-add to output
        output = output.at[idx_y, idx_x].add(
            values, mode="drop", wrap_negative_indices=False
        )
        return output, None

    output, _ = jax.lax.scan(process_chunk, output, jnp.arange(n_chunks))

    return output


def make_galaxy_disperser(psf_payload, optical_payload, chunk_size=500):
    """Create a JIT-compiled galaxy disperser with payloads captured in closure.

    Parameters
    ----------
    psf_payload : dict
        PSF payload from psf_model.get_or_make_psf_payload()
        Must use even oversampling (e.g., 4x)
    optical_payload : dict
        Optical model payload from optical_model_jax.make_sca_payload()
    chunk_size : int, optional
        Wavelengths per chunk (default: 500)

    Returns
    -------
    disperse_fn : Callable
        JIT-compiled function with signature:
        (image, x0, y0, spectrum, wavelengths, output) -> output

    Raises
    ------
    ValueError
        If psf_payload uses odd oversampling
    """
    oversample = psf_payload['oversample']
    if oversample % 2 != 0:
        raise ValueError(
            f"PSF payload must use even oversampling for correct PSF center "
            f"geometry, got {oversample}. Even oversampling (e.g., 2, 4) "
            f"places the PSF center at the cross-hairs of central pixels."
        )

    @jax.jit
    def disperse_fn(image, x0, y0, spectrum, wavelengths, output):
        return disperse_galaxy(
            optical_payload,
            psf_payload,
            image,
            x0,
            y0,
            spectrum,
            wavelengths,
            output,
            chunk_size,
        )

    return disperse_fn

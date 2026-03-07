"""
JAX functional optical model utilities.

Notes:
- This JAX implementation follows the modern path of the class-based model
    and does not implement the legacy `old_format` behavior. Specifically,
    coefficients are evaluated in FPA space and offsets are not divided by
    `plate_scale`. If `old_format` parity is needed, we can add a payload flag
    and mirror the class’s alternate code path.
- Function naming: the polynomial mapping of FPA→MPA coordinates is exposed
    as `get_mpa_coords` (renamed from `get_map_coords` for clarity).
"""

import jax
import jax.numpy as jnp


def make_sca_payload(model, sca: int, order: str = "1"):
    """
    Extract minimal per-SCA payload from RomanOpticalModel.
    Returns a dict of JAX arrays suitable for jit compilation.
    
    Args:
        model: RomanOpticalModel instance
        sca: SCA number (int)
        order: spectral order as string (e.g., "1", "0", "2", "m1")
    
    Returns:
        dict with keys: wl, det, poly
    """
    if sca not in model.sca_list:
        raise ValueError(f"SCA {sca} not in {model.sca_list.tolist()}")
    if order not in model.beam_coeffs:
        raise ValueError(f"Order {order!r} not in {list(model.beam_coeffs.keys())}")

    xcen, ycen = model.coords.get_sca_center(sca=sca)

    return {
        "wl": {
            "transform": str(model.wl_transform).lower(),
            "reference": jnp.asarray(model.wl_reference, dtype=jnp.float32),
            "min": jnp.asarray(model.wl_min, dtype=jnp.float32),
            "max": jnp.asarray(model.wl_max, dtype=jnp.float32),
        },
        "det": {
            "naxis1": int(model.detmod["naxis1"]),
            "naxis2": int(model.detmod["naxis2"]),
            "crpix1": jnp.asarray(model.detmod["crpix1"], dtype=jnp.float32),
            "crpix2": jnp.asarray(model.detmod["crpix2"], dtype=jnp.float32),
            "pixel_scale": jnp.asarray(model.detmod["pixel_scale"], dtype=jnp.float32),
            "plate_scale": jnp.asarray(model.detmod["plate_scale"], dtype=jnp.float32),
            "xy_center": jnp.asarray([xcen, ycen], dtype=jnp.float32),
        },
        "poly": {
            # dimen_map
            "map_i": int(model.dimen_map["i"]),
            "map_j": int(model.dimen_map["j"]),
            # dimen_crv
            "crv_i": int(model.dimen_crv["i"]),
            "crv_j": int(model.dimen_crv["j"]),
            "crv_k": int(model.dimen_crv["k"]),
            # dimen_ids
            "ids_i": int(model.dimen_ids["i"]),
            "ids_j": int(model.dimen_ids["j"]),
            "ids_k": int(model.dimen_ids["k"]),
            # Coefficients
            "X_ij": jnp.asarray(model.beam_coeffs[order]["X_ij"], dtype=jnp.float32),
            "Y_ij": jnp.asarray(model.beam_coeffs[order]["Y_ij"], dtype=jnp.float32),
            "C_ijk": jnp.asarray(model.beam_coeffs[order]["C_ijk"], dtype=jnp.float32),
            "D_ijk": jnp.asarray(model.beam_coeffs[order]["D_ijk"], dtype=jnp.float32),
        },
    }


# -------- coordinate transforms --------


def sca_to_mpa(payload, xsca, ysca):
    """
    Convert SCA pixel coordinates to MPA (mm).
    
    Args:
        payload: dict from make_sca_payload
        xsca, ysca: detector pixel coordinates
        
    TODO: Verify SCA coordinate range. Currently assumes [0.5, naxis+0.5].
    
    Returns:
        xmpa, ympa: position in mm
    """
    crpix1 = payload["det"]["crpix1"]
    crpix2 = payload["det"]["crpix2"]
    plate_scale = payload["det"]["plate_scale"]
    xcen, ycen = payload["det"]["xy_center"]
    
    dx = (xsca - crpix1) * -1.0
    dy = ysca - crpix2
    
    xoff = dx / plate_scale
    yoff = dy / plate_scale
    
    xmpa = xoff + xcen
    ympa = yoff + ycen
    
    return xmpa, ympa


def mpa_to_sca(payload, xmpa, ympa):
    """
    Convert MPA (mm) coordinates to SCA pixel coordinates.
    
    Args:
        payload: dict from make_sca_payload
        xmpa, ympa: position in mm
        
    TODO: Verify SCA coordinate range. Currently assumes [0.5, naxis+0.5].
    
    Returns:
        xsca, ysca: detector pixel coordinates
    """
    crpix1 = payload["det"]["crpix1"]
    crpix2 = payload["det"]["crpix2"]
    plate_scale = payload["det"]["plate_scale"]
    xcen, ycen = payload["det"]["xy_center"]
    
    xoff = xmpa - xcen
    yoff = ympa - ycen
    
    xrot = xoff * plate_scale
    yrot = yoff * plate_scale
    
    xsca = -xrot + crpix1
    ysca = yrot + crpix2
    
    return xsca, ysca


def sca_to_fpa(payload, xsca, ysca):
    """
    Convert SCA pixel coordinates to FPA (arcsec, as degrees).
    
    Args:
        payload: dict from make_sca_payload
        xsca, ysca: detector pixel coordinates
    
    Returns:
        xfpa, yfpa: position in degrees
    """
    crpix1 = payload["det"]["crpix1"]
    crpix2 = payload["det"]["crpix2"]
    plate_scale = payload["det"]["plate_scale"]
    pixel_scale = payload["det"]["pixel_scale"]
    xcen, ycen = payload["det"]["xy_center"]
    
    dx = (xsca - crpix1) * -1.0
    dy = ysca - crpix2
    
    xfpa = (xcen * plate_scale + dx) * pixel_scale / 3600.0
    yfpa = (ycen * plate_scale + dy) * pixel_scale / 3600.0
    
    return xfpa, yfpa


def fpa_to_sca(payload, xfpa, yfpa):
    """
    Convert FPA (degrees) coordinates to SCA pixel coordinates.
    
    Args:
        payload: dict from make_sca_payload
        xfpa, yfpa: position in degrees
    
    Returns:
        xsca, ysca: detector pixel coordinates
    """
    crpix1 = payload["det"]["crpix1"]
    crpix2 = payload["det"]["crpix2"]
    plate_scale = payload["det"]["plate_scale"]
    pixel_scale = payload["det"]["pixel_scale"]
    xcen, ycen = payload["det"]["xy_center"]
    
    dx = (xfpa * 3600.0 / pixel_scale) - (xcen * plate_scale)
    dy = (yfpa * 3600.0 / pixel_scale) - (ycen * plate_scale)
    
    xsca = -dx + crpix1
    ysca = dy + crpix2
    
    return xsca, ysca


def mpa_to_fpa(payload, xmpa, ympa):
    """
    Convert MPA (mm) coordinates to FPA (degrees).
    
    Args:
        payload: dict from make_sca_payload
        xmpa, ympa: position in mm
    
    Returns:
        xfpa, yfpa: position in degrees
    """
    plate_scale = payload["det"]["plate_scale"]
    pixel_scale = payload["det"]["pixel_scale"]
    
    xfpa = xmpa * plate_scale * pixel_scale / 3600.0
    yfpa = ympa * plate_scale * pixel_scale / 3600.0
    
    return xfpa, yfpa


def fpa_to_mpa(payload, xfpa, yfpa):
    """
    Convert FPA (degrees) coordinates to MPA (mm).
    
    Args:
        payload: dict from make_sca_payload
        xfpa, yfpa: position in degrees
    
    Returns:
        xmpa, ympa: position in mm
    """
    plate_scale = payload["det"]["plate_scale"]
    pixel_scale = payload["det"]["pixel_scale"]
    
    xmpa = xfpa * 3600.0 / pixel_scale / plate_scale
    ympa = yfpa * 3600.0 / pixel_scale / plate_scale
    
    return xmpa, ympa


def get_pa_rotation(pa):
    """Return 2x2 rotation matrix for a given position angle (JAX version).

    Args:
        pa: position angle in degrees (scalar)

    Returns:
        2x2 rotation matrix (JAX array)
    """
    theta = jnp.deg2rad(pa + 180 - 60)
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta),  jnp.cos(theta)],
    ])


def get_fpa_pos(ra, dec, pointing_ra, pointing_dec, pointing_pa):
    """Convert sky coordinates to FPA position (JAX version).

    Args:
        ra, dec: source coordinates in degrees (1D arrays)
        pointing_ra, pointing_dec: telescope pointing in degrees (scalars)
        pointing_pa: position angle in degrees (scalar)

    Returns:
        xfpa, yfpa: FPA coordinates in degrees
    """
    dx = (ra - pointing_ra) * jnp.cos(jnp.deg2rad(dec))
    dy = dec - pointing_dec
    xy = jnp.stack([dx, dy])  # [2, N]
    rot_matrix = get_pa_rotation(pa=pointing_pa)
    xy = rot_matrix @ xy
    xfpa = -xy[0, :]
    yfpa = -xy[1, :]
    return xfpa, yfpa


# -------- polynomial functions --------


def get_mpa_coords(payload, xfpa, yfpa):
    """
    Return the trace offset location for a given reference pixel.
    Trace offset location defines the "wl_reference" wavelength.
    
    Args:
        payload: dict from make_sca_payload
        xfpa, yfpa: reference position in degrees (FPA coords), as 1D arrays
    
    Returns:
        xmpa, ympa: trace offset position in mm (MPA coords)
    """
    # Create Vandermonde matrices for polynomial evaluation: V[i,j] = x[i]^j
    # Use manual power computation instead of jnp.vander for flexibility:
    # - vander only accepts 1D arrays; manual powers broadcast naturally to 3D grids
    # - Performance is identical (~same HLO after optimization)
    # - Keeps door open for future vectorization with vmap over higher dimensions
    map_i = payload["poly"]["map_i"]
    map_j = payload["poly"]["map_j"]
    x_powers = xfpa[:, jnp.newaxis] ** jnp.arange(map_i)
    y_powers = yfpa[:, jnp.newaxis] ** jnp.arange(map_j)
    
    # Get coefficients
    X_ij = payload["poly"]["X_ij"]
    Y_ij = payload["poly"]["Y_ij"]
    
    # Evaluate 2D polynomials using einsum (more efficient than matmul + diagonal)
    xmpa = jnp.einsum('ni,ij,nj->n', x_powers, X_ij, y_powers, precision='highest')
    ympa = jnp.einsum('ni,ij,nj->n', x_powers, Y_ij, y_powers, precision='highest')
    
    return xmpa, ympa


def get_trace_coeffs(payload, xfpa, yfpa):
    """
    Return curvature and inverse dispersion solution coefficients.
    
    Args:
        payload: dict from make_sca_payload
        xfpa, yfpa: trace offset location in degrees (FPA coords), as 1D arrays
    
    Returns:
        crv: curvature coefficients, shape [i, n]
        ids: inverse dispersion coefficients, shape [i, n]
    """
    # Curvature coefficients (C_ijk)
    crv_j = payload["poly"]["crv_j"]
    crv_k = payload["poly"]["crv_k"]
    x_powers_crv = xfpa[:, jnp.newaxis] ** jnp.arange(crv_j)  # [n, j]
    y_powers_crv = yfpa[:, jnp.newaxis] ** jnp.arange(crv_k)  # [n, k]
    
    C_ijk = payload["poly"]["C_ijk"]  # [i, j, k]
    
    # Compute crv[i, n] using einsum: sum over j,k for each i,n
    # Original: x @ C_ijk @ y.T -> [n,i,n], then diagonal(axis1=1,axis2=2) -> [n,i], then transpose -> [i,n]
    # Einsum: directly compute shape [i, n]
    crv = jnp.einsum('nj,ijk,nk->in', x_powers_crv, C_ijk, y_powers_crv, precision='highest')
    
    # Inverse dispersion coefficients (D_ijk)
    ids_j = payload["poly"]["ids_j"]
    ids_k = payload["poly"]["ids_k"]
    x_powers_ids = xfpa[:, jnp.newaxis] ** jnp.arange(ids_j)  # [n, j]
    y_powers_ids = yfpa[:, jnp.newaxis] ** jnp.arange(ids_k)  # [n, k]
    
    D_ijk = payload["poly"]["D_ijk"]  # [i, j, k]
    
    # Compute ids[i, n] using einsum
    ids = jnp.einsum('nj,ijk,nk->in', x_powers_ids, D_ijk, y_powers_ids, precision='highest')
    
    return crv, ids


def trace_beam(payload, xfpa, yfpa, wavelength):
    """
    Trace beam position at given wavelength(s).
    
    Computes the position on the MPA (mm) for given FPA (degrees) and wavelength(s).
    This combines the reference position from get_mpa_coords with wavelength-dependent
    offsets computed from the curvature and inverse dispersion coefficients.
    
    Args:
        payload: dict from make_sca_payload containing model parameters
        xfpa, yfpa: FPA coordinates in degrees, shape [n]
        wavelength: wavelength(s) in microns, shape [n]
        
    Returns:
        xmpa, ympa: position in mm, shape [n]
    """
    # Get reference position (at wl_reference)
    xmpa_ref, ympa_ref = get_mpa_coords(payload, xfpa, yfpa)
    
    # Get trace coefficients
    crv, ids = get_trace_coeffs(payload, xfpa, yfpa)  # [i, n]
    
    # Transform wavelength
    wl_ref = payload["wl"]["reference"]
    wl_transform = payload["wl"]["transform"]
    
    if wl_transform == "linear":
        wl = wavelength - wl_ref
    elif wl_transform == "log":
        wl = jnp.log10(wavelength / wl_ref)
    else:
        raise ValueError(f"Invalid wavelength transform: {wl_transform}")
    
    # Compute dely: sum over i of ids[i] * wl^i
    ids_i = payload["poly"]["ids_i"]
    wl_powers = wl[:, jnp.newaxis] ** jnp.arange(ids_i)  # [n, i]
    dely = jnp.einsum('ni,in->n', wl_powers, ids, precision='highest')

    # Compute delx: sum over i of crv[i] * dely^i
    crv_i = payload["poly"]["crv_i"]
    dely_powers = dely[:, jnp.newaxis] ** jnp.arange(crv_i)  # [n, i]
    delx = jnp.einsum('ni,in->n', dely_powers, crv, precision='highest')
    
    return xmpa_ref + delx, ympa_ref + dely

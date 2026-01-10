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


# -------- polynomial functions --------


def get_map_coords(payload, xfpa, yfpa):
    """
    Return the trace offset location for a given reference pixel.
    Trace offset location defines the "wl_reference" wavelength.
    
    Args:
        payload: dict from make_sca_payload
        xfpa, yfpa: reference position in degrees (FPA coords), as arrays
    
    Returns:
        xmpa, ympa: trace offset position in mm (MPA coords)
    """
    # Create Vandermonde matrices: [n, i] and [n, j]
    map_i = payload["poly"]["map_i"]
    map_j = payload["poly"]["map_j"]
    x_powers = xfpa[:, jnp.newaxis] ** jnp.arange(map_i)
    y_powers = yfpa[:, jnp.newaxis] ** jnp.arange(map_j)
    
    # Get coefficients
    X_ij = payload["poly"]["X_ij"]
    Y_ij = payload["poly"]["Y_ij"]
    
    # Evaluate 2D polynomials using einsum (more efficient than matmul + diagonal)
    xmpa = jnp.einsum('ni,ij,nj->n', x_powers, X_ij, y_powers)
    ympa = jnp.einsum('ni,ij,nj->n', x_powers, Y_ij, y_powers)
    
    return xmpa, ympa

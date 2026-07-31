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

Precision convention
--------------------
Two rules, both learned the hard way (see `get_fpa_pos`):

1. **Absolute sky coordinates are float64 and stay on the host in NumPy;
   everything downstream of the tangent-plane difference is float32 and lives
   in JAX.** Right ascension in degrees is a large number whose *difference*
   from the pointing is small, so it must be differenced at float64 before it
   ever reaches JAX. Once differenced, the surviving quantities are bounded by
   the field radius (~0.4 deg) and float32 carries them to ~1e-3 px, which is
   far below anything we care about.

2. **Every matmul-class JAX op carries `precision='highest'`.** With
   `jax_enable_x64` off, XLA:GPU serves an unannotated float32 `dot_general` as
   TF32 on Ampere and later — a 10-bit mantissa, eps ~ 4.9e-4 — while the same
   op on CPU is exact. That divergence is invisible to a CPU-only test suite.

   Annotate each site individually; do **not** reach for the process-global
   `JAX_DEFAULT_MATMUL_PRECISION` or `jax.config.update(...)`. A global default
   would mask a missing annotation rather than surface it, and setting it from
   library code would silently change the numerics of any other JAX work
   sharing the interpreter — including code that legitimately wants TF32.
   `tests/test_precision_convention.py` enforces the annotation by AST scan.
"""

import jax
import jax.numpy as jnp
import numpy as np


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


# Fixed orientation of the FPA coordinate system relative to the telescope,
# in degrees East of North. An instrument constant, not a tunable: the total
# on-sky orientation of the focal plane is (pointing PA + FOCAL_PA_DEG).
FOCAL_PA_DEG = -60.0


def get_pa_rotation(pa):
    """Return 2x2 rotation matrix for a given position angle (JAX version).

    Convention note (this construction confuses everyone, including its
    authors): together with the negation of both components in
    `get_fpa_pos_from_offsets`, the net operation is R_math(pa - 60) -- the
    +180 here and the negation there cancel exactly. This matrix is the math
    (counterclockwise) convention, which on the sky (x = East pointing left)
    rotates North toward *West*; with the opposite angle sign it is identical
    to the North-toward-East matrix R_NE(-(pa + FOCAL_PA_DEG)) used in the
    derivation notebook, i.e. the inverse rotation that takes sky-aligned
    tangent-plane coordinates into a focal plane oriented at
    (pa + FOCAL_PA_DEG) on the sky. Full derivation and a numerical proof of
    the equivalence: docs/reference/tangent_plane_derivation.ipynb, and the
    executable pin in
    `tests/test_optical_model_jax.py::TestGnomonicNotebookOracle`.

    Args:
        pa: position angle in degrees, East of North (scalar)

    Returns:
        2x2 rotation matrix (JAX array)
    """
    theta = jnp.deg2rad(pa + 180 + FOCAL_PA_DEG)
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta),  jnp.cos(theta)],
    ])


def sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec):
    """Gnomonic (TAN) offsets of sources from the pointing (degrees).

    Projects (ra, dec) onto the plane tangent to the celestial sphere at the
    pointing, returning sky-aligned tangent-plane coordinates
    (:math:`\\xi` = East, :math:`\\eta` = North) in degrees. The algorithm
    rotates the pointing to the North Pole in 3-D (an RA shift folded into the
    Cartesian conversion, then :math:`R_y(\\delta_0 - \\pi/2)`) and projects by
    dividing by the z component. It is a **verbatim transcription of steps
    1-3 of the derivation notebook**,
    ``docs/reference/tangent_plane_derivation.ipynb`` — same operations in the
    same order, so
    `tests/test_optical_model_jax.py::TestGnomonicNotebookOracle` can assert
    exact (bitwise) equality against the notebook function executed straight
    from that file. Do not "clean up" the arithmetic here without updating
    that contract.

    This replaced the flat-sky approximation
    (:math:`\\Delta\\alpha \\cos\\delta`, :math:`\\Delta\\delta`; issues #5 and
    #19). The flat-sky errors it removes — derived in the notebook's Taylor
    expansion — are third order in the offset at the equator
    (:math:`\\theta^3/3`) but pick up a **second-order** North error
    :math:`\\Delta\\alpha^2 \\sin(2\\delta_0)/4` off the equator (worst at
    :math:`\\delta_0 = 45^\\circ`): over a ±0.4 deg field, a median 0.12 px at
    Dec 0 grows to ~20 px at Dec 60. Because the projection uses sin/cos of
    (ra - ra0), it is periodic in RA by construction, so a field straddling
    RA = 0 needs no wrap handling — the former meridian-crossing ValueError
    guard is gone.

    Host-side (NumPy, float64) by design — see rule 1 of the precision
    convention in the module docstring. Right ascension in degrees is a large
    number whose difference from the pointing is small, so differencing it at
    float32 destroys the very quantity we need:

        pointing RA     cast-then-subtract      subtract-then-cast
              10 deg           0.0062 px             0.000098 px
             150 deg           0.0999 px             0.000098 px
             260 deg           0.3995 px             0.000098 px

    Differencing first makes the error independent of where the telescope
    points, which is the entire purpose of this function existing separately
    from the JAX rotation that follows it. The gnomonic projection does not
    change this: float32 quantisation of *absolute* RA (~0.5 px at RA ≈ 260
    deg) happens before sin/cos of the difference can help, so the float32
    guard below is permanent.

    Args:
        ra, dec: source coordinates in degrees (1D arrays)
        pointing_ra, pointing_dec: telescope pointing in degrees (scalars)

    Returns:
        dx, dy: tangent-plane offsets (xi East, eta North) in degrees,
            float64 NumPy arrays

    Raises:
        TypeError: if ra or dec arrive as float32 (e.g. a JAX array with
            x64 disabled). By then the quantisation has already happened and
            upcasting cannot undo it, so refuse rather than silently produce
            the pointing-dependent error this function exists to remove.
    """
    for name, value in (("ra", ra), ("dec", dec)):
        dtype = getattr(value, "dtype", None)
        if dtype is not None and np.dtype(dtype) == np.float32:
            raise TypeError(
                f"{name} arrived as float32: absolute sky coordinates were "
                "quantised before the tangent-plane differencing, which is "
                "the error this function exists to prevent (rule 1 of the "
                "precision convention). Pass float64 host arrays -- e.g. "
                "df['ra'].values, not jnp.array(df['ra'].values)."
            )
    # Transcription of tangent_plane() steps 1-3 from
    # docs/reference/tangent_plane_derivation.ipynb (bit-exact contract).
    ra_rad = np.deg2rad(np.asarray(ra, dtype=float))
    dec_rad = np.deg2rad(np.asarray(dec, dtype=float))
    ra0 = np.deg2rad(np.float64(pointing_ra))
    dec0 = np.deg2rad(np.float64(pointing_dec))

    # Step 1 + 2a: shift RA so the pointing is at RA = 0, then convert to 3-D
    # Cartesian coordinates on the unit sphere.
    da = ra_rad - ra0
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(da)
    y = cos_dec * np.sin(da)
    z = np.sin(dec_rad)

    # Step 2b: rotate about the y-axis by (dec0 - pi/2) to bring the pointing
    # to the North Pole: cos(dec0 - pi/2) = sin(dec0), sin(dec0 - pi/2) =
    # -cos(dec0).
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)
    xr = sin_dec0 * x - cos_dec0 * z
    yr = y
    zr = cos_dec0 * x + sin_dec0 * z

    # Step 3: gnomonic projection onto the tangent plane at the pole. yr
    # points East, xr points -Dec, so xi = East, eta = North.
    xi = yr / zr
    eta = -xr / zr

    return np.rad2deg(xi), np.rad2deg(eta)


def get_fpa_pos_from_offsets(dx, dy, pointing_pa):
    """Rotate tangent-plane offsets into FPA coordinates (JAX, jittable).

    This is the JAX half of the sky->FPA transform. Its inputs are already
    small (bounded by the field radius, ~0.4 deg), so float32 is accurate to
    ~1e-3 px here and no float64 is needed — see rule 1 of the precision
    convention.

    Args:
        dx, dy: tangent-plane offsets in degrees, from `sky_to_tangent_offsets`
        pointing_pa: position angle in degrees (scalar)

    Returns:
        xfpa, yfpa: FPA coordinates in degrees
    """
    xy = jnp.stack([jnp.asarray(dx), jnp.asarray(dy)])  # [2, N]
    rot_matrix = get_pa_rotation(pa=pointing_pa)
    # precision='highest' is mandatory, not stylistic: without it XLA:GPU runs
    # this float32 dot as TF32 (10-bit mantissa) on Ampere and later, which
    # displaced every source in the 2026-07 SSC line-grid package by a median
    # 1.84 px and up to 7.08 px. The same op on CPU is exact, so no CPU-only
    # test can see it. Note `@` takes no precision argument -- jnp.matmul does.
    xy = jnp.matmul(rot_matrix, xy, precision='highest')
    xfpa = -xy[0, :]
    yfpa = -xy[1, :]
    return xfpa, yfpa


def get_fpa_pos(ra, dec, pointing_ra, pointing_dec, pointing_pa):
    """Convert sky coordinates to FPA position.

    Thin composition of `sky_to_tangent_offsets` (host, float64) and
    `get_fpa_pos_from_offsets` (JAX, float32). Split so that the differencing
    happens at float64 before anything reaches the GPU.

    .. note::
        **Not jit-compilable**, deliberately: the float64 differencing is host
        NumPy. Jit `get_fpa_pos_from_offsets` instead, which is the part that
        belongs on the device anyway. In production this runs once per pointing
        on a few thousand sources, so it is not on any hot path.

    Args:
        ra, dec: source coordinates in degrees (1D arrays)
        pointing_ra, pointing_dec: telescope pointing in degrees (scalars)
        pointing_pa: position angle in degrees (scalar)

    Returns:
        xfpa, yfpa: FPA coordinates in degrees
    """
    dx, dy = sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec)
    return get_fpa_pos_from_offsets(dx, dy, pointing_pa)


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

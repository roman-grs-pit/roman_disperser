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

Position contractions: einsum vs elementwise
--------------------------------------------
The source-position math (the 2x2 PA rotation, the 2D position polynomial,
the trace-coefficient polynomials and their wavelength/dely evaluation, here
and in `disperser`) contracts over at most ~6 terms. Written as matmul/einsum,
XLA:GPU lowers these to Triton GEMM fusions whose tile configuration the
autotuner picks by *timing* at each compile. Two configurations sum in
different orders, so a fresh compile can move a source by 1-2 float32 ulp
(~5e-4 px), which shows as a ~1.5e-4-of-peak dipole on steep traces
(agent_work/tests_cluster/RESULTS_determinism.md). The ``"elementwise"`` mode
writes the same sums as broadcast-multiply + ``jnp.sum``: plain fusions with no
autotuned kernel choice, so every compile gives the same bits. ``"einsum"`` is
the original code, kept for comparison.

Select with the environment variable ``ROMAN_DISPERSER_POSITION_CONTRACTION``
(read at import) or :func:`set_position_contraction` (which clears JAX's jit
caches, because the mode is read when a function is traced).
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np

POSITION_CONTRACTION_MODES = ("elementwise", "horner", "einsum")
POSITION_CONTRACTION = os.environ.get("ROMAN_DISPERSER_POSITION_CONTRACTION", "einsum")
if POSITION_CONTRACTION not in POSITION_CONTRACTION_MODES:
    raise ValueError(
        f"ROMAN_DISPERSER_POSITION_CONTRACTION={POSITION_CONTRACTION!r}; "
        f"expected one of {POSITION_CONTRACTION_MODES}")


def set_position_contraction(mode):
    """Select ``"elementwise"`` or ``"einsum"`` position contractions.

    The mode is read when a function is traced, so this clears JAX's jit
    caches; functions compiled before the call recompile on next use.
    """
    global POSITION_CONTRACTION
    if mode not in POSITION_CONTRACTION_MODES:
        raise ValueError(f"mode {mode!r}; expected one of {POSITION_CONTRACTION_MODES}")
    POSITION_CONTRACTION = mode
    jax.clear_caches()


def _elementwise():
    """True for both GEMM-free modes ("elementwise" broadcast-sum, "horner")."""
    return POSITION_CONTRACTION in ("elementwise", "horner")


def _horner():
    return POSITION_CONTRACTION == "horner"


def _horner1d(t, coef):
    """sum_i coef[i] t^i by Horner; coef is a static-length sequence of arrays broadcastable with t."""
    p = coef[-1]
    if len(coef) == 1:
        return p + jnp.zeros_like(t)
    for c in coef[-2::-1]:
        p = p * t + c
    return p


def _horner2d(x, y, coef):
    """sum_ij coef[..., i, j] x^i y^j by nested Horner.

    Vectorised over the coefficient indices: one Horner pass in y over the
    last axis (all i at once), then one in x. Per element this is the same
    sequence of multiply-adds as a fully unrolled Horner, but the traced
    program has only n_i + n_j steps, which keeps compiles of the galaxy
    batch (vmap/jvp/transpose of this) small.
    """
    ni, nj = coef.shape[-2:]
    t = coef[..., nj - 1, None]                 # [..., ni, 1]
    for j in range(nj - 2, -1, -1):
        t = t * y + coef[..., j, None]          # [..., ni, n]
    t = t + jnp.zeros_like(y)                   # nj == 1: broadcast to [..., ni, n]
    p = t[..., ni - 1, :]
    for i in range(ni - 2, -1, -1):
        p = p * x + t[..., i, :]
    return p


def _poly2d(x_powers, coef, y_powers):
    """sum_ij x_powers[n,i] coef[i,j] y_powers[n,j] -> [n]  ('ni,ij,nj->n')."""
    if _elementwise():
        return jnp.sum(x_powers[:, :, None] * coef[None] * y_powers[:, None, :], axis=(1, 2))
    return jnp.einsum('ni,ij,nj->n', x_powers, coef, y_powers, precision='highest')


def _poly2d_coeffs(x_powers, coef, y_powers):
    """sum_jk x_powers[n,j] coef[i,j,k] y_powers[n,k] -> [i,n]  ('nj,ijk,nk->in')."""
    if _elementwise():
        return jnp.sum(x_powers[None, :, :, None] * coef[:, None, :, :]
                       * y_powers[None, :, None, :], axis=(2, 3))
    return jnp.einsum('nj,ijk,nk->in', x_powers, coef, y_powers, precision='highest')


def _poly1d(powers, coef):
    """sum_i powers[n,i] coef[i,n] -> [n]  ('ni,in->n')."""
    if _elementwise():
        return jnp.sum(powers * coef.T, axis=1)
    return jnp.einsum('ni,in->n', powers, coef, precision='highest')


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
        # SCA-local re-expansion of the same polynomials (float64 on the host,
        # stored float32), evaluated by `trace_sca`. See its docstring.
        "local": make_local_poly(model, sca, order),
        # float64 host copies of the detector constants (Python floats: a
        # label, never a jit argument) for the host-side float64 maps.
        "host": {
            "crpix1": float(model.detmod["crpix1"]),
            "crpix2": float(model.detmod["crpix2"]),
            "pixel_scale": float(model.detmod["pixel_scale"]),
            "plate_scale": float(model.detmod["plate_scale"]),
            "xy_center": (float(xcen), float(ycen)),
        },
    }


# -------- SCA-local trace (precision) --------
#
# The legacy chain sca_to_fpa -> trace_beam -> mpa_to_sca carries absolute
# FPA degrees (~0.3 deg) and MPA millimetres (~100 mm) in float32 and
# subtracts the SCA centre at the very end. One float32 ulp of 100 mm is
# 7.6e-6 mm ~ 7.6e-4 px, and the coefficients themselves (X_ij in mm) lose as
# much when cast to float32, so the dispersed positions came out 1-7e-4 px
# off (median) and up to 4.5e-3 px (SCAs far from the FPA centre). At a
# bright, steep trace edge (~0.3 peak/px) that is a dipole of up to ~7e-4 of
# peak. Re-expanding every polynomial on the host, in float64, in the
# *normalised pixel offsets from the SCA reference pixel* and folding the
# SCA-centre subtraction and the mm->px scale into the coefficients keeps every
# float32 intermediate small; what is left is the float32 rounding of the
# final absolute pixel position (<= 1.2e-4 px at x > 2048).

LOCAL_SCALE = 2048.0   # u = (crpix1 - xsca) / LOCAL_SCALE, v = (ysca - crpix2) / LOCAL_SCALE
POSITION_MODES = ("local", "legacy")


def _shift_poly2d(C, x0, sx, y0, sy):
    """Re-expand ``sum_ij C[i,j] x^i y^j`` with ``x = x0 + sx*u``,
    ``y = y0 + sy*v`` as ``sum_ab D[a,b] u^a v^b`` (float64, host)."""
    C = np.asarray(C, np.float64)
    I, J = C.shape
    D = np.zeros((I, J), np.float64)
    for i in range(I):
        for j in range(J):
            if C[i, j] == 0.0:
                continue
            for a in range(i + 1):
                ca = math.comb(i, a) * x0 ** (i - a) * sx ** a
                for b in range(j + 1):
                    D[a, b] += C[i, j] * ca * math.comb(j, b) * y0 ** (j - b) * sy ** b
    return D


def make_local_poly(model, sca, order):
    """Host float64 SCA-local polynomial coefficients for :func:`trace_sca`.

    Returns float32 JAX arrays with the shapes of the ``poly`` coefficients:

    * ``X``, ``Y`` [map_i, map_j]: reference trace offset from the SCA
      reference pixel in pixels, ``(xmpa_ref - xcen) * plate_scale`` (and y),
      as polynomials in ``u, v``;
    * ``C`` [crv_i, crv_j, crv_k], ``D`` [ids_i, ids_j, ids_k]: the curvature
      and inverse-dispersion coefficients (mm, as before) as polynomials in
      ``u, v``.
    """
    det = model.detmod
    xc, yc = model.coords.get_sca_center(sca=sca)
    ps = float(det["plate_scale"])
    pix = float(det["pixel_scale"])
    # sca_to_fpa: xfpa = (xc*ps - (xsca - crpix1)) * pix/3600 = xf0 + s*u
    xf0 = float(xc) * ps * pix / 3600.0
    yf0 = float(yc) * ps * pix / 3600.0
    s = LOCAL_SCALE * pix / 3600.0
    b = model.beam_coeffs[order]
    X = _shift_poly2d(b["X_ij"], xf0, s, yf0, s)
    Y = _shift_poly2d(b["Y_ij"], xf0, s, yf0, s)
    X[0, 0] -= xc
    Y[0, 0] -= yc
    X *= ps
    Y *= ps
    C = np.stack([_shift_poly2d(c, xf0, s, yf0, s) for c in b["C_ijk"]])
    D = np.stack([_shift_poly2d(d, xf0, s, yf0, s) for d in b["D_ijk"]])
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    return {"X": f32(X), "Y": f32(Y), "C": f32(C), "D": f32(D)}


def _poly2(C, u, v):
    """``sum_ab C[..., a, b] u^a v^b`` for C [L..., A, B]; result [shape(u, v), L...].

    Horner in a fixed order with elementwise ops only (vectorised over the
    leading coefficient axes L). No ``dot_general`` (so no autotuned GEMM
    choice, see RESULTS_determinism) and no reduction (whose association
    order XLA may pick per array shape, which broke the bit-identity of
    sharded vs single-device runs)."""
    lead = C.ndim - 2
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    uu = u.reshape(u.shape + (1,) * lead)
    vv = v.reshape(v.shape + (1,) * lead)
    A, B = C.shape[-2], C.shape[-1]
    out = None
    for a in range(A - 1, -1, -1):
        row = C[..., a, B - 1] + jnp.zeros_like(vv)
        for b in range(B - 2, -1, -1):
            row = row * vv + C[..., a, b]
        out = row if out is None else out * uu + row
    return out


def position_mode(mode=None):
    """Resolve the trace position mode: argument, else the
    ``ROMAN_DISPERSER_POSITIONS`` environment variable, else ``"local"``."""
    mode = mode or os.environ.get("ROMAN_DISPERSER_POSITIONS") or "local"
    if mode not in POSITION_MODES:
        raise ValueError(f"position mode {mode!r} not in {POSITION_MODES}")
    return mode


def trace_sca(payload, xsca, ysca, wavelength, mode=None):
    """Dispersed SCA position of a source at (xsca, ysca) and wavelength(s).

    Same result as ``mpa_to_sca(trace_beam(sca_to_fpa(...)))`` but evaluated
    in SCA-local coordinates (see the comment above ``LOCAL_SCALE``), which
    cuts the float32 position error ~10x (p99 ~2.5e-4 px instead of
    ~2e-3 px). Inputs broadcast elementwise; the output has their broadcast
    shape. Jittable and differentiable.

    ``mode="legacy"`` (or ``ROMAN_DISPERSER_POSITIONS=legacy``, read at trace
    time) selects the original chain; payloads without a ``local`` entry
    always use it.
    """
    if position_mode(mode) == "legacy" or "local" not in payload:
        xs, ys, wl = jnp.broadcast_arrays(jnp.asarray(xsca), jnp.asarray(ysca),
                                          jnp.asarray(wavelength))
        shape = xs.shape
        xf, yf = sca_to_fpa(payload, xs.reshape(-1), ys.reshape(-1))
        xm, ym = trace_beam(payload, xf, yf, wl.reshape(-1))
        xo, yo = mpa_to_sca(payload, xm, ym)
        return xo.reshape(shape), yo.reshape(shape)
    loc = payload["local"]
    det = payload["det"]
    c1, c2 = det["crpix1"], det["crpix2"]
    u = (c1 - xsca) * (1.0 / LOCAL_SCALE)
    v = (ysca - c2) * (1.0 / LOCAL_SCALE)
    wl_ref = payload["wl"]["reference"]
    if payload["wl"]["transform"] == "linear":
        w = wavelength - wl_ref
    elif payload["wl"]["transform"] == "log":
        w = jnp.log10(wavelength / wl_ref)
    else:
        raise ValueError(f"Invalid wavelength transform: {payload['wl']['transform']}")
    ids = _poly2(loc["D"], u, v)             # [..., ids_i] (mm)
    crv = _poly2(loc["C"], u, v)             # [..., crv_i] (mm)
    dely = ids[..., -1]
    for i in range(ids.shape[-1] - 2, -1, -1):
        dely = dely * w + ids[..., i]
    delx = crv[..., -1]
    for i in range(crv.shape[-1] - 2, -1, -1):
        delx = delx * dely + crv[..., i]
    ps = det["plate_scale"]
    xo = c1 - (_poly2(loc["X"], u, v) + delx * ps)
    yo = c2 + (_poly2(loc["Y"], u, v) + dely * ps)
    return jnp.broadcast_arrays(xo, yo)


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
    :math:`\\delta_0 = 45^\\circ`): over a ±0.4 deg field, median 0.12 px
    (max 0.72 px) at Dec 0 growing to median ~9.6 px (max ~55 px) at Dec 60;
    the golden-value literals shifted 0.064-19.7 px on regeneration.
    Because the projection uses sin/cos of
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
    if _elementwise():
        # Same two products and one add per output as the 2x2 matmul, as a
        # plain fusion (no autotuned GEMM; see "Position contractions").
        xfpa = -(rot_matrix[0, 0] * xy[0] + rot_matrix[0, 1] * xy[1])
        yfpa = -(rot_matrix[1, 0] * xy[0] + rot_matrix[1, 1] * xy[1])
        return xfpa, yfpa
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


# ---------------------------------------------------------------------------
# Differentiable sky -> detector transform (small perturbations)
# ---------------------------------------------------------------------------
#
# sky_to_tangent_offsets is host float64 by design (rule 1 above), so RA/Dec
# cannot be traced. For gradients w.r.t. source positions and the pointing,
# the absolute geometry is evaluated once in float64 at the nominal pointing
# (make_sky_reference) and the perturbed position is that reference plus a
# *change* computed in float32 JAX from small quantities only. Every term of
# the change is a small number times an O(1) one, so float32 carries it to
# ~1e-7 of its size (~2e-4 px for a 0.1 deg pointing change), and the only
# other rounding is the final reference + change.


def get_sca_pos_f64(ra, dec, pointing_ra, pointing_dec, pointing_pa, payload):
    """Undispersed SCA position of sky sources, float64 on the host.

    The production driver used to compute this as ``fpa_to_sca`` of the
    float32 FPA position (:func:`get_fpa_pos`), which rounds the absolute FPA
    degrees (~0.3 deg) to ~1e-3 px. Here the tangent-plane offsets, the PA
    rotation and the FPA -> SCA map are all float64 NumPy (with the payload's
    float64 ``host`` constants when present); cast the result to float32 only
    at the end (<= 1.2e-4 px).

    Returns:
        xsca, ysca: float64 NumPy arrays (1-indexed FITS pixels)
    """
    xi, eta = sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec)
    theta = np.deg2rad(np.float64(pointing_pa) + 180.0 + FOCAL_PA_DEG)
    c, s = np.cos(theta), np.sin(theta)
    xfpa = -(c * xi - s * eta)
    yfpa = -(s * xi + c * eta)
    det = payload["host"] if "host" in payload else payload["det"]
    return _fpa_to_sca_f64(det, xfpa, yfpa)


def _fpa_to_sca_f64(det, xfpa, yfpa):
    """float64 NumPy :func:`fpa_to_sca` (host reference)."""
    pixel_scale = float(det["pixel_scale"])
    plate_scale = float(det["plate_scale"])
    xcen, ycen = (float(v) for v in det["xy_center"])
    dx = xfpa * 3600.0 / pixel_scale - xcen * plate_scale
    dy = yfpa * 3600.0 / pixel_scale - ycen * plate_scale
    return -dx + float(det["crpix1"]), dy + float(det["crpix2"])


def make_sky_reference(ra, dec, pointing_ra, pointing_dec, pointing_pa,
                       payload=None):
    """Host float64 reference for the ``perturbed_sky_to_*`` functions.

    Runs the float64 tangent-plane differencing of
    :func:`sky_to_tangent_offsets`, the PA rotation and (with ``payload``)
    the FPA -> SCA map once at the nominal pointing, in float64 NumPy, and
    keeps the results as float32 JAX arrays together with sin/cos of the
    nominal Dec and PA rotation angle. Everything downstream is float32 JAX
    (rule 1 of the precision convention).

    Args:
        ra, dec: source coordinates in degrees (float64 host arrays, as for
            :func:`sky_to_tangent_offsets`, which rejects float32)
        pointing_ra, pointing_dec, pointing_pa: nominal pointing in degrees
        payload: optional :func:`make_sca_payload` result; adds the nominal
            SCA position (needed by :func:`perturbed_sky_to_sca` for its
            full accuracy)

    Returns:
        dict of float32 JAX arrays: ``xi0``, ``eta0``, ``xfpa0``, ``yfpa0``
        [N] (degrees), optionally ``xsca0``, ``ysca0`` [N] (pixels), and the
        scalars ``sin_dec0``, ``cos_dec0``, ``cos_pa0``, ``sin_pa0``
    """
    xi, eta = sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec)
    dec0 = np.deg2rad(np.float64(pointing_dec))
    theta0 = np.deg2rad(np.float64(pointing_pa) + 180.0 + FOCAL_PA_DEG)
    c, s = np.cos(theta0), np.sin(theta0)
    xfpa = -(c * xi - s * eta)
    yfpa = -(s * xi + c * eta)
    f32 = lambda v: jnp.asarray(v, jnp.float32)
    ref = {
        "xi0": f32(xi), "eta0": f32(eta),
        "xfpa0": f32(xfpa), "yfpa0": f32(yfpa),
        "sin_dec0": f32(np.sin(dec0)), "cos_dec0": f32(np.cos(dec0)),
        "cos_pa0": f32(c), "sin_pa0": f32(s),
    }
    if payload is not None:
        xs, ys = _fpa_to_sca_f64(payload["det"], xfpa, yfpa)
        ref["xsca0"], ref["ysca0"] = f32(xs), f32(ys)
    return ref


def _two_sin2_half(a):
    """``1 - cos(a)`` without cancellation: ``2 sin^2(a/2)``."""
    h = jnp.sin(0.5 * a)
    return 2.0 * h * h


def _tangent_change(ref, d_xi, d_eta, d_ra, d_dec):
    """Change (degrees) of the tangent-plane offsets, see
    :func:`perturbed_tangent_offsets`."""
    deg = jnp.float32(np.pi / 180.0)
    dxi = jnp.asarray(d_xi, jnp.float32)
    deta = jnp.asarray(d_eta, jnp.float32)
    da = jnp.asarray(d_ra, jnp.float32) * deg
    dd = jnp.asarray(d_dec, jnp.float32) * deg
    s0, c0 = ref["sin_dec0"], ref["cos_dec0"]

    # Source vector in the nominal pointing frame, unnormalised and scaled
    # by 180/pi so that its x, y components are offsets in degrees
    # (xr = -eta, yr = xi, zr = 180/pi).
    big = jnp.float32(180.0 / np.pi)
    sx = -(ref["eta0"] + deta)
    sy = ref["xi0"] + dxi
    sz = jnp.full_like(sy, big)

    # New frame = M @ nominal, M = F(Dec0 + d_dec) Rz(-d_ra) F(Dec0)^T, with
    # F(d) = [[sin d, 0, -cos d], [0, 1, 0], [cos d, 0, sin d]] the
    # RA-shifted equatorial -> pointing-frame rotation of
    # sky_to_tangent_offsets. M - I = (rotation about y by d_dec) - I
    # + F(d1) Z F(d0)^T with Z = Rz(-d_ra) - I; only sin and 2 sin^2(x/2)
    # of the small angles enter, so nothing O(1) cancels.
    sin_dd, omc_dd = jnp.sin(dd), _two_sin2_half(dd)
    s1 = s0 * (1.0 - omc_dd) + c0 * sin_dd           # sin(Dec0 + d_dec)
    c1 = c0 * (1.0 - omc_dd) - s0 * sin_dd           # cos(Dec0 + d_dec)
    sin_da, omc_da = jnp.sin(da), _two_sin2_half(da)
    ux = s0 * sx + c0 * sz                           # u = F(d0)^T s
    uy = sy
    wx = -omc_da * ux + sin_da * uy                  # w = Z u (wz = 0)
    wy = -sin_da * ux - omc_da * uy
    # (M - I) s, componentwise
    ex = -omc_dd * sx + sin_dd * sz + s1 * wx
    ey = wy
    ez = -sin_dd * sx - omc_dd * sz + c1 * wx

    # Tangent ratios: xi' = big * (sy + ey) / (sz + ez), and the change
    # xi' - xi = big * (ey * sz - sy * ez) / ((sz + ez) * sz), likewise eta.
    nz = sz + ez
    scale = big / (nz * sz)
    dxi_t = dxi + scale * (ey * sz - sy * ez)
    deta_t = deta + scale * (-(ex * sz - sx * ez))
    return dxi_t, deta_t


def perturbed_tangent_offsets(ref, d_xi=0.0, d_eta=0.0, d_ra=0.0, d_dec=0.0):
    """Tangent-plane offsets after small source and pointing perturbations.

    JAX, float32, jittable and differentiable in every perturbation. The
    source is moved by ``(d_xi, d_eta)`` in the *nominal* tangent plane and
    the pointing moves to ``(RA0 + d_ra, Dec0 + d_dec)``; the result is the
    source's gnomonic offset (xi East, eta North) from the new pointing,
    exact up to float32 rounding (no small-angle approximation).

    Args:
        ref: from :func:`make_sky_reference`
        d_xi, d_eta: source offsets in degrees in the nominal tangent plane,
            scalars or [N]
        d_ra, d_dec: pointing offsets in degrees (``d_ra`` is a coordinate
            offset, ``d_ra * cos(Dec)`` on the sky)

    Returns:
        xi, eta: tangent-plane offsets in degrees
    """
    dxi, deta = _tangent_change(ref, d_xi, d_eta, d_ra, d_dec)
    return ref["xi0"] + dxi, ref["eta0"] + deta


def _fpa_change(ref, d_xi, d_eta, d_ra, d_dec, d_pa):
    dxi, deta = _tangent_change(ref, d_xi, d_eta, d_ra, d_dec)
    dp = jnp.asarray(d_pa, jnp.float32) * jnp.float32(np.pi / 180.0)
    sin_dp, omc_dp = jnp.sin(dp), _two_sin2_half(dp)
    c0, s0 = ref["cos_pa0"], ref["sin_pa0"]
    c = c0 * (1.0 - omc_dp) - s0 * sin_dp            # cos(theta0 + d_pa)
    s = s0 * (1.0 - omc_dp) + c0 * sin_dp            # sin(theta0 + d_pa)
    dc = -c0 * omc_dp - s0 * sin_dp                  # c - c0
    ds = -s0 * omc_dp + c0 * sin_dp                  # s - s0
    xi0, eta0 = ref["xi0"], ref["eta0"]
    # xfpa = -(c xi - s eta), yfpa = -(s xi + c eta) (get_fpa_pos_from_
    # offsets); elementwise, so there is no matmul to annotate.
    dx = -(c * dxi - s * deta) - (dc * xi0 - ds * eta0)
    dy = -(s * dxi + c * deta) - (ds * xi0 + dc * eta0)
    return dx, dy


def perturbed_sky_to_fpa(ref, d_xi=0.0, d_eta=0.0, d_ra=0.0, d_dec=0.0,
                         d_pa=0.0):
    """FPA position (degrees) after small source/pointing perturbations.

    :func:`perturbed_tangent_offsets` followed by the PA rotation of
    :func:`get_fpa_pos_from_offsets` at ``PA0 + d_pa`` (``d_pa`` in degrees,
    applied as a separate small rotation so it is not lost to the float32
    rounding of PA0 + d_pa). Jittable and differentiable. The result is the
    float64 reference plus a float32 change; its float32 rounding in degrees
    is ~5e-4 px at the field edge.
    """
    dx, dy = _fpa_change(ref, d_xi, d_eta, d_ra, d_dec, d_pa)
    return ref["xfpa0"] + dx, ref["yfpa0"] + dy


def perturbed_sky_to_sca(ref, payload, d_xi=0.0, d_eta=0.0, d_ra=0.0,
                         d_dec=0.0, d_pa=0.0, wavelengths=None):
    """SCA pixel position (and optionally the trace) after perturbations.

    The undispersed SCA position (:func:`fpa_to_sca`, as in the production
    driver) of :func:`perturbed_sky_to_fpa`. With ``ref`` built with this
    ``payload`` it is the float64 nominal pixel position plus the float32
    change (the FPA -> SCA map is linear), ~2e-4 px from a float64
    evaluation; without, it is ``fpa_to_sca`` of the float32 FPA position
    (~1e-3 px, like production). If ``wavelengths`` (microns) is given, the
    dispersed trace from :func:`trace_sca` is
    returned too. Jittable and differentiable (``jax.grad`` /
    ``jax.jacfwd``) in all perturbations.

    Returns:
        xsca, ysca [N]; with ``wavelengths``, also xtrace, ytrace [N, N_wl]
    """
    dx, dy = _fpa_change(ref, d_xi, d_eta, d_ra, d_dec, d_pa)
    xfpa, yfpa = ref["xfpa0"] + dx, ref["yfpa0"] + dy
    if "xsca0" in ref:
        k = jnp.float32(3600.0) / payload["det"]["pixel_scale"]
        xsca, ysca = ref["xsca0"] - k * dx, ref["ysca0"] + k * dy
    else:
        xsca, ysca = fpa_to_sca(payload, xfpa, yfpa)
    if wavelengths is None:
        return xsca, ysca
    wavelengths = jnp.asarray(wavelengths, jnp.float32)

    def trace_one(xs, ys):   # as star_disperser._compute_dispersed_positions
        xt, yt = trace_sca(payload, xs[None], ys[None], wavelengths)
        return xt.reshape(-1), yt.reshape(-1)

    xt, yt = jax.vmap(trace_one)(jnp.atleast_1d(xsca), jnp.atleast_1d(ysca))
    return xsca, ysca, xt, yt

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
    if _horner():
        return _horner2d(xfpa, yfpa, X_ij), _horner2d(xfpa, yfpa, Y_ij)
    xmpa = _poly2d(x_powers, X_ij, y_powers)
    ympa = _poly2d(x_powers, Y_ij, y_powers)
    
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
    if _horner():
        return (_horner2d(xfpa, yfpa, payload["poly"]["C_ijk"]),
                _horner2d(xfpa, yfpa, payload["poly"]["D_ijk"]))
    crv = _poly2d_coeffs(x_powers_crv, C_ijk, y_powers_crv)
    
    # Inverse dispersion coefficients (D_ijk)
    ids_j = payload["poly"]["ids_j"]
    ids_k = payload["poly"]["ids_k"]
    x_powers_ids = xfpa[:, jnp.newaxis] ** jnp.arange(ids_j)  # [n, j]
    y_powers_ids = yfpa[:, jnp.newaxis] ** jnp.arange(ids_k)  # [n, k]
    
    D_ijk = payload["poly"]["D_ijk"]  # [i, j, k]
    
    # Compute ids[i, n] using einsum
    ids = _poly2d_coeffs(x_powers_ids, D_ijk, y_powers_ids)
    
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
    if _horner():
        dely = _horner1d(wl, list(ids))
        delx = _horner1d(dely, list(crv))
        return xmpa_ref + delx, ympa_ref + dely
    dely = _poly1d(wl_powers, ids)

    # Compute delx: sum over i of crv[i] * dely^i
    crv_i = payload["poly"]["crv_i"]
    dely_powers = dely[:, jnp.newaxis] ** jnp.arange(crv_i)  # [n, i]
    delx = _poly1d(dely_powers, crv)
    
    return xmpa_ref + delx, ympa_ref + dely

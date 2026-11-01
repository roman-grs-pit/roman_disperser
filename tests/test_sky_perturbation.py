"""Differentiable sky -> detector transform for small perturbations.

``optical_model_jax.make_sky_reference`` (host float64) +
``perturbed_sky_to_fpa`` / ``perturbed_sky_to_sca`` (JAX float32) against
(a) a float64 NumPy evaluation of the same geometry and (b) the existing
``get_fpa_pos`` chain re-run at the perturbed pointing, plus gradients vs
finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj

PX_PER_DEG = 3600.0 / 0.11          # nominal WFI pixel scale
POINTINGS = [(10.195391, 0.102452, 60.0), (150.1, 2.2, 13.0),
             (260.3, -65.4, 287.5), (35.0, 80.0, 200.0), (359.9, -30.0, 0.0)]


def _inv_gnomonic(xi, eta, ra0, dec0):
    """float64 inverse of ``sky_to_tangent_offsets`` (degrees in and out)."""
    xi, eta = np.deg2rad(xi), np.deg2rad(eta)
    d0 = np.deg2rad(dec0)
    xr, yr, zr = -eta, xi, np.ones_like(xi)
    n = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
    xr, yr, zr = xr / n, yr / n, zr / n
    x = np.sin(d0) * xr + np.cos(d0) * zr
    y = yr
    z = -np.cos(d0) * xr + np.sin(d0) * zr
    return np.rad2deg(np.arctan2(y, x)) + ra0, np.rad2deg(np.arcsin(z))


def _fpa64(xi, eta, pa):
    """float64 ``get_fpa_pos_from_offsets``."""
    th = np.deg2rad(pa + 180.0 + omj.FOCAL_PA_DEG)
    return (-(np.cos(th) * xi - np.sin(th) * eta),
            -(np.sin(th) * xi + np.cos(th) * eta))


def _field(rng, pointing, n=1500, det=None):
    """Random sources over the field; with ``det`` only those within ~200 px
    of that SCA (SCA pixel coordinates far off the chip are large numbers,
    whose float32 rounding is not what is being tested)."""
    ra0, dec0, pa0 = pointing
    m = n if det is None else 40 * n
    xi0 = rng.uniform(-0.4, 0.4, m)
    eta0 = rng.uniform(-0.4, 0.4, m)
    if det is not None:
        xs, ys = omj._fpa_to_sca_f64(det, *_fpa64(xi0, eta0, pa0))
        on = (xs > -200) & (xs < 4300) & (ys > -200) & (ys < 4300)
        xi0, eta0 = xi0[on][:n], eta0[on][:n]
    ra, dec = _inv_gnomonic(xi0, eta0, ra0, dec0)
    return xi0, eta0, ra, dec


@pytest.fixture(scope="module")
def payload(optical_model):
    return omj.make_sca_payload(optical_model, sca=5, order="1")


def _cases(rng, n):
    """(dxi, deta, dra, ddec, dpa): none, then up to 10 arcsec per source
    and 0.1 deg in pointing."""
    out = [(np.zeros(n), np.zeros(n), 0.0, 0.0, 0.0)]
    for _ in range(3):
        out.append((rng.uniform(-10, 10, n) / 3600.0,
                    rng.uniform(-10, 10, n) / 3600.0,
                    *rng.uniform(-0.1, 0.1, 3)))
    return out


def _truth(pointing, xi0, eta0, case, det):
    """float64: move the source in the nominal tangent plane, re-project at
    the perturbed pointing; FPA (deg) and SCA (px)."""
    ra0, dec0, pa0 = pointing
    dxi, deta, dra, ddec, dpa = case
    ra1, dec1 = _inv_gnomonic(xi0 + dxi, eta0 + deta, ra0, dec0)
    t1, e1 = omj.sky_to_tangent_offsets(ra1, dec1, ra0 + dra, dec0 + ddec)
    X, Y = _fpa64(t1, e1, pa0 + dpa)
    return (X, Y), omj._fpa_to_sca_f64(det, X, Y), (ra1, dec1)


@pytest.mark.parametrize("pointing", POINTINGS)
def test_matches_float64_and_existing_chain(pointing, payload):
    ra0, dec0, pa0 = pointing
    det = payload["det"]
    rng = np.random.default_rng(int(abs(dec0) * 10) + 1)
    xi0, eta0, ra, dec = _field(rng, pointing, n=1000, det=det)
    assert len(ra) > 500
    ref = omj.make_sky_reference(ra, dec, ra0, dec0, pa0, payload=payload)
    f_sca = jax.jit(lambda *a: omj.perturbed_sky_to_sca(ref, payload, *a))
    f_fpa = jax.jit(lambda *a: omj.perturbed_sky_to_fpa(ref, *a))
    for k, case in enumerate(_cases(rng, len(ra))):
        (X, Y), (XS, YS), (ra1, dec1) = _truth(pointing, xi0, eta0, case, det)
        x, y = (np.asarray(v, np.float64) for v in f_sca(*case))
        err = np.hypot(x - XS, y - YS).max()
        assert err < (5e-4 if k == 0 else 1.5e-3), (k, err)
        # FPA output: float32 degrees, rounding ~5e-4 px per axis
        xf, yf = (np.asarray(v, np.float64) for v in f_fpa(*case))
        errf = np.hypot(xf - X, yf - Y).max() * PX_PER_DEG
        assert errf < 2.5e-3, (k, errf)
        # existing chain (float64 differencing, float32 JAX rotation and
        # FPA -> SCA), re-run at the perturbed pointing. It carries 1-5e-3
        # px of float32 rounding of its own, hence the bound.
        dra, ddec, dpa = case[2:]
        xc, yc = omj.fpa_to_sca(payload, *omj.get_fpa_pos(
            ra1, dec1, ra0 + dra, dec0 + ddec, pa0 + dpa))
        xc, yc = np.asarray(xc, np.float64), np.asarray(yc, np.float64)
        chain64 = np.hypot(xc - XS, yc - YS).max()
        assert chain64 < 1e-2
        assert np.hypot(x - xc, y - yc).max() < chain64 + 1.5e-3


def test_without_payload_reference_and_trace(payload):
    """``ref`` without the payload: fpa_to_sca of the float32 FPA position
    (production accuracy); the trace option matches trace_beam."""
    ra0, dec0, pa0 = POINTINGS[0]
    rng = np.random.default_rng(3)
    _, _, ra, dec = _field(rng, POINTINGS[0], n=300, det=payload["det"])
    ref = omj.make_sky_reference(ra, dec, ra0, dec0, pa0)
    ref_p = omj.make_sky_reference(ra, dec, ra0, dec0, pa0, payload=payload)
    x, y = omj.perturbed_sky_to_sca(ref, payload)
    xp, yp = omj.perturbed_sky_to_sca(ref_p, payload)
    xf, yf = omj.get_fpa_pos(ra, dec, ra0, dec0, pa0)
    xs, ys = omj.fpa_to_sca(payload, xf, yf)
    for a in (x, xp):
        np.testing.assert_allclose(np.asarray(a), np.asarray(xs), atol=5e-3)
    for a in (y, yp):
        np.testing.assert_allclose(np.asarray(a), np.asarray(ys), atol=5e-3)
    wl = jnp.linspace(1.0, 1.9, 7)
    _, _, xt, yt = omj.perturbed_sky_to_sca(ref_p, payload, wavelengths=wl)
    xm, ym = jax.vmap(lambda a, b: omj.trace_beam(payload, a[None], b[None],
                                                  wl))(xf, yf)
    xt2, yt2 = omj.mpa_to_sca(payload, xm, ym)
    assert xt.shape == (len(ra), 7)
    np.testing.assert_allclose(np.asarray(xt), np.asarray(xt2), atol=5e-3)
    np.testing.assert_allclose(np.asarray(yt), np.asarray(yt2), atol=5e-3)


def test_rejects_float32_sky_coordinates():
    with pytest.raises(TypeError, match="float32"):
        omj.make_sky_reference(np.float32([10.0]), np.float32([0.1]),
                               10.0, 0.1, 60.0)


def _fd_jacobian(fun, p0, h):
    """Central differences of the float64 truth (well conditioned)."""
    cols = []
    for k in range(len(p0)):
        e = np.zeros(len(p0))
        e[k] = h[k]
        cols.append((np.asarray(fun(p0 + e)) - np.asarray(fun(p0 - e)))
                    / (2 * h[k]))
    return np.stack(cols, axis=-1)


@pytest.mark.parametrize("pointing", [POINTINGS[0], POINTINGS[2]])
def test_gradients_vs_finite_differences(payload, pointing):
    """jacfwd and grad w.r.t. (d_xi, d_eta, d_ra, d_dec, d_pa) of the SCA
    position vs central differences of the float64 truth."""
    ra0, dec0, pa0 = pointing
    rng = np.random.default_rng(11)
    xi0, eta0, ra, dec = _field(rng, pointing, n=6, det=payload["det"])
    ref = omj.make_sky_reference(ra, dec, ra0, dec0, pa0, payload=payload)
    p0 = np.array([2.0 / 3600, -3.0 / 3600, 0.02, -0.03, 0.05])

    def ad(p):
        p = jnp.asarray(p, jnp.float32)
        x, y = omj.perturbed_sky_to_sca(ref, payload, p[0], p[1], p[2], p[3],
                                        p[4])
        return jnp.stack([x, y])            # [2, N]

    J = np.asarray(jax.jacfwd(ad)(p0))     # [2, N, 5]

    def truth(p):
        _, (xs, ys), _ = _truth(pointing, xi0, eta0, tuple(p), payload["det"])
        return np.stack([xs, ys])

    Jfd = _fd_jacobian(truth, p0, h=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    scale = np.abs(Jfd).max(axis=(0, 1))   # per-parameter (px / deg)
    assert np.all(scale > 0)
    np.testing.assert_allclose(J, Jfd, atol=1e-4 * scale.max(),
                               rtol=1e-4)
    # reverse mode agrees with forward mode
    g = jax.grad(lambda p: jnp.sum(ad(p)[0] + 2.0 * ad(p)[1]))(
        jnp.asarray(p0, jnp.float32))
    np.testing.assert_allclose(np.asarray(g),
                               (J[0] + 2.0 * J[1]).sum(axis=0), rtol=1e-4,
                               atol=1e-4 * scale.max())

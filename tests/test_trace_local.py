"""SCA-local trace evaluation (``optical_model_jax.trace_sca``) and the
float64 host sky -> SCA map (``get_sca_pos_f64``): float32 position accuracy
against a float64 NumPy evaluation of the optical model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roman_disperser import optical_model_jax as omj

WL = {"grism": np.linspace(0.9, 2.0, 56), "prism": np.linspace(0.75, 1.85, 56)}


def _ref64(model, sca, order, xs, ys, wl):
    """float64 NumPy SCA -> dispersed SCA (the class model's formulas)."""
    det = model.detmod
    xc, yc = model.coords.get_sca_center(sca=sca)
    ps, pix = float(det["plate_scale"]), float(det["pixel_scale"])
    c1, c2 = float(det["crpix1"]), float(det["crpix2"])
    xf = (xc * ps - (xs - c1)) * pix / 3600.0
    yf = (yc * ps + (ys - c2)) * pix / 3600.0
    b = model.beam_coeffs[order]

    def p2(C, x, y):
        return np.einsum("ni,ij,nj->n", x[:, None] ** np.arange(C.shape[0]), C,
                         y[:, None] ** np.arange(C.shape[1]))

    def p3(C, x, y):
        return np.einsum("nj,ijk,nk->in", x[:, None] ** np.arange(C.shape[1]), C,
                         y[:, None] ** np.arange(C.shape[2]))

    xm, ym = p2(b["X_ij"], xf, yf), p2(b["Y_ij"], xf, yf)
    crv, ids = p3(b["C_ijk"], xf, yf), p3(b["D_ijk"], xf, yf)
    wr = float(model.wl_reference)
    w = (wl - wr) if str(model.wl_transform).lower() == "linear" else np.log10(wl / wr)
    dely = np.einsum("wi,in->nw", w[:, None] ** np.arange(ids.shape[0]), ids)
    delx = np.einsum("nwi,in->nw", dely[..., None] ** np.arange(crv.shape[0]), crv)
    return -(xm[:, None] + delx - xc) * ps + c1, (ym[:, None] + dely - yc) * ps + c2


def _element(model):
    return "prism" if "PRISM" in str(getattr(model, "config_file", "")).upper() \
        or float(model.wl_reference) < 0.8 else "grism"


@pytest.mark.parametrize("sca", [1, 9])
def test_local_trace_float32_accuracy(optical_model, sca):
    """local: <= 1e-3 px everywhere (float32 rounding of the absolute
    position); the legacy FPA/MPA chain is several times worse."""
    wl = WL[_element(optical_model)]
    rng = np.random.default_rng(sca)
    for order in optical_model.beam_coeffs:
        payload = omj.make_sca_payload(optical_model, sca=sca, order=order)
        xs = rng.uniform(100, 3988, 60).astype(np.float32)
        ys = rng.uniform(100, 3988, 60).astype(np.float32)
        rx, ry = _ref64(optical_model, sca, order, xs.astype(np.float64),
                        ys.astype(np.float64), wl.astype(np.float32).astype(np.float64))
        on = (rx > 0) & (rx < 4089) & (ry > 0) & (ry < 4089)
        if not on.any():
            continue
        out = {}
        for mode in ("local", "legacy"):
            f = jax.jit(lambda a, b: omj.trace_sca(
                payload, a[:, None], b[:, None], jnp.asarray(wl, jnp.float32)[None, :],
                mode=mode))
            tx, ty = map(np.asarray, f(jnp.asarray(xs), jnp.asarray(ys)))
            out[mode] = max(np.abs(tx - rx)[on].max(), np.abs(ty - ry)[on].max())
        assert out["local"] < 1e-3, (order, out)
        assert out["local"] < out["legacy"], (order, out)


def test_legacy_mode_is_the_old_chain(payload):
    """Same arithmetic as the old chain. The batch shape may change the float32
    summation order of its einsums (on GPU also the GEMM kernel), which moves
    the legacy result by up to ~5e-4 px (16 ulp at x ~ 380, L40S)."""
    xs = jnp.asarray([300.0, 2000.5, 3900.25], jnp.float32)
    ys = jnp.asarray([3500.0, 1000.75, 50.5], jnp.float32)
    wl = jnp.linspace(0.9, 1.9, 7).astype(jnp.float32)
    for x, y in zip(xs, ys):
        xf, yf = omj.sca_to_fpa(payload, x[None], y[None])
        xm, ym = omj.trace_beam(payload, xf, yf, wl)
        ox, oy = omj.mpa_to_sca(payload, xm, ym)
        tx, ty = omj.trace_sca(payload, x[None], y[None], wl, mode="legacy")
        np.testing.assert_allclose(np.asarray(tx).ravel(), np.asarray(ox).ravel(), rtol=0, atol=1e-3)
        np.testing.assert_allclose(np.asarray(ty).ravel(), np.asarray(oy).ravel(), rtol=0, atol=1e-3)


def test_local_and_legacy_agree_and_differentiate(payload):
    xs = jnp.asarray([300.0, 2000.5, 3900.25], jnp.float32)
    ys = jnp.asarray([3500.0, 1000.75, 50.5], jnp.float32)
    wl = jnp.full(3, 1.5, jnp.float32)
    a = np.asarray(omj.trace_sca(payload, xs, ys, wl, mode="local"))
    b = np.asarray(omj.trace_sca(payload, xs, ys, wl, mode="legacy"))
    assert np.abs(a - b).max() < 1e-2
    jac = jax.jacobian(lambda x, y: jnp.stack(omj.trace_sca(payload, x, y, 1.5)),
                       argnums=(0, 1))(jnp.float32(2000.0), jnp.float32(1500.0))
    J = np.asarray(jnp.stack(jac, -1))
    assert np.all(np.isfinite(J)) and abs(np.linalg.det(J)) > 0.5


def test_mode_env(monkeypatch, payload):
    monkeypatch.setenv("ROMAN_DISPERSER_POSITIONS", "legacy")
    assert omj.position_mode() == "legacy"
    monkeypatch.setenv("ROMAN_DISPERSER_POSITIONS", "bogus")
    with pytest.raises(ValueError):
        omj.position_mode()
    monkeypatch.delenv("ROMAN_DISPERSER_POSITIONS")
    assert omj.position_mode() == "local"


def test_sca_pos_f64_matches_float64_chain(payload):
    """get_sca_pos_f64 vs a float64 evaluation of get_fpa_pos + fpa_to_sca,
    and vs the float32 production route (~1e-3 px)."""
    rng = np.random.default_rng(3)
    pra, pdec, ppa = 10.195391, 0.102452, 60.0
    ra = pra + rng.uniform(-0.3, 0.3, 200)
    dec = pdec + rng.uniform(-0.3, 0.3, 200)
    xs, ys = omj.get_sca_pos_f64(ra, dec, pra, pdec, ppa, payload)
    assert xs.dtype == np.float64
    xi, eta = omj.sky_to_tangent_offsets(ra, dec, pra, pdec)
    th = np.deg2rad(ppa + 180.0 + omj.FOCAL_PA_DEG)
    xf = -(np.cos(th) * xi - np.sin(th) * eta)
    yf = -(np.sin(th) * xi + np.cos(th) * eta)
    h = payload["host"]
    xr = -(xf * 3600.0 / h["pixel_scale"] - h["xy_center"][0] * h["plate_scale"]) + h["crpix1"]
    yr = (yf * 3600.0 / h["pixel_scale"] - h["xy_center"][1] * h["plate_scale"]) + h["crpix2"]
    np.testing.assert_allclose(xs, xr, atol=1e-9)
    np.testing.assert_allclose(ys, yr, atol=1e-9)
    xf32, yf32 = omj.get_fpa_pos(ra, dec, pra, pdec, ppa)
    x32, y32 = map(np.asarray, omj.fpa_to_sca(payload, xf32, yf32))
    assert np.abs(x32 - xs).max() < 0.02 and np.abs(y32 - ys).max() < 0.02

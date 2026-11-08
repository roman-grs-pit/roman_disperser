"""Position contractions: every mode evaluates the same polynomials.

``optical_model_jax.POSITION_CONTRACTION`` selects how the small position
contractions are written ("Position contractions" in that module's
docstring). The GEMM-free modes exist so XLA:GPU has no autotuned matmul
kernel to choose per compile. All modes must reproduce a float64 evaluation of
the same formulas to float32 accuracy.

The float32 error is not one ulp of the result: the position polynomials sum
large, cancelling monomials, so the error scales with the largest term. The
measured worst cases over 140k random field positions are 2-3e-3 px for
einsum and 1.5e-3 px for Horner (RESULTS_det_impl.md); the tolerance below
leaves room above that and is still far below anything the image can see.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import disperser as dsp

PX_PER_MM = 100.0      # 10 um pixels
TOL_PX = 5e-3


@pytest.fixture
def restore_mode():
    old = omj.POSITION_CONTRACTION
    yield
    omj.set_position_contraction(old)


def _inputs(n=4000):
    rng = np.random.default_rng(3)
    dx = rng.uniform(-0.25, 0.25, n).astype(np.float32)
    dy = rng.uniform(-0.25, 0.25, n).astype(np.float32)
    wl = rng.uniform(0.95, 1.85, n).astype(np.float32)
    wlg = np.linspace(0.95, 1.85, 56).astype(np.float32)
    return dx, dy, wl, wlg


def _jax_positions(payload, dx, dy, wl, wlg):
    xf, yf = omj.get_fpa_pos_from_offsets(dx, dy, jnp.float32(60.0))
    xm, ym = omj.trace_beam(payload, xf, yf, jnp.asarray(wl))
    crv, ids = omj.get_trace_coeffs(payload, xf[:200], yf[:200])
    dely = dsp._eval_ids_polynomial(dsp._transform_wavelength(jnp.asarray(wlg), payload), ids)
    delx = dsp._eval_crv_polynomial(dely, crv)
    return [np.asarray(a, np.float64) for a in (xf, yf, xm, ym, dely, delx)]


def _f64_positions(payload, xf, yf, wl, wlg):
    """Float64 NumPy evaluation from the (float32) FPA positions and payload coefficients."""
    Q = {k: np.asarray(v, np.float64) for k, v in payload["poly"].items() if np.ndim(v) > 0}
    wl_ref = float(payload["wl"]["reference"])
    if payload["wl"]["transform"] == "linear":
        wlt = lambda w: w - wl_ref  # noqa: E731
    else:
        wlt = lambda w: np.log10(w / wl_ref)  # noqa: E731

    def poly2(x, y, C):
        return np.einsum("ni,ij,nj->n", x[:, None] ** np.arange(C.shape[0]), C,
                         y[:, None] ** np.arange(C.shape[1]))

    def coeffs(x, y, C):
        return np.einsum("nj,ijk,nk->in", x[:, None] ** np.arange(C.shape[1]), C,
                         y[:, None] ** np.arange(C.shape[2]))

    xm0, ym0 = poly2(xf, yf, Q["X_ij"]), poly2(xf, yf, Q["Y_ij"])
    crv, ids = coeffs(xf, yf, Q["C_ijk"]), coeffs(xf, yf, Q["D_ijk"])
    t = wlt(wl.astype(np.float64))
    dely = np.einsum("ni,in->n", t[:, None] ** np.arange(ids.shape[0]), ids)
    delx = np.einsum("ni,in->n", dely[:, None] ** np.arange(crv.shape[0]), crv)

    crv_g, ids_g = crv[:, :200], ids[:, :200]
    tg = wlt(wlg.astype(np.float64))
    dely_g = np.einsum("wi,is->sw", tg[:, None] ** np.arange(ids_g.shape[0]), ids_g)
    delx_g = np.einsum("swi,is->sw", dely_g[:, :, None] ** np.arange(crv_g.shape[0]), crv_g)
    return [xm0 + delx, ym0 + dely, dely_g, delx_g]


@pytest.mark.parametrize("mode", omj.POSITION_CONTRACTION_MODES)
def test_mode_matches_float64(payload, mode, restore_mode):
    omj.set_position_contraction(mode)
    dx, dy, wl, wlg = _inputs()
    xf, yf, *got = _jax_positions(payload, dx, dy, wl, wlg)

    # The 2x2 rotation: two products and an add, exact to a float32 ulp
    # (3e-8 deg at 0.25 deg, ~1e-3 px)
    R = np.asarray(omj.get_pa_rotation(pa=np.float32(60.0)), np.float64)
    xy = R @ np.stack([dx, dy]).astype(np.float64)
    np.testing.assert_allclose(xf, -xy[0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(yf, -xy[1], rtol=0, atol=1e-7)

    ref = _f64_positions(payload, xf, yf, wl, wlg)
    for name, a, b in zip(("xmpa", "ympa", "dely_grid", "delx_grid"), got, ref):
        err_px = np.max(np.abs(a - b)) * PX_PER_MM
        assert err_px < TOL_PX, f"{mode} {name}: {err_px:.2e} px"


def test_modes_agree(payload, restore_mode):
    """GEMM-free modes and einsum agree within twice the float32 tolerance."""
    dx, dy, wl, wlg = _inputs()
    out = {}
    for mode in omj.POSITION_CONTRACTION_MODES:
        omj.set_position_contraction(mode)
        out[mode] = _jax_positions(payload, dx, dy, wl, wlg)
    for mode in omj.POSITION_CONTRACTION_MODES:
        for a, b in zip(out[mode][2:], out["einsum"][2:]):
            assert np.max(np.abs(a - b)) * PX_PER_MM < 2 * TOL_PX


def test_set_position_contraction_rejects_unknown_mode(restore_mode):
    with pytest.raises(ValueError):
        omj.set_position_contraction("matmul")

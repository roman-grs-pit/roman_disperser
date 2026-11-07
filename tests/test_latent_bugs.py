"""Fixes for the latent bugs confirmed on the cluster (T18, T35).

T18: (a) ``get_beam_trace_wl`` read keys that do not exist; (b) an order
listed without coefficients ("m1" in v0.8) got 0-d arrays and failed deep in
einsum; (c) ``match_pos_to_sca`` raised on any vector input.
T35: Sersic parameter gradients were NaN for odd npix (the r = 0 pixel).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import paths, sersic
from roman_disperser.optical_model import RomanOpticalModel


@pytest.fixture(scope="module")
def model():
    return RomanOpticalModel(str(paths.optical_model_path()))


def test_get_beam_trace_wl_inverts_the_trace(model):
    xf, yf = np.array([0.1]), np.array([0.1])
    coeff = model._get_beam_trace(xf, yf, sca=1, width=25)
    k = len(model.wl_grid) // 2
    x, y = coeff["trace_sca_x"][12, k], coeff["trace_sca_y"][12, k]
    wl = model.get_beam_trace_wl(x, y, xf, yf, sca=1, width=25)
    np.testing.assert_allclose(wl, model.wl_grid[k], rtol=1e-6)


def test_order_without_coefficients_is_not_offered(model):
    assert "m1" in model.orders_defined
    assert "m1" not in model.beam_coeffs
    with pytest.raises(ValueError, match="m1"):
        omj.make_sca_payload(model, 1, "m1")


def test_match_pos_to_sca_vector(model):
    c = model.coords
    xs = np.array([c.get_sca_center(sca=s)[0] for s in (1, 5, 10)])
    ys = np.array([c.get_sca_center(sca=s)[1] for s in (1, 5, 10)])
    np.testing.assert_array_equal(c.match_pos_to_sca(xs, ys), [1, 5, 10])
    assert c.match_pos_to_sca(xs[1], ys[1]) == 5
    # far outside every SCA: 0, not the "closest" SCA
    assert c.match_pos_to_sca(500.0, 500.0) == 0
    # sca=None callers need one SCA
    with pytest.raises(ValueError, match="several SCAs"):
        c._single_sca(xs, ys)
    with pytest.raises(ValueError, match="chip gap"):
        c._single_sca(500.0, 500.0)
    assert c._single_sca(xs[[1, 1]], ys[[1, 1]]) == 5


def _old_kernel(r_eff_pix, n, ba, theta, npix):
    """The pre-fix formula, for the bit-for-bit check off r = 0."""
    center = (npix - 1) / 2.0
    y, x = jnp.mgrid[0:npix, 0:npix] - center
    n = jnp.clip(n, 0.01, 10.0)
    ba = jnp.clip(ba, 0.01, 1.0)
    r_eff_pix = jnp.clip(r_eff_pix, 0.1, None)
    xr = x * jnp.cos(theta) + y * jnp.sin(theta)
    yr = -x * jnp.sin(theta) + y * jnp.cos(theta)
    r = jnp.sqrt(xr ** 2 + (yr / ba) ** 2)
    p = jnp.exp(-sersic.compute_bn(n) * ((r / r_eff_pix) ** (1.0 / n) - 1.0))
    return p / p.sum()


@pytest.mark.parametrize("npix", [120, 121])
def test_sersic_gradient_finite(npix):
    rng = np.random.default_rng(3)
    W = jnp.asarray(rng.normal(size=(npix, npix)), jnp.float32)
    f = lambda p: jnp.sum(sersic.make_sersic_image(p[0], p[1], p[2], p[3], npix) * W)
    g = jax.grad(f)(jnp.array([8.0, 1.5, 0.6, 0.3]))
    assert np.all(np.isfinite(np.asarray(g)))
    img = sersic.make_sersic_image(8.0, 1.5, 0.6, 0.3, npix)
    assert np.all(np.isfinite(np.asarray(img)))
    old = _old_kernel(8.0, 1.5, 0.6, 0.3, npix)
    np.testing.assert_allclose(np.asarray(img), np.asarray(old), rtol=1e-6)
    if npix % 2 == 0:
        np.testing.assert_array_equal(np.asarray(img), np.asarray(old))

"""Fixes for the latent bugs confirmed on the cluster (T18).

T18: (a) ``get_beam_trace_wl`` read keys that do not exist; (b) an order
listed without coefficients ("m1" in v0.8) got 0-d arrays and failed deep in
einsum; (c) ``match_pos_to_sca`` raised on any vector input.
"""

import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import paths
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

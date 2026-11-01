"""Tests for the experimental matrix-free operator (``linear_operator``).

Mock PSF (2x2 nodes, 5 wavelengths, 20x20 os-4 stamps) + the real grism
optical model of SCA 5, so they run in the fast suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import galaxy_disperser, linear_operator as lo, star_disperser
from roman_disperser.pipeline import make_batched_galaxy_fori, make_batched_star_fori

N_WL = 41
WL = jnp.linspace(1.0, 1.8, N_WL).astype(jnp.float32)
DLAM = 200.0  # A, matches the grid spacing (0.02 um)
STARS = np.array([[1000.3, 900.55], [1003.71, 905.2]], np.float32)
GALS = np.array([[1001.12, 902.9]], np.float32)


def _mock_psf(n_wl=5, size=20):
    # Smooth (Gaussian, width varying with node and wavelength) so that
    # finite differences in position are well conditioned.
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    sig = 2.5 + 0.4 * np.arange(n_wl)[None, None, :] + 0.3 * np.arange(2)[:, None, None]
    grid = np.exp(-0.5 * (xx ** 2 + (1.2 * yy) ** 2) / sig[..., None, None] ** 2)
    grid = np.broadcast_to(grid, (2, 2, n_wl, size, size)).astype(np.float32)
    grid = grid / grid.sum(axis=(-2, -1), keepdims=True)
    return {
        'psf_grid': jnp.asarray(grid),
        'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
        'spatial_x': jnp.linspace(1, 4088, 2),
        'spatial_y': jnp.linspace(1, 4088, 2),
        'oversample': 4,
    }


def _galaxy_image():
    return jnp.zeros((16, 16), jnp.float32).at[6:10, 5:11].set(1.0 / 24)


@pytest.fixture(scope="module")
def setup():
    from roman_disperser.optical_model import RomanOpticalModel
    from roman_disperser import elements, paths
    model = RomanOpticalModel(str(paths.optical_model_path(
        element=elements.get_element("grism"))))
    psf = _mock_psf()
    opt = omj.make_sca_payload(model, sca=5, order="1")
    sens = jnp.linspace(0.5, 1.5, N_WL).astype(jnp.float32)
    return psf, opt, sens


def _operator(setup, rebin=1, basis=None, precompute=True, checkpoint=False,
              deposit="interp", window="auto"):
    psf, opt, sens = setup
    orders = {"1": lo.make_order_setup(psf, opt, sens, WL, rebin=rebin)}
    xy = np.concatenate([STARS, GALS])
    if window == "auto":
        window = lo.auto_window(orders, xy, margin=12)
    return lo.SceneOperator(
        orders, WL, DLAM, star_xy=STARS, galaxy_xy=GALS,
        galaxy_images=_galaxy_image()[None], window=window, basis=basis,
        flux_unit=1.0, precompute=precompute, checkpoint=checkpoint,
        deposit=deposit)


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.1, 1.0, shape),
                       jnp.float32)


def test_linearity(setup):
    A = _operator(setup)
    x, y = _rand(A.theta_shape, 0), _rand(A.theta_shape, 1)
    lhs = np.asarray(A.forward(2.0 * x - 0.5 * y), np.float64)
    rhs = 2.0 * np.asarray(A.forward(x), np.float64) - 0.5 * np.asarray(
        A.forward(y), np.float64)
    np.testing.assert_allclose(lhs, rhs, atol=1e-5 * np.abs(rhs).max())
    assert np.abs(rhs).max() > 0


@pytest.mark.parametrize("rebin,basis_dA", [(1, None), (2, None), (1, 600.0)])
def test_adjoint_dot_product(setup, rebin, basis_dA):
    basis = None if basis_dA is None else lo.LinearBasis(WL, basis_dA)
    A = _operator(setup, rebin=rebin, basis=basis)
    x = _rand(A.theta_shape, 2)
    y = _rand(A.shape, 3)
    ax = np.asarray(A.forward(x), np.float64)
    aty = np.asarray(A.transpose(y), np.float64)
    lhs = (ax * np.asarray(y, np.float64)).sum()
    rhs = (np.asarray(x, np.float64) * aty).sum()
    assert abs(lhs - rhs) <= 1e-5 * abs(lhs)
    # vjp transpose agrees with linear_transpose
    aty_vjp = np.asarray(A.transpose_vjp(y), np.float64)
    np.testing.assert_allclose(aty_vjp, aty, rtol=1e-5,
                               atol=1e-6 * np.abs(aty).max())


# Galaxy tolerance: the dispersed positions computed in a different jit
# context differ by up to 1 float32 ulp at x ~ 1000 (6e-5 px); with the
# sharp-edged mock galaxy the continuous interp deposit turns that into
# <= ~2e-4 of the peak. Stars agree to ~1e-7.
POS_ULP_ATOL = 5e-4


def test_precompute_checkpoint_and_window_equivalent(setup):
    x = _rand((3, N_WL), 4)
    a = _operator(setup)
    ref = np.asarray(a.forward(x), np.float64)
    for kw in (dict(precompute=False), dict(checkpoint=True)):
        b = np.asarray(_operator(setup, **kw).forward(x), np.float64)
        np.testing.assert_allclose(b, ref, atol=POS_ULP_ATOL * ref.max())
        np.testing.assert_allclose(b.sum(), ref.sum(), rtol=1e-5)
    full = np.asarray(_operator(setup, window=None).forward(x), np.float64)
    y0, x0, ny, nx = a.window
    np.testing.assert_allclose(full[y0:y0 + ny, x0:x0 + nx], ref,
                               atol=1e-6 * ref.max())
    np.testing.assert_allclose(full.sum(), ref.sum(), rtol=1e-6)


@pytest.mark.parametrize("rebin", [1, 2])
def test_matches_production_batch_path(setup, rebin):
    psf, opt, sens = setup
    x = _rand((3, N_WL), 5)
    A = _operator(setup, rebin=rebin, window=None)
    ours = np.asarray(A.forward(x), np.float64)
    z = jnp.zeros((4088, 4088), jnp.float32)
    sfn = star_disperser.make_star_disperser(psf, opt, deposit="interp",
                                             hoist_psf=True)
    gfn = galaxy_disperser.make_galaxy_disperser(psf, opt, deposit="interp")
    img = make_batched_star_fori(sfn, sens, WL, DLAM, sed_rebin=rebin)(
        2, x[:2], jnp.asarray(STARS[:, 0]), jnp.asarray(STARS[:, 1]), z)
    img = make_batched_galaxy_fori(gfn, sens, WL, DLAM, sed_rebin=rebin)(
        1, x[2:], jnp.asarray(GALS[:, 0]), jnp.asarray(GALS[:, 1]),
        _galaxy_image()[None], img)
    prod = np.asarray(img, np.float64)
    np.testing.assert_allclose(ours, prod, atol=POS_ULP_ATOL * prod.max())
    np.testing.assert_allclose(ours.sum(), prod.sum(), rtol=1e-5)


def test_basis_reproduces_hat_functions():
    b = lo.LinearBasis(WL, 600.0)
    M = b.matrix()
    assert M.shape == (N_WL, b.n)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-6)  # partition of unity
    c = np.arange(b.n, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(b(jnp.asarray(c))), M @ c, atol=1e-5)


def test_gradients_vs_finite_differences(setup):
    A = _operator(setup, precompute=False)
    f = A.render_fn()
    theta = _rand(A.theta_shape, 6)
    # Smooth weight field: the interp model is piecewise linear in position
    # with kinks every 1/os px, so FD of a pixel-noise W is ill-conditioned.
    from scipy import ndimage
    W = ndimage.gaussian_filter(np.random.default_rng(7).normal(size=A.shape), 2.0)
    W = jnp.asarray(W / W.std(), jnp.float32)
    loss = jax.jit(lambda th, d: jnp.sum(f(th, d) * W))
    d0 = jnp.zeros((A.n_sources, 2), jnp.float32)
    g_th, g_d = jax.jit(jax.grad(loss, argnums=(0, 1)))(theta, d0)
    # spectra: loss is linear in theta, gradient = A^T W exactly
    # (up to the float32 position-ulp effect, see POS_ULP_ATOL; the pixel-
    # uncorrelated W makes A^T W sensitive to it).
    ref = np.asarray(A.transpose(W), np.float64)
    err = np.linalg.norm(np.asarray(g_th, np.float64) - ref) / np.linalg.norm(ref)
    assert err < 1e-3
    h = 0.02
    for s in range(A.n_sources):
        for ax in range(2):
            e = d0.at[s, ax].set(h)
            fd = (float(loss(theta, e)) - float(loss(theta, -e))) / (2 * h)
            assert abs(float(g_d[s, ax]) - fd) <= 0.03 * abs(fd) + 1e-6, (s, ax)


def test_floor_position_gradient_is_zero(setup):
    A = _operator(setup, precompute=False, deposit="floor")
    f = A.render_fn()
    theta = _rand(A.theta_shape, 6)
    W = _rand(A.shape, 7) - 0.55
    g = jax.grad(lambda d: jnp.sum(f(theta, d) * W))(
        jnp.zeros((A.n_sources, 2), jnp.float32))
    A2 = _operator(setup, precompute=False)
    g2 = jax.grad(lambda d: jnp.sum(A2.render_fn()(theta, d) * W))(
        jnp.zeros((A.n_sources, 2), jnp.float32))
    assert float(jnp.abs(g).max()) < 0.05 * float(jnp.abs(g2).max())

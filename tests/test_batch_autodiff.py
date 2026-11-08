"""Reverse-mode autodiff through the production batch loops.

``pipeline.run_star_batch`` / ``run_galaxy_batch`` loop over the static batch
length with a checkpointed ``lax.scan`` (padded rows skipped by ``lax.cond``),
so ``jax.grad`` works through them. Mock PSF + the real grism optical model
of SCA 5, as in ``test_linear_operator``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import (galaxy_disperser, jit_payload, pipeline,
                             star_disperser)
from roman_disperser.pipeline import (make_batched_galaxy_fori,
                                      make_batched_star_fori, rebin_counts)

N_WL = 41
WL = jnp.linspace(1.0, 1.8, N_WL).astype(jnp.float32)
DLAM = 200.0
# Two valid stars + one padded row; one valid galaxy + one padded row.
STAR_X = jnp.array([1000.3, 1003.71, 0.0], jnp.float32)
STAR_Y = jnp.array([900.55, 905.2, 0.0], jnp.float32)
GAL_X = jnp.array([1001.12, 0.0], jnp.float32)
GAL_Y = jnp.array([902.9, 0.0], jnp.float32)


def _mock_psf(n_wl=5, size=20):
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    sig = (2.5 + 0.4 * np.arange(n_wl)[None, None, :]
           + 0.3 * np.arange(2)[:, None, None])
    grid = np.exp(-0.5 * (xx ** 2 + (1.2 * yy) ** 2)
                  / sig[..., None, None] ** 2)
    grid = np.broadcast_to(grid, (2, 2, n_wl, size, size)).astype(np.float32)
    grid = grid / grid.sum(axis=(-2, -1), keepdims=True)
    return {'psf_grid': jnp.asarray(grid),
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
            'spatial_x': jnp.linspace(1, 4088, 2),
            'spatial_y': jnp.linspace(1, 4088, 2),
            'oversample': 4}


def _galaxy_images():
    img = jnp.zeros((16, 16), jnp.float32).at[5:11, 4:12].set(1.0 / 48)
    from scipy import ndimage
    img = jnp.asarray(ndimage.gaussian_filter(np.asarray(img), 1.0),
                      jnp.float32)
    return jnp.stack([img, jnp.zeros_like(img)])


@pytest.fixture(scope="module")
def setup():
    from roman_disperser import elements, paths
    from roman_disperser.optical_model import RomanOpticalModel
    model = RomanOpticalModel(str(paths.optical_model_path(
        element=elements.get_element("grism"))))
    psf = _mock_psf()
    opt = omj.make_sca_payload(model, sca=5, order="1")
    sens = jnp.linspace(0.5, 1.5, N_WL).astype(jnp.float32)
    return psf, opt, sens


def _rand(shape, seed, lo=0.1, hi=1.0):
    return jnp.asarray(np.random.default_rng(seed).uniform(lo, hi, shape),
                       jnp.float32)


@pytest.fixture(scope="module")
def weight(setup):
    """Smooth random weight over the bounding box of all traces (FD of a
    pixel-noise weight is ill-conditioned for the piecewise-linear interp
    deposit)."""
    from scipy import ndimage
    star_img, gal_img, _, _ = _batch_fns(setup)
    img = (np.asarray(star_img(_rand((3, N_WL), 0), STAR_X, STAR_Y))
           + np.asarray(gal_img(_rand((2, N_WL), 0), GAL_X, GAL_Y,
                                _galaxy_images())))
    ys, xs = np.nonzero(img)
    y0, y1 = ys.min() - 15, ys.max() + 16
    x0, x1 = xs.min() - 15, xs.max() + 16
    rng = np.random.default_rng(7)
    patch = ndimage.gaussian_filter(rng.normal(size=(y1 - y0, x1 - x0)), 2.0)
    w = np.zeros((4088, 4088), np.float32)
    w[y0:y1, x0:x1] = patch / patch.std()
    return jnp.asarray(w)


def _batch_fns(setup, rebin=1):
    psf, opt, sens = setup
    sd = star_disperser.make_star_disperser(psf, opt, deposit="interp",
                                            hoist_psf=True)
    gd = galaxy_disperser.make_galaxy_disperser(psf, opt, deposit="interp")
    sargs, gargs = sd.payload_args, gd.payload_args

    def star_img(spec, x, y, n=2):
        a, st, ch, dep = sargs
        return pipeline.run_star_batch(
            a, sens, WL, jnp.float32(DLAM), n, spec, x, y,
            jnp.zeros((4088, 4088), jnp.float32),
            static=st, chunk_size=ch, deposit=dep, sed_rebin=rebin)

    def gal_img(spec, x, y, images, n=1):
        a, st, ch, dep = gargs
        return pipeline.run_galaxy_batch(
            a, sens, WL, jnp.float32(DLAM), n, spec, x, y, images,
            jnp.zeros((4088, 4088), jnp.float32),
            static=st, chunk_size=ch, deposit=dep, sed_rebin=rebin)
    return star_img, gal_img, sd, gd


def _fori_reference(setup, kind, rebin=1):
    """The pre-scan ``run_*_batch`` (fori_loop over the traced n_sources),
    with the same jit arguments as the production function."""
    psf, opt, sens = setup
    if kind == "star":
        fn = star_disperser.make_star_disperser(psf, opt, deposit="interp",
                                                hoist_psf=True)
    else:
        fn = galaxy_disperser.make_galaxy_disperser(psf, opt,
                                                    deposit="interp")
    arrays, static, chunk, dep = fn.payload_args

    @jax.jit
    def run_args(arrays, sens, wavelengths, dlam, n, spec, x, y, images,
                 out):
        p, o = jit_payload.merge_payloads(arrays, static)
        _, wl_dep = rebin_counts(sens, wavelengths, rebin)

        def body(i, out):
            c, _ = rebin_counts(spec[i] * sens * dlam, wavelengths, rebin)
            if kind == "star":
                return star_disperser.disperse_star_psf(
                    p, o, x[i], y[i], wl_dep, c, out, chunk_size=chunk,
                    deposit=dep)
            return galaxy_disperser.disperse_galaxy(
                o, p, images[i], x[i], y[i], c, wl_dep, out, chunk, dep)
        return jax.lax.fori_loop(0, n, body, out)
    return lambda *args: run_args(arrays, sens, WL, jnp.float32(DLAM), *args)


@pytest.mark.parametrize("rebin", [1, 2])
def test_forward_bit_identical_to_fori_loop(setup, rebin):
    star_img, gal_img, _, _ = _batch_fns(setup, rebin)
    spec_s, spec_g = _rand((3, N_WL), 1), _rand((2, N_WL), 2)
    imgs = _galaxy_images()
    z = lambda: jnp.zeros((4088, 4088), jnp.float32)
    new_s = np.asarray(star_img(spec_s, STAR_X, STAR_Y))
    old_s = np.asarray(_fori_reference(setup, "star", rebin)(
        2, spec_s, STAR_X, STAR_Y, imgs, z()))
    new_g = np.asarray(gal_img(spec_g, GAL_X, GAL_Y, imgs))
    old_g = np.asarray(_fori_reference(setup, "gal", rebin)(
        1, spec_g, GAL_X, GAL_Y, imgs, z()))
    assert old_s.sum() > 0 and old_g.sum() > 0
    np.testing.assert_array_equal(new_s, old_s)
    np.testing.assert_array_equal(new_g, old_g)


def test_padded_rows_are_skipped(setup):
    """Rows >= n_sources contribute nothing (even with nonzero inputs)."""
    star_img, gal_img, _, _ = _batch_fns(setup)
    spec_s = _rand((3, N_WL), 3)
    x = STAR_X.at[2].set(1500.0)
    y = STAR_Y.at[2].set(1500.0)
    full = np.asarray(star_img(spec_s, x, y, n=3))
    two = np.asarray(star_img(spec_s, x, y, n=2))
    only = np.asarray(star_img(spec_s[:2], x[:2], y[:2], n=2))
    np.testing.assert_array_equal(two, only)
    assert full.sum() > two.sum() * 1.2


def test_grad_star_batch_vs_finite_differences(setup, weight):
    star_img, _, _, _ = _batch_fns(setup)
    spec = _rand((3, N_WL), 4)
    loss = jax.jit(lambda s, x, y: jnp.sum(star_img(s, x, y) * weight))
    gs, gx, gy = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(
        spec, STAR_X, STAR_Y)
    # padded row: zero gradient
    assert float(jnp.abs(gs[2]).max()) == 0.0
    assert float(gx[2]) == 0.0 and float(gy[2]) == 0.0
    # spectra: the loss is linear, FD along a random direction is exact up
    # to float32 rounding
    v = _rand((3, N_WL), 5, -1.0, 1.0).at[2].set(0.0)
    fd = (float(loss(spec + v, STAR_X, STAR_Y))
          - float(loss(spec - v, STAR_X, STAR_Y))) / 2.0
    ad = float(jnp.vdot(gs, v, precision="highest"))
    assert abs(ad - fd) <= 1e-3 * abs(fd)
    # positions: reverse mode == forward mode through the old fori_loop
    ref = _fori_reference(setup, "star")
    z = jnp.zeros((4088, 4088), jnp.float32)
    imgs = _galaxy_images()
    loss_ref = lambda x, y: jnp.sum(ref(2, spec, x, y, imgs, z) * weight)
    for s in range(2):
        e = jnp.zeros(3, jnp.float32).at[s].set(1.0)
        _, tx = jax.jvp(lambda x: loss_ref(x, STAR_Y), (STAR_X,), (e,))
        _, ty = jax.jvp(lambda y: loss_ref(STAR_X, y), (STAR_Y,), (e,))
        assert abs(float(gx[s]) - float(tx)) <= 1e-4 * abs(float(tx))
        assert abs(float(gy[s]) - float(ty)) <= 1e-4 * abs(float(ty))
    # ... and agrees with finite differences. The FD losses are summed in
    # float64 on the host (the float32 jnp.sum over 4088^2 pixels is
    # quantised at ~1 count). Even so the one-sided slopes scatter with the
    # step (y: 180-250 for h = 0.001-0.1, both position modes, CPU and L40S,
    # job 32478582): the float32 sample positions move in ulp steps and the
    # interp deposit is piecewise linear with kinks wherever a subpixel edge
    # crosses a pixel edge. AD (== forward mode, checked above) must lie
    # within the range of the one-sided slopes over h = 0.003-0.03 (3 % margin).
    w64 = np.asarray(weight, np.float64)
    img = jax.jit(star_img)
    for s in range(2):
        for g, which in ((gx, 0), (gy, 1)):
            def l(d):
                e = jnp.zeros(3, jnp.float32).at[s].set(d)
                out = (img(spec, STAR_X + e, STAR_Y) if which == 0
                       else img(spec, STAR_X, STAR_Y + e))
                return float(np.sum(np.asarray(out, np.float64) * w64))
            l0 = l(0.0)
            slopes = []
            for h in (0.003, 0.01, 0.03):
                slopes += [(l0 - l(-h)) / h, (l(h) - l0) / h]
            lo, hi = min(slopes), max(slopes)
            pad = 0.03 * max(abs(lo), abs(hi))
            assert lo - pad <= float(g[s]) <= hi + pad, (s, which, slopes,
                                                         float(g[s]))


def test_grad_galaxy_batch_vs_finite_differences(setup, weight):
    _, gal_img, _, _ = _batch_fns(setup)
    spec = _rand((2, N_WL), 6)
    imgs = _galaxy_images()
    loss = jax.jit(lambda s, x, y: jnp.sum(gal_img(s, x, y, imgs) * weight))
    gs, gx, gy = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(spec, GAL_X, GAL_Y)
    assert float(jnp.abs(gs[1]).max()) == 0.0
    assert float(gx[1]) == 0.0 and float(gy[1]) == 0.0
    v = _rand((2, N_WL), 7, -1.0, 1.0).at[1].set(0.0)
    fd = (float(loss(spec + v, GAL_X, GAL_Y))
          - float(loss(spec - v, GAL_X, GAL_Y))) / 2.0
    ad = float(jnp.vdot(gs, v, precision="highest"))
    assert abs(ad - fd) <= 1e-3 * abs(fd)
    h = 0.02
    e = jnp.zeros(2, jnp.float32).at[0].set(h)
    fdx = (float(loss(spec, GAL_X + e, GAL_Y))
           - float(loss(spec, GAL_X - e, GAL_Y))) / (2 * h)
    fdy = (float(loss(spec, GAL_X, GAL_Y + e))
           - float(loss(spec, GAL_X, GAL_Y - e))) / (2 * h)
    assert abs(fdx) > 0 and abs(fdy) > 0
    assert abs(float(gx[0]) - fdx) <= 0.05 * abs(fdx) + 1e-6
    assert abs(float(gy[0]) - fdy) <= 0.05 * abs(fdy) + 1e-6


def test_factory_batch_still_donates(setup):
    """The batched entry points keep donating ``output``."""
    _, _, sd, gd = _batch_fns(setup)
    psf, opt, sens = setup
    run = make_batched_star_fori(sd, sens, WL, DLAM)
    out = jnp.zeros((4088, 4088), jnp.float32)
    new = run(2, _rand((3, N_WL), 8), STAR_X, STAR_Y, out)
    assert out.is_deleted()
    run_g = make_batched_galaxy_fori(gd, sens, WL, DLAM)
    new2 = run_g(1, _rand((2, N_WL), 9), GAL_X, GAL_Y, _galaxy_images(), new)
    assert new.is_deleted() and float(new2.sum()) > 0

"""
Tests for the trace-aligned segment render (``segment_render``).

The segment render regroups the production deposit per wavelength node:
with the hat weights of the deposit's own wavelength interpolation, the
sum over samples of shifted stamps equals the sum over nodes of
(line image) * (node stamp). So:

- with every PSF grid wavelength as a node it must reproduce the
  production ``interp`` deposit;
- with a subset of nodes it must reproduce the ``interp`` deposit of the
  stack restricted to those nodes (same piecewise-linear wavelength model);
- with ``psf_at="landing"`` it must reproduce the deposit of the stack whose
  node k is the PSF interpolated at the trace position of node k.

All comparisons use the same float32 dispersed positions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import galaxy_disperser, psf_model, segment_render as sr
from roman_disperser import star_disperser
from roman_disperser.pipeline import make_batched_star_fori

N_GRID = 6
WL = jnp.linspace(1.0, 1.8, 161).astype(jnp.float32)
X0, Y0 = jnp.float32(1500.37), jnp.float32(1800.81)
ZERO = jnp.zeros((4088, 4088), jnp.float32)


def _mock_psf(n_wl=N_GRID, size=20):
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    sig = (1.5 + 0.5 * np.arange(n_wl)[None, None, :]
           + 0.4 * np.arange(2)[:, None, None])
    grid = np.exp(-0.5 * (xx ** 2 + (1.3 * yy) ** 2)
                  / sig[..., None, None] ** 2)
    grid = np.broadcast_to(grid, (2, 2, n_wl, size, size)).copy()
    grid[1] *= 1.0 + 0.2 * np.sin(xx / 3.0)  # spatial variation in y
    grid = (grid / grid.sum(axis=(-2, -1), keepdims=True)).astype(np.float32)
    return {'psf_grid': jnp.asarray(grid),
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl).astype(jnp.float32),
            'spatial_x': jnp.linspace(1, 4088, 2).astype(jnp.float32),
            'spatial_y': jnp.linspace(1, 4088, 2).astype(jnp.float32),
            'oversample': 4}


def _galaxy_image(n=16):
    from scipy import ndimage
    img = np.zeros((n, n), np.float32)
    img[5:11, 4:12] = 1.0
    img = ndimage.gaussian_filter(img, 1.0)
    return jnp.asarray(img / img.sum(), jnp.float32)


@pytest.fixture(scope="module")
def payloads():
    from roman_disperser import elements, paths
    from roman_disperser.optical_model import RomanOpticalModel
    model = RomanOpticalModel(str(paths.optical_model_path(
        element=elements.get_element("grism"))))
    return _mock_psf(), omj.make_sca_payload(model, sca=5, order="1")


@pytest.fixture(scope="module")
def flux():
    rng = np.random.default_rng(3)
    return jnp.asarray(rng.uniform(0.5, 2.0, WL.size), jnp.float32)


@pytest.fixture(scope="module")
def dispersed(payloads):
    _, opt = payloads
    return jax.jit(lambda x, y: star_disperser._compute_dispersed_positions(
        opt, x, y, WL))(X0, Y0)


def _np(a):
    return np.asarray(a, np.float64)


def _deposit(stack, grid_wl, flux, dispersed):
    xd, yd = dispersed
    return _np(star_disperser.deposit_stack_native(
        stack, grid_wl, WL, flux, xd, yd, ZERO, oversample=4,
        chunk_size=2000, deposit="interp"))


def _assert_close(a, b, tol=2e-5):
    assert b.max() > 0
    np.testing.assert_allclose(a, b, rtol=0, atol=tol * b.max())


class TestPlan:

    def test_node_indices(self):
        assert sr.segment_node_indices(56, 56) == tuple(range(56))
        assert sr.segment_node_indices(56, 100) == tuple(range(56))
        assert sr.segment_node_indices(56, 1) == (0, 55)
        idx = sr.segment_node_indices(56, 12)
        assert idx[0] == 0 and idx[-1] == 55 and len(idx) == 12
        assert all(isinstance(i, int) for i in idx)

    def test_plan_shapes_and_errors(self, payloads):
        psf, opt = payloads
        plan = sr.make_segment_plan(psf, opt, 3, "star")
        hash(plan)
        assert plan.n_nodes == 3 and plan.kernel_shape == (20, 20)
        for n_line, n_fft, n_out in zip(plan.line_shape, plan.fft_shape,
                                        plan.out_shape):
            assert n_line % 4 == 0
            assert n_fft >= n_line + 19
            assert 4 * n_out >= n_line + 19
        g = sr.make_segment_plan(psf, opt, 3, "galaxy", galaxy_npix=16)
        assert g.kernel_shape == (35, 35)
        with pytest.raises(ValueError):
            sr.make_segment_plan(psf, opt, 3, "galaxy")
        with pytest.raises(ValueError):
            sr.make_segment_plan(psf, opt, 3, "quasar")
        with pytest.raises(ValueError):
            sr.make_segment_plan(psf, opt, 3, "star", psf_at="pupil")


class TestStar:

    def _seg(self, payloads, n_nodes, flux, dispersed, psf_at="source"):
        psf, opt = payloads
        plan = sr.make_segment_plan(psf, opt, n_nodes, "star", psf_at=psf_at)
        fn = jax.jit(lambda f: sr.render_star_segments(
            psf, opt, X0, Y0, WL, f, ZERO, plan, dispersed=dispersed))
        return _np(fn(flux)), plan

    def test_all_nodes_equal_interp_deposit(self, payloads, flux, dispersed):
        psf, _ = payloads
        seg, _ = self._seg(payloads, N_GRID, flux, dispersed)
        stack = psf_model.interpolate_psf_spatial(psf, X0, Y0)
        ref = _deposit(stack, psf['wavelengths'], flux, dispersed)
        _assert_close(seg, ref)
        assert abs(seg.sum() / ref.sum() - 1) < 1e-5

    @pytest.mark.parametrize("n_nodes", [2, 4])
    def test_node_subset_equals_deposit_of_subset(self, payloads, flux,
                                                  dispersed, n_nodes):
        psf, _ = payloads
        seg, plan = self._seg(payloads, n_nodes, flux, dispersed)
        idx = jnp.asarray(plan.node_idx)
        stack = psf_model.interpolate_psf_spatial(psf, X0, Y0)[idx]
        ref = _deposit(stack, psf['wavelengths'][idx], flux, dispersed)
        _assert_close(seg, ref)

    def test_landing_lookup(self, payloads, flux, dispersed):
        psf, opt = payloads
        seg, plan = self._seg(payloads, N_GRID, flux, dispersed, "landing")
        node_wl = psf['wavelengths'][jnp.asarray(plan.node_idx)]
        nx, ny = star_disperser._compute_dispersed_positions(
            opt, X0, Y0, node_wl)
        stack = jnp.stack([psf_model.interpolate_psf_spatial(
            psf, nx[k], ny[k])[k] for k in range(N_GRID)])
        ref = _deposit(stack, node_wl, flux, dispersed)
        _assert_close(seg, ref)
        src, _ = self._seg(payloads, N_GRID, flux, dispersed, "source")
        assert np.abs(seg - src).max() > 1e-3 * src.max()  # lookup matters

    def test_flux_conserved(self, payloads, flux, dispersed):
        seg, _ = self._seg(payloads, 2, flux, dispersed)
        psf = payloads[0]
        stack = psf_model.interpolate_psf_spatial(psf, X0, Y0)
        # every sample's stamp sums to its PSF sum (~1), so the image sum is
        # sum(flux) up to the wavelength interpolation of the stamp sums
        assert abs(seg.sum() / float(flux.sum()) - 1) < 1e-4
        assert np.isclose(float(stack.sum(axis=(1, 2)).mean()), 1.0, atol=1e-5)

    def test_batch_matches_single_and_skips_padding(self, payloads, flux):
        psf, opt = payloads
        plan = sr.make_segment_plan(psf, opt, 3, "star")
        sens = jnp.ones_like(WL)
        run = sr.make_batched_star_segment(psf, opt, sens, WL, 2.0, plan)
        xs = jnp.array([1000.3, 2044.71, 7.0], jnp.float32)
        ys = jnp.array([900.55, 2044.2, 7.0], jnp.float32)
        spectra = jnp.stack([flux, 0.5 * flux, flux])
        out = _np(run(2, spectra, xs, ys, jnp.zeros((4088, 4088),
                                                    jnp.float32)))
        ref = np.zeros_like(out)
        for i in range(2):
            ref += _np(sr.render_star_segments(
                psf, opt, xs[i], ys[i], WL, spectra[i] * 2.0, ZERO, plan))
        # Batched (jitted) and single (eager) renders compute the trace in
        # differently compiled programs, so positions differ by ~1 ulp,
        # which the sharp mock PSF shows at the 1e-3 level; plumbing errors
        # (padding, counts = spectra * sens * dlam) would be O(1).
        _assert_close(out, ref, 5e-3)
        assert abs(out.sum() / ref.sum() - 1) < 1e-5
        # same as the production batch path with the node subset's
        # piecewise-linear wavelength model when every grid node is used
        plan_all = sr.make_segment_plan(psf, opt, N_GRID, "star")
        run_all = sr.make_batched_star_segment(psf, opt, sens, WL, 2.0,
                                               plan_all)
        prod = make_batched_star_fori(
            star_disperser.make_star_disperser(psf, opt), sens, WL, 2.0)
        a = _np(run_all(2, spectra, xs, ys, jnp.zeros((4088, 4088),
                                                      jnp.float32)))
        b = _np(prod(2, spectra, xs, ys, jnp.zeros((4088, 4088),
                                                   jnp.float32)))
        _assert_close(a, b, 5e-3)  # positions may differ by ~1 ulp
        assert abs(a.sum() / b.sum() - 1) < 1e-5

    def test_gradients(self, payloads, flux):
        """Reverse mode through the batch function vs central differences:
        spectra (linear, tight) and position (piecewise linear deposit;
        smooth weight)."""
        psf, opt = payloads
        plan = sr.make_segment_plan(psf, opt, 3, "star")
        sens = jnp.ones_like(WL)
        arrays, static = sr.jit_payload.split_payloads(psf, opt)
        img0 = _np(sr.render_star_segments(psf, opt, X0, Y0, WL, flux * 2.0,
                                           ZERO, plan))
        ys_, xs_ = np.nonzero(img0 > 1e-3 * img0.max())
        y0, y1 = ys_.min() - 5, ys_.max() + 6
        x0, x1 = xs_.min() - 5, xs_.max() + 6
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wgt = np.zeros((4088, 4088), np.float32)
        wgt[y0:y1, x0:x1] = np.cos(xx / 7.0) * np.sin(yy / 11.0) + 1.5
        wgt = jnp.asarray(wgt)

        def loss(spec, x, y):
            out = sr.run_star_batch_segment(
                arrays, sens, WL, jnp.float32(2.0), 1, spec[None],
                x[None], y[None], jnp.zeros((4088, 4088), jnp.float32),
                static=static, plan=plan)
            return jnp.sum(out * wgt)

        g_spec, g_x, g_y = jax.grad(loss, argnums=(0, 1, 2))(flux, X0, Y0)
        # spectrum: linear -> FD exact up to float32
        e = np.zeros(WL.size, np.float32)
        e[80] = 1.0
        h = 0.5
        fd = (float(loss(flux + h * e, X0, Y0))
              - float(loss(flux - h * e, X0, Y0))) / (2 * h)
        assert np.isclose(float(g_spec[80]), fd, rtol=2e-3)
        for g, args in ((g_x, lambda d: (flux, X0 + d, Y0)),
                        (g_y, lambda d: (flux, X0, Y0 + d))):
            d = 0.05
            fd = (float(loss(*args(d))) - float(loss(*args(-d)))) / (2 * d)
            assert np.isfinite(float(g)) and abs(fd) > 0
            assert np.isclose(float(g), fd, rtol=0.05), (float(g), fd)


class TestGalaxy:

    def test_all_nodes_equal_production(self, payloads, flux, dispersed):
        psf, opt = payloads
        img = _galaxy_image()
        plan = sr.make_segment_plan(psf, opt, N_GRID, "galaxy",
                                    galaxy_npix=16)
        seg = _np(jax.jit(lambda f: sr.render_galaxy_segments(
            psf, opt, img, X0, Y0, f, WL, ZERO, plan,
            dispersed=dispersed))(flux))
        ref = _np(jax.jit(lambda f: galaxy_disperser.disperse_galaxy(
            opt, psf, img, X0, Y0, f, WL, ZERO,
            dispersed=dispersed))(flux))
        _assert_close(seg, ref)

    def test_batch_and_node_batch(self, payloads, flux):
        psf, opt = payloads
        img = _galaxy_image()
        plan = sr.make_segment_plan(psf, opt, 3, "galaxy", galaxy_npix=16)
        plan_b = plan._replace(node_batch=2)
        a = _np(sr.render_galaxy_segments(psf, opt, img, X0, Y0, flux, WL,
                                          ZERO, plan))
        b = _np(sr.render_galaxy_segments(psf, opt, img, X0, Y0, flux, WL,
                                          ZERO, plan_b))
        _assert_close(b, a, 1e-6)
        run = sr.make_batched_galaxy_segment(psf, opt, jnp.ones_like(WL), WL,
                                             1.0, plan)
        out = _np(run(1, jnp.stack([flux, flux]),
                      jnp.array([X0, 0.0]), jnp.array([Y0, 0.0]),
                      jnp.stack([img, img]),
                      jnp.zeros((4088, 4088), jnp.float32)))
        _assert_close(out, a, 5e-3)  # ~1 ulp position differences, as above
        assert abs(out.sum() / a.sum() - 1) < 1e-5
        assert abs(out.sum() / float(flux.sum()) - 1) < 1e-4


class TestDriverOptions:
    """Opt-in wiring in scripts/build_dispersed_image.py (off by default)."""

    @pytest.fixture(scope="class")
    def bdi(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).parent.parent / "scripts" / "build_dispersed_image.py"
        spec = importlib.util.spec_from_file_location("bdi_segment", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parse_and_resolve(self, bdi):
        orders = ("0", "1", "2")
        assert bdi.parse_segment_render(None) is None
        assert bdi.parse_segment_render("none") is None
        assert bdi.parse_segment_render("12") == 12
        assert bdi.parse_segment_render("0:6, 1:12,2:none") == {
            "0": 6, "1": 12, "2": None}
        assert bdi.resolve_segment_render(None, orders) is None
        assert bdi.resolve_segment_render({"2": None}, orders) is None
        r = bdi.resolve_segment_render(12, orders, "landing")
        assert r == {"nodes": {"0": 12, "1": 12, "2": 12}, "psf_at": "landing"}
        r = bdi.resolve_segment_render({0: 6}, orders)
        assert r["nodes"] == {"0": 6, "1": None, "2": None}
        for bad in ({"3": 6}, 1):
            with pytest.raises(ValueError):
                bdi.resolve_segment_render(bad, orders)
        with pytest.raises(ValueError):
            bdi.resolve_segment_render(6, orders, "pupil")

    def test_default_off(self, bdi):
        import inspect
        for fn in (bdi.setup_pipeline, bdi.build_dispersed_image):
            p = inspect.signature(fn).parameters
            assert p["segment_render"].default is None
            assert p["segment_psf_at"].default == "source"
        assert inspect.signature(bdi.make_order_fori_fns).parameters[
            "segment"].default is None


def test_galaxy_gradients(payloads, flux):
    """Reverse mode through run_galaxy_batch_segment vs central differences."""
    psf, opt = payloads
    img = _galaxy_image()
    plan = sr.make_segment_plan(psf, opt, 3, "galaxy", galaxy_npix=16)
    sens = jnp.ones_like(WL)
    arrays, static = sr.jit_payload.split_payloads(psf, opt)
    img0 = _np(sr.render_galaxy_segments(psf, opt, img, X0, Y0, flux, WL,
                                         ZERO, plan))
    ys_, xs_ = np.nonzero(img0 > 1e-3 * img0.max())
    y0, y1, x0, x1 = ys_.min() - 5, ys_.max() + 6, xs_.min() - 5, xs_.max() + 6
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wgt = np.zeros((4088, 4088), np.float32)
    wgt[y0:y1, x0:x1] = np.cos(xx / 7.0) * np.sin(yy / 11.0) + 1.5
    wgt = jnp.asarray(wgt)

    def loss(spec, x, y):
        out = sr.run_galaxy_batch_segment(
            arrays, sens, WL, jnp.float32(1.0), 1, spec[None], x[None],
            y[None], img[None], jnp.zeros((4088, 4088), jnp.float32),
            static=static, plan=plan)
        return jnp.sum(out * wgt)

    g_spec, g_x, g_y = jax.grad(loss, argnums=(0, 1, 2))(flux, X0, Y0)
    e = np.zeros(WL.size, np.float32)
    e[80] = 1.0
    fd = (float(loss(flux + 0.5 * e, X0, Y0))
          - float(loss(flux - 0.5 * e, X0, Y0)))
    assert np.isclose(float(g_spec[80]), fd, rtol=2e-3)
    for g, args in ((g_x, lambda d: (flux, X0 + d, Y0)),
                    (g_y, lambda d: (flux, X0, Y0 + d))):
        d = 0.05
        fd = (float(loss(*args(d))) - float(loss(*args(-d)))) / (2 * d)
        assert abs(fd) > 0
        assert np.isclose(float(g), fd, rtol=0.05), (float(g), fd)

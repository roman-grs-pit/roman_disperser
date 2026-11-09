"""Differentiability features for forward-model fitting.

* ``deposit="cubic"`` (C1 in position) and cubic spatial PSF interpolation;
* ``SceneOperator.model_fn`` (spectra, positions, Sérsic shapes) and
  ``sky_dxy``;
* Sérsic sub-pixel sampling and unconstrained reparametrisations;
* JAX photometry (vs synphot) and redshifting;
* differentiable selection functions (hard limit = production cuts);
* the opt-in window-overflow check;
* likelihood / IPC helpers.

Mock PSFs + the real grism optical model of SCA 5 (fast suite).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import (accumulate as acc, catalog, likelihood,
                             linear_operator as lo, photometry, pipeline,
                             psf_model, segment_render as sr, selection,
                             sersic, star_disperser, transforms)

OS = 4
N_WL = 41
WL = jnp.linspace(1.0, 1.8, N_WL).astype(jnp.float32)
DLAM = 200.0


def _mock_psf(n_wl=5, size=20, nodes=2):
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    sig = (2.5 + 0.4 * np.arange(n_wl)[None, None, :]
           + 0.3 * np.arange(nodes)[:, None, None])
    grid = np.exp(-0.5 * (xx ** 2 + (1.2 * yy) ** 2) / sig[..., None, None] ** 2)
    grid = np.broadcast_to(grid, (nodes, nodes, n_wl, size, size)).copy()
    grid = (grid / grid.sum(axis=(-2, -1), keepdims=True)).astype(np.float32)
    return {'psf_grid': jnp.asarray(grid),
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl).astype(jnp.float32),
            'spatial_x': jnp.linspace(1, 4088, nodes).astype(jnp.float32),
            'spatial_y': jnp.linspace(1, 4088, nodes).astype(jnp.float32),
            'oversample': OS}


@pytest.fixture(scope="module")
def model():
    from roman_disperser import elements, paths
    from roman_disperser.optical_model import RomanOpticalModel
    return RomanOpticalModel(str(paths.optical_model_path(
        element=elements.get_element("grism"))))


@pytest.fixture(scope="module")
def opt(model):
    return omj.make_sca_payload(model, sca=5, order="1")


def _stack(n_grid=3, size=20):
    """Smooth [n_grid, size, size] oversampled stamps."""
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    s = 2.2 + 0.3 * np.arange(n_grid)[:, None, None]
    st = np.exp(-0.5 * ((xx - 0.7) ** 2 + (1.3 * yy) ** 2) / s ** 2)
    return jnp.asarray(st / st.sum(axis=(-2, -1), keepdims=True), jnp.float32)


def _deposit_one(x, y, deposit, stack=None, shape=(40, 40)):
    stack = _stack() if stack is None else stack
    grid_wl = jnp.linspace(1.0, 1.8, stack.shape[0]).astype(stack.dtype)
    binned = star_disperser.bin_stack_phases(stack, OS)
    wl = jnp.full((1,), 1.33, stack.dtype)
    out = jnp.zeros(shape, stack.dtype)
    return star_disperser.deposit_binned(
        binned, stack.shape[-2:], grid_wl, wl, jnp.ones(1, stack.dtype),
        jnp.atleast_1d(x), jnp.atleast_1d(y), out, oversample=OS,
        chunk_size=8, deposit=deposit)


# ---------------------------------------------------------------------------
# Cubic deposit
# ---------------------------------------------------------------------------

def _aligned(base, k, size=20):
    # u = x - 0.5 - (S-1)/(2 os); aligned when frac(u) * os - 0.5 = k
    return base + 0.5 + (size - 1) / (2.0 * OS) + (k + 0.5) / OS


class TestCubicDeposit:
    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_equals_interp_at_aligned_positions(self, k):
        x = jnp.float32(_aligned(15, k))
        y = jnp.float32(_aligned(17, (k + 2) % OS))
        a = np.asarray(_deposit_one(x, y, "cubic"), np.float64)
        b = np.asarray(_deposit_one(x, y, "interp"), np.float64)
        np.testing.assert_allclose(a, b, atol=2e-7 * b.max())

    def test_flux_conserved(self):
        rng = np.random.default_rng(0)
        for _ in range(6):
            x, y = (jnp.float32(v) for v in rng.uniform(15, 25, 2))
            a = float(_deposit_one(x, y, "cubic").sum())
            b = float(_deposit_one(x, y, "interp").sum())
            assert abs(a - b) <= 2e-6 * b

    def test_gradient_continuous_across_phase_boundary(self):
        w = jnp.asarray(np.random.default_rng(1).normal(size=(40, 40)),
                        jnp.float32)
        y = jnp.float32(20.3)

        def jump(deposit):
            g = jax.grad(lambda x: jnp.sum(w * _deposit_one(x, y, deposit)))
            xb = _aligned(18, 1)              # a phase boundary
            return abs(float(g(jnp.float32(xb + 1e-3)))
                       - float(g(jnp.float32(xb - 1e-3)))), \
                abs(float(g(jnp.float32(xb + 0.1))))

        j_cub, scale = jump("cubic")
        j_lin, _ = jump("interp")
        assert j_cub < 0.02 * scale
        assert j_lin > 5 * j_cub   # interp is only C0

    def test_gradient_matches_finite_difference_x64(self):
        with jax.enable_x64(True):
            st = _stack().astype(jnp.float64)
            w = jnp.asarray(np.random.default_rng(2).normal(size=(40, 40)))
            f = lambda x, y: jnp.sum(w * _deposit_one(x, y, "cubic", st))
            x0, y0 = jnp.float64(19.37), jnp.float64(21.81)
            gx, gy = jax.grad(f, argnums=(0, 1))(x0, y0)
            h = 1e-5
            fx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
            fy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)
            np.testing.assert_allclose(float(gx), float(fx), rtol=1e-6)
            np.testing.assert_allclose(float(gy), float(fy), rtol=1e-6)

    def test_window_accumulator_refuses_cubic(self, opt):
        with pytest.raises(ValueError, match="cubic"):
            acc.make_window_plan(_mock_psf(), opt, "star",
                                 np.asarray(WL), deposit="cubic")


# ---------------------------------------------------------------------------
# Cubic PSF spatial interpolation
# ---------------------------------------------------------------------------

class TestCubicPSF:
    def _payload(self):
        rng = np.random.default_rng(3)
        grid = rng.uniform(0.5, 1.5, (4, 4, 2, 3, 3)).astype(np.float32)
        return {'psf_grid': jnp.asarray(grid),
                'spatial_x': jnp.linspace(1, 4088, 4).astype(jnp.float32),
                'spatial_y': jnp.linspace(1, 4088, 4).astype(jnp.float32)}

    def test_nodes_reproduced(self):
        p = self._payload()
        for iy in range(4):
            for ix in range(4):
                v = psf_model.interpolate_psf_spatial(
                    p, p['spatial_x'][ix], p['spatial_y'][iy], method='cubic')
                np.testing.assert_allclose(v, p['psf_grid'][iy, ix], rtol=1e-5)

    def test_c1_across_node(self):
        p = self._payload()
        f = lambda x, m: psf_model.interpolate_psf_spatial(
            p, x, jnp.float32(1700.0), method=m).sum()
        xn = float(p['spatial_x'][1])
        for m, small in (("cubic", True), ("linear", False)):
            g = jax.grad(f)
            d = abs(float(g(jnp.float32(xn + 0.5), m))
                    - float(g(jnp.float32(xn - 0.5), m)))
            s = np.mean([abs(float(g(jnp.float32(xn + o), m)))
                         for o in (-400.0, -200.0, 200.0, 400.0)])
            assert (d < 0.02 * s) == small

    def test_operator_cubic_close_to_linear(self, opt):
        psf = _mock_psf(nodes=4)
        sens = jnp.ones(N_WL, jnp.float32)
        orders = {"1": lo.make_order_setup(psf, opt, sens, WL)}
        xy = np.array([[1000.3, 900.55]], np.float32)
        th = jnp.ones((1, N_WL), jnp.float32)
        imgs = [np.asarray(lo.SceneOperator(
            orders, WL, DLAM, star_xy=xy, window=lo.auto_window(orders, xy, 12),
            flux_unit=1.0, psf_spatial=m).forward(th)) for m in ("linear", "cubic")]
        np.testing.assert_allclose(imgs[1].sum(), imgs[0].sum(), rtol=1e-4)


# ---------------------------------------------------------------------------
# SceneOperator.model_fn and sky_dxy
# ---------------------------------------------------------------------------

STARS = np.array([[1000.3, 900.55]], np.float32)
GALS = np.array([[1001.12, 902.9]], np.float32)
NPIX = 16


@pytest.fixture(scope="module")
def fit_op(opt):
    psf = _mock_psf()
    sens = jnp.linspace(0.5, 1.5, N_WL).astype(jnp.float32)
    orders = {"1": lo.make_order_setup(psf, opt, sens, WL)}
    xy = np.concatenate([STARS, GALS])
    return lo.SceneOperator(
        orders, WL, DLAM, star_xy=STARS, galaxy_xy=GALS, galaxy_npix=NPIX,
        window=lo.auto_window(orders, xy, margin=14), flux_unit=1.0,
        precompute=False, deposit="cubic", psf_spatial="cubic",
        sersic_subsample=3)


class TestModelFn:
    def test_gradients_vs_finite_differences(self, fit_op):
        f = fit_op.model_fn()
        rng = np.random.default_rng(4)
        th = jnp.asarray(rng.uniform(0.5, 1.0, fit_op.theta_shape), jnp.float32)
        w = jnp.asarray(rng.normal(size=fit_op.shape), jnp.float32)
        u0 = transforms.sersic_to_unconstrained(
            jnp.array([3.0]), jnp.array([1.5]), jnp.array([0.6]),
            jnp.array([0.4]))

        def loss(dxy, u):
            (r, n, ba, t), _ = transforms.sersic_from_unconstrained(u)
            return jnp.sum(w * f(th, dxy, (r, n, ba, t)))

        loss_j = jax.jit(loss)
        dxy0 = jnp.zeros((2, 2), jnp.float32)
        g_dxy, g_u = jax.jit(jax.grad(loss, argnums=(0, 1)))(dxy0, u0)
        assert np.all(np.isfinite(g_dxy)) and np.all(np.isfinite(g_u))
        for (i, j), h in (((0, 0), 0.02), ((1, 1), 0.02)):
            e = dxy0.at[i, j].set(h)
            fd = (float(loss_j(dxy0 + e, u0)) - float(loss_j(dxy0 - e, u0))) / (2 * h)
            np.testing.assert_allclose(float(g_dxy[i, j]), fd, rtol=0.03,
                                       atol=1e-3 * abs(fd) + 1e-6)
        for k, h in ((0, 0.02), (1, 0.02), (2, 0.02)):
            e = jnp.zeros_like(u0).at[0, k].set(h)
            fd = (float(loss_j(dxy0, u0 + e)) - float(loss_j(dxy0, u0 - e))) / (2 * h)
            np.testing.assert_allclose(float(g_u[0, k]), fd, rtol=0.03,
                                       atol=1e-3 * abs(fd) + 1e-6)

    def test_matches_forward_with_images(self, opt, fit_op):
        params = (jnp.array([3.0]), jnp.array([1.5]), jnp.array([0.6]),
                  jnp.array([0.4]))
        img = sersic.make_sersic_images(*params, npix=NPIX, subsample=3)
        psf = _mock_psf()
        sens = jnp.linspace(0.5, 1.5, N_WL).astype(jnp.float32)
        orders = {"1": lo.make_order_setup(psf, opt, sens, WL)}
        ref_op = lo.SceneOperator(
            orders, WL, DLAM, star_xy=STARS, galaxy_xy=GALS,
            galaxy_images=img, window=fit_op.window, flux_unit=1.0,
            deposit="cubic", psf_spatial="cubic")
        th = jnp.ones(fit_op.theta_shape, jnp.float32)
        splat = lo.SceneOperator(
            orders, WL, DLAM, star_xy=STARS, galaxy_xy=GALS, galaxy_npix=NPIX,
            window=fit_op.window, flux_unit=1.0, precompute=False,
            deposit="cubic", psf_spatial="cubic", sersic_subsample=3,
            sersic_warp="splat")
        a = np.asarray(splat.model_fn()(th, None, params), np.float64)
        b = np.asarray(ref_op.forward(th), np.float64)
        np.testing.assert_allclose(a, b, atol=5e-4 * b.max())
        # analytic warp: same flux, close image (no bilinear resampling blur)
        c = np.asarray(fit_op.model_fn()(th, None, params), np.float64)
        np.testing.assert_allclose(c.sum(), b.sum(), rtol=2e-3)
        assert np.abs(c - b).max() < 0.05 * b.max()

    def test_floor_refused(self, opt):
        psf = _mock_psf()
        orders = {"1": lo.make_order_setup(psf, opt, jnp.ones(N_WL), WL)}
        A = lo.SceneOperator(orders, WL, DLAM, star_xy=STARS, deposit="floor",
                             window=(800, 900, 200, 400))
        with pytest.raises(ValueError, match="floor"):
            A.model_fn()
        with pytest.raises(ValueError, match="floor"):
            A.render(jnp.ones(A.theta_shape), jnp.zeros((1, 2)))

    def test_sky_dxy(self, opt):
        ra = np.array([150.1, 150.12]); dec = np.array([2.2, 2.21])
        pnt = (150.1, 2.2, 30.0)
        for with_payload in (True, False):
            ref = omj.make_sky_reference(ra, dec, *pnt,
                                         payload=opt if with_payload else None)
            kw = dict(d_xi=jnp.array([1e-5, -2e-5]), d_ra=3e-6, d_pa=1e-3)
            d = lo.sky_dxy(ref, opt, **kw)
            x1, y1 = omj.perturbed_sky_to_sca(ref, opt, **kw)
            x0, y0 = omj.perturbed_sky_to_sca(ref, opt)
            np.testing.assert_allclose(d[:, 0], x1 - x0, atol=2e-3)
            np.testing.assert_allclose(d[:, 1], y1 - y0, atol=2e-3)
            g = jax.grad(lambda a: lo.sky_dxy(ref, opt, d_pa=a).sum())(0.0)
            assert np.isfinite(float(g)) and float(g) != 0.0


# ---------------------------------------------------------------------------
# Sérsic and transforms
# ---------------------------------------------------------------------------

class TestSersic:
    def test_subsample_one_is_production(self):
        p = (jnp.array([2.5, 6.0]), jnp.array([1.0, 3.0]),
             jnp.array([0.7, 0.4]), jnp.array([0.1, 1.2]))
        npix = 21
        c = (npix - 1) / 2.0
        yg, xg = jnp.mgrid[0:npix, 0:npix] - c
        old = jax.vmap(sersic._sersic_kernel, in_axes=(0, 0, 0, 0, None, None))(
            *p, yg, xg)
        new = sersic.make_sersic_images(*p, npix=npix)
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
        one = sersic.make_sersic_image(*(v[0] for v in p), npix=npix)
        # scalar vs batched compile: the b_n spline rounds differently (~1 ulp)
        np.testing.assert_allclose(np.asarray(one), np.asarray(old[0]),
                                   rtol=1e-5)

    def test_subsample_normalised_and_converges(self):
        args = (1.2, 2.0, 0.8, 0.3)
        imgs = [np.asarray(sersic.make_sersic_image(*args, npix=21, subsample=s),
                           np.float64) for s in (1, 3, 9, 27)]
        for im in imgs:
            assert abs(im.sum() - 1.0) < 1e-5
        e = [np.abs(im - imgs[-1]).sum() for im in imgs[:-1]]
        assert e[0] > e[1] > e[2]

    def test_transforms_roundtrip_and_logdet(self):
        u = jnp.array([[0.3, -1.2, 0.8, 2.0], [-2.0, 1.5, -0.5, -1.0]])
        (r, n, ba, t), ld = transforms.sersic_from_unconstrained(u)
        assert np.all((n > 0.2) & (n < 8.0) & (ba > 0.05) & (ba < 1) & (r > 0.1))
        np.testing.assert_allclose(
            transforms.sersic_to_unconstrained(r, n, ba, t), u, atol=1e-4)
        for i in range(2):
            J = jax.jacfwd(lambda v: jnp.stack(
                transforms.sersic_from_unconstrained(v)[0]))(u[i])
            np.testing.assert_allclose(float(ld[i]),
                                       float(jnp.log(jnp.abs(jnp.linalg.det(J)))),
                                       atol=1e-4)

    def test_bound_values_are_moved_inside(self):
        v = transforms.sersic_to_unconstrained(
            jnp.array([0.05]), jnp.array([1.0]), jnp.array([1.0]), jnp.array([0.]))
        assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# Photometry
# ---------------------------------------------------------------------------

class TestPhotometry:
    def test_ab_mag_matches_synphot(self):
        syn = pytest.importorskip("synphot")
        import astropy.units as u
        from roman_disperser import refdata
        band = refdata.get_f158_band()
        sp = syn.SourceSpectrum(syn.models.BlackBody1D, temperature=4500)
        sp = sp.normalize(22.0 * u.ABmag, band)
        wl = np.arange(0.9, 2.0, 0.0002)
        flam = sp(wl * 1e4 * u.AA, flux_unit="flam").value
        ref = syn.Observation(sp, band, force="taper").effstim(u.ABmag).value
        bp = photometry.load_bandpass("F158")
        m = float(photometry.ab_mag(jnp.asarray(wl, jnp.float32),
                                    jnp.asarray(flam, jnp.float32), bp))
        assert abs(m - ref) < 3e-3

    def test_normalise_redshift_and_gradients(self):
        bp = photometry.load_bandpass("F158")
        wl = jnp.linspace(0.9, 2.0, 2201).astype(jnp.float32)
        rest_wl = jnp.linspace(0.3, 2.0, 1701).astype(jnp.float32)
        rest = 1e-17 * (1.0 + jnp.exp(-0.5 * ((rest_wl - 0.6563) / 0.002) ** 2))

        def spec(z, mag):
            s = photometry.redshift(wl, rest_wl, rest, z)
            return photometry.normalise_to_mag(wl, s, mag, bp)

        s = spec(1.2, 21.5)
        np.testing.assert_allclose(float(photometry.ab_mag(wl, s, bp)), 21.5,
                                   atol=1e-4)
        np.testing.assert_allclose(
            float(photometry.maggies(wl, s, bp)), 10 ** (-0.4 * 21.5), rtol=1e-4)
        # line lands at 0.6563 * 2.2
        i = int(jnp.argmax(s))
        assert abs(float(wl[i]) - 0.6563 * 2.2) < 1e-3
        gz = jax.grad(lambda z: spec(z, 21.5)[i])(1.2)
        gm = jax.grad(lambda m: spec(1.2, m).sum())(21.5)
        assert np.isfinite(float(gz)) and float(gz) != 0.0
        np.testing.assert_allclose(float(gm), -0.4 * np.log(10) * float(s.sum()),
                                   rtol=1e-4)

    def test_nonpositive_flux(self):
        bp = photometry.load_bandpass("F158")
        wl = jnp.linspace(0.9, 2.0, 101)
        assert np.isinf(float(photometry.ab_mag(wl, jnp.zeros(101), bp)))
        def f(a):
            m = photometry.ab_mag(wl, a * jnp.zeros(101), bp)
            return jnp.where(jnp.isfinite(m), m, 0.0)
        assert np.isfinite(float(jax.grad(f)(1.0)))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fpa(opt):
    xs = jnp.linspace(-1200.0, 5300.0, 60).astype(jnp.float32)
    X, Y = jnp.meshgrid(xs, jnp.linspace(-900.0, 5000.0, 9))
    return omj.sca_to_fpa(opt, X.ravel(), Y.ravel())


class TestSelection:
    def test_hard_limit_equals_production_cut(self, opt, fpa):
        xf, yf = fpa
        mask = np.asarray(catalog.select_sources(opt, xf, yf, 0.9, 2.0))
        assert 0 < mask.sum() < mask.size
        lp = np.asarray(selection.trace_overlap_logp(opt, xf, yf, 0.9, 2.0))
        np.testing.assert_array_equal(lp == 0.0, mask)
        np.testing.assert_array_equal(np.isneginf(lp), ~mask)
        soft = np.exp(np.asarray(selection.trace_overlap_logp(
            opt, xf, yf, 0.9, 2.0, width=1e-2)))
        np.testing.assert_array_equal(soft > 0.5, mask)

    def test_soft_gradients(self, opt, fpa):
        xf, yf = fpa
        g = jax.grad(lambda d: selection.trace_overlap_logp(
            opt, xf + d, yf, 0.9, 2.0, width=20.0).sum())(0.0)
        assert np.isfinite(float(g)) and float(g) != 0.0

    def test_mag_logp_matches_apply_order_mag_limits(self):
        flux = np.array([1e-9, 1e-10, 0.0, np.nan, 3.98e-10, -1.0])
        mag = pipeline.flux_to_ab_mag(flux)
        masks = {"0": np.ones(6, bool), "1": np.ones(6, bool)}
        out, _ = pipeline.apply_order_mag_limits(masks, mag, {"0": 23.5})
        jm = selection.ab_mag_from_maggies(jnp.asarray(flux, jnp.float32))
        lp = np.asarray(selection.mag_logp(jm, 23.5))
        np.testing.assert_array_equal(lp == 0.0, out["0"])
        np.testing.assert_array_equal(np.asarray(selection.mag_logp(jm, None)), 0)
        g = jax.grad(lambda m: selection.mag_logp(m, 23.5, 0.1).sum())(
            jnp.array([23.4, 23.6]))
        assert np.all(np.asarray(g) < 0)

    def test_on_chip_fraction_and_footprint(self, opt):
        xd, yd = star_disperser._compute_dispersed_positions(
            opt, jnp.float32(3900.0), jnp.float32(2000.0), WL)
        c = jnp.ones(N_WL)
        hard = float(selection.on_chip_fraction(xd, yd, c, width=0))
        frac = float(selection.on_chip_fraction(xd, yd, c))
        on = np.mean((np.asarray(xd) >= 0.5) & (np.asarray(xd) <= 4088.5)
                     & (np.asarray(yd) >= 0.5) & (np.asarray(yd) <= 4088.5))
        assert hard == pytest.approx(on)
        assert abs(frac - on) <= 1.5 / N_WL
        g = jax.grad(lambda x: selection.on_chip_fraction(
            *star_disperser._compute_dispersed_positions(
                opt, x, jnp.float32(2000.0), WL), c, width=5.0))(
            jnp.float32(3900.0))
        assert np.isfinite(float(g))
        assert float(selection.footprint_logp(10.0, 10.0)) == 0.0
        assert np.isneginf(float(selection.footprint_logp(-10.0, 10.0)))

    def test_log_normaliser(self):
        eps = jnp.asarray(np.random.default_rng(5).normal(size=20000))

        def log_z(mu):
            return selection.log_normaliser(
                lambda m: selection.soft_le(m, 24.0, 0.2), mu + 1.0 * eps)
        from scipy.stats import norm
        np.testing.assert_allclose(float(jnp.exp(log_z(23.5))),
                                   norm.cdf(0.5 / np.hypot(1.0, 0.2)), atol=0.01)
        assert float(jax.grad(log_z)(23.5)) < 0


# ---------------------------------------------------------------------------
# Window-overflow check
# ---------------------------------------------------------------------------

class TestWindowCheck:
    @pytest.fixture
    def check_on(self):
        sr.set_window_check(True)
        yield
        sr.set_window_check(False)

    def _star(self, opt, plan):
        counts = jnp.ones(N_WL, jnp.float32)
        return acc.render_star_window(_mock_psf(), opt, jnp.float32(1500.3),
                                      jnp.float32(1700.6), WL, counts, plan)

    def test_deposit_window(self, opt, check_on):
        plan = acc.make_window_plan(_mock_psf(), opt, "star", np.asarray(WL),
                                    inner="scatter")
        self._star(opt, plan)                         # fits: no error
        small = plan._replace(window_shape=(plan.window_shape[0], 20))
        with pytest.raises(Exception, match="outside the static window"):
            jax.block_until_ready(self._star(opt, small))

    def test_segment_line_image(self, opt, check_on):
        psf = _mock_psf()
        plan = sr.make_segment_plan(psf, opt, 5, "star")
        out = jnp.zeros((4088, 4088), jnp.float32)
        args = (psf, opt, jnp.float32(1500.3), jnp.float32(1700.6), WL,
                jnp.ones(N_WL, jnp.float32), out)
        sr.render_star_segments(*args, plan)
        small = plan._replace(line_shape=(plan.line_shape[0], 8))
        with pytest.raises(Exception, match="outside the static window"):
            jax.block_until_ready(sr.render_star_segments(*args, small))

    def test_off_by_default(self, opt):
        assert not sr.window_check_enabled()
        plan = acc.make_window_plan(_mock_psf(), opt, "star", np.asarray(WL),
                                    inner="scatter")
        small = plan._replace(window_shape=(plan.window_shape[0], 20))
        self._star(opt, small)                        # silently cropped


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

class TestLikelihood:
    def test_poisson_and_gaussian(self):
        rng = np.random.default_rng(6)
        m = jnp.asarray(rng.uniform(0.5, 2.0, (8, 8)), jnp.float32)
        d = jnp.asarray(rng.poisson(np.asarray(m) * 100.0), jnp.float32)
        g = jax.grad(lambda m: likelihood.poisson_loglike(m, d, 100.0))(m)
        np.testing.assert_allclose(g, 100.0 * (d / (100.0 * m) - 1.0), rtol=1e-4)
        from scipy.stats import norm, poisson
        np.testing.assert_allclose(
            float(likelihood.poisson_loglike(m, d, 100.0)),
            poisson.logpmf(np.asarray(d), np.asarray(m) * 100.0).sum(), rtol=1e-5)
        np.testing.assert_allclose(
            float(likelihood.gaussian_loglike(m, d / 100.0, 0.01)),
            norm.logpdf(np.asarray(d) / 100.0, np.asarray(m), 0.1).sum(),
            rtol=1e-5)

    def test_ipc(self):
        img = jnp.zeros((7, 7)).at[3, 3].set(1.0)
        k = jnp.array([[0.0, 0.01, 0.0], [0.02, 0.94, 0.0], [0.0, 0.03, 0.0]])
        out = likelihood.ipc_convolve(img, k)
        np.testing.assert_allclose(out[2:5, 2:5], k, atol=1e-7)
        g = jax.grad(lambda k: likelihood.ipc_convolve(img, k)[3, 4])(k)
        assert float(g[1, 2]) == pytest.approx(1.0)

    def test_selected_loglike(self):
        v = likelihood.selected_loglike(-10.0, jnp.array([-0.1, -0.2]), -0.5)
        assert float(v) == pytest.approx(-10.0 - 0.3 + 1.0)


# ---------------------------------------------------------------------------
# b_n spline and analytic Sérsic warp
# ---------------------------------------------------------------------------

class TestBnSpline:
    def test_accuracy(self):
        from scipy.special import gammaincinv
        n = np.linspace(0.2, 9.9, 3001)
        ref = gammaincinv(2 * n, 0.5)
        got = np.asarray(sersic.compute_bn(jnp.asarray(n, jnp.float32)),
                         np.float64)
        lin = np.asarray(sersic.compute_bn(jnp.asarray(n, jnp.float32),
                                           "linear"), np.float64)
        assert np.max(np.abs(got / ref - 1)) < 2e-6
        assert np.max(np.abs(got / ref - 1)) <= np.max(np.abs(lin / ref - 1))

    def test_gradient_continuous_at_nodes(self):
        # n ~ 0.21, where b_n is curved enough for the linear table's
        # slope jump (~0.6 %) to stand out; the spline's slope changes by
        # only b_n'' * 2e-5 across the same gap.
        node = float(sersic._BN_LO + 40 * sersic._BN_STEP)
        g = jax.grad(lambda n, m: sersic.compute_bn(n, m))
        for m, smooth in (("spline", True), ("linear", False)):
            lo_, hi_ = (float(g(jnp.float32(node + d), m)) for d in (-1e-5, 1e-5))
            assert (abs(hi_ - lo_) < 1e-3 * abs(hi_)) == smooth, (m, lo_, hi_)

    def test_bad_method(self):
        with pytest.raises(ValueError):
            sersic.compute_bn(1.0, "cubic")


class TestAnalyticWarp:
    P = (jnp.float32(5.0), jnp.float32(1.3), jnp.float32(0.6),
         jnp.float32(0.5))

    def test_identity_is_plain_image(self):
        from roman_disperser import galaxy_disperser as gd
        w = gd.warped_sersic_images(self.P, jnp.eye(2)[None], 0.25, 0.25, 31,
                                    subsample=3)[0]
        ref = sersic.make_sersic_image(*self.P, npix=31, subsample=3)
        np.testing.assert_allclose(w, ref, atol=1e-7)

    def test_close_to_splat_and_flux(self):
        from roman_disperser import galaxy_disperser as gd
        J = jnp.array([[1.01, 0.02], [-0.01, 1.03]], jnp.float32)
        img = sersic.make_sersic_image(*self.P, npix=41)
        splat = np.asarray(gd.disperse_galaxy_shape(img, J, 0.25, 0.25))
        ana = np.asarray(gd.warped_sersic_images(self.P, J[None], 0.25, 0.25,
                                                 41)[0])
        assert abs(ana.sum() - 1) < 1e-5
        assert np.abs(ana - splat).max() < 0.05 * splat.max()
        # A stretch by det J lowers the peak surface brightness by det J.
        # The analytic warp has that; the splat keeps the unwarped peak (the
        # central pixel maps onto itself with weight 1).
        det = float(jnp.linalg.det(J))
        np.testing.assert_allclose(ana.max(), img.max() / det, rtol=3e-3)
        np.testing.assert_allclose(splat.max(), img.max(), rtol=3e-3)

    def test_smooth_in_jacobian(self):
        from roman_disperser import galaxy_disperser as gd
        img = sersic.make_sersic_image(*self.P, npix=41)
        w = jnp.asarray(np.random.default_rng(8).normal(size=(41, 41)),
                        jnp.float32)

        def jumps(fn):
            g = jax.grad(lambda a: jnp.sum(w * fn(
                jnp.array([[1.0 + a, 0.0], [0.0, 1.0]], jnp.float32))))
            # wide enough that splatted pixels cross output pixel edges
            # (stretch a moves pixel k by a * k px)
            vals = [float(g(jnp.float32(a))) for a in
                    np.linspace(0.10, 0.20, 81)]
            return np.max(np.abs(np.diff(vals))) / np.max(np.abs(vals))

        ana = jumps(lambda J: gd.warped_sersic_images(
            self.P, J[None], 0.25, 0.25, 41)[0])
        spl = jumps(lambda J: gd.disperse_galaxy_shape(img, J, 0.25, 0.25))
        assert ana < 0.05 and spl > 3 * ana

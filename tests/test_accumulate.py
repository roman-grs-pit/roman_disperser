"""
Tests for per-source windows and deterministic accumulation (``accumulate``).

A source rendered into its own window and added to the frame at the window
origin must give the production image (same arithmetic, other summation
order), the ordered (unique-index) inner accumulation must equal the plain
scatter, and the window / tiled frame sums must not depend on batch sizes,
vmap widths, tile batching or (after canonical sorting) catalogue order.
"""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import accumulate as acc
from roman_disperser import galaxy_disperser, jit_payload, pipeline
from roman_disperser import segment_render as sr
from roman_disperser import star_disperser

N_GRID = 6
WL = jnp.linspace(1.0, 1.8, 161).astype(jnp.float32)
WL_NP = np.asarray(WL)
DLAM = 50.0


def _z():
    """Fresh accumulator (the batch functions donate theirs)."""
    return jnp.zeros((4088, 4088), jnp.float32)


NPIX = 16


def _mock_psf(n_wl=N_GRID, size=20):
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    sig = (1.5 + 0.5 * np.arange(n_wl)[None, None, :]
           + 0.4 * np.arange(2)[:, None, None])
    grid = np.exp(-0.5 * (xx ** 2 + (1.3 * yy) ** 2)
                  / sig[..., None, None] ** 2)
    grid = np.broadcast_to(grid, (2, 2, n_wl, size, size)).copy()
    grid[1] *= 1.0 + 0.2 * np.sin(xx / 3.0)
    grid = (grid / grid.sum(axis=(-2, -1), keepdims=True)).astype(np.float32)
    return {'psf_grid': jnp.asarray(grid),
            'wavelengths': jnp.linspace(1.0, 1.8, n_wl).astype(jnp.float32),
            'spatial_x': jnp.linspace(1, 4088, 2).astype(jnp.float32),
            'spatial_y': jnp.linspace(1, 4088, 2).astype(jnp.float32),
            'oversample': 4}


def _galaxy_images(n):
    from scipy import ndimage
    rng = np.random.default_rng(5)
    imgs = []
    for _ in range(n):
        img = np.zeros((NPIX, NPIX), np.float32)
        a, b = rng.integers(3, 7, 2)
        img[8 - a:8 + a, 8 - b:8 + b] = 1.0
        img = ndimage.gaussian_filter(img, 1.0)
        imgs.append(img / img.sum())
    return np.asarray(imgs, np.float32)


def _payloads():
    from roman_disperser import elements, paths
    from roman_disperser.optical_model import RomanOpticalModel
    model = RomanOpticalModel(str(paths.optical_model_path(
        element=elements.get_element("grism"))))
    return _mock_psf(), omj.make_sca_payload(model, sca=5, order="1")


def _scene():
    """12 stars/galaxies, two near the low detector edges (windows cut)."""
    rng = np.random.default_rng(11)
    n = 12
    x = rng.uniform(200, 3900, n).astype(np.float32)
    y = rng.uniform(200, 3900, n).astype(np.float32)
    x[0], y[0] = 5.3, 1200.7
    x[1], y[1] = 2100.2, -150.4
    spec = rng.uniform(0.5, 2.0, (n, WL.size)).astype(np.float32)
    return x, y, spec, _galaxy_images(n)


@pytest.fixture(scope="module")
def payloads():
    return _payloads()


@pytest.fixture(scope="module")
def scene():
    return _scene()


@pytest.fixture(scope="module")
def sens():
    return jnp.ones(WL.size, jnp.float32)


def _plan(payloads, kind="star", **kw):
    psf, opt = payloads
    return acc.make_window_plan(psf, opt, kind, WL_NP, galaxy_npix=NPIX,
                                vmap_width=kw.pop("vmap_width", 3), **kw)


def _np(a):
    return np.asarray(a, np.float64)


def _close(a, b, tol=2e-6):
    a, b = _np(a), _np(b)
    assert np.abs(a - b).max() <= tol * np.abs(b).max(), (
        np.abs(a - b).max() / np.abs(b).max())


def _place(window, oy, ox):
    return acc.add_window(_z(), window, oy, ox)


class TestPlan:
    def test_shapes_and_errors(self, payloads):
        p = _plan(payloads)
        assert p.renderer == "deposit" and p.inner == "ordered"
        assert p.n_samples == WL.size and 1 <= p.stride <= WL.size
        s = _plan(payloads, renderer="segment", n_nodes=3)
        assert s.segment is not None and s.segment.n_nodes == 3
        g = _plan(payloads, "galaxy")
        assert g.stamp_shape == (20 + NPIX - 1, 20 + NPIX - 1)
        with pytest.raises(ValueError):
            _plan(payloads, renderer="nope")
        with pytest.raises(ValueError):
            _plan(payloads, inner="nope")
        with pytest.raises(ValueError):
            _plan(payloads, renderer="segment")      # n_nodes missing
        with pytest.raises(ValueError):
            acc.make_window_plan(*payloads, "galaxy", WL_NP)  # npix missing

    def test_ordered_classes_have_unique_cells(self, payloads, scene):
        """Two samples of one class never share a window pixel (deposit)
        or an oversampled line cell (segment tents)."""
        psf, opt = payloads
        x, y, _, _ = scene
        p = _plan(payloads)
        n_y, n_x = star_disperser._native_stamp_size(*p.stamp_shape, 4)
        n_y, n_x = n_y + 1, n_x + 1
        rel = -(p.stamp_shape[0] - 1) / 8.0
        for xi, yi in zip(x, y):
            xd, yd = star_disperser._compute_dispersed_positions(
                opt, jnp.float32(xi), jnp.float32(yi), WL)
            _, _, _, m_y, _ = star_disperser._interp_taps(
                np.asarray(yd) - 0.5 + rel, 4)
            _, _, _, m_x, _ = star_disperser._interp_taps(
                np.asarray(xd) - 0.5 + rel, 4)
            m_y, m_x = np.asarray(m_y), np.asarray(m_x)
            for c in range(p.stride):
                my, mx = m_y[c::p.stride], m_x[c::p.stride]
                dy = np.abs(my[:, None] - my[None, :])
                dx = np.abs(mx[:, None] - mx[None, :])
                clash = (dy < n_y) & (dx < n_x)
                np.fill_diagonal(clash, False)
                assert not clash.any()


class TestWindows:
    @pytest.mark.parametrize("deposit", ["interp", "floor"])
    @pytest.mark.parametrize("inner", ["ordered", "scatter"])
    def test_star_window_is_deposit(self, payloads, scene, deposit, inner):
        psf, opt = payloads
        x, y, spec, _ = scene
        p = _plan(payloads, deposit=deposit, inner=inner)
        for i in (0, 2):
            xi, yi = jnp.float32(x[i]), jnp.float32(y[i])
            disp = star_disperser._compute_dispersed_positions(opt, xi, yi, WL)
            c = jnp.asarray(spec[i])
            ref = star_disperser.disperse_star_psf(
                psf, opt, xi, yi, WL, c, _z(), deposit=deposit,
                dispersed=disp)
            win = acc.render_star_window(psf, opt, xi, yi, WL, c, p,
                                         dispersed=disp)
            _close(_place(*win), ref)

    def test_star_window_segment(self, payloads, scene):
        psf, opt = payloads
        x, y, spec, _ = scene
        p = _plan(payloads, renderer="segment", n_nodes=4)
        for i in (0, 3):
            xi, yi = jnp.float32(x[i]), jnp.float32(y[i])
            disp = star_disperser._compute_dispersed_positions(opt, xi, yi, WL)
            c = jnp.asarray(spec[i])
            ref = sr.render_star_segments(psf, opt, xi, yi, WL, c, _z(),
                                          p.segment, dispersed=disp)
            win = acc.render_star_window(psf, opt, xi, yi, WL, c, p,
                                         dispersed=disp)
            _close(_place(*win), ref)

    @pytest.mark.parametrize("renderer", ["deposit", "segment"])
    def test_galaxy_window(self, payloads, scene, renderer):
        psf, opt = payloads
        x, y, spec, imgs = scene
        p = _plan(payloads, "galaxy", renderer=renderer, n_nodes=4)
        i = 4
        xi, yi = jnp.float32(x[i]), jnp.float32(y[i])
        img, c = jnp.asarray(imgs[i]), jnp.asarray(spec[i])
        if renderer == "deposit":
            ref = galaxy_disperser.disperse_galaxy(opt, psf, img, xi, yi, c,
                                                   WL, _z())
        else:
            ref = sr.render_galaxy_segments(psf, opt, img, xi, yi, c, WL,
                                            _z(), p.segment)
        win = acc.render_galaxy_window(psf, opt, img, xi, yi, c, WL, p)
        _close(_place(*win), ref)


def _star_window_fn(payloads, sens, plan):
    psf, opt = payloads
    return acc.make_batched_star_window(psf, opt, sens, WL, DLAM, plan,
                                        hoist_psf=False)


def _production_stars(payloads, sens, scene, batch=5):
    psf, opt = payloads
    x, y, spec, _ = scene
    disp = star_disperser.make_star_disperser(psf, opt)
    fn = pipeline.make_batched_star_fori(disp, sens, WL, DLAM)
    return pipeline.disperse_batched_stars(fn, spec, x, y, _z(), batch)


class TestWindowMode:
    def _render(self, payloads, sens, scene, batch, width, vectorize=True):
        x, y, spec, _ = scene
        fn = _star_window_fn(payloads, sens, _plan(
            payloads, vmap_width=width, vectorize=vectorize))
        return np.asarray(pipeline.disperse_batched_stars(
            fn, spec, x, y, _z(), batch))

    @pytest.mark.parametrize("vectorize", [True, False])
    def test_matches_production_and_batch_invariant(self, payloads, sens,
                                                    scene, vectorize):
        """Bit-identical across batch sizes (also below the width) at a
        fixed vmap width, on every backend."""
        ref = _production_stars(payloads, sens, scene)
        outs = [self._render(payloads, sens, scene, batch, 4, vectorize)
                for batch in (5, 12, 7, 2)]
        _close(outs[0], ref)
        for o in outs[1:]:
            assert np.array_equal(o, outs[0])

    @pytest.mark.parametrize("vectorize", [True, False])
    def test_width_invariant(self, payloads, sens, scene, vectorize):
        """Across vmap widths: bit-identical on CPU and with lax.map
        (vectorize=False). On GPU each vmap width is its own compiled
        program, so vmapped widths agree only to float32 rounding (L40S:
        width 2 vs 3/4/8 differ by 1.3e-7 of peak); 1e-6 of peak is ~8x that."""
        outs = [self._render(payloads, sens, scene, batch, width, vectorize)
                for batch, width in ((5, 3), (12, 4), (7, 1), (12, 8))]
        exact = jax.default_backend() == "cpu" or not vectorize
        for o in outs[1:]:
            if exact:
                assert np.array_equal(o, outs[0])
            else:
                _close(o, outs[0], tol=1e-6)

    def test_padded_rows_skipped(self, payloads, sens, scene):
        x, y, spec, _ = scene
        p = _plan(payloads)
        psf, opt = payloads
        arrays, static = jit_payload.split_payloads(psf, opt)
        out = acc.run_star_batch_window(
            arrays, sens, WL, jnp.float32(DLAM), 2, jnp.asarray(spec[:6]),
            jnp.asarray(x[:6]), jnp.asarray(y[:6]), _z(), static=static,
            plan=p)
        ref = acc.run_star_batch_window(
            arrays, sens, WL, jnp.float32(DLAM), 2, jnp.asarray(spec[:2]),
            jnp.asarray(x[:2]), jnp.asarray(y[:2]), _z(), static=static,
            plan=p)
        assert np.array_equal(np.asarray(out), np.asarray(ref))

    def test_canonical_order_is_permutation_invariant(self, payloads, sens,
                                                      scene):
        x, y, spec, _ = scene
        fn = _star_window_fn(payloads, sens, _plan(payloads))
        perm = np.random.default_rng(2).permutation(x.size)
        outs = []
        for idx in (np.arange(x.size), perm):
            c = acc.canonical_order(x[idx], y[idx], spec[idx])
            sel = idx[c]
            outs.append(np.asarray(pipeline.disperse_batched_stars(
                fn, spec[sel], x[sel], y[sel], _z(), 5)))
        assert np.array_equal(outs[0], outs[1])

    @pytest.mark.parametrize("renderer", ["deposit", "segment"])
    def test_galaxies_match_production(self, payloads, sens, scene,
                                       renderer):
        psf, opt = payloads
        x, y, spec, imgs = scene
        p = _plan(payloads, "galaxy", renderer=renderer, n_nodes=4)
        if renderer == "deposit":
            ref_fn = pipeline.make_batched_galaxy_fori(
                galaxy_disperser.make_galaxy_disperser(psf, opt), sens, WL,
                DLAM)
        else:
            ref_fn = sr.make_batched_galaxy_segment(psf, opt, sens, WL, DLAM,
                                                    p.segment)
        fn = acc.make_batched_galaxy_window(psf, opt, sens, WL, DLAM, p)
        imgs = jnp.asarray(imgs)
        ref = pipeline.disperse_batched_galaxies(ref_fn, spec, x, y, imgs,
                                                 _z(), 5)
        out = pipeline.disperse_batched_galaxies(fn, spec, x, y, imgs,
                                                 _z(), 5)
        _close(out, ref)


class TestTiled:
    def test_tiled_matches_window_and_is_invariant(self, payloads, sens,
                                                   scene):
        psf, opt = payloads
        x, y, spec, _ = scene
        p = _plan(payloads, vmap_width=1)
        ref = _production_stars(payloads, sens, scene)
        outs = []
        perm = np.random.default_rng(4).permutation(x.size)
        for idx, rounds, tb in ((np.arange(x.size), 2, 8),
                                (np.arange(x.size), 1, 16),
                                (perm, 3, 4)):
            td = acc.TiledDisperser(psf, opt, sens, WL, DLAM, p,
                                    hoist_psf=False, n_tiles=4,
                                    rounds=rounds, tile_batch=tb)
            outs.append(np.asarray(td.stars(spec[idx], x[idx], y[idx],
                                            _z())))
        _close(outs[0], ref)
        for o in outs[1:]:
            assert np.array_equal(o, outs[0])

    def test_tile_helpers(self):
        lay = acc.make_tile_layout((100, 30), n_tiles=8)
        assert lay.tile == 511 and lay.count == 64
        assert lay.buffer_shape == (511 + 100 + 8, 511 + 30 + 8)
        t = acc.assign_tiles(lay, np.array([-50, 0, 511, 5000]),
                             np.array([0, 600, -3, 4087]))
        assert list(t) == [0, 1, 8, 63]
        table = acc.tile_table(np.array([1, 1, 0]), np.array([2, 1, 0]), 3,
                               rounds_multiple=4)
        assert table.shape == (3, 4)
        assert list(table[1, :2]) == [1, 0] and table[0, 0] == 2
        assert (table[2] == -1).all()


def test_gradients_match_finite_differences(payloads, sens, scene):
    """Reverse mode through run_star_batch_window: spectrum vs FD (linear,
    tight); x vs the production batch's reverse mode (same arithmetic) and
    vs FD (piecewise-linear deposit; smooth weight, 5 %)."""
    psf, opt = payloads
    x, y, spec, _ = scene
    p = _plan(payloads)
    arrays, static = jit_payload.split_payloads(psf, opt)
    yy, xx = np.mgrid[:4088, :4088].astype(np.float32)
    w = jnp.asarray(np.cos(xx / 7.0) * np.sin(yy / 11.0) + 1.5)
    idx = [2, 3, 4]
    y_b = jnp.asarray(y[idx])

    def loss(sp, xx_):
        out = acc.run_star_batch_window(
            arrays, sens, WL, jnp.float32(DLAM), 3, sp, xx_, y_b,
            jnp.zeros((4088, 4088), jnp.float32), static=static, plan=p)
        return jnp.sum(out * w)

    def loss_prod(sp, xx_):
        out = pipeline.run_star_batch(
            arrays, sens, WL, jnp.float32(DLAM), 3, sp, xx_, y_b,
            jnp.zeros((4088, 4088), jnp.float32), static=static)
        return jnp.sum(out * w)

    sp0, x0 = jnp.asarray(spec[idx]), jnp.asarray(x[idx])
    g_sp, g_x = jax.grad(loss, argnums=(0, 1))(sp0, x0)
    g_x_prod = jax.grad(loss_prod, argnums=1)(sp0, x0)
    np.testing.assert_allclose(np.asarray(g_x), np.asarray(g_x_prod),
                               rtol=1e-3)
    k = 80
    e = jnp.zeros_like(sp0).at[1, k].set(0.25)
    fd = (loss(sp0 + e, x0) - loss(sp0 - e, x0)) / 0.5
    assert abs(float(g_sp[1, k]) - float(fd)) <= 2e-3 * abs(float(fd))
    h = 0.05
    ex = jnp.zeros_like(x0).at[0].set(h)
    fdx = (loss(sp0, x0 + ex) - loss(sp0, x0 - ex)) / (2 * h)
    assert abs(float(g_x[0]) - float(fdx)) <= 0.05 * abs(float(fdx))


def test_sharded_tiles_bit_identical(tmp_path):
    """shard_map over 4 emulated CPU devices = one device, bit for bit."""
    script = textwrap.dedent(f"""
        import os, sys
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        import numpy as np, jax, jax.numpy as jnp
        from jax.sharding import Mesh
        import test_accumulate as t
        from roman_disperser import accumulate as acc
        assert jax.device_count() == 4
        payloads = t._payloads()
        x, y, spec, _ = t._scene()
        sens = jnp.ones(t.WL.size, jnp.float32)
        p = t._plan(payloads, vmap_width=1)
        outs = []
        for mesh in (None, Mesh(np.array(jax.devices()), ("t",)),
                     Mesh(np.array(jax.devices()[:2]), ("t",))):
            td = acc.TiledDisperser(*payloads, sens, t.WL, t.DLAM, p,
                                    hoist_psf=False, n_tiles=4, rounds=2,
                                    tile_batch=2, mesh=mesh)
            outs.append(np.asarray(td.stars(spec, x, y, t._z())))
        assert outs[0].max() > 0
        for o in outs[1:]:
            assert np.array_equal(o, outs[0]), np.abs(o - outs[0]).max()
        print("SHARDED_OK")
    """)
    path = tmp_path / "sharded.py"
    path.write_text(script)
    # Emulated CPU devices also on a GPU host, where the default backend
    # would otherwise be the GPU (one device).
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    res = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, env=env)
    assert "SHARDED_OK" in res.stdout, res.stdout + res.stderr


class TestDriverOptions:
    """Opt-in wiring in scripts/build_dispersed_image.py (off by default)."""

    @pytest.fixture(scope="class")
    def bdi(self):
        import importlib.util
        from pathlib import Path
        path = (Path(__file__).parent.parent / "scripts"
                / "build_dispersed_image.py")
        spec = importlib.util.spec_from_file_location("bdi_accumulate", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_default_off(self, bdi):
        import inspect
        for fn in (bdi.setup_pipeline, bdi.build_dispersed_image,
                   bdi.make_order_fori_fns):
            assert inspect.signature(fn).parameters[
                "accumulate"].default is None
        with pytest.raises(ValueError):
            bdi._accumulate_fns("nope", None, None, None, None, None, 1,
                                deposit="interp", hoist_star_psf=False,
                                n_nodes=None, psf_at=None, galaxy_npix_os=4)

    @pytest.mark.parametrize("mode", ["window", "tiled"])
    def test_driver_modes_match_production(self, bdi, payloads, sens, scene,
                                           mode):
        psf, opt = payloads
        x, y, spec, _ = scene
        ref = _production_stars(payloads, sens, scene)
        star_fn, _ = bdi._accumulate_fns(
            mode, psf, opt, sens, WL, DLAM, 1, deposit="interp",
            hoist_star_psf=False, n_nodes=None, psf_at=None,
            galaxy_npix_os=NPIX)
        perm = np.random.default_rng(6).permutation(x.size)
        outs = [np.asarray(bdi.disperse_order_sources(
            star_fn, "star", spec[i], x[i], y[i], _z(), 5,
            accumulate=mode)) for i in (np.arange(x.size), perm)]
        _close(outs[0], ref)
        assert np.array_equal(outs[0], outs[1])

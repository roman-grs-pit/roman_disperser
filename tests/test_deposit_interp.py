"""
Tests for the interpolating deposit (``deposit="interp"``), the count
rebinning before the deposit (``pipeline.rebin_counts``) and the star PSF
hoist (``make_star_disperser(hoist_psf=True)``).

The interpolating deposit claims to be the *exact* box-overlap deposit of
the oversampled stamp: each subpixel is a box of width 1/os native pixels
whose flux is split between the (at most two) native pixels it overlaps, by
overlap length, per axis. The reference below computes exactly that in
float64, independently of the phase-binning machinery.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import galaxy_disperser, star_disperser
from roman_disperser.pipeline import (
    make_batched_galaxy_fori, make_batched_star_fori, parse_rebin_dlam,
    rebin_counts, sed_rebin_factor, sed_rebin_factors,
)

SHAPE = (64, 64)


def _overlap_matrix(centers, size, oversample, n_out):
    """[K, n_out, size] per-axis box-overlap weights (float64).

    Subpixel j of a stamp centred at ``c`` (1-indexed FITS) has its centre
    at ``c + (j - (size-1)/2)/os``, i.e. at array coordinate
    ``w = c - 0.5 + (j - (size-1)/2)/os`` where native pixel i spans
    [i, i+1) in w for the floor deposit; the box spans [w - h/2, w + h/2].
    """
    h = 1.0 / oversample
    k = len(centers)
    out = np.zeros((k, n_out, size))
    rel = (np.arange(size) - (size - 1) / 2.0) * h
    for a, c in enumerate(np.asarray(centers, np.float64)):
        lo = c - 0.5 + rel - h / 2
        hi = lo + h
        for j in range(size):
            i0 = int(np.floor(lo[j]))
            for i in (i0, i0 + 1):
                ov = min(hi[j], i + 1) - max(lo[j], i)
                if ov > 0 and 0 <= i < n_out:
                    out[a, i, j] += ov / h
    return out


def _reference_exact_f64(stack, grid_wl, wavelengths, flux, x_disp, y_disp,
                         shape, oversample):
    stack = np.asarray(stack, np.float64)
    grid_wl = np.asarray(grid_wl, np.float64)
    n_grid = stack.shape[0]
    s_y, s_x = stack.shape[-2:]
    wy = _overlap_matrix(y_disp, s_y, oversample, shape[0])
    wx = _overlap_matrix(x_disp, s_x, oversample, shape[1])
    out = np.zeros(shape)
    for k, (wl, fx) in enumerate(zip(np.asarray(wavelengths, np.float64),
                                     np.asarray(flux, np.float64))):
        i0 = int(np.clip(np.searchsorted(grid_wl, wl) - 1, 0, n_grid - 2))
        t = np.clip((wl - grid_wl[i0]) / (grid_wl[i0 + 1] - grid_wl[i0]),
                    0.0, 1.0)
        stamp = (stack[i0] + t * (stack[i0 + 1] - stack[i0])) * fx
        out += wy[k] @ stamp @ wx[k].T
    return out


def _random_case(s_y, s_x, n_wl, seed, lo=25.0, hi=35.0):
    rng = np.random.default_rng(seed)
    stack = jnp.array(rng.uniform(0.1, 1.0, (3, s_y, s_x)), jnp.float32)
    grid_wl = jnp.array([1.0, 1.5, 2.0], jnp.float32)
    x = jnp.array(rng.uniform(lo, hi, n_wl), jnp.float32)
    y = jnp.array(rng.uniform(lo, hi, n_wl), jnp.float32)
    wl = jnp.linspace(0.9, 2.1, n_wl).astype(jnp.float32)
    flux = jnp.array(rng.uniform(0.5, 2.0, n_wl), jnp.float32)
    return stack, grid_wl, wl, flux, x, y


class TestInterpDeposit:

    @pytest.mark.parametrize("s_y,s_x,os_", [
        (12, 12, 4), (9, 9, 4), (10, 10, 4), (11, 11, 4), (10, 13, 4),
        (11, 8, 4), (7, 10, 2),
    ])
    def test_matches_exact_box_overlap(self, s_y, s_x, os_):
        stack, grid_wl, wl, flux, x, y = _random_case(s_y, s_x, 23,
                                                      100 * s_y + s_x)
        # include exact phase-aligned positions and boundaries
        extra = jnp.array([30.0, 30.125, 30.25, 30.375, 30.5, 30.875],
                          jnp.float32)
        x = jnp.concatenate([x, extra])
        y = jnp.concatenate([y, extra[::-1]])
        wl = jnp.linspace(0.9, 2.1, len(x)).astype(jnp.float32)
        flux = jnp.concatenate([flux, jnp.ones(len(extra), jnp.float32)])
        got = star_disperser.deposit_stack_native(
            stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32),
            oversample=os_, chunk_size=8, deposit="interp")
        want = _reference_exact_f64(stack, grid_wl, wl, flux, x, y, SHAPE,
                                    os_)
        atol = 2e-5 * float(np.abs(want).max())
        np.testing.assert_allclose(np.asarray(got, np.float64), want,
                                   rtol=2e-5, atol=atol)

    def test_floor_differs_from_exact(self):
        """Sanity check of the reference: the floor deposit is NOT exact
        off the phase-aligned positions (else the test above is vacuous)."""
        stack, grid_wl, wl, flux, x, y = _random_case(12, 12, 10, 5)
        got = star_disperser.deposit_stack_native(
            stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32),
            oversample=4, chunk_size=4, deposit="floor")
        want = _reference_exact_f64(stack, grid_wl, wl, flux, x, y, SHAPE, 4)
        assert np.abs(np.asarray(got) - want).max() > 1e-2 * want.max()

    def test_aligned_positions_equal_floor(self):
        """At phase-aligned centres (frac(u)*os = p + 1/2) the exact and the
        floor deposit coincide."""
        s = 12
        rel0 = -(s - 1) / 8.0
        # u = c - 0.5 + rel0 with frac(u) = (p + 0.5)/4
        x = jnp.array([30 + (p + 0.5) / 4 + 0.5 - rel0 % 1 for p in range(4)],
                      jnp.float32)
        y = x[::-1]
        stack, grid_wl, _, _, _, _ = _random_case(s, s, 4, 9)
        wl = jnp.array([1.1, 1.3, 1.6, 1.9], jnp.float32)
        flux = jnp.ones(4, jnp.float32)
        out = {d: np.asarray(star_disperser.deposit_stack_native(
            stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32),
            oversample=4, chunk_size=4, deposit=d), np.float64)
            for d in ("floor", "interp")}
        np.testing.assert_allclose(out["interp"], out["floor"], rtol=1e-5,
                                   atol=1e-5 * out["floor"].max())

    def test_flux_conservation(self):
        stack, grid_wl, wl, flux, x, y = _random_case(13, 11, 40, 3)
        got = star_disperser.deposit_stack_native(
            stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32),
            oversample=4, chunk_size=16, deposit="interp")
        # expected: sum over samples of flux * (lambda-interpolated stamp sum)
        sums = np.asarray(stack, np.float64).sum(axis=(1, 2))
        g = np.asarray(grid_wl, np.float64)
        w = np.asarray(wl, np.float64)
        i0 = np.clip(np.searchsorted(g, w) - 1, 0, 1)
        t = np.clip((w - g[i0]) / (g[i0 + 1] - g[i0]), 0, 1)
        want = (np.asarray(flux, np.float64)
                * (sums[i0] + t * (sums[i0 + 1] - sums[i0]))).sum()
        np.testing.assert_allclose(float(np.asarray(got, np.float64).sum()),
                                   want, rtol=1e-6)

    def test_edge_drop_matches_reference(self):
        """Stamps straddling the detector edge: dropped flux = the part of
        the boxes outside the detector, as in the exact reference."""
        rng = np.random.default_rng(7)
        stack = jnp.array(rng.uniform(0.1, 1.0, (2, 11, 10)), jnp.float32)
        grid_wl = jnp.array([1.0, 2.0], jnp.float32)
        x = jnp.array([1.3, 63.6, -0.37, 0.9], jnp.float32)
        y = jnp.array([0.7, 62.9, 64.2, 2.2], jnp.float32)
        wl = jnp.array([1.1, 1.5, 1.9, 1.2], jnp.float32)
        flux = jnp.ones(4, jnp.float32)
        got = star_disperser.deposit_stack_native(
            stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32),
            oversample=2, chunk_size=2, deposit="interp")
        want = _reference_exact_f64(stack, grid_wl, wl, flux, x, y, SHAPE, 2)
        np.testing.assert_allclose(np.asarray(got, np.float64), want,
                                   rtol=2e-5, atol=2e-5 * want.max())

    def _scan(self, deposit, deltas):
        stack, grid_wl, _, _, _, _ = _random_case(12, 12, 1, 11)
        wl = jnp.array([1.4], jnp.float32)
        flux = jnp.ones(1, jnp.float32)

        @jax.jit
        def render(d):
            x = jnp.array([30.0], jnp.float32) + d
            y = jnp.array([31.0], jnp.float32) + 0.7 * d
            return star_disperser.deposit_stack_native(
                stack, grid_wl, wl, flux, x, y,
                jnp.zeros(SHAPE, jnp.float32), oversample=4, chunk_size=1,
                deposit=deposit)
        return render, np.stack([np.asarray(render(jnp.float32(d)))
                                 for d in deltas])

    def test_continuity_in_subpixel_position(self):
        """Lipschitz in the centre position: a step of 1e-3 px changes the
        image by O(1e-3 * stamp scale), across all phase boundaries; the
        floor deposit jumps by O(1) there."""
        step = 1.0 / 512
        deltas = np.arange(0, 1.0 + 1e-9, step)
        _, imgs = self._scan("interp", deltas)
        jumps = np.abs(np.diff(imgs, axis=0)).max(axis=(1, 2))
        scale = np.abs(imgs).max()
        # max |d image / d pos| <= ~ 2 * os * peak native value
        assert jumps.max() <= 8 * 2 * step * scale
        _, imgs_f = self._scan("floor", deltas)
        jumps_f = np.abs(np.diff(imgs_f, axis=0)).max(axis=(1, 2))
        assert jumps_f.max() > 50 * jumps.max()

    def test_position_gradient_matches_finite_difference(self):
        render, _ = self._scan("interp", [0.0])
        weights = jnp.asarray(np.random.default_rng(0).normal(size=SHAPE),
                              jnp.float32)

        def loss(d):
            return jnp.sum(render(d) * weights)
        for d0 in (0.137, 0.41, 0.73):
            g = float(jax.grad(loss)(jnp.float32(d0)))
            h = 1e-3
            fd = float((loss(jnp.float32(d0 + h)) - loss(jnp.float32(d0 - h)))
                       / (2 * h))
            assert g != 0.0
            assert abs(g - fd) <= 2e-2 * abs(fd) + 1e-3

    def test_unknown_mode_raises(self):
        stack, grid_wl, wl, flux, x, y = _random_case(8, 8, 2, 1)
        with pytest.raises(ValueError, match="deposit"):
            star_disperser.deposit_stack_native(
                stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE),
                oversample=4, chunk_size=2, deposit="bilinear")


class TestRebinCounts:

    def test_sums_and_centres(self):
        wl = jnp.arange(9000.0, 9000.0 + 2 * 11, 2.0) / 1e4   # 11 samples
        counts = jnp.arange(11, dtype=jnp.float32)[None, :] * jnp.ones((3, 1))
        cb, wb = rebin_counts(counts, wl, 2)
        assert cb.shape == (3, 6) and wb.shape == (6,)
        np.testing.assert_allclose(np.asarray(cb).sum(axis=1),
                                   np.asarray(counts).sum(axis=1))
        np.testing.assert_allclose(np.asarray(cb[0]), [1, 5, 9, 13, 17, 10])
        # bin centres; the last, partial bin sits on its single member
        np.testing.assert_allclose(
            np.asarray(wb) * 1e4, [9001, 9005, 9009, 9013, 9017, 9020],
            rtol=1e-6)

    def test_factor_one_is_identity(self):
        wl = jnp.linspace(1.0, 2.0, 7)
        c = jnp.arange(7.0)
        cb, wb = rebin_counts(c, wl, 1)
        assert cb is c and wb is wl

    def test_rebin_factor(self):
        assert sed_rebin_factor(2.0, None) == 1
        assert sed_rebin_factor(2.0, 4.0) == 2
        assert sed_rebin_factor(2.0, 2.0) == 1
        with pytest.raises(ValueError):
            sed_rebin_factor(2.0, 5.0)

    def test_rebin_factors_per_order(self):
        orders = ("0", "1", "2")
        assert sed_rebin_factors(2.0, None, orders) == {"0": 1, "1": 1, "2": 1}
        assert sed_rebin_factors(2.0, 4.0, orders) == {"0": 2, "1": 2, "2": 2}
        # int keys (unquoted YAML) and orders left out (not rebinned)
        assert sed_rebin_factors(2.0, {0: 4, "1": 4}, orders) == {
            "0": 2, "1": 2, "2": 1}
        assert sed_rebin_factors(2.0, {"1": 4}, ("1",)) == {"1": 2}
        with pytest.raises(ValueError):
            sed_rebin_factors(2.0, {"3": 4}, orders)
        with pytest.raises(ValueError):
            sed_rebin_factors(2.0, {"2": 5}, orders)

    def test_parse_rebin_dlam(self):
        assert parse_rebin_dlam(None) is None
        assert parse_rebin_dlam("4") == 4.0
        assert parse_rebin_dlam("0:4, 1:4,2:2") == {"0": 4.0, "1": 4.0, "2": 2.0}


def _mock_psf(n_wl=5, size=20):
    rng = np.random.default_rng(42)
    grid = rng.uniform(0, 1, (2, 2, n_wl, size, size)).astype(np.float32)
    grid /= grid.sum(axis=(-2, -1), keepdims=True)
    return {
        'psf_grid': jnp.asarray(grid),
        'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
        'spatial_x': jnp.linspace(1, 4088, 2),
        'spatial_y': jnp.linspace(1, 4088, 2),
        'oversample': 4,
    }


@pytest.fixture(scope="module")
def grism_payloads():
    from roman_disperser.optical_model import RomanOpticalModel
    from roman_disperser import elements, paths
    model = RomanOpticalModel(str(paths.optical_model_path(element=elements.get_element("grism"))))
    return _mock_psf(), omj.make_sca_payload(model, sca=5, order="1")


class TestBatchPath:

    def _stars(self, psf, opt, deposit, hoist, rebin):
        wl = jnp.linspace(1.0, 1.8, 41).astype(jnp.float32)
        sens = jnp.ones_like(wl)
        fn = star_disperser.make_star_disperser(psf, opt, deposit=deposit,
                                                hoist_psf=hoist)
        run = make_batched_star_fori(fn, sens, wl, 2.0, sed_rebin=rebin)
        spectra = jnp.ones((3, 41), jnp.float32)
        xs = jnp.array([1000.3, 2044.71, 3000.12], jnp.float32)
        ys = jnp.array([900.55, 2044.2, 1500.9], jnp.float32)
        return np.asarray(run(3, spectra, xs, ys,
                              jnp.zeros((4088, 4088), jnp.float32)),
                          np.float64)

    @pytest.mark.parametrize("deposit", ["floor", "interp"])
    def test_hoist_matches_unhoisted(self, grism_payloads, deposit):
        psf, opt = grism_payloads
        a = self._stars(psf, opt, deposit, False, 1)
        b = self._stars(psf, opt, deposit, True, 1)
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6 * a.max())

    def test_rebin_conserves_flux(self, grism_payloads):
        psf, opt = grism_payloads
        a = self._stars(psf, opt, "interp", False, 1)
        b = self._stars(psf, opt, "interp", True, 2)
        np.testing.assert_allclose(b.sum(), a.sum(), rtol=1e-5)

    def test_galaxy_interp_flux(self, grism_payloads):
        psf, opt = grism_payloads
        wl = jnp.linspace(1.0, 1.8, 21).astype(jnp.float32)
        sens = jnp.ones_like(wl)
        img = jnp.zeros((16, 16), jnp.float32).at[7:9, 6:10].set(1.0 / 8)
        out = {}
        for dep in ("floor", "interp"):
            fn = galaxy_disperser.make_galaxy_disperser(psf, opt, deposit=dep)
            run = make_batched_galaxy_fori(fn, sens, wl, 2.0)
            out[dep] = np.asarray(run(
                1, jnp.ones((1, 21), jnp.float32),
                jnp.array([2044.37], jnp.float32),
                jnp.array([2044.81], jnp.float32), img[None],
                jnp.zeros((4088, 4088), jnp.float32)), np.float64)
        np.testing.assert_allclose(out["interp"].sum(), out["floor"].sum(),
                                   rtol=1e-5)
        assert not np.allclose(out["interp"], out["floor"])


class TestExtendedTable:
    """``deposit="interp"`` reads an extended phase table (one contiguous
    block per sample); ``"interp_gather"`` is the pre-2026-11 computation of
    the same deposit. They must agree to float32 summation order."""

    @pytest.mark.parametrize("s_y,s_x,os_", [
        (12, 12, 4), (9, 9, 4), (10, 13, 4), (11, 8, 4), (7, 10, 2),
        (47, 47, 4)])
    def test_extend_equals_direct(self, s_y, s_x, os_):
        rng = np.random.default_rng(s_y * 31 + s_x)
        stack = jnp.asarray(rng.uniform(0, 1, (2, 3, s_y, s_x)), jnp.float32)
        direct = star_disperser.bin_stack_phases(stack, os_, extended=True)
        via = star_disperser.extend_phases(
            star_disperser.bin_stack_phases(stack, os_))
        assert direct.shape == via.shape
        assert direct.shape[:2] == (os_ + 1, os_ + 1)
        np.testing.assert_array_equal(np.asarray(direct), np.asarray(via))

    @pytest.mark.parametrize("s_y,s_x,os_", [
        (12, 12, 4), (9, 9, 4), (10, 13, 4), (11, 8, 4), (7, 10, 2)])
    @pytest.mark.parametrize("chunk", [5, 64])
    def test_matches_gather(self, s_y, s_x, os_, chunk):
        stack, grid_wl, wl, flux, x, y = _random_case(s_y, s_x, 301,
                                                      7 * s_y + s_x)
        # phase-aligned and wrap positions (p0 = -1 and p0 = os - 1)
        extra = jnp.array([30.0, 30.01, 30.125, 30.24, 30.5, 30.99, 31.0],
                          jnp.float32)
        x = jnp.concatenate([x, extra, extra[::-1]])
        y = jnp.concatenate([y, extra[::-1], extra])
        wl = jnp.concatenate([wl, wl[:14]])
        flux = jnp.concatenate([flux, flux[:14]])
        out = {}
        for d in ("interp", "interp_gather"):
            out[d] = np.asarray(star_disperser.deposit_stack_native(
                stack, grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32),
                oversample=os_, chunk_size=chunk, deposit=d), np.float64)
        peak = out["interp_gather"].max()
        assert np.abs(out["interp"] - out["interp_gather"]).max() <= 1e-6 * peak
        np.testing.assert_allclose(out["interp"].sum(),
                                   out["interp_gather"].sum(), rtol=1e-6)

    def test_standard_table_is_extended_once(self):
        stack, grid_wl, wl, flux, x, y = _random_case(12, 12, 50, 3)
        std = star_disperser.bin_stack_phases(stack, 4)
        ext = star_disperser.bin_stack_phases(stack, 4, extended=True)
        args = (grid_wl, wl, flux, x, y, jnp.zeros(SHAPE, jnp.float32))
        a = star_disperser.deposit_binned(std, (12, 12), *args, oversample=4,
                                          chunk_size=16, deposit="interp")
        b = star_disperser.deposit_binned(ext, (12, 12), *args, oversample=4,
                                          chunk_size=16, deposit="interp")
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_native_stamps_rejects_standard_table(self):
        stack, grid_wl, wl, flux, x, y = _random_case(12, 12, 4, 4)
        std = star_disperser.bin_stack_phases(stack, 4)
        with pytest.raises(ValueError, match="extended"):
            star_disperser.native_stamps(std, (12, 12), grid_wl, wl, flux, x,
                                         y, 4, "interp")

    def test_position_gradient_matches_gather(self):
        stack, grid_wl, wl, flux, x, y = _random_case(10, 10, 40, 5)
        w = jnp.asarray(np.random.default_rng(6).uniform(0, 1, SHAPE),
                        jnp.float32)

        def loss(dx, d):
            o = star_disperser.deposit_stack_native(
                stack, grid_wl, wl, flux, x + dx, y + 0.5 * dx,
                jnp.zeros(SHAPE, jnp.float32), oversample=4, chunk_size=16,
                deposit=d)
            return jnp.sum(o * w)

        for dx in (0.0, 0.037, 0.2):
            g_new = jax.grad(loss)(jnp.float32(dx), "interp")
            g_old = jax.grad(loss)(jnp.float32(dx), "interp_gather")
            np.testing.assert_allclose(float(g_new), float(g_old),
                                       rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("hoist", [False, True])
    def test_stars_batch_matches_gather(self, grism_payloads, hoist):
        psf, opt = grism_payloads
        a = TestBatchPath()._stars(psf, opt, "interp", hoist, 1)
        b = TestBatchPath()._stars(psf, opt, "interp_gather", hoist, 1)
        assert np.abs(a - b).max() <= 1e-6 * b.max()
        np.testing.assert_allclose(a.sum(), b.sum(), rtol=1e-6)

    def test_galaxy_batch_matches_gather(self, grism_payloads):
        psf, opt = grism_payloads
        wl = jnp.linspace(1.0, 1.8, 21).astype(jnp.float32)
        sens = jnp.ones_like(wl)
        img = jnp.zeros((16, 16), jnp.float32).at[7:9, 6:10].set(1.0 / 8)
        out = {}
        for dep in ("interp", "interp_gather"):
            fn = galaxy_disperser.make_galaxy_disperser(psf, opt, deposit=dep)
            run = make_batched_galaxy_fori(fn, sens, wl, 2.0)
            out[dep] = np.asarray(run(
                1, jnp.ones((1, 21), jnp.float32),
                jnp.array([2044.37], jnp.float32),
                jnp.array([1500.81], jnp.float32), img[None],
                jnp.zeros((4088, 4088), jnp.float32)), np.float64)
        b = out["interp_gather"]
        assert np.abs(out["interp"] - b).max() <= 1e-6 * b.max()
        np.testing.assert_allclose(out["interp"].sum(), b.sum(), rtol=1e-6)

    def test_default_chunk_size(self):
        assert star_disperser.default_chunk_size("cpu") == 2000
        assert star_disperser.default_chunk_size("gpu") == 8000
        assert star_disperser.default_chunk_size("cuda") == 8000

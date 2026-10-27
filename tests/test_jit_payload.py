"""
Tests for payload-as-argument JIT (``jit_payload`` and its users).

Guards the property that motivated it: dispersers for *different SCAs* share
one compiled program (the payload arrays are jit arguments, not constants),
and the result equals the old closure-captured path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import galaxy_disperser, jit_payload, star_disperser
from roman_disperser.pipeline import (
    make_batched_galaxy_fori, make_batched_star_fori,
    run_galaxy_batch, run_star_batch,
)


def _closure_star_fori(disperser_fn, sens, wl, dlam):
    """Reference: the pre-0.16 closure-captured batched star loop."""
    @jax.jit
    def run(n_sources, spectra, xsca, ysca, output):
        def body_fn(i, output):
            counts = spectra[i] * sens * dlam
            return disperser_fn(xsca[i], ysca[i], wl, counts, output)
        return jax.lax.fori_loop(0, n_sources, body_fn, output)
    return run


def _closure_galaxy_fori(disperser_fn, sens, wl, dlam):
    """Reference: the pre-0.16 closure-captured batched galaxy loop."""
    @jax.jit
    def run(n_sources, spectra, xsca, ysca, images, output):
        def body_fn(i, output):
            counts = spectra[i] * sens * dlam
            return disperser_fn(images[i], xsca[i], ysca[i], counts, wl, output)
        return jax.lax.fori_loop(0, n_sources, body_fn, output)
    return run


def _mock_psf(detector, scale):
    """Small mock PSF payload; ``scale`` makes the arrays SCA-specific."""
    n_y, n_x, n_wl, size = 2, 2, 5, 20
    rng = np.random.default_rng(int(scale * 100))
    grid = rng.uniform(0, 1, (n_y, n_x, n_wl, size, size)).astype(np.float32)
    grid /= grid.sum(axis=(-2, -1), keepdims=True)
    return {
        'psf_grid': jnp.asarray(grid),
        'wavelengths': jnp.linspace(1.0, 1.8, n_wl),
        'wl_grid': jnp.linspace(1.0, 1.8, n_wl),
        'spatial_x': jnp.linspace(1, 4088, n_x),
        'spatial_y': jnp.linspace(1, 4088, n_y),
        'psf_fov_pixels': size,
        'pixel_scale': 0.11,
        'oversample': 4,
        'detector': detector,
        'order': '1',
        'stpsf_filter': 'GRISM0',
        'timing': {'total_time': scale, 'n_psfs': 3},
    }


@pytest.fixture(scope="module")
def two_scas(optical_model):
    """(psf, optical) payload pairs for two SCAs of the same element."""
    return [
        (_mock_psf('WFI05', 1.0), omj.make_sca_payload(optical_model, sca=5, order="1")),
        (_mock_psf('WFI11', 2.0), omj.make_sca_payload(optical_model, sca=11, order="1")),
    ]


class TestSplitMerge:

    def test_roundtrip_and_labels_dropped(self, two_scas):
        psf, opt = two_scas[0]
        arrays, static = jit_payload.split_payloads(psf, opt)
        psf2, opt2 = jit_payload.merge_payloads(arrays, static)
        for k in ('psf_grid', 'wavelengths', 'spatial_x', 'spatial_y'):
            assert psf2[k] is psf[k]
        assert psf2['oversample'] == psf['oversample']
        for k in ('detector', 'order', 'stpsf_filter', 'timing', 'pixel_scale'):
            assert k not in psf2
        assert opt2['wl']['transform'] == opt['wl']['transform']
        assert opt2['poly']['C_ijk'] is opt['poly']['C_ijk']
        assert opt2['poly']['map_i'] == opt['poly']['map_i']

    def test_static_identical_across_scas(self, two_scas):
        """Different SCAs must give equal (hashable) static parts."""
        (_, s1), (_, s2) = [jit_payload.split_payloads(*p) for p in two_scas]
        assert s1 == s2 and hash(s1) == hash(s2)


def _compiles(fn, call):
    """Number of new entries in ``fn``'s jit cache caused by ``call()``."""
    before = fn._cache_size()
    jax.block_until_ready(call())
    return fn._cache_size() - before


class TestOneCompileAcrossSCAs:

    WL = jnp.linspace(1.1, 1.7, 40)
    FLUX = jnp.ones(40)

    def test_star_factory(self, two_scas):
        out = jnp.zeros((4088, 4088), jnp.float32)
        d1 = star_disperser.make_star_disperser(*two_scas[0])
        d2 = star_disperser.make_star_disperser(*two_scas[1])
        jax.block_until_ready(d1(2000.0, 2000.0, self.WL, self.FLUX, out))
        assert _compiles(star_disperser.disperse_star_jit,
                         lambda: d2(2000.0, 2000.0, self.WL, self.FLUX, out)) == 0

    def test_star_matches_closure(self, two_scas):
        """New path == old closure-captured path."""
        psf, opt = two_scas[1]
        out = jnp.zeros((4088, 4088), jnp.float32)
        new = star_disperser.make_star_disperser(psf, opt)(
            1500.0, 2500.0, self.WL, self.FLUX, out)
        old = jax.jit(lambda x, y, w, f, o: star_disperser.disperse_star_psf(
            psf, opt, x, y, w, f, o))(1500.0, 2500.0, self.WL, self.FLUX, out)
        assert float(old.sum()) > 0
        np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-7 * float(old.max()))

    def test_batched_star_and_galaxy(self, two_scas):
        batch, npix = 2, 16
        out = jnp.zeros((4088, 4088), jnp.float32)
        spec = jnp.ones((batch, 40), jnp.float32)
        xs = jnp.array([1800.0, 2200.0], jnp.float32)
        ys = jnp.array([2000.0, 2100.0], jnp.float32)
        imgs = jnp.ones((batch, npix, npix), jnp.float32) / npix**2

        def build(psf, opt, sens):
            sd = star_disperser.make_star_disperser(psf, opt)
            gd = galaxy_disperser.make_galaxy_disperser(psf, opt)
            return (make_batched_star_fori(sd, sens, self.WL, 5.0),
                    make_batched_galaxy_fori(gd, sens, self.WL, 5.0),
                    _closure_star_fori(sd, sens, self.WL, 5.0),
                    _closure_galaxy_fori(gd, sens, self.WL, 5.0))

        s1, g1, _, _ = build(*two_scas[0], jnp.ones(40))
        jax.block_until_ready(s1(batch, spec, xs, ys, out))
        jax.block_until_ready(g1(batch, spec, xs, ys, imgs, out))

        # SCA 2 (different payload arrays AND sensitivity): no new compile
        s2, g2, s2_old, g2_old = build(*two_scas[1], 0.7 * jnp.ones(40))
        assert _compiles(run_star_batch,
                         lambda: s2(batch, spec, xs, ys, out)) == 0
        assert _compiles(run_galaxy_batch,
                         lambda: g2(batch, spec, xs, ys, imgs, out)) == 0

        for new, old in [(s2(batch, spec, xs, ys, out), s2_old(batch, spec, xs, ys, out)),
                         (g2(batch, spec, xs, ys, imgs, out),
                          g2_old(batch, spec, xs, ys, imgs, out))]:
            assert float(old.sum()) > 0
            np.testing.assert_allclose(new, old, rtol=1e-5,
                                       atol=1e-7 * float(old.max()))


def test_batched_fori_rejects_non_factory_disperser():
    """A callable without ``.payload_args`` must fail loudly (v0.16.0), not
    fall back to a closure that recompiles per SCA."""
    sens, wl = jnp.ones(4), jnp.linspace(1.0, 1.8, 4)
    with pytest.raises(TypeError, match="make_star_disperser"):
        make_batched_star_fori(lambda *a: a[-1], sens, wl, 5.0)
    with pytest.raises(TypeError, match="make_galaxy_disperser"):
        make_batched_galaxy_fori(lambda *a: a[-1], sens, wl, 5.0)

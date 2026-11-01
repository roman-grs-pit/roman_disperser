"""Recompilation triggers off the production path.

* Factory dispersers called with Python float / np.float32 / np.float64 /
  jnp.float32 positions share one compiled program.
* PSF payload grids are float32 even with ``jax_enable_x64`` on, so the
  payload is not mixed precision and does not key a separate compile.
* ``build_dispersed_image.check_psf_payloads`` refuses PSF caches of another
  grid geometry within one run.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser import galaxy_disperser, psf_model, star_disperser

SCRIPTS = Path(__file__).parent.parent / "scripts"
WL = jnp.linspace(1.1, 1.7, 40)
FLUX = jnp.ones(40)


def _mock_psf(size=20, n_wl=5, float64=False):
    rng = np.random.default_rng(size)
    grid = rng.uniform(0, 1, (2, 2, n_wl, size, size)).astype(np.float32)
    grid /= grid.sum(axis=(-2, -1), keepdims=True)
    arr = (lambda v: np.asarray(v, np.float64)) if float64 else jnp.asarray
    return {
        'psf_grid': jnp.asarray(grid),
        'wavelengths': arr(np.linspace(1.0, 1.8, n_wl)),
        'wl_grid': arr(np.linspace(1.0, 1.8, n_wl)),
        'spatial_x': arr(np.linspace(1, 4088, 2)),
        'spatial_y': arr(np.linspace(1, 4088, 2)),
        'psf_fov_pixels': size, 'pixel_scale': 0.11, 'oversample': 4,
        'detector': 'WFI05', 'order': '1', 'stpsf_filter': 'GRISM0',
        'timing': {'total_time': 1.0, 'n_psfs': 4},
    }


@pytest.fixture(scope="module")
def payloads(optical_model):
    return _mock_psf(), omj.make_sca_payload(optical_model, sca=5, order="1")


POSITIONS = [2000.0, np.float32(2000.5), np.float64(2001.0),
             jnp.float32(2001.5), jnp.asarray(2002.0)]


def _new_compiles(fn, calls):
    before = fn._cache_size()
    for call in calls:
        jax.block_until_ready(call())
    return fn._cache_size() - before


def test_star_factory_one_compile_for_all_position_types(payloads):
    sd = star_disperser.make_star_disperser(*payloads)
    out = jnp.zeros((4088, 4088), jnp.float32)
    first = _new_compiles(star_disperser.disperse_star_jit,
                          [lambda: sd(2000.0, 2000.0, WL, FLUX, out)])
    assert first <= 1
    calls = [lambda p=p: sd(p, p, WL, FLUX, out) for p in POSITIONS]
    assert _new_compiles(star_disperser.disperse_star_jit, calls) == 0


def test_galaxy_factory_one_compile_for_all_position_types(payloads):
    gd = galaxy_disperser.make_galaxy_disperser(*payloads)
    out = jnp.zeros((4088, 4088), jnp.float32)
    img = jnp.ones((16, 16), jnp.float32) / 256
    first = _new_compiles(galaxy_disperser.disperse_galaxy_jit,
                          [lambda: gd(img, 2000.0, 2000.0, FLUX, WL, out)])
    assert first <= 1
    calls = [lambda p=p: gd(img, p, p, FLUX, WL, out) for p in POSITIONS]
    assert _new_compiles(galaxy_disperser.disperse_galaxy_jit, calls) == 0


def test_psf_payload_float32_under_x64(tmp_path, payloads):
    path = tmp_path / "psf.npz"
    psf_model.save_psf_payload(_mock_psf(float64=True), path, verbose=False)
    assert np.load(path)["wavelengths"].dtype == np.float64
    with jax.enable_x64(True):
        p = psf_model.load_psf_payload(path, verbose=False)
        for k in ("psf_grid", "wavelengths", "wl_grid", "spatial_x",
                  "spatial_y"):
            assert p[k].dtype == jnp.float32, k
        # Same leaf dtypes as a payload loaded with x64 off: x64 does not
        # key a separate compile of the disperser through the PSF payload.
        sd = star_disperser.make_star_disperser(p, payloads[1])
        arrays_x64 = sd.payload_args[0][0]
    p32 = psf_model.load_psf_payload(path, verbose=False)
    arrays_32 = star_disperser.make_star_disperser(
        p32, payloads[1]).payload_args[0][0]
    assert jax.tree.map(lambda a: a.dtype, arrays_x64) == \
        jax.tree.map(lambda a: a.dtype, arrays_32)


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image_recompiles",
        SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_psf_geometry_check(bdi, payloads):
    psf, opt = payloads
    geom = bdi.check_psf_payloads({"0": psf, "1": psf})
    assert geom == ((2, 2, 5, 20, 20), 4)
    other = _mock_psf(size=24)
    with pytest.raises(ValueError, match="psf_grid shape"):
        bdi.check_psf_payloads({"0": psf, "1": other})
    with pytest.raises(ValueError, match="WFI07 order 1"):
        bdi.check_psf_payloads({"1": other}, geom, "WFI07")
    sens = jnp.ones(40)
    with pytest.raises(ValueError, match="grid geometry"):
        bdi.make_order_fori_fns(
            {"0": psf, "1": other}, {"0": opt, "1": opt},
            {"0": sens, "1": sens}, WL, 150.0, ("0", "1"))
    # matching geometry passes and builds the functions
    s, g = bdi.make_order_fori_fns({"1": psf}, {"1": opt}, {"1": sens}, WL,
                                   150.0, ("1",), psf_geometry=geom)
    assert set(s) == set(g) == {"1"}

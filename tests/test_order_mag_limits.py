"""Per-order brightness cut (``order_mag_limits`` / ``--order-mag-limits``).

Parsing and the mask helpers in ``pipeline``, and the record in the FITS
header, meta YAML and per-order source counts through ``process_pointing``
(with the disperser, PSF and trace calls replaced by light fakes).
"""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits

from roman_disperser import elements
from roman_disperser.pipeline import (
    apply_order_mag_limits, flux_to_ab_mag, order_mag_limits_by_order,
    parse_order_mag_limits,
)

SCRIPTS = Path(__file__).parent.parent / "scripts"
ORDERS = ("0", "1", "2")


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image_mag", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- parsing / normalisation --------------------------------------------------

def test_parse():
    assert parse_order_mag_limits(None) is None
    assert parse_order_mag_limits("0:24,2:24") == {"0": 24.0, "2": 24.0}
    assert parse_order_mag_limits(" 0 : 22.5 , 2:26") == {"0": 22.5, "2": 26.0}
    for bad in ("24", "0:24,2", ":24", "0:abc"):
        with pytest.raises(ValueError):
            parse_order_mag_limits(bad)


def test_by_order():
    assert order_mag_limits_by_order(None, ORDERS) == {}
    assert order_mag_limits_by_order({}, ORDERS) == {}
    assert order_mag_limits_by_order({0: 24, "2": "23.5", "1": None},
                                     ORDERS) == {"0": 24.0, "2": 23.5}
    with pytest.raises(ValueError, match="not in"):
        order_mag_limits_by_order({"3": 24}, ORDERS)
    with pytest.raises(ValueError, match="not in"):       # prism: order 1 only
        order_mag_limits_by_order({"0": 24}, ("1",))
    with pytest.raises(ValueError, match="finite"):
        order_mag_limits_by_order({"0": float("nan")}, ORDERS)
    with pytest.raises(TypeError):
        order_mag_limits_by_order(24.0, ORDERS)


def test_flux_to_ab_mag():
    m = flux_to_ab_mag([1.0, 1e-8, 10 ** (-0.4 * 24.3), 0.0, -1.0, np.nan])
    np.testing.assert_allclose(m[:3], [0.0, 20.0, 24.3])
    assert np.all(np.isposinf(m[3:]))       # non-positive and NaN flux: +inf


def test_apply_masks():
    mag = np.array([18.0, 24.0, 24.01, 30.0, np.inf, np.nan])
    on = {"0": np.array([1, 1, 1, 1, 1, 1], bool),
          "1": np.array([1, 1, 1, 1, 1, 0], bool),
          "2": np.array([0, 1, 1, 0, 1, 1], bool)}
    masks, skipped = apply_order_mag_limits(on, mag, {"0": 24.0, "2": 24.0})
    np.testing.assert_array_equal(masks["0"], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(skipped["0"], [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(masks["1"], on["1"])      # not limited
    assert not skipped["1"].any() and masks["1"] is not on["1"]
    np.testing.assert_array_equal(masks["2"], [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(skipped["2"], [0, 0, 1, 0, 1, 1])
    for o in ORDERS:                      # partition of the on-detector set
        np.testing.assert_array_equal(masks[o] | skipped[o], on[o])
        assert not (masks[o] & skipped[o]).any()
    # no limits: identity
    masks, skipped = apply_order_mag_limits(on, mag, {})
    for o in ORDERS:
        np.testing.assert_array_equal(masks[o], on[o])
        assert not skipped[o].any()


def test_example_config_and_cli(bdi):
    cfg = yaml.safe_load(bdi.EXAMPLE_CONFIG.replace(
        "# order_mag_limits:", "order_mag_limits:"))
    assert order_mag_limits_by_order(cfg["order_mag_limits"], ORDERS) == \
        {"0": 24.0, "2": 24.0}
    import inspect
    for fn in (bdi.setup_pipeline, bdi.build_dispersed_image):
        assert inspect.signature(fn).parameters["order_mag_limits"].default is None


# -- process_pointing record --------------------------------------------------

MAGS = np.array([16.0, 20.0, 23.0, 25.0, 19.0, 24.5, 21.0])
TYPES = ["PSF", "PSF", "PSF", "PSF", "SER", "SER", "SER"]
N_WL = 8


def _fake_pipeline(limits):
    n = len(MAGS)
    meta = pd.DataFrame({
        "ra": np.full(n, 10.0), "dec": np.full(n, 0.1), "type": TYPES,
        "F158": 10 ** (-0.4 * MAGS), "flux_scale": 10 ** (-0.4 * MAGS),
        "sed_index": np.zeros(n, int), "half_light_radius": np.full(n, 0.3),
        "n": np.ones(n), "ba": np.ones(n), "pa": np.zeros(n),
    })
    el = elements.get_element("grism")
    wl = np.linspace(0.9, 2.0, N_WL).astype(np.float32)
    return {
        "model": None, "element": el, "meta": meta, "store": None,
        "star_seds_all": np.ones((1, N_WL), np.float32), "wl_mask": None,
        "wavelengths_um": wl, "wavelengths_ang": wl * 1e4,
        "wavelengths_jax": jnp.asarray(wl), "dlam_angstroms": 2.0,
        "sca_list": [5],
        "sca_data": {5: {"optical_payloads": {o: {} for o in el.orders},
                         "sensitivities": {o: None for o in el.orders}}},
        "psf_cache_dir": "unused", "optical_model_file": "fake.yaml",
        "star_batch_size": 4, "galaxy_batch_size": 4, "galaxy_npix": 30,
        "galaxy_npix_os": 8, "deposit": "interp",
        "sed_rebin": {"0": 20, "1": 2, "2": 1}, "hoist_star_psf": True,
        "order_mag_limits": order_mag_limits_by_order(limits, el.orders),
        "oversample": 4, "n_catalog_rows": n, "source_cuts": [],
        "timings": {},
    }


class _Fn:
    """Stands in for a batched disperser: remembers its order."""

    def __init__(self, order):
        self.order = order

    def __call__(self, *args):
        return args[-1]


def _patch(monkeypatch, bdi, calls):
    size = 16
    monkeypatch.setattr(bdi, "DETECTOR_SIZE", size)
    monkeypatch.setattr(bdi.psf_model, "get_or_make_psf_payload",
                        lambda **kw: {})
    monkeypatch.setattr(
        bdi, "make_order_fori_fns",
        lambda *a, **kw: ({o: _Fn(o) for o in ORDERS},
                          {o: _Fn(o) for o in ORDERS}))
    n = len(MAGS)
    monkeypatch.setattr(bdi.omj, "get_fpa_pos",
                        lambda ra, dec, *a: (jnp.zeros(len(ra)),
                                             jnp.zeros(len(ra))))
    monkeypatch.setattr(bdi.omj, "fpa_to_sca",
                        lambda p, x, y: (jnp.full(x.shape, 8.0),
                                         jnp.full(x.shape, 8.0)))
    # every source on the detector in every order
    monkeypatch.setattr(
        bdi, "select_sources_per_order",
        lambda p, x, y, orders, lo, hi: (
            {o: np.ones(n, bool) for o in orders}, np.ones(n, bool)))
    monkeypatch.setattr(bdi, "load_galaxy_seds",
                        lambda store, m, wl_mask: np.ones((len(m), N_WL),
                                                          np.float32))

    def stars(fn, spec, x, y, out, bs):
        calls.append((fn.order, "PSF", len(x)))
        return out + len(x)

    def gals(fn, spec, x, y, ser, npix, out, bs):
        calls.append((fn.order, "SER", len(x)))
        return out + len(x)
    monkeypatch.setattr(bdi, "disperse_batched_stars", stars)
    monkeypatch.setattr(bdi, "disperse_batched_sersic_galaxies", gals)
    monkeypatch.setattr(bdi, "write_png", lambda *a, **kw: None)


def _run(bdi, tmp_path, limits, monkeypatch):
    calls = []
    _patch(monkeypatch, bdi, calls)
    out_dir = tmp_path / "pt"
    bdi.process_pointing(_fake_pipeline(limits), 10.0, 0.1, 0.0, str(out_dir),
                         cone_radius=1.0, verbose=False)
    fits_path, = out_dir.glob("*_detSCA05.fits")
    meta_path, = out_dir.glob("*_meta.yaml")
    manifest, = out_dir.glob("*_sources.parquet")
    with fits.open(fits_path) as h:
        hdr = h[0].header.copy()
    return calls, hdr, yaml.safe_load(meta_path.read_text()), \
        pd.read_parquet(manifest)


def test_process_pointing_records_cut(bdi, tmp_path, monkeypatch):
    calls, hdr, meta, man = _run(bdi, tmp_path, {"0": 22, "2": 24.0},
                                 monkeypatch)
    got = {(o, t): n for o, t, n in calls}
    # order 0 (<= 22): stars 16, 20; galaxies 19, 21
    assert got[("0", "PSF")] == 2 and got[("0", "SER")] == 2
    # order 1 uncut
    assert got[("1", "PSF")] == 4 and got[("1", "SER")] == 3
    # order 2 (<= 24): stars 16, 20, 23; galaxies 19, 21
    assert got[("2", "PSF")] == 3 and got[("2", "SER")] == 2
    assert hdr["ORDMAG0"] == 22.0 and hdr["ORDMAG2"] == 24.0
    assert "ORDMAG1" not in hdr
    assert hdr["DEPDLAM0"] == 40.0
    assert meta["order_mag_limits"] == {"0": 22.0, "2": 24.0}
    sc = meta["source_counts"]["SCA5"]
    assert sc["0"] == {"stars": 2, "galaxies": 2,
                       "stars_mag_cut": 2, "galaxies_mag_cut": 1}
    assert sc["1"] == {"stars": 4, "galaxies": 3}
    assert sc["2"] == {"stars": 3, "galaxies": 2,
                       "stars_mag_cut": 1, "galaxies_mag_cut": 1}
    assert man.groupby("order").size().to_dict() == {"0": 4, "1": 7, "2": 5}
    assert (-2.5 * np.log10(man[man.order == "0"].F158)).max() <= 22.0 + 1e-6


def test_process_pointing_no_cut(bdi, tmp_path, monkeypatch):
    calls, hdr, meta, man = _run(bdi, tmp_path, None, monkeypatch)
    got = {(o, t): n for o, t, n in calls}
    for o in ORDERS:
        assert got[(o, "PSF")] == 4 and got[(o, "SER")] == 3
    assert not any(k.startswith("ORDMAG") for k in hdr)
    assert meta["order_mag_limits"] == {}
    for o in ORDERS:
        assert meta["source_counts"]["SCA5"][o] == {"stars": 4, "galaxies": 3}


def test_cut_every_order_drops_source(bdi, tmp_path, monkeypatch):
    """A source cut from all orders leaves the on-detector set (no manifest row)."""
    calls, hdr, meta, man = _run(bdi, tmp_path, {"0": 20, "1": 20, "2": 20},
                                 monkeypatch)
    assert sorted(set(man.catalog_index)) == [0, 1, 4]
    assert hdr["ORDMAG1"] == 20.0

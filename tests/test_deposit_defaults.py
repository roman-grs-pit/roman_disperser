"""Production deposit defaults: interp, per-order count rebinning, star hoist.

The defaults live in ``scripts/build_dispersed_image.py`` (``DEFAULT_DEPOSIT``,
``DEFAULT_REBIN_DLAM``, ``DEFAULT_HOIST_STAR_PSF``); the historical setting
(floor, no rebinning, no hoist) stays reachable from the CLI and the config.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import yaml

from roman_disperser import galaxy_disperser, pipeline, star_disperser
from roman_disperser.pipeline import parse_rebin_dlam, sed_rebin_factors

SCRIPTS = Path(__file__).parent.parent / "scripts"
GRISM = ("0", "1", "2")


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image_defaults", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_library_defaults_are_interp():
    for fn in (star_disperser.make_star_disperser,
               star_disperser.disperse_star_psf,
               star_disperser.deposit_binned,
               star_disperser.deposit_stack_native,
               galaxy_disperser.make_galaxy_disperser,
               galaxy_disperser.disperse_galaxy,
               pipeline.run_star_batch, pipeline.run_galaxy_batch):
        sig = inspect.signature(getattr(fn, "__wrapped__", fn))
        assert sig.parameters["deposit"].default == "interp", fn.__name__
    # low-level factories keep the hoist off and do not rebin
    sig = inspect.signature(star_disperser.make_star_disperser)
    assert sig.parameters["hoist_psf"].default is False
    for fn in (pipeline.make_batched_star_fori, pipeline.make_batched_galaxy_fori):
        assert inspect.signature(fn).parameters["sed_rebin"].default == 1


def test_script_defaults(bdi):
    assert bdi.DEFAULT_DEPOSIT == "interp"
    assert bdi.DEFAULT_REBIN_DLAM == {"0": 40.0, "1": 4.0, "2": 2.0}
    assert bdi.DEFAULT_HOIST_STAR_PSF is True
    for fn in (bdi.setup_pipeline, bdi.build_dispersed_image):
        p = inspect.signature(fn).parameters
        assert p["deposit"].default is None
        assert p["rebin_dlam"].default is None
        assert p["hoist_star_psf"].default is None
        assert p["order_mag_limits"].default is None     # cut stays off


def test_resolve_rebin_dlam(bdi):
    r = bdi.resolve_rebin_dlam
    assert r(None, GRISM, 2.0) == {"0": 40.0, "1": 4.0, "2": 2.0}
    assert sed_rebin_factors(2.0, r(None, GRISM, 2.0), GRISM) == \
        {"0": 20, "1": 2, "2": 1}
    assert sed_rebin_factors(1.0, r(None, GRISM, 1.0), GRISM) == \
        {"0": 40, "1": 4, "2": 2}
    assert r(None, ("1",), 2.0) == {"1": 4.0}               # prism
    # defaults round down to a multiple of the catalog spacing (>= 1 sample)
    assert r(None, GRISM, 3.0) == {"0": 39.0, "1": 3.0, "2": 3.0}
    # switching off
    for off in ("none", "None", " NONE ", False):
        assert r(off, GRISM, 2.0) is None
        assert sed_rebin_factors(2.0, r(off, GRISM, 2.0), GRISM) == \
            {"0": 1, "1": 1, "2": 1}
    # explicit values pass through unchanged
    assert r(4.0, GRISM, 2.0) == 4.0
    assert r({"1": 4.0}, GRISM, 2.0) == {"1": 4.0}


def test_resolve_hoist(bdi):
    assert bdi.resolve_hoist_star_psf(None) is True
    assert bdi.resolve_hoist_star_psf(False) is False
    assert bdi.resolve_hoist_star_psf(True) is True


def test_parse_rebin_dlam_none():
    assert parse_rebin_dlam("none") == "none"
    assert parse_rebin_dlam("None") == "none"
    assert parse_rebin_dlam("0:40,1:none,2:2") == {"0": 40.0, "1": None,
                                                  "2": 2.0}
    assert sed_rebin_factors(2.0, parse_rebin_dlam("0:40,1:none,2:2"),
                             GRISM) == {"0": 20, "1": 1, "2": 1}


def test_example_config(bdi):
    cfg = yaml.safe_load(bdi.EXAMPLE_CONFIG.replace(
        "# deposit: interp", "deposit: interp").replace(
        "# rebin_dlam:", "rebin_dlam:").replace(
        "# hoist_star_psf:", "hoist_star_psf:"))
    assert cfg["deposit"] == bdi.DEFAULT_DEPOSIT
    assert {str(k): float(v) for k, v in cfg["rebin_dlam"].items()} == \
        bdi.DEFAULT_REBIN_DLAM
    assert cfg["hoist_star_psf"] is bdi.DEFAULT_HOIST_STAR_PSF


def _cli_kwargs(bdi, monkeypatch, *flags):
    seen = {}
    monkeypatch.setattr(bdi, "build_dispersed_image",
                        lambda **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", [
        "build_dispersed_image.py", "--pointing-ra", "10", "--pointing-dec",
        "0", "--pointing-pa", "60", "--sca", "5", "--output", "x.fits",
        "--seed", "1", *flags])
    bdi.main()
    return seen


def test_cli_defaults_and_off_switches(bdi, monkeypatch):
    kw = _cli_kwargs(bdi, monkeypatch)
    assert kw["deposit"] is None and kw["rebin_dlam"] is None
    assert kw["hoist_star_psf"] is None and kw["order_mag_limits"] is None
    kw = _cli_kwargs(bdi, monkeypatch, "--deposit", "floor",
                     "--rebin-dlam", "none", "--no-hoist-star-psf")
    assert kw["deposit"] == "floor"
    assert bdi.resolve_rebin_dlam(kw["rebin_dlam"], GRISM, 2.0) is None
    assert kw["hoist_star_psf"] is False
    kw = _cli_kwargs(bdi, monkeypatch, "--hoist-star-psf",
                     "--rebin-dlam", "0:40,1:4,2:2")
    assert kw["hoist_star_psf"] is True
    assert kw["rebin_dlam"] == {"0": 40.0, "1": 4.0, "2": 2.0}


def test_benchmark_settings_match_driver_defaults(bdi):
    spec = importlib.util.spec_from_file_location(
        "bench_deposit_defaults",
        Path(__file__).parent.parent / "benchmarks" / "bench_deposit.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    d = bench.SETTINGS["default"]
    assert d["deposit"] == bdi.DEFAULT_DEPOSIT
    assert d["rebin_dlam"] == bdi.DEFAULT_REBIN_DLAM
    assert d["hoist_star_psf"] is bdi.DEFAULT_HOIST_STAR_PSF
    assert bench.SETTINGS["floor"] == {"deposit": "floor", "rebin_dlam": None,
                                       "hoist_star_psf": False}

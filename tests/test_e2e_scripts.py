"""Tests for the end-to-end pipeline scripts (wrapper command, catalog builder)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses (wrap_with_romanisim) look the module up
    spec.loader.exec_module(mod)
    return mod


HDR = dict(ra=10.0, dec=0.0, pa=-60.0, sca=5, ma_table=1036, rng_seed=7, optelem="grism")


def test_wrap_inject_command_uses_model_rate_and_no_psf():
    w = _load("wrap_with_romanisim")
    cmd = w.build_command(HDR, Path("in.fits"), Path("out.asdf"), "2026-01-01T00:00:00",
                          "GRISM", 2)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("romanisim_inject_rate.py")
    assert cmd[cmd.index("--rate-file") + 1] == "in.fits"
    assert cmd[cmd.index("--rate-hdu") + 1] == "MODEL"
    tail = cmd[cmd.index("--") + 1:]
    assert tail[0] == "out.asdf"
    assert "--extra-counts" not in cmd and "--stpsf" not in cmd
    assert tail[tail.index("--nobj") + 1] == "0"
    assert tail[tail.index("--sca") + 1] == "5"


def test_wrap_legacy_command_unchanged():
    w = _load("wrap_with_romanisim")
    cmd = w.build_command(HDR, Path("in.fits"), Path("out.asdf"), "D", "GRISM", 2,
                          counts_mode="extra-counts")
    assert cmd[:2] == ["romanisim-make-image", "out.asdf"]
    assert cmd[cmd.index("--extra-counts") + 1:cmd.index("--extra-counts") + 3] == ["in.fits", "ISIM"]
    assert "--stpsf" in cmd
    with pytest.raises(ValueError):
        w.build_command(HDR, Path("a"), Path("b"), "D", "GRISM", 2, counts_mode="nope")


def test_make_catalog_ab_normalization_and_roundtrip(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("pyarrow")
    mc = _load("make_catalog")
    try:
        wnum, den = mc.f158_weights(mc.wl_grid())
    except Exception as exc:  # no synphot data hydrated
        pytest.skip(f"F158 band unavailable: {exc}")
    wl = mc.wl_grid()
    flat_ab = mc.C_AA * mc.FNU_AB0 / wl**2
    # a flat-f_nu source is AB mag 0 in every band, exactly
    assert abs(mc.ab_mag_f158(flat_ab, wnum, den)) < 1e-6
    from astropy.table import Table
    Table(dict(id=["s", "g"], ra=[10.0, 10.01], dec=[0.0, 0.01], type=["PSF", "SER"],
               mag=[20.0, 21.5], n=[0.0, 1.0], half_light_radius=[0.0, 0.3],
               spectrum=["flat", "red"])).write(tmp_path / "src.ecsv")
    Table(dict(wavelength=wl, flat=flat_ab, red=flat_ab * (wl / 1e4) ** 2)).write(
        tmp_path / "spec.ecsv")
    sys.argv = ["make_catalog", "--sources", str(tmp_path / "src.ecsv"),
                "--spectra", str(tmp_path / "spec.ecsv"), "--out", str(tmp_path / "cat")]
    mc.main()
    import pyarrow.parquet as pq
    import zarr
    meta = pq.read_table(tmp_path / "cat" / "metadata.parquet").to_pandas()
    store = zarr.open(str(tmp_path / "cat" / "seds.zarr"), mode="r")
    np.testing.assert_allclose(-2.5 * np.log10(meta["F158"]), [20.0, 21.5], atol=1e-4)
    star = np.asarray(store["star_seds"])[meta.loc[0, "sed_index"]] * meta.loc[0, "flux_scale"]
    gal = np.asarray(store["galaxy_seds/sim_000"])[meta.loc[1, "sed_index"]] * meta.loc[1, "flux_scale"]
    assert abs(mc.ab_mag_f158(star, wnum, den) - 20.0) < 1e-3
    assert abs(mc.ab_mag_f158(gal, wnum, den) - 21.5) < 1e-3
    assert list(meta["type"]) == ["PSF", "SER"]

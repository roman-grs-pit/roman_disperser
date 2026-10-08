"""Tests for the HEALPix SED-store afterburner (scripts/healpix_repartition_seds.py)
and the healpix branch of the pipeline reader (load_galaxy_seds).

The afterburner is a pure copy: every galaxy's SED must come back bit-for-bit,
in metadata row order, whichever layout the reader is pointed at. The
end-to-end test pins that on a tiny synthetic sim-layout catalog. The unit
tests pin the HEALPix bookkeeping it rests on: UNIQ packing, NESTED
parent = child >> 2, and density-driven splitting.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("pyarrow")
pytest.importorskip("astropy_healpix")

import pyarrow as pa
import pyarrow.parquet as pq
import zarr

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hr():
    return _load("healpix_repartition_seds")


class TestUniq:
    def test_roundtrip(self, hr):
        for order in [0, 1, 5, 12, 20]:
            npix = 12 * 4 ** order
            ipix = np.array([0, 1, npix // 2, npix - 1])
            u = hr.uniq_from_nested(order, ipix)
            o, p = hr.nested_from_uniq(u)
            assert (o == order).all() and (p == ipix).all()

    def test_known_values(self, hr):
        # order 0: uniq 4..15 ; order 1: 16..63
        assert hr.uniq_from_nested(0, 0) == 4
        assert hr.uniq_from_nested(0, 11) == 15
        assert hr.uniq_from_nested(1, 0) == 16
        assert hr.nested_from_uniq(63) == (1, 47)


def test_nested_parent_is_bit_shift():
    """The afterburner derives coarse pixels as fine >> 2*(dk). Check vs astropy-healpix."""
    from astropy_healpix import HEALPix
    import astropy.units as u
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, 2000)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))
    fine = HEALPix(2 ** 12, order="nested").lonlat_to_healpix(ra * u.deg, dec * u.deg)
    for k in [0, 3, 7, 11]:
        coarse = HEALPix(2 ** k, order="nested").lonlat_to_healpix(ra * u.deg, dec * u.deg)
        np.testing.assert_array_equal(coarse, fine >> (2 * (12 - k)))


class TestChoosePixels:
    def test_dense_region_is_split_sparse_is_not(self, hr):
        max_order = 6
        # 100 rows in fine pixel 0 (dense), 5 rows spread in base pixel 11.
        base11 = 11 * 4 ** max_order
        ipix = np.r_[np.zeros(100, int), base11 + np.arange(5) * 1000]
        uniq, n_over = hr.choose_pixels(ipix, max_order, bytes_per_row=1.0,
                                        target_bytes=30.0)
        order, p = hr.nested_from_uniq(np.unique(uniq))
        # sparse rows stay in one order-0 pixel (base 11)
        assert hr.uniq_from_nested(0, 11) in set(uniq[100:])
        # dense rows sit at max_order and are flagged oversize (can't split further)
        assert (uniq[:100] == hr.uniq_from_nested(max_order, 0)).all()
        assert n_over == 1

    def test_every_pixel_under_target_when_splittable(self, hr):
        rng = np.random.default_rng(1)
        max_order = 10
        ipix = rng.integers(0, 12 * 4 ** max_order, 20000)
        uniq, n_over = hr.choose_pixels(ipix, max_order, 1.0, 500.0)
        _, counts = np.unique(uniq, return_counts=True)
        assert n_over == 0 and counts.max() <= 500
        # pixels are disjoint: each row's fine pixel lies inside its chosen pixel
        o, p = hr.nested_from_uniq(uniq)
        assert (ipix >> (2 * (max_order - o)) == p).all()

    def test_fixed_order(self, hr):
        ipix = np.array([0, 5, 4 ** 4, 12 * 4 ** 6 - 1])
        uniq, _ = hr.choose_pixels(ipix, 6, 1.0, 0.0, fixed_order=4)
        assert (hr.nested_from_uniq(uniq)[0] == 4).all()


def test_rows_within_group(hr):
    np.testing.assert_array_equal(hr.rows_within_group([7, 3, 7, 7, 3]),
                                  [0, 0, 1, 2, 1])


def test_assign_tasks(hr):
    np.testing.assert_array_equal(hr.assign_tasks([4, 4, 4, 10, 1], 8),
                                  [0, 0, 1, 2, 3])


# ---------------------------------------------------------------------------
# End to end: tiny sim-layout catalog -> plan/fill/verify -> reader on both
# ---------------------------------------------------------------------------

N_WL = 37


def _make_sim_catalog(root, rng):
    """Two sims, galaxies over a 2x2 deg field, a few stars, one padded copy."""
    from zarr.codecs import BloscCodec
    root.mkdir()
    store = zarr.open_group(str(root / "seds.zarr"), mode="w")
    store.create_array("wavelengths", data=np.linspace(7500, 21000, N_WL))
    store.create_array("star_seds", data=rng.random((3, N_WL), dtype=np.float32))
    store.create_group("galaxy_seds").attrs["n_partitions"] = 2
    rows = []
    for sim, n in [(1, 57), (2, 43)]:
        seds = rng.random((60, N_WL), dtype=np.float32) * 1e-17  # 60 >= n: padded rows
        store.create_array(f"galaxy_seds/sim_{sim:03d}", data=seds, chunks=(10, N_WL),
                           shards=(60, N_WL), compressors=BloscCodec(cname="zstd"))
        for i in range(n):
            rows.append(dict(type="SER", sim=sim, sed_index=i))
    for i in range(3):
        rows.append(dict(type="PSF", sim=0, sed_index=i))
    n = len(rows)
    tab = {
        "ra": rng.uniform(9, 11, n), "dec": rng.uniform(-1, 1, n),
        "type": [r["type"] for r in rows],
        "n": np.full(n, 1.0), "half_light_radius": np.full(n, 0.3),
        "pa": np.zeros(n), "ba": np.full(n, 0.7),
        "sed_index": np.array([r["sed_index"] for r in rows], np.int32),
        "flux_scale": rng.uniform(0.5, 2, n),
        "sim": np.array([r["sim"] for r in rows], np.int16),
    }
    t = pa.table(tab)
    # RA-padded copy, as pad_catalog.py makes: same SEDs referenced twice.
    t2 = pa.table({**tab, "ra": tab["ra"] + 2.0})
    pq.write_table(pa.concat_tables([t, t2]), root / "metadata.parquet")


def test_end_to_end_roundtrip(hr, tmp_path):
    rng = np.random.default_rng(42)
    src = tmp_path / "cat"
    out = tmp_path / "cat_hp"
    _make_sim_catalog(src, rng)

    # Small target so the field splits into several pixels and tasks.
    hr.cmd_plan(argparse.Namespace(
        input=str(src), output=str(out), target_gb=40 * 1e-9 * 200, nside=None,
        min_order=0, max_order=12, task_mem_gb=60 * N_WL * 4 / 1e9,
        dry_run=False, overwrite=False))
    plan = pq.read_table(out / hr.PLAN_FILE).to_pandas()
    assert len(plan) > 1 and plan["task"].max() > 0
    for t in range(plan["task"].max() + 1):
        hr.cmd_fill(argparse.Namespace(output=str(out), task=t))
    with pytest.raises(SystemExit) as e:
        hr.cmd_verify(argparse.Namespace(output=str(out), n_sample=10_000,
                                         seed=0, no_record=False))
    assert e.value.code == 0

    # metadata: same rows, same order, two appended columns
    m_src = pq.read_table(src / "metadata.parquet").to_pandas()
    m_out = pq.read_table(out / "metadata.parquet").to_pandas()
    assert list(m_out.columns) == list(m_src.columns) + ["hp_uniq", "hp_row"]
    assert m_out[m_src.columns].equals(m_src)
    assert ((m_out["type"] == "PSF") == (m_out["hp_uniq"] == -1)).all()

    # Reader returns identical spectra from both layouts
    bdi = _load("build_dispersed_image")
    s_old = zarr.open(str(src / "seds.zarr"), mode="r")
    s_new = zarr.open(str(out / "seds.zarr"), mode="r")
    assert bdi.galaxy_sed_layout(s_old) == "sim"
    assert bdi.galaxy_sed_layout(s_new) == "healpix"
    wl = np.asarray(s_old["wavelengths"][:])
    wl_mask = (wl >= 9000) & (wl <= 20000)
    gal = m_out[m_out["type"] == "SER"].sample(frac=0.6, random_state=1)  # shuffled subset
    a = bdi.load_galaxy_seds(s_old, gal, wl_mask)
    b = bdi.load_galaxy_seds(s_new, gal, wl_mask)
    np.testing.assert_array_equal(a, b)
    assert np.all(a > 0)

    # validate_catalog accepts the new store; and rejects it without hp columns
    class _E:  # minimal element: grism band
        lam_min, lam_max = 0.9, 2.0
    bdi.validate_catalog(m_out, s_new, wl, _E)
    with pytest.raises(ValueError, match="healpix"):
        bdi.validate_catalog(m_src, s_new, wl, _E)

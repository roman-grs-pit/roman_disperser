"""Tests for the water-ice throughput model (roman_disperser.ice).

Everything runs on synthetic data written to ``tmp_path`` with the same
schema as the (embargoed) toy-model files: a rate mosaic ``.npz`` with
per-SCA offsets, an ECSV ratio table with ``T_ratio_d<N>nm`` columns, and an
``ice_map.yaml``. No real ice data are needed or shipped.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from astropy.table import Table

from roman_disperser import ice
from roman_disperser.star_disperser import disperse_star_psf, make_star_disperser
from roman_disperser.galaxy_disperser import disperse_galaxy, make_galaxy_disperser
from roman_disperser.pipeline import (
    make_batched_star_fori, disperse_batched_stars,
)

PIX_PER_BIN = 32
N_BINS = ice.SCA_NPIX // PIX_PER_BIN          # 128
THICK_NM = np.array([0.0, 30.0, 60.0, 90.0])  # 4 columns, step 30
TABLE_WL_NM = np.linspace(480.0, 2300.0, 400)


def synthetic_ratio(wl_nm, d_nm):
    """Smooth, known T(lambda; d)/T(lambda; 0): 1 at d=0, +-10% ripples."""
    return 1.0 + 0.1 * (d_nm / THICK_NM[-1]) * np.sin(2 * np.pi * wl_nm / 400.0)


def write_synthetic_ice_dir(root, scas=(5, 10), rate_seed=0, epoch=None,
                            period=None, rate_scale=1.0):
    """Create ice_map.yaml + rate mosaic + one ratio table under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rate_seed)
    # Mosaic: each SCA block placed at a distinct offset; NaN elsewhere.
    n_sca = len(scas)
    mosaic = np.full((N_BINS + 10, n_sca * (N_BINS + 10)), np.nan, np.float32)
    row_off, col_off = {}, {}
    for j, sca in enumerate(scas):
        r0, c0 = 5, j * (N_BINS + 10) + 3
        row_off[sca], col_off[sca] = r0, c0
        block = rate_scale * rng.uniform(1.0, 3.0, (N_BINS, N_BINS)).astype(np.float32)
        mosaic[r0:r0 + N_BINS, c0:c0 + N_BINS] = block
    np.savez(root / "rate.npz", rate_mosaic=mosaic,
             det_row_offsets=np.array(row_off, dtype=object),
             det_col_offsets=np.array(col_off, dtype=object),
             pix_per_bin=np.int64(PIX_PER_BIN))
    tab = Table()
    tab["wavelength"] = TABLE_WL_NM
    for d in THICK_NM:
        tab[f"T_ratio_d{int(d)}nm"] = synthetic_ratio(TABLE_WL_NM, d)
    tab.write(root / "table.ecsv", format="ascii.ecsv", overwrite=True)
    ice_map = {"rate_mosaic": "rate.npz",
               "tables": {f"SCA{s}": "table.ecsv" for s in scas}}
    if epoch is not None:
        ice_map["epoch_mjd"] = epoch
    if period is not None:
        ice_map["decon_period_days"] = period
    with open(root / ice.ICE_MAP_FILE, "w") as f:
        yaml.safe_dump(ice_map, f)
    return root


@pytest.fixture
def ice_dir(tmp_path):
    return write_synthetic_ice_dir(tmp_path / "ice")


@pytest.fixture
def wavelengths_um():
    return np.linspace(0.95, 1.95, 300).astype(np.float32)


@pytest.fixture
def ice_payload(ice_dir, wavelengths_um):
    return ice.load_ice_payload(ice_dir, 5, wavelengths_um)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_read_map_defaults(self, ice_dir):
        m = ice.read_ice_map(ice_dir)
        assert m["epoch_mjd"] == ice.DEFAULT_EPOCH_MJD
        assert m["decon_period_days"] == ice.DEFAULT_DECON_PERIOD_DAYS
        assert m["tables"]["SCA5"] == "table.ecsv"

    def test_read_map_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="embargoed"):
            ice.read_ice_map(tmp_path)

    def test_rate_map_block_extraction(self, ice_dir):
        rate, ppb = ice.load_rate_map(ice_dir / "rate.npz", 10)
        assert ppb == PIX_PER_BIN
        assert rate.shape == (N_BINS, N_BINS)
        assert rate.dtype == np.float32
        assert np.all(np.isfinite(rate))
        with np.load(ice_dir / "rate.npz", allow_pickle=True) as z:
            r0 = z["det_row_offsets"].item()[10]
            c0 = z["det_col_offsets"].item()[10]
            expected = z["rate_mosaic"][r0:r0 + N_BINS, c0:c0 + N_BINS]
        np.testing.assert_array_equal(rate, expected)

    def test_rate_map_unknown_sca(self, ice_dir):
        with pytest.raises(ValueError, match="SCA 7"):
            ice.load_rate_map(ice_dir / "rate.npz", 7)

    def test_table_interpolation(self, ice_dir, wavelengths_um):
        trans, thick = ice.load_transmission_table(
            ice_dir / "table.ecsv", wavelengths_um)
        np.testing.assert_array_equal(thick, THICK_NM)
        assert trans.shape == (len(THICK_NM), len(wavelengths_um))
        assert trans.dtype == np.float32
        # Column 0 is identically 1; others match the generating function
        # up to the table's own linear-in-wavelength sampling.
        np.testing.assert_allclose(trans[0], 1.0, atol=1e-7)
        wl_nm = wavelengths_um.astype(np.float64) * 1e3
        for i, d in enumerate(THICK_NM):
            ref = np.interp(wl_nm, TABLE_WL_NM, synthetic_ratio(TABLE_WL_NM, d))
            np.testing.assert_allclose(trans[i], ref, rtol=1e-6)

    def test_table_rejects_nonuniform(self, tmp_path, wavelengths_um):
        tab = Table()
        tab["wavelength"] = TABLE_WL_NM
        for d in (0, 30, 90):
            tab[f"T_ratio_d{d}nm"] = np.ones_like(TABLE_WL_NM)
        tab.write(tmp_path / "bad.ecsv", format="ascii.ecsv")
        with pytest.raises(ValueError, match="uniformly"):
            ice.load_transmission_table(tmp_path / "bad.ecsv", wavelengths_um)

    def test_payload_contents(self, ice_payload, wavelengths_um):
        p = ice_payload
        assert p["rate_nm_per_day"].shape == (N_BINS, N_BINS)
        assert p["trans"].shape == (len(THICK_NM), len(wavelengths_um))
        assert p["thickness_step_nm"] == 30.0
        assert p["thickness_max_nm"] == 90.0
        assert p["n_wl"] == len(wavelengths_um)
        assert p["sca"] == 5 and p["table_file"] == "table.ecsv"

    def test_payload_warns_when_table_too_short(self, tmp_path, wavelengths_um):
        # rates 1-3 nm/day x 20 d = up to 60 nm < 90: no warning ...
        d_ok = write_synthetic_ice_dir(tmp_path / "ok")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ice.load_ice_payload(d_ok, 5, wavelengths_um)
        # ... but x 40 d = up to 120 nm > 90: warn.
        d_warn = write_synthetic_ice_dir(tmp_path / "warn", period=40.0)
        with pytest.warns(UserWarning, match="clamped"):
            ice.load_ice_payload(d_warn, 5, wavelengths_um)


# ---------------------------------------------------------------------------
# Time -> thickness
# ---------------------------------------------------------------------------

class TestTime:
    def test_time_since_decon_wraps(self):
        t0 = ice.DEFAULT_EPOCH_MJD
        assert ice.time_since_decon(t0) == 0.0
        assert ice.time_since_decon(t0 + 7.5) == 7.5
        assert ice.time_since_decon(t0 + 20.0) == 0.0
        assert ice.time_since_decon(t0 + 47.25) == pytest.approx(7.25)
        assert ice.time_since_decon(t0 + 5, period_days=3.0) == 2.0

    def test_time_before_epoch_raises(self):
        with pytest.raises(ValueError, match="before"):
            ice.time_since_decon(ice.DEFAULT_EPOCH_MJD - 1)

    def test_thickness_map(self, ice_payload):
        mjd = ice.DEFAULT_EPOCH_MJD + 25.0   # 5 d into the second cycle
        d = ice.thickness_map(ice_payload, mjd)
        assert d.shape == (N_BINS, N_BINS) and d.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(d), np.asarray(ice_payload["rate_nm_per_day"]) * 5.0,
            rtol=1e-6)


# ---------------------------------------------------------------------------
# The factor
# ---------------------------------------------------------------------------

def toy_model_reference(rate, pix_per_bin, thick_nm, table_wl_nm, table_cols,
                        xsca, ysca, dt_days, wl_nm):
    """numpy transcription of ice_rep_sim.ice_relative_transmission.

    Per position: nearest 32-px bin (0-based pixel // pix_per_bin) ->
    thickness = rate * dt -> np.digitize into the thickness columns ->
    linear weights between the two bracketing columns; then evaluate the
    resulting passband at the requested wavelength (np.interp, as our
    pre-interpolated table does).
    """
    out = np.empty(len(xsca))
    for k in range(len(xsca)):
        iy = int((ysca[k] - 1) // pix_per_bin)
        ix = int((xsca[k] - 1) // pix_per_bin)
        d = rate[iy, ix] * dt_days
        r = np.digitize(d, thick_nm)                 # right edge index
        lo_d, hi_d = thick_nm[r - 1], thick_nm[r]
        w_hi = (d - lo_d) / (hi_d - lo_d)
        passband = (1 - w_hi) * table_cols[r - 1] + w_hi * table_cols[r]
        out[k] = np.interp(wl_nm[k], table_wl_nm, passband)
    return out


class TestFactor:
    def test_matches_toy_model(self, ice_dir, ice_payload, wavelengths_um):
        rng = np.random.default_rng(1)
        n = len(wavelengths_um)
        # In-detector positions, strictly inside so the toy model's
        # un-clipped indexing is valid; thickness stays inside the table.
        xsca = rng.uniform(1.0, 4088.0, n)
        ysca = rng.uniform(1.0, 4088.0, n)
        dt = 12.3                                      # 1-3 nm/day -> <= 37 nm
        rate = np.asarray(ice_payload["rate_nm_per_day"])
        cols = [synthetic_ratio(TABLE_WL_NM, d) for d in THICK_NM]
        ref = toy_model_reference(rate, PIX_PER_BIN, THICK_NM, TABLE_WL_NM,
                                  cols, xsca, ysca, dt,
                                  wavelengths_um.astype(np.float64) * 1e3)
        thick = (ice_payload["rate_nm_per_day"] * jnp.float32(dt)).astype(jnp.float32)
        # The payload is a closure constant (it holds strings), as in the
        # dispersers; only the thickness map and positions are traced.
        factor_jit = jax.jit(
            lambda th, x, y: ice.transmission_factor(ice_payload, th, x, y))
        got = factor_jit(thick, jnp.asarray(xsca, jnp.float32),
                         jnp.asarray(ysca, jnp.float32))
        np.testing.assert_allclose(np.asarray(got), ref, rtol=2e-5)

    def test_zero_thickness_is_unity(self, ice_payload, wavelengths_um):
        n = len(wavelengths_um)
        thick = jnp.zeros((N_BINS, N_BINS), jnp.float32)
        got = ice.transmission_factor(
            ice_payload, thick, jnp.full(n, 1000.0), jnp.full(n, 2000.0))
        np.testing.assert_allclose(np.asarray(got), 1.0, atol=1e-7)

    def test_clamps_to_last_column(self, ice_payload, wavelengths_um):
        n = len(wavelengths_um)
        thick = jnp.full((N_BINS, N_BINS), 500.0, jnp.float32)   # > 90 nm
        got = ice.transmission_factor(
            ice_payload, thick, jnp.full(n, 1000.0), jnp.full(n, 2000.0))
        np.testing.assert_allclose(np.asarray(got),
                                   np.asarray(ice_payload["trans"][-1]),
                                   rtol=1e-6)

    def test_off_detector_positions_use_edge_bin(self, ice_payload, wavelengths_um):
        n = len(wavelengths_um)
        thick = ice_payload["rate_nm_per_day"] * 10.0
        far = ice.transmission_factor(
            ice_payload, thick, jnp.full(n, -500.0), jnp.full(n, 9000.0))
        edge = ice.transmission_factor(
            ice_payload, thick, jnp.full(n, 1.0), jnp.full(n, 4096.0))
        np.testing.assert_array_equal(np.asarray(far), np.asarray(edge))

    def test_wavelength_count_mismatch_raises(self, ice_payload):
        thick = jnp.zeros((N_BINS, N_BINS), jnp.float32)
        with pytest.raises(ValueError, match="wavelength"):
            ice.transmission_factor(ice_payload, thick, jnp.ones(7), jnp.ones(7))


# ---------------------------------------------------------------------------
# Through the dispersers
# ---------------------------------------------------------------------------

def _delta_psf_payload(n_wl=6, psf_size=20, oversample=4):
    n_y = n_x = 3
    grid = jnp.zeros((n_y, n_x, n_wl, psf_size, psf_size), jnp.float32)
    c = psf_size // 2
    grid = grid.at[:, :, :, c, c].set(1.0)
    wl = jnp.linspace(1.0, 1.8, n_wl)
    return {"psf_grid": grid, "wavelengths": wl, "wl_grid": wl,
            "spatial_x": jnp.linspace(1, 4088, n_x),
            "spatial_y": jnp.linspace(1, 4088, n_y),
            "oversample": oversample, "detector": "WFI05", "order": "1"}


def _constant_payload(ice_payload, value):
    """Same payload with the ratio table replaced by a constant."""
    p = dict(ice_payload)
    p["trans"] = jnp.full_like(ice_payload["trans"], value)
    return p


class TestDispersers:
    @pytest.fixture
    def run_wl(self):
        return jnp.linspace(1.1, 1.7, 40)

    @pytest.fixture
    def run_payload(self, ice_dir, run_wl):
        return ice.load_ice_payload(ice_dir, 5, np.asarray(run_wl))

    def test_star_constant_factor_scales_exactly(self, payload, run_wl, run_payload):
        """A table of 0.5 everywhere must halve the deposit bit-for-bit
        (scaling by a power of two is exact in float32)."""
        psf = _delta_psf_payload()
        flux = jnp.linspace(1.0, 2.0, len(run_wl))
        out0 = jnp.zeros((4088, 4088), jnp.float32)
        ref = disperse_star_psf(psf, payload, 2000.0, 2000.0, run_wl, flux, out0)
        half = _constant_payload(run_payload, 0.5)
        thick = ice.thickness_map(half, ice.DEFAULT_EPOCH_MJD + 3.0)
        got = disperse_star_psf(psf, payload, 2000.0, 2000.0, run_wl, flux, out0,
                                ice_payload=half, ice_thickness_nm=thick)
        np.testing.assert_array_equal(np.asarray(got), 0.5 * np.asarray(ref))

    def test_star_requires_thickness_with_payload(self, payload, run_wl, run_payload):
        psf = _delta_psf_payload()
        out0 = jnp.zeros((4088, 4088), jnp.float32)
        with pytest.raises(ValueError, match="ice_thickness_nm"):
            disperse_star_psf(psf, payload, 2000.0, 2000.0, run_wl,
                              jnp.ones(len(run_wl)), out0, ice_payload=run_payload)

    def test_star_factory_signature_and_dynamic_thickness(self, payload, run_wl,
                                                          run_payload):
        """The ice-enabled compiled function takes the thickness map as a
        traced argument: two exposure times reuse one compilation."""
        psf = _delta_psf_payload()
        fn = make_star_disperser(psf, payload, ice_payload=run_payload)
        flux = jnp.ones(len(run_wl))
        out0 = jnp.zeros((4088, 4088), jnp.float32)
        d1 = ice.thickness_map(run_payload, ice.DEFAULT_EPOCH_MJD + 1.0)
        d2 = ice.thickness_map(run_payload, ice.DEFAULT_EPOCH_MJD + 19.0)
        r1 = fn(2000.0, 2000.0, run_wl, flux, out0, d1)
        r2 = fn(2000.0, 2000.0, run_wl, flux, out0, d2)
        assert fn._cache_size() == 1
        # Thicker ice -> different image (the synthetic ratio is not flat).
        assert not np.array_equal(np.asarray(r1), np.asarray(r2))
        # And the total deposited flux equals sum(flux * factor at the trace).
        xd, yd = _trace(payload, 2000.0, 2000.0, run_wl)
        for d, r in ((d1, r1), (d2, r2)):
            f = ice.transmission_factor(run_payload, d, xd, yd)
            np.testing.assert_allclose(float(r.sum()), float((flux * f).sum()),
                                       rtol=1e-5)

    def test_galaxy_constant_factor_scales_exactly(self, payload, run_wl, run_payload):
        psf = _delta_psf_payload()
        img = jnp.zeros((40, 40), jnp.float32).at[18:22, 18:22].set(0.0625)
        spec = jnp.linspace(1.0, 2.0, len(run_wl))
        out0 = jnp.zeros((4088, 4088), jnp.float32)
        ref = disperse_galaxy(payload, psf, img, 2000.0, 2000.0, spec, run_wl, out0)
        half = _constant_payload(run_payload, 0.5)
        thick = ice.thickness_map(half, ice.DEFAULT_EPOCH_MJD + 3.0)
        got = disperse_galaxy(payload, psf, img, 2000.0, 2000.0, spec, run_wl, out0,
                              ice_payload=half, ice_thickness_nm=thick)
        np.testing.assert_array_equal(np.asarray(got), 0.5 * np.asarray(ref))
        # Factory path: compare jitted-with-ice against jitted-without
        # (jit vs eager differ at the 1e-8 level in the FFT convolution's
        # fusion, unrelated to ice).
        fn_off = make_galaxy_disperser(psf, payload)
        fn_ice = make_galaxy_disperser(psf, payload, ice_payload=half)
        ref2 = fn_off(img, 2000.0, 2000.0, spec, run_wl, out0)
        got2 = fn_ice(img, 2000.0, 2000.0, spec, run_wl, out0, thick)
        np.testing.assert_array_equal(np.asarray(got2), 0.5 * np.asarray(ref2))

    def test_batched_star_fori_threads_thickness(self, payload, run_wl, run_payload):
        psf = _delta_psf_payload()
        half = _constant_payload(run_payload, 0.5)
        sens = jnp.ones(len(run_wl))
        fn_off = make_batched_star_fori(
            make_star_disperser(psf, payload), sens, run_wl, 1.0)
        fn_ice = make_batched_star_fori(
            make_star_disperser(psf, payload, ice_payload=half), sens, run_wl,
            1.0, ice=True)
        spectra = np.ones((3, len(run_wl)), np.float32)
        x = np.array([1500.0, 2000.0, 2500.0], np.float32)
        y = np.array([2000.0, 2100.0, 2200.0], np.float32)
        out0 = jnp.zeros((4088, 4088), jnp.float32)
        thick = ice.thickness_map(half, ice.DEFAULT_EPOCH_MJD + 3.0)
        ref = disperse_batched_stars(fn_off, spectra, x, y, out0, 2)
        got = disperse_batched_stars(fn_ice, spectra, x, y, out0, 2,
                                     ice_thickness_nm=thick)
        np.testing.assert_array_equal(np.asarray(got), 0.5 * np.asarray(ref))


def _trace(payload, x0, y0, wl):
    from roman_disperser.star_disperser import _compute_dispersed_positions
    return _compute_dispersed_positions(payload, x0, y0, wl)

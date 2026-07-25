"""Tests for the dispersing-element mode abstraction.

Two kinds of check live here:

* pure statements about the mode records (defaults, resolution order, aliasing),
  which need no reference data; and
* consistency between each mode and its *delivered optical model* — the band
  and the order list. Those are statements about the reference data, so a test
  is where they belong: they catch a delivered model drifting away from what we
  declare, without waiting for a run to produce quietly wrong traces.
"""

import os

import numpy as np
import pytest

from roman_disperser import modes, paths, psf_model


class TestModeResolution:
    def test_default_is_grism(self, monkeypatch):
        """No argument and no environment => grism, so old calls are unchanged."""
        monkeypatch.delenv(modes.MODE_ENV_VAR, raising=False)
        assert modes.get_mode() is modes.GRISM

    def test_env_var_selects(self, monkeypatch):
        monkeypatch.setenv(modes.MODE_ENV_VAR, "prism")
        assert modes.get_mode() is modes.PRISM

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv(modes.MODE_ENV_VAR, "prism")
        assert modes.get_mode("grism") is modes.GRISM

    def test_passthrough(self):
        """An already-resolved mode is returned unchanged, so callers can thread
        either a name or a record through without caring which they hold."""
        assert modes.get_mode(modes.PRISM) is modes.PRISM

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown disperser mode"):
            modes.get_mode("g150")

    def test_picklable(self):
        """The multiprocessing PSF-cache generator ships modes across a process
        boundary, so a mode must pickle."""
        import pickle
        for mode in modes.MODES.values():
            assert pickle.loads(pickle.dumps(mode)) == mode


class TestModeRecords:
    def test_names_match_registry(self):
        for name, mode in modes.MODES.items():
            assert mode.name == name

    def test_band_is_ordered(self, mode):
        assert mode.lam_min < mode.lam_max

    def test_every_order_has_a_psf(self, mode):
        """Each rendered order must resolve to some PSF payload, directly or by
        alias — otherwise the pipeline KeyErrors partway through a pointing."""
        for order in mode.orders:
            assert mode.psf_order(order) in mode.stpsf_filter

    def test_grism_order_2_aliases_order_1(self):
        """STPSF ships only GRISM0/GRISM1, so order 2 borrows the order-1 PSF."""
        assert modes.GRISM.psf_order("2") == "1"

    def test_prism_has_a_single_beam(self):
        assert modes.PRISM.orders == ("1",)
        assert modes.PRISM.psf_order_alias == {}

    def test_filter_for_unknown_order_raises(self):
        with pytest.raises(ValueError, match="no STPSF filter"):
            modes.PRISM.filter_for("0")

    def test_psf_cache_names_do_not_collide(self):
        """Grism and prism caches share one psf_cache/ directory, so their
        filenames must differ for the same detector and order."""
        grid = {"x": np.linspace(1, 4088, 4), "y": np.linspace(1, 4088, 4)}
        names = {
            m.name: psf_model.get_cache_filename(
                "WFI05", "1", psf_model.default_wavelengths(m), grid, 5.0, 4, mode=m
            )
            for m in modes.MODES.values()
        }
        assert names["grism"] != names["prism"]
        assert "GRISM1" in names["grism"] and "PRISM" in names["prism"]


class TestDefaultWavelengths:
    def test_spans_the_band(self, mode):
        wl = psf_model.default_wavelengths(mode)
        assert wl[0] == pytest.approx(mode.lam_min)
        assert wl[-1] == pytest.approx(mode.lam_max)

    def test_spacing(self, mode):
        wl = psf_model.default_wavelengths(mode)
        assert np.allclose(np.diff(wl), psf_model.PSF_WL_STEP_UM)

    def test_matches_the_historical_grids(self, mode):
        """Both elements' PSF caches on disk were generated with these exact
        grids; changing them would silently invalidate every cached payload."""
        expected = {"grism": np.arange(0.9, 2.01, 0.02),
                    "prism": np.arange(0.75, 1.86, 0.02)}[mode.name]
        assert np.allclose(psf_model.default_wavelengths(mode), expected)


class TestPathResolution:
    def test_mode_is_keyword_optional(self):
        """Existing no-argument callers (the tutorials repo, line_test scripts)
        must keep resolving to the grism."""
        assert paths.optical_model_path() == paths.optical_model_path(mode="grism")
        assert paths.sensitivity_dir() == paths.sensitivity_dir(mode="grism")

    def test_modes_resolve_to_different_files(self):
        assert (paths.optical_model_path(mode="grism")
                != paths.optical_model_path(mode="prism"))
        assert (paths.sensitivity_dir(mode="grism")
                != paths.sensitivity_dir(mode="prism"))

    def test_explicit_overrides_mode(self, tmp_path):
        assert paths.optical_model_path(tmp_path / "x.yaml", mode="prism") \
            == tmp_path / "x.yaml"


class TestBandConsistency:
    """The mode record vs. the delivered optical model."""

    def test_band_matches_model(self, mode_model):
        """Declared band equals the model's validity range, for both elements
        today. If a future delivery changes wl_min/wl_max, this is the tripwire."""
        mode, model = mode_model
        assert mode.lam_min == pytest.approx(model.wl_min)
        assert mode.lam_max == pytest.approx(model.wl_max)

    def test_orders_are_defined_by_the_model(self, mode_model):
        mode, model = mode_model
        assert set(mode.orders) <= set(model.orders_defined)

    def test_optical_element_matches(self, mode_model):
        mode, model = mode_model
        assert modes.mode_for_optical_model(model) is mode

    def test_check_band_passes(self, mode_model):
        mode, model = mode_model
        modes.check_band(mode, model)  # must not raise

    def test_mismatched_element_raises(self, optical_model):
        """Prism mode + grism model: the polynomials would evaluate happily and
        give wrong traces, so this has to be caught at load time."""
        with pytest.raises(ValueError, match="mode/model mismatch"):
            modes.check_band(modes.PRISM, optical_model)

    def test_extrapolation_raises(self, mode_model):
        """A band wider than the model's fit range is an error; narrowing is not."""
        import dataclasses
        mode, model = mode_model
        too_blue = dataclasses.replace(mode, lam_min=mode.lam_min - 0.1)
        with pytest.raises(ValueError, match="extends outside"):
            modes.check_band(too_blue, model)
        too_red = dataclasses.replace(mode, lam_max=mode.lam_max + 0.1)
        with pytest.raises(ValueError, match="extends outside"):
            modes.check_band(too_red, model)

    def test_narrowing_is_allowed(self, mode_model):
        import dataclasses
        mode, model = mode_model
        narrow = dataclasses.replace(
            mode, lam_min=mode.lam_min + 0.05, lam_max=mode.lam_max - 0.05
        )
        modes.check_band(narrow, model)  # must not raise

    def test_padding_permits_slight_overrun(self, mode_model):
        import dataclasses
        mode, model = mode_model
        wide = dataclasses.replace(mode, lam_max=mode.lam_max + 0.01)
        with pytest.raises(ValueError):
            modes.check_band(wide, model)
        modes.check_band(wide, model, padding=0.02)  # must not raise

    def test_undefined_order_raises(self, mode_model):
        import dataclasses
        mode, model = mode_model
        bogus = dataclasses.replace(mode, orders=mode.orders + ("m7",))
        with pytest.raises(ValueError, match="does not define"):
            modes.check_band(bogus, model)

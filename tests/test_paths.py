"""Tests for the optical-model resolution ladder (paths.optical_model_path).

The design rule under test: which delivery file loads is always *declared* —
an explicit path, an explicit version, or the delivery hydrate recorded in
``data-versions.lock`` — never inferred from what happens to sit in the data
directory. A stray YAML must not silently become the calibration; a missing
declaration must fail loudly with actionable hints.
"""

import json
from pathlib import Path

import pytest

from roman_disperser import paths
from roman_disperser.elements import GRISM, PRISM


def _make_data_dir(tmp_path, monkeypatch, files=(), lock=None):
    """Point the resolver at a synthetic data dir."""
    d = tmp_path / "data"
    d.mkdir()
    for name in files:
        (d / name).write_text("stub\n")
    if lock is not None:
        (d / paths.LOCK_NAME).write_text(json.dumps(lock))
    monkeypatch.setenv("ROMAN_DISPERSER_DATA", str(d))
    return d


class TestOpticalModelFilename:
    def test_template_matches_vendored_names(self):
        assert (paths.optical_model_filename(GRISM, "v0.8")
                == "Roman_grism_OpticalModel_v0.8.yaml")
        assert (paths.optical_model_filename(PRISM, "v0.8")
                == "Roman_prism_OpticalModel_v0.8.yaml")

    def test_version_leniency(self):
        # "0.8" and "v0.8" are the same declaration
        assert (paths.optical_model_filename(GRISM, "0.8")
                == paths.optical_model_filename(GRISM, "v0.8"))


class TestOpticalModelPath:
    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        _make_data_dir(tmp_path, monkeypatch)
        p = paths.optical_model_path("/somewhere/else/model.yaml")
        assert p == Path("/somewhere/else/model.yaml")

    def test_explicit_version(self, tmp_path, monkeypatch):
        d = _make_data_dir(
            tmp_path, monkeypatch,
            files=["Roman_grism_OpticalModel_v0.9.yaml"],
            lock={"optical_model": "optical-model-v0.8"},
        )
        # version beats the lock
        p = paths.optical_model_path(version="v0.9")
        assert p == d / "Roman_grism_OpticalModel_v0.9.yaml"

    def test_explicit_version_missing_file_raises(self, tmp_path, monkeypatch):
        _make_data_dir(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError, match="v0.9"):
            paths.optical_model_path(version="v0.9")

    def test_lock_resolves_per_element(self, tmp_path, monkeypatch):
        d = _make_data_dir(
            tmp_path, monkeypatch,
            files=["Roman_grism_OpticalModel_v0.8.yaml",
                   "Roman_prism_OpticalModel_v0.8.yaml"],
            lock={"optical_model": "optical-model-v0.8",
                  "optical_model_prism": "optical-model-prism-v0.8"},
        )
        assert (paths.optical_model_path()
                == d / "Roman_grism_OpticalModel_v0.8.yaml")
        assert (paths.optical_model_path(element=PRISM)
                == d / "Roman_prism_OpticalModel_v0.8.yaml")

    def test_lock_names_missing_file_raises(self, tmp_path, monkeypatch):
        _make_data_dir(tmp_path, monkeypatch,
                       lock={"optical_model": "optical-model-v0.8"})
        with pytest.raises(FileNotFoundError, match="hydrate"):
            paths.optical_model_path()

    def test_stray_file_is_not_adopted(self, tmp_path, monkeypatch):
        # THE design case: a matching file exists, but nothing declared it.
        # It must be listed as a hint, never used.
        _make_data_dir(tmp_path, monkeypatch,
                       files=["Roman_grism_OpticalModel_v0.9.yaml"])
        with pytest.raises(FileNotFoundError) as exc:
            paths.optical_model_path()
        msg = str(exc.value)
        assert "Roman_grism_OpticalModel_v0.9.yaml" in msg   # hinted...
        assert "not used" in msg                              # ...not adopted
        assert "hydrate" in msg

    def test_no_lock_no_files_fails_with_guidance(self, tmp_path, monkeypatch):
        _make_data_dir(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError, match="optical_model_version"):
            paths.optical_model_path()

    def test_unparseable_lock_tag_fails_loudly(self, tmp_path, monkeypatch):
        _make_data_dir(tmp_path, monkeypatch,
                       files=["Roman_grism_OpticalModel_v0.8.yaml"],
                       lock={"optical_model": "some-strange-tag"})
        with pytest.raises(FileNotFoundError):
            paths.optical_model_path()

    def test_vendored_data_dir_resolves(self):
        # Against the real hydrated data dir (whatever the environment
        # provides): both elements resolve to existing files via the lock.
        for element in (GRISM, PRISM):
            p = paths.optical_model_path(element=element)
            assert p.exists(), p


class TestSensitivityDir:
    def test_default_is_grism(self, tmp_path, monkeypatch):
        d = _make_data_dir(tmp_path, monkeypatch)
        assert paths.sensitivity_dir() == d / "sensitivities"

    def test_per_element(self, tmp_path, monkeypatch):
        d = _make_data_dir(tmp_path, monkeypatch)
        assert paths.sensitivity_dir(element=GRISM) == d / "sensitivities"
        assert (paths.sensitivity_dir(element=PRISM)
                == d / "sensitivities_prism")
        assert (paths.sensitivity_dir(element="prism")
                == d / "sensitivities_prism")

    def test_explicit_directory_wins(self, tmp_path, monkeypatch):
        _make_data_dir(tmp_path, monkeypatch)
        explicit = tmp_path / "elsewhere"
        assert (paths.sensitivity_dir(explicit, element=PRISM)
                == explicit)

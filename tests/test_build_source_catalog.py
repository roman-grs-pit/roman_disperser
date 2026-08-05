"""Tests for the catalog builder's wavelength grid (scripts/build_source_catalog.py).

The grid is the contract between the catalog and the pipeline: the catalog is
built as a superset of every element's band (floor at the prism's 7500 A, not
the grism's 9000 A), and the slice into the Galacticus SED grid is *derived*
from the requested range rather than hardcoded. These tests pin that
derivation — a wrong slice reads the wrong SED window silently, which is
exactly the failure the derivation exists to prevent.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def bsc():
    """Import scripts/build_source_catalog.py as a module (not a package)."""
    path = Path(__file__).parent.parent / "scripts" / "build_source_catalog.py"
    spec = importlib.util.spec_from_file_location("build_source_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestWlGrid:
    def test_default_is_prism_grism_superset(self, bsc):
        wavelengths, n_wl, sl = bsc.wl_grid()
        assert wavelengths[0] == 7500.0
        assert wavelengths[-1] == 21000.0
        assert n_wl == 6751
        assert np.allclose(np.diff(wavelengths), bsc.WL_STEP)
        assert np.allclose(bsc.GALACTICUS_WL[sl], wavelengths)

    def test_default_covers_both_element_bands(self, bsc):
        from roman_disperser.elements import ELEMENTS

        wavelengths, _, _ = bsc.wl_grid()
        for el in ELEMENTS.values():
            assert wavelengths[0] <= el.lam_min * 1e4, el.name
            assert wavelengths[-1] >= el.lam_max * 1e4, el.name

    def test_wl_min_9000_reproduces_catalog_v2_grid(self, bsc):
        # The published catalog-v2 was built with the grism-era constants:
        # np.linspace(9000, 21000, 6001) read through slice(3500, 9501).
        wavelengths, n_wl, sl = bsc.wl_grid(wl_min=9000.0)
        assert wavelengths[0] == 9000.0
        assert n_wl == 6001
        assert sl == slice(3500, 9501)

    def test_off_grid_floor_fails_loudly(self, bsc):
        # A floor that is not a sample of the Galacticus grid cannot be
        # served by any slice; the derivation must refuse, not round.
        with pytest.raises(AssertionError):
            bsc.wl_grid(wl_min=7501.0)

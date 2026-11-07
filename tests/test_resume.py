"""Resuming over existing products: incomplete FITS are re-rendered (T66).

``process_pointing`` and quick mode skip an existing product only if it is
complete (``pipeline.fits_product_problem``); ``write_fits`` writes through a
``.part`` file so an interrupted write never leaves a truncated product.
"""

import os
import warnings

import jax
import numpy as np
import pytest
from astropy.io import fits

from roman_disperser.pipeline import fits_product_problem, write_fits
from tests.test_order_mag_limits import _fake_pipeline, _patch, bdi  # noqa: F401


def _write(path, shape=(8, 8)):
    img = np.ones(shape, np.float32)
    write_fits(img, img, str(path), pointing_ra=1.0, pointing_dec=2.0,
               pointing_pa=0.0, sca=5, exptime=100.0,
               rng_key_data=np.array([1, 2], np.uint32), seed=1)


class TestProductCheck:
    def test_complete(self, tmp_path):
        p = tmp_path / "a.fits"
        _write(p)
        assert fits_product_problem(p, (8, 8)) is None
        assert not list(tmp_path.glob("*.part"))

    def test_wrong_shape(self, tmp_path):
        p = tmp_path / "a.fits"
        _write(p)
        assert "shape" in fits_product_problem(p, (16, 16))

    @pytest.mark.parametrize("cut", [2880, 4000, 0.5, 0.9, -1])
    def test_truncated(self, tmp_path, cut):
        """Cut in a header, in a data block, or by the last byte."""
        p = tmp_path / "a.fits"
        _write(p, (64, 64))
        size = os.path.getsize(p)
        keep = (size + cut if cut < 0 else int(size * cut) if cut < 1
                else cut)
        os.truncate(p, keep)
        assert fits_product_problem(p, (64, 64)) is not None

    def test_not_fits(self, tmp_path):
        p = tmp_path / "a.fits"
        p.write_bytes(b"")
        assert fits_product_problem(p, (8, 8)) is not None

    def test_missing_isim(self, tmp_path):
        p = tmp_path / "a.fits"
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(np.ones((8, 8), np.float32), name="MODEL")]
                     ).writeto(p)
        assert "ISIM" in fits_product_problem(p, (8, 8))


def _render(bdi, out_dir, force=False):
    return bdi.process_pointing(_fake_pipeline(None), 10.0, 0.1, 0.0,
                                str(out_dir), cone_radius=1.0, verbose=False,
                                pointing_key=jax.random.key(0), force=force)


def test_resume_rerenders_truncated_fits(bdi, tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, bdi, calls)
    out = tmp_path / "pt"
    _render(bdi, out)
    fits_path, = out.glob("*_detSCA05.fits")
    good = fits_path.read_bytes()
    n_first = len(calls)

    # complete product: skipped, nothing rendered, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _render(bdi, out)
    assert len(calls) == n_first

    # truncated product: re-rendered with a warning
    os.truncate(fits_path, len(good) // 2)
    with pytest.warns(RuntimeWarning, match="incomplete"):
        _render(bdi, out)
    assert len(calls) == 2 * n_first
    assert fits_product_problem(fits_path, (16, 16)) is None

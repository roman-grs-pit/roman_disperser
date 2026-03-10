"""Tests for the bundled synphot reference data (roman_disperser.refdata).

These tests only require synphot, NOT stsynphot or PYSYN_CDBS.
"""

import numpy as np
import pytest
import synphot as syn
from astropy import units as u

from roman_disperser import refdata


class TestF158Band:
    def test_loads(self):
        band = refdata.get_f158_band()
        assert isinstance(band, syn.SpectralElement)

    def test_pivot_wavelength(self):
        band = refdata.get_f158_band()
        pivot = band.pivot().to(u.AA).value
        # F158 pivot should be ~15749 Angstrom
        assert 15700 < pivot < 15800

    def test_throughput_range(self):
        band = refdata.get_f158_band()
        tp = band(band.waveset)
        assert tp.min().value >= 0.0
        assert tp.max().value < 1.0
        assert tp.max().value > 0.5  # peak throughput should be substantial

    def test_normalize_spectrum(self):
        """Verify that normalization to ABmag works with the bundled bandpass."""
        band = refdata.get_f158_band()
        star = refdata.get_template("g0v")
        norm = star.normalize(20.0 * u.ABmag, band=band)
        # Evaluate at a wavelength in-band
        flux = norm(1.55 * u.um, flux_unit=syn.units.FLAM)
        assert flux.value > 0


class TestTemplates:
    @pytest.mark.parametrize("name", ["g0v", "bz77_bz_24", "kc96_elliptical", "kc96_starb1"])
    def test_loads(self, name):
        sp = refdata.get_template(name)
        assert isinstance(sp, syn.SourceSpectrum)

    def test_g0v_alias(self):
        sp1 = refdata.get_template("g0v")
        sp2 = refdata.get_template("bz77_bz_24")
        wl = np.linspace(5000, 20000, 100) * u.AA
        np.testing.assert_array_equal(
            sp1(wl, flux_unit=syn.units.FLAM).value,
            sp2(wl, flux_unit=syn.units.FLAM).value,
        )

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            refdata.get_template("nonexistent")

    @pytest.mark.parametrize("name", ["g0v", "kc96_elliptical", "kc96_starb1"])
    def test_has_flux_in_wfi_range(self, name):
        """Templates should have non-zero flux in the WFI grism range."""
        sp = refdata.get_template(name)
        wl = np.linspace(9000, 20000, 100) * u.AA
        flux = sp(wl, flux_unit=syn.units.FLAM).value
        assert np.any(flux > 0)

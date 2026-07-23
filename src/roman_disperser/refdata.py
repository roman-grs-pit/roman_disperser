"""Synphot reference data for the Roman WFI grism pipeline.

Provides the F158 bandpass and a few spectral templates without requiring
stsynphot or a PYSYN_CDBS installation.  The files are vendored reference data
located via :func:`roman_disperser.paths.synphot_dir` (see that module for how
the data directory is resolved); they were extracted from the STScI
calibration database (see ``synphot/README.md`` for provenance).
"""

from roman_disperser.paths import synphot_dir

# f_lambda of a 0-AB-mag flat-f_nu source: f_nu(0 AB) = 3631 Jy
# = 3.631e-20 erg s^-1 cm^-2 Hz^-1, and f_lambda = f_nu * c / lambda^2 with
# c in Å/s, so f_lambda = FLAM_0AB_COEFF / lambda_Å^2 in erg s^-1 cm^-2 Å^-1.
# Stored as the product so the derivation *is* the value (= 0.1088546...);
# earlier per-script copies carried two hand-rounded values (0.108866,
# 0.10885) — a 1.5e-4 relative spread, harmless but confusing.
FLAM_0AB_COEFF = 3.631e-20 * 2.99792458e18

# Available template names and their filenames
_TEMPLATES = {
    "bz77_bz_24": "bz77_bz_24.fits",
    "g0v": "bz77_bz_24.fits",  # alias
    "kc96_elliptical": "kc96_elliptical_template.fits",
    "kc96_starb1": "kc96_starb1_template.fits",
}


def get_f158_band():
    """Load the Roman WFI F158 bandpass.

    Returns
    -------
    synphot.SpectralElement
        The F158 throughput curve.
    """
    import synphot as syn

    path = synphot_dir() / "roman_wfi_f158.fits"
    return syn.SpectralElement.from_file(str(path))


def get_f184_band():
    """Load the Roman WFI F184 bandpass.

    Returns
    -------
    synphot.SpectralElement
        The F184 throughput curve.
    """
    import synphot as syn

    path = synphot_dir() / "roman_wfi_f184.fits"
    return syn.SpectralElement.from_file(str(path))



def get_template(name):
    """Load a bundled spectral template.

    Parameters
    ----------
    name : str
        Template name. Available templates:

        - ``"g0v"`` or ``"bz77_bz_24"`` — Bruzual 1977 G0V stellar template
        - ``"kc96_elliptical"`` — Kinney-Calzetti elliptical galaxy
        - ``"kc96_starb1"`` — Kinney-Calzetti starburst galaxy

    Returns
    -------
    synphot.SourceSpectrum
        The spectral template.

    Raises
    ------
    ValueError
        If ``name`` is not a recognized template.
    """
    if name not in _TEMPLATES:
        available = [k for k in _TEMPLATES if k != "g0v"]  # skip alias in listing
        raise ValueError(
            f"Unknown template {name!r}. Available: {available}"
        )
    import synphot as syn

    path = synphot_dir() / _TEMPLATES[name]
    return syn.SourceSpectrum.from_file(str(path))

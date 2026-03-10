"""Pre-bundled synphot reference data for the Roman WFI grism pipeline.

Provides the F158 bandpass and a few spectral templates without requiring
stsynphot or a PYSYN_CDBS installation.  All data files live in
``data/synphot/`` and were extracted from the STScI calibration database
(see ``data/synphot/README.md`` for provenance).
"""

from pathlib import Path

import synphot as syn

# Resolve data directory relative to this file:
# src/roman_disperser/refdata.py -> ../../data/synphot/
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synphot"

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
    path = _DATA_DIR / "roman_wfi_f158.fits"
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
    path = _DATA_DIR / _TEMPLATES[name]
    return syn.SourceSpectrum.from_file(str(path))

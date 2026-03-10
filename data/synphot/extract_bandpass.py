#!/usr/bin/env python
"""Extract the Roman WFI F158 bandpass from stsynphot and save to FITS.

This script requires stsynphot and a configured PYSYN_CDBS environment
variable. It is NOT part of the regular test suite — run it manually
when the upstream throughput curve needs to be refreshed.

Usage:
    pixi run python data/synphot/extract_bandpass.py
"""

from pathlib import Path

import numpy as np


def extract_and_verify():
    import stsynphot as stsyn
    import synphot as syn

    out_path = Path(__file__).parent / "roman_wfi_f158.fits"

    # Extract from stsynphot (requires PYSYN_CDBS)
    f158_orig = stsyn.band("roman, wfi, f158")
    pivot_orig = f158_orig.pivot().value
    print(f"Extracted F158 bandpass: pivot = {pivot_orig:.1f} Angstrom")
    print(f"  Waveset: {f158_orig.waveset.shape[0]} points, "
          f"{f158_orig.waveset[0]:.0f} to {f158_orig.waveset[-1]:.0f}")

    # Save
    f158_orig.to_fits(str(out_path), overwrite=True)
    print(f"  Saved to {out_path}")

    # Verify round-trip: reload and compare
    f158_loaded = syn.SpectralElement.from_file(str(out_path))
    pivot_loaded = f158_loaded.pivot().value
    assert abs(pivot_orig - pivot_loaded) < 0.1, (
        f"Pivot mismatch: {pivot_orig:.4f} vs {pivot_loaded:.4f}"
    )
    print(f"  Round-trip OK: pivot = {pivot_loaded:.1f} Angstrom")

    # Verify normalization produces the same result
    from astropy import units as u

    star = syn.SourceSpectrum.from_file(
        str(Path(__file__).parent / "bz77_bz_24.fits")
    )
    wl = np.linspace(9000, 20000, 1000) * u.AA

    norm_orig = star.normalize(0.0 * u.ABmag, band=f158_orig)
    norm_loaded = star.normalize(0.0 * u.ABmag, band=f158_loaded)

    flux_orig = norm_orig(wl, flux_unit=syn.units.FLAM).value
    flux_loaded = norm_loaded(wl, flux_unit=syn.units.FLAM).value

    max_reldiff = np.max(np.abs(flux_orig - flux_loaded) / (np.abs(flux_orig) + 1e-30))
    assert max_reldiff < 1e-6, f"Normalization mismatch: max relative diff = {max_reldiff:.2e}"
    print(f"  Normalization test OK: max relative diff = {max_reldiff:.2e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    extract_and_verify()

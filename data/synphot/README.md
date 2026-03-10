# Synphot Reference Data

Pre-extracted reference spectra and bandpasses from the STScI
[synphot](https://synphot.readthedocs.io/) calibration database (CDBS).
These files allow the pipeline and notebooks to run without requiring
`stsynphot` or a full `PYSYN_CDBS` installation.

## Files

| File | Source | Description |
|------|--------|-------------|
| `roman_wfi_f158.fits` | `stsyn.band("roman, wfi, f158")` | WFI F158 bandpass throughput curve (109 points, 13150-18550 A) |
| `bz77_bz_24.fits` | `$PYSYN_CDBS/grid/bz77/bz_24.fits` | Bruzual 1977 G0V stellar template |
| `kc96_elliptical_template.fits` | `$PYSYN_CDBS/grid/kc96/elliptical_template.fits` | Kinney-Calzetti 1996 elliptical galaxy template |
| `kc96_starb1_template.fits` | `$PYSYN_CDBS/grid/kc96/starb1_template.fits` | Kinney-Calzetti 1996 starburst galaxy template |

## Provenance

The F158 bandpass was extracted from `stsynphot` using `extract_bandpass.py`
(see below). The template spectra were copied directly from the CDBS
`grid/` directory.

All files are in standard synphot FITS format and can be loaded with
`synphot.SpectralElement.from_file()` (bandpass) or
`synphot.SourceSpectrum.from_file()` (templates).

## Regenerating the bandpass

If the upstream F158 throughput curve is updated, re-extract it:

```bash
# Requires stsynphot and PYSYN_CDBS to be configured
pixi run python data/synphot/extract_bandpass.py
```

This extracts the bandpass, verifies the round-trip, and compares against
the bundled version. See `extract_bandpass.py` for details.

# Spectral Lookup (2026-03-26)

Looked up input catalog SEDs for two sources from the full-field simulation
(`ra10_dec0_pa0` pointing) and verified pipeline self-consistency by extracting
spectra from the simulated grism images using our own optical model trace.

## Scripts

- **`lookup_spectra.py`** — Given a list of (RA, Dec), matches against the
  pipeline source manifest (order 1), retrieves the input SED from the
  galacticus catalog, computes expected count rates using per-SCA sensitivity
  curves, and writes per-source ECSV files and plots.

- **`extract_and_compare.py`** — Reads the ECSV files, loads the FITS MODEL
  image, extracts a 1D spectrum along the trace (boxcar, cross-dispersion
  sum scaled by dy/dlambda), and plots the extracted vs expected spectrum
  with a 2D spectral cutout.

## Targets

| # | RA | Dec | Type | F158 | z | Notes |
|---|----|----|------|------|---|-------|
| 1 | 9.98153 | -0.02915 | PSF (star) | 16.50 | — | Clean extraction, good match |
| 2 | 9.91600 | -0.01960 | SER (galaxy) | 19.26 | 0.061 | Paschen series lines visible |

## Key Findings

1. **Pipeline is self-consistent.** Extracted spectra from the simulated images
   match the input SED x sensitivity x dlam to high accuracy when properly
   scaled by the dispersion (dy/dlambda).

2. **Emission lines land at correct wavelengths.** The galaxy's Paschen-beta
   line at 13606 A appears at the right position in both the input SED and the
   self-extracted spectrum.

3. **Zeroth-order contamination.** A bright zeroth-order image from a nearby
   star lands just blueward of the Pa-beta line in the galaxy's first-order
   trace. In an external extraction (see `external_extraction_galaxy.png`),
   this zeroth-order contaminant masquerades as an emission line near 13000 A,
   which could be mistaken for a real spectral feature.

## TODO

- **Zeroth-order masking.** The zeroth-order contamination is clearly visible
  in the galaxy extraction. We should investigate identifying and masking
  zeroth-order images in first-order extractions. The pipeline already tracks
  which sources have order-0 traces overlapping each SCA (in the source
  manifest), so the information needed to build a contamination model is
  available. This would be important for any spectral extraction pipeline
  built on top of these simulations.

- **PA=10 comparison.** Re-run the same two targets with the PA=10 pointing.
  The zeroth-order contaminant should land at a different position on the
  detector, confirming that the spurious feature near Pa-beta disappears or
  moves. This would cleanly separate the real emission line from the
  contamination.

## Files

| File | Description |
|------|-------------|
| `targets.txt` | Input (RA, Dec) list |
| `source_01_cat42936.ecsv` | Star SED + counts/s (ECSV) |
| `source_01_cat42936.png` | Star input SED plot |
| `source_01_cat42936_extracted.png` | Star extracted vs expected |
| `source_02_cat660940.ecsv` | Galaxy SED + counts/s (ECSV) |
| `source_02_cat660940.png` | Galaxy input SED plot |
| `source_02_cat660940_extracted.png` | Galaxy extracted vs expected |
| `external_extraction_galaxy.png` | External extraction for comparison |

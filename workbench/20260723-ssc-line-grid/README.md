# SSC line-grid simulation campaign (2026-07-23)

Simulations of the line-test field to ship to the SSC, built on the validated
line-centering harness (see `docs/` + `scripts/check_line_centering.py`, and
the 2026-07-03 validation). Every star carries the *same* known spectrum —
five narrow emission lines at exactly known wavelengths — so the SSC can
check their disperser wavelength→pixel solution line by line against ours,
with full per-line truth tables for localizing any disagreement.

## Matrix

| Axis | Values |
|---|---|
| SED | lines-only (`--no-continuum`) · lines+continuum (wave 2, TBD level) |
| PSF | STPSF caches · synthetic centred Gaussians (FWHM 2.5 native px) |
| PA  | 0° · 10° (same field: RA 10°, Dec 0°, EXPTIME 190.22 s) |
| SCAs | all 18, order 0/1/2, seed 42 |

Wave 1 = the four lines-only runs. The line set (centers 11000/12500/15500/
17000/19000 Å, FWHM 10 Å, amps 5–10× local continuum) is identical across
all runs; the continuum is toggled by the catalog builder after the summed
SED is built, so line FLAM is preserved exactly between SED sets.

## Output layout (`/mnt/roman-science/grs/line-tests-20260723/`)

```
catalog_lines/                    # shared lines-only catalog + lines.ecsv + provenance.json
lines_{stpsf,gauss}/pointing_{pa000,pa010}_001.001.001.001.001.001/
    grism_*_detSCA??.fits         # MODEL (noise-free counts/s) + ISIM (Poisson counts)
    grism_*_sources.parquet       # per (source, SCA, order) manifest
    grism_*_meta.yaml             # pointing metadata + RNG keys
    residuals.parquet             # closure check (measured - predicted centroids)
    truth_lines.parquet           # per (source, SCA, order, line) predicted positions
    truth_lines_provenance.json
lines_{stpsf,gauss}_l2/...        # romanisim L2 ASDF (extra_counts branch), same tree
scripts/                          # verbatim copy of this directory at submission
slurm-meta/                       # per-submission task script + env audit
logs/                             # romanisim per-file logs
```

## Reproduce

```bash
# 0. Gaussian PSF caches for all 18 SCAs (one-time; correctness gate for gauss runs)
pixi run python scripts/make_gaussian_psf_cache.py --scas $(seq 1 18) \
    --out-dir /data/npadman/1-Projects/roman/roman_disperser/line_test_outputs/psf_cache_gaussian

# 1. Shared lines-only catalog (fixed-src_index builder, commit >= 1fa0a7a)
pixi run python scripts/build_line_test_catalog.py --no-continuum \
    --out-dir /mnt/roman-science/grs/line-tests-20260723/catalog_lines

# 2. Per-run pipeline (from this directory)
bash submit.sh sim     stpsf pa000     # ... and {stpsf,gauss} x {pa000,pa010}
bash submit.sh wrap    stpsf pa000     # after the sim lands
bash submit.sh closure stpsf pa000     # after the sim lands
```

Acceptance bar for the closure check (from the 2026-07-03 validation):
along-dispersion median ≈ 0.001 px, within-SCA MAD 0.012–0.028 px for STPSF;
Gaussian runs should null the per-SCA PSF-asymmetry offset to ≈ 0.001 px.

Caveats that travel with the deliverable: a "measured position" is the
flux-weighted centroid (≈0.08 px from the PSF peak for real STPSF PSFs), and
any centroid on grism data must remove the dispersed continuum first (wave 2
images); see the validation report for both.

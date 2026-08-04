# Workbench

Workspace for comparisons, experiments, and analyses that use `roman_disperser` but aren't part of the library itself.

## Convention

Each study lives in a date-prefixed subdirectory: `YYYYMMDD-short-description/`.

Large data files go in a `data/` subdirectory within each study (ignored by git).

## Contents

| Directory | Description |
|-----------|-------------|
| `20260220-grizli-comparison/` | Cross-code comparison against grizli grism simulation |
| `20260321-galacticus-sed-units/` | Galacticus SED unit investigation |
| `20260323-pipeline-validation/` | Unified pipeline visual validation (cutouts, spectra) |
| `20260323-batch-tuning/` | Batch size and performance optimization |
| `20260324-benchmarks/` | GPU scaling and compilation benchmarks |
| `20260326-spectral-lookup/` | Input SED lookup and self-extraction consistency check |
| `20260414-acceptance-testing/` | NERSC acceptance test driver (single 4xA100 node, ramdisk-staged) |
| `20260505-acceptance-testing-aws/` | AWS rerun after catalog SED scrubber fix; SLURM-array on gpu-med a10g |
| `20260508-romanisim-wrap/` | Romanisim L2 wrap of the spectro acceptance run (mem-lg array + S3 archive) |
| `20260731-ssc-line-grid-30x/` | SSC line-grid wave 1b: lines-only SED, 30x line flux, MA 1036 (STPSF/Gaussian × PA 0/10; truth tables) |
| `20260731-ssc-line-grid-cont/` | SSC line-grid wave 2: as wave 1b plus a flat-f_nu continuum pedestal |
| `20260731-ssc-line-grid-gal/` | SSC line-grid wave 3: as wave 1b on Sersic galaxies (morphology-induced centroid shift) |
| `20260731-ssc-line-grid-analysis/` | v0.13.0 rerun before/after analysis + addendum deck generator |

### A note on the deleted campaigns

The `20260723` and `20260724-*` line-grid dirs were removed on the
`feature/optical-model-line-test` merge. `20260723` was an exploratory run
superseded by `20260724` (different MA table and line flux); the `20260724-*`
campaigns are the **pre-fix** runs that the `-analysis/` before/after compares
against, and their products are still live at
`/mnt/roman-science/grs/line-tests-20260724{,-cont,-gal}/`, each with its
`submit.sh`, configs, pointing ECSVs and `make_truth_tables.py` mirrored into
`scripts/` beside the data. The wave-1b deck lives at that store too, and its
figures and generator are in the research log
(`2026-07-24/assets/`). Their repo history is preserved in this branch's merge
commit — do not squash-merge it.

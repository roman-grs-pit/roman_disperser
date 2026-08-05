# Workbench

Workspace for comparisons, experiments, and analyses that use `roman_disperser` but aren't part of the library itself.

## Convention

Each study lives in a date-prefixed subdirectory: `YYYYMMDD-short-description/`.
A study that ran as several related campaigns may nest one level below that
(see `20260731-ssc-line-grid/`).

Large data files go in a `data/` subdirectory within a study (ignored by git at
any depth, via `workbench/**/data/`).

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
| `20260731-ssc-line-grid/` | SSC line-grid campaign, v0.13.0 rerun — three waves plus the before/after analysis (see below) |

### `20260731-ssc-line-grid/`

The v0.13.0 rerun of the SSC line-grid waves, on the line-test field
(STPSF/Gaussian × PA 0/10). Each wave has its own run store; the drivers,
configs, pointing ECSVs and `make_truth_tables.py` are mirrored into
`scripts/` beside the data at submit time, so the recipe travels with the
products it made.

| Subdirectory | Wave | Run store |
|---|---|---|
| `wave1b/` | lines-only SED, 30× line flux, MA 1036 | `/mnt/roman-science/grs/line-tests-20260731/` |
| `wave2/` | as wave 1b plus a flat-f_nu continuum pedestal (anchor mag 16.3071968) | `…/line-tests-20260731-cont/` |
| `wave3/` | as wave 1b on Sérsic galaxies (morphology-induced centroid shift) | `…/line-tests-20260731-gal/` |
| `analysis/` | before/after vs the 20260724 pre-fix runs; addendum deck generator | writes to `…/line-tests-20260731/analysis/` |

### A note on the deleted campaigns

The `20260723` and `20260724-*` line-grid dirs were removed on the
`feature/optical-model-line-test` merge. `20260723` was an exploratory run
superseded by `20260724` (different MA table and line flux); the `20260724-*`
campaigns are the **pre-fix** runs that the `-analysis/` before/after compares
against, and their products are still live at
`/mnt/roman-science/grs/line-tests-20260724{,-cont,-gal}/`, each with its
`submit.sh`, configs, pointing ECSVs and `make_truth_tables.py` mirrored into
`scripts/` beside the data. The wave-1b deck lives at that store too, and its
figures and generator are in the research log (`2026-07-24/assets/`).

Their repo history is preserved: the branch was merged with a merge commit, not
squashed, precisely so that deleting them lost nothing. **But `git log` will not
show it by default** — history simplification follows a single parent through the
merge, and these paths do not exist in the result, so a plain
`git log -- <path>` returns *empty* and reads as "never existed". Use
`--full-history`:

```bash
# find the commits that added and removed a deleted campaign file
git log --full-history --oneline -- workbench/20260724-ssc-line-grid-30x/submit.sh

# recover its content from any commit that had it
git show <commit>:workbench/20260724-ssc-line-grid-30x/submit.sh
```

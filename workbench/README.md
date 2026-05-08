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

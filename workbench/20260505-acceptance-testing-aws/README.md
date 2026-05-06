# Spectro acceptance test, AWS rerun (2026-05-05)

Rerun on AWS after the 2026-04-30 NERSC run hit 5/32 failures: 3 from
corrupt galaxy SEDs in `catalogs_padded/seds.zarr`, 2 from SLURM
wall-time kills.

Fixes: load-time SED scrubber in `build_grism_image.py:load_galaxy_seds`
(commit `29e0f6e`); 1 pointing/task layout removes the wall-time risk.

## Files

| File | Purpose |
|---|---|
| `slurm_warmup_grism.sh` | 5-task array, parallel JIT cache warmup. |
| `slurm_run_grism.sh`    | 5-task production array. Each task is one long-running Python process owning ~6-7 pointings via `--num-workers 5` round-robin partitioning. |
| `config.yaml`           | Pipeline config consumed by both drivers. |

Verbatim runtime copies live at
`/mnt/roman-science/.../spectro/2026-05-05/acceptance/scripts/`.
This directory is the version-controlled mirror.

## How to run

Both drivers are idempotent: they figure out paths from their own
location and write all artifacts under `acceptance/`. Run them in
order from any login node:

```bash
ACCEPT=/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance

# 1. Parallel JIT warmup (~12 min, 5 a10g, ~$0.35).
bash $ACCEPT/scripts/slurm_warmup_grism.sh

# 2. Production run (5 a10g, ~10-11 h wall, ~$19).
bash $ACCEPT/scripts/slurm_run_grism.sh

# Sanity check on a single pointing: use a one-row ECSV (copy the row
# of interest into a new file alongside the full ECSV, change the
# driver's ECSV path or symlink, and run with NUM_WORKERS=1).
```

The production run is safely re-runnable: each pointing dir is
skipped if it already exists (`build_grism_image.py:1464`), so
re-submitting after a partial completion just fills in the missing
pointings — useful when a long-running worker hits a wall-time limit
or transient failure.

Both drivers expect:
- The SED scrubber commit (`29e0f6e`) on the active branch — the
  pre-fix code crashes on 3 known-bad catalog galaxies.
- `acceptance-testing-spectro.sim.ecsv` at the top of
  `/mnt/roman-science/grs/acceptance-testing-20260430/`.
- The unified catalog at
  `/mnt/roman-science/grs/acceptance-testing-20260430/catalogs_padded/`.
- PSF cache, sensitivities, and optical model under
  `/data/npadman/1-Projects/roman_disperser/data/`.

## Outputs

Per pointing (in `acceptance/output/<ecsv>_<plan>.<pass>...<exposure>/`):
18 FITS + 18 PNG + mosaic PNG + sources.parquet + meta.yaml.
See `docs/grism_pipeline.md` for the FITS/parquet schema.

Per-submission audit lands in `acceptance/slurm-meta/grism-<JobID>.env`
(git SHA, partition, mem, time, ECSV path, cache dir, etc.).

## Hardware + cache notes

`--gres=gpu:a10g:1` pins to g5.2xlarge (4 CPU, 31 GB, 1xa10g 24GB,
~$0.35/hr); `gpu-med` also has l4 / l40s nodes that we don't want
mixed in.

`--cache-dir /data/npadman/jax-cache-grism` is passed explicitly on
the CLI so the location shows up in scheduler logs and audit files;
`/data/npadman/` is shared NFS visible to every compute node.

Per-SCA timing observed on a10g (validation run, warm cache):
~5 min/SCA, ~1h40/pointing wall (18 SCAs serialized). Full-pipeline
JIT compile is ~36 s/SCA on a warm cache vs ~75 s cold — NFS read
of 6 x 111 MB cache files dominates the residual cost on warm hits.

# Spectro acceptance test, AWS rerun (2026-05-05)

Rerun of the `acceptance-testing-spectro` pointing table on AWS after the
2026-04-30 NERSC run hit 5 of 32 failures (3 caused by 3 corrupt galaxy
SEDs in `catalogs_padded/seds.zarr`, 2 by SLURM wall-time kills from a
4-worker x 8-pointing layout).

The catalog SED corruption is fixed by a load-time scrubber in
`scripts/build_grism_image.py:load_galaxy_seds` (commit `29e0f6e`). The
wall-time kills are eliminated by going to one pointing per SLURM task.

## Files in this directory

| File | Purpose |
|---|---|
| `slurm_warmup_grism.sh` | 5-task array, parallel JIT cache warmup (one a10g per task, ~12 min wall). |
| `slurm_run_grism.sh`    | 32-task array, one pointing per task, gpu-med a10g (max 5 concurrent). |
| `config.yaml`           | Pipeline config (paths to /mnt/roman-science catalog and /data/npadman project data). |

The verbatim run-time copies live alongside the outputs at
`/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance/scripts/`.
This directory is the version-controlled mirror.

## Layout assumed by the SLURM drivers

```
/mnt/roman-science/grs/acceptance-testing-20260430/
|-- acceptance-testing-spectro.sim.ecsv         # 32 GRISM rows, top-level
`-- spectro/
    `-- 2026-05-05/
        `-- acceptance/
            |-- config.yaml                      # this file's twin
            |-- output/                          # per-pointing dirs
            |-- logs/                            # per-task app logs
            |-- slurm-meta/                      # per-submission audit
            `-- scripts/                         # this directory's twin
```

## How to run

```bash
ACCEPT=/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance

# 1. Parallel warmup (~12 min wall, 5 a10g, ~$0.35).
bash $ACCEPT/scripts/slurm_warmup_grism.sh

# 2. Single-pointing validation (016.001 = ECSV row 31), ~70 min, ~$0.40.
SUBSET=31 bash $ACCEPT/scripts/slurm_run_grism.sh

# 3. Full 32-pointing array, ~7.5 h wall on 5 a10g, ~$13.
bash $ACCEPT/scripts/slurm_run_grism.sh
```

The cache_dir is `/data/npadman/jax-cache-grism` (EBS, persistent across
SLURM jobs and reboots — distinct from the cluster's default `/tmp`).
Both drivers pass `--cache-dir` explicitly so the cache location appears
in scheduler logs and audit env files.

## Hardware notes

| Partition | Node | Cost/hr | CPUs | Mem | GPU |
|---|---|---|---|---|---|
| `gpu-med` | g5.2xlarge (a10g) | ~$0.35 | 4 | 31 GB | 1xa10g 24GB |

The drivers pin `--gres=gpu:a10g:1` because gpu-med also includes l4 and
l40s nodes; performance baseline below was measured on a10g.

## Performance baseline (per-SCA, this a10g)

From `/data/npadman/tmp/debug-grism-nan/repro_after_fix.log` (008.001
SCA 2 with the scrubber active):

- JIT compile (cold): 169 s
- Dispersion compute: 221 s
- Total per SCA: 390 s (~6.5 min)
- 18 SCAs serialized per pointing: ~120 min cold, ~70 min warm

These numbers feed the `TIME=02:30:00` and `TIME=30:00` defaults in the
respective drivers.

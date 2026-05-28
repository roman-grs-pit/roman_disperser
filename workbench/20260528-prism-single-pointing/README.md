# Prism single-pointing test (2026-05-28)

First end-to-end PRISM (P127) grism simulation: one pointing at
RA,Dec = (10, 0), the center of the `prism-testing-20260527` catalog
footprint. Produces the disperser output files (FITS + PNG + mosaic +
sources.parquet + meta.yaml) for downstream use. The romanisim L2 wrap is
deferred (see `workbench/20260508-romanisim-wrap/` for that chain).

Built in analogy to `20260505-acceptance-testing-aws/`, but a single
pointing run interactively on one a10g rather than a SLURM array.

## Files

| File | Purpose |
|---|---|
| `config.yaml`            | Pipeline config (prism paths, output on /mnt, catalog on EBS). |
| `prism-single.sim.ecsv`  | One-row APT pointing table, `BANDPASS=PRISM`. |
| `run.sh`                 | Stages the catalog to EBS, then disperses. Optional SCA arg for a smoke test. |
| `data/`                  | Git-ignored. Holds the EBS-staged catalog + derived smoke config. |

## How to run

The `cuda` env only activates on a GPU node (it needs the `__cuda` virtual
package), so grab an a10g first. Pin the GRES -- `gpu-med` also has pricier
l40s nodes:

```bash
salloc --partition=gpu-med --gres=gpu:a10g:1 --cpus-per-task=4 \
       --mem=24G --time=4:00:00

cd /data/npadman/1-Projects/roman_disperser_prism/workbench/20260528-prism-single-pointing

bash run.sh 5      # smoke test: SCA 5 only  -> .../output-smoke   (~5-6 min)
bash run.sh        # full: all 18 SCAs       -> .../output         (~1h40 cold)

scancel <jobid>    # release the a10g when done (~$1.2/h on-demand)
```

The first run compiles JIT cold (~170 s/SCA) into `/data/npadman/jax-cache-prism`;
later runs hit the warm cache (~15 s/SCA). No separate warmup step -- not
worth it for a single pointing.

## Key parameters

| Parameter | Value | Source / rationale |
|---|---|---|
| Pointing | RA=10, Dec=0, PA=0 | Catalog center (RA [9,11], Dec [-1,1]); PA arbitrary for a test. |
| BANDPASS | `PRISM` | Pipeline filters `BANDPASS == "PRISM"` (build_grism_image.py:1430). |
| MA table | `1041` | `SP_450_16` (`SP_450_16_HLTDS`), from CRDS `roman_wfi_matable_0003.asdf`. |
| EXPOSURE_TIME | `452.42195 s` | SP_450_16 effective exposure (15 science resultants). Drives Poisson noise. |
| Order / band | "1", 0.75-1.85 um | Prism defaults baked into `pipeline.py` (ORDERS, LAM_MIN/MAX). |
| seed | 42 | RNG reproducibility. |

`DURATION` and `TARGET_NAME` in the ECSV are not read by the pipeline
(cosmetic, kept for APT schema parity).

## Path strategy

- **Catalog on EBS:** `run.sh` rsyncs the catalog from `/mnt` (S3-Files) to
  `data/catalogs` first. S3-Files reads ~2.8x slower than EBS for the many
  small SED chunks. The rsync is incremental, so re-runs are cheap.
- **Outputs on /mnt:** `output_dir` is under
  `/mnt/roman-science/grs/prism-testing-20260527/spectro/2026-05-28/`. FITS/PNG
  are plain writes -- the zarr/hardlink limitation of the S3 mount (which
  bit the catalog build) does not apply here.
- **JAX cache on EBS:** `/data/npadman/jax-cache-prism`, persists across runs.

## Notes / open items

- **No catalog padding needed.** The AWS acceptance run used `catalogs_padded`
  for RA-edge pointings; at center (10,0) with `cone_radius=0.6` we stay well
  inside the footprint, so the raw catalog is fine. Edge pointings would need
  padding.
- **Fresh-catalog QC:** watch the run log for the `load_galaxy_seds` scrubber
  warning count -- a non-zero count flags pathological SED bins in this
  freshly built catalog.
- **Smoke vs full output dirs are separate** (`-smoke` suffix) on purpose; see
  the skip-if-exists note in `run.sh`.

## Outputs (per pointing dir)

`prism-single_001.001.001.001.001.001/`: up to 18 FITS + 18 PNG + mosaic PNG
+ `*_sources.parquet` + `*_meta.yaml`. FITS/parquet schema in
`docs/grism_pipeline.md`.

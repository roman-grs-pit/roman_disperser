# Prism single-pointing test (2026-05-28)

First end-to-end PRISM (P127) grism simulation: one pointing at
RA,Dec = (10, 0), the center of the `prism-testing-20260527` catalog
footprint. Produces the disperser output files (FITS + PNG + mosaic +
sources.parquet + meta.yaml) and then wraps them through romanisim to
L2 ASDF -- both steps stay in this directory rather than spinning off
a separate romanisim-wrap workbench (the grism acceptance run did that
at scale; for one pointing it's overkill).

Built in analogy to `20260505-acceptance-testing-aws/`, but a single
pointing run interactively on one a10g rather than a SLURM array.

## Roll extension (2026-06-03)

Extended the original PA=0 pointing with three additional roll angles at
the same RA,Dec=(10,0): **PA = 10, 170, 180**. Each is a new row in
`prism-single.sim.ecsv` with a distinct `OBSERVATION` id (2, 3, 4) so it
gets its own output dir and RNG key; the original PA=0 row (OBSERVATION 1)
is untouched and auto-skipped on re-run (directory-level skip-if-exists).

| Roll | OBSERVATION | Pointing dir suffix |
|---|---|---|
| PA=0   | 1 | `_001.001.001.001.001.001` (original) |
| PA=10  | 2 | `_001.001.001.002.001.001` |
| PA=170 | 3 | `_001.001.001.003.001.001` |
| PA=180 | 4 | `_001.001.001.004.001.001` |

No catalog padding needed: the cone search is roll-invariant and the
center is unchanged, so `cone_radius=0.6` around (10,0) stays inside the
footprint exactly as for PA=0. Dispersion of the 3 new rolls ran
2026-06-03 on one a10g (`gpu-int`) against the warm JAX cache: 4.0 h
total (~67 min/roll), clean (no NaN / SED-scrubber warnings). The
romanisim wrap + L2 quick-looks (Steps 2-3 below) then cover all four
pointings -- the wrap globs `**/grism_*_detSCA*.fits` recursively and
skips per-file, so the original PA=0 ASDFs are reused.

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

- **Catalog read directly from /mnt** (S3-Files). For a single pointing the
  S3 read penalty (~2.8x; a few min total across 18 SCAs, since galaxy SEDs
  are read per-SCA for the on-detector subset) is smaller than the ~5 min
  cost of staging the 19 GB catalog to EBS, so direct is simpler and no
  slower end-to-end. This matches the acceptance run. **Stage to EBS only
  for multi-pointing runs**, where the one-time copy amortizes
  (project_s3_vs_ebs_seds).
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

## Step 2: romanisim L2 wrap (interactive, head node)

After the dispersion run is on disk, wrap the 18 FITS through romanisim
to produce L2 ASDF products. The wrap is CPU-only (no JAX/GPU), so it
runs in the `romanisim` pixi env on whatever node you happen to be on
-- no SLURM array, no separate workbench dir. For 18 files this is
trivially small (~1-2 hours on a 2-core head node at N=2 threads).

`scripts/wrap_with_romanisim.py` is the underlying tool; it walks the
input dir for `grism_*_detSCA*.fits`, reads pointing + RNG headers,
and shells out to `romanisim-make-image --extra-counts <fits> ISIM`
per file via a `ThreadPoolExecutor`. Per-file skip-if-exists makes
restarts cheap.

Header-to-romanisim argument mapping (see `scripts/wrap_with_romanisim.py`
docstring for the full list):

| FITS header        | Romanisim arg |
|--------------------|---|
| `WFICENRA, WFICENDEC` | `--radec` |
| `WFICENPA - 60`       | `--roll`  (focal-plane vs spacecraft) |
| `DETNUM`              | `--sca` |
| `MA_TABLE`            | `--ma_table_number` |
| `RNDSEED0 ^ RNDSEED1` | `--rng_seed` (XOR fold to 32-bit) |

Static args: `--bandpass PRISM` (prism branch default; flipped from
`GRISM` in `wrap_with_romanisim.py`), `--usecrds`, `--stpsf`, `--nobj 0`,
`--date 2026-01-01T12:00:00.000`, `--level 2`.

### How to run

```bash
RUN=/mnt/roman-science/grs/prism-testing-20260527/spectro/2026-05-28
mkdir -p $RUN/logs/romanisim
cd /data/npadman/1-Projects/roman_disperser_prism

pixi run -e romanisim python scripts/wrap_with_romanisim.py \
    --input-dir  $RUN/output \
    --output-dir $RUN/output_l2 \
    --log-dir    $RUN/logs/romanisim/per-file \
    --num-threads 2 \
    > $RUN/logs/romanisim/worker.log 2>&1 &
```

`--num-threads` should match available cores. On this head node (2
cores, 30 GB) N=2 keeps memory comfortable; the acceptance run hit
~48 GB peak at N=8 on `mem-lg` for reference.

### Output layout

L2 ASDFs land in a sibling tree mirroring the input dir name:

```
output_l2/prism-single.sim_001.001.001.001.001.001/
    grism_prism-single.sim_001.001.001.001.001.001_detSCA01_l2.asdf
    ...
    grism_prism-single.sim_001.001.001.001.001.001_detSCA18_l2.asdf
```

Per-file size ~216 MB, total ~4 GB.

### Why bandpass PRISM (not GRISM)

Romanisim supports both as first-class options
(`romanisim/models/bandpass.py:64-65`). `PRISM` maps to the SNPrism
throughput internally and triggers the spectral STPSF path
(`ris_make_utils.py:402`). The fork in use is
`roman-grs-pit/romanisim @ extra_counts`, pinned via `pixi.toml`.

## Step 3: L2 quick-look PNGs

The romanisim wrap only writes ASDF; render quick-looks from the L2 rate
images with `render_l2_per_sca.py` (this dir) for the 18 per-SCA PNGs and
the grism archetype `../20260508-romanisim-wrap/render_l2_pointing_mosaic.py`
for the sky-coordinate mosaic. Both run in the `romanisim` env (asdf +
matplotlib come in transitively).

```bash
L2=$RUN/output_l2/prism-single.sim_001.001.001.001.001.001
cd /data/npadman/1-Projects/roman_disperser_prism

# 18 per-SCA detector-space PNGs (written alongside the ASDFs)
pixi run -e romanisim python \
    workbench/20260528-prism-single-pointing/render_l2_per_sca.py "$L2"

# Sky-coordinate RA/Dec mosaic of the full focal plane
pixi run -e romanisim python \
    workbench/20260508-romanisim-wrap/render_l2_pointing_mosaic.py \
    "$L2" "$L2/grism_prism-single.sim_001.001.001.001.001.001_l2_mosaic.png"
```

L2 prism rate images have a sky floor ~3.5 DN/s with ramp-fit negative
outliers, so the per-SCA renderer clamps `vmin=0` and takes `vmax` from the
p99.99 quantile of the rebinned pixels (`AsinhNorm`, `linear_width=0.5`).
Dispersed sources show as bright vertical streaks (the single-beam prism
signature). At 0.5"/px the mosaic reads as a footprint/layout check -- the
streaks are too thin to pop; the per-SCA PNGs are where the spectra show.

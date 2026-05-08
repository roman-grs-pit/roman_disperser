# Romanisim wrap + L2 mosaic + S3 archive (2026-05-08)

Post-processing chain for the 2026-05-05 spectro acceptance disperser run
(`workbench/20260505-acceptance-testing-aws/`):

1. **Wrap** disperser FITS through romanisim → L2 ASDF (`output_l2/`).
2. **Render** RA/Dec mosaic PNGs from the L2 ASDFs (`output_l2_mosaics/`).
3. **Archive** the run + pointing ECSV to the staging S3 bucket.

Verbatim runtime copies of the SLURM driver live at
`/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance/scripts/`.
This directory is the version-controlled mirror.

## Files

| File | Purpose |
|---|---|
| `slurm_run_romanisim.sh` | Step 1 driver: 4-task array on `mem-lg`, 8 internal threads each. |
| `render_l2_pointing_mosaic.py` | Step 2: per-pointing RA/Dec mosaic PNG from 18 L2 ASDFs. |
| `s3_archive.sh` | Step 3: idempotent `aws s3 sync` of the run + pointing ECSV. |

## Step 1 — Romanisim wrap

`scripts/wrap_with_romanisim.py` is the underlying tool;
`slurm_run_romanisim.sh` wraps it as a 4-task SLURM array. Each task is
one Python orchestrator processing 144/576 files via 8 internal threads
(`ThreadPoolExecutor`). Per-file outputs go to a sibling tree under
`<ACCEPT>/output_l2/<pointing>/`.

```bash
ACCEPT=/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance

# Copy the SLURM driver into the runtime mirror (one-time, after edits).
cp workbench/20260508-romanisim-wrap/slurm_run_romanisim.sh \
   $ACCEPT/scripts/

# Submit (one-shot; idempotent on re-runs).
bash $ACCEPT/scripts/slurm_run_romanisim.sh
```

### Argument derivation (per FITS header)

| FITS header | Romanisim arg |
|---|---|
| `WFICENRA, WFICENDEC` | `--radec` |
| `WFICENPA - 60` | `--roll` (focal-plane vs spacecraft offset) |
| `DETNUM` | `--sca` |
| `MA_TABLE` | `--ma_table_number` |
| `RNDSEED0 ^ RNDSEED1` | `--rng_seed` (XOR fold to 32-bit, deterministic per-SCA) |

Static: `--bandpass GRISM`, `--usecrds`, `--stpsf`, `--nobj 0`,
`--extra-counts <fits> ISIM`, `--date 2026-01-01T12:00:00.000`, `--level 2`.

`--date` is a single placeholder (drives zodi only); per-pointing dates
can be added later via a sidecar table if a science run needs it.

### Partitioning

Each worker globs `<ACCEPT>/output/**/grism_*_detSCA*.fits`, sorts the
list, and processes `[idx % num_workers == worker_index]`. All workers
log a manifest hash at startup; matching hashes confirm a clean
partition. For 576 files / 4 workers each worker gets 144 files.

### Idempotency / restart

`wrap_with_romanisim.py` skips files whose `_l2.asdf` already exists, so
re-submitting after a partial completion just fills in the gaps. A worker
dying mid-run drops at most 8 files in flight.

### Hardware + cost

`mem-lg` (r6i.4xlarge: 8 cores SMT-off, 124 GB, ~$1.01/hr). Memory peaks
around 48 GB at N=8 — comfortable headroom. Production projection from
the 2026-05-08 scaling test (jobs 5740 + 5741, audit at
`/data/npadman/tmp/romanisim-smoke-20260508/scaling_*`):

- Throughput: ~47 jobs/h-node at N=8.
- 576 files / 4 nodes / 8 threads ≈ 18 files/proc serially.
- Wall: ~3 hours (5 h SLURM time-limit gives margin).
- Cost: 4 nodes × 3 h × $1.01 ≈ **~$12**.

Compared alternatives (full scaling table in the smoke-test artifacts):

| Configuration | Wall | Cost |
|---|---|---|
| 1× `cpun-2xlg` N=22 | 5.2 h | ~$20 |
| 4× `cpun-2xlg` N=22 | 80 min | ~$20 |
| 1× `mem-lg` N=8 | 12 h | ~$12 |
| **4× `mem-lg` N=8** | **~3 h** | **~$12** |

`mem-lg` wins on $/file: the 3.85× lower hourly rate beats the per-proc
speed advantage of c5n.

### Outputs

Per pointing dir layout after the run:

```
<ACCEPT>/output_l2/<ecsv>_<plan>.<pass>.<segment>.<obs>.<visit>.<exposure>/
    grism_<dirname>_detSCA01_l2.asdf
    ...
    grism_<dirname>_detSCA18_l2.asdf
```

Per file: a single `*_l2.asdf` (~216 MB). Total ~120 GB under
`output_l2/`.

Per-submission audit lands in `<ACCEPT>/slurm-meta/romanisim-<JobID>.env`
(git SHA, partition, mem, time, input/output dirs, timestamp). Per-file
romanisim logs land in `<ACCEPT>/logs/romanisim/per-file/`.

### Sanity checks during the run

```bash
# Job + node assignment
squeue -u $USER

# Manifest hash agreement (should be identical across all 4 workers)
grep manifest_sha $ACCEPT/logs/romanisim/worker-*.log | head

# Live progress
find $ACCEPT/output_l2 -name '*.asdf' | wc -l

# Per-worker counts at end
grep DONE $ACCEPT/logs/romanisim/worker-*.log
```

### Failure handling

Per-file failures are logged but don't kill the worker; the orchestrator
prints a final `ok=N skipped=N failed=N` summary and exits 1 if any
failed. Re-running is the recovery — idempotency means the next submit
retries only the missing files.

## Step 2 — L2 mosaic PNGs

`render_l2_pointing_mosaic.py` builds a sky-coordinate (RA/Dec) mosaic
PNG of one pointing's 18 L2 ASDFs. For each detector it reads the GWCS
from `roman.meta.wcs`, rebins the image in detector pixel space, projects
each rebinned pixel center through the GWCS to (RA, Dec), then onto a
common tangent-plane WCS centered on the pointing. The result is a
single PNG per pointing showing the full WFI focal-plane "banana" in
sky coordinates with detector orientations preserved.

### One pointing

```bash
pixi run -e romanisim python \
    workbench/20260508-romanisim-wrap/render_l2_pointing_mosaic.py \
    $ACCEPT/output_l2/<pointing> \
    $ACCEPT/output_l2_mosaics/<pointing>_l2_mosaic.png
```

### All 32 pointings

Sequential loop on the head node, ~30-50 sec/pointing wall:

```bash
for d in $ACCEPT/output_l2/*/; do
    name=$(basename "$d")
    out=$ACCEPT/output_l2_mosaics/${name}_l2_mosaic.png
    [ -f "$out" ] && { echo "skip $name"; continue; }
    pixi run -e romanisim python \
        workbench/20260508-romanisim-wrap/render_l2_pointing_mosaic.py \
        "$d" "$out"
done
```

~25 min total wall on the head node for the full 32-pointing acceptance
set, ~9 MB of PNGs.

### Stretch defaults

`vmin=0`, `vmax = p99.99` (~71 DN/s for grism L2), matplotlib
`AsinhNorm(linear_width=0.5)`. This puts the sky at ~37% colormap (clear
magenta), bright dispersed sources at 80-100% (yellow), with the very
brightest pixels saturated.

To tune, see `--vmin`, `--vmax-quantile`, `--linear-width` flags. Lower
`--linear-width` makes sources brighter relative to sky (with the
brightest also saturating sooner); raising `--vmax-quantile` reduces the
saturated-pixel count.

### Output layout

Flat directory, one PNG per pointing:

```
<ACCEPT>/output_l2_mosaics/
    <ecsv>_001.001.001.001.001.001_l2_mosaic.png
    <ecsv>_001.001.001.001.001.002_l2_mosaic.png
    ...
```

## Step 3 — S3 archive

`s3_archive.sh` syncs the entire acceptance run (FITS, L2 ASDF, mosaics,
audit, logs) plus the pointing ECSV to the staging bucket. Layout
mirrors the imaging-side archive (`grs/acceptance-testing-20260430/imaging/2026-04-30/...`).

```bash
bash workbench/20260508-romanisim-wrap/s3_archive.sh
```

Run from the head node; `aws s3 sync` is idempotent and resumable.

### S3 destination layout

```
s3://spinup-003131-romanisim-l3/grs/acceptance-testing-20260430/
├── acceptance-testing-spectro.sim.ecsv         # pointing file
└── spectro/2026-05-05/acceptance/
    ├── output/                  73 GB   # disperser FITS (input to romanisim)
    ├── output_l2/              117 GB   # romanisim L2 ASDF
    ├── output_l2_mosaics/      9 MB     # quicklook PNGs (one per pointing)
    ├── slurm-meta/             56 KB    # submission audit
    ├── scripts/                20 KB    # SLURM drivers (runtime mirror)
    └── logs/                   19 MB    # worker + per-file logs
```

Total ~190 GB / 2464 objects. AWS profile `spinup-003131-romanisim-l3`.

### Throughput

For the 2026-05-08 archive (head-node, intra-region same-AZ to bucket):

| Phase | Size | Wall | Throughput |
|---|---|---|---|
| Pointing ECSV + small dirs | < 20 MB | < 10 sec | 3 MB/s (per-file overhead) |
| `output_l2/` | 117 GB | 8:11 | 240 MB/s |
| `output/` | 73 GB | 5:25 | 230 MB/s |
| **total** | **190 GB / 2464 obj** | **13:47** | **245 MB/s** |

Cost: $0 (intra-region S3 transfer, no egress charges).

### Verification

The script's final step calls `aws s3 ls --recursive` on `output_l2/`
and `output/` and prints object counts, which should match the local
`find ... -type f | wc -l`.

## Dependencies

- Romanisim pixi env solved (`pixi run -e romanisim hydrate-romanisim`
  has populated `$CRDS_PATH`).
- `$CRDS_PATH`, `$CRDS_SERVER_URL`, `$STPSF_PATH` exported in the
  worker's shell (via `~/.bashrc`).
- The 2026-05-05 dispersion run complete at `<ACCEPT>/output/`.
- AWS profile `spinup-003131-romanisim-l3` configured with
  `s3:PutObject` on `s3://spinup-003131-romanisim-l3/`.

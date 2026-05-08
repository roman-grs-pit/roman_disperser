# Romanisim wrap, AWS run (2026-05-08)

Wraps the 2026-05-05 spectro acceptance disperser FITS through romanisim
(`scripts/wrap_with_romanisim.py`) to produce L2 ASDF products. Sibling
output tree under `<ACCEPT>/output_l2/`.

This is the version-controlled mirror; the runtime copy lives at
`/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance/scripts/`.

## Files

| File | Purpose |
|---|---|
| `slurm_run_romanisim.sh` | 4-task array, one mem-lg node each, 8 internal threads. |

## How to run

```bash
ACCEPT=/mnt/roman-science/grs/acceptance-testing-20260430/spectro/2026-05-05/acceptance

# Copy this script into the runtime mirror (one-time, after edits).
cp workbench/20260508-romanisim-wrap/slurm_run_romanisim.sh \
   $ACCEPT/scripts/

# Submit (one-shot; idempotent on re-runs).
bash $ACCEPT/scripts/slurm_run_romanisim.sh
```

Per-pointing dir layout after the run:

```
$ACCEPT/output_l2/<ecsv>_<plan>.<pass>.<segment>.<obs>.<visit>.<exposure>/
    grism_<dirname>_detSCA01_l2.asdf
    ...
    grism_<dirname>_detSCA18_l2.asdf
```

## Dependencies

- Romanisim pixi env solved (`pixi run -e romanisim hydrate-romanisim` has
  populated `$CRDS_PATH`).
- `$CRDS_PATH`, `$CRDS_SERVER_URL`, `$STPSF_PATH` exported in the worker's
  shell (via `~/.bashrc`).
- The acceptance disperser run complete at `$ACCEPT/output/`.

## Argument derivation (per FITS header)

| FITS header | Romanisim arg |
|---|---|
| `WFICENRA, WFICENDEC` | `--radec` |
| `WFICENPA - 60` | `--roll` (focal-plane vs spacecraft offset) |
| `DETNUM` | `--sca` |
| `MA_TABLE` | `--ma_table_number` |
| `RNDSEED0 ^ RNDSEED1` | `--rng_seed` (XOR fold to 32-bit) |

Static: `--bandpass GRISM`, `--usecrds`, `--stpsf`, `--nobj 0`,
`--extra-counts <fits> ISIM`, `--date 2026-01-01T12:00:00.000`, `--level 2`.

Date is a single placeholder (zodi calc only); per-pointing dates can be
added later via a sidecar table if the science run needs it.

## Partitioning

Each worker globs `<ACCEPT>/output/**/grism_*_detSCA*.fits`, sorts the
list, and processes `[idx % num_workers == worker_index]`. All workers
log a manifest hash at startup; matching hashes confirm a clean partition.

For 576 files / 4 workers: each worker gets 144 files.

## Idempotency / restart

`wrap_with_romanisim.py` skips files whose `_l2.asdf` already exists, so
re-submitting after a partial completion just fills in the gaps. A worker
dying mid-run drops at most 8 files in flight.

## Hardware + cost

`mem-lg` (r6i.4xlarge: 8 cores SMT-off, 124 GB, ~$1.01/hr). Memory peaks
around 48 GB at N=8 — comfortable headroom. Production projection from
the 2026-05-08 scaling test (job 5740 + 5741, audit at
`/data/npadman/tmp/romanisim-smoke-20260508/scaling_*`):

- Throughput: ~47 jobs/h-node at N=8
- 576 files / 4 nodes / 8 threads ≈ 18 files/proc serially
- Wall: ~3 hours (5 h SLURM time-limit gives margin)
- Cost: 4 nodes × 3 h × $1.01 ≈ **~$12**

Compare alternatives:
- 1× `cpun-2xlg` N=22: 5.2 h wall, ~$20 (single node, slower).
- 4× `cpun-2xlg` N=22: 80 min wall, ~$20 (faster but pricier).

`mem-lg` wins on $/file because the 3.85× lower hourly rate beats the
per-proc speed advantage of c5n.

## Outputs

Per file: a single `*_l2.asdf` (~216 MB). Total ~120 GB under
`output_l2/`.

Per-submission audit lands in `$ACCEPT/slurm-meta/romanisim-<JobID>.env`
(git SHA, partition, mem, time, input/output dirs, timestamp). Per-file
romanisim logs land in `$ACCEPT/logs/romanisim/per-file/`.

## Sanity checks during the run

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

## Failure handling

Per-file failures are logged but don't kill the worker; the orchestrator
prints a final `ok=N skipped=N failed=N` summary and exits 1 if any
failed. Re-running is the recovery: idempotency means the next submit
retries only the missing files.

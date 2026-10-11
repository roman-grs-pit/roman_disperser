# HEALPix galaxy-SED store

A second, sky-ordered layout for a catalog's galaxy SEDs. A pointing then
reads only the few arrays its cone touches, instead of scattered rows from
every array in the catalog. It is produced from an existing catalog by an
afterburner, `scripts/healpix_repartition_seds.py`. `build_dispersed_image.py`
reads either layout; it detects the layout from the store and needs no new
config. The base catalog format (`metadata.parquet` + `seds.zarr`, units,
wavelength grid, stars) is unchanged and is specified in
[`data/catalogs/README.md`](../data/catalogs/README.md).

## Why

`build_source_catalog.py` writes galaxy SEDs as `galaxy_seds/sim_NNN`, one
sharded array per Galacticus sim. Every sim covers the whole field, so the
rows a pointing needs are spread across *every* array. For example, the 0.6°
cone at (RA, Dec) = (10, 0) of the acceptance catalog holds 343,212 galaxies
drawn from all 100 arrays, with a median of 14% of each array's rows. The
reader then issues one request per 10-row inner chunk. On the S3-backed
`/mnt/roman-science` mount each request costs ~50 ms, so reads are latency-
bound: 120 scattered rows took 6.2 s, while one whole 160 MB array took 1.7 s
(2-core r6i node, 2026-10-08).

In the 4-roll v0.15.1 run (SLURM array 10117, 4× A10G) galaxy-SED reads were
**~48% of wall time**: 2171 of 4530 s per pointing, 75–180 s per SCA and
330 s for the cold first SCA.

Regrouping the SEDs by sky position turns this into a few large sequential
reads, one per HEALPix pixel the cone touches.

## Layout

A HEALPix catalog is a **new sibling directory**; the source catalog is only
read and stays valid.

```
<catalog>_hp/
  metadata.parquet        # same rows, same order as the source + hp_uniq, hp_row
  healpix_plan.parquet    # one row per pixel: hp_uniq, order, n_rows, est_bytes, task
  seds.zarr/
    wavelengths           # copied unchanged
    star_seds             # copied unchanged
    galaxy_seds/          # group attrs: layout="healpix", ordering="nested", key="uniq", ...
      hp_<uniq>/          # one array (one shard) per HEALPix pixel
      ...
  slurm-meta/             # audit .env per submission (driver only)
```

### HEALPix convention

- **NESTED** ordering. Pixel `p` at order `k` (nside = 2^k) has children
  `4p … 4p+3` at order `k+1`. A fine index therefore maps to a coarser order
  by a bit shift, `p_k = p_fine >> 2(k_max − k)`, which is what makes
  adaptive splitting cheap.
- Pixels are keyed by the MOC **UNIQ** integer, which encodes (order, p) in
  one number, so pixels of different orders can coexist:

  $$\mathrm{uniq} = 4\,n_\mathrm{side}^2 + p = 4^{k+1} + p .$$

  The array name is `galaxy_seds/hp_<uniq>`; each array's attributes repeat
  `order`, `nside`, `ipix` and `n_sources`.

### Metadata columns

`metadata.parquet` keeps the source schema and row order exactly, including
the provenance columns `sim` and `sed_index`, and appends two columns:

| Column | Type | Meaning |
|---|---|---|
| `hp_uniq` | int64 | UNIQ key of the pixel holding this galaxy's SED; `-1` for stars |
| `hp_row` | int32 | Row of this galaxy's SED inside `galaxy_seds/hp_<hp_uniq>`; `-1` for stars |

Within a pixel, rows keep source-metadata order. Stars still use `sed_index`
into `star_seds`, exactly as before.

**The metadata and store are a pair.** A HEALPix `seds.zarr` must be read
with the `metadata.parquet` written beside it. `validate_catalog` raises if
the store says `layout="healpix"` but `hp_uniq` / `hp_row` are missing.

### Storage

- One array per pixel, stored as a **single shard** with 100-row inner
  chunks. Codec as in `build_source_catalog.py`: blosc zstd level 3, byte
  shuffle, float32.
- The shard is zero-padded up to a multiple of 100 rows. `n_sources` holds
  the real count, and `hp_row < n_sources` always.
- Each pixel array gets `complete=True` written last. `fill` reruns skip
  complete pixels, and `verify` requires every pixel to have it.
- **File count** is ~2 per pixel (one `zarr.json` and one shard), e.g. 361
  files for the whole acceptance store.

### Duplicated SEDs

One SED is stored **per metadata row**. If the metadata references a SED
more than once, its bytes are duplicated. This happens in the RA-padded
acceptance catalog, where each template appears at two RA shifts 2° apart:
2,356,744 galaxies against 1,178,372 unique (sim, sed_index). The cost there
is ~17 GB more on S3 (34.1 GB in total). In exchange the store is
self-contained and the reader has a single code path; copies 2° apart would
almost never share a pixel anyway.

## Pixel sizes and adaptive splitting

`plan` places each galaxy at order `--max-order` (default 12, nside 4096,
~0.86′ pixels). Starting from `--min-order` (default 0), any pixel whose
estimated compressed size exceeds `--target-gb` (default **0.5 GB**) is
split into its 4 children, recursively, down to `--max-order`. The size
estimate is `n_rows × bytes_per_row`, where `bytes_per_row` is the source's
total galaxy-shard bytes divided by its stored rows (from `stat` alone).
`--nside N` instead forces one fixed order with no splitting.

The 0.5 GB target is a *reasoned, untuned* default. At ~100 MB/s single
stream a pixel reads in a few seconds, and a g5 node's page cache can hold
tens of pixels for re-reads by later SCAs of the same pointing.

For the acceptance catalog this gave **177 pixels at orders 6, 7 and 8**.
Order 8 (~0.05 deg²) is the interior working size; the order-6 pixels are
sparse edge pixels cut by the field boundary. Measured compressed size per
pixel is 0.008 / 0.218 / 0.490 GB (min / median / max).

## Reader changes (`scripts/build_dispersed_image.py`)

- **Detection.** `galaxy_sed_layout(store)` reads the `galaxy_seds` group
  attribute `layout`. If it is absent the layout is `"sim"`, so every existing
  catalog reads exactly as before. An unknown value raises.
- **Dispatch table** `_GALAXY_SED_LAYOUTS`, giving for each layout the
  metadata column that names the array, the array-key format, and the
  metadata column giving the row inside it:

  | layout | array chosen by | array key | row column |
  |---|---|---|---|
  | `sim` | `sim` | `galaxy_seds/sim_{:03d}` | `sed_index` |
  | `healpix` | `hp_uniq` | `galaxy_seds/hp_{}` | `hp_row` |

- **`load_galaxy_seds`** groups the selected galaxies by array as before.
  - For `healpix` it reads each needed pixel **whole** (`arr[:]`) and then
    selects `hp_row` in memory. One sequential read beats per-chunk requests
    on S3.
  - The `sim` path still reads only the needed rows.
  - There is no in-process cache: re-reads by later SCAs of the same pointing
    hit the OS page cache, measured at 0.09 s raw and 0.52 s through zarr.
  - The SED corruption scrubber still reports bad galaxies by
    `(sim, sed_index)`, the template identity both layouts carry.
- **`validate_catalog`** checks that every referenced array exists with the
  right wavelength length, in either layout, and for `healpix` that the
  metadata has `hp_uniq` / `hp_row`.

No config key changes: point `catalog_dir` (or `--catalog-dir`) at the HEALPix
catalog directory.

### Looking up one SED by hand

```python
import numpy as np, pyarrow.parquet as pq, zarr

cat = "/mnt/roman-science/grs/acceptance-testing-20260430/catalogs_padded_hp"
meta = pq.read_table(f"{cat}/metadata.parquet").to_pandas()
store = zarr.open(f"{cat}/seds.zarr", mode="r")

row = meta[meta["type"] == "SER"].iloc[0]
arr = store[f"galaxy_seds/hp_{row['hp_uniq']}"]
sed = np.asarray(arr[row["hp_row"]]) * row["flux_scale"]   # FLAM, as for sim layout
```

## Building a HEALPix store

There are three steps; normally the SLURM driver runs all of them:

```bash
A=/mnt/roman-science/grs/acceptance-testing-20260430
SLURM_PARTITION=mem-med SLURM_MEM=58G \
  scripts/slurm_healpix_repartition.sh $A/catalogs_padded $A/catalogs_padded_hp \
  --task-mem-gb 35
```

1. **`plan`** runs inline on the head node, in seconds to minutes. It writes
   `metadata.parquet`, `healpix_plan.parquet` and the store skeleton.
   - It also copies `wavelengths` and `star_seds`, and records provenance in
     the `galaxy_seds` attrs: source paths, code commit, the size estimates.
   - It refuses to overwrite an existing plan, so rerunning the driver
     reuses it.
   - `--dry-run` prints the plan and writes nothing.
2. **`fill`** is a SLURM array with one task per planned task. Each task
   streams every source `sim_NNN` array whole (the fast read path), scatters
   its rows into an in-memory buffer holding that task's pixels, then writes
   each pixel shard **once**.
   - At exit it logs its peak memory (`VmHWM`), because `sacct` is
     unavailable on this cluster.
3. **`verify`** is a dependent job. It checks that every planned pixel is
   complete with the planned row count and that no unplanned arrays exist.
   - It then compares 10,000 random rows **bitwise** against the source
     (seed 20261008).
   - On success it records the measured sizes and a `verified_utc` timestamp
     in the attrs.

The driver refuses to run if the script has uncommitted changes, so the
commit it records is the code that ran. It writes
`<output>/slurm-meta/healpix-<JobID>.env`, and its environment overrides are
`SLURM_PARTITION`, `SLURM_MEM`, `SLURM_TIME` and `SLURM_LOG_DIR`.

### Memory model

Every source array spans the whole field, so every pixel needs rows from
every source array. `fill` therefore holds its pixels whole and streams the
*entire* source past them. `plan` packs pixels, in plan order, into tasks of
at most `--task-mem-gb` of **raw float32** each (a single oversized pixel
gets its own task). So:

$$
N_\mathrm{tasks} \approx \frac{\text{raw output}}{\texttt{--task-mem-gb}},
\qquad
\text{source bytes read} \approx N_\mathrm{tasks} \times \text{source size}.
$$

Memory per task trades directly against how many times the source is
re-read; pick the largest budget the node allows. Measured overhead above
the buffer is **~1.5–2.4 GB**: one source array plus temporaries
(acceptance, tasks of 34.7 and 21.9 GB peaked at 37.1 and 23.4 GB). So
`--mem ≈ budget + 5 GB` is safe there. Node limits as advertised to SLURM:
`mem-med` 62,259 MB, `mem-lg` 124,518 MB, `cpun-2xlg` 186,777 MB.

A stream-once alternative was considered and set aside on 2026-10-09: read
each source array once and append rows to per-pixel scratch files, then
convert each to a shard. A zarr shard itself cannot be appended to (zarr 3
read-modify-writes the whole shard), so this needs scratch the size of the
compressed output plus a sequential append pass. It only pays off for
catalogs that would need many tasks even on the largest nodes. XMM
(202 GB compressed, ~350 GB raw *estimated*) should fit in ~3 tasks on
`cpun-2xlg`.

## Measured results (acceptance catalog)

**Build.** This used fill array 11823 and verify 11824 at commit `5dca81b`,
on `mem-med` with `--task-mem-gb 35`.

- There were 2 fill tasks of ~7 min each. Each read the 17.04 GB source in
  168–183 s, which is **~95 MB/s per task**.
- `verify` PASSED: 177/177 pixels complete and 0 mismatches in 10,000 rows.
  The store totals 34.1 GB.
- The build cost well under \$1.

**Rendering.** The test pointing was 4-roll VISIT 1 (RA 10, Dec 0, PA 0),
18 SCAs, run on A10G as jobs 12130 and 12131 (2026-10-11). It is documented
in `workbench/20261011-sed-store-ab/`, with outputs in
`/mnt/roman-science/grs/20261011_sed-store-ab/`.

| Store | Galaxy-SED read, sum over 18 SCAs | Wall time per pointing |
|---|---|---|
| sim (10117 reference, 2026-10-07) | 2171 s | 76 min |
| sim (rerun, same day as healpix) | 2493 s | 70 min |
| **healpix** | **44 s** (1–5 s/SCA) | **29 min** (~93 s/SCA) |

- **Images.** The noiseless `MODEL` extension passes the GPU golden
  tolerance (`rtol=1e-5`, `atol=1e-5·max|ref|`) on all 18 SCAs, with relative
  sum differences ≤ 1.5e-9. That is the GPU scatter-add nondeterminism floor,
  the same as sim-vs-sim reruns.
- **`ISIM`** differs in 11–64 Poisson draws per SCA, the same as sim-vs-sim
  reruns (see `pipeline.make_sca_keys`).
- **Where the time goes now.** With SED reads gone, the remaining time is
  galaxy dispersion (927 s, 54%) and JIT compilation (601 s, ~33 s/SCA, 35%).

## Caveats

- Read timings for the diagnosis came from a 2-core r6i node; the A/B
  result above is the end-to-end check on A10G.
- The 0.5 GB pixel target has not been tuned. Smaller pixels mean more files
  but less wasted read per cone.
- A HEALPix store is derived data. If the source catalog changes, rebuild it.
  The provenance is in the `galaxy_seds` attrs (`source`,
  `source_metadata`, `code_commit`).
- Only `build_dispersed_image.py` reads the new layout. Other readers of
  `galaxy_seds/sim_NNN` must use the source catalog, or follow
  `hp_uniq` / `hp_row` as above. In this repo that means
  `scripts/verify_source_catalog.py`, which is sim-only (use the
  afterburner's `verify` for HEALPix stores), plus notebooks and ad-hoc
  scripts.

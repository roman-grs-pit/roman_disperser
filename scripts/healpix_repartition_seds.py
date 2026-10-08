#!/usr/bin/env python
"""
Repartition a catalog's galaxy SEDs by sky position (HEALPix), so that a
pointing's SED read cost scales with its cone instead of the whole catalog.

Why
---
Catalogs from ``build_source_catalog.py`` store galaxy SEDs as
``galaxy_seds/sim_NNN``: one array per Galacticus sim. Every sim covers the
whole field, so a ~0.6 deg cone needs ~7-14% of the rows of *every* array.
The reader then issues one ~50 ms request per 10-row inner chunk on the
S3-backed mount. In the 4-roll array 10117 that was ~48% of per-pointing wall
time. (Diagnosis: RobotResearchLog, roman-disperser, 2026-10-08,
thread ``catalog-store-layout``.)

This script writes a **new sibling catalog**. The source is only read. Its
galaxy SEDs are regrouped into one array per HEALPix pixel,
``galaxy_seds/hp_<uniq>``, so the reader can fetch the few whole pixels a
cone touches.

Layout written
--------------
``<output>/metadata.parquet``
    Same rows and the same order as the source metadata, with the source
    schema and the ``sim`` / ``sed_index`` provenance columns kept. Two
    columns are appended:

    - ``hp_uniq`` (int64): the MOC UNIQ key of the row's pixel, -1 for stars.
    - ``hp_row`` (int32): the row's index inside that pixel array, -1 for
      stars.

    Rows within a pixel keep source metadata order.
``<output>/seds.zarr/``
    ``wavelengths`` and ``star_seds`` are copied unchanged.
    ``galaxy_seds/hp_<uniq>`` holds one array (one shard) per pixel.
    The ``galaxy_seds`` group attributes are ``layout="healpix"``,
    ``ordering="nested"``, ``key="uniq"`` plus provenance; the reader
    dispatches on ``layout``.
``<output>/healpix_plan.parquet``
    One row per pixel: ``hp_uniq``, ``order``, ``n_rows``, ``est_bytes``
    (estimated compressed size) and ``task`` (fill-task index).

One SED is stored per metadata row. If the metadata references a SED twice,
as in the RA-padded acceptance catalog where each template appears at two
RA shifts, the bytes are duplicated. This keeps the store self-contained and
the reader single-path. The copies are 2 deg apart, so they would almost
never share a pixel anyway.

HEALPix conventions
-------------------
NESTED ordering. A pixel at order k (nside = 2**k) with index p has children
4p..4p+3 at order k+1, so a pixel's index at a coarser order is a bit shift
of its fine index: ``p_k = p_fine >> 2*(max_order - k)``. The UNIQ key packs
(order, p) into one integer:

    uniq = 4 * nside**2 + p = 4**(k+1) + p

Pixel choice (density-driven)
-----------------------------
Start at ``--min-order``. Any pixel whose estimated compressed size,
``n_rows * bytes_per_row``, exceeds ``--target-gb`` is split into its 4
children, recursively, down to ``--max-order``. ``bytes_per_row`` is the
source's total galaxy-shard bytes divided by its total stored rows, from
``stat`` only. ``--nside`` forces a single fixed order instead.

The 0.5 GB default target is reasoned, not tuned. At ~100 MB/s
single-stream, a pixel reads in a few seconds, and ~15 GB of page cache on
a g5 holds ~30 pixels for cross-SCA re-reads. The g5 process footprint is
unmeasured.

Usage (three steps)
-------------------
::

    # 1. plan: metadata + pixel/task plan + store skeleton (seconds-minutes)
    pixi run python scripts/healpix_repartition_seds.py plan \\
        --input  /mnt/roman-science/grs/acceptance-testing-20260430/catalogs_padded \\
        --output /mnt/roman-science/grs/acceptance-testing-20260430/catalogs_padded_hp

    # 2. fill: one call per task (SLURM array on mem-lg). Each task streams
    #    every source shard whole and writes the pixels it owns.
    pixi run python scripts/healpix_repartition_seds.py fill --output ... --task K

    # 3. verify: per-pixel counts + exact comparison of random rows vs source
    pixi run python scripts/healpix_repartition_seds.py verify --output ...

``fill`` skips pixels that already carry ``complete=True``, so reruns are
cheap.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BloscCodec

# Same codec as build_source_catalog.py.
COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

# Rows per inner chunk in each pixel shard. The reader reads whole pixels,
# so this only sets the decompression granularity. 100 rows x 6001-6751 bins
# x 4 B is ~2.4-2.7 MB raw per chunk.
INNER_CHUNK_ROWS = 100

PLAN_FILE = "healpix_plan.parquet"


# ---------------------------------------------------------------------------
# Pure functions (unit-tested in tests/test_healpix_repartition.py)
# ---------------------------------------------------------------------------

def uniq_from_nested(order, ipix):
    """MOC UNIQ key ``4**(order+1) + ipix`` for a NESTED pixel at ``order``."""
    return (np.int64(4) ** (np.asarray(order, dtype=np.int64) + 1)
            + np.asarray(ipix, dtype=np.int64))


def nested_from_uniq(uniq):
    """Inverse of :func:`uniq_from_nested`: return ``(order, ipix)``.

    order = floor(log4(uniq)) - 1. Integer bit-length arithmetic avoids
    float log round-off: for uniq in [4**(k+1), 4**(k+2)) the bit length is
    2k+3 or 2k+4.
    """
    uniq = np.asarray(uniq, dtype=np.int64)
    nbits = np.array([int(u).bit_length() for u in np.atleast_1d(uniq)])
    order = (nbits - 1) // 2 - 1
    ipix = np.atleast_1d(uniq) - np.int64(4) ** (order + 1)
    if uniq.ndim == 0:
        return int(order[0]), int(ipix[0])
    return order, ipix


def choose_pixels(ipix_fine, max_order, bytes_per_row, target_bytes,
                  min_order=0, fixed_order=None):
    """Assign each row to a HEALPix pixel, splitting dense pixels.

    Parameters
    ----------
    ipix_fine : ndarray int64 [N]
        NESTED pixel index of each row at ``max_order``.
    max_order : int
        Order of ``ipix_fine``; the finest order a pixel may be split to.
    bytes_per_row : float
        Estimated compressed bytes per row.
    target_bytes : float
        Split any pixel whose estimated size ``n_rows * bytes_per_row``
        exceeds this.
    min_order : int
        Coarsest order considered.
    fixed_order : int or None
        If given, put every row at this single order (no splitting).

    Returns
    -------
    uniq : ndarray int64 [N]
        UNIQ key of each row's pixel.
    n_oversize : int
        Number of pixels still over target at ``max_order``. They cannot be
        split further; the caller should warn.
    """
    ipix_fine = np.asarray(ipix_fine, dtype=np.int64)
    if fixed_order is not None:
        if not 0 <= fixed_order <= max_order:
            raise ValueError(f"fixed_order {fixed_order} not in [0, {max_order}]")
        p = ipix_fine >> (2 * (max_order - fixed_order))
        return uniq_from_nested(fixed_order, p), 0

    n = len(ipix_fine)
    uniq = np.full(n, -1, dtype=np.int64)
    pending = np.arange(n)
    n_oversize = 0
    for order in range(min_order, max_order + 1):
        if len(pending) == 0:
            break
        p = ipix_fine[pending] >> (2 * (max_order - order))
        _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
        too_big = counts * bytes_per_row > target_bytes
        if order == max_order:
            n_oversize = int(too_big.sum())
            too_big[:] = False
        keep = ~too_big[inv]
        uniq[pending[keep]] = uniq_from_nested(order, p[keep])
        pending = pending[~keep]
    assert (uniq >= 0).all()
    return uniq, n_oversize


def rows_within_group(keys):
    """For each element, its rank among equal keys in original order.

    ``rows_within_group([7, 3, 7, 7, 3]) -> [0, 0, 1, 2, 1]``.
    """
    keys = np.asarray(keys)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(keys)]))
    rank = np.empty(len(keys), dtype=np.int64)
    rank[order] = np.arange(len(keys)) - group_start
    return rank


def assign_tasks(raw_bytes, budget_bytes):
    """Pack pixels, in the given order, into consecutive tasks.

    A task closes when adding the next pixel would exceed ``budget_bytes``
    (raw, in-memory). A single pixel larger than the budget gets its own
    task.
    """
    task = np.empty(len(raw_bytes), dtype=np.int32)
    t, used = 0, 0
    for i, b in enumerate(raw_bytes):
        if used > 0 and used + b > budget_bytes:
            t, used = t + 1, 0
        task[i] = t
        used += b
    return task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit():
    """Code commit of this script, with ``-dirty`` if the tree is modified."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            text=True).strip()
        dirty = subprocess.call(
            ["git", "-C", str(here), "diff", "--quiet", "HEAD", "--",
             str(Path(__file__).resolve())])
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dir_bytes(path):
    """Total size of regular files under ``path`` (S3 mounts: use stat)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.stat(os.path.join(root, f)).st_size
    return total


def _sim_key(sim):
    return f"galaxy_seds/sim_{int(sim):03d}"


def _hp_key(uniq):
    return f"galaxy_seds/hp_{int(uniq)}"


def _source_zarr(input_dir):
    """Resolved source ``seds.zarr`` path (follows pad_catalog's symlink)."""
    return (Path(input_dir) / "seds.zarr").resolve()


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def cmd_plan(args):
    from astropy_healpix import HEALPix
    import astropy.units as u

    src_dir = Path(args.input)
    out_dir = Path(args.output)
    if (out_dir / "metadata.parquet").exists() and not args.overwrite:
        sys.exit(f"{out_dir}/metadata.parquet exists; pass --overwrite to redo the plan")

    src_zarr_path = _source_zarr(src_dir)
    src = zarr.open(str(src_zarr_path), mode="r")
    src_gal = src["galaxy_seds"]
    if src_gal.attrs.get("layout", "sim") != "sim":
        sys.exit(f"source galaxy_seds layout is {src_gal.attrs.get('layout')!r}, expected sim")
    n_wl = src["wavelengths"].shape[0]

    table = pq.read_table(src_dir / "metadata.parquet")
    meta = table.select(["ra", "dec", "type", "sim", "sed_index"]).to_pandas()
    is_gal = (meta["type"] == "SER").values
    n_gal = int(is_gal.sum())
    print(f"Source: {src_dir}  ({len(meta)} rows, {n_gal} galaxies)")
    print(f"  seds.zarr -> {src_zarr_path}")

    # bytes per row: compressed galaxy-shard bytes / stored rows (all sims).
    sim_keys = sorted(k for k in src_gal.array_keys())
    stored_rows = sum(src_gal[k].shape[0] for k in sim_keys)
    shard_bytes = _dir_bytes(src_zarr_path / "galaxy_seds")
    bytes_per_row = shard_bytes / stored_rows
    print(f"  {len(sim_keys)} sim arrays, {stored_rows} stored rows, "
          f"{shard_bytes / 1e9:.2f} GB -> {bytes_per_row / 1e3:.2f} kB/row compressed")

    # Fine NESTED index for galaxies.
    hp = HEALPix(nside=2 ** args.max_order, order="nested")
    ra = meta.loc[is_gal, "ra"].values
    dec = meta.loc[is_gal, "dec"].values
    ipix_fine = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg).astype(np.int64)

    fixed_order = None
    if args.nside is not None:
        fixed_order = int(np.log2(args.nside))
        if 2 ** fixed_order != args.nside:
            sys.exit(f"--nside {args.nside} is not a power of 2")
    target_bytes = args.target_gb * 1e9
    gal_uniq, n_oversize = choose_pixels(
        ipix_fine, args.max_order, bytes_per_row, target_bytes,
        min_order=args.min_order, fixed_order=fixed_order)
    if n_oversize:
        print(f"  WARNING: {n_oversize} pixel(s) exceed the target at max_order "
              f"{args.max_order}; raise --max-order to split them")
    gal_row = rows_within_group(gal_uniq)

    hp_uniq = np.full(len(meta), -1, dtype=np.int64)
    hp_row = np.full(len(meta), -1, dtype=np.int32)
    hp_uniq[is_gal] = gal_uniq
    hp_row[is_gal] = gal_row

    # Per-pixel plan.
    pix, n_rows = np.unique(gal_uniq, return_counts=True)
    order, _ = nested_from_uniq(pix)
    est_bytes = n_rows * bytes_per_row
    raw_bytes = n_rows.astype(np.int64) * n_wl * 4
    task = assign_tasks(raw_bytes, args.task_mem_gb * 1e9)
    plan = pa.table({
        "hp_uniq": pix, "order": order.astype(np.int16), "n_rows": n_rows,
        "est_bytes": est_bytes, "task": task,
    })

    print(f"Pixels: {len(pix)}  orders {sorted(set(order.tolist()))}")
    print(f"  rows/pixel min/median/max: {n_rows.min()} / "
          f"{int(np.median(n_rows))} / {n_rows.max()}")
    print(f"  est GB/pixel min/median/max: {est_bytes.min() / 1e9:.3f} / "
          f"{np.median(est_bytes) / 1e9:.3f} / {est_bytes.max() / 1e9:.3f}")
    print(f"  est total {est_bytes.sum() / 1e9:.1f} GB compressed, "
          f"{raw_bytes.sum() / 1e9:.1f} GB raw")
    print(f"Fill tasks: {task.max() + 1} (budget {args.task_mem_gb} GB raw each)")

    if args.dry_run:
        return

    # Write metadata (source schema + 2 appended columns), plan, store skeleton.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_table = (table.append_column("hp_uniq", pa.array(hp_uniq))
                      .append_column("hp_row", pa.array(hp_row)))
    pq.write_table(out_table, out_dir / "metadata.parquet")
    pq.write_table(plan, out_dir / PLAN_FILE)

    store = zarr.open_group(str(out_dir / "seds.zarr"), mode="a")
    store.attrs.update({**dict(src.attrs),
                        "derived_from": str(src_zarr_path),
                        "derived_by": "scripts/healpix_repartition_seds.py"})
    for name in ("wavelengths", "star_seds"):
        if name in store:
            continue
        a = src[name]
        store.create_array(name, data=np.asarray(a[:]),
                           attributes=dict(a.attrs), compressors=COMPRESSOR)
    gal = store.require_group("galaxy_seds")
    gal.attrs.update({
        "layout": "healpix",
        "ordering": "nested",
        "key": "uniq",
        "array_name": "hp_<uniq>",
        "metadata_columns": ["hp_uniq", "hp_row"],
        "n_pixels": int(len(pix)),
        "orders": sorted(set(order.tolist())),
        "max_order": int(args.max_order),
        "min_order": int(args.min_order),
        "fixed_order": fixed_order,
        "target_bytes": float(target_bytes),
        "est_bytes_per_row": float(bytes_per_row),
        "est_pixel_bytes_min_median_max": [float(est_bytes.min()),
                                           float(np.median(est_bytes)),
                                           float(est_bytes.max())],
        "source": str(src_zarr_path),
        "source_metadata": str((src_dir / "metadata.parquet").resolve()),
        "code_commit": _git_commit(),
        "planned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"Wrote {out_dir}/metadata.parquet, {PLAN_FILE}, seds.zarr skeleton")


# ---------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------

def _write_pixel(gal_group, uniq, seds):
    """Write one pixel as a single shard, zero-padded to the inner chunk."""
    n, n_wl = seds.shape
    inner = min(INNER_CHUNK_ROWS, n)
    shard_rows = -(-n // inner) * inner
    if shard_rows > n:
        seds = np.concatenate([seds, np.zeros((shard_rows - n, n_wl), seds.dtype)])
    order, ipix = nested_from_uniq(uniq)
    arr = gal_group.create_array(
        f"hp_{int(uniq)}", data=seds, chunks=(inner, n_wl),
        shards=(shard_rows, n_wl), compressors=COMPRESSOR, overwrite=True,
        attributes={
            "units": "FLAM (erg/s/cm^2/Å, apparent)",
            "axes": ["hp_row", "wavelength"],
            "frame": "observed",
            "n_sources": int(n),  # actual count (before padding)
            "order": int(order), "nside": int(2 ** order), "ipix": int(ipix),
        })
    arr.attrs["complete"] = True  # set last: marks the shard as fully written


def cmd_fill(args):
    out_dir = Path(args.output)
    plan = pq.read_table(out_dir / PLAN_FILE).to_pandas()
    store = zarr.open_group(str(out_dir / "seds.zarr"), mode="a")
    gal = store["galaxy_seds"]
    src = zarr.open(gal.attrs["source"], mode="r")
    n_wl = store["wavelengths"].shape[0]

    mine = plan[plan["task"] == args.task]
    if mine.empty:
        sys.exit(f"task {args.task}: no pixels (plan has {plan['task'].max() + 1} tasks)")
    todo = [int(u) for u in mine["hp_uniq"]
            if not (f"hp_{u}" in gal and gal[f"hp_{u}"].attrs.get("complete"))]
    print(f"task {args.task}: {len(mine)} pixels, {len(todo)} to write")
    if not todo:
        return

    meta = pq.read_table(out_dir / "metadata.parquet",
                         columns=["sim", "sed_index", "hp_uniq", "hp_row"]).to_pandas()
    meta = meta[meta["hp_uniq"].isin(todo)]

    # One contiguous buffer for all of this task's pixels; pixel u occupies
    # rows offset[u] : offset[u] + n[u], and a galaxy lands at offset + hp_row.
    n_rows = mine.set_index("hp_uniq").loc[todo, "n_rows"].values
    offsets = dict(zip(todo, np.r_[0, np.cumsum(n_rows)[:-1]]))
    buf = np.empty((int(n_rows.sum()), n_wl), dtype=np.float32)
    print(f"  buffer {buf.nbytes / 1e9:.1f} GB raw")
    filled = np.zeros(len(buf), dtype=bool)
    dest = meta["hp_uniq"].map(offsets).values + meta["hp_row"].values

    t0 = time.time()
    sims = np.unique(meta["sim"].values)
    for i, sim in enumerate(sims):
        sel = (meta["sim"].values == sim)
        shard = np.asarray(src[_sim_key(sim)][:])  # whole-shard read: fast path
        buf[dest[sel]] = shard[meta["sed_index"].values[sel]]
        filled[dest[sel]] = True
        if (i + 1) % 10 == 0 or i + 1 == len(sims):
            print(f"  read {i + 1}/{len(sims)} sim shards  ({time.time() - t0:.0f}s)")
    if not filled.all():
        sys.exit(f"task {args.task}: {(~filled).sum()} buffer rows never filled")

    t0 = time.time()
    for j, u in enumerate(todo):
        o = offsets[u]
        _write_pixel(gal, u, buf[o:o + n_rows[j]])
    print(f"  wrote {len(todo)} pixels ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args):
    out_dir = Path(args.output)
    plan = pq.read_table(out_dir / PLAN_FILE).to_pandas()
    store = zarr.open_group(str(out_dir / "seds.zarr"), mode="r")
    gal = store["galaxy_seds"]
    src = zarr.open(gal.attrs["source"], mode="r")
    meta = pq.read_table(out_dir / "metadata.parquet",
                         columns=["type", "sim", "sed_index", "hp_uniq", "hp_row"]
                         ).to_pandas()
    ok = True

    # 1. Every planned pixel complete, with row count = metadata count.
    counts = meta.loc[meta["type"] == "SER", "hp_uniq"].value_counts()
    bad = []
    for u, n in zip(plan["hp_uniq"], plan["n_rows"]):
        key = f"hp_{u}"
        if key not in gal or not gal[key].attrs.get("complete"):
            bad.append((u, "missing/incomplete"))
        elif gal[key].attrs["n_sources"] != n or counts.get(u, 0) != n:
            bad.append((u, f"n_sources {gal[key].attrs['n_sources']} plan {n} "
                           f"meta {counts.get(u, 0)}"))
    extra = set(gal.array_keys()) - {f"hp_{u}" for u in plan["hp_uniq"]}
    print(f"[counts] {len(plan)} pixels, {len(bad)} bad, {len(extra)} unplanned arrays")
    for b in bad[:10]:
        print(f"   {b}")
    ok &= not bad and not extra
    if bad:
        print("verify: FAIL (pixels incomplete; skipping row comparison)")
        sys.exit(1)

    # 2. Exact comparison of random galaxy rows against the source.
    rng = np.random.default_rng(args.seed)
    gidx = np.flatnonzero((meta["type"] == "SER").values)
    sample = meta.iloc[np.sort(rng.choice(gidx, min(args.n_sample, len(gidx)),
                                          replace=False))]
    n_mismatch = 0
    t0 = time.time()
    for u, grp in sample.groupby("hp_uniq"):
        new = np.asarray(gal[f"hp_{u}"].get_orthogonal_selection(
            (grp["hp_row"].values, slice(None))))
        for sim, g2 in grp.groupby("sim"):
            old = np.asarray(src[_sim_key(sim)].get_orthogonal_selection(
                (g2["sed_index"].values, slice(None))))
            pos = grp.index.get_indexer(g2.index)
            # Bitwise equality (NaN-safe): this is a pure copy.
            n_mismatch += int((~np.all(new[pos].view(np.uint32)
                                       == old.view(np.uint32), axis=1)).sum())
    print(f"[rows] {len(sample)} random rows (seed {args.seed}) compared bitwise: "
          f"{n_mismatch} mismatches  ({time.time() - t0:.0f}s)")
    ok &= n_mismatch == 0

    # 3. Measured sizes (informational; also recorded in attrs).
    sizes = np.array([_dir_bytes(Path(out_dir) / "seds.zarr" / "galaxy_seds" / f"hp_{u}")
                      for u in plan["hp_uniq"]])
    print(f"[sizes] GB/pixel min/median/max: {sizes.min() / 1e9:.3f} / "
          f"{np.median(sizes) / 1e9:.3f} / {sizes.max() / 1e9:.3f}; "
          f"total {sizes.sum() / 1e9:.1f} GB")
    if ok and not args.no_record:
        g = zarr.open_group(str(out_dir / "seds.zarr"), mode="a")["galaxy_seds"]
        g.attrs.update({
            "measured_pixel_bytes_min_median_max": [int(sizes.min()),
                                                    int(np.median(sizes)),
                                                    int(sizes.max())],
            "measured_total_bytes": int(sizes.sum()),
            "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verified_n_sample": int(len(sample)), "verified_seed": int(args.seed),
        })
    print("verify: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="write metadata, pixel plan, store skeleton")
    pp.add_argument("--input", required=True, help="source catalog dir")
    pp.add_argument("--output", required=True, help="new catalog dir (sibling)")
    pp.add_argument("--target-gb", type=float, default=0.5,
                    help="max estimated compressed GB per pixel (default 0.5)")
    pp.add_argument("--nside", type=int, default=None,
                    help="force a single fixed nside (power of 2); no splitting")
    pp.add_argument("--min-order", type=int, default=0)
    pp.add_argument("--max-order", type=int, default=12,
                    help="finest order (nside 4096, ~0.86 arcmin pixels)")
    pp.add_argument("--task-mem-gb", type=float, default=40.0,
                    help="raw float32 GB of pixels per fill task (mem-lg has 124 GB)")
    pp.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    pp.add_argument("--overwrite", action="store_true")
    pp.set_defaults(func=cmd_plan)

    pf = sub.add_parser("fill", help="write the pixels of one task")
    pf.add_argument("--output", required=True)
    pf.add_argument("--task", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)),
                    help="task index (default: $SLURM_ARRAY_TASK_ID)")
    pf.set_defaults(func=cmd_fill)

    pv = sub.add_parser("verify", help="check counts and compare rows vs source")
    pv.add_argument("--output", required=True)
    pv.add_argument("--n-sample", type=int, default=10_000)
    pv.add_argument("--seed", type=int, default=20261008)
    pv.add_argument("--no-record", action="store_true",
                    help="do not write measured sizes into galaxy_seds attrs")
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args()
    if args.cmd == "fill" and args.task < 0:
        p.error("fill needs --task or $SLURM_ARRAY_TASK_ID")
    args.func(args)


if __name__ == "__main__":
    main()

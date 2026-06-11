#!/usr/bin/env python3
"""Combine partial SED catalogs (disk or spheroid) into one complete catalog.

Takes 3 batch directories (sim_001–033, 034–066, 067–100) and produces
one complete catalog with all 100 sims:
  - metadata.parquet with global sed_index
  - seds.zarr with a single combined galaxy_seds array

Usage:
    python combine_catalogs.py --input-dir /Volumes/SSD/batches \
                               --output-dir /path/to/catalog_disk \
                               --type disk

    python combine_catalogs.py --input-dir /Volumes/SSD/batches \
                               --output-dir /path/to/catalog_spheroid \
                               --type spheroid
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BloscCodec

COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
INNER_CHUNK = 10  # matches original inner chunk size


def parse_args():
    p = argparse.ArgumentParser(description="Combine partial SED catalogs into one complete catalog")
    p.add_argument("--input-dir", required=True, help="Root dir with sed_calc_batch_*_disk/ or spheroid/")
    p.add_argument("--output-dir", required=True, help="Output directory for the combined catalog")
    p.add_argument("--type", required=True, choices=["disk", "spheroid"])
    p.add_argument("--n-batches", type=int, default=3, help="Number of batch dirs (default 3)")
    return p.parse_args()


# ── discovery ────────────────────────────────────────────────────────────────

def get_batch_dirs(input_dir: Path, n_batches: int, catalog_type: str):
    batches = []
    for i in range(1, n_batches + 1):
        d = input_dir / f"sed_calc_batch_{i}_{catalog_type}"
        if not d.exists():
            raise FileNotFoundError(f"Batch directory not found: {d}")
        batches.append(d)
    return batches


def get_sims_with_batch_info(batch_dirs: list[Path]) -> list[tuple[str, Path, int]]:
    """Return [(sim_name, sim_dir, batch_idx)] sorted by sim number."""
    out: list[tuple[str, Path, int]] = []
    for batch_idx, batch_dir in enumerate(batch_dirs):
        galaxy_dir = batch_dir / "seds.zarr" / "galaxy_seds"
        for d in sorted(galaxy_dir.iterdir()):
            if d.is_dir() and d.name != "zarr.json":
                out.append((d.name, d, batch_idx))
    return out


def get_source_info(sim_dir: Path) -> tuple[int, int]:
    """Return (actual_array_rows, n_sources_from_attr).

    The actual zarr array may have one extra row of zero-padding beyond
    n_sources.  We always use the *actual* array size so the combined
    buffer is large enough.
    """
    with open(sim_dir / "zarr.json") as f:
        data = json.load(f)
    n_sources = data.get("attributes", {}).get("n_sources", None)
    arr_n = data.get("shape", [0, 0])[0]
    return (arr_n, n_sources)


def compute_offsets(sims: list[tuple[str, Path, int]]) -> dict[str, tuple[int, int, int]]:
    """Return {sim_name: (global_n_sources_offset, arr_rows, n_sources)}"""
    offsets: dict[str, tuple[int, int, int]] = {}
    global_off = 0
    for sim_name, sim_dir, _ in sims:
        arr_n, n_s = get_source_info(sim_dir)
        offsets[sim_name] = (global_off, arr_n, n_s)
        global_off += n_s if n_s is not None else arr_n
    return offsets


# ── zarr building ────────────────────────────────────────────────────────────

def build_zarr(batch_dirs: list[Path], sims: list[tuple[str, Path, int]],
               offsets: dict[str, int], output_zarr: Path) -> int:
    """Create combined seds.zarr with galaxy_seds, star_seds, wavelengths.

    Returns total galaxy count.
    """
    src_path = batch_dirs[0] / "seds.zarr"

    src = zarr.open_group(str(src_path), mode="r")
    wavelengths = src["wavelengths"][:]
    star_seds = src["star_seds"][:]

    n_wl_grid = len(wavelengths)       # 14001 (wavelength array / star templates)
    n_wl_gal = src[f"galaxy_seds/{sims[0][0]}"][:].shape[1]  # 14000 (galaxy SEDs)
    n_star = star_seds.shape[0]
    # total_gal = sum of n_sources (actual sources, matching metadata row count)
    total_gal = sum(info[2] for info in offsets.values() if info[2] is not None)
    if total_gal == 0:
        total_gal = sum(info[1] for info in offsets.values())  # fallback to arr rows
    # Pad to multiple of INNER_CHUNK for sharding compatibility
    padded_gal = ((total_gal + INNER_CHUNK - 1) // INNER_CHUNK) * INNER_CHUNK
    if padded_gal != total_gal:
        print(f"  NOTE: padding {total_gal:,} → {padded_gal:,} rows (for INNER_CHUNK={INNER_CHUNK})")

    if n_wl_gal != n_wl_grid:
        print(f"  NOTE: galaxy_seds use {n_wl_gal:,} wl pts, grid uses {n_wl_grid:,}")

    print(f"\n{'='*60}")
    print(f"Building combined Zarr store at {output_zarr}")
    print(f"{'='*60}")
    print(f"  Wavelengths : {n_wl_grid:,} points")
    print(f"  Star templates: {n_star:,}")
    print(f"  Galaxy sources: {total_gal:,} ({len(sims)} sims)")
    print(f"  Galaxy output : {total_gal:,} × {n_wl_gal:,} × 4 B = {total_gal * n_wl_gal * 4 / 1e9:.1f} GB")
    print(f"  Padded to     : {padded_gal:,} rows for sharding")

    # Pre-allocate + read
    print(f"\nReading galaxy SEDs ({len(sims)} sims) …")
    combined_gal = np.zeros((padded_gal, n_wl_gal), dtype=np.float32)

    t0 = time.time()
    for i, (sim_name, sim_dir, batch_idx) in enumerate(sims):
        batch_zarr = batch_dirs[batch_idx] / "seds.zarr"
        store = zarr.open_group(str(batch_zarr), mode="r")
        sim_seds = store[f"galaxy_seds/{sim_name}"][:]

        off, arr_n, n_s = offsets[sim_name]
        # Trim to n_sources (remove any padding rows) to match metadata
        n = n_s if n_s is not None else arr_n
        combined_gal[off : off + n] = sim_seds[:n]

        pct = (i + 1) / len(sims) * 100
        if pct % 10 == 0 or pct == 100:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sims) - i - 1)
            print(f"  {sim_name:>8s}  {pct:5.1f}%  (eta ~{eta:6.0f}s)")

    print(f"  Read & assembled {total_gal:,} sources in {time.time()-t0:.1f}s")

    # Remove pre-existing output (if any)
    if output_zarr.exists():
        print(f"\nRemoving existing store at {output_zarr} …")
        shutil.rmtree(output_zarr)
    output_zarr.mkdir(parents=True)

    # Write arrays
    print(f"\nWriting Zarr arrays …")
    t0 = time.time()

    root = zarr.open_group(str(output_zarr), mode="w")

    # wavelengths (single chunk, no sharding needed)
    root.create_array(
        "wavelengths", data=wavelengths, chunks=(n_wl_grid,),
        compressors=COMPRESSOR,
        attributes={
            "units": "Angstrom",
            "description": "Common wavelength grid for all SEDs",
            "grid_definition": f"np.linspace(7000.0, 21000.0, {n_wl_grid})",
        },
    )

    # star_seds (single chunk)
    root.create_array(
        "star_seds", data=star_seds, chunks=star_seds.shape,
        compressors=COMPRESSOR,
        attributes={
            "units": "FLAM (erg/s/cm^2/Å, normalized to 0 mag F158)",
            "axes": ["template_index", "wavelength"],
        },
    )

    # combined galaxy_seds (sharded)
    root.create_array(
        "galaxy_seds", data=combined_gal,
        chunks=(INNER_CHUNK, n_wl_gal), shards=(padded_gal, n_wl_gal),
        compressors=COMPRESSOR,
        attributes={
            "units": "FLAM (erg/s/cm^2/Å, apparent)",
            "axes": ["sed_index", "wavelength"],
            "frame": "observed",
            "n_sources": total_gal,
            "n_partitions": len(sims),
        },
    )

    elapsed = time.time() - t0
    print(f"  Written in {elapsed:.1f}s")

    return total_gal


# ── metadata building ────────────────────────────────────────────────────────

def build_metadata(sims: list[tuple[str, Path, int]],
                   offsets: dict[str, int],
                   batch_dirs: list[Path]) -> pa.Table:
    """Merge metadata.parquet across batches.

    - Stars: kept once (from first batch).
    - Galaxies: included from every sim, sed_index re-indexed globally.
    - Original schema preserved.
    """
    print(f"\nMerging metadata from {len(batch_dirs)} batches …")
    t0 = time.time()

    combined_tables: list[pa.Table] = []

    # ── stars (keep from first batch only) ────────────────────────────
    first_meta = pq.read_table(batch_dirs[0] / "metadata.parquet")
    stars = first_meta.filter(pa.compute.equal(first_meta["type"], "PSF"))
    print(f"  Stars:  {stars.num_rows:,} rows (from batch 1, deduplicated)")
    combined_tables.append(stars)

    # ── galaxies (every batch) ────────────────────────────────────────
    n_gal_total = 0
    for sim_name, sim_dir, batch_idx in sims:
        batch_meta = pq.read_table(batch_dirs[batch_idx] / "metadata.parquet")
        sim_val = int(sim_name.replace("sim_", ""))

        # Keep only SER (galaxy) rows for this sim
        type_ok = pa.compute.equal(batch_meta["type"], "SER")
        sim_ok = pa.compute.equal(batch_meta["sim"], pa.array([sim_val] * batch_meta.num_rows, type=pa.int16()))
        mask = pa.compute.and_(type_ok, sim_ok)
        gal = batch_meta.filter(mask)

        # Offset sed_index (replace in-place to preserve type)
        global_off, _, _ = offsets[sim_name]
        old_idx = gal["sed_index"].to_numpy(zero_copy_only=False).astype(np.int32)
        new_idx = (old_idx + np.int32(global_off)).astype(np.int32)
        new_col = pa.array(new_idx, type=pa.int32())
        # Replace the sed_index column directly
        col_names = gal.column_names
        new_col_names = ["sed_index" if c == "sed_index" else c for c in col_names]
        gal = gal.set_column(
            col_names.index("sed_index"),
            "sed_index",
            new_col,
        )

        combined_tables.append(gal)
        n_gal_total += gal.num_rows
        if len(combined_tables) % 10 == 0:
            print(f"  … processed {len(combined_tables)} sims, {n_gal_total:,} galaxies so far")

    combined = pa.concat_tables(combined_tables, promote_options="default")
    print(f"  Galaxies: {n_gal_total:,} rows (total metadata: {combined.num_rows:,} rows)")
    print(f"  Metadata merge complete in {time.time()-t0:.1f}s")
    return combined


# ── verification ─────────────────────────────────────────────────────────────

def verify(output_zarr: Path, metadata_path: Path, expected_gal: int, n_sims: int):
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    ok = True

    # ── metadata ────────────────────────────────────────────────────────
    tbl = pq.read_table(str(metadata_path))
    gal_mask = pa.compute.equal(tbl["type"], "SER")
    n_gal = pa.compute.sum(gal_mask.cast(pa.int32())).as_py()
    n_stars = tbl.num_rows - n_gal

    print(f"  Metadata rows    : {tbl.num_rows:,}  (galaxies={n_gal:,}, stars={n_stars:,})")
    if n_gal != expected_gal:
        print(f"  ⚠  Galaxy count mismatch: expected {expected_gal:,}, got {n_gal:,}")
        ok = False

    max_idx = pa.compute.max(tbl.filter(gal_mask)["sed_index"]).as_py()
    print(f"  Max galaxy sed_idx: {max_idx:,}  (expected {expected_gal-1:,})")
    if max_idx != expected_gal - 1:
        print(f"  ⚠  Max sed_index mismatch!")
        ok = False

    sims_in_meta = set(tbl.filter(gal_mask)["sim"].to_pylist())
    expected_sims = set(range(1, n_sims + 1))
    if sims_in_meta != expected_sims:
        print(f"  ⚠  Sim IDs in metadata: {sorted(sims_in_meta)[:5]}…{sorted(sims_in_meta)[-5:]}")
        print(f"      Expected: {sorted(expected_sims)[:5]}…{sorted(expected_sims)[-5:]}")
        ok = False

    # ── zarr store ──────────────────────────────────────────────────────
    store = zarr.open_group(str(output_zarr), mode="r")
    gal_arr = store["galaxy_seds"]
    arr_shape = gal_arr.shape
    arr_n = gal_arr.attrs.get("n_sources", None)
    arr_np = gal_arr.attrs.get("n_partitions", None)
    print(f"  galaxy_seds shape : {arr_shape}  (n_sources={arr_n}, n_partitions={arr_np})")
    if arr_n != expected_gal:
        print(f"  ⚠  galaxy_seds n_sources mismatch!")
        ok = False
    if arr_np != n_sims:
        print(f"  ⚠  galaxy_seds n_partitions mismatch!")
        ok = False

    # Spot-check a few sims against originals
    # Compute per-sim info from the output zarr attributes
    sims_info = []
    for i in range(1, n_sims + 1):
        sn = f"sim_{i:03d}"
        path = os.path.join(str(output_zarr), "galaxy_seds", sn, "zarr.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        n_s = data.get("attributes", {}).get("n_sources", data.get("shape", [0, 0])[0])
        # Determine batch index
        batch_idx = (i - 1) // 33 if n_sims > 33 else 0
        sims_info.append((sn, i, n_s, batch_idx))

    print(f"\n  Spot-checking against source batches …")
    for sim_name, sim_num, n_s, batch_idx in sims_info[:3]:  # check first 3
        if batch_idx >= len(batch_dirs_orig):
            print(f"    ⏭ {sim_name}: skip (batch {batch_idx} not in sources)")
            continue
        src = zarr.open_group(str(batch_dirs_orig[batch_idx] / "seds.zarr"), mode="r")
        orig = src[f"galaxy_seds/{sim_name}"][:]
        # Trim source to n_sources (remove padding)
        orig_trimmed = orig[:n_s]
        # Find offset from prior sims
        offset = 0
        for sn, si, ns, _ in sims_info:
            if sn == sim_name:
                break
            offset += ns
        combined_slice = gal_arr[offset:offset + n_s][:]
        match = np.allclose(orig_trimmed, combined_slice, equal_nan=True)
        status = "✅" if match else "❌"
        print(f"    {status} {sim_name}: {n_s:,} rows match={match}")
        if not match:
            ok = False

    # ── summary ─────────────────────────────────────────────────────────
    print(f"\n  {'✅ ALL CHECKS PASSED' if ok else '⚠  SOME CHECKS FAILED'}")
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

batch_dirs_orig: list  # module-level reference for verify()

def main():
    global batch_dirs_orig

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    catalog_type = args.type

    print(f"Combining {catalog_type} catalogs:")
    print(f"  Input:  {input_dir}/sed_calc_batch_*_{catalog_type}/")
    print(f"  Output: {output_dir}/")

    batch_dirs = get_batch_dirs(input_dir, args.n_batches, catalog_type)
    batch_dirs_orig = batch_dirs  # for verify()

    sims = get_sims_with_batch_info(batch_dirs)
    if len(sims) != 100:
        print(f"  WARNING: expected 100 sims, got {len(sims)}")

    # Check for duplicate sim names
    sim_names = [s[0] for s in sims]
    if len(sim_names) != len(set(sim_names)):
        print(f"  ERROR: duplicate sim names found!")
        sys.exit(1)

    offsets = compute_offsets(sims)
    total_gal = build_zarr(batch_dirs, sims, offsets, output_dir / "seds.zarr")

    combined_meta = build_metadata(sims, offsets, batch_dirs)
    pq.write_table(combined_meta, output_dir / "metadata.parquet")
    print(f"\n  metadata.parquet written ({combined_meta.num_rows:,} rows)")

    # Verify
    all_ok = verify(output_dir / "seds.zarr", output_dir / "metadata.parquet", total_gal, len(sims))

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    if not all_ok:
        print("Some checks failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

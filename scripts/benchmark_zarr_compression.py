#!/usr/bin/env python
"""Benchmark Zarr compression strategies for galaxy SED storage.

Tests:
1. Bitshuffle vs byteshuffle (blosc + zstd)
2. Inner chunk sizes (1, 10, 50, 100, 1000) × 5501 within a single shard
3. Random-gather read time for ~1000 non-consecutive indices

Usage:
    pixi run python scripts/benchmark_zarr_compression.py
"""

import time
import tempfile
import shutil
from pathlib import Path

import h5py
import numpy as np
import zarr
from zarr.codecs import BloscCodec

# --- Config ---
HDF5_DIR = Path.home() / "data/Roman/galacticus_4deg2_mock"
HDF5_FILE = HDF5_DIR / "galacticus_FOV_EVERY100_sub_1.hdf5"
SED_PATH = "Outputs/SED:observed:dust:Av1.6523"
GRISM_SLICE = slice(3500, 9001)  # 5501 wavelengths, 9000-20000 Å

N_GATHER = 1000  # number of random sources to gather
N_GATHER_REPEATS = 5  # repeat gather timing for stability


def load_sed_data():
    """Load and trim SED data from HDF5."""
    print(f"Loading SEDs from {HDF5_FILE.name}...")
    with h5py.File(HDF5_FILE, "r") as f:
        data = f[SED_PATH][:, GRISM_SLICE].astype(np.float32)
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Raw size: {data.nbytes / 1e6:.1f} MB")
    return data


def write_zarr_flat(data, path, compressor, chunk_rows):
    """Write Zarr array with given compressor and chunk size (no sharding)."""
    store = zarr.open(path, mode="w")
    store.create_array(
        "seds",
        data=data,
        chunks=(chunk_rows, data.shape[1]),
        compressors=compressor,
    )
    return store


def write_zarr_sharded(data, path, compressor, inner_chunk_rows, shard_rows=None):
    """Write Zarr array with sharding.

    In zarr-python v3: chunks = inner (small), shards = outer (big).
    """
    n_src = data.shape[0]
    n_wl = data.shape[1]
    if shard_rows is None:
        # Round up to nearest multiple of inner_chunk_rows
        shard_rows = ((n_src + inner_chunk_rows - 1) // inner_chunk_rows) * inner_chunk_rows
    store = zarr.open(path, mode="w")
    # Pad data to shard-aligned size (trailing zeros)
    if n_src < shard_rows:
        padded = np.zeros((shard_rows, n_wl), dtype=data.dtype)
        padded[:n_src] = data
    else:
        padded = data
    store.create_array(
        "seds",
        data=padded,
        chunks=(inner_chunk_rows, n_wl),
        shards=(shard_rows, n_wl),
        compressors=compressor,
    )
    return store


def measure_dir_size(path):
    """Total bytes on disk for a directory."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def measure_gather_read(store_path, indices, n_repeats=N_GATHER_REPEATS):
    """Time random-gather reads (one row at a time)."""
    store = zarr.open(store_path, mode="r")
    arr = store["seds"]

    # Warmup
    _ = np.array(arr[indices[0]])

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for idx in indices:
            _ = np.array(arr[idx])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times)


def measure_bulk_read(store_path, n_repeats=3):
    """Time reading the entire array."""
    store = zarr.open(store_path, mode="r")
    arr = store["seds"]

    # Warmup
    _ = np.array(arr[:10])

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = np.array(arr[:])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times)


def run_benchmarks():
    data = load_sed_data()
    n_sources = data.shape[0]
    raw_mb = data.nbytes / 1e6

    # Fixed random indices for gather test
    rng = np.random.default_rng(42)
    gather_indices = rng.choice(n_sources, size=N_GATHER, replace=False)
    gather_indices.sort()  # sorted but non-consecutive

    tmpdir = tempfile.mkdtemp(prefix="zarr_bench_")
    print(f"Working in {tmpdir}\n")

    results = []

    try:
        # =====================================================
        # Benchmark 1: Bitshuffle vs byteshuffle (flat chunks)
        # =====================================================
        print("=" * 70)
        print("BENCHMARK 1: Bitshuffle vs Byteshuffle (chunk=1000, no sharding)")
        print("=" * 70)

        for shuffle_name in ["shuffle", "bitshuffle"]:
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle=shuffle_name)
            path = Path(tmpdir) / f"flat_{shuffle_name}"

            t0 = time.perf_counter()
            write_zarr_flat(data, path, compressor, chunk_rows=1000)
            write_time = time.perf_counter() - t0

            disk_mb = measure_dir_size(path) / 1e6
            ratio = raw_mb / disk_mb
            bulk_time = measure_bulk_read(path)
            gather_time = measure_gather_read(path, gather_indices)

            result = {
                "config": f"flat_1000_{shuffle_name}",
                "shuffle": shuffle_name,
                "chunk_rows": 1000,
                "sharded": False,
                "disk_mb": disk_mb,
                "ratio": ratio,
                "write_s": write_time,
                "bulk_read_s": bulk_time,
                "gather_1k_s": gather_time,
            }
            results.append(result)
            print(f"  {shuffle_name:12s}: {disk_mb:7.1f} MB  ({ratio:.2f}x)  "
                  f"write={write_time:.2f}s  bulk={bulk_time:.3f}s  "
                  f"gather({N_GATHER})={gather_time:.3f}s")

        # =====================================================
        # Benchmark 2: Chunk/shard sizes with bitshuffle
        # =====================================================
        print()
        print("=" * 70)
        print("BENCHMARK 2: Sharded (shard=whole array), bitshuffle")
        print("=" * 70)

        for inner_rows in [1, 10, 50, 100, 1000]:
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
            path = Path(tmpdir) / f"sharded_bit_inner{inner_rows}"

            t0 = time.perf_counter()
            write_zarr_sharded(data, path, compressor, inner_chunk_rows=inner_rows)
            write_time = time.perf_counter() - t0

            disk_mb = measure_dir_size(path) / 1e6
            ratio = raw_mb / disk_mb
            bulk_time = measure_bulk_read(path)
            gather_time = measure_gather_read(path, gather_indices)

            result = {
                "config": f"sharded_{inner_rows}_bitshuffle",
                "shuffle": "bitshuffle",
                "chunk_rows": inner_rows,
                "sharded": True,
                "disk_mb": disk_mb,
                "ratio": ratio,
                "write_s": write_time,
                "bulk_read_s": bulk_time,
                "gather_1k_s": gather_time,
            }
            results.append(result)
            print(f"  inner={inner_rows:5d}: {disk_mb:7.1f} MB  ({ratio:.2f}x)  "
                  f"write={write_time:.2f}s  bulk={bulk_time:.3f}s  "
                  f"gather({N_GATHER})={gather_time:.3f}s")

        # =====================================================
        # Benchmark 3: Same with byteshuffle for comparison
        # =====================================================
        print()
        print("=" * 70)
        print("BENCHMARK 3: Sharded (shard=whole array), shuffle (byteshuffle)")
        print("=" * 70)

        for inner_rows in [1, 10, 50, 100, 1000]:
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
            path = Path(tmpdir) / f"sharded_byte_inner{inner_rows}"

            t0 = time.perf_counter()
            write_zarr_sharded(data, path, compressor, inner_chunk_rows=inner_rows)
            write_time = time.perf_counter() - t0

            disk_mb = measure_dir_size(path) / 1e6
            ratio = raw_mb / disk_mb
            gather_time = measure_gather_read(path, gather_indices)

            result = {
                "config": f"sharded_{inner_rows}_shuffle",
                "shuffle": "shuffle",
                "chunk_rows": inner_rows,
                "sharded": True,
                "disk_mb": disk_mb,
                "ratio": ratio,
                "write_s": write_time,
                "gather_1k_s": gather_time,
            }
            results.append(result)
            print(f"  inner={inner_rows:5d}: {disk_mb:7.1f} MB  ({ratio:.2f}x)  "
                  f"write={write_time:.2f}s  "
                  f"gather({N_GATHER})={gather_time:.3f}s")

        # =====================================================
        # Summary
        # =====================================================
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Raw data: {raw_mb:.1f} MB  ({n_sources} sources x 5501 wavelengths, float32)")
        print(f"Gather test: {N_GATHER} random non-consecutive rows, median of {N_GATHER_REPEATS} repeats")
        print()
        print(f"{'Config':<35s} {'Size MB':>8s} {'Ratio':>6s} {'Bulk':>8s} {'Gather':>10s}")
        print("-" * 70)
        for r in results:
            bulk_str = f"{r['bulk_read_s']:.3f}s" if 'bulk_read_s' in r else "—"
            gather_str = f"{r['gather_1k_s']:.3f}s"
            print(f"{r['config']:<35s} {r['disk_mb']:>8.1f} {r['ratio']:>6.2f} "
                  f"{bulk_str:>8s} {gather_str:>10s}")

    finally:
        print(f"\nCleaning up {tmpdir}")
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    run_benchmarks()

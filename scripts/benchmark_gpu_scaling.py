#!/usr/bin/env python
"""
GPU Scaling Benchmarks for Roman Disperser

This script benchmarks the disperser performance across:
- Galaxy counts: 100, 250, 500, 1000
- Spectral orders: "1", "0", "2"
- Wavelength chunk sizes: 50, 100, 200

Outputs:
- benchmark_results.json: Timing data with GPU hardware info
- dispersed_1000_galaxies.png: Combined detector image (all 3 orders)

Usage:
    pixi run -e cuda python scripts/benchmark_gpu_scaling.py
    pixi run -e cuda python scripts/benchmark_gpu_scaling.py --output-dir scripts/output
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj
import roman_disperser.disperser as disperser
import roman_disperser.demo_utils as demo_utils


# ============================================================================
# CONFIGURATION
# ============================================================================

# Galaxy counts to test
GALAXY_COUNTS = [100, 250, 500, 1000]

# Spectral orders
ORDERS = ["1", "0", "2"]

# Wavelength chunk sizes
CHUNK_SIZES = [50, 100, 200]

# Galaxy parameters (fixed)
NPIX_NATIVE = 50           # 50x50 native pixels
OVERSAMPLE = 3             # 3x oversampling -> 150x150 input
N_WAVELENGTH = 1000        # 1000 wavelength samples
LAM_MIN = 1.0              # microns
LAM_MAX = 2.0              # microns
HALF_LIGHT_RADIUS = 0.3    # arcsec

# SCA to test
SCA = 5

# Galaxy position range (full detector)
X_RANGE = (0.0, 4088.0)
Y_RANGE = (0.0, 4088.0)

# Number of warmup runs before timing
N_WARMUP = 2

# Number of timed runs for averaging
N_RUNS = 3

# Order efficiency factors (accounts for grism throughput)
# These scale the spectrum to reflect realistic relative flux levels
ORDER_EFFICIENCIES = {
    "1": 1.0,    # Order +1: full efficiency (reference)
    "0": 0.02,   # Order 0: 2% efficiency
    "2": 0.01,   # Order +2: 1% efficiency
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark configuration."""
    n_galaxies: int
    order: str
    chunk_size: int
    compile_time_s: float
    run_times_s: list
    mean_time_s: float
    std_time_s: float
    time_per_galaxy_ms: float
    total_flux: float
    peak_memory_bytes: int


# ============================================================================
# GPU INFO
# ============================================================================

def get_gpu_info() -> dict:
    """Get GPU hardware information."""
    gpu_info = {
        "name": "Unknown",
        "memory_total_gb": 0.0,
        "cuda_version": "Unknown",
        "driver_version": "Unknown",
    }

    # Try nvidia-smi for detailed info
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                gpu_info["name"] = parts[0].strip()
                gpu_info["memory_total_gb"] = float(parts[1].strip()) / 1024
                gpu_info["driver_version"] = parts[2].strip()
    except Exception:
        pass

    # Get CUDA version from nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        # Also get CUDA version from the header
        result2 = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result2.returncode == 0:
            for line in result2.stdout.split('\n'):
                if 'CUDA Version' in line:
                    # Parse "CUDA Version: 12.4"
                    cuda_part = line.split('CUDA Version:')[-1].strip()
                    cuda_version = cuda_part.split()[0] if cuda_part else "Unknown"
                    gpu_info["cuda_version"] = cuda_version
                    break
    except Exception:
        pass

    return gpu_info


def get_gpu_memory_usage() -> Optional[dict]:
    """Get GPU memory stats from JAX."""
    try:
        devices = jax.devices('gpu')
        if devices:
            device = devices[0]
            stats = device.memory_stats()
            if stats:
                return {
                    'bytes_in_use': stats.get('bytes_in_use', 0),
                    'peak_bytes': stats.get('peak_bytes_in_use', 0),
                }
    except Exception:
        pass
    return None


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_test_data(n_galaxies_max: int, model: RomanOpticalModel, seed: int = 42) -> dict:
    """
    Pre-generate test data for the largest galaxy count.

    Args:
        n_galaxies_max: Maximum number of galaxies to generate
        model: Optical model for pixel scale
        seed: Random seed for reproducibility

    Returns:
        dict with images, x0s, y0s, specs, lam0s, dlams, dx, dy
    """
    pixel_scale = model.detmod["pixel_scale"]

    # Generate galaxy positions over full detector
    x_centers, y_centers = demo_utils.make_random_galaxy_positions(
        n_galaxies=n_galaxies_max,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        seed=seed,
    )

    # Create single galaxy image (same for all)
    galaxy_image = demo_utils.make_exponential_galaxy(
        npix=NPIX_NATIVE,
        half_light_radius_arcsec=HALF_LIGHT_RADIUS,
        pixel_scale_arcsec=pixel_scale,
        oversample=OVERSAMPLE,
        normalize=True,
    )

    # Create sloped spectrum
    spectrum, lam0, dlam = demo_utils.make_sloped_spectrum(
        lam_min=LAM_MIN,
        lam_max=LAM_MAX,
        n_wavelength=N_WAVELENGTH,
        slope_min=0.5,
        slope_max=1.5,
        taper_fraction=0.2,
    )

    # Pixel spacing for oversampled input
    dx = 1.0 / OVERSAMPLE
    dy = 1.0 / OVERSAMPLE

    # Convert center positions to corner positions
    npix_oversampled = NPIX_NATIVE * OVERSAMPLE
    x0s, y0s = demo_utils.center_to_corner(
        x_centers, y_centers, npix_oversampled, npix_oversampled, dx, dy
    )

    # Create batched arrays
    images = jnp.stack([galaxy_image] * n_galaxies_max)
    specs = jnp.stack([spectrum] * n_galaxies_max)
    x0s_jax = jnp.array(x0s, dtype=jnp.float32)
    y0s_jax = jnp.array(y0s, dtype=jnp.float32)
    lam0s_jax = jnp.ones(n_galaxies_max, dtype=jnp.float32) * lam0
    dlams_jax = jnp.ones(n_galaxies_max, dtype=jnp.float32) * dlam

    return {
        'images': images,
        'x0s': x0s_jax,
        'y0s': y0s_jax,
        'specs': specs,
        'lam0s': lam0s_jax,
        'dlams': dlams_jax,
        'dx': dx,
        'dy': dy,
        'lam0': lam0,
        'dlam': dlam,
    }


# ============================================================================
# BENCHMARKING
# ============================================================================

def run_single_benchmark(
    model: RomanOpticalModel,
    test_data: dict,
    n_galaxies: int,
    order: str,
    chunk_size: int,
) -> tuple[BenchmarkResult, jnp.ndarray]:
    """
    Run a single benchmark configuration.

    Returns:
        BenchmarkResult and the final output array
    """
    # Slice test data to n_galaxies
    images = test_data['images'][:n_galaxies]
    x0s = test_data['x0s'][:n_galaxies]
    y0s = test_data['y0s'][:n_galaxies]
    specs = test_data['specs'][:n_galaxies]
    lam0s = test_data['lam0s'][:n_galaxies]
    dlams = test_data['dlams'][:n_galaxies]
    dx = test_data['dx']
    dy = test_data['dy']

    # Apply order efficiency scaling
    efficiency = ORDER_EFFICIENCIES.get(order, 1.0)
    specs = specs * efficiency

    # Create payload and JIT-compiled function
    payload = omj.make_sca_payload(model, sca=SCA, order=order)

    @jax.jit
    def disperse_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams):
        return disperser.disperse_galaxies_sequential(
            payload, images, x0s, y0s, dx, dy, specs, lam0s, dlams,
            wavelength_chunk_size=chunk_size
        )

    # First call: compilation
    t0 = time.perf_counter()
    output = disperse_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams)
    output.block_until_ready()
    compile_time = time.perf_counter() - t0

    # Warmup (already compiled, just warming up GPU)
    for _ in range(N_WARMUP - 1):
        output = disperse_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams)
        output.block_until_ready()

    # Timed runs
    run_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        output = disperse_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams)
        output.block_until_ready()
        run_times.append(time.perf_counter() - t0)

    # Memory stats
    memory_stats = get_gpu_memory_usage()
    peak_bytes = memory_stats['peak_bytes'] if memory_stats else 0

    mean_time = float(np.mean(run_times))
    std_time = float(np.std(run_times))

    result = BenchmarkResult(
        n_galaxies=n_galaxies,
        order=order,
        chunk_size=chunk_size,
        compile_time_s=compile_time,
        run_times_s=run_times,
        mean_time_s=mean_time,
        std_time_s=std_time,
        time_per_galaxy_ms=mean_time / n_galaxies * 1000,
        total_flux=float(output.sum()),
        peak_memory_bytes=peak_bytes,
    )

    return result, output


def disperse_all_orders(
    model: RomanOpticalModel,
    test_data: dict,
    n_galaxies: int,
    chunk_size: int,
    orders: list[str] = ORDERS,
) -> jnp.ndarray:
    """
    Disperse galaxies through all orders, accumulating onto single detector.

    Returns:
        Combined output array with all orders
    """
    dx = test_data['dx']
    dy = test_data['dy']

    # Slice test data
    images = test_data['images'][:n_galaxies]
    x0s = test_data['x0s'][:n_galaxies]
    y0s = test_data['y0s'][:n_galaxies]
    specs = test_data['specs'][:n_galaxies]
    lam0s = test_data['lam0s'][:n_galaxies]
    dlams = test_data['dlams'][:n_galaxies]

    combined_output = jnp.zeros((4088, 4088), dtype=jnp.float32)

    for order in orders:
        # Apply order efficiency scaling
        efficiency = ORDER_EFFICIENCIES.get(order, 1.0)
        specs_scaled = specs * efficiency

        payload = omj.make_sca_payload(model, sca=SCA, order=order)

        @jax.jit
        def disperse_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams):
            return disperser.disperse_galaxies_sequential(
                payload, images, x0s, y0s, dx, dy, specs, lam0s, dlams,
                wavelength_chunk_size=chunk_size
            )

        output = disperse_jit(images, x0s, y0s, dx, dy, specs_scaled, lam0s, dlams)
        output.block_until_ready()
        combined_output = combined_output + output

    return combined_output


# ============================================================================
# OUTPUT
# ============================================================================

def save_results_json(results: list[BenchmarkResult], gpu_info: dict, output_path: Path):
    """Save benchmark results to JSON."""
    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'gpu': gpu_info,
            'jax_backend': str(jax.default_backend()),
            'jax_devices': [str(d) for d in jax.devices()],
            'config': {
                'sca': SCA,
                'npix_native': NPIX_NATIVE,
                'oversample': OVERSAMPLE,
                'n_wavelength': N_WAVELENGTH,
                'lam_range': [LAM_MIN, LAM_MAX],
                'x_range': list(X_RANGE),
                'y_range': list(Y_RANGE),
                'half_light_radius': HALF_LIGHT_RADIUS,
                'n_warmup': N_WARMUP,
                'n_runs': N_RUNS,
                'order_efficiencies': ORDER_EFFICIENCIES,
            }
        },
        'results': [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_detector_png(output_array: jnp.ndarray, filepath: Path, title: str):
    """Save detector image as PNG."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use log scale for better visibility
    vmax = float(output_array.max())
    vmin = max(1e-8, vmax * 1e-5)

    im = ax.imshow(
        np.array(output_array),
        origin='lower',
        cmap='viridis',
        norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X (SCA pixels)', fontsize=12)
    ax.set_ylabel('Y (SCA pixels)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Flux', shrink=0.8)

    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_summary_table(results: list[BenchmarkResult]):
    """Print formatted summary table to console."""
    print("\n" + "=" * 80)
    print("GPU SCALING BENCHMARK RESULTS")
    print("=" * 80)

    # Group by chunk_size for cleaner display
    for chunk_size in sorted(set(r.chunk_size for r in results)):
        print(f"\nWavelength Chunk Size: {chunk_size}")
        print("-" * 75)
        print(f"{'N_gal':>8} {'Order':>6} {'Mean(s)':>10} {'Std(s)':>10} {'ms/gal':>10} {'Peak MB':>12}")
        print("-" * 75)

        for r in sorted(results, key=lambda x: (x.n_galaxies, x.order)):
            if r.chunk_size == chunk_size:
                peak_mb = r.peak_memory_bytes / 1024**2
                print(f"{r.n_galaxies:>8} {r.order:>6} {r.mean_time_s:>10.3f} "
                      f"{r.std_time_s:>10.4f} {r.time_per_galaxy_ms:>10.3f} {peak_mb:>12.1f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU Scaling Benchmarks for Roman Disperser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('scripts/output'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--galaxy-counts',
        type=int,
        nargs='+',
        default=GALAXY_COUNTS,
        help='Galaxy counts to test'
    )
    parser.add_argument(
        '--orders',
        nargs='+',
        default=ORDERS,
        help='Spectral orders to test'
    )
    parser.add_argument(
        '--chunk-sizes',
        type=int,
        nargs='+',
        default=CHUNK_SIZES,
        help='Wavelength chunk sizes to test'
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    print("=" * 60)
    print("GPU SCALING BENCHMARKS")
    print("=" * 60)

    backend = jax.default_backend()
    print(f"\nJAX Backend: {backend}")
    print(f"JAX Devices: {jax.devices()}")

    if backend != 'gpu':
        print("\nWARNING: Not running on GPU! Results may not be meaningful.")
        print("Use: pixi run -e cuda python scripts/benchmark_gpu_scaling.py")

    # Get GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Info:")
    print(f"  Name: {gpu_info['name']}")
    print(f"  Memory: {gpu_info['memory_total_gb']:.1f} GB")
    print(f"  CUDA Version: {gpu_info['cuda_version']}")
    print(f"  Driver Version: {gpu_info['driver_version']}")

    # Load model
    print("\nLoading optical model...")
    project_root = Path(os.environ.get("PIXI_PROJECT_ROOT", "."))
    config_path = project_root / "data" / "Roman_grism_OpticalModel_v0.8.yaml"
    model = RomanOpticalModel(config_file=str(config_path))
    print(f"  Loaded from {config_path.name}")

    # Generate test data
    max_galaxies = max(args.galaxy_counts)
    print(f"\nGenerating test data for {max_galaxies} galaxies...")
    test_data = generate_test_data(max_galaxies, model)
    print(f"  Images: {test_data['images'].shape}")
    print(f"  Spectra: {test_data['specs'].shape}")

    # Run benchmarks
    results = []
    total_configs = len(args.galaxy_counts) * len(args.orders) * len(args.chunk_sizes)

    print(f"\nRunning {total_configs} benchmark configurations...")
    print(f"  Galaxy counts: {args.galaxy_counts}")
    print(f"  Orders: {args.orders}")
    print(f"  Chunk sizes: {args.chunk_sizes}")
    print()

    for i, (n_gal, order, chunk) in enumerate(
        product(args.galaxy_counts, args.orders, args.chunk_sizes)
    ):
        print(f"[{i+1:3d}/{total_configs}] n_galaxies={n_gal:4d}, order={order}, chunk_size={chunk:3d} ... ", end="", flush=True)
        result, _ = run_single_benchmark(model, test_data, n_gal, order, chunk)
        results.append(result)
        print(f"{result.mean_time_s:.3f}s ({result.time_per_galaxy_ms:.2f} ms/galaxy)")

    # Save JSON results
    json_path = args.output_dir / "benchmark_results.json"
    save_results_json(results, gpu_info, json_path)
    print(f"\nResults saved to {json_path}")

    # Generate combined detector image for 1000 galaxies
    print(f"\nGenerating combined detector image for 1000 galaxies (all orders)...")
    combined_output = disperse_all_orders(
        model, test_data,
        n_galaxies=1000,
        chunk_size=100,  # Use default chunk size
    )

    png_path = args.output_dir / "dispersed_1000_galaxies.png"
    save_detector_png(
        combined_output,
        png_path,
        title=f"1000 Galaxies - All Orders Combined (SCA {SCA})"
    )
    print(f"Detector image saved to {png_path}")

    # Print summary table
    print_summary_table(results)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  {json_path}")
    print(f"  {png_path}")

    # Total time for 1000 galaxies at default chunk size
    chunk_100_results = [r for r in results if r.chunk_size == 100 and r.n_galaxies == 1000]
    if chunk_100_results:
        total_1000 = sum(r.mean_time_s for r in chunk_100_results)
        print(f"\n1000 galaxies, all 3 orders (chunk_size=100):")
        print(f"  Total time: {total_1000:.2f}s")
        print(f"  Per galaxy (all orders): {total_1000/1000*1000:.2f} ms")


if __name__ == '__main__':
    main()

"""
Test get_map_coords on 3D grids (x, y spatial dimensions).
Compare single reshape vs vmap for memory efficiency.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import time
from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj

# Load model
pixi_root = os.environ.get("PIXI_PROJECT_ROOT", ".")
fn = os.path.join(pixi_root, "data/Roman_grism_OpticalModel_v0.8.yaml")
model = RomanOpticalModel(fn)
payload = omj.make_sca_payload(model, sca=1, order="1")

print("3D Grid Scaling Test for get_map_coords")
print("=" * 70)
print("Grid shape: (nx, ny, nlambda) where we compute xmpa, ympa for each point")
print()

# Test different grid sizes
grid_configs = [
    (10, 10, 100),      # 10k points
    (50, 50, 100),      # 250k points
    (100, 100, 100),    # 1M points
    (100, 100, 1000),   # 10M points - your use case
]

results = []

for nx, ny, nlambda in grid_configs:
    total_points = nx * ny * nlambda
    print(f"\nGrid: {nx}x{ny}x{nlambda} = {total_points:,} points")
    print("-" * 70)
    
    # Create 3D grid of FPA coordinates
    xfpa_grid = jnp.linspace(-0.01, 0.01, nx)
    yfpa_grid = jnp.linspace(-0.01, 0.01, ny)
    
    # Create meshgrid
    xx, yy = jnp.meshgrid(xfpa_grid, yfpa_grid, indexing='ij')
    
    # Replicate for wavelength dimension
    xx_3d = jnp.tile(xx[:, :, jnp.newaxis], (1, 1, nlambda))
    yy_3d = jnp.tile(yy[:, :, jnp.newaxis], (1, 1, nlambda))
    
    # Approach 1: Reshape to 1D, compute, reshape back
    def compute_reshape(xx_3d, yy_3d):
        shape_3d = xx_3d.shape
        xx_flat = xx_3d.reshape(-1)
        yy_flat = yy_3d.reshape(-1)
        xmpa_flat, ympa_flat = omj.get_map_coords(payload, xx_flat, yy_flat)
        return xmpa_flat.reshape(shape_3d), ympa_flat.reshape(shape_3d)
    
    # Approach 2: vmap over spatial dimensions
    def compute_vmap(xx_3d, yy_3d):
        def compute_1d_slice(xx_slice, yy_slice):
            def compute_wavelength_slice(xx_1d, yy_1d):
                return omj.get_map_coords(payload, xx_1d, yy_1d)
            xmpa, ympa = jax.vmap(compute_wavelength_slice)(xx_slice, yy_slice)
            return xmpa, ympa
        xmpa_3d, ympa_3d = jax.vmap(compute_1d_slice)(xx_3d, yy_3d)
        return xmpa_3d, ympa_3d
    
    # JIT compile both
    compute_reshape_jit = jax.jit(compute_reshape)
    compute_vmap_jit = jax.jit(compute_vmap)
    
    # Warmup
    _ = compute_reshape_jit(xx_3d, yy_3d)
    _ = compute_vmap_jit(xx_3d, yy_3d)
    
    # Verify correctness
    xmpa_reshape, ympa_reshape = compute_reshape_jit(xx_3d, yy_3d)
    xmpa_vmap, ympa_vmap = compute_vmap_jit(xx_3d, yy_3d)
    
    match_x = jnp.allclose(xmpa_reshape, xmpa_vmap, rtol=1e-5, atol=1e-3)
    match_y = jnp.allclose(ympa_reshape, ympa_vmap, rtol=1e-5, atol=1e-3)
    print(f"Results match: x={match_x}, y={match_y}")
    
    # Timing
    n_runs = 5
    
    start = time.time()
    for _ in range(n_runs):
        result = compute_reshape_jit(xx_3d, yy_3d)
        jax.block_until_ready(result)
    reshape_time = time.time() - start
    
    start = time.time()
    for _ in range(n_runs):
        result = compute_vmap_jit(xx_3d, yy_3d)
        jax.block_until_ready(result)
    vmap_time = time.time() - start
    
    reshape_ms = reshape_time / n_runs * 1000
    vmap_ms = vmap_time / n_runs * 1000
    
    print(f"Reshape:  {reshape_ms:.2f} ms/run ({reshape_time:.3f}s total)")
    print(f"Vmap:     {vmap_ms:.2f} ms/run ({vmap_time:.3f}s total)")
    
    if vmap_time < reshape_time:
        print(f"Vmap speedup: {reshape_time/vmap_time:.2f}x")
    else:
        print(f"Reshape speedup: {vmap_time/reshape_time:.2f}x")
    
    results.append({
        'grid': (nx, ny, nlambda),
        'points': total_points,
        'reshape_ms': reshape_ms,
        'vmap_ms': vmap_ms,
        'speedup': reshape_time / vmap_time if vmap_time > 0 else 1.0,
    })

print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(f"{'Grid':<20} {'Points':<12} {'Reshape':<12} {'Vmap':<12} {'Speedup':<10}")
print("-" * 70)
for r in results:
    grid_str = f"{r['grid'][0]}x{r['grid'][1]}x{r['grid'][2]}"
    print(f"{grid_str:<20} {r['points']:<12,} {r['reshape_ms']:<11.2f}ms {r['vmap_ms']:<11.2f}ms {r['speedup']:<9.2f}x")

print("\n" + "=" * 70)
print("Conclusions")
print("=" * 70)
print("1. For your 100x100x1000 grid (10M points):")
if results:
    r = results[-1]
    speedup_factor = 1 / r['speedup'] if r['speedup'] < 1 else r['speedup']
    if r['speedup'] < 1:
        # reshape_time < vmap_time, so reshape is faster
        print(f"   - Reshape is {speedup_factor:.2f}x faster (~214 ms vs 265 ms per run)")
    else:
        # vmap_time < reshape_time, so vmap is faster
        print(f"   - Vmap is {speedup_factor:.2f}x faster")

print("\n2. When to use each:")
print("   - Reshape: Simple, direct computation, works for most grid sizes")
print("   - Vmap: Better memory locality, can help if memory is tight")
print("     (Vmap overhead visible on CPU; would differ on GPU)")

print("\n3. For memory-constrained scenarios:")
print("   - Can vmap one spatial dimension at a time")
print("   - Process wavelengths in batches with outer loop")
print("   - Trade off computation speed for lower peak memory usage")

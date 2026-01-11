"""Test if jnp.vander has advantages over manual power computation."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import time

n = 1000
map_i = 6
map_j = 6

xfpa = jnp.linspace(-0.01, 0.01, n)
yfpa = jnp.linspace(-0.01, 0.01, n)

# Manual powers approach
def manual_powers(x, y, map_i, map_j):
    x_powers = x[:, jnp.newaxis] ** jnp.arange(map_i)
    y_powers = y[:, jnp.newaxis] ** jnp.arange(map_j)
    return x_powers, y_powers

# Using jnp.vander
def using_vander(x, y, map_i, map_j):
    x_powers = jnp.vander(x, N=map_i, increasing=True)
    y_powers = jnp.vander(y, N=map_j, increasing=True)
    return x_powers, y_powers

# Test correctness
x1, y1 = manual_powers(xfpa, yfpa, map_i, map_j)
x2, y2 = using_vander(xfpa, yfpa, map_i, map_j)

print("Results match:", jnp.allclose(x1, x2) and jnp.allclose(y1, y2))
print(f"Max diff x: {jnp.max(jnp.abs(x1 - x2))}")
print(f"Max diff y: {jnp.max(jnp.abs(y1 - y2))}")

# JIT compile
manual_jit = jax.jit(manual_powers, static_argnums=(2, 3))
vander_jit = jax.jit(using_vander, static_argnums=(2, 3))

# Warmup
_ = manual_jit(xfpa, yfpa, map_i, map_j)
_ = vander_jit(xfpa, yfpa, map_i, map_j)

# Timing
n_runs = 1000

start = time.time()
for _ in range(n_runs):
    result = manual_jit(xfpa, yfpa, map_i, map_j)
    jax.block_until_ready(result)
manual_time = time.time() - start

start = time.time()
for _ in range(n_runs):
    result = vander_jit(xfpa, yfpa, map_i, map_j)
    jax.block_until_ready(result)
vander_time = time.time() - start

print(f"\nTiming (n={n}, {n_runs} runs):")
print(f"Manual powers: {manual_time:.4f}s ({manual_time/n_runs*1000:.3f} ms/run)")
print(f"jnp.vander:    {vander_time:.4f}s ({vander_time/n_runs*1000:.3f} ms/run)")
print(f"Speedup: {manual_time/vander_time:.2f}x" if vander_time < manual_time else f"Slowdown: {vander_time/manual_time:.2f}x")

# Check HLO
print("\n--- Manual powers HLO (first 800 chars) ---")
hlo_manual = jax.jit(manual_powers, static_argnums=(2, 3)).lower(xfpa, yfpa, map_i, map_j).compile().as_text()
print(hlo_manual[:800])

print("\n--- jnp.vander HLO (first 800 chars) ---")
hlo_vander = jax.jit(using_vander, static_argnums=(2, 3)).lower(xfpa, yfpa, map_i, map_j).compile().as_text()
print(hlo_vander[:800])

print("\n--- Code readability comparison ---")
print("Manual: x_powers = x[:, jnp.newaxis] ** jnp.arange(map_i)")
print("Vander: x_powers = jnp.vander(x, N=map_i, increasing=True)")
print("\nVander is more explicit about creating a Vandermonde matrix.")

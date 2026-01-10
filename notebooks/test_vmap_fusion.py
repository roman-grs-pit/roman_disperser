"""Test if JAX fuses vmapped operations into batched matrix ops."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from functools import partial
import time

print("Testing if vmap + einsum fuses into batched operations")
print("=" * 70)

# Setup
batch_size = 16
n_per_batch = 1000
map_i = 6
map_j = 6

# Create test data
xfpa_batch = jnp.linspace(-0.01, 0.01, n_per_batch)  # [n_per_batch]
yfpa_batch = jnp.linspace(-0.01, 0.01, n_per_batch)

# Different coefficient matrices for each batch
coeff = jnp.ones((map_i, map_j))

# Approach 1: Loop over batches
def loop_approach(xfpa_list, yfpa_list):
    results_x = []
    results_y = []
    for xfpa, yfpa in zip(xfpa_list, yfpa_list):
        x_powers = jnp.vander(xfpa, N=map_i, increasing=True)
        y_powers = jnp.vander(yfpa, N=map_j, increasing=True)
        xmpa = jnp.einsum('ni,ij,nj->n', x_powers, coeff, y_powers)
        ympa = jnp.einsum('ni,ij,nj->n', x_powers, coeff, y_powers)
        results_x.append(xmpa)
        results_y.append(ympa)
    return jnp.stack(results_x), jnp.stack(results_y)

# Approach 2: vmap the function
def single_compute(xfpa, yfpa):
    x_powers = jnp.vander(xfpa, N=map_i, increasing=True)
    y_powers = jnp.vander(yfpa, N=map_j, increasing=True)
    xmpa = jnp.einsum('ni,ij,nj->n', x_powers, coeff, y_powers)
    ympa = jnp.einsum('ni,ij,nj->n', x_powers, coeff, y_powers)
    return xmpa, ympa

vmapped_compute = jax.vmap(single_compute)

# Create batch data
xfpa_batches = jnp.tile(xfpa_batch[jnp.newaxis, :], (batch_size, 1))  # [batch_size, n_per_batch]
yfpa_batches = jnp.tile(yfpa_batch[jnp.newaxis, :], (batch_size, 1))

# Test correctness (first batch)
print("\nTesting correctness:")
x_single, y_single = single_compute(xfpa_batch, yfpa_batch)
x_vmapped, y_vmapped = vmapped_compute(xfpa_batches, yfpa_batches)
print(f"Single batch shape: {x_single.shape}")
print(f"Vmapped output shape: {x_vmapped.shape}")
print(f"First batch matches: {jnp.allclose(x_single, x_vmapped[0])}")

# JIT compile
loop_jit = jax.jit(loop_approach)
vmapped_jit = jax.jit(vmapped_compute)

# Warmup
xfpa_list = [xfpa_batch for _ in range(batch_size)]
yfpa_list = [yfpa_batch for _ in range(batch_size)]
_ = loop_jit(xfpa_list, yfpa_list)
_ = vmapped_jit(xfpa_batches, yfpa_batches)

# Timing
n_runs = 50

print(f"\nTiming (batch_size={batch_size}, n_per_batch={n_per_batch}, {n_runs} runs):")
print("-" * 70)

start = time.time()
for _ in range(n_runs):
    result = loop_jit(xfpa_list, yfpa_list)
    jax.block_until_ready(result)
loop_time = time.time() - start

start = time.time()
for _ in range(n_runs):
    result = vmapped_jit(xfpa_batches, yfpa_batches)
    jax.block_until_ready(result)
vmapped_time = time.time() - start

print(f"Loop:   {loop_time:.4f}s ({loop_time/n_runs*1000:.2f} ms/run)")
print(f"Vmap:   {vmapped_time:.4f}s ({vmapped_time/n_runs*1000:.2f} ms/run)")
if vmapped_time < loop_time:
    print(f"Speedup: {loop_time/vmapped_time:.2f}x")
else:
    print(f"Slowdown: {vmapped_time/loop_time:.2f}x")

# Check HLO to see fusion
print("\n" + "=" * 70)
print("HLO Analysis (checking for fusion):")
print("=" * 70)

print("\n--- Vmapped HLO (first 1000 chars) ---")
hlo_vmapped = jax.jit(vmapped_compute).lower(xfpa_batches, yfpa_batches).compile().as_text()
print(hlo_vmapped[:1000])

print("\n--- HLO summary ---")
# Count certain operations
n_vander = hlo_vmapped.count('vander')
n_einsum = hlo_vmapped.count('einsum')
n_dot = hlo_vmapped.count('dot')
n_mul = hlo_vmapped.count('multiply')
print(f"'vander' mentions: {n_vander}")
print(f"'einsum' mentions: {n_einsum}")
print(f"'dot' mentions: {n_dot}")
print(f"'multiply' mentions: {n_mul}")

print("\nConclusion:")
print("-" * 70)
print("If vmap fuses operations into batched matrix ops, we should see:")
print("1. Fewer total operations (consolidated into larger matmuls)")
print("2. Vmap should be faster or same speed as loop")
print("3. HLO should show batched/fusion operations rather than separate ones")

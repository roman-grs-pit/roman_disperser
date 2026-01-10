"""
Efficiency comparison: einsum vs matmul+diagonal for get_map_coords

Summary:
--------
When evaluating 2D polynomials of the form:
    result[i] = sum_j sum_k x_powers[i,j] * coeff[j,k] * y_powers[i,k]

We only need the diagonal elements where the input index matches (i.e., x[i] paired with y[i]).

Two approaches:
1. matmul+diagonal: Compute full matrix x_powers @ coeff @ y_powers.T -> [n,n], then extract diagonal
2. einsum: Directly compute only the diagonal elements using 'ni,ij,nj->n'

Results: einsum is ~12x faster because it avoids creating the full [n,n] intermediate matrix.

Einstein Summation (einsum) Notation:
--------------------------------------
The notation 'ni,ij,nj->n' means:
  - First arg (x_powers): shape [n, i] with indices 'ni'
  - Second arg (coeff):   shape [i, j] with indices 'ij'  
  - Third arg (y_powers): shape [n, j] with indices 'nj'
  - Output:               shape [n] with index 'n'

Repeated indices (i, j) are summed over (contracted).
The index 'n' appears in input and output, so we compute one result per n.
Critically, the 'n' index is the same across x_powers and y_powers, so we only
compute x_powers[k,:] @ coeff @ y_powers[k,:] for each k, giving us the diagonal.

This is equivalent to:
    result[k] = sum_i sum_j x_powers[k,i] * coeff[i,j] * y_powers[k,j]

If we used matmul instead:
    temp[k,m] = sum_i sum_j x_powers[k,i] * coeff[i,j] * y_powers[m,j]
we'd compute all (k,m) pairs, wasting computation on off-diagonal elements.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj

# Load model
fn = "../data/Roman_grism_OpticalModel_v0.8.yaml"
model = RomanOpticalModel(fn)
payload = omj.make_sca_payload(model, sca=1, order="1")

# Test data
n = 1000
xfpa = jnp.linspace(-0.01, 0.01, n)
yfpa = jnp.linspace(-0.01, 0.01, n)

map_i = payload["poly"]["map_i"]
map_j = payload["poly"]["map_j"]
X_ij = payload["poly"]["X_ij"]
Y_ij = payload["poly"]["Y_ij"]

# Current approach: matmul + diagonal
def matmul_diagonal(xfpa, yfpa):
    x_powers = xfpa[:, jnp.newaxis] ** jnp.arange(map_i)
    y_powers = yfpa[:, jnp.newaxis] ** jnp.arange(map_j)
    xmpa = jnp.diagonal(x_powers @ X_ij @ y_powers.T)
    ympa = jnp.diagonal(x_powers @ Y_ij @ y_powers.T)
    return xmpa, ympa

# Einsum approach
def einsum_approach(xfpa, yfpa):
    x_powers = xfpa[:, jnp.newaxis] ** jnp.arange(map_i)
    y_powers = yfpa[:, jnp.newaxis] ** jnp.arange(map_j)
    xmpa = jnp.einsum('ni,ij,nj->n', x_powers, X_ij, y_powers)
    ympa = jnp.einsum('ni,ij,nj->n', x_powers, Y_ij, y_powers)
    return xmpa, ympa

# Test correctness
xmpa1, ympa1 = matmul_diagonal(xfpa, yfpa)
xmpa2, ympa2 = einsum_approach(xfpa, yfpa)

print(f"Results match: {jnp.allclose(xmpa1, xmpa2) and jnp.allclose(ympa1, ympa2)}")
print(f"Max diff x: {jnp.max(jnp.abs(xmpa1 - xmpa2))}")
print(f"Max diff y: {jnp.max(jnp.abs(ympa1 - ympa2))}")

# JIT compile
matmul_jit = jax.jit(matmul_diagonal)
einsum_jit = jax.jit(einsum_approach)

# Warmup
_ = matmul_jit(xfpa, yfpa)
_ = einsum_jit(xfpa, yfpa)

# Time them
import time

n_runs = 100

start = time.time()
for _ in range(n_runs):
    result = matmul_jit(xfpa, yfpa)
    jax.block_until_ready(result)
matmul_time = time.time() - start

start = time.time()
for _ in range(n_runs):
    result = einsum_jit(xfpa, yfpa)
    jax.block_until_ready(result)
einsum_time = time.time() - start

print(f"\nTiming (n={n}, {n_runs} runs):")
print(f"matmul+diagonal: {matmul_time:.4f}s ({matmul_time/n_runs*1000:.2f} ms/run)")
print(f"einsum:          {einsum_time:.4f}s ({einsum_time/n_runs*1000:.2f} ms/run)")
print(f"Speedup: {matmul_time/einsum_time:.2f}x")

# Check HLO to see if JAX optimizes the matmul+diagonal
print("\n--- Matmul+diagonal HLO (optimized) ---")
print(jax.jit(matmul_diagonal).lower(xfpa, yfpa).compile().as_text()[:500])
print("\n--- Einsum HLO (optimized) ---")
print(jax.jit(einsum_approach).lower(xfpa, yfpa).compile().as_text()[:500])

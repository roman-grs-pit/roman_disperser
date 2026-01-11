"""Test if our code works with multidimensional arrays."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np

# Test jnp.vander with 2D input
print("Testing jnp.vander with different input shapes:")
print("=" * 60)

# 1D case (what we currently use)
x_1d = jnp.array([1.0, 2.0, 3.0])
print(f"\n1D input: {x_1d.shape}")
try:
    result = jnp.vander(x_1d, N=4, increasing=True)
    print(f"Output shape: {result.shape}")
    print(f"Output:\n{result}")
except Exception as e:
    print(f"Error: {e}")

# 2D case
x_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
print(f"\n2D input: {x_2d.shape}")
try:
    result = jnp.vander(x_2d, N=4, increasing=True)
    print(f"Output shape: {result.shape}")
    print(f"Output:\n{result}")
except Exception as e:
    print(f"Error: {e}")

# Test our einsum with different shapes
print("\n" + "=" * 60)
print("Testing einsum with different input shapes:")
print("=" * 60)

# Mock coefficients
coeff = jnp.ones((4, 4))

# 1D case
print("\n1D arrays (current usage):")
x_1d = jnp.array([1.0, 2.0, 3.0])
y_1d = jnp.array([1.0, 2.0, 3.0])
x_powers_1d = jnp.vander(x_1d, N=4, increasing=True)
y_powers_1d = jnp.vander(y_1d, N=4, increasing=True)
print(f"x_powers shape: {x_powers_1d.shape}, y_powers shape: {y_powers_1d.shape}")
try:
    result = jnp.einsum('ni,ij,nj->n', x_powers_1d, coeff, y_powers_1d)
    print(f"Result shape: {result.shape}")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# 2D case - if vander worked
print("\n2D arrays (hypothetical):")
x_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
print(f"x_2d shape: {x_2d.shape}")
try:
    x_powers_2d = jnp.vander(x_2d, N=4, increasing=True)
    print(f"x_powers_2d shape: {x_powers_2d.shape}")
    # This won't work with our einsum
except Exception as e:
    print(f"Vander error: {e}")

# What if we wanted to support 2D?
print("\n" + "=" * 60)
print("How to handle 2D arrays if needed:")
print("=" * 60)
print("Option 1: Flatten, compute, reshape")
print("Option 2: Use different einsum notation")
print("Option 3: Use vmap to vectorize over extra dimensions")
print("\nFor research code, simplest is to require 1D arrays (current approach)")

# JIT Compilation for the Disperser

This document explains the JIT (Just-In-Time) compilation strategy for the disperser module.

## The Challenge

The `disperse_2d1d_sca` and `disperse_galaxies_sequential` functions take a `payload` argument that cannot be easily traced by JAX:

1. **Non-traceable strings**: The payload contains `payload["wl"]["transform"]` which is a string ("linear" or "log"). JAX can't trace through Python string comparisons.

2. **Non-hashable dicts**: The payload contains nested dicts with JAX arrays. This means you can't use `static_argnums` (which requires hashable arguments for caching).

## The Solution: Closure Pattern

The solution is to capture the payload in a closure, then JIT-compile the wrapper function:

```python
import jax
import roman_disperser.optical_model_jax as omj
import roman_disperser.disperser as disperser

# Create payload for specific SCA and order
payload = omj.make_sca_payload(model, sca=5, order="1")

# Capture payload in closure, then JIT
@jax.jit
def disperse_jit(image, x0, y0, dx, dy, spec, lam0, dlam, output):
    return disperser.disperse_2d1d_sca(
        payload, image, x0, y0, dx, dy, spec, lam0, dlam, output,
        wavelength_chunk_size=100
    )

# Use the compiled function
output = disperse_jit(image, x0, y0, dx, dy, spec, lam0, dlam, output)
```

**Why this works:** JAX captures the payload at trace time, baking it into the compiled function. The string fields and JAX arrays become constants in the compiled XLA code.

## Single Galaxy Example

From the `single_galaxy_demo.ipynb` notebook (now in `notebooks/archive/` —
it demos the legacy `disperser.py` module, but the closure pattern shown here
is unchanged and used throughout the package):

```python
# Create payload for this SCA and order
payload = omj.make_sca_payload(model, sca=SCA, order=order)

# Create JIT-compiled version with payload captured in closure
@jax.jit
def disperse_jit(image, x0, y0, dx, dy, spec, lam0, dlam, output):
    return disperser.disperse_2d1d_sca(
        payload, image, x0, y0, dx, dy, spec, lam0, dlam, output,
        wavelength_chunk_size=WAVELENGTH_CHUNK_SIZE
    )

# Initialize output detector
output = jnp.zeros((4088, 4088), dtype=jnp.float32)

# First call includes JIT compilation time
output = disperse_jit(
    image=galaxy_image,
    x0=X0, y0=Y0,
    dx=dx, dy=dy,
    spec=spectrum,
    lam0=lam0, dlam=dlam,
    output=output,
)
```

## Multi-Galaxy Example

For processing multiple galaxies with the same (SCA, order), use the same pattern with `disperse_galaxies_sequential`:

```python
payload = omj.make_sca_payload(model, sca=5, order="1")

@jax.jit
def disperse_batch_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams):
    return disperser.disperse_galaxies_sequential(
        payload, images, x0s, y0s, dx, dy, specs, lam0s, dlams,
        wavelength_chunk_size=100
    )

# Process 1000s of galaxies in one JIT'd call
output = disperse_batch_jit(images, x0s, y0s, dx, dy, specs, lam0s, dlams)
```

The `disperse_galaxies_sequential` function uses `jax.lax.fori_loop` internally, which is fully compatible with JIT compilation.

## Performance Notes

Typical performance on CPU (from notebook benchmarks):

| Stage | Time |
|-------|------|
| First call (includes compilation) | ~0.5s |
| Cached calls | ~0.3s |
| Speedup | 1.3-1.6x |

Key points:
- Compilation is cached per unique closure (payload baked in)
- Each (SCA, order) combination requires a separate compiled function
- For 1000s of galaxies through the same SCA/order, compilation overhead is negligible

## Why Not Other Approaches?

### `static_argnums`

Won't work because:
- The payload dict contains JAX arrays, which are not hashable
- `static_argnums` requires hashable arguments to cache compilations

### Making payload traceable

Converting strings to integers and using `jax.lax.cond` is possible but:
- Adds code complexity
- `jax.lax.cond` has overhead (evaluates both branches)
- No performance benefit over closures
- The nested dict structure still can't be passed as a static argument

### Factory functions

We considered adding factory functions like `make_jit_disperser(model, sca, order)` but:
- The closure pattern is simple and explicit
- Users already need to manage payloads for different (SCA, order) combinations
- Factory functions add abstraction without clear benefit for typical workflows

## Tips for Efficient Usage

1. **Reuse compiled functions**: Create one closure per (SCA, order) and reuse it for all galaxies in that configuration.

2. **Don't recreate closures unnecessarily**: Each `@jax.jit` decorator creates a new function object, triggering recompilation.

3. **Cache across orders**: If processing multiple orders, create a dict of compiled functions:
   ```python
   dispersers = {}
   for order in ["1", "0", "2"]:
       payload = omj.make_sca_payload(model, sca=5, order=order)
       @jax.jit
       def disperse(image, x0, y0, dx, dy, spec, lam0, dlam, output, _payload=payload):
           return disperser.disperse_2d1d_sca(
               _payload, image, x0, y0, dx, dy, spec, lam0, dlam, output,
               wavelength_chunk_size=100
           )
       dispersers[order] = disperse
   ```
   Note the `_payload=payload` default argument trick to capture the current loop value.

4. **Block until ready**: JAX operations are asynchronous. Use `output.block_until_ready()` for accurate timing.

# JIT Compilation for the Disperser

How the star and galaxy dispersers are compiled, and why the payloads are
passed as jit **arguments** rather than captured in closures (changed on
branch `perf/jit-payload-args`, 2026-10; earlier versions used closures —
see "History" below).

## The challenge

The dispersion functions take two payload dicts:

- the optical payload (`optical_model_jax.make_sca_payload`)
- the PSF payload (`psf_model.get_or_make_psf_payload`)

Neither can be handed to `jax.jit` as-is:

1. **Non-traceable leaves.**
   - `payload["wl"]["transform"]` is a string (`"linear"` / `"log"`) that
     selects a Python branch in `trace_beam`.
   - The polynomial dimensions (`map_i`, `crv_j`, `ids_i`, …) and the PSF
     `oversample` are ints that set array shapes, e.g. `jnp.arange(map_i)`.

   A traced value can do neither job.
2. **Non-hashable dicts.** Dicts of arrays cannot be `static_argnums`.

## The pattern: split arrays from structure

`roman_disperser.jit_payload` splits each payload into two parts and drops
the labels:

| Part | Contents | Passed to jit as |
|---|---|---|
| `arrays` | every array leaf (PSF grid, wavelengths, spatial grid, polynomial coefficients, detector constants) | ordinary (traced) argument — a pytree |
| `static` | a whitelisted tuple of the ints/strings the traced code reads (`PSF_STATIC_KEYS`, `OPTICAL_STATIC_KEYS`) | `static_argnames` (hashable) |
| labels | `detector`, `order`, `stpsf_filter`, `timing`, `pixel_scale`, … | dropped |

The traced code calls `jit_payload.merge_payloads(arrays, static)` and then
indexes the payloads exactly as before.

```python
from roman_disperser import jit_payload, star_disperser

arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
out = star_disperser.disperse_star_jit(
    arrays, x, y, wavelengths, flux, out, static=static, chunk_size=2000)
```

The arrays have the same shapes on every SCA, and `static` is the same for
every SCA of a dispersing element. **One compiled program therefore serves
all 18 SCAs and every pointing.** The persistent-cache entries are small,
because the PSF grid is no longer embedded in them.

`static` is a whitelist on purpose. A generic rule ("keep every non-array
leaf static") would key the cache on `detector` and silently recompile per
SCA. With the whitelist, a missing key fails loudly at trace time
(`KeyError`).

## Public entry points

You normally never call `jit_payload` yourself; the factories do it:

- `star_disperser.make_star_disperser(psf, opt)` /
  `galaxy_disperser.make_galaxy_disperser(psf, opt)`
  - Same signatures as before.
  - The returned function calls the module-level jitted
    `disperse_star_jit` / `disperse_galaxy_jit`.
  - It carries `.payload_args = (arrays, static, chunk_size)`.
- `pipeline.make_batched_star_fori` / `make_batched_galaxy_fori`
  - When given a factory-built disperser, they route to
    `pipeline.run_star_batch` / `run_galaxy_batch`.
  - Those take the payload arrays, the sensitivity curve, the wavelengths
    and `dlam` as arguments.
  - Any other callable (no `.payload_args`) raises `TypeError`. Until
    v0.16.0 it silently fell back to the closure path (correct, but one
    compile per SCA); that was removed so the slow path cannot be hit by
    accident. To loop a custom disperser, write the `jax.jit(fori_loop)`
    yourself, as `benchmarks/bench_deposit.py` does for its noscatter
    reference.

### Pitfall: wrapping a disperser in your own `jax.jit`

Calling a factory-built disperser **inside your own jitted function** works.
But the payload arrays it holds are then closed over by *your* function, so
they become constants of *your* program again: one compile per SCA. For
loops over many sources, use `pipeline.make_batched_*_fori` (or
`run_*_batch`) instead of hand-rolling `jax.jit(fori_loop(... disperser
...))`.

## Multi-source loops

`run_star_batch` / `run_galaxy_batch` run a `jax.lax.fori_loop` over a
padded batch. `n_sources` is a traced argument, so one compile also serves
every batch fill level (see the "Multi-Source fori_loop Pattern" in
CLAUDE.md).

## History: why it used to be closures

Until 2026-10 the payloads were captured in a closure before `@jax.jit`.
That was the simplest way around the two problems above. It turns the
captured arrays into **compile-time constants**: each (SCA, order) was a
distinct XLA program.

This was judged harmless when it was written: the legacy CPU disperser
compiled in ~0.5 s. By the 2026-10 production runs it cost far more:

- 6 programs per SCA (3 orders × star/galaxy), at ~30–70 s per SCA on A10G.
- Each persistent-cache entry was ~111 MB, because it carried its own PSF
  copy.
- Array 10117 measured ~74 s/SCA of compile (29% of wall time).
- The HEALPix A/B (job 12131) measured 601 s per pointing (35% of wall
  time), once SED reads were fixed.

See the research log thread `jit-payload-args`.

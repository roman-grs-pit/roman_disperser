"""Compile-count + equivalence probe for payload-as-argument JIT.

For SCAs given on the command line (order "1", GRISM), build the batched
star and galaxy loops two ways and time the first call on each SCA:

  new : make_batched_*_fori(make_*_disperser(...))  -> run_*_batch, payload args
  old : same, but the disperser is wrapped in a plain lambda (no
        .payload_args), forcing the pre-change closure path (payload baked in)

Reports per-SCA first-call (compile) time, steady-state call time, the
number of compiled programs in the new path's jit cache, and max |new-old|
relative to the image sum. Run with the persistent compilation cache OFF so
the timings are real compiles.

Usage: pixi run python workbench/20261011-jit-payload-args/compile_probe.py 1 2 3
"""
import os
import sys
import time

os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_compilation_cache", False)

from roman_disperser import paths, psf_model, star_disperser, galaxy_disperser
from roman_disperser import optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.pipeline import (make_batched_star_fori,
                                      make_batched_galaxy_fori,
                                      run_star_batch, run_galaxy_batch)

ORDER = "1"
N_WL = int(os.environ.get("PROBE_NWL", 2000))
BATCH = 4
NPIX = 64  # galaxy stamp at 4x oversampling
model = RomanOpticalModel(str(paths.optical_model_path()))
psf_dir = paths.data_dir() / "psf_cache"
wl = jnp.linspace(1.0, 1.9, N_WL, dtype=jnp.float32)
dlam = float((wl[1] - wl[0]) * 1e4)
rng = np.random.default_rng(0)
spec = jnp.asarray(rng.uniform(0.5, 1.5, (BATCH, N_WL)), jnp.float32) * 1e-3
xs = jnp.asarray([1000., 2000., 3000., 1500.], jnp.float32)
ys = jnp.asarray([1200., 2100., 2900., 3300.], jnp.float32)
yy, xx = np.mgrid[:NPIX, :NPIX] - (NPIX - 1) / 2
img = np.exp(-(xx**2 + yy**2) / (2 * 6.0**2)).astype(np.float32)
imgs = jnp.asarray(np.broadcast_to(img / img.sum(), (BATCH, NPIX, NPIX)).copy())
zero = jnp.zeros((4088, 4088), jnp.float32)


def timed(f, *a):
    t = time.time()
    out = f(*a).block_until_ready()
    return out, time.time() - t


print(f"backend={jax.default_backend()} N_WL={N_WL} batch={BATCH}")
for sca in [int(s) for s in sys.argv[1:]]:
    psf = psf_model.get_or_make_psf_payload(
        detector=f"WFI{sca:02d}", order=ORDER, cache_dir=psf_dir, verbose=False)
    opt = omj.make_sca_payload(model, sca=sca, order=ORDER)
    sens = jnp.ones(N_WL, jnp.float32) * (0.5 + 0.1 * sca)

    sd = star_disperser.make_star_disperser(psf, opt)
    gd = galaxy_disperser.make_galaxy_disperser(psf, opt)
    s_new = make_batched_star_fori(sd, sens, wl, dlam)
    g_new = make_batched_galaxy_fori(gd, sens, wl, dlam)
    s_old = make_batched_star_fori(lambda *a: sd(*a), sens, wl, dlam)
    g_old = make_batched_galaxy_fori(lambda *a: gd(*a), sens, wl, dlam)

    row = [f"SCA{sca:02d}"]
    outs = {}
    for name, f, args in [
        ("star_new", s_new, (BATCH, spec, xs, ys, zero)),
        ("star_old", s_old, (BATCH, spec, xs, ys, zero)),
        ("gal_new", g_new, (BATCH, spec, xs, ys, imgs, zero)),
        ("gal_old", g_old, (BATCH, spec, xs, ys, imgs, zero)),
    ]:
        out, t1 = timed(f, *args)
        _, t2 = timed(f, *args)
        outs[name] = np.asarray(out)
        row.append(f"{name}: first {t1:6.1f}s steady {t2:6.2f}s")
    for k in ("star", "gal"):
        a, b = outs[f"{k}_new"], outs[f"{k}_old"]
        row.append(f"{k} max|new-old|/sum = {np.abs(a - b).max() / b.sum():.2e}"
                   f" sum rel {abs(a.sum() - b.sum()) / b.sum():.2e}")
    print("\n  ".join(row), flush=True)

print("compiled programs in cache: run_star_batch",
      run_star_batch._cache_size(), " run_galaxy_batch",
      run_galaxy_batch._cache_size())

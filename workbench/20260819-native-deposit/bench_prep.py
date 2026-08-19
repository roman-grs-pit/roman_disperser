"""Decompose the prepare_galaxy_images floor (issue #30, follow-on to job 7144).

Job 7144 put the native16 per-galaxy floor at the *prepare* stage
(~3.0 ms/gal on a10g vs ~0.5 binning + ~1.9 interp+deposit). This script
splits prepare into its four sub-stages and benchmarks structural
variants that give the same answer up to floating-point regrouping —
no algorithmic changes.

Sub-stages (each jitted separately, materialized boundaries, so readings
are per-stage upper bounds — stage boundaries block the cross-stage
fusion the real fused prep gets):

1. ``psf_interp`` : bilinear spatial interpolation of the PSF grid at
                    (x0, y0) -> [56, P, P].
2. ``jac``        : vmapped trace_beam_sca_with_jacobian over the 56
                    grid wavelengths (autodiff Jacobian).
3. ``warp``       : vmapped bilinear forward-scatter warp of the galaxy
                    image through each Jacobian -> [56, N, N].
4. ``conv_fft``   : vmapped jax.scipy.signal.fftconvolve(mode='full')
                    -> [56, X, X], X = N + P - 1.

Structural variants (each equivalence-gated against its baseline):

- ``warp_fused``  : the warp's 4 scatter-adds concatenated into ONE
                    scatter-add. Same additions, different grouping.
- ``conv_padS``   : manual rfft2/irfft2 at FFT size S >= X, cropped to
                    [:X, :X]. jax's fftconvolve transforms at the exact
                    full shape (fft_shape = full_shape, with a literal
                    TODO(jakevdp) about next_fast_len in jax 0.7.2);
                    X = 303 = 3*101 has a large prime factor, which is a
                    worst case for cuFFT. Circular conv at S >= X equals
                    the linear conv on the first X samples (no
                    wraparound: the linear support is exactly X), so
                    this is the same answer up to fp rounding.
- ``conv_pre``    : PSF-grid FFTs precomputed once per (SCA, order) as
                    closure constants; the bilinear *spatial*
                    interpolation is applied to the complex spectra
                    instead of the real PSFs. FFT is linear, so
                    interpolation and FFT commute exactly (fp rounding
                    aside): this removes 56 forward PSF FFTs per galaxy.
                    Cost: the precomputed spectra are
                    [4, 4, 56, S, S//2+1] complex64 (~370 MB at S=320) —
                    resident per (SCA, order-filter).
- ``prep_fast``   : fused end-to-end prepare = jac + warp + conv_pre,
                    directly comparable to ``prep_base``
                    (= galaxy_disperser.prepare_galaxy_images).
- ``prep_*_bB``   : vmap of prep over B galaxies (per-galaxy amortized
                    time). Same per-galaxy computation; batching only
                    raises GPU occupancy for these small kernels.

Usage (from the repo root, cuda env):
    pixi run -e cuda python workbench/20260819-native-deposit/bench_prep.py \
        --out results/a10g/prep.json
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from roman_disperser import elements, galaxy_disperser, psf_model, sersic
from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj
from roman_disperser.pipeline import resolve_paths


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "none"
    except Exception:
        return "unavailable"


def warp_fused(image, jacobian, dx, dy):
    """disperse_galaxy_shape with the 4 scatter-adds fused into one.

    Identical additions to galaxy_disperser.disperse_galaxy_shape,
    regrouped into a single scatter (fp summation order differs).
    """
    Ny, Nx = image.shape
    j_idx = jnp.arange(Nx)
    i_idx = jnp.arange(Ny)
    ii, jj = jnp.meshgrid(i_idx, j_idx, indexing='ij')
    rel_x = (jj - (Nx - 1) / 2.0) * dx
    rel_y = (ii - (Ny - 1) / 2.0) * dy
    warped_x = jacobian[0, 0] * rel_x + jacobian[0, 1] * rel_y
    warped_y = jacobian[1, 0] * rel_x + jacobian[1, 1] * rel_y
    out_j = warped_x / dx + (Nx - 1) / 2.0
    out_i = warped_y / dy + (Ny - 1) / 2.0
    out_j_floor = jnp.floor(out_j).astype(jnp.int32)
    out_i_floor = jnp.floor(out_i).astype(jnp.int32)
    fj = out_j - out_j_floor
    fi = out_i - out_i_floor

    w00 = ((1 - fj) * (1 - fi)).ravel()
    w10 = (fj * (1 - fi)).ravel()
    w01 = ((1 - fj) * fi).ravel()
    w11 = (fj * fi).ravel()
    flux = image.ravel()
    idx_j = out_j_floor.ravel()
    idx_i = out_i_floor.ravel()

    all_i = jnp.concatenate([idx_i, idx_i, idx_i + 1, idx_i + 1])
    all_j = jnp.concatenate([idx_j, idx_j + 1, idx_j, idx_j + 1])
    all_v = jnp.concatenate([flux * w00, flux * w10, flux * w01, flux * w11])
    warped = jnp.zeros_like(image)
    return warped.at[all_i, all_j].add(
        all_v, mode="drop", wrap_negative_indices=False)


def make_fns(psf_payload, optical_payload, oversample, npix_os,
             fast_sizes, precomp_size, batch_sizes):
    """Build all jitted stage/variant functions for one order."""
    dx = dy = 1.0 / oversample
    grid_wl = psf_payload['wavelengths']
    P = int(psf_payload['psf_fov_pixels'])
    X = npix_os + P - 1

    fns = {}

    # --- sub-stages of the baseline prepare ---
    @jax.jit
    def psf_interp_fn(x0, y0):
        return psf_model.interpolate_psf_spatial(psf_payload, x0, y0)

    @jax.jit
    def jac_fn(x0, y0):
        def one(wl):
            return galaxy_disperser.trace_beam_sca_with_jacobian(
                optical_payload, x0, y0, wl)
        return jax.vmap(one)(grid_wl)

    @jax.jit
    def warp_fn(image, jacobians):
        return jax.vmap(
            lambda J: galaxy_disperser.disperse_galaxy_shape(image, J, dx, dy)
        )(jacobians)

    @jax.jit
    def warp_fused_fn(image, jacobians):
        return jax.vmap(lambda J: warp_fused(image, J, dx, dy))(jacobians)

    @jax.jit
    def conv_fft_fn(warped, psfs):
        return jax.vmap(
            lambda w, p: jax.scipy.signal.fftconvolve(w, p, mode='full')
        )(warped, psfs)

    fns.update(psf_interp=psf_interp_fn, jac=jac_fn, warp=warp_fn,
               warp_fused=warp_fused_fn, conv_fft=conv_fft_fn)

    # --- padded-FFT convolution at each candidate size ---
    def conv_pad(warped, psfs, S):
        sp1 = jnp.fft.rfft2(warped, s=(S, S))
        sp2 = jnp.fft.rfft2(psfs, s=(S, S))
        return jnp.fft.irfft2(sp1 * sp2, s=(S, S))[..., :X, :X]

    for S in fast_sizes:
        fns[f"conv_pad{S}"] = jax.jit(
            lambda warped, psfs, S=S: conv_pad(warped, psfs, S))

    # --- precomputed PSF-grid FFTs, spatial interp in Fourier domain ---
    S0 = precomp_size
    # [N_y, N_x, N_wl, S0, S0//2+1] complex64; computed once, closed over.
    fft_grid = jnp.fft.rfft2(jnp.asarray(psf_payload['psf_grid']),
                             s=(S0, S0))
    fft_payload = dict(psf_payload, psf_grid=fft_grid)

    def conv_pre(warped, x0, y0):
        # bilinear spatial interp of the complex spectra: same weights and
        # lerp as interpolate_psf_spatial, commutes with the (linear) FFT
        sp2 = psf_model.interpolate_psf_spatial(fft_payload, x0, y0)
        sp1 = jnp.fft.rfft2(warped, s=(S0, S0))
        return jnp.fft.irfft2(sp1 * sp2, s=(S0, S0))[..., :X, :X]

    fns["conv_pre"] = jax.jit(conv_pre)

    # --- end-to-end prepare: baseline and fast ---
    def prep_base(image, x0, y0):
        convolved, cx, cy, _ = galaxy_disperser.prepare_galaxy_images(
            optical_payload, psf_payload, image, x0, y0, dx, dy)
        return convolved, cx, cy

    fns["prep_base"] = jax.jit(prep_base)

    def prep_fast(image, x0, y0):
        cx, cy, jacobians = jax.vmap(
            lambda wl: galaxy_disperser.trace_beam_sca_with_jacobian(
                optical_payload, x0, y0, wl))(grid_wl)
        warped = jax.vmap(
            lambda J: galaxy_disperser.disperse_galaxy_shape(image, J, dx, dy)
        )(jacobians)
        return conv_pre(warped, x0, y0), cx, cy

    fns["prep_fast"] = jax.jit(prep_fast)

    # --- galaxy-batched prepare (per-galaxy amortized) ---
    for B in batch_sizes:
        fns[f"prep_base_b{B}"] = jax.jit(jax.vmap(prep_base))
        fns[f"prep_fast_b{B}"] = jax.jit(jax.vmap(prep_fast))

    return fns, X, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sca", type=int, default=1)
    ap.add_argument("--orders", default="1")
    ap.add_argument("--n-gal", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--fast-sizes", default="303,315,320,324")
    ap.add_argument("--precomp-size", type=int, default=320)
    ap.add_argument("--batch-sizes", default="4,16")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    orders = args.orders.split(",")
    fast_sizes = [int(s) for s in args.fast_sizes.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    rng = np.random.default_rng(args.seed)

    meta = {
        "tag": args.tag, "jax": jax.__version__,
        "backend": jax.default_backend(), "device": str(jax.devices()[0]),
        "gpu": gpu_name(), "host": platform.node(),
        "n_gal": args.n_gal, "seed": args.seed, "sca": args.sca,
        "fast_sizes": fast_sizes, "precomp_size": args.precomp_size,
        "batch_sizes": batch_sizes,
    }
    print(json.dumps(meta, indent=1))

    element = elements.get_element(None)
    catalog_dir, sensitivity_dir, model_path, psf_cache_dir = resolve_paths()
    model = RomanOpticalModel(str(model_path))
    detector_name = f"WFI{args.sca:02d}"

    results = []
    for order in orders:
        fname = element.stpsf_filters[order]
        psf_payload = psf_model.get_or_make_psf_payload(
            detector=detector_name, order=order,
            wavelengths=elements.psf_cache_wavelengths(element),
            stpsf_filter=fname, cache_dir=psf_cache_dir, verbose=False)
        optical_payload = omj.make_sca_payload(model, sca=args.sca,
                                               order=order)
        oversample = int(psf_payload["oversample"])
        npix_os = 30 * oversample

        fns, X, P = make_fns(psf_payload, optical_payload, oversample,
                             npix_os, fast_sizes, args.precomp_size,
                             batch_sizes)
        print(f"order {order}: npix_os={npix_os}, psf={P}, X={X}")

        n = args.n_gal
        x_gal = rng.uniform(400, 3688, n).astype(np.float32)
        y_gal = rng.uniform(400, 3688, n).astype(np.float32)
        r_eff = sersic.catalog_r_eff_to_pixels(
            jnp.array(np.exp(rng.normal(np.log(0.25), 0.5, n)),
                      dtype=jnp.float32),
            oversample=oversample)
        n_ser = jnp.array(rng.uniform(1.0, 4.0, n), dtype=jnp.float32)
        ba = jnp.array(rng.uniform(0.3, 1.0, n), dtype=jnp.float32)
        theta = jnp.array(rng.uniform(0, np.pi, n), dtype=jnp.float32)
        images = sersic.make_sersic_images(r_eff, n_ser, ba, theta, npix_os)
        images.block_until_ready()
        x_gal_j = jnp.asarray(x_gal)
        y_gal_j = jnp.asarray(y_gal)

        # --- equivalence gates on galaxy 0 ---
        im0, x0, y0 = images[0], x_gal_j[0], y_gal_j[0]
        psfs0 = fns["psf_interp"](x0, y0)
        _, _, jac0 = fns["jac"](x0, y0)
        warped0 = fns["warp"](im0, jac0)
        conv0 = fns["conv_fft"](warped0, psfs0)
        ref = np.array(conv0)

        def gate(name, val, refv, rtol=1e-5):
            val = np.array(val)
            refv = np.array(refv)
            a = 1e-5 * max(float(np.abs(refv).max()), 1e-30)
            ok = np.allclose(val, refv, rtol=rtol, atol=a)
            mad = float(np.abs(val - refv).max())
            print(f"  gate {name}: max_abs={mad:.3e} allclose={ok}")
            results.append({"order": order, "check": name,
                            "max_abs_diff": mad, "allclose": bool(ok)})
            return ok

        gate("warp_fused_vs_warp", fns["warp_fused"](im0, jac0), warped0)
        for S in fast_sizes:
            gate(f"conv_pad{S}_vs_fft", fns[f"conv_pad{S}"](warped0, psfs0),
                 ref)
        gate("conv_pre_vs_fft", fns["conv_pre"](warped0, x0, y0), ref)
        pf, pcx, pcy = fns["prep_fast"](im0, x0, y0)
        pb, bcx, bcy = fns["prep_base"](im0, x0, y0)
        gate("prep_fast_vs_base", pf, pb)
        gate("prep_fast_centers", np.stack([pcx, pcy]),
             np.stack([bcx, bcy]))

        # --- warmup / compile ---
        singles = (["psf_interp", "jac", "warp", "warp_fused", "conv_fft",
                    "conv_pre", "prep_base", "prep_fast"]
                   + [f"conv_pad{S}" for S in fast_sizes])
        t_compile = {}
        arg_for = {
            "psf_interp": lambda i: (x_gal_j[i], y_gal_j[i]),
            "jac": lambda i: (x_gal_j[i], y_gal_j[i]),
            "warp": lambda i: (images[i], jac0),
            "warp_fused": lambda i: (images[i], jac0),
            "conv_fft": lambda i: (warped0, psfs0),
            "conv_pre": lambda i: (warped0, x_gal_j[i], y_gal_j[i]),
            "prep_base": lambda i: (images[i], x_gal_j[i], y_gal_j[i]),
            "prep_fast": lambda i: (images[i], x_gal_j[i], y_gal_j[i]),
        }
        for S in fast_sizes:
            arg_for[f"conv_pad{S}"] = lambda i: (warped0, psfs0)
        for name in singles:
            t0 = time.time()
            out = fns[name](*arg_for[name](0))
            jax.block_until_ready(out)
            t_compile[name] = time.time() - t0
        for B in batch_sizes:
            for kind in ("prep_base", "prep_fast"):
                name = f"{kind}_b{B}"
                t0 = time.time()
                out = fns[name](images[:B], x_gal_j[:B], y_gal_j[:B])
                jax.block_until_ready(out)
                t_compile[name] = time.time() - t0
        print("  compile s: " + " ".join(
            f"{k}={v:.2f}" for k, v in t_compile.items()))

        # --- timed loops ---
        for rep in range(args.repeats):
            times = {k: 0.0 for k in t_compile}
            # per-galaxy stage inputs: reuse each galaxy's own upstream
            # outputs so stage inputs vary realistically
            for i in range(n):
                psfs_i = fns["psf_interp"](x_gal_j[i], y_gal_j[i])
                _, _, jac_i = fns["jac"](x_gal_j[i], y_gal_j[i])
                warped_i = fns["warp"](images[i], jac_i)
                jax.block_until_ready((psfs_i, jac_i, warped_i))

                stage_args = {
                    "psf_interp": (x_gal_j[i], y_gal_j[i]),
                    "jac": (x_gal_j[i], y_gal_j[i]),
                    "warp": (images[i], jac_i),
                    "warp_fused": (images[i], jac_i),
                    "conv_fft": (warped_i, psfs_i),
                    "conv_pre": (warped_i, x_gal_j[i], y_gal_j[i]),
                    "prep_base": (images[i], x_gal_j[i], y_gal_j[i]),
                    "prep_fast": (images[i], x_gal_j[i], y_gal_j[i]),
                }
                for S in fast_sizes:
                    stage_args[f"conv_pad{S}"] = (warped_i, psfs_i)
                for name in singles:
                    t0 = time.time()
                    out = fns[name](*stage_args[name])
                    jax.block_until_ready(out)
                    times[name] += time.time() - t0
            for B in batch_sizes:
                for kind in ("prep_base", "prep_fast"):
                    name = f"{kind}_b{B}"
                    t0 = time.time()
                    for s in range(0, n - B + 1, B):
                        out = fns[name](images[s:s + B],
                                        x_gal_j[s:s + B], y_gal_j[s:s + B])
                        jax.block_until_ready(out)
                    times[name] += time.time() - t0
            ms = {k: 1e3 * v / n for k, v in times.items()}
            print(f"  rep {rep}: " + " ".join(
                f"{k}={v:.3f}" for k, v in ms.items()) + " ms/gal")
            results.append({"order": order, "rep": rep, "ms_per_gal": ms,
                            "compile_s": t_compile})

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"meta": meta, "results": results},
                                   indent=1))
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()

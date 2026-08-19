"""Per-stage decomposition of the native16 deposit floor (issue #30).

The 2026-08-19 a10g benchmark (job 7143) measured native16 end-to-end at
~3.85 ms/gal/order vs baseline 9-10 ms. This script splits that floor into
its stages, each timed as a separate jitted function with materialized
inputs/outputs (block_until_ready between stages):

1. ``prep``      : prepare_galaxy_images — 56 Jacobian warps + FFT
                   convolutions -> [56, X, X] convolved stack (X ~ 303).
2. ``bin_joint`` : the prototype's 16-phase pre-binning — 16 independent
                   pad -> reshape -> sum(axes 2,4) passes (~370 MB moved).
3. ``bin_sep``   : separable rewrite — bin y per y-phase (4 passes), then
                   x per x-phase on the y-binned stack (16 cheap passes);
                   ~165 MB moved. Same additions regrouped (f32
                   summation-order differences only) — checked vs
                   bin_joint with allclose.
4. ``interp``    : the wavelength scan WITHOUT the scatter — trace eval,
                   per-wavelength phase/base-index math, the
                   [p_y, p_x, i0] gathers, lerp, flux scale — consumed by
                   an iota-weighted einsum reduction so XLA cannot fold
                   the gathers away.
5. ``deposit``   : the full scan including the native scatter-add into
                   the detector. scatter cost ~ deposit - interp.

Caveats: stage boundaries prevent the cross-stage fusion the real
native16 path gets, so the stage sum will exceed the fused 3.85 ms;
readings are per-stage upper bounds and identify the dominant term, not a
reconstruction of the fused time. The interp/deposit stages are run at
each --chunk-sizes value (scan/dynamic-slice overhead is now a larger
fraction of the smaller per-chunk work).

Usage (from the repo root, cuda env):
    pixi run -e cuda python workbench/20260819-native-deposit/bench_stages.py \
        --out results/a10g/stages.json
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

from roman_disperser import elements, galaxy_disperser, psf_model, sersic, star_disperser
from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj
from roman_disperser.pipeline import (
    DETECTOR_SIZE,
    load_sensitivities,
    resolve_paths,
)


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "none"
    except Exception:
        return "unavailable"


def make_stage_fns(psf_payload, optical_payload, n_wl_padded, chunk_size,
                   X, oversample):
    """Build the five jitted stage functions for one (order, chunk_size)."""
    os_ = oversample
    n_native = (X - 2) // os_ + 2
    rel0 = -(X - 1) / (2.0 * os_)
    grid_wl = psf_payload['wavelengths']
    n_chunks = n_wl_padded // chunk_size
    dx = dy = 1.0 / os_

    @jax.jit
    def prep_fn(image, x0, y0):
        convolved, _, _, _ = galaxy_disperser.prepare_galaxy_images(
            optical_payload, psf_payload, image, x0, y0, dx, dy)
        return convolved

    @jax.jit
    def bin_joint_fn(convolved):
        def bin_phase(py, px):
            padded = jnp.pad(
                convolved,
                ((0, 0),
                 (py, os_ * n_native - X - py),
                 (px, os_ * n_native - X - px)))
            return padded.reshape(
                convolved.shape[0], n_native, os_, n_native, os_
            ).sum(axis=(2, 4))
        return jnp.stack([
            jnp.stack([bin_phase(py, px) for px in range(os_)])
            for py in range(os_)
        ])  # [os, os, N_grid, n, n]

    @jax.jit
    def bin_sep_fn(convolved):
        n_grid = convolved.shape[0]
        ybinned = []
        for py in range(os_):
            padded = jnp.pad(
                convolved, ((0, 0), (py, os_ * n_native - X - py), (0, 0)))
            ybinned.append(padded.reshape(
                n_grid, n_native, os_, X).sum(axis=2))  # [N_grid, n, X]
        out = []
        for py in range(os_):
            row = []
            for px in range(os_):
                padded = jnp.pad(
                    ybinned[py],
                    ((0, 0), (0, 0), (px, os_ * n_native - X - px)))
                row.append(padded.reshape(
                    n_grid, n_native, n_native, os_).sum(axis=3))
            out.append(jnp.stack(row))
        return jnp.stack(out)

    # Iota weights: force every gathered/interpolated element to be
    # computed (a plain sum could in principle be folded into per-image
    # sums, since binning/interp are linear).
    wy = 1.0 + 1e-3 * jnp.arange(n_native, dtype=jnp.float32)
    wx = 1.0 + 7e-4 * jnp.arange(n_native, dtype=jnp.float32)

    def scan_setup(x0, y0, wavelengths_padded):
        xsca_disp, ysca_disp = star_disperser._compute_dispersed_positions(
            optical_payload, x0, y0, wavelengths_padded)
        u_x = xsca_disp - 0.5 + rel0
        u_y = ysca_disp - 0.5 + rel0
        return u_x, u_y

    def chunk_values(binned, u_x, u_y, wl_chunk, flux_chunk):
        m_x = jnp.floor(u_x).astype(jnp.int32)
        m_y = jnp.floor(u_y).astype(jnp.int32)
        p_x = jnp.clip(jnp.floor((u_x - m_x) * os_), 0, os_ - 1
                       ).astype(jnp.int32)
        p_y = jnp.clip(jnp.floor((u_y - m_y) * os_), 0, os_ - 1
                       ).astype(jnp.int32)
        i0 = jnp.clip(jnp.searchsorted(grid_wl, wl_chunk) - 1,
                      0, len(grid_wl) - 2)
        t = (wl_chunk - grid_wl[i0]) / (grid_wl[i0 + 1] - grid_wl[i0])
        t = jnp.clip(t, 0.0, 1.0)
        lo = binned[p_y, p_x, i0]
        hi = binned[p_y, p_x, i0 + 1]
        nat = lo + t[:, None, None] * (hi - lo)
        return nat * flux_chunk[:, None, None], m_x, m_y

    @jax.jit
    def interp_fn(binned, x0, y0, wavelengths_padded, counts_padded):
        u_x, u_y = scan_setup(x0, y0, wavelengths_padded)

        def body(acc, ci):
            s = ci * chunk_size
            wl_c = jax.lax.dynamic_slice(wavelengths_padded, [s], [chunk_size])
            fl_c = jax.lax.dynamic_slice(counts_padded, [s], [chunk_size])
            ux_c = jax.lax.dynamic_slice(u_x, [s], [chunk_size])
            uy_c = jax.lax.dynamic_slice(u_y, [s], [chunk_size])
            nat, _, _ = chunk_values(binned, ux_c, uy_c, wl_c, fl_c)
            acc = acc + jnp.einsum('cyx,y,x->', nat, wy, wx)
            return acc, None

        acc, _ = jax.lax.scan(body, jnp.float32(0.0), jnp.arange(n_chunks))
        return acc

    @jax.jit
    def deposit_fn(binned, x0, y0, wavelengths_padded, counts_padded, output):
        u_x, u_y = scan_setup(x0, y0, wavelengths_padded)
        k = jnp.arange(n_native)

        def body(out_c, ci):
            s = ci * chunk_size
            wl_c = jax.lax.dynamic_slice(wavelengths_padded, [s], [chunk_size])
            fl_c = jax.lax.dynamic_slice(counts_padded, [s], [chunk_size])
            ux_c = jax.lax.dynamic_slice(u_x, [s], [chunk_size])
            uy_c = jax.lax.dynamic_slice(u_y, [s], [chunk_size])
            nat, m_x, m_y = chunk_values(binned, ux_c, uy_c, wl_c, fl_c)
            idx_y = m_y[:, None, None] + k[None, :, None]
            idx_x = m_x[:, None, None] + k[None, None, :]
            idx_y = jnp.broadcast_to(idx_y, nat.shape)
            idx_x = jnp.broadcast_to(idx_x, nat.shape)
            out_c = out_c.at[idx_y.ravel(), idx_x.ravel()].add(
                nat.ravel(), mode="drop", wrap_negative_indices=False)
            return out_c, None

        output, _ = jax.lax.scan(body, output, jnp.arange(n_chunks))
        return output

    return prep_fn, bin_joint_fn, bin_sep_fn, interp_fn, deposit_fn, n_native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sca", type=int, default=1)
    ap.add_argument("--orders", default="1")
    ap.add_argument("--n-gal", type=int, default=30)
    ap.add_argument("--chunk-sizes", default="500,2000")
    ap.add_argument("--dlam", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    orders = args.orders.split(",")
    chunk_sizes = [int(c) for c in args.chunk_sizes.split(",")]
    rng = np.random.default_rng(args.seed)

    meta = {
        "tag": args.tag, "jax": jax.__version__,
        "backend": jax.default_backend(), "device": str(jax.devices()[0]),
        "gpu": gpu_name(), "host": platform.node(),
        "n_gal": args.n_gal, "chunk_sizes": chunk_sizes,
        "dlam_A": args.dlam, "seed": args.seed, "sca": args.sca,
    }
    print(json.dumps(meta, indent=1))

    element = elements.get_element(None)
    catalog_dir, sensitivity_dir, model_path, psf_cache_dir = resolve_paths()
    model = RomanOpticalModel(str(model_path))

    wl_ang = np.arange(element.lam_min * 1e4, element.lam_max * 1e4 + 0.1,
                       args.dlam)
    wl_um = (wl_ang / 1e4).astype(np.float32)
    n_wl = len(wl_um)
    print(f"wavelengths: {n_wl} samples, {args.dlam} A")
    sens = load_sensitivities(sensitivity_dir, args.sca, wl_um, orders)

    detector_name = f"WFI{args.sca:02d}"
    payloads_by_filter = {}
    psf_payloads = {}
    optical_payloads = {}
    for order in orders:
        fname = element.stpsf_filters[order]
        if fname not in payloads_by_filter:
            payloads_by_filter[fname] = psf_model.get_or_make_psf_payload(
                detector=detector_name, order=order,
                wavelengths=elements.psf_cache_wavelengths(element),
                stpsf_filter=fname, cache_dir=psf_cache_dir, verbose=False)
        psf_payloads[order] = payloads_by_filter[fname]
        optical_payloads[order] = omj.make_sca_payload(
            model, sca=args.sca, order=order)

    oversample = int(psf_payloads[orders[0]]["oversample"])
    npix_os = 30 * oversample
    psf_dim = int(psf_payloads[orders[0]]["psf_fov_pixels"])
    X = npix_os + psf_dim - 1

    n = args.n_gal
    x_gal = rng.uniform(400, 3688, n).astype(np.float32)
    y_gal = rng.uniform(400, 3688, n).astype(np.float32)
    r_eff = sersic.catalog_r_eff_to_pixels(
        jnp.array(np.exp(rng.normal(np.log(0.25), 0.5, n)), dtype=jnp.float32),
        oversample=oversample)
    n_ser = jnp.array(rng.uniform(1.0, 4.0, n), dtype=jnp.float32)
    ba = jnp.array(rng.uniform(0.3, 1.0, n), dtype=jnp.float32)
    theta = jnp.array(rng.uniform(0, np.pi, n), dtype=jnp.float32)
    images = sersic.make_sersic_images(r_eff, n_ser, ba, theta, npix_os)
    images.block_until_ready()

    sed = (1e-16 * (1.0 + 0.3 * np.sin(wl_um * 20.0))).astype(np.float32)
    spectra = np.tile(sed, (n, 1)) * rng.uniform(0.3, 3.0, (n, 1)).astype(np.float32)

    results = []

    for order in orders:
        max_cs = max(chunk_sizes)
        n_padded = ((n_wl + max_cs - 1) // max_cs) * max_cs
        # pad so every chunk size divides n_padded
        for cs in chunk_sizes:
            n_padded = ((n_padded + cs - 1) // cs) * cs
        wl_padded = jnp.array(np.pad(wl_um, (0, n_padded - n_wl),
                                     constant_values=wl_um[-1]))

        def counts_for(i):
            c = spectra[i] * np.array(sens[order]) * args.dlam
            return jnp.array(np.pad(c.astype(np.float32),
                                    (0, n_padded - n_wl)))

        stage_fns = {}
        for cs in chunk_sizes:
            stage_fns[cs] = make_stage_fns(
                psf_payloads[order], optical_payloads[order],
                n_padded, cs, X, oversample)
        prep_fn, bin_joint_fn, bin_sep_fn = stage_fns[chunk_sizes[0]][:3]
        n_native = stage_fns[chunk_sizes[0]][5]
        print(f"order {order}: X={X}, native stamp {n_native}")

        # --- consistency: joint vs separable binning ---
        conv0 = prep_fn(images[0], x_gal[0], y_gal[0])
        bj = np.array(bin_joint_fn(conv0))
        bs = np.array(bin_sep_fn(conv0))
        ok = np.allclose(bj, bs, rtol=1e-5,
                         atol=1e-5 * max(float(bj.max()), 1e-30))
        print(f"  bin_sep vs bin_joint: max_abs={np.abs(bj-bs).max():.3e} "
              f"allclose={ok}")
        results.append({"order": order, "check": "bin_sep_equiv",
                        "max_abs_diff": float(np.abs(bj - bs).max()),
                        "allclose": bool(ok)})

        # --- warm up (compile) everything on galaxy 0 ---
        t_compile = {}
        t0 = time.time()
        conv0 = prep_fn(images[0], x_gal[0], y_gal[0])
        conv0.block_until_ready(); t_compile["prep"] = time.time() - t0
        t0 = time.time()
        binned0 = bin_joint_fn(conv0)
        binned0.block_until_ready(); t_compile["bin_joint"] = time.time() - t0
        t0 = time.time()
        bin_sep_fn(conv0).block_until_ready()
        t_compile["bin_sep"] = time.time() - t0
        for cs in chunk_sizes:
            _, _, _, interp_fn, deposit_fn, _ = stage_fns[cs]
            t0 = time.time()
            interp_fn(binned0, x_gal[0], y_gal[0], wl_padded,
                      counts_for(0)).block_until_ready()
            t_compile[f"interp_c{cs}"] = time.time() - t0
            out = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
            t0 = time.time()
            deposit_fn(binned0, x_gal[0], y_gal[0], wl_padded,
                       counts_for(0), out).block_until_ready()
            t_compile[f"deposit_c{cs}"] = time.time() - t0
        print("  compile s: " + " ".join(
            f"{k}={v:.1f}" for k, v in t_compile.items()))

        # --- timed loops ---
        for rep in range(args.repeats):
            times = {k: 0.0 for k in t_compile}
            out = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
            for i in range(n):
                counts_i = counts_for(i)
                t0 = time.time()
                conv = prep_fn(images[i], x_gal[i], y_gal[i])
                conv.block_until_ready(); times["prep"] += time.time() - t0
                t0 = time.time()
                binned = bin_joint_fn(conv)
                binned.block_until_ready()
                times["bin_joint"] += time.time() - t0
                t0 = time.time()
                bin_sep_fn(conv).block_until_ready()
                times["bin_sep"] += time.time() - t0
                for cs in chunk_sizes:
                    _, _, _, interp_fn, deposit_fn, _ = stage_fns[cs]
                    t0 = time.time()
                    interp_fn(binned, x_gal[i], y_gal[i], wl_padded,
                              counts_i).block_until_ready()
                    times[f"interp_c{cs}"] += time.time() - t0
                    t0 = time.time()
                    out = deposit_fn(binned, x_gal[i], y_gal[i], wl_padded,
                                     counts_i, out)
                    out.block_until_ready()
                    times[f"deposit_c{cs}"] += time.time() - t0
            ms = {k: 1e3 * v / n for k, v in times.items()}
            print(f"  rep {rep}: " + " ".join(
                f"{k}={v:.2f}" for k, v in ms.items()) + " ms/gal")
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

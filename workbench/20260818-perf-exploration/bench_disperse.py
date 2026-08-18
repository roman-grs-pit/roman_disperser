"""Microbenchmark for the dispersal hot path (galaxy + star deposit).

Purpose (2026-08-18 performance exploration, branch explore/performance):
separate the three candidate costs in ``galaxy_disperser.disperse_galaxy``
per order — (a) non-scatter compute (PSF interp + flux scale + index math),
(b) scatter volume, (c) scatter *collision* structure — and measure the
environment multipliers (allocator env vars) suspected in the slow
20260803_test_catalog run (150/99/19 ms/galaxy vs the historical 7-9 ms flat).

Method: synthetic galaxies (Sérsic stamps at production geometry: 30 px native
= 120 px oversampled, 184^2 4x-oversampled PSFs, 5501-sample 2 A wavelength
grid, batched fori_loop with batch=100 exactly like
``scripts/build_dispersed_image.py``), timed per (order, variant):

- ``baseline``   : the production deposit — ~505M scatter-adds/galaxy into the
                   4088^2 detector (Conv 303^2 x 5501 wavelengths).
- ``noscatter``  : deposit replaced by a scalar reduction added to output[0,0].
                   Keeps interpolation/flux compute, removes the scatter (and
                   lets XLA drop the index math). Measures (a).
- ``spread``     : same scatter volume, but indices decorrelated in y by a
                   per-wavelength stride so deposits spread over the full
                   detector column instead of piling on the trace. Measures
                   scatter without same-address collisions (b vs c).
                   NOT flux-equivalent to baseline (diagnostic only).
- ``local``      : per-galaxy accumulation into a small footprint buffer
                   (sized per order to cover trace span + conv stamp), then
                   one uncollided scatter of the buffer into the detector.
                   Flux-equivalent to baseline up to f32 summation order —
                   checked with allclose (rtol from --rtol) + relative-sum.
                   This is the "restructure the deposit" candidate.

Stars run baseline-only (184^2 x 5501 = ~186M adds/star).

All arrays float32; wavelengths in microns. Timing = wall time around the
batched call with block_until_ready, after a compile+warmup call. Reported
per-source ms is wall / n_sources.

Usage (from the repo root, cuda env):
    pixi run -e cuda python workbench/20260818-perf-exploration/bench_disperse.py \
        --out results/gpu-main.json
Environment-variant runs are driven by run_bench.sh (allocator env vars must
be set before JAX initializes).
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
    disperse_batched_galaxies,
    disperse_batched_stars,
    load_sensitivities,
    make_batched_galaxy_fori,
    make_batched_star_fori,
    resolve_paths,
)


# ---------------------------------------------------------------------------
# Variant galaxy disperser: a copy of galaxy_disperser.disperse_galaxy with a
# switchable deposit stage. Steps 1-4 are identical to the package function
# (same helpers, same order of operations); only Step 5's scatter differs.
# ---------------------------------------------------------------------------

def disperse_galaxy_variant(
    optical_payload, psf_payload, image, x0, y0, spectrum, wavelengths,
    output, chunk_size, mode, buf_shape=None,
):
    oversample = psf_payload['oversample']
    dx = dy = 1.0 / oversample

    convolved, _, _, grid_wl = galaxy_disperser.prepare_galaxy_images(
        optical_payload, psf_payload, image, x0, y0, dx, dy
    )

    conv_shape = convolved.shape[1:]
    rel_y, rel_x = star_disperser.make_psf_pixel_grid(conv_shape, oversample)
    rel_x_flat = rel_x.ravel()
    rel_y_flat = rel_y.ravel()

    xsca_disp, ysca_disp = star_disperser._compute_dispersed_positions(
        optical_payload, x0, y0, wavelengths
    )

    n_wl = len(wavelengths)
    n_padded = ((n_wl + chunk_size - 1) // chunk_size) * chunk_size
    pad_size = n_padded - n_wl
    n_chunks = n_padded // chunk_size

    wavelengths_padded = jnp.pad(wavelengths, (0, pad_size),
                                 constant_values=wavelengths[-1])
    spectrum_padded = jnp.pad(spectrum, (0, pad_size), constant_values=0.0)
    x_disp_padded = jnp.pad(xsca_disp, (0, pad_size),
                            constant_values=xsca_disp[-1])
    y_disp_padded = jnp.pad(ysca_disp, (0, pad_size),
                            constant_values=ysca_disp[-1])

    # 'local' mode: footprint buffer base = min dispersed position minus the
    # conv stamp half-width and margin, so every deposit index is in-buffer.
    if mode == "local":
        conv_half = int(np.ceil(conv_shape[0] / (2 * oversample))) + 2
        base_x = jnp.floor(jnp.min(xsca_disp)).astype(jnp.int32) - conv_half - 2
        base_y = jnp.floor(jnp.min(ysca_disp)).astype(jnp.int32) - conv_half - 2
        buf_iy, buf_ix = jnp.meshgrid(
            jnp.arange(buf_shape[0]), jnp.arange(buf_shape[1]), indexing="ij"
        )

    def process_chunk(carry, chunk_idx):
        output_c = carry
        start = chunk_idx * chunk_size

        wl_chunk = jax.lax.dynamic_slice(wavelengths_padded, [start], [chunk_size])
        flux_chunk = jax.lax.dynamic_slice(spectrum_padded, [start], [chunk_size])
        x_chunk = jax.lax.dynamic_slice(x_disp_padded, [start], [chunk_size])
        y_chunk = jax.lax.dynamic_slice(y_disp_padded, [start], [chunk_size])

        conv_chunk = psf_model.interp_wavelength_chunk(convolved, grid_wl, wl_chunk)
        conv_chunk = conv_chunk * flux_chunk[:, None, None]

        det_x = x_chunk[:, None] + rel_x_flat[None, :]
        det_y = y_chunk[:, None] + rel_y_flat[None, :]

        values = conv_chunk.ravel()

        if mode == "noscatter":
            # Reduction keeps the interp/scale compute; index math is dead.
            output_c = output_c.at[0, 0].add(values.sum())
        elif mode == "spread":
            idx_x = jnp.floor(det_x - 0.5).astype(jnp.int32)
            idx_y = jnp.floor(det_y - 0.5).astype(jnp.int32)
            # Decorrelate: per-wavelength y stride spreads the pile-up over
            # the full column while keeping per-wavelength locality.
            lam_idx = start + jnp.arange(chunk_size)
            idx_y = (idx_y + (lam_idx * 997)[:, None]) % (DETECTOR_SIZE - 8)
            output_c = output_c.at[idx_y.ravel(), idx_x.ravel()].add(
                values, mode="drop", wrap_negative_indices=False)
        elif mode == "local":
            idx_x = jnp.floor(det_x - 0.5).astype(jnp.int32) - base_x
            idx_y = jnp.floor(det_y - 0.5).astype(jnp.int32) - base_y
            output_c = output_c.at[idx_y.ravel(), idx_x.ravel()].add(
                values, mode="drop", wrap_negative_indices=False)
        else:  # baseline — identical to the package deposit
            idx_x = jnp.floor(det_x - 0.5).astype(jnp.int32)
            idx_y = jnp.floor(det_y - 0.5).astype(jnp.int32)
            output_c = output_c.at[idx_y.ravel(), idx_x.ravel()].add(
                values, mode="drop", wrap_negative_indices=False)
        return output_c, None

    if mode == "local":
        buf = jnp.zeros(buf_shape, dtype=jnp.float32)
        buf, _ = jax.lax.scan(process_chunk, buf, jnp.arange(n_chunks))
        # Un-collided flush: every buffer cell has a unique detector target.
        output = output.at[buf_iy + base_y, buf_ix + base_x].add(
            buf.ravel().reshape(buf_shape), mode="drop",
            wrap_negative_indices=False)
        return output
    output, _ = jax.lax.scan(process_chunk, output, jnp.arange(n_chunks))
    return output


def make_variant_disperser(psf_payload, optical_payload, chunk_size, mode,
                           buf_shape=None):
    @jax.jit
    def fn(image, x0, y0, spectrum, wavelengths, output):
        return disperse_galaxy_variant(
            optical_payload, psf_payload, image, x0, y0, spectrum,
            wavelengths, output, chunk_size, mode, buf_shape)
    return fn


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def trace_span(optical_payload, wavelengths_um, x=2044.0, y=2044.0):
    """Host-side trace extent (native px) for buffer sizing."""
    xs, ys = star_disperser._compute_dispersed_positions(
        optical_payload, x, y, jnp.array(wavelengths_um[::25]))
    xs, ys = np.array(xs), np.array(ys)
    return (float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "none"
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sca", type=int, default=1)
    ap.add_argument("--orders", default="0,1,2")
    ap.add_argument("--variants", default="baseline,noscatter,spread,local")
    ap.add_argument("--n-gal", type=int, default=300)
    ap.add_argument("--n-star", type=int, default=300)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--dlam", type=float, default=2.0,
                    help="wavelength spacing in Angstrom (2.0 = production)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--skip-stars", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    orders = args.orders.split(",")
    variants = args.variants.split(",")
    rng = np.random.default_rng(args.seed)

    meta = {
        "tag": args.tag,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "gpu": gpu_name(),
        "host": platform.node(),
        "n_gal": args.n_gal, "n_star": args.n_star, "batch": args.batch,
        "chunk_size": args.chunk_size, "dlam_A": args.dlam,
        "seed": args.seed, "sca": args.sca,
        "env": {k: v for k, v in __import__("os").environ.items()
                if k.startswith(("XLA_", "TF_GPU", "JAX_"))},
    }
    print(json.dumps(meta, indent=1))

    element = elements.get_element(None)  # grism
    catalog_dir, sensitivity_dir, model_path, psf_cache_dir = resolve_paths()
    model = RomanOpticalModel(str(model_path))

    wl_ang = np.arange(element.lam_min * 1e4, element.lam_max * 1e4 + 0.1,
                       args.dlam)
    wl_um = (wl_ang / 1e4).astype(np.float32)
    wl_jax = jnp.array(wl_um)
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

    # Synthetic galaxies: production-like geometry, positions in the interior
    # so full traces stay on-detector for the equivalence check.
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

    # Smooth synthetic SED at production magnitude scale; per-order counts are
    # spectra*sens*dlam inside the fori wrapper, exactly as in the pipeline.
    sed = (1e-16 * (1.0 + 0.3 * np.sin(wl_um * 20.0))).astype(np.float32)
    spectra = np.tile(sed, (n, 1)) * rng.uniform(0.3, 3.0, (n, 1)).astype(np.float32)

    results = []

    for order in orders:
        span_x, span_y = trace_span(optical_payloads[order], wl_um)
        conv_native = int(np.ceil((npix_os + 184 - 1) / oversample))
        buf_y = int(np.ceil(span_y)) + conv_native + 24
        buf_x = int(np.ceil(span_x)) + conv_native + 24
        buf_shape = (buf_y + (-buf_y) % 8, buf_x + (-buf_x) % 8)
        print(f"order {order}: trace span x={span_x:.0f} y={span_y:.0f} px, "
              f"local buf {buf_shape}")

        # Flux-equivalence gate for 'local' vs baseline on a few galaxies.
        if "local" in variants:
            base_fn = make_variant_disperser(
                psf_payloads[order], optical_payloads[order],
                args.chunk_size, "baseline")
            loc_fn = make_variant_disperser(
                psf_payloads[order], optical_payloads[order],
                args.chunk_size, "local", buf_shape)
            out_b = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
            out_l = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
            for i in range(min(3, n)):
                counts = jnp.array(spectra[i]) * sens[order] * args.dlam
                out_b = base_fn(images[i], x_gal[i], y_gal[i], counts, wl_jax, out_b)
                out_l = loc_fn(images[i], x_gal[i], y_gal[i], counts, wl_jax, out_l)
            out_b, out_l = np.array(out_b), np.array(out_l)
            s_b, s_l = out_b.sum(), out_l.sum()
            rel_sum = abs(s_l - s_b) / max(abs(s_b), 1e-30)
            atol = args.rtol * max(out_b.max(), 1e-30)
            ok = np.allclose(out_l, out_b, rtol=args.rtol, atol=atol)
            print(f"  local-vs-baseline: rel_sum_diff={rel_sum:.3e} "
                  f"allclose(rtol={args.rtol})={ok}")
            results.append({"order": order, "check": "local_equiv",
                            "rel_sum_diff": float(rel_sum), "allclose": bool(ok)})
            if not ok:
                print("  WARNING: local variant NOT equivalent — timing still "
                      "reported, but treat 'local' as broken for this order")

        for variant in variants:
            buf = buf_shape if variant == "local" else None
            gd_fn = make_variant_disperser(
                psf_payloads[order], optical_payloads[order],
                args.chunk_size, variant, buf)
            fori = make_batched_galaxy_fori(gd_fn, sens[order], wl_jax, args.dlam)

            warm_out = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
            t0 = time.time()
            fori(1, jnp.zeros((args.batch, n_wl), jnp.float32),
                 jnp.full(args.batch, 2044.0), jnp.full(args.batch, 2044.0),
                 jnp.zeros((args.batch, npix_os, npix_os), jnp.float32),
                 warm_out).block_until_ready()
            t_compile = time.time() - t0

            times = []
            for _ in range(args.repeats):
                out = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
                t0 = time.time()
                out = disperse_batched_galaxies(
                    fori, spectra, x_gal, y_gal, images, out, args.batch)
                times.append(time.time() - t0)
            ms = [1e3 * t / n for t in times]
            print(f"  order {order} {variant:9s}: "
                  f"{' '.join(f'{m:7.2f}' for m in ms)} ms/gal "
                  f"(compile {t_compile:.1f}s)")
            results.append({"order": order, "variant": variant,
                            "ms_per_gal": ms, "compile_s": t_compile})

    if not args.skip_stars:
        ns = args.n_star
        x_st = rng.uniform(400, 3688, ns).astype(np.float32)
        y_st = rng.uniform(400, 3688, ns).astype(np.float32)
        st_spectra = np.tile(sed, (ns, 1)).astype(np.float32)
        for order in orders:
            sd_fn = star_disperser.make_star_disperser(
                psf_payloads[order], optical_payloads[order])
            fori = make_batched_star_fori(sd_fn, sens[order], wl_jax, args.dlam)
            warm = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
            t0 = time.time()
            fori(1, jnp.zeros((1000, n_wl), jnp.float32),
                 jnp.full(1000, 2044.0), jnp.full(1000, 2044.0),
                 warm).block_until_ready()
            t_compile = time.time() - t0
            times = []
            for _ in range(args.repeats):
                out = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), jnp.float32)
                t0 = time.time()
                out = disperse_batched_stars(
                    fori, st_spectra, x_st, y_st, out, 1000)
                times.append(time.time() - t0)
            ms = [1e3 * t / ns for t in times]
            print(f"  order {order} star     : "
                  f"{' '.join(f'{m:7.2f}' for m in ms)} ms/star "
                  f"(compile {t_compile:.1f}s)")
            results.append({"order": order, "variant": "star_baseline",
                            "ms_per_star": ms, "compile_s": t_compile})

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()

"""Performance-regression benchmark for the dispersal hot path.

Times the *production* galaxy disperser (``galaxy_disperser.make_galaxy_disperser``,
the exact code path used by ``scripts/build_dispersed_image.py``) against a
"noscatter" reference on the same synthetic workload, and writes a JSON result
that ``check_perf.py`` gates on.

Why two variants
----------------
Scatter-add performance is what this benchmark exists to police: backend
regressions there (e.g. the jax 0.11.0 GPU scatter-add regression,
jax-ml/jax#39959, ~16x slower) produce correct images slowly. Since the
16-phase native-deposit port the scatter is ~32M elements per galaxy (was
~505M oversampled), but a scatter regression still lands on it. Absolute
times vary between GPUs, so the primary regression signal is the
hardware-insensitive *ratio*

    baseline ms/gal  /  noscatter ms/gal

where ``noscatter`` is the identical computation with the scatter replaced by
a scalar reduction (keeps the 16-phase binning, phase selection, wavelength
interpolation, and flux scaling; XLA drops the dead index math). The healthy
ratio is ~0.73-0.75, *below* 1: the baseline fuses gather+interp+scatter into
one kernel while the reduction variant pays for a separate reduction kernel
(2.26 vs 3.03 ms/gal on an A10G, SLURM 7183, 2026-08-25). A scatter
regression inflates only the baseline, so the ``check_perf.py`` gate at 3.0
still trips — a 0.11.0-class 16x scatter slowdown puts the ratio far above
it. (Pre-native16 history, oversampled deposit: healthy jax 0.7.2/0.10.1/
0.11.1 gave ~1.1-1.2; jax 0.11.0 gave 15-21.)

Workload (matches the 2026-08-18 exploration bench so numbers are comparable):
synthetic Sersic galaxies at production geometry — 30 px native stamps (120 px
at 4x oversample), 184^2 4x-oversampled PSFs, full grism band at 2 A spacing
(5501 wavelength samples), batched fori_loop with batch=100 exactly like the
pipeline script. Seeded; all arrays float32; wavelengths in microns.

Timing: wall time around the batched call with ``block_until_ready``, after a
compile+warmup call; reported per-galaxy ms = wall / n_gal for each repeat.
``check_perf.py`` uses the min over repeats (least-noise estimator).

Requires hydrated reference data (``pixi run hydrate``) and, for GPU numbers,
a CUDA jax. CPU runs work but the ratio gate threshold was chosen from GPU
behaviour; treat CPU results as informational.

Usage (repo root):
    pixi run -e cuda python benchmarks/bench_deposit.py --out results.json
    python benchmarks/check_perf.py results.json
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
    load_sensitivities,
    make_batched_galaxy_fori,
    resolve_paths,
)


def make_noscatter_disperser(psf_payload, optical_payload, chunk_size):
    """Reference disperser: identical to ``disperse_galaxy`` through the
    16-phase binning, phase selection, wavelength interpolation, and flux
    scaling, but the deposit scatter is replaced by a scalar reduction into
    output[0,0]. This is the compute floor the baseline is compared against;
    it is NOT flux-equivalent output (diagnostic only).

    Mirrors the steps of ``star_disperser.deposit_stack_native`` (the shared
    native-resolution hot loop, since the 16-phase pre-binning port), so a
    change in the shared prepare/bin/interp path moves both variants and
    cancels in the ratio. If ``deposit_stack_native`` changes, keep this in
    step or the ratio gate loses its meaning.
    """

    def disperse(image, x0, y0, spectrum, wavelengths, output):
        oversample = psf_payload["oversample"]
        dx = dy = 1.0 / oversample
        os_ = int(oversample)

        convolved, _, _, grid_wl = galaxy_disperser.prepare_galaxy_images(
            optical_payload, psf_payload, image, x0, y0, dx, dy
        )
        xsca_disp, ysca_disp = star_disperser._compute_dispersed_positions(
            optical_payload, x0, y0, wavelengths
        )

        # 16-phase pre-binning, exactly as deposit_stack_native does it.
        n_grid = convolved.shape[0]
        s_y, s_x = convolved.shape[-2:]
        n_y = (s_y - 2) // os_ + 2
        n_x = (s_x - 2) // os_ + 2
        rel0_y = -(s_y - 1) / (2.0 * os_)
        rel0_x = -(s_x - 1) / (2.0 * os_)

        def bin_phase(p_y, p_x):
            padded = jnp.pad(
                convolved,
                ((0, 0),
                 (p_y, os_ * n_y - s_y - p_y),
                 (p_x, os_ * n_x - s_x - p_x)))
            return padded.reshape(n_grid, n_y, os_, n_x, os_).sum(axis=(2, 4))

        binned = jnp.stack([
            jnp.stack([bin_phase(p_y, p_x) for p_x in range(os_)])
            for p_y in range(os_)
        ])

        n_wl = len(wavelengths)
        n_padded = ((n_wl + chunk_size - 1) // chunk_size) * chunk_size
        pad = n_padded - n_wl
        n_chunks = n_padded // chunk_size

        wl_p = jnp.pad(wavelengths, (0, pad), constant_values=wavelengths[-1])
        sp_p = jnp.pad(spectrum, (0, pad), constant_values=0.0)
        x_p = jnp.pad(xsca_disp, (0, pad), constant_values=xsca_disp[-1])
        y_p = jnp.pad(ysca_disp, (0, pad), constant_values=ysca_disp[-1])

        def process_chunk(carry, chunk_idx):
            out = carry
            start = chunk_idx * chunk_size
            wl_c = jax.lax.dynamic_slice(wl_p, [start], [chunk_size])
            fx_c = jax.lax.dynamic_slice(sp_p, [start], [chunk_size])
            x_c = jax.lax.dynamic_slice(x_p, [start], [chunk_size])
            y_c = jax.lax.dynamic_slice(y_p, [start], [chunk_size])

            u_x = x_c - 0.5 + rel0_x
            u_y = y_c - 0.5 + rel0_y
            m_x = jnp.floor(u_x).astype(jnp.int32)
            m_y = jnp.floor(u_y).astype(jnp.int32)
            p_x = jnp.clip(jnp.floor((u_x - m_x) * os_), 0, os_ - 1
                           ).astype(jnp.int32)
            p_y = jnp.clip(jnp.floor((u_y - m_y) * os_), 0, os_ - 1
                           ).astype(jnp.int32)

            i0 = jnp.clip(jnp.searchsorted(grid_wl, wl_c) - 1, 0, n_grid - 2)
            t = jnp.clip((wl_c - grid_wl[i0])
                         / (grid_wl[i0 + 1] - grid_wl[i0]), 0.0, 1.0)
            lo = binned[p_y, p_x, i0]
            hi = binned[p_y, p_x, i0 + 1]
            nat = (lo + t[:, None, None] * (hi - lo)) * fx_c[:, None, None]
            # Scatter replaced by a reduction; the base indices m_x/m_y
            # become dead index math that XLA eliminates.
            out = out.at[0, 0].add(nat.ravel().sum())
            return out, None

        output, _ = jax.lax.scan(process_chunk, output, jnp.arange(n_chunks))
        return output

    return jax.jit(disperse)


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "none"
    except Exception:
        return "unavailable"


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sca", type=int, default=1)
    ap.add_argument("--orders", default="0,1,2")
    ap.add_argument("--n-gal", type=int, default=100)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="wavelengths per chunk (2000 = the production "
                         "default since the native-deposit port)")
    ap.add_argument("--dlam", type=float, default=2.0,
                    help="wavelength spacing in Angstrom (2.0 = production)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    orders = args.orders.split(",")
    rng = np.random.default_rng(args.seed)

    meta = {
        "tag": args.tag,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "gpu": gpu_name(),
        "host": platform.node(),
        "code_commit": git_commit(),
        "n_gal": args.n_gal, "batch": args.batch,
        "chunk_size": args.chunk_size, "dlam_A": args.dlam,
        "seed": args.seed, "sca": args.sca,
        "env": {k: v for k, v in __import__("os").environ.items()
                if k.startswith(("XLA_", "JAX_"))},
    }
    print(json.dumps(meta, indent=1))

    element = elements.get_element(None)  # grism
    _, sensitivity_dir, model_path, psf_cache_dir = resolve_paths()
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

    # Synthetic galaxies, interior positions so full traces stay on-detector.
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

    # Smooth synthetic SED at production magnitude scale.
    sed = (1e-16 * (1.0 + 0.3 * np.sin(wl_um * 20.0))).astype(np.float32)
    spectra = np.tile(sed, (n, 1)) * rng.uniform(0.3, 3.0, (n, 1)).astype(np.float32)

    results = []
    for order in orders:
        variants = {
            "baseline": galaxy_disperser.make_galaxy_disperser(
                psf_payloads[order], optical_payloads[order],
                chunk_size=args.chunk_size),
            "noscatter": make_noscatter_disperser(
                psf_payloads[order], optical_payloads[order],
                args.chunk_size),
        }
        for variant, gd_fn in variants.items():
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

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()

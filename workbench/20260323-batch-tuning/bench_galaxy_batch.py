#!/usr/bin/env python
"""
Benchmark galaxy batch size to find optimal throughput.

Runs galaxy dispersion on a single SCA with varying batch sizes,
measuring wall-clock time per galaxy. Uses the full catalog (100% galaxies).

Usage:
    pixi run -e cuda python workbench/20260323-batch-tuning/bench_galaxy_batch.py
"""

import os
import tempfile
import time

os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "jax-cache-bench"),
)

import jax
import jax.numpy as jnp
import numpy as np

from roman_disperser import (
    galaxy_disperser, psf_model, sersic, star_disperser,
)
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.pipeline import (
    DETECTOR_SIZE,
    resolve_paths, cone_search, select_sources_per_order,
    load_sensitivities,
    make_batched_galaxy_fori, disperse_batched_galaxies,
)
from roman_disperser.elements import GRISM
import roman_disperser.optical_model_jax as omj

# -- Add parent so we can import build_dispersed_image helpers --
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
from build_dispersed_image import (
    load_catalog, validate_catalog, trim_wavelength_grid, load_galaxy_seds,
)

# Grism-only benchmark (as originally run, 2026-03-23). These lived on
# pipeline as module constants until the prism merge moved them to the
# element records.
ORDERS = GRISM.orders
LAM_MIN = GRISM.lam_min
LAM_MAX = GRISM.lam_max


# ── Configuration ─────────────────────────────────────────────────────────────

SCA = 5
POINTING_RA = 10.0
POINTING_DEC = 0.0
POINTING_PA = 0.0
CONE_RADIUS = 0.6
GALAXY_NPIX = 30
ORDER = "1"  # benchmark on order 1 (the main science order)

BATCH_SIZES = [50, 100, 200, 500, 1000]


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # ── Load catalog & model ──────────────────────────────────────────────────

    catalog_dir, sensitivity_dir, optical_model_path, psf_cache_dir = \
        resolve_paths(None, None, None, None)

    print("Loading catalog...")
    t0 = time.time()
    meta, store, wavelengths_full = load_catalog(catalog_dir)
    n_stars = int((meta["type"] == "PSF").sum())
    n_galaxies = int((meta["type"] == "SER").sum())
    print(f"  {len(meta)} sources ({n_stars} stars, {n_galaxies} galaxies) "
          f"in {time.time() - t0:.2f}s")

    validate_catalog(meta, store, wavelengths_full, GRISM)

    wavelengths_ang, wl_mask, dlam_angstroms = trim_wavelength_grid(
        wavelengths_full, GRISM)
    wavelengths_um = (wavelengths_ang / 1e4).astype(np.float32)
    wavelengths_jax = jnp.array(wavelengths_um)
    n_wavelength = len(wavelengths_um)
    print(f"Wavelength grid: {LAM_MIN}-{LAM_MAX} um, "
          f"{dlam_angstroms:.1f} A spacing, {n_wavelength} samples")

    # ── Optical model + PSF ───────────────────────────────────────────────────

    print("Loading optical model...")
    model = RomanOpticalModel(config_file=str(optical_model_path))

    detector_name = f"WFI{SCA:02d}"
    optical_payload = omj.make_sca_payload(model, sca=SCA, order=ORDER)

    # PSF order mapping (order 2 reuses order 1)
    psf_order = "1" if ORDER == "2" else ORDER
    psf_payload = psf_model.get_or_make_psf_payload(
        detector=detector_name, order=psf_order,
        cache_dir=str(psf_cache_dir), verbose=True,
    )
    oversample = int(psf_payload["oversample"])
    galaxy_npix_os = GALAXY_NPIX * oversample

    # Sensitivity
    sensitivities = load_sensitivities(sensitivity_dir, SCA, wavelengths_um,
                                       ORDERS)
    sens = sensitivities[ORDER]

    print(f"  PSF oversample: {oversample}x, "
          f"galaxy image: {GALAXY_NPIX}px native = {galaxy_npix_os}px oversampled")

    # ── Galaxy disperser (order-independent of batch size) ────────────────────

    disperser_fn = galaxy_disperser.make_galaxy_disperser(
        psf_payload, optical_payload,
    )

    # ── Source selection ──────────────────────────────────────────────────────

    print(f"\nCone search around ({POINTING_RA}, {POINTING_DEC})...")
    ra_all = meta["ra"].values
    dec_all = meta["dec"].values
    cone_mask = cone_search(ra_all, dec_all, POINTING_RA, POINTING_DEC, CONE_RADIUS)
    meta_cone = meta[cone_mask].reset_index(drop=True)
    is_galaxy = (meta_cone["type"] == "SER").values
    n_galaxies_cone = int(is_galaxy.sum())
    print(f"  {int(cone_mask.sum())} sources in cone, {n_galaxies_cone} galaxies")

    # Sky -> FPA -> SCA
    xfpa, yfpa = omj.get_fpa_pos(
        jnp.array(meta_cone["ra"].values),
        jnp.array(meta_cone["dec"].values),
        POINTING_RA, POINTING_DEC, POINTING_PA,
    )

    # Select for this order
    optical_payloads = {
        order: omj.make_sca_payload(model, sca=SCA, order=order)
        for order in ORDERS
    }
    order_masks, any_mask = select_sources_per_order(optical_payloads, xfpa,
                                                     yfpa, ORDERS,
                                                     LAM_MIN, LAM_MAX)
    omask = order_masks[ORDER]

    # Galaxy subset for this order
    any_mask_np = np.asarray(any_mask)
    is_galaxy_sel = is_galaxy[any_mask_np]
    gal_omask = omask[any_mask_np] & is_galaxy_sel
    n_gal_order = int(gal_omask.sum())

    # SCA coordinates
    xsca_all, ysca_all = omj.fpa_to_sca(optical_payloads["1"], xfpa[any_mask], yfpa[any_mask])
    x_gal = np.asarray(xsca_all[gal_omask])
    y_gal = np.asarray(ysca_all[gal_omask])

    print(f"  Order {ORDER}: {n_gal_order} galaxies on detector")

    # ── Load galaxy spectra ───────────────────────────────────────────────────

    print("Loading galaxy spectra...")
    t0 = time.time()
    galaxy_meta_cone = meta_cone[is_galaxy]
    galaxy_spectra_all = load_galaxy_seds(store, galaxy_meta_cone, wl_mask)
    print(f"  {n_galaxies_cone} spectra ({galaxy_spectra_all.nbytes / 1e6:.1f} MB) "
          f"in {time.time() - t0:.2f}s")

    # Map order galaxies -> spectra indices
    cone_indices = np.where(any_mask_np)[0]
    galaxy_cone_indices = np.where(is_galaxy)[0]
    gal_order_in_sel = np.where(gal_omask)[0]
    gal_cone_pos = cone_indices[gal_order_in_sel]
    gal_rank_in_cone = np.searchsorted(galaxy_cone_indices, gal_cone_pos)
    spec_gal = galaxy_spectra_all[gal_rank_in_cone]

    # ── Generate Sersic images ────────────────────────────────────────────────

    print("Generating Sersic images...")
    t0 = time.time()
    gal_sel_positions = np.where(is_galaxy_sel)[0]
    gal_rank_in_sel = np.searchsorted(gal_sel_positions, gal_order_in_sel)

    # Get all selected galaxies for image generation
    all_gal_sel_idx = cone_indices[is_galaxy_sel]
    gal_meta_for_images = meta_cone.iloc[all_gal_sel_idx]

    r_eff_pix = sersic.catalog_r_eff_to_pixels(
        jnp.array(gal_meta_for_images["half_light_radius"].values, dtype=jnp.float32),
        oversample=oversample,
    )
    n_sersic = jnp.array(gal_meta_for_images["n"].values, dtype=jnp.float32)
    ba = jnp.array(gal_meta_for_images["ba"].values, dtype=jnp.float32)
    theta = sersic.sky_pa_to_sca_theta(
        jnp.array(gal_meta_for_images["pa"].values, dtype=jnp.float32),
        POINTING_PA,
    )
    all_galaxy_images = sersic.make_sersic_images(r_eff_pix, n_sersic, ba, theta, galaxy_npix_os)
    all_galaxy_images.block_until_ready()
    imgs_gal = all_galaxy_images[gal_rank_in_sel]
    print(f"  {len(all_galaxy_images)} images ({all_galaxy_images.nbytes / 1e6:.1f} MB) "
          f"in {time.time() - t0:.2f}s")

    # ── Benchmark loop ────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {n_gal_order} galaxies, order {ORDER}, SCA {SCA}")
    print(f"{'='*70}\n")

    results = []

    for bs in BATCH_SIZES:
        print(f"--- batch_size = {bs} ---")

        # Build fresh fori function for this batch size
        fori_fn = make_batched_galaxy_fori(
            disperser_fn, sens, wavelengths_jax, dlam_angstroms,
        )

        # JIT warmup: single dummy call to compile
        print("  Warmup (JIT compile)...", end=" ", flush=True)
        t0 = time.time()
        dummy_output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
        dummy_spec = jnp.zeros((bs, n_wavelength), dtype=jnp.float32)
        dummy_x = jnp.zeros(bs, dtype=jnp.float32) + 2000.0
        dummy_y = jnp.zeros(bs, dtype=jnp.float32) + 2000.0
        dummy_img = jnp.zeros((bs, galaxy_npix_os, galaxy_npix_os), dtype=jnp.float32)
        _ = fori_fn(1, dummy_spec, dummy_x, dummy_y, dummy_img, dummy_output)
        _.block_until_ready()
        warmup_time = time.time() - t0
        print(f"{warmup_time:.1f}s")

        # Timed run
        output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
        print(f"  Dispersing {n_gal_order} galaxies...", end=" ", flush=True)
        t0 = time.time()
        output = disperse_batched_galaxies(
            fori_fn, spec_gal, x_gal, y_gal, imgs_gal,
            output, bs,
        )
        elapsed = time.time() - t0
        ms_per = elapsed / n_gal_order * 1e3
        rate = n_gal_order / elapsed
        print(f"{elapsed:.2f}s  ({ms_per:.2f} ms/gal, {rate:.1f} gal/s)")

        # Second run to check consistency
        output2 = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
        t0 = time.time()
        output2 = disperse_batched_galaxies(
            fori_fn, spec_gal, x_gal, y_gal, imgs_gal,
            output2, bs,
        )
        elapsed2 = time.time() - t0
        ms_per2 = elapsed2 / n_gal_order * 1e3

        # Verify outputs match
        max_diff = float(jnp.max(jnp.abs(output - output2)))
        print(f"  Run 2: {elapsed2:.2f}s ({ms_per2:.2f} ms/gal), "
              f"max diff = {max_diff:.2e}")

        results.append({
            "batch_size": bs,
            "n_galaxies": n_gal_order,
            "warmup_s": warmup_time,
            "run1_s": elapsed,
            "run2_s": elapsed2,
            "ms_per_gal_run1": ms_per,
            "ms_per_gal_run2": ms_per2,
            "gal_per_s": n_gal_order / elapsed2,  # use run2 (warm cache)
            "max_diff": max_diff,
        })
        print()

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch':>6} {'Warmup':>8} {'Run1':>8} {'Run2':>8} "
          f"{'ms/gal':>8} {'gal/s':>8}")
    print("-" * 56)
    for r in results:
        print(f"{r['batch_size']:>6d} {r['warmup_s']:>8.1f} {r['run1_s']:>8.2f} "
              f"{r['run2_s']:>8.2f} {r['ms_per_gal_run2']:>8.2f} "
              f"{r['gal_per_s']:>8.1f}")

    # ── Extrapolation to full run ─────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("FULL RUN TIME ESTIMATES")
    print(f"{'='*70}")
    print(f"Assumptions: 18 SCAs, 3 orders, ~{n_gal_order} galaxies/order/SCA")
    print(f"(This is the actual count for SCA {SCA} order {ORDER} with this catalog)")
    print()

    # Use run2 timings (warm)
    best = min(results, key=lambda r: r["ms_per_gal_run2"])
    worst = max(results, key=lambda r: r["ms_per_gal_run2"])

    for label, r in [("Best", best), ("Worst", worst)]:
        gal_time_per_sca = r["ms_per_gal_run2"] * n_gal_order * 3 / 1e3  # 3 orders
        gal_time_total = gal_time_per_sca * 18  # 18 SCAs
        warmup_total = r["warmup_s"] * 3 * 18  # per order per SCA
        total = gal_time_total + warmup_total
        print(f"  {label} (batch_size={r['batch_size']}):")
        print(f"    Galaxy dispersion: {gal_time_total/60:.1f} min "
              f"({gal_time_per_sca:.1f}s/SCA)")
        print(f"    JIT warmup:        {warmup_total/60:.1f} min "
              f"({r['warmup_s']:.1f}s/order/SCA)")
        print(f"    Total (galaxies):  {total/60:.1f} min")
        print()


if __name__ == "__main__":
    main()

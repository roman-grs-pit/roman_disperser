"""Bisect to find which source(s) introduce NaN/Inf into a per-SCA grism dispersion.

Reproduces the SCA-level dispersion for a single pointing without writing files,
but processes the source list in halves with a finiteness check between calls
so we can binary-search down to the offending galaxy.

Usage::

    pixi run -e cuda python scripts/debug_dispersion_nan.py \
        --config /data/npadman/tmp/debug-grism-nan/config.yaml \
        --pointings /mnt/roman-science/.../acceptance-testing.sim.ecsv \
        --pointing-row 14 \
        --sca 2

`--pointing-row` is 0-indexed into the (already-bandpass-filtered) ECSV.
For the NERSC failed pointings:
    008.001 = row 14, SCA 02
    008.002 = row 15, SCA 10
    016.001 = row 30, SCA 18
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set JAX cache before importing jax
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    "/data/npadman/tmp/debug-grism-nan/jax-cache",
)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent))

# Reuse the pipeline + helpers from build_grism_image
from build_grism_image import (
    setup_pipeline, load_galaxy_seds, load_catalog,
)
from roman_disperser import (
    galaxy_disperser, psf_model, sersic, star_disperser,
)
from roman_disperser import optical_model_jax as omj
from roman_disperser.pipeline import (
    cone_search, select_sources_per_order,
    make_batched_galaxy_fori, make_batched_star_fori,
    disperse_batched_galaxies, disperse_batched_stars,
    DETECTOR_SIZE, ORDERS,
)


def is_finite_all(arr):
    """Return (all_finite, n_nan, n_inf)."""
    nn = int(jnp.isnan(arr).sum())
    ni = int(jnp.isinf(arr).sum())
    return (nn == 0 and ni == 0), nn, ni


def disperse_subset(fori_fn, spec, x, y, imgs, batch_size):
    """Disperse a subset of galaxies onto a fresh output buffer.

    Returns (output, n_nan, n_inf).
    """
    output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
    output = disperse_batched_galaxies(
        fori_fn, spec, x, y, imgs, output, batch_size,
    )
    n_nan = int(jnp.isnan(output).sum())
    n_inf = int(jnp.isinf(output).sum())
    return output, n_nan, n_inf


def bisect_galaxies(fori_fn, spec, x, y, imgs, batch_size, indices, prefix=""):
    """Recursively bisect to find the smallest set of galaxies whose
    dispersion alone produces non-finite values.

    `indices` are absolute indices into the original galaxy list (for
    reporting; the slicing here is on the local arrays passed in).

    Returns the list of offending absolute indices.
    """
    n = len(indices)
    print(f"{prefix}bisect on {n} galaxies (idx [{indices[0]}..{indices[-1]}])")

    _, n_nan, n_inf = disperse_subset(fori_fn, spec, x, y, imgs, batch_size)
    if n_nan == 0 and n_inf == 0:
        print(f"{prefix}  clean ({n} galaxies)")
        return []
    print(f"{prefix}  dirty: NaN={n_nan} Inf={n_inf}")

    if n == 1:
        return [int(indices[0])]

    half = n // 2
    left = bisect_galaxies(
        fori_fn,
        spec[:half], x[:half], y[:half], imgs[:half],
        batch_size, indices[:half], prefix + "  ",
    )
    right = bisect_galaxies(
        fori_fn,
        spec[half:], x[half:], y[half:], imgs[half:],
        batch_size, indices[half:], prefix + "  ",
    )
    return left + right


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--pointings", required=True)
    p.add_argument("--pointing-row", type=int, required=True,
                   help="0-indexed row in the bandpass-filtered ECSV")
    p.add_argument("--sca", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Read ECSV via astropy (handles GRISM filter)
    tab = Table.read(args.pointings)
    grism_mask = np.array([str(b) == "GRISM" for b in tab["BANDPASS"]])
    tab = tab[grism_mask]
    row = tab[args.pointing_row]
    pointing_ra = float(row["RA"])
    pointing_dec = float(row["DEC"])
    pointing_pa = float(row["PA"])
    print(f"Pointing row {args.pointing_row}: RA={pointing_ra}, "
          f"Dec={pointing_dec}, PA={pointing_pa}")
    print(f"  APT: {row['PLAN']}.{row['PASS']}.{row['SEGMENT']}."
          f"{row['OBSERVATION']}.{row['VISIT']}.{row['EXPOSURE']}")

    # -- Setup --
    pipeline = setup_pipeline(
        sca_list=[args.sca],
        catalog_dir=config["catalog_dir"],
        sensitivity_dir=config["sensitivity_dir"],
        optical_model_path=config["optical_model"],
        psf_cache_dir=config["psf_cache_dir"],
        star_batch_size=config.get("star_batch_size", 1000),
        galaxy_batch_size=config.get("galaxy_batch_size", 100),
        galaxy_npix=config.get("galaxy_npix", 30),
        verbose=True,
    )

    sca = args.sca
    sd = pipeline["sca_data"][sca]
    galaxy_batch_size = pipeline["galaxy_batch_size"]
    galaxy_npix_os = pipeline["galaxy_npix_os"]
    oversample = pipeline["oversample"]
    wavelengths_jax = pipeline["wavelengths_jax"]
    dlam_angstroms = pipeline["dlam_angstroms"]
    n_wavelength = len(pipeline["wavelengths_um"])

    # -- Build dispersers (per-SCA) --
    detector_name = f"WFI{sca:02d}"
    psf_payloads = {}
    for psf_order in ["0", "1"]:
        psf_payloads[psf_order] = psf_model.get_or_make_psf_payload(
            detector=detector_name, order=psf_order,
            cache_dir=pipeline["psf_cache_dir"], verbose=False,
        )
    psf_payloads["2"] = psf_payloads["1"]

    galaxy_fori_fns = {}
    for order in ORDERS:
        gd_fn = galaxy_disperser.make_galaxy_disperser(
            psf_payloads[order], sd["optical_payloads"][order],
        )
        galaxy_fori_fns[order] = make_batched_galaxy_fori(
            gd_fn, sd["sensitivities"][order],
            wavelengths_jax, dlam_angstroms,
        )

    # -- Cone search & FPA conversion --
    meta = pipeline["meta"]
    store = pipeline["store"]
    cone_mask = cone_search(
        meta["ra"].values, meta["dec"].values,
        pointing_ra, pointing_dec,
        config.get("cone_radius", 0.6),
    )
    meta_cone = meta[cone_mask].copy()
    meta_cone_reset = meta_cone.reset_index(drop=True)
    print(f"Cone search: {len(meta_cone)} sources")

    is_star = (meta_cone_reset["type"].values == "PSF")
    is_galaxy = (meta_cone_reset["type"].values == "SER")

    # Float64 straight through: get_fpa_pos differences on the host before
    # JAX sees anything. A jnp.array() wrap here would quantise absolute RA
    # to float32 first -- and now raises TypeError rather than silently
    # diverging from the pipeline this script exists to debug.
    xfpa, yfpa = omj.get_fpa_pos(
        meta_cone_reset["ra"].values,
        meta_cone_reset["dec"].values,
        pointing_ra, pointing_dec, pointing_pa,
    )

    order_masks, any_mask = select_sources_per_order(
        sd["optical_payloads"], xfpa, yfpa,
    )
    any_mask_np = np.asarray(any_mask)
    is_galaxy_sel = is_galaxy[any_mask_np]
    n_galaxies_sel = int(is_galaxy_sel.sum())
    print(f"On detector {sca}: {int(any_mask_np.sum())} sources "
          f"({n_galaxies_sel} galaxies)")

    if n_galaxies_sel == 0:
        print("No galaxies on this detector. Nothing to bisect.")
        return

    # -- Galaxy SEDs + Sersic images --
    cone_indices = np.where(any_mask_np)[0]
    gal_sel_meta = meta_cone_reset.iloc[cone_indices[is_galaxy_sel]]
    galaxy_spectra_sel = load_galaxy_seds(
        store, gal_sel_meta, pipeline["wl_mask"],
    )
    print(f"Galaxy SEDs: {n_galaxies_sel} loaded "
          f"({galaxy_spectra_sel.nbytes / 1e6:.1f} MB)")

    r_eff_pix = sersic.catalog_r_eff_to_pixels(
        jnp.array(gal_sel_meta["half_light_radius"].values, dtype=jnp.float32),
        oversample=oversample,
    )
    n_sersic = jnp.array(gal_sel_meta["n"].values, dtype=jnp.float32)
    ba = jnp.array(gal_sel_meta["ba"].values, dtype=jnp.float32)
    theta = sersic.sky_pa_to_sca_theta(
        jnp.array(gal_sel_meta["pa"].values, dtype=jnp.float32),
        pointing_pa,
    )
    galaxy_images = sersic.make_sersic_images(
        r_eff_pix, n_sersic, ba, theta, galaxy_npix_os,
    )
    galaxy_images.block_until_ready()

    # Sanity: any non-finite in Sersic input?
    n_si_nan = int(jnp.isnan(galaxy_images).sum())
    n_si_inf = int(jnp.isinf(galaxy_images).sum())
    print(f"Sersic images: shape={tuple(galaxy_images.shape)} "
          f"NaN={n_si_nan} Inf={n_si_inf}")

    # SCA coordinates
    xsca_all, ysca_all = omj.fpa_to_sca(
        sd["optical_payloads"]["1"], xfpa[any_mask], yfpa[any_mask],
    )
    xsca_all_np = np.asarray(xsca_all)
    ysca_all_np = np.asarray(ysca_all)

    # -- For each order, bisect among galaxies in that order --
    for order in ORDERS:
        omask = order_masks[order]
        omask_np = np.asarray(omask)
        gal_omask = omask_np[any_mask_np] & is_galaxy_sel
        n_gal_order = int(gal_omask.sum())
        if n_gal_order == 0:
            print(f"\n=== Order {order}: no galaxies ===")
            continue

        # Index mapping: rank-in-galaxy-list for galaxies in this order
        gal_sel_positions = np.where(is_galaxy_sel)[0]
        gal_order_in_sel = np.where(gal_omask)[0]
        gal_rank_in_sel = np.searchsorted(gal_sel_positions, gal_order_in_sel)

        x_gal = xsca_all_np[gal_omask]
        y_gal = ysca_all_np[gal_omask]
        spec_gal = galaxy_spectra_sel[gal_rank_in_sel]
        imgs_gal = galaxy_images[gal_rank_in_sel]

        print(f"\n=== Order {order}: {n_gal_order} galaxies ===")

        # Disperse all to confirm non-finite present
        _, n_nan, n_inf = disperse_subset(
            galaxy_fori_fns[order], spec_gal, x_gal, y_gal, imgs_gal,
            galaxy_batch_size,
        )
        print(f"  full: NaN={n_nan} Inf={n_inf}")
        if n_nan == 0 and n_inf == 0:
            continue

        # Bisect
        offending = bisect_galaxies(
            galaxy_fori_fns[order],
            spec_gal, x_gal, y_gal, imgs_gal,
            galaxy_batch_size,
            np.arange(n_gal_order),
        )
        print(f"\n  Order {order} offending galaxies (rank in order): "
              f"{offending}")

        # Dump full info per offender
        for rank in offending:
            sel_idx = gal_order_in_sel[rank]  # rank in is_galaxy_sel
            cone_idx = cone_indices[sel_idx]
            meta_idx = int(meta_cone.index[cone_idx])
            row = meta_cone_reset.iloc[cone_idx]
            sed = np.asarray(galaxy_spectra_sel[gal_rank_in_sel[rank]])
            img = np.asarray(galaxy_images[gal_rank_in_sel[rank]])
            print(f"\n  --- Offender (order {order}, rank {rank}) ---")
            print(f"    catalog_index: {meta_idx}")
            print(f"    src_index/sim: {int(row['src_index'])}/"
                  f"{int(row['sim'])}")
            print(f"    sed_index:     {int(row['sed_index'])}")
            print(f"    RA, Dec:       {row['ra']:.6f}, {row['dec']:.6f}")
            print(f"    xsca, ysca:    {x_gal[rank]:.3f}, {y_gal[rank]:.3f}")
            print(f"    type:          {row['type']}")
            print(f"    n, hlr, ba, pa, F158, fs: "
                  f"{row['n']} {row['half_light_radius']} "
                  f"{row['ba']} {row['pa']} {row['F158']} {row['flux_scale']}")
            print(f"    SED stats:     min={sed.min():.3e} "
                  f"max={sed.max():.3e} sum={sed.sum():.3e} "
                  f"NaN={int(np.isnan(sed).sum())} "
                  f"Inf={int(np.isinf(sed).sum())} "
                  f"neg={int((sed<0).sum())}")
            print(f"    Sersic stats:  shape={img.shape} "
                  f"min={img.min():.3e} max={img.max():.3e} "
                  f"sum={img.sum():.3e} "
                  f"NaN={int(np.isnan(img).sum())} "
                  f"Inf={int(np.isinf(img).sum())}")


if __name__ == "__main__":
    main()

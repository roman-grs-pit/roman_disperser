"""Golden-frame scene: the end-to-end regression reference for the disperser.

This module defines a small, fixed scene of stars and galaxies, renders it
through the *production* dispersal path (``make_galaxy_disperser`` /
``make_star_disperser`` + the batched fori wrappers, exactly as
``scripts/build_dispersed_image.py`` uses them), and reads/writes the golden
reference frames that ``tests/test_golden_frame.py`` compares against.

Why this exists
---------------
The suite's property tests (flux conservation, delta-galaxy vs star, CPU vs
GPU) run on mock PSFs at small dimensions and check invariants, not values; the
``benchmarks/`` suite catches *correct-but-slow*. Neither pins the actual
output of the production path at production geometry, so a restructuring of
the dispersal internals (e.g. the planned FFT pad-to-fast-size change and the
native-deposit rewrite) had no durable end-to-end guard: old-vs-new
equivalence checks lived in throwaway workbench scripts and die with the old
code path. The golden frames are that guard.

Design decisions (agreed 2026-08-24, see the research log)
----------------------------------------------------------
* **Few curated objects, not a statistical population.** The disperser runs
  the identical code per source; extra objects add runtime, not sensitivity.
  Each object is chosen to hit a branch: detector center (vanilla), stamp
  clipped at the detector edge, off-grid sub-pixel position (bilinear deposit
  weights), an overlapping close pair (scatter-add accumulation), and two
  stars (the ``star_disperser`` path). Whatever the code currently does at
  the edges is what the golden pins — the test guards *equivalence under
  refactoring*, not correctness of the edge convention.
* **Full 4088x4088 frame, compressed.** The frame is exactly zero outside the
  trace footprints and zeros compress away, so a full-frame
  ``savez_compressed`` costs about the same as storing crops — and the
  comparison collapses to one ``allclose`` with flux outside any window
  caught by construction (no bbox bookkeeping, no separate sum tripwire).
* **Two tiers, because coarse wavelength sampling has real blind spots.**
  The per-pixel accumulation depth and the deposit-collision regime both
  scale with the number of wavelength samples (order 0 runs at ~59k
  collisions/px at production sampling), and those are exactly what the
  native-deposit restructure touches. So: a *coarse* tier (20 A, all
  element/order configs) that runs in the default suite, and a *full* tier
  (2 A = production, grism orders 1 and 0 — order 0 is the
  collision-extreme case) marked ``slow`` and required before merging any
  PR that touches the dispersal path.
* **References are a vendored asset, pinned by name in the test.** Frames
  live in the ``roman_disperser_data`` store (this test already needs
  hydrated PSF/sensitivity data, so this adds no new dependency) under a
  version-named directory (``GOLDEN_VERSION``). The test hard-codes that
  name: a PR that intentionally changes results must bump the pin *and*
  publish the new asset in the same change, keeping code and reference
  atomic. Provenance (code version, git sha, jax version, platform, seed)
  is stored inside each ``.npz``.

Regenerating (only for an *intentional* results change)::

    pixi run python -m tests.golden_frame            # writes into <data>/golden_frames/<GOLDEN_VERSION>/
    pixi run python -m tests.golden_frame --out-dir /tmp/golden   # elsewhere

then publish the directory as a ``roman_disperser_data`` release named
``GOLDEN_VERSION`` (tarball of the version-named directory), update
``manifest.json`` there, and bump ``GOLDEN_VERSION`` here. References are
generated on **CPU** (``JAX_PLATFORMS=cpu``) so the blessed values come from
the deterministic backend; the comparison gate absorbs CPU/GPU differences
(see ``tests/test_golden_frame.py``).
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import roman_disperser
import roman_disperser.optical_model_jax as omj
from roman_disperser import elements, galaxy_disperser, paths, psf_model, sersic, star_disperser
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.pipeline import (
    DETECTOR_SIZE,
    disperse_batched_galaxies,
    disperse_batched_stars,
    load_sensitivities,
    make_batched_galaxy_fori,
    make_batched_star_fori,
    resolve_paths,
)

# The pinned reference version. Bumping this is a *results-changing* act:
# do it only together with publishing the regenerated frames as a
# roman_disperser_data release of the same name (see module docstring).
GOLDEN_VERSION = "golden-frames-v1"

SCA = 1  # matches all perf/benchmark work (bench_deposit.py, the workbench runs)

# (element, order) configurations per tier, and their wavelength spacings.
DLAM_COARSE_A = 20.0
DLAM_FULL_A = 2.0  # production spacing (matches build_dispersed_image.py default)
CONFIGS_COARSE = (("grism", "0"), ("grism", "1"), ("grism", "2"), ("prism", "1"))
CONFIGS_FULL = (("grism", "1"), ("grism", "0"))

# ---------------------------------------------------------------------------
# The scene. Positions are 1-indexed SCA pixels (package convention).
# Explicit literals, not RNG draws: the scene is part of the reference's
# identity and should be readable at a glance.
# ---------------------------------------------------------------------------

# name, x, y, r_eff [arcsec], sersic n, b/a, theta [rad]
GALAXIES = (
    # vanilla: detector center, on-grid position
    ("center", 2044.0, 2044.0, 0.30, 1.5, 0.70, 0.30),
    # stamp clipped by the detector edge (30 native-px stamp, half-width 15)
    ("edge", 12.3, 900.7, 0.25, 2.5, 0.60, 1.10),
    # off-grid sub-pixel position; also carries the emission line (below)
    ("offgrid", 1500.37, 2600.81, 0.20, 1.0, 0.90, 0.00),
    # overlapping close pair: traces accumulate into the same pixels
    ("pair_a", 3000.25, 1200.50, 0.35, 4.0, 0.45, 2.00),
    ("pair_b", 3004.75, 1203.25, 0.15, 1.0, 0.80, 0.70),
)

# name, x, y  (stars: point sources through the star_disperser path)
STARS = (
    ("star_offgrid", 2500.60, 2200.40),
    # near the detector corner: trace partially off-detector
    ("star_corner", 100.20, 3900.80),
)

# Per-object continuum scale factors (x the common SED shape).
GALAXY_SCALES = (1.0, 2.0, 0.8, 1.5, 0.6)
STAR_SCALES = (3.0, 1.2)

# The "offgrid" galaxy gets a Gaussian emission line at the center of the
# element band (5x continuum peak, sigma 30 A) so the spectral interpolation
# is probed with a sharp feature, not just a smooth continuum.
LINE_GALAXY = "offgrid"
LINE_SIGMA_A = 30.0
LINE_AMPLITUDE = 5.0


def wavelength_grid(element, dlam_A):
    """Wavelength grid over the element band: microns, float32, dlam in A."""
    wl_ang = np.arange(element.lam_min * 1e4, element.lam_max * 1e4 + 0.1, dlam_A)
    return (wl_ang / 1e4).astype(np.float32)


def _sed(wl_um, scale, line=False, lam_c_um=None):
    """Smooth synthetic SED at production magnitude scale (matches the
    bench_deposit.py workload family), optionally with an emission line."""
    sed = 1e-16 * (1.0 + 0.3 * np.sin(wl_um * 20.0)) * scale
    if line:
        sigma_um = LINE_SIGMA_A / 1e4
        sed = sed + (LINE_AMPLITUDE * 1e-16 * scale
                     * np.exp(-0.5 * ((wl_um - lam_c_um) / sigma_um) ** 2))
    return sed.astype(np.float32)


def render_frame(element_name, order, dlam_A):
    """Render the golden scene for one (element, order) at spacing dlam_A.

    Runs the production dispersal path end to end on the real (hydrated)
    optical model, PSF cache, and sensitivities for SCA 1. Returns the
    4088x4088 float32 detector frame.
    """
    element = elements.get_element(element_name)
    _, sensitivity_dir, model_path, psf_cache_dir = resolve_paths(element=element)
    model = RomanOpticalModel(str(model_path))

    wl_um = wavelength_grid(element, dlam_A)
    wl_jax = jnp.array(wl_um)
    sens = load_sensitivities(sensitivity_dir, SCA, wl_um, [order])[order]

    psf_payload = psf_model.get_or_make_psf_payload(
        detector=f"WFI{SCA:02d}", order=order,
        wavelengths=elements.psf_cache_wavelengths(element),
        stpsf_filter=element.stpsf_filters[order],
        cache_dir=psf_cache_dir, verbose=False)
    optical_payload = omj.make_sca_payload(model, sca=SCA, order=order)

    oversample = int(psf_payload["oversample"])
    npix_os = 30 * oversample  # production stamp geometry (30 native px)

    # --- galaxies ---
    names, xs, ys, r_effs, n_sers, bas, thetas = zip(*GALAXIES)
    r_eff_pix = sersic.catalog_r_eff_to_pixels(
        jnp.array(r_effs, dtype=jnp.float32), oversample=oversample)
    images = sersic.make_sersic_images(
        r_eff_pix, jnp.array(n_sers, dtype=jnp.float32),
        jnp.array(bas, dtype=jnp.float32), jnp.array(thetas, dtype=jnp.float32),
        npix_os)

    lam_c_um = 0.5 * (element.lam_min + element.lam_max)
    gal_spectra = np.stack([
        _sed(wl_um, scale, line=(name == LINE_GALAXY), lam_c_um=lam_c_um)
        for name, scale in zip(names, GALAXY_SCALES)])

    gd_fn = galaxy_disperser.make_galaxy_disperser(psf_payload, optical_payload)
    gal_fori = make_batched_galaxy_fori(gd_fn, sens, wl_jax, dlam_A)

    output = jnp.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=jnp.float32)
    output = disperse_batched_galaxies(
        gal_fori, gal_spectra,
        np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32),
        images, output, batch_size=len(GALAXIES))

    # --- stars ---
    s_names, s_xs, s_ys = zip(*STARS)
    star_spectra = np.stack([
        _sed(wl_um, scale) for scale in STAR_SCALES])

    sd_fn = star_disperser.make_star_disperser(psf_payload, optical_payload)
    star_fori = make_batched_star_fori(sd_fn, sens, wl_jax, dlam_A)
    output = disperse_batched_stars(
        star_fori, star_spectra,
        np.array(s_xs, dtype=np.float32), np.array(s_ys, dtype=np.float32),
        output, batch_size=len(STARS))

    return np.asarray(output.block_until_ready())


# ---------------------------------------------------------------------------
# Reference file IO
# ---------------------------------------------------------------------------

def golden_dir(version=GOLDEN_VERSION):
    """Directory holding the pinned golden frames in the vendored data dir."""
    return Path(paths.data_dir()) / "golden_frames" / version


def frame_filename(element_name, order, tier):
    return f"{element_name}_order{order}_{tier}.npz"


def load_reference(element_name, order, tier, version=GOLDEN_VERSION):
    """Return (frame, provenance dict) for a pinned reference, or raise
    FileNotFoundError with a hydration hint."""
    path = golden_dir(version) / frame_filename(element_name, order, tier)
    if not path.exists():
        raise FileNotFoundError(
            f"golden reference {path} not hydrated "
            f"(pixi run hydrate fetches the '{version}' asset)")
    with np.load(path) as f:
        return f["frame"], json.loads(str(f["provenance"]))


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_references(out_dir):
    """Render and write every reference frame plus a provenance.json index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_prov = {
        "golden_version": GOLDEN_VERSION,
        "code_version": roman_disperser.__version__,
        "code_commit": _git_sha(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "host": platform.node(),
        "numpy": np.__version__,
        "sca": SCA,
    }
    index = dict(base_prov, frames={})
    tiers = ([(e, o, "coarse", DLAM_COARSE_A) for e, o in CONFIGS_COARSE]
             + [(e, o, "full", DLAM_FULL_A) for e, o in CONFIGS_FULL])
    for element_name, order, tier, dlam in tiers:
        fname = frame_filename(element_name, order, tier)
        print(f"rendering {fname} (dlam={dlam} A) ...", flush=True)
        frame = render_frame(element_name, order, dlam)
        prov = dict(base_prov, element=element_name, order=order,
                    tier=tier, dlam_A=dlam)
        np.savez_compressed(out_dir / fname, frame=frame,
                            provenance=json.dumps(prov))
        nz = int((frame != 0).sum())
        total = float(frame.sum(dtype=np.float64))
        index["frames"][fname] = dict(prov, nonzero_px=nz, frame_sum=total)
        print(f"  nonzero px: {nz}  sum: {total:.6e}  "
              f"size: {(out_dir / fname).stat().st_size / 1e6:.1f} MB")
    (out_dir / "provenance.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {out_dir}/provenance.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None,
                    help="destination (default: <data>/golden_frames/"
                         f"{GOLDEN_VERSION}/)")
    args = ap.parse_args()
    if jax.default_backend() != "cpu":
        raise SystemExit(
            "golden references must be generated on CPU "
            "(run with JAX_PLATFORMS=cpu) — see module docstring")
    write_references(args.out_dir or golden_dir())


if __name__ == "__main__":
    main()

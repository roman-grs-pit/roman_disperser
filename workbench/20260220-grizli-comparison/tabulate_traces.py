"""Tabulate spectral traces from class-based and JAX optical models.

Produces one ECSV file per star with trace positions at each wavelength
from both models, plus differences. Used for diagnosing offsets between
roman_disperser and grizli.
"""

import os
from pathlib import Path

import numpy as np
from astropy.table import Table

from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj

# ============================================================================
# Configuration
# ============================================================================

SCA = 1
ORDER = "1"
WAVELENGTHS = np.linspace(1.0, 1.93, 94)  # 0.01 μm spacing

STARS = {
    "center": (2043.0, 2043.0),
    "BL": (100.0, 100.0),
    "TL": (100.0, 3900.0),
    "BR": (3900.0, 100.0),
    "TR": (3900.0, 3900.0),
}

# ============================================================================
# Setup
# ============================================================================

project_root = Path(os.environ.get("PIXI_PROJECT_ROOT", "."))
config_path = project_root / "data" / "Roman_grism_OpticalModel_v0.8.yaml"
output_dir = project_root / "workbench" / "20260220-grizli-comparison"

# Load class-based model
model = RomanOpticalModel(config_file=str(config_path))
model.wl_grid = WAVELENGTHS

# Load JAX payload
payload = omj.make_sca_payload(model, sca=SCA, order=ORDER)

print(f"SCA: {SCA}, Order: {ORDER}")
print(f"Wavelengths: {WAVELENGTHS[0]:.2f} – {WAVELENGTHS[-1]:.2f} μm, "
      f"N = {len(WAVELENGTHS)}, Δλ = {WAVELENGTHS[1] - WAVELENGTHS[0]:.4f} μm")
print(f"Config: {config_path.name}")
print()

# ============================================================================
# Tabulate traces
# ============================================================================

for label, (xsca_star, ysca_star) in STARS.items():
    # --- Class-based model ---
    xfpa_class_input, yfpa_class_input = model.coords.convert_sca_to_fpa(
        xsca_star, ysca_star, sca=SCA
    )
    trace = model._get_beam_trace(
        xfpa_class_input, yfpa_class_input, sca=SCA, width=1, order=ORDER
    )
    xsca_class = trace["trace_sca_x"].ravel()
    ysca_class = trace["trace_sca_y"].ravel()
    xfpa_class = trace["trace_fpa_x"].ravel()
    yfpa_class = trace["trace_fpa_y"].ravel()
    xmpa_class = trace["trace_mpa_x"].ravel()
    ympa_class = trace["trace_mpa_y"].ravel()
    wl_class = trace["trace_wvl"].ravel()

    # --- JAX model ---
    xfpa_jax_input, yfpa_jax_input = omj.sca_to_fpa(
        payload, np.array([xsca_star]), np.array([ysca_star])
    )
    # Broadcast single FPA coord to match wavelength array length
    xfpa_broadcast = np.full(len(WAVELENGTHS), float(xfpa_jax_input[0]))
    yfpa_broadcast = np.full(len(WAVELENGTHS), float(yfpa_jax_input[0]))

    xmpa_jax, ympa_jax = omj.trace_beam(
        payload, xfpa_broadcast, yfpa_broadcast, WAVELENGTHS
    )
    xmpa_jax = np.asarray(xmpa_jax)
    ympa_jax = np.asarray(ympa_jax)

    xsca_jax, ysca_jax = omj.mpa_to_sca(payload, xmpa_jax, ympa_jax)
    xsca_jax = np.asarray(xsca_jax)
    ysca_jax = np.asarray(ysca_jax)

    # FPA from dispersed MPA positions (JAX)
    xfpa_jax, yfpa_jax = omj.mpa_to_fpa(payload, xmpa_jax, ympa_jax)
    xfpa_jax = np.asarray(xfpa_jax)
    yfpa_jax = np.asarray(yfpa_jax)

    # --- Differences ---
    dx_sca = xsca_jax - xsca_class
    dy_sca = ysca_jax - ysca_class
    dx_mpa = xmpa_jax - xmpa_class
    dy_mpa = ympa_jax - ympa_class

    # --- Build table ---
    t = Table()
    t["wavelength_um"] = WAVELENGTHS
    t["xsca_class"] = xsca_class
    t["ysca_class"] = ysca_class
    t["xfpa_class"] = xfpa_class
    t["yfpa_class"] = yfpa_class
    t["xmpa_class"] = xmpa_class
    t["ympa_class"] = ympa_class
    t["xsca_jax"] = xsca_jax
    t["ysca_jax"] = ysca_jax
    t["xfpa_jax"] = xfpa_jax
    t["yfpa_jax"] = yfpa_jax
    t["xmpa_jax"] = xmpa_jax
    t["ympa_jax"] = ympa_jax
    t["dx_sca"] = dx_sca
    t["dy_sca"] = dy_sca
    t["dx_mpa"] = dx_mpa
    t["dy_mpa"] = dy_mpa

    # Units
    t["wavelength_um"].unit = "um"
    for col in ["xsca_class", "ysca_class", "xsca_jax", "ysca_jax", "dx_sca", "dy_sca"]:
        t[col].unit = "pix"
    for col in ["xfpa_class", "yfpa_class", "xfpa_jax", "yfpa_jax"]:
        t[col].unit = "deg"
    for col in ["xmpa_class", "ympa_class", "xmpa_jax", "ympa_jax", "dx_mpa", "dy_mpa"]:
        t[col].unit = "mm"

    # Metadata
    t.meta["sca"] = SCA
    t.meta["order"] = ORDER
    t.meta["star_label"] = label
    t.meta["star_xsca"] = xsca_star
    t.meta["star_ysca"] = ysca_star
    t.meta["xfpa_class_input"] = float(xfpa_class_input)
    t.meta["yfpa_class_input"] = float(yfpa_class_input)
    t.meta["xfpa_jax_input"] = float(xfpa_jax_input[0])
    t.meta["yfpa_jax_input"] = float(yfpa_jax_input[0])
    t.meta["config"] = config_path.name
    t.meta["wavelength_range"] = f"{WAVELENGTHS[0]:.2f}-{WAVELENGTHS[-1]:.2f} um"
    t.meta["n_wavelengths"] = len(WAVELENGTHS)
    t.meta["coord_note"] = "SCA coords are 1-indexed FITS (pixel center at integer)"
    t.meta["precision_note"] = (
        "Class model uses NumPy float64; JAX payload uses float32. "
        "Differences reflect both algorithmic and precision effects."
    )

    # Write ECSV
    fname = f"star_trace_SCA{SCA}_order{ORDER}_{label}.ecsv"
    outpath = output_dir / fname
    t.write(outpath, format="ascii.ecsv", overwrite=True)

    # Summary
    print(f"{label:>8s} ({xsca_star:.0f}, {ysca_star:.0f}):  "
          f"max |dx_sca| = {np.abs(dx_sca).max():.6e} pix,  "
          f"max |dy_sca| = {np.abs(dy_sca).max():.6e} pix  → {fname}")

print("\nDone.")

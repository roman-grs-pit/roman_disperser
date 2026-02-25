"""Compare grizli/aXe beam traces with roman_disperser optical model traces.

Computes beam traces from both the aXe configuration (as used by grizli) and
the roman_disperser class-based optical model for 5 test star positions.
Produces per-star ECSV tables and a text report summarising the offsets.

Run with:
    pixi run python workbench/20260220-grizli-comparison/compare_traces.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

# Add workbench dir to path so we can import the local grism_dispersion module
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grism_dispersion

from roman_disperser.optical_model import RomanOpticalModel

# ============================================================================
# Configuration
# ============================================================================

SCA = 1
ORDER = "1"
ROTATION_CENTER = 2044  # From axe_trace.md

STARS = {
    "center": (2043.0, 2043.0),
    "BL": (100.0, 100.0),
    "TL": (100.0, 3900.0),
    "BR": (3900.0, 100.0),
    "TR": (3900.0, 3900.0),
}

# aXe pixel offset range (matching Keith's notebook)
DX_AXE = np.flip(np.arange(-380, 624, 1))  # 623 down to -380

# ============================================================================
# Setup paths
# ============================================================================

project_root = Path(os.environ.get("PIXI_PROJECT_ROOT", "."))
config_path = project_root / "data" / "Roman_grism_OpticalModel_v0.8.yaml"
sens_path = project_root / "data" / "sensitivities" / "SCA1_sens_12052024.fits"
conf_path = SCRIPT_DIR / "TestBuild_rot_det1.conf"
output_dir = SCRIPT_DIR

# ============================================================================
# Load sensitivity → determine wavelength range where sensitivity > 0
# ============================================================================

sens_table = Table.read(sens_path)
sens_wl_ang = np.array(sens_table["WAVELENGTH"])  # Angstroms
sens_val = np.array(sens_table["SENSITIVITY"])

mask_pos = sens_val > 0
wl_min_ang = sens_wl_ang[mask_pos].min()
wl_max_ang = sens_wl_ang[mask_pos].max()
wl_min_um = wl_min_ang / 1e4
wl_max_um = wl_max_ang / 1e4

# Common wavelength grid: 10 Angstrom steps in the sensitivity > 0 range
wl_common_ang = np.arange(wl_min_ang, wl_max_ang + 1.0, 10.0)
wl_common_um = wl_common_ang / 1e4

# ============================================================================
# Load aXe configuration
# ============================================================================

grizli_conf = grism_dispersion.aXeConf(conf_file=str(conf_path))

# ============================================================================
# Load optical model (class-based, float64)
# ============================================================================

model = RomanOpticalModel(config_file=str(config_path))
model.wl_grid = wl_common_um

# ============================================================================
# Rotation helper
# ============================================================================


def rotate_90(xc, yc, center=ROTATION_CENTER):
    """Rotate (xc, yc) by 90 degrees CCW about (center, center).

    Uses int() truncation to match grizli convention (see axe_trace.md).
    """
    theta = np.pi / 2
    x_rot = int(
        np.cos(theta) * (xc - center)
        - np.sin(theta) * (yc - center)
        + center
    )
    y_rot = int(
        np.sin(theta) * (xc - center)
        + np.cos(theta) * (yc - center)
        + center
    )
    return x_rot, y_rot


# ============================================================================
# Compare traces
# ============================================================================

print("=" * 78)
print("grizli/aXe vs roman_disperser Beam Trace Comparison")
print("=" * 78)
print()
print(f"SCA: {SCA},  Order: {ORDER}")
print(f"Wavelength range (sensitivity > 0): {wl_min_um:.4f} – {wl_max_um:.4f} μm")
print(f"Common grid: {len(wl_common_um)} wavelengths at 10 Å spacing")
print(f"aXe dx range: {DX_AXE[-1]} to {DX_AXE[0]}  ({len(DX_AXE)} pixels)")
print(f"Rotation: 90° CCW about ({ROTATION_CENTER}, {ROTATION_CENTER}), int() truncation")
print(f"Sign convention: offset = grizli − optical_model")
print(f"Coordinate system: SCA 1-indexed FITS pixels")
print(f"aXe config: {conf_path.name}")
print(f"Optical model: {config_path.name}")
print()

# Representative wavelengths for trend reporting
wl_short = wl_common_um[0]
wl_mid = wl_common_um[len(wl_common_um) // 2]
wl_long = wl_common_um[-1]
idx_short = 0
idx_mid = len(wl_common_um) // 2
idx_long = len(wl_common_um) - 1

summary_rows = []

for label, (xc, yc) in STARS.items():
    # --- aXe/grizli trace ---
    x_rot, y_rot = rotate_90(xc, yc)

    dy_axe, lam_axe_ang = grizli_conf.get_beam_trace(
        x=x_rot, y=y_rot, dx=DX_AXE, beam="A"
    )

    # Convert to SCA coordinates (un-rotate)
    axe_x_raw = xc + dy_axe          # cross-dispersion
    axe_y_raw = yc - DX_AXE          # dispersion direction
    axe_lam_um = lam_axe_ang / 1e4   # Angstrom → microns

    # Sort by wavelength for interpolation
    sort_idx = np.argsort(axe_lam_um)
    axe_lam_sorted = axe_lam_um[sort_idx]
    axe_x_sorted = axe_x_raw[sort_idx]
    axe_y_sorted = axe_y_raw[sort_idx]

    # Filter to sensitivity > 0 range for safe interpolation
    in_range = (axe_lam_sorted >= wl_min_um) & (axe_lam_sorted <= wl_max_um)
    axe_lam_filt = axe_lam_sorted[in_range]
    axe_x_filt = axe_x_sorted[in_range]
    axe_y_filt = axe_y_sorted[in_range]

    # Interpolate to common wavelength grid
    axe_x_interp = np.interp(wl_common_um, axe_lam_filt, axe_x_filt)
    axe_y_interp = np.interp(wl_common_um, axe_lam_filt, axe_y_filt)

    # --- Optical model trace (class-based, float64) ---
    xfpa, yfpa = model.coords.convert_sca_to_fpa(xc, yc, sca=SCA)
    trace = model._get_beam_trace(xfpa, yfpa, sca=SCA, width=1, order=ORDER)

    xsca_om = trace["trace_sca_x"].ravel()
    ysca_om = trace["trace_sca_y"].ravel()

    # --- Differences (grizli minus optical model) ---
    dx_sca = axe_x_interp - xsca_om
    dy_sca = axe_y_interp - ysca_om

    # --- Build and write ECSV table ---
    t = Table()
    t["wavelength_um"] = wl_common_um
    t["xsca_axe"] = axe_x_interp
    t["ysca_axe"] = axe_y_interp
    t["xsca_om"] = xsca_om
    t["ysca_om"] = ysca_om
    t["dx_sca"] = dx_sca
    t["dy_sca"] = dy_sca

    t["wavelength_um"].unit = "um"
    for col in ["xsca_axe", "ysca_axe", "xsca_om", "ysca_om", "dx_sca", "dy_sca"]:
        t[col].unit = "pix"

    t.meta["sca"] = SCA
    t.meta["order"] = ORDER
    t.meta["star_label"] = label
    t.meta["star_xsca"] = xc
    t.meta["star_ysca"] = yc
    t.meta["rotated_x"] = x_rot
    t.meta["rotated_y"] = y_rot
    t.meta["rotation_center"] = ROTATION_CENTER
    t.meta["sign_convention"] = "offset = grizli - optical_model"
    t.meta["axe_config"] = conf_path.name
    t.meta["optical_model_config"] = config_path.name
    t.meta["n_wavelengths"] = len(wl_common_um)

    fname = f"compare_trace_SCA{SCA}_order{ORDER}_{label}.ecsv"
    outpath = output_dir / fname
    t.write(outpath, format="ascii.ecsv", overwrite=True)

    # --- Per-star report ---
    print("-" * 78)
    print(f"Star: {label}  ({xc:.0f}, {yc:.0f})  →  rotated ({x_rot}, {y_rot})")
    print(f"  Output: {fname}")
    print()
    print(f"  {'':20s} {'min':>10s} {'mean':>10s} {'max':>10s} {'std':>10s}")
    print(
        f"  {'Δx (cross-disp)':20s} "
        f"{dx_sca.min():+10.4f} {dx_sca.mean():+10.4f} "
        f"{dx_sca.max():+10.4f} {dx_sca.std():10.4f}"
    )
    print(
        f"  {'Δy (dispersion)':20s} "
        f"{dy_sca.min():+10.4f} {dy_sca.mean():+10.4f} "
        f"{dy_sca.max():+10.4f} {dy_sca.std():10.4f}"
    )
    print()
    print(f"  Wavelength trend (Δx, Δy):")
    print(
        f"    λ = {wl_short:.4f} μm:  "
        f"Δx = {dx_sca[idx_short]:+.4f},  Δy = {dy_sca[idx_short]:+.4f}"
    )
    print(
        f"    λ = {wl_mid:.4f} μm:  "
        f"Δx = {dx_sca[idx_mid]:+.4f},  Δy = {dy_sca[idx_mid]:+.4f}"
    )
    print(
        f"    λ = {wl_long:.4f} μm:  "
        f"Δx = {dx_sca[idx_long]:+.4f},  Δy = {dy_sca[idx_long]:+.4f}"
    )
    print()

    summary_rows.append({
        "label": label,
        "xc": xc,
        "yc": yc,
        "dx_mean": dx_sca.mean(),
        "dx_max_abs": np.abs(dx_sca).max(),
        "dy_mean": dy_sca.mean(),
        "dy_max_abs": np.abs(dy_sca).max(),
    })

# ============================================================================
# Summary table
# ============================================================================

print("=" * 78)
print("Summary: grizli − optical_model offsets (pixels)")
print("=" * 78)
print()
print(
    f"  {'Star':>8s}  {'(xc, yc)':>16s}  "
    f"{'<Δx>':>8s}  {'|Δx|max':>8s}  "
    f"{'<Δy>':>8s}  {'|Δy|max':>8s}"
)
print("  " + "-" * 70)
for row in summary_rows:
    print(
        f"  {row['label']:>8s}  "
        f"({row['xc']:6.0f},{row['yc']:6.0f})  "
        f"{row['dx_mean']:+8.4f}  {row['dx_max_abs']:8.4f}  "
        f"{row['dy_mean']:+8.4f}  {row['dy_max_abs']:8.4f}"
    )
print()
print("Done.")

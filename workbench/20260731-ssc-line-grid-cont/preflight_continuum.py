#!/usr/bin/env python
"""Wave-2 pre-flight: predicted continuum ridge level, line contrast, full well.

Nothing here is simulated fresh. The line side is *measured* from the wave-1b
MODEL image (whose lines are bit-identical to wave 2's by construction), and
the continuum side is computed analytically:

    ridge peak [counts/s/px] = f_lam(lam) * flux_scale * sens(lam)
                               * (dlam/dpx)(lam) * f_central

where f_lam = FLAM_0AB_COEFF/lam^2 (flat-fnu continuum, 0-mag anchor),
flux_scale = 3.0000e-7 maggies (mag 16.3071968), sens is the per-SCA order-1
sensitivity [counts/s per FLAM-unit... as used by the pipeline: counts/s =
f_lam * sens * dlam], dlam/dpx is the local dispersion from the optical model,
and f_central is the fraction of the cross-dispersion profile falling in the
peak pixel row -- measured from a wave-1b line's cross profile, since the
continuum trace sees the same PSF.

L2 conversion uses the wave-1b measurements: L2[DN/s] ~ 0.82 * MODEL[counts/s],
per-pixel noise sigma ~ 0.105 DN/s on a 1.96 DN/s background (MA 1036,
190.22287 s). The continuum adds its own Poisson term, included below.

Outputs a JSON + printed table. Run on the head node (CPU, seconds):
    pixi run python preflight_continuum.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.paths import data_dir
from roman_disperser.pipeline import load_sensitivities
from roman_disperser.refdata import FLAM_0AB_COEFF

W1B = Path("/mnt/roman-science/grs/line-tests-20260724")
PDIR = W1B / "lines_stpsf" / "pointing_pa000_001.001.001.001.001.001"
OUT = Path(__file__).parent / "preflight_continuum.json"

FLUX_SCALE = 3.0000001e-07          # maggies, mag 16.3071968 (wave-1b/2 anchor)
LINE_CENTERS_A = [11000.0, 12500.0, 15500.0, 17000.0, 19000.0]
LINE_AMPS = [5.0, 6.25, 7.5, 8.75, 10.0]
EXPTIME = 190.22287                 # s (MA 1036 effective)
L2_PER_MODEL = 0.82                 # DN/s per counts/s (wave-1b measured)
SIGMA_L2 = 0.105                    # DN/s per-pixel robust noise (wave-1b)
GAIN = 1.22                         # e-/DN (nominal; only used for Poisson term)
FULL_WELL_E = 1.0e5                 # e-, order of magnitude for headroom check

sca = 1
model = RomanOpticalModel(str(data_dir() / "Roman_grism_OpticalModel_v0.8.yaml"))
payload = omj.make_sca_payload(model, sca=sca, order="1")
wl_um = np.array(LINE_CENTERS_A) / 1e4

# --- local dispersion dlam/dpx at SCA center, per line wavelength ---
def predict(xsca, ysca, lam_um):
    import jax.numpy as jnp
    xf, yf = omj.sca_to_fpa(payload, jnp.array([xsca]), jnp.array([ysca]))
    xm, ym = omj.trace_beam(payload, xf, yf, jnp.asarray(lam_um))
    xp, yp = omj.mpa_to_sca(payload, xm, ym)
    return np.asarray(xp).ravel(), np.asarray(yp).ravel()

x0, y0 = predict(2044.0, 2044.0, wl_um)
x1, y1 = predict(2044.0, 2044.0, wl_um + 1e-4)   # +1 Angstrom
px_per_A = np.hypot(x1 - x0, y1 - y0) / 1.0
disp_A_per_px = 1.0 / px_per_A

# --- sensitivity on the fine grid ---
wl_grid_um = np.linspace(0.9, 2.0, 5501)
sens = load_sensitivities(data_dir() / "sensitivities", sca, wl_grid_um)
sens1 = np.asarray(sens["1"])
sens_at = np.interp(wl_um, wl_grid_um, sens1)
sens0 = np.asarray(sens["0"]) if "0" in sens else None

# --- f_central: cross-dispersion peak-pixel fraction, measured from wave-1b ---
truth = pd.read_parquet(PDIR / "truth_lines.parquet")
sel = truth[(truth.sca == sca) & (truth.order == "1") & truth.on_detector
            & (truth.line_id == 2)]
# a line near the detector center, away from edges
sel = sel.assign(r=np.hypot(sel.x_pred - 2044, sel.y_pred - 2044)).sort_values("r")
row = sel.iloc[0]
fits_path = sorted(PDIR.glob(f"*detSCA{sca:02d}.fits"))[0]
img = np.asarray(fits.open(fits_path)["MODEL"].data, float)
cx, cy = int(round(row.x_pred)) - 1, int(round(row.y_pred)) - 1   # FITS->0-idx
box = img[cy - 8:cy + 9, cx - 8:cx + 9]
# dispersion axis: from the local tangent
t = np.array([x1[2] - x0[2], y1[2] - y0[2]])
t /= np.hypot(*t)
if abs(t[1]) > abs(t[0]):          # dispersion mostly along y: cross axis = x
    cross_profile = box.sum(axis=0)
else:                              # dispersion along x: cross axis = y
    cross_profile = box.sum(axis=1)
f_central = cross_profile.max() / cross_profile.sum()
line_peak_meas = box.max()          # counts/s, brightest pixel of this line

# --- continuum ridge, per line wavelength ---
flam_cont = FLAM_0AB_COEFF / np.array(LINE_CENTERS_A) ** 2 * FLUX_SCALE
ridge = flam_cont * sens_at * disp_A_per_px * f_central     # counts/s/px

ridge_dn = ridge * L2_PER_MODEL
sigma_tot = np.sqrt(SIGMA_L2**2 + ridge_dn / (GAIN * EXPTIME))
snr_cont = ridge_dn / sigma_tot

# line contrast: the measured wave-1b line peak sits ON the ridge in wave 2
contrast = line_peak_meas / ridge[2]     # line 2 measured vs its local ridge

# --- full well: brightest wave-1b pixel + local ridge, through the ramp ---
peak_rate = img.max() + ridge.max()
peak_e = peak_rate * EXPTIME             # MODEL counts/s are electrons/s here

# --- order 0: whole-band continuum lands compact; check its loading ---
if sens0 is not None and np.any(sens0 > 0):
    flam_grid = FLAM_0AB_COEFF / (wl_grid_um * 1e4) ** 2 * FLUX_SCALE
    tot0 = np.trapezoid(flam_grid * sens0, wl_grid_um * 1e4)   # counts/s total
    # order-0 image is a short trace ~ few px; assume <=30% in peak pixel
    peak0_e = 0.3 * tot0 * EXPTIME
else:
    tot0, peak0_e = 0.0, 0.0

report = {
    "sca": sca, "f_central_measured": float(f_central),
    "line2_peak_meas_counts_s": float(line_peak_meas),
    "disp_A_per_px": dict(zip(map(str, LINE_CENTERS_A),
                              np.round(disp_A_per_px, 3).tolist())),
    "ridge_counts_s_px": dict(zip(map(str, LINE_CENTERS_A),
                                  np.round(ridge, 4).tolist())),
    "ridge_DN_s_px": dict(zip(map(str, LINE_CENTERS_A),
                              np.round(ridge_dn, 4).tolist())),
    "continuum_perpx_SNR_L2": dict(zip(map(str, LINE_CENTERS_A),
                                       np.round(snr_cont, 1).tolist())),
    "line2_peak_over_local_ridge": float(np.round(contrast, 2)),
    "max_pixel_e_over_ramp": float(np.round(peak_e, 0)),
    "order0_total_counts_s": float(np.round(tot0, 2)),
    "order0_peak_e_est": float(np.round(peak0_e, 0)),
    "full_well_e_assumed": FULL_WELL_E,
}
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

ok = (snr_cont.min() > 5) and (contrast > 1.5) and (peak_e < 0.5 * FULL_WELL_E) \
     and (peak0_e < 0.5 * FULL_WELL_E)
print("\nPRE-FLIGHT:", "PASS" if ok else "ATTENTION -- check numbers above")

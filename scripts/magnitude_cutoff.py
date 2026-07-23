#!/usr/bin/env python
"""Estimate grism count rates and SNR to inform the catalog magnitude cutoff.

For a flat-spectrum (AB) source at a given F158 magnitude, computes:
1. Total detected count rate (counts/s) integrated over the 1st-order bandpass
2. Count rate per resolution element (spectral R~461 at 1.45 μm)
3. SNR per resolution element accounting for zodiacal background noise

Physics
-------
An AB flat source has constant f_nu, which in FLAM (f_lambda) is:

    f_lambda = f_nu * c / lambda^2

where f_nu(AB=0) = 3631 Jy = 3.631e-20 erg/s/cm^2/Hz, and the conversion is:

    f_lambda = 0.1089 / (lambda_angstrom)^2   [erg/s/cm^2/Å, for 0 ABmag]

For magnitude m:

    f_lambda(m) = f_lambda(0) * 10^(-0.4 * m)

The grism sensitivity S(lambda) converts FLAM to count rate:

    counts/s = integral[ f_lambda(lambda) * S(lambda) * d_lambda ]

where S(lambda) has units [counts/s / (erg/s/cm^2/Å)], i.e. it already
includes the collecting area, quantum efficiency, and optical throughput.
This is consistent with the disperser pipeline usage (see
notebooks/galaxy/stars_and_galaxies_gpu_demo.ipynb):

    counts_per_pixel = spectrum_flam * sensitivity * dlam_angstroms

SNR per resolution element
--------------------------
For a point source extracted with an optimal aperture of width N_cross pixels
in the cross-dispersion direction, and one resolution element of width
N_disp pixels in the dispersion direction:

    signal = source_counts_per_resel [counts in exptime]
    noise^2 = signal + N_pix * (B_zodi * exptime + read_noise^2)
    SNR = signal / noise

where:
    N_pix = N_cross * N_disp  (pixels per resolution element)
    B_zodi = zodiacal background rate [counts/s/pixel]

The source count rate per resolution element is:

    rate_per_resel = f_lambda(mag) * S(lambda_ref) * delta_lambda_resel

where delta_lambda_resel = lambda_ref / R is the width of one resolution
element in wavelength units.

Usage
-----
    pixi run python scripts/magnitude_cutoff.py
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# AB zeropoint f_lambda coefficient: f_lam = AB_FLAM_CONST / lam_A^2
# [erg/s/cm^2/Å]; canonical value + derivation in roman_disperser.refdata.
from roman_disperser.refdata import FLAM_0AB_COEFF as AB_FLAM_CONST

# Standard Roman grism exposure time (seconds)
EXPTIME = 190.22

# Grism parameters
# Spectral resolution R at reference wavelength (from Roman documentation:
# R ~ 461 * (lambda/1.45um) for 1st order; we use R at the reference wavelength)
LAMBDA_REF_UM = 1.45  # reference wavelength in microns
R_GRISM = 461.0       # spectral resolution at lambda_ref

# Extraction aperture
# Cross-dispersion: PSF FWHM ~ 2 pixels (0.22") at native sampling,
# optimal extraction uses ~3 pixels (1.5 * FWHM)
N_CROSS = 3  # pixels in cross-dispersion direction

# Dispersion direction: one resolution element.
# The 1st-order dispersion is ~10.3 Å/pixel at 1.45 μm (from the optical model:
# the trace spans ~1070 pixels over the 0.9–2.0 μm bandpass = 11000 Å,
# giving ~10.3 Å/pixel). One resolution element at R=461 and λ=1.45 μm
# is Δλ = 14500/461 = 31.5 Å, which spans ~3 pixels.
DISPERSION_A_PER_PIX = 10.3  # Angstroms per pixel (approximate, 1st order)
DELTA_LAMBDA_RESEL_A = (LAMBDA_REF_UM * 1e4) / R_GRISM  # ~31.5 Å
N_DISP = DELTA_LAMBDA_RESEL_A / DISPERSION_A_PER_PIX     # ~3 pixels

# Noise sources
ZODI_RATE = 0.65      # counts/s/pixel (minimum zodiacal background)
READ_NOISE = 5.0      # effective electrons per exposure (up-the-ramp sampling)
                      # CDS (2 reads) gives ~15 e-; up-the-ramp with many reads
                      # beats this down to ~5 e- effective total noise per exposure.

# Survey exposure counts
N_EXP_WIDE = 8        # standard wide survey: 8 exposures per field
N_EXP_DEEP = 32       # deep fields: 4× the wide survey

# Magnitudes to evaluate
MAGNITUDES = np.arange(18, 31, 0.5)


def load_sensitivities(sensitivity_dir, order="1"):
    """Load sensitivity curves for all 18 SCAs.

    Parameters
    ----------
    sensitivity_dir : Path
        Directory containing sensitivity FITS files and sensitivity_map.yaml.
    order : str
        Spectral order to load.

    Returns
    -------
    dict mapping SCA number (int) to (wavelength_A, sensitivity) arrays.
    """
    sensitivity_dir = Path(sensitivity_dir)
    with open(sensitivity_dir / "sensitivity_map.yaml") as f:
        sens_map = yaml.safe_load(f)

    sensitivities = {}
    for sca_key, orders in sens_map.items():
        sca_num = int(sca_key.replace("SCA", ""))
        fname = orders.get(order)
        if fname is None:
            continue
        fpath = sensitivity_dir / fname
        with fits.open(fpath) as hdul:
            wl = hdul[1].data["WAVELENGTH"].astype(np.float64)  # Angstroms
            sens = hdul[1].data["SENSITIVITY"].astype(np.float64)
            sensitivities[sca_num] = (wl, sens)

    return sensitivities


def flat_ab_flam(wavelength_angstrom, magnitude):
    """Compute f_lambda for a flat AB source at given magnitude.

    Parameters
    ----------
    wavelength_angstrom : array
        Wavelengths in Angstroms.
    magnitude : float
        AB magnitude.

    Returns
    -------
    f_lambda : array
        Flux density in erg/s/cm^2/Å.
    """
    return (AB_FLAM_CONST / wavelength_angstrom**2) * 10 ** (-0.4 * magnitude)


def integrate_count_rate(wavelength_angstrom, sensitivity, magnitude):
    """Compute total count rate for a flat AB source.

    Integrates f_lambda(mag) * sensitivity over wavelength using the
    trapezoidal rule.

    Parameters
    ----------
    wavelength_angstrom : array
        Wavelengths in Angstroms.
    sensitivity : array
        Sensitivity curve (counts/s per FLAM).
    magnitude : float
        AB magnitude.

    Returns
    -------
    count_rate : float
        Total counts per second integrated over the bandpass.
    """
    flam = flat_ab_flam(wavelength_angstrom, magnitude)
    return np.trapezoid(flam * sensitivity, wavelength_angstrom)


def count_rate_per_resel(wavelength_angstrom, sensitivity, magnitude,
                         lambda_ref_a, delta_lambda_resel_a):
    """Compute count rate in one resolution element at the reference wavelength.

    Uses the sensitivity interpolated to lambda_ref, not the full integral.

    Parameters
    ----------
    wavelength_angstrom : array
        Sensitivity curve wavelengths in Angstroms.
    sensitivity : array
        Sensitivity curve values.
    magnitude : float
        AB magnitude.
    lambda_ref_a : float
        Reference wavelength in Angstroms.
    delta_lambda_resel_a : float
        Width of one resolution element in Angstroms.

    Returns
    -------
    count_rate : float
        Counts per second in one resolution element.
    """
    sens_at_ref = np.interp(lambda_ref_a, wavelength_angstrom, sensitivity)
    flam_at_ref = flat_ab_flam(lambda_ref_a, magnitude)
    return flam_at_ref * sens_at_ref * delta_lambda_resel_a


def snr_per_resel(source_rate, n_pix, zodi_rate, read_noise, exptime):
    """Compute SNR per resolution element.

    Parameters
    ----------
    source_rate : float
        Source count rate in one resolution element [counts/s].
    n_pix : float
        Number of pixels in the extraction aperture (N_cross × N_disp).
    zodi_rate : float
        Zodiacal background rate [counts/s/pixel].
    read_noise : float
        Read noise per pixel [electrons].
    exptime : float
        Exposure time [seconds].

    Returns
    -------
    snr : float
        Signal-to-noise ratio per resolution element.
    """
    signal = source_rate * exptime
    noise_sq = signal + n_pix * (zodi_rate * exptime + read_noise**2)
    return signal / np.sqrt(noise_sq)


def main():
    project_root = Path(__file__).resolve().parent.parent
    sensitivity_dir = project_root / "data" / "sensitivities"

    # Load 1st-order sensitivities for all SCAs
    print("Loading 1st-order sensitivity curves...")
    sensitivities = load_sensitivities(sensitivity_dir, order="1")
    print(f"  Loaded {len(sensitivities)} SCAs\n")

    # Derived quantities
    lambda_ref_a = LAMBDA_REF_UM * 1e4
    n_pix = N_CROSS * N_DISP

    print("Assumptions:")
    print(f"  Single exposure time: {EXPTIME:.2f} s")
    print(f"  Wide survey:          {N_EXP_WIDE} exposures")
    print(f"  Deep survey:          {N_EXP_DEEP} exposures")
    print(f"  Reference wavelength: {LAMBDA_REF_UM} μm ({lambda_ref_a:.0f} Å)")
    print(f"  Spectral resolution:  R = {R_GRISM:.0f} at {LAMBDA_REF_UM} μm")
    print(f"  Resolution element:   Δλ = {DELTA_LAMBDA_RESEL_A:.1f} Å"
          f" = {N_DISP:.1f} pixels (dispersion)")
    print(f"  Dispersion:           {DISPERSION_A_PER_PIX:.1f} Å/pixel")
    print(f"  Extraction aperture:  {N_CROSS} × {N_DISP:.1f}"
          f" = {n_pix:.1f} pixels per resolution element")
    print(f"  Zodiacal background:  {ZODI_RATE} counts/s/pixel (minimum)")
    print(f"  Read noise:           {READ_NOISE:.0f} e-/exposure (up-the-ramp)")
    print()

    # -----------------------------------------------------------------------
    # Part 1: Total integrated count rates
    # -----------------------------------------------------------------------
    sca_nums = sorted(sensitivities.keys())
    n_sca = len(sca_nums)
    n_mag = len(MAGNITUDES)

    count_rates = np.zeros((n_mag, n_sca))
    for j, sca in enumerate(sca_nums):
        wl, sens = sensitivities[sca]
        for i, mag in enumerate(MAGNITUDES):
            count_rates[i, j] = integrate_count_rate(wl, sens, mag)

    cr_med = np.median(count_rates, axis=1)
    total_counts_med = cr_med * EXPTIME

    print("=" * 85)
    print("PART 1: Total integrated counts (full 1st-order bandpass, median SCA)")
    print("=" * 85)
    print(f"{'Mag':>5s}  {'Rate':>10s}  {'Counts':>12s}")
    print(f"{'':>5s}  {'(ct/s)':>10s}  {'(1 exp)':>12s}")
    print("-" * 30)
    for i, mag in enumerate(MAGNITUDES):
        print(f"{mag:5.1f}  {cr_med[i]:10.2f}  {total_counts_med[i]:12.1f}")

    # -----------------------------------------------------------------------
    # Part 2: Per-resolution-element SNR
    # -----------------------------------------------------------------------
    print()
    print("=" * 85)
    print("PART 2: SNR per resolution element at λ_ref = {:.2f} μm (median SCA)".format(
        LAMBDA_REF_UM))
    print("=" * 85)

    # Compute per-resel rates for median SCA
    # Use median sensitivity at the reference wavelength
    sens_at_ref_all = []
    for sca in sca_nums:
        wl, sens = sensitivities[sca]
        sens_at_ref_all.append(np.interp(lambda_ref_a, wl, sens))
    sens_at_ref_median = np.median(sens_at_ref_all)

    print(f"\nSensitivity at {LAMBDA_REF_UM} μm (median SCA): {sens_at_ref_median:.3e}")
    print()

    # Noise budget at mag=25 for illustration
    mag_example = 25.0
    flam_example = flat_ab_flam(lambda_ref_a, mag_example)
    rate_example = flam_example * sens_at_ref_median * DELTA_LAMBDA_RESEL_A

    print(f"Noise budget at mag={mag_example:.0f} (per resolution element):")
    print()
    scenarios = [
        ("1 exposure", 1),
        (f"Wide ({N_EXP_WIDE} exp)", N_EXP_WIDE),
        (f"Deep ({N_EXP_DEEP} exp)", N_EXP_DEEP),
    ]
    for label, n_exp in scenarios:
        total_exptime = EXPTIME * n_exp
        signal = rate_example * total_exptime
        zodi_counts = n_pix * ZODI_RATE * total_exptime
        # Read noise adds per exposure (each exposure has independent read noise)
        read_counts = n_pix * READ_NOISE**2 * n_exp
        total_noise = np.sqrt(signal + zodi_counts + read_counts)
        snr = signal / total_noise
        print(f"  {label}  (total {total_exptime:.0f}s):")
        print(f"    Source signal:    {signal:8.1f} counts")
        print(f"    Zodi background:  {zodi_counts:8.1f} counts")
        print(f"    Read noise²:      {read_counts:8.1f}"
              f" ({n_pix:.0f} pix × {READ_NOISE:.0f}² × {n_exp} exp)")
        print(f"    Total noise:      {total_noise:8.1f}")
        print(f"    SNR/resel:        {snr:8.1f}")
        print()

    # Full table: show SNR for all three scenarios
    print(f"{'Mag':>5s}  {'Rate/resel':>12s}  {'Signal':>10s}  "
          f"{'SNR 1exp':>10s}  {'SNR wide':>10s}  {'SNR deep':>10s}")
    print(f"{'':>5s}  {'(ct/s)':>12s}  {'(1 exp)':>10s}  "
          f"{'':>10s}  {'({} exp)':>10s}  {'({} exp)':>10s}".format(N_EXP_WIDE, N_EXP_DEEP))
    print("-" * 65)

    snr_1exp = np.zeros(n_mag)
    snr_wide = np.zeros(n_mag)
    snr_deep = np.zeros(n_mag)
    for i, mag in enumerate(MAGNITUDES):
        flam = flat_ab_flam(lambda_ref_a, mag)
        rate = flam * sens_at_ref_median * DELTA_LAMBDA_RESEL_A
        signal_1 = rate * EXPTIME
        snr_1exp[i] = snr_per_resel(rate, n_pix, ZODI_RATE, READ_NOISE, EXPTIME)
        # For N exposures: signal scales as N, noise² = N*(signal_1 + n_pix*zodi*t + n_pix*rn²)
        for n_exp, arr in [(N_EXP_WIDE, snr_wide), (N_EXP_DEEP, snr_deep)]:
            total_t = EXPTIME * n_exp
            signal_n = rate * total_t
            noise_sq = signal_n + n_pix * (ZODI_RATE * total_t + READ_NOISE**2 * n_exp)
            arr[i] = signal_n / np.sqrt(noise_sq)
        print(f"{mag:5.1f}  {rate:12.4f}  {signal_1:10.1f}  "
              f"{snr_1exp[i]:10.1f}  {snr_wide[i]:10.1f}  {snr_deep[i]:10.1f}")

    # -----------------------------------------------------------------------
    # Part 3: SNR thresholds
    # -----------------------------------------------------------------------
    print()
    print("=" * 85)
    print("PART 3: Magnitude at SNR thresholds")
    print("=" * 85)
    print()
    print(f"{'SNR/resel':>10s}  {'1 exp':>8s}  {'Wide':>8s}  {'Deep':>8s}")
    print("-" * 40)
    for snr_threshold in [10, 5, 3, 1, 0.5, 0.1]:
        mags = []
        for arr in [snr_1exp, snr_wide, snr_deep]:
            log_snr = np.log10(arr)
            mag_at = np.interp(np.log10(snr_threshold), log_snr[::-1],
                               MAGNITUDES[::-1])
            mags.append(mag_at)
        print(f"{snr_threshold:10.1f}  {mags[0]:8.1f}  {mags[1]:8.1f}  {mags[2]:8.1f}")

    # Catalog cut recommendation
    print()
    print("=" * 85)
    print("RECOMMENDATION")
    print("=" * 85)
    print()
    print("For a catalog magnitude cut, consider:")
    print("  - SNR/resel ≥ 1: minimum for per-resolution-element detection")
    print("  - SNR/resel ~ 0.1: source contributes ~10% of noise per resel")
    print("    (relevant for contamination modeling)")
    print("  - The deep survey reaches ~1.5 mag deeper than a single exposure")


if __name__ == "__main__":
    main()

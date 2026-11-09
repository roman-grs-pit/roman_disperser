"""Differentiable synthetic photometry and spectral templates (JAX).

The production driver takes spectra as catalogue template x ``flux_scale``
and the F158 magnitudes from the catalogue, both on the host, and the
notebooks normalise templates with synphot. None of that is traceable, so a
fit could not vary a magnitude, a template mixture or a redshift. This
module redoes those steps in JAX:

* :class:`Bandpass` - a throughput curve resampled once (host) onto a fine
  integration grid;
* :func:`ab_mag` / :func:`maggies` - synphot's photon-weighted AB magnitude
  (``Observation.effstim('ABmag')``):
  ``f_nu = int f_lambda T lambda dlambda / int T c / lambda dlambda``;
* :func:`normalise_to_mag` - scale a spectrum to a given AB magnitude;
* :func:`redshift` - shift a rest-frame template to redshift ``z``.

All are differentiable in the spectrum, the magnitude and ``z``.
Wavelengths are in microns and ``f_lambda`` in erg s^-1 cm^-2 A^-1, as
everywhere in roman_disperser.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

C_ANGSTROM = 2.99792458e18          # speed of light, A/s
AB_ZEROPOINT = 48.6                 # m_AB = -2.5 log10(f_nu / cgs) - 48.6


class Bandpass(NamedTuple):
    """Throughput ``thru`` on a uniform integration grid ``wl`` (microns)."""
    wl: jax.Array
    thru: jax.Array


def make_bandpass(wl_um, thru, step_angstroms=2.0):
    """Resample a throughput curve onto a uniform grid (host, once).

    ``step_angstroms`` sets the integration step (default 2 A, well below
    the catalogue SED spacing and the bandpass structure).
    """
    wl_um = np.asarray(wl_um, np.float64)
    thru = np.asarray(thru, np.float64)
    nz = np.nonzero(thru > 0)[0]
    lo, hi = wl_um[max(nz[0] - 1, 0)], wl_um[min(nz[-1] + 1, wl_um.size - 1)]
    n = int(np.ceil((hi - lo) * 1e4 / step_angstroms)) + 1
    grid = np.linspace(lo, hi, n)
    t = np.interp(grid, wl_um, thru, left=0.0, right=0.0)
    return Bandpass(jnp.asarray(grid, jnp.float32), jnp.asarray(t, jnp.float32))


def load_bandpass(name="F158", step_angstroms=2.0):
    """Vendored Roman WFI bandpass (``"F158"`` or ``"F184"``) via synphot."""
    from . import refdata
    loaders = {"F158": refdata.get_f158_band, "F184": refdata.get_f184_band}
    if name not in loaders:
        raise ValueError(f"unknown bandpass {name!r}; have {sorted(loaders)}")
    band = loaders[name]()
    wav = band.waveset.to_value("Angstrom")
    return make_bandpass(wav / 1e4, band(band.waveset).value, step_angstroms)


def _trapz(y, x):
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * jnp.diff(x), axis=-1)


def mean_fnu(wl_um, flam, band):
    """Photon-weighted mean f_nu (erg s^-1 cm^-2 Hz^-1) of ``flam`` [..., N]
    sampled at ``wl_um`` [N] through ``band``. ``flam`` is linearly
    interpolated onto the band grid and is zero outside ``wl_um``."""
    lam = band.wl * 1e4                                         # A
    f = jnp.interp(band.wl, wl_um, flam, left=0.0, right=0.0) \
        if flam.ndim == 1 else jax.vmap(
            lambda s: jnp.interp(band.wl, wl_um, s, left=0.0, right=0.0)
        )(flam.reshape(-1, flam.shape[-1])).reshape(flam.shape[:-1] + lam.shape)
    num = _trapz(f * band.thru * lam, lam)
    den = _trapz(band.thru * C_ANGSTROM / lam, lam)
    return num / den


def ab_mag(wl_um, flam, band):
    """AB magnitude of ``flam`` through ``band`` (+inf for non-positive flux;
    the gradient there is zero, not NaN)."""
    fnu = mean_fnu(wl_um, flam, band)
    pos = fnu > 0
    safe = jnp.where(pos, fnu, 1.0)
    return jnp.where(pos, -2.5 * jnp.log10(safe) - AB_ZEROPOINT, jnp.inf)


def maggies(wl_um, flam, band):
    """Band flux in maggies (``10**(-0.4 m_AB)``), the catalogue F158 unit."""
    return mean_fnu(wl_um, flam, band) * 10 ** (0.4 * AB_ZEROPOINT)


def normalise_to_mag(wl_um, flam, mag, band):
    """Scale ``flam`` [..., N] so its AB magnitude through ``band`` is
    ``mag`` [...]. Differentiable in ``flam`` and ``mag``."""
    scale = 10 ** (-0.4 * (jnp.asarray(mag) - ab_mag(wl_um, flam, band)))
    return flam * scale[..., None]


def redshift(wl_obs_um, wl_rest_um, flam_rest, z):
    """Observed-frame shape of a rest-frame template at redshift ``z``.

    ``f_obs(lambda) = f_rest(lambda / (1 + z)) / (1 + z)`` (flux density per
    unit observed wavelength for a fixed bolometric flux; distance dimming
    is left to the normalisation, e.g. :func:`normalise_to_mag`), sampled at
    ``wl_obs_um``, zero outside the template range. Linear interpolation of
    the template, so the derivative in ``z`` is that of the piecewise-linear
    template (continuous in ``z``; smooth to the extent the template grid
    resolves its features).
    """
    zp1 = 1.0 + jnp.asarray(z)
    return jnp.interp(wl_obs_um / zp1, wl_rest_um, flam_rest,
                      left=0.0, right=0.0) / zp1

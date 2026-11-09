"""Unconstrained reparametrisations for gradient-based fitting.

Several model inputs are clipped to a valid range inside the forward model
(``sersic._sersic_kernel`` clips ``n`` to [0.01, 10], ``b/a`` to [0.01, 1]
and ``r_eff`` to >= 0.1 px; the PSF and wavelength interpolations hold the
edge value outside their grids). A clip has zero gradient on its flat side,
so an optimiser or sampler that steps past a bound gets stuck there.

Fit in unconstrained coordinates instead: the maps here send all of R into
the *open* valid interval, so the clips are never active, and they return
the log-Jacobian needed to put a prior on the physical parameter while
sampling the unconstrained one (HMC/NUTS, VI)::

    params, logdet = sersic_from_unconstrained(u)        # u: [N, 4]
    log_post = log_like(params) + log_prior(params) + logdet.sum()

All functions are pure JAX, elementwise and differentiable.
"""

import jax
import jax.numpy as jnp

# Strictly inside the clip ranges of sersic._sersic_kernel.
SERSIC_N_BOUNDS = (0.2, 8.0)
SERSIC_BA_BOUNDS = (0.05, 1.0)
SERSIC_R_EFF_MIN = 0.1     # oversampled pixels, the kernel's clip


def bounded(u, lo, hi):
    """``(lo, hi)``-valued map of real ``u``; returns (value, log|dv/du|)."""
    s = jax.nn.sigmoid(u)
    v = lo + (hi - lo) * s
    logdet = jnp.log(hi - lo) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)
    return v, logdet


def unbounded(v, lo, hi):
    """Inverse of :func:`bounded`. ``v`` on or past a bound (e.g. a catalogue
    ``b/a`` of exactly 1) is moved 1e-4 of the range inside it."""
    s = jnp.clip((v - lo) / (hi - lo), 1e-4, 1.0 - 1e-4)
    return jnp.log(s) - jnp.log1p(-s)


def positive(u, floor=0.0):
    """``(floor, inf)``-valued softplus map; returns (value, log|dv/du|)."""
    return floor + jax.nn.softplus(u), jax.nn.log_sigmoid(u)


def unpositive(v, floor=0.0):
    """Inverse of :func:`positive` (``v`` at or below ``floor`` is moved
    to ``floor + 1e-4``)."""
    x = jnp.maximum(v - floor, 1e-4)
    return x + jnp.log(-jnp.expm1(-x))


def sersic_from_unconstrained(u, n_bounds=SERSIC_N_BOUNDS,
                              ba_bounds=SERSIC_BA_BOUNDS,
                              r_eff_min=SERSIC_R_EFF_MIN):
    """Map ``u`` [..., 4] to Sérsic ``(r_eff_pix, n, ba, theta)``.

    ``r_eff`` = floor + softplus, ``n`` and ``b/a`` sigmoid-bounded,
    ``theta`` unchanged (periodic; no Jacobian). ``ba_bounds`` may reach 1
    (the kernel clips at 1 but ``b/a`` = 1 is only approached).

    Returns
    -------
    params : tuple of 4 arrays [...]
    logdet : [...] summed log-Jacobian of the map
    """
    r, lr = positive(u[..., 0], r_eff_min)
    n, ln = bounded(u[..., 1], *n_bounds)
    ba, lb = bounded(u[..., 2], *ba_bounds)
    theta = u[..., 3]
    return (r, n, ba, theta), lr + ln + lb


def sersic_to_unconstrained(r_eff_pix, n, ba, theta,
                            n_bounds=SERSIC_N_BOUNDS,
                            ba_bounds=SERSIC_BA_BOUNDS,
                            r_eff_min=SERSIC_R_EFF_MIN):
    """Inverse of :func:`sersic_from_unconstrained` -> [..., 4]."""
    return jnp.stack([unpositive(jnp.asarray(r_eff_pix), r_eff_min),
                      unbounded(jnp.asarray(n), *n_bounds),
                      unbounded(jnp.asarray(ba), *ba_bounds),
                      jnp.asarray(theta, jnp.result_type(float))], axis=-1)

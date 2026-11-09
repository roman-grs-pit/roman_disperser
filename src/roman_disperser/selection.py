"""Differentiable selection functions, for selection terms in a likelihood.

The production driver keeps or drops sources with hard cuts:

* the trace bounding box must overlap the padded detector
  (``catalog.select_sources``, per order);
* a per-order F158 brightness limit (``pipeline.apply_order_mag_limits``);
* implicitly, the light that lands on the chip (the deposit drops the rest).

A forward-model fit wants these as *probabilities* with gradients: in a
hierarchical model the selection enters the likelihood of the selected
sample, typically as ``S(d) / Z(Lambda)`` with ``Z = int S(d) p(d|Lambda) dd``
(e.g. Stiskalek & Desmond's treatment of selection in their forward
models). This module gives the per-source terms as smooth functions of the
same quantities the cuts use, built from the cut code's own arithmetic, so
that ``width -> 0`` reproduces the production masks exactly (``width=0``
returns them as ``log S`` = 0 / -inf):

* :func:`soft_le` / :func:`soft_ge` - Gaussian-smoothed step,
  ``log Phi((limit - x) / width)``; ``width`` is the scatter of the
  selected quantity (photometric error in mag, astrometric error in px);
* :func:`mag_logp` - per-order F158 brightness selection;
* :func:`trace_overlap_logp` - the trace-on-detector cut of
  ``catalog.select_sources`` with soft box edges;
* :func:`on_chip_fraction` - counts-weighted fraction of a source's
  dispersed light that falls on the chip (soft pixel-edge indicator);
* :func:`footprint_logp` - undispersed position inside the chip;
* :func:`log_normaliser` - Monte Carlo ``log Z`` from reparametrised draws.

All return log-probabilities (or fractions) as JAX arrays and are
differentiable in positions, magnitudes and fluxes.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from . import catalog

DETECTOR_SIZE = 4088


def _hard(cond):
    return jnp.where(cond, 0.0, -jnp.inf)


def soft_le(x, limit, width):
    """``log P(x <= limit)`` with Gaussian scatter ``width`` on ``x``.

    ``width`` = 0 (a Python number) gives the hard cut (0 or -inf, ``x <=
    limit`` exactly as the production masks). NaN ``x`` fails, as in
    ``pipeline.apply_order_mag_limits``.
    """
    x = jnp.asarray(x)
    if isinstance(width, (int, float)) and width == 0:
        return _hard(x <= limit)
    out = log_ndtr((limit - x) / width)
    return jnp.where(jnp.isnan(x), -jnp.inf, out)


def soft_ge(x, limit, width):
    """``log P(x >= limit)``; see :func:`soft_le`."""
    x = jnp.asarray(x)
    if isinstance(width, (int, float)) and width == 0:
        return _hard(x >= limit)
    out = log_ndtr((x - limit) / width)
    return jnp.where(jnp.isnan(x), -jnp.inf, out)


def ab_mag_from_maggies(flux):
    """``-2.5 log10(flux)``, +inf for non-positive flux (as
    ``pipeline.flux_to_ab_mag``) with a finite (zero) gradient there."""
    flux = jnp.asarray(flux)
    pos = flux > 0
    return jnp.where(pos, -2.5 * jnp.log10(jnp.where(pos, flux, 1.0)),
                     jnp.inf)


def mag_logp(mag, limit, width=0.0):
    """``log S`` of the per-order brightness cut ``mag <= limit``
    (``limit`` None: no cut, log S = 0)."""
    mag = jnp.asarray(mag)
    if limit is None:
        return jnp.zeros(mag.shape)
    return soft_le(mag, limit, width)


def _box_logp(lo_edge, hi_edge, lo, hi, width):
    """log P(interval [lo_edge, hi_edge] overlaps [lo, hi]): both
    ``hi_edge >= lo`` and ``lo_edge <= hi``."""
    return soft_ge(hi_edge, lo, width) + soft_le(lo_edge, hi, width)


def trace_overlap_logp(payload, xfpa, yfpa, wl_min, wl_max,
                       detector_size=DETECTOR_SIZE, padding=300, n_wl=5,
                       width=0.0):
    """``log S`` of ``catalog.select_sources`` (trace bounding box overlaps
    the padded detector), with soft edges of scale ``width`` px.

    Same bounding box as the production cut (``catalog.trace_bbox``), so
    ``width=0`` gives exactly ``log(select_sources(...))``. Differentiable
    in ``xfpa``, ``yfpa`` (e.g. through ``optical_model_jax.
    perturbed_sky_to_fpa``) and the optical payload.
    """
    x_min, x_max, y_min, y_max = catalog.trace_bbox(
        payload, xfpa, yfpa, wl_min, wl_max, n_wl)
    lo, hi = -padding, detector_size + padding
    return (_box_logp(x_min, x_max, lo, hi, width)
            + _box_logp(y_min, y_max, lo, hi, width))


def _in_chip(v, detector_size, width):
    """Soft indicator of FITS coordinate ``v`` in [0.5, N + 0.5]."""
    if isinstance(width, (int, float)) and width == 0:
        return ((v >= 0.5) & (v <= detector_size + 0.5)).astype(jnp.float32)
    return jnp.exp(log_ndtr((v - 0.5) / width)
                   + log_ndtr((detector_size + 0.5 - v) / width))


def on_chip_fraction(xd, yd, counts, detector_size=DETECTOR_SIZE,
                     width=0.5):
    """Counts-weighted fraction of a dispersed source on the chip.

    ``xd, yd`` [..., N_wl] are the trace (FITS SCA px, e.g.
    ``star_disperser._compute_dispersed_positions``) and ``counts``
    [..., N_wl] the counts per sample. Each sample's indicator is smoothed
    over ``width`` px (default 0.5: about the PSF core; the deposited
    stamp leaves the chip gradually anyway). Returns [...] in [0, 1].
    """
    w = _in_chip(xd, detector_size, width) * _in_chip(yd, detector_size, width)
    tot = jnp.sum(counts, axis=-1)
    return jnp.sum(counts * w, axis=-1) / jnp.where(tot > 0, tot, 1.0)


def footprint_logp(xsca, ysca, detector_size=DETECTOR_SIZE, width=0.0,
                   padding=0.0):
    """``log S`` of the undispersed position lying on the (padded) chip,
    i.e. a sky-position selection for a given pointing and SCA."""
    lo, hi = 0.5 - padding, detector_size + 0.5 + padding
    return (soft_ge(xsca, lo, width) + soft_le(xsca, hi, width)
            + soft_ge(ysca, lo, width) + soft_le(ysca, hi, width))


def log_normaliser(log_s_fn, draws):
    """Monte Carlo ``log Z = log E_p[S]`` from draws of the population.

    ``draws`` is any pytree whose leaves have a leading sample axis K,
    generated by reparametrisation (``mu + sigma * eps`` with fixed
    ``eps``) so ``log Z`` is differentiable in the hyperparameters;
    ``log_s_fn(draws) -> [K]`` is the log selection probability of each
    draw. Returns ``logsumexp(log S) - log K``.
    """
    ls = log_s_fn(draws)
    return logsumexp(ls) - jnp.log(ls.shape[0])

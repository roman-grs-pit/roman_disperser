"""Differentiable likelihoods and detector terms for forward-model fits.

The simulator's noise step (Poisson draws) and the romanisim detector stage
(ramps, non-linearity, saturation, IPC, read noise) are not differentiable
and should not be differentiated through. A fit compares the noiseless
model (e-/s, e.g. ``linear_operator.SceneOperator``) with the data through
a likelihood instead. The linear detector terms that matter at the model
level are applied in JAX:

* :func:`ipc_convolve` - inter-pixel capacitance as a (3x3) convolution;
* :func:`poisson_loglike` - exact Poisson log-likelihood of counts;
* :func:`gaussian_loglike` - Gaussian (e.g. rate images with variance maps,
  including read noise);
* :func:`selected_loglike` - adds the selection terms of
  :mod:`roman_disperser.selection` (``sum log S_i - N log Z``).

All are pure JAX and differentiable in the model (and in the IPC kernel).
"""

import jax
import jax.numpy as jnp
import jax.scipy.signal
from jax.scipy.special import gammaln


def ipc_convolve(image, kernel):
    """Apply an IPC kernel (odd size, typically 3x3 summing to ~1) to a
    [ny, nx] image; zero padding at the edges ('same' output).

    ``kernel[c + dy, c + dx]`` is the fraction of a pixel's charge that
    appears ``(dy, dx)`` pixels away (``c`` the centre index), i.e. a
    convolution.
    """
    return jax.scipy.signal.convolve2d(image, kernel, mode="same",
                                       precision=jax.lax.Precision.HIGHEST)


def _masked_sum(x, mask):
    return jnp.sum(x if mask is None else jnp.where(mask, x, 0.0))


def poisson_loglike(model_rate, data_counts, exptime, background_rate=0.0,
                    mask=None, floor=1e-12):
    """``sum d log(mu) - mu - log d!`` with ``mu = (model + bkg) * exptime``.

    ``floor`` keeps ``log(mu)`` finite (and its gradient bounded) where the
    model is zero. ``mask`` (bool, True = use) drops pixels.
    """
    mu = jnp.maximum((model_rate + background_rate) * exptime, floor)
    d = data_counts
    ll = d * jnp.log(mu) - mu - gammaln(d + 1.0)
    return _masked_sum(ll, mask)


def gaussian_loglike(model, data, variance, mask=None):
    """``-0.5 sum [(d - m)^2 / var + log(2 pi var)]`` (``mask`` as above)."""
    r2 = (data - model) ** 2 / variance
    ll = -0.5 * (r2 + jnp.log(2.0 * jnp.pi * variance))
    return _masked_sum(ll, mask)


def selected_loglike(loglike, log_s, log_z):
    """Likelihood of a selected sample: ``loglike + sum(log_s) - N log_z``.

    ``log_s`` [N] are the per-source log selection probabilities of the
    observed sources (:mod:`roman_disperser.selection`) and ``log_z`` the
    population normaliser (``selection.log_normaliser``). ``log_s`` is
    constant in the parameters when selection depends only on the data; it
    is kept so the same expression covers selection on latent quantities.
    """
    log_s = jnp.asarray(log_s)
    return loglike + jnp.sum(log_s) - log_s.shape[0] * log_z

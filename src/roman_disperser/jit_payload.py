"""
Split PSF / optical payloads into jit arguments and compile-time constants.

Why this exists
---------------
The payload dicts (``psf_model.get_or_make_psf_payload``,
``optical_model_jax.make_sca_payload``) mix three kinds of leaves:

1. **Arrays** — PSF grid, polynomial coefficients, detector constants. These
   differ per SCA but have the *same shapes* on every SCA.
2. **Compile-time structure** — ints that set array shapes inside the traced
   code (``jnp.arange(map_i)`` etc. in ``optical_model_jax``), the PSF
   ``oversample`` (stamp geometry), and the ``wl.transform`` string that picks
   a Python branch in ``trace_beam``. These cannot be traced, and are the same
   for every SCA of an element.
3. **Labels** — ``detector``, ``order``, ``stpsf_filter``, ``timing``, ... .
   Never read by the traced code, but differ per SCA.

Capturing a whole payload in a ``jax.jit`` closure (the original pattern)
embeds the arrays as constants in the compiled program, so every (SCA, order)
is a distinct XLA program: one compile (~30-70 s on A10G) and one ~111 MB
persistent-cache entry each. Passing (1) as arguments and (2) as a hashable
static value — and dropping (3) — makes the program depend only on shapes and
structure, so one compile serves every SCA and pointing.

The static keys are an explicit *whitelist*: any non-array leaf the traced
code needs but which is not listed here raises a ``KeyError`` at trace time
(loud), rather than silently re-keying the compile cache per SCA (which a
generic "keep every non-array leaf" rule would do with ``detector``).
"""

import numpy as np
import jax

# Non-array leaves the traced code reads, by payload kind, as key paths.
PSF_STATIC_KEYS = (("oversample",),)
OPTICAL_STATIC_KEYS = (
    ("wl", "transform"),
    ("det", "naxis1"), ("det", "naxis2"),
    ("poly", "map_i"), ("poly", "map_j"),
    ("poly", "crv_i"), ("poly", "crv_j"), ("poly", "crv_k"),
    ("poly", "ids_i"), ("poly", "ids_j"), ("poly", "ids_k"),
)


def _is_array(x):
    return isinstance(x, (jax.Array, np.ndarray))


def _array_leaves(d):
    """Nested dict -> same nesting with only the array leaves kept."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub = _array_leaves(v)
            if sub:
                out[k] = sub
        elif _is_array(v):
            out[k] = v
    return out


def split_payload(payload, static_keys):
    """Split a payload into ``(arrays, static)``.

    Parameters
    ----------
    payload : dict
        PSF or optical payload (possibly nested).
    static_keys : tuple of tuple of str
        Key paths of the non-array leaves to keep as compile-time constants
        (``PSF_STATIC_KEYS`` or ``OPTICAL_STATIC_KEYS``).

    Returns
    -------
    arrays : dict
        Pytree of the array leaves (pass to jit as an argument).
    static : tuple of (path, value)
        Hashable; pass to jit as a static argument. Values are coerced to
        plain Python ``int``/``str`` so equal payloads hash equal.
    """
    static = []
    for path in static_keys:
        v = payload
        for k in path:
            v = v[k]
        static.append((path, v if isinstance(v, str) else int(v)))
    return _array_leaves(payload), tuple(static)


def merge_payload(arrays, static):
    """Inverse of :func:`split_payload` (minus the dropped labels).

    Returns a nested dict the dispersion functions can index exactly as they
    index the original payload. Works inside a traced function (``arrays``
    leaves are tracers there).
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in arrays.items()}
    for path, v in static:
        d = out
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = v
    return out


def split_payloads(psf_payload, optical_payload):
    """Split both payloads; returns ``(arrays, static)`` for the pair.

    ``arrays = (psf_arrays, optical_arrays)``,
    ``static = (psf_static, optical_static)``.
    """
    pa, ps = split_payload(psf_payload, PSF_STATIC_KEYS)
    oa, os_ = split_payload(optical_payload, OPTICAL_STATIC_KEYS)
    return (pa, oa), (ps, os_)


def merge_payloads(arrays, static):
    """Inverse of :func:`split_payloads`: returns ``(psf_payload, optical_payload)``."""
    return (merge_payload(arrays[0], static[0]),
            merge_payload(arrays[1], static[1]))

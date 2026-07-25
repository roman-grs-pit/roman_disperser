"""
Catalog utilities for source selection and assignment to detectors.
"""

import jax.numpy as jnp

from . import modes
from . import optical_model_jax as omj


def select_sources(payload, xfpa, yfpa, wl_min=None, wl_max=None,
                   detector_size=4088, padding=300, n_wl=5, mode=None):
    """Return boolean mask of sources whose dispersed trace overlaps the detector.

    Traces each source at multiple wavelengths between wl_min and wl_max,
    computes the bounding box of the trace, and checks overlap with the
    padded detector region.

    The band matters: culling over a *narrower* range than is actually
    dispersed produces false negatives — sources whose trace only enters the
    detector at the wavelengths that were skipped are dropped silently. Hence
    the defaults follow the mode rather than being fixed at the grism's band.

    Args:
        payload: dict from make_sca_payload (specific SCA and order)
        xfpa, yfpa: FPA coordinates in degrees (1D arrays, from get_fpa_pos).
            Must be non-empty.
        wl_min: minimum wavelength in microns (default: the mode's lam_min)
        wl_max: maximum wavelength in microns (default: the mode's lam_max)
        detector_size: detector size in pixels (default 4088)
        padding: padding in pixels around detector edges (default 300)
        n_wl: number of wavelength samples for trace bounding box (default 5)
        mode: dispersing element (default grism); supplies the band defaults

    Returns:
        mask: boolean JAX array [N_sources], True if trace overlaps detector

    Raises:
        ValueError: if xfpa or yfpa is empty
    """
    if wl_min is None or wl_max is None:
        _mode = modes.get_mode(mode)
        wl_min = _mode.lam_min if wl_min is None else wl_min
        wl_max = _mode.lam_max if wl_max is None else wl_max

    n = xfpa.shape[0]
    if n == 0:
        raise ValueError("xfpa and yfpa must be non-empty")

    # Sample wavelengths across the range to capture trace curvature
    wavelengths = jnp.linspace(wl_min, wl_max, n_wl)

    # Tile sources × wavelengths: [n_wl * n]
    xfpa_tiled = jnp.tile(xfpa, n_wl)
    yfpa_tiled = jnp.tile(yfpa, n_wl)
    wl_tiled = jnp.repeat(wavelengths, n)

    xmpa, ympa = omj.trace_beam(payload, xfpa_tiled, yfpa_tiled, wl_tiled)
    xsca, ysca = omj.mpa_to_sca(payload, xmpa, ympa)

    # Reshape to [n_wl, n] and compute bounding box across wavelengths
    xsca = xsca.reshape(n_wl, n)
    ysca = ysca.reshape(n_wl, n)

    x_min = xsca.min(axis=0)
    x_max = xsca.max(axis=0)
    y_min = ysca.min(axis=0)
    y_max = ysca.max(axis=0)

    # Padded detector region
    lo = -padding
    hi = detector_size + padding

    # Overlap: trace bbox intersects detector bbox
    x_overlap = (x_max >= lo) & (x_min <= hi)
    y_overlap = (y_max >= lo) & (y_min <= hi)

    return x_overlap & y_overlap

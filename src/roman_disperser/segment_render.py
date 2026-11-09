"""Trace-aligned segment rendering of stars and galaxies (prototype).

The production dispersers (:mod:`star_disperser`, :mod:`galaxy_disperser`)
deposit one full stamp per wavelength sample: a 47x47 (star) or 77x77
(galaxy) native stamp, linearly interpolated in wavelength between the PSF
grid nodes, is scattered at every sample's dispersed position. The image is

    I = sum_lambda c_lambda * box_os[ K_lambda shifted to x(lambda), y(lambda) ]

where ``K_lambda = sum_k h_k(lambda) K_k`` is the stamp interpolated linearly
between wavelength nodes ``lambda_k`` (hat weights ``h_k``) and ``box_os[]``
is the exact box-overlap deposit of the oversampled stamp (the ``interp``
deposit). Because everything is linear, the sum can be regrouped per node:

    I = sum_k  K_k (*) L_k,   L_k = sum_lambda c_lambda h_k(lambda) delta(x - x(lambda))

``L_k`` is a thin "line image" of the counts laid along the (tilted, curved)
trace over the support of hat ``k`` (two node intervals), and ``(*)`` is a
2-D convolution. This module renders exactly that:

1. lay the counts of every sample along the trace onto an oversampled line
   image per node, with bilinear (tent) weights in the oversampled pixel
   index (sub-pixel exact; see below);
2. convolve each node's line image with the node's oversampled kernel by
   FFT (stars: the PSF; galaxies: PSF (*) the Jacobian-warped galaxy image,
   as a three-way product in Fourier space, so the PSF (*) galaxy image is
   never formed per wavelength);
3. bin the oversampled result to native pixels and add each node window to
   the detector.

**Exactness.** With the oversampled-index tent deposit, step 1-3 is
algebraically the ``interp`` deposit: a subpixel box of width 1/os placed at
a continuous offset overlaps two oversampled cells with exactly the tent
weights, and native pixels are unions of oversampled cells. If the nodes are
*all* PSF grid wavelengths (``n_nodes = N_grid``, 56 in production) the hat
weights equal the deposit's own wavelength interpolation, so the result is
the production ``interp`` image up to float32 FFT rounding (and the
deposit's own wavelength sampling is kept: no extra approximation). Fewer
nodes (a subset of the grid wavelengths) approximate the PSF's wavelength
dependence by a coarser piecewise-linear function; that is the only
approximation, and its size is the speed/accuracy knob.

**PSF lookup position** (``psf_at``). ``"source"`` (default) interpolates
the spatially varying PSF at the source's direct-image position, as the
production code does. ``"landing"`` interpolates node ``k``'s PSF at the
trace position of ``lambda_k``, i.e. where that light lands on the detector
(36-412 px from the direct image on SCA 5). Which is physically right
depends on where the field-dependent aberrations arise (before the grism:
field angle = direct position; after it: landing position); the switch lets
the effect be measured.

**Cost.** Per source and order: ``n_nodes`` FFT convolutions over a window
that holds the trace part of one hat plus the kernel (3 transforms per node
for stars, 4 for galaxies), instead of ``N_lambda`` stamp scatters, the
per-source 16-phase binning of the stamp stack and, for galaxies, the
PSF (*) galaxy convolution at all 56 grid wavelengths.

Static geometry (window sizes, FFT shapes, node indices) lives in a
hashable :class:`SegmentPlan` built on the host by :func:`make_segment_plan`
from the payloads; one plan per (element, order, n_nodes, kind, psf_at)
serves every SCA (window sizes are computed with a margin over the probed
source positions; samples outside a window would be dropped, which the
plan's margin prevents and the tests check via flux conservation).
"""

import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import galaxy_disperser
from . import jit_payload
from . import optical_model_jax as omj
from . import pipeline
from . import star_disperser

PSF_AT_MODES = ("source", "landing")


# ---------------------------------------------------------------------------
# Window overflow check
# ---------------------------------------------------------------------------
# Window sizes are fixed on the host (static shapes); light that a sample
# would put outside its window is dropped silently by the scatter. The plans'
# margins prevent that for the probed geometry, but a fit that moves sources
# (or a plan reused with other payloads) can break it, and then both the
# image and its gradient are wrong without any error. With the check on,
# each renderer adds a host callback that raises WindowOverflowError when
# more than WINDOW_DROP_RTOL of a source's counts would be dropped. It is a
# trace-time switch (on: one host round trip per render), off by default for
# production speed; ROMAN_DISPERSER_CHECK_WINDOWS=1 turns it on.

WINDOW_DROP_RTOL = 1e-5
_CHECK_WINDOWS = [os.environ.get("ROMAN_DISPERSER_CHECK_WINDOWS", "0") == "1"]


class WindowOverflowError(RuntimeError):
    """A renderer would drop light outside its static window."""


def set_window_check(enabled):
    """Turn the window-overflow check on/off (affects later traces only;
    already-compiled functions keep the setting they were traced with)."""
    _CHECK_WINDOWS[0] = bool(enabled)


def window_check_enabled():
    return _CHECK_WINDOWS[0]


def _raise_if_dropped(dropped, total, where):
    dropped = float(dropped)
    total = float(total)
    if dropped > WINDOW_DROP_RTOL * max(total, 1e-30):
        raise WindowOverflowError(
            f"{where}: {dropped:.4g} of {total:.4g} counts "
            f"({dropped / max(total, 1e-30):.2%}) fall outside the static "
            f"window and would be dropped; rebuild the plan with a larger "
            f"margin (or for these positions)")


def check_dropped(dropped, total, where):
    """Raise (at run time) if ``dropped`` > rtol x ``total``; no-op unless
    the check is enabled at trace time."""
    if _CHECK_WINDOWS[0]:
        jax.debug.callback(partial(_raise_if_dropped, where=where),
                           jnp.sum(dropped), jnp.sum(total))
KINDS = ("star", "galaxy")


class SegmentPlan(NamedTuple):
    """Static (hashable) geometry of a segment render.

    Attributes
    ----------
    kind : {"star", "galaxy"}
    node_idx : tuple of int
        Indices of the node wavelengths in the PSF grid wavelengths.
    oversample : int
    psf_shape : (int, int)
        Oversampled PSF stamp shape.
    image_shape : (int, int)
        Oversampled galaxy image shape ((1, 1) for stars).
    line_shape : (int, int)
        Oversampled line-window shape per node (multiples of ``oversample``).
    fft_shape : (int, int)
        FFT size (>= line + kernel - 1 per axis; a fast size).
    out_shape : (int, int)
        Native output window per node.
    psf_at : {"source", "landing"}
    node_batch : int or None
        Nodes per vectorised block (``jax.lax.map`` batch size); None = all
        nodes at once.
    """
    kind: str
    node_idx: tuple
    oversample: int
    psf_shape: tuple
    image_shape: tuple
    line_shape: tuple
    fft_shape: tuple
    out_shape: tuple
    psf_at: str = "source"
    node_batch: object = None

    @property
    def kernel_shape(self):
        """Oversampled shape of the effective stamp (PSF or PSF (*) image)."""
        return (self.psf_shape[0] + self.image_shape[0] - 1,
                self.psf_shape[1] + self.image_shape[1] - 1)

    @property
    def n_nodes(self):
        return len(self.node_idx)


def segment_node_indices(n_grid, n_nodes):
    """Node indices into a grid of ``n_grid`` wavelengths: ``n_nodes``
    (clipped to [2, n_grid]) indices spread evenly from first to last, as a
    tuple of Python ints. ``n_nodes >= n_grid`` returns every grid index."""
    n_nodes = int(min(max(int(n_nodes), 2), n_grid))
    idx = np.unique(np.round(np.linspace(0, n_grid - 1, n_nodes)).astype(int))
    return tuple(int(i) for i in idx)


def _hat_extents(optical_payload, node_wl, wl_range, positions, n_fine):
    """Largest x and y extent (native px) of the trace inside each hat's
    support, over the probe positions. Host-side helper of the plan."""
    lo, hi = wl_range
    wl = np.linspace(lo, hi, n_fine).astype(np.float32)
    xs = np.asarray([p[0] for p in positions], np.float32)
    ys = np.asarray([p[1] for p in positions], np.float32)
    n_p = xs.size
    xd, yd = omj.trace_sca(optical_payload, jnp.asarray(xs)[:, None],
                           jnp.asarray(ys)[:, None], jnp.asarray(wl)[None, :])
    xd = np.asarray(xd, np.float64).reshape(n_p, wl.size)
    yd = np.asarray(yd, np.float64).reshape(n_p, wl.size)
    node_wl = np.asarray(node_wl, np.float64)
    m = node_wl.size
    ext = np.zeros((m, 2))
    for k in range(m):
        a = node_wl[k - 1] if k > 0 else -np.inf
        b = node_wl[k + 1] if k < m - 1 else np.inf
        sel = (wl >= a - 1e-6) & (wl <= b + 1e-6)
        if not sel.any():
            continue
        ext[k, 0] = (yd[:, sel].max(axis=1) - yd[:, sel].min(axis=1)).max()
        ext[k, 1] = (xd[:, sel].max(axis=1) - xd[:, sel].min(axis=1)).max()
    return ext.max(axis=0)  # (y, x)


def make_segment_plan(psf_payload, optical_payload, n_nodes, kind="star", *,
                      galaxy_npix=None, psf_at="source", wl_range=None,
                      margin_px=2, extent_factor=1.05, probe=9,
                      node_batch=None):
    """Build the static :class:`SegmentPlan` for one (element, order).

    Parameters
    ----------
    psf_payload, optical_payload : dict
        As for the production dispersers. Only shapes, the grid wavelengths
        and the trace geometry are read (host-side).
    n_nodes : int
        Number of wavelength nodes (a subset of the PSF grid wavelengths,
        always including the first and last). ``N_grid`` (56) = every grid
        wavelength = the production ``interp`` image.
    kind : {"star", "galaxy"}
    galaxy_npix : int
        Oversampled galaxy image size (required for galaxies; 120 in BDI).
    psf_at : {"source", "landing"}
        PSF lookup position, see the module docstring.
    wl_range : (float, float), optional
        Wavelength range (microns) the deposit samples can span; default
        the PSF grid range. Edge hats extend to it.
    margin_px : int
        Extra native pixels per axis in the line window.
    extent_factor : float
        Safety factor on the probed trace extents (the probe grid does not
        hit every source position).
    probe : int
        Probe positions per axis (``probe x probe`` grid from -300 to
        naxis + 300 px).
    node_batch : int, optional
        Nodes per vectorised block (memory knob); None = all at once.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if psf_at not in PSF_AT_MODES:
        raise ValueError(f"psf_at must be one of {PSF_AT_MODES}, got {psf_at!r}")
    os_ = int(psf_payload["oversample"])
    psf_shape = tuple(int(s) for s in psf_payload["psf_grid"].shape[-2:])
    if kind == "galaxy":
        if galaxy_npix is None:
            raise ValueError("galaxy_npix is required for kind='galaxy'")
        image_shape = (int(galaxy_npix), int(galaxy_npix))
    else:
        image_shape = (1, 1)
    grid_wl = np.asarray(psf_payload["wavelengths"], np.float64)
    node_idx = segment_node_indices(grid_wl.size, n_nodes)
    if wl_range is None:
        wl_range = (float(grid_wl[0]), float(grid_wl[-1]))
    naxis = (int(optical_payload["det"]["naxis1"]),
             int(optical_payload["det"]["naxis2"]))
    px = np.linspace(-300.0, naxis[0] + 300.0, probe)
    py = np.linspace(-300.0, naxis[1] + 300.0, probe)
    positions = [(x, y) for y in py for x in px]
    ext = _hat_extents(optical_payload, grid_wl[list(node_idx)], wl_range,
                       positions, n_fine=4001)
    # Oversampled index span of one hat: ext*os, +1 for the tent's upper
    # cell, + (os-1) because the window starts at a native boundary, +1 so
    # that floor() rounding never falls off; then margin and round to os.
    line_shape = tuple(
        os_ * int(np.ceil((np.ceil(e * extent_factor * os_) + os_ + 1) / os_)
                  + margin_px)
        for e in ext)
    s_y = psf_shape[0] + image_shape[0] - 1
    s_x = psf_shape[1] + image_shape[1] - 1
    full = (line_shape[0] + s_y - 1, line_shape[1] + s_x - 1)
    fft_shape = tuple(galaxy_disperser._next_fast_fft_size(n) for n in full)
    out_shape = tuple(-(-n // os_) for n in full)
    return SegmentPlan(kind=kind, node_idx=node_idx, oversample=os_,
                       psf_shape=psf_shape, image_shape=image_shape,
                       line_shape=line_shape, fft_shape=fft_shape,
                       out_shape=out_shape, psf_at=psf_at,
                       node_batch=node_batch)


# ---------------------------------------------------------------------------
# Traced pieces
# ---------------------------------------------------------------------------

def _psf_at_nodes(psf_payload, node_idx, x, y):
    """Spatially interpolated PSF at each node wavelength: [M, S_y, S_x].

    ``x, y`` are scalars (one lookup position) or [M] (per-node positions).
    The arithmetic is that of ``psf_model.interpolate_psf_spatial``.
    """
    from . import psf_model
    grid = psf_payload["psf_grid"][:, :, jnp.asarray(node_idx)]
    sub = {"spatial_x": psf_payload["spatial_x"],
           "spatial_y": psf_payload["spatial_y"]}
    m = len(node_idx)
    # A scalar position is broadcast to every node: the per-node form lets
    # XLA fuse the node gather into the blend (measured ~20 % faster per
    # source than blending the gathered [Ny, Nx, M, S, S] sub-grid).
    x = jnp.broadcast_to(x, (m,))
    y = jnp.broadcast_to(y, (m,))

    def one(xk, yk, g_k):
        return psf_model.interpolate_psf_spatial(
            dict(sub, psf_grid=g_k[:, :, None]), xk, yk)[0]
    return jax.vmap(one, in_axes=(0, 0, 2))(x, y, grid)


def _hat_weights(wavelengths, node_wl):
    """Bracketing node ``k`` and weight ``t`` of node ``k+1`` per sample;
    the same formula as the deposit's wavelength interpolation (searchsorted
    - 1, clipped; t clipped to [0, 1])."""
    m = node_wl.shape[0]
    k = jnp.clip(jnp.searchsorted(node_wl, wavelengths) - 1, 0, m - 2)
    t = (wavelengths - node_wl[k]) / (node_wl[k + 1] - node_wl[k])
    return k, jnp.clip(t, 0.0, 1.0)


def _line_entries(plan, wavelengths, counts, xd, yd, node_wl):
    """Scatter entries of the per-node line images (see :func:`_line_images`).

    Returns
    -------
    (idx_n, idx_y, idx_x), vals : flat [8 N] arrays, laid out as
        ``reshape(4, 2, N)`` = (tent cell, hat, sample): node, oversampled
        row/column inside the node's line window, value. The 8 entries of
        one sample are distinct cells.
    origin_y, origin_x : [M] int32 native index (0-based) of each window.
    """
    os_ = plan.oversample
    m = plan.n_nodes
    s_y, s_x = plan.kernel_shape
    # Continuous oversampled index of the lower edge of stamp subpixel 0
    # (FITS centre c: native pixel i spans [i + 0.5, i + 1.5)).
    u_y = (yd - 0.5) * os_ - s_y / 2.0
    u_x = (xd - 0.5) * os_ - s_x / 2.0
    q_y = jnp.floor(u_y)
    q_x = jnp.floor(u_x)
    f_y = u_y - q_y
    f_x = u_x - q_x
    q_y = q_y.astype(jnp.int32)
    q_x = q_x.astype(jnp.int32)

    k, t = _hat_weights(wavelengths, node_wl)
    node = jnp.concatenate([k, k + 1])
    w = jnp.concatenate([counts * (1.0 - t), counts * t])
    qy2 = jnp.concatenate([q_y, q_y])
    qx2 = jnp.concatenate([q_x, q_x])
    fy2 = jnp.concatenate([f_y, f_y])
    fx2 = jnp.concatenate([f_x, f_x])

    big = jnp.iinfo(jnp.int32).max
    qmin_y = jax.ops.segment_min(qy2, node, num_segments=m)
    qmin_x = jax.ops.segment_min(qx2, node, num_segments=m)
    qmin_y = jnp.where(qmin_y == big, 0, qmin_y)  # node without samples
    qmin_x = jnp.where(qmin_x == big, 0, qmin_x)
    origin_y = jnp.floor_divide(qmin_y, os_)
    origin_x = jnp.floor_divide(qmin_x, os_)
    r_y = qy2 - os_ * origin_y[node]
    r_x = qx2 - os_ * origin_x[node]

    idx_n = jnp.concatenate([node] * 4)
    idx_y = jnp.concatenate([r_y, r_y, r_y + 1, r_y + 1])
    idx_x = jnp.concatenate([r_x, r_x + 1, r_x, r_x + 1])
    vals = jnp.concatenate([
        w * (1.0 - fy2) * (1.0 - fx2), w * (1.0 - fy2) * fx2,
        w * fy2 * (1.0 - fx2), w * fy2 * fx2])
    if _CHECK_WINDOWS[0]:
        l_y, l_x = plan.line_shape
        inside = ((idx_y >= 0) & (idx_y < l_y) & (idx_x >= 0) & (idx_x < l_x))
        check_dropped(jnp.where(inside, 0.0, jnp.abs(vals)), jnp.abs(vals),
                      "segment line image")
    return (idx_n, idx_y, idx_x), vals, origin_y, origin_x


def _line_images(plan, wavelengths, counts, xd, yd, node_wl):
    """Per-node oversampled line images and their native window origins.

    Returns
    -------
    lines : [M, L_y, L_x] float
    origin_y, origin_x : [M] int32 native index (0-based) of each window.
    """
    (idx_n, idx_y, idx_x), vals, origin_y, origin_x = _line_entries(
        plan, wavelengths, counts, xd, yd, node_wl)
    lines = jnp.zeros((plan.n_nodes,) + tuple(plan.line_shape),
                      counts.dtype).at[idx_n, idx_y, idx_x].add(
        vals, mode="drop", wrap_negative_indices=False)
    return lines, origin_y, origin_x


def _convolve_bin(plan, lines, kernels):
    """FFT-convolve each line image with its kernel(s) and bin to native.

    ``kernels`` is a tuple of [M, ., .] arrays whose full linear convolution
    is the effective stamp (PSF for stars; PSF and warped galaxy image).
    Returns [M, W_y, W_x] native windows.
    """
    os_ = plan.oversample
    n_y, n_x = plan.fft_shape
    w_y, w_x = plan.out_shape
    l_y, l_x = plan.line_shape
    s_y, s_x = plan.kernel_shape
    full_y, full_x = l_y + s_y - 1, l_x + s_x - 1

    def block(args):
        lines_b, kernels_b = args
        spec = jnp.fft.rfft2(lines_b, s=(n_y, n_x))
        for kern in kernels_b:
            spec = spec * jnp.fft.rfft2(kern, s=(n_y, n_x))
        img = jnp.fft.irfft2(spec, s=(n_y, n_x))[..., :full_y, :full_x]
        img = jnp.pad(img, [(0, 0)] * (img.ndim - 2)
                      + [(0, os_ * w_y - full_y), (0, os_ * w_x - full_x)])
        return img.reshape(img.shape[:-2] + (w_y, os_, w_x, os_)).sum(
            axis=(-3, -1))

    if plan.node_batch is None:
        return block((lines, kernels))

    def one(args):
        lines_1, kernels_1 = args
        return block((lines_1[None], tuple(k[None] for k in kernels_1)))[0]
    return jax.lax.map(one, (lines, kernels), batch_size=int(plan.node_batch))


def _add_windows(output, windows, origin_y, origin_x):
    """Scatter-add the [M, W_y, W_x] native windows at their origins."""
    m, w_y, w_x = windows.shape
    iy = origin_y[:, None, None] + jnp.arange(w_y)[None, :, None]
    ix = origin_x[:, None, None] + jnp.arange(w_x)[None, None, :]
    iy = jnp.broadcast_to(iy, windows.shape)
    ix = jnp.broadcast_to(ix, windows.shape)
    return output.at[iy.ravel(), ix.ravel()].add(
        windows.ravel(), mode="drop", wrap_negative_indices=False)


def _node_positions(optical_payload, x0, y0, node_wl):
    return star_disperser._compute_dispersed_positions(
        optical_payload, x0, y0, node_wl)


def render_star_segments(psf_payload, optical_payload, xsca, ysca,
                         wavelengths, counts, output, plan, dispersed=None):
    """Render one star by trace-aligned segments (see the module docstring).

    Same contract as ``star_disperser.disperse_star_psf`` (``counts`` =
    per-sample counts, already multiplied by sensitivity and dlam), plus the
    static ``plan`` from :func:`make_segment_plan` (``kind="star"``).
    ``dispersed`` = precomputed dispersed positions at ``wavelengths``.
    """
    return _add_windows(output, *star_node_windows(
        psf_payload, optical_payload, xsca, ysca, wavelengths, counts, plan,
        dispersed=dispersed))


def star_node_windows(psf_payload, optical_payload, xsca, ysca, wavelengths,
                      counts, plan, dispersed=None, line_fn=None):
    """Per-node native windows of one star: ``(windows [M, W_y, W_x],
    origin_y [M], origin_x [M])``; :func:`render_star_segments` adds them
    to the detector. ``line_fn`` replaces :func:`_line_images` (same
    signature; ``roman_disperser.accumulate`` passes an ordered variant).
    """
    line_fn = _line_images if line_fn is None else line_fn
    if plan.kind != "star":
        raise ValueError("plan.kind must be 'star'")
    grid_wl = psf_payload["wavelengths"]
    node_wl = grid_wl[jnp.asarray(plan.node_idx)]
    if dispersed is None:
        dispersed = star_disperser._compute_dispersed_positions(
            optical_payload, xsca, ysca, wavelengths)
    xd, yd = dispersed
    if plan.psf_at == "landing":
        nx, ny = _node_positions(optical_payload, xsca, ysca, node_wl)
        psfs = _psf_at_nodes(psf_payload, plan.node_idx, nx, ny)
    else:
        psfs = _psf_at_nodes(psf_payload, plan.node_idx, xsca, ysca)
    lines, oy, ox = line_fn(plan, wavelengths, counts, xd, yd, node_wl)
    return _convolve_bin(plan, lines, (psfs,)), oy, ox


def render_galaxy_segments(psf_payload, optical_payload, image, x0, y0,
                           counts, wavelengths, output, plan, dispersed=None,
                           jacobians=None):
    """Render one galaxy by trace-aligned segments.

    Same contract as ``galaxy_disperser.disperse_galaxy`` (``image`` at the
    PSF oversampling, ``counts`` per sample), plus the static ``plan``
    (``kind="galaxy"``, ``galaxy_npix = image.shape[0]``). Per node, the
    galaxy image is warped by the trace Jacobian at the node wavelength
    (as the production code does at every grid wavelength) and the
    effective stamp PSF (*) warped image enters only through the product of
    their spectra with the line image's.

    ``jacobians`` = precomputed [M, 2, 2] trace Jacobians at the node
    wavelengths (``galaxy_disperser.galaxy_trace_geometry(...)[2]``).
    """
    return _add_windows(output, *galaxy_node_windows(
        psf_payload, optical_payload, image, x0, y0, counts, wavelengths,
        plan, dispersed=dispersed, jacobians=jacobians))


def galaxy_node_windows(psf_payload, optical_payload, image, x0, y0, counts,
                        wavelengths, plan, dispersed=None, jacobians=None,
                        line_fn=None):
    """Per-node native windows of one galaxy (see :func:`star_node_windows`)."""
    line_fn = _line_images if line_fn is None else line_fn
    if plan.kind != "galaxy":
        raise ValueError("plan.kind must be 'galaxy'")
    if tuple(image.shape) != tuple(plan.image_shape):
        raise ValueError(f"image shape {image.shape} != plan {plan.image_shape}")
    os_ = plan.oversample
    grid_wl = psf_payload["wavelengths"]
    node_wl = grid_wl[jnp.asarray(plan.node_idx)]
    if dispersed is None:
        dispersed = star_disperser._compute_dispersed_positions(
            optical_payload, x0, y0, wavelengths)
    xd, yd = dispersed
    if jacobians is None:
        jacobians = galaxy_disperser.galaxy_trace_geometry(
            optical_payload, x0, y0, node_wl)[2]
    if plan.psf_at == "landing":
        nx, ny = _node_positions(optical_payload, x0, y0, node_wl)
        psfs = _psf_at_nodes(psf_payload, plan.node_idx, nx, ny)
    else:
        psfs = _psf_at_nodes(psf_payload, plan.node_idx, x0, y0)
    warped = jax.vmap(lambda j: galaxy_disperser.disperse_galaxy_shape(
        image, j, 1.0 / os_, 1.0 / os_))(jacobians)
    lines, oy, ox = line_fn(plan, wavelengths, counts, xd, yd, node_wl)
    return _convolve_bin(plan, lines, (psfs, warped)), oy, ox


# ---------------------------------------------------------------------------
# Jitted single-source and batch entry points (payloads as jit arguments)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("static", "plan"))
def render_star_jit(arrays, xsca, ysca, wavelengths, counts, output, *,
                    static, plan):
    """Jitted :func:`render_star_segments` with the payloads as arguments
    (``arrays, static = jit_payload.split_payloads(psf, optical)``)."""
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    return render_star_segments(psf_payload, optical_payload, xsca, ysca,
                                wavelengths, counts, output, plan)


@partial(jax.jit, static_argnames=("static", "plan"))
def render_galaxy_jit(arrays, image, x0, y0, counts, wavelengths, output, *,
                      static, plan):
    """Jitted :func:`render_galaxy_segments` with the payloads as arguments."""
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    return render_galaxy_segments(psf_payload, optical_payload, image, x0, y0,
                                  counts, wavelengths, output, plan)


@partial(jax.jit, static_argnames=("static", "plan", "sed_rebin"),
         donate_argnames=("output",))
def run_star_batch_segment(arrays, sens, wavelengths, dlam_angstroms,
                           n_sources, spectra, xsca, ysca, output, *,
                           static, plan, sed_rebin=1):
    """Segment-render the first ``n_sources`` stars of a padded batch.

    Same contract as ``pipeline.run_star_batch`` (same ``_source_loop``,
    count rebinning and padded-row handling; reverse-differentiable), with
    the static ``plan`` in place of ``chunk_size``/``deposit``.
    """
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    _, wl_deposit = pipeline.rebin_counts(sens, wavelengths, sed_rebin)

    def geometry_fn(i):
        return star_disperser._compute_dispersed_positions(
            optical_payload, xsca[i], ysca[i], wl_deposit)

    def body_fn(i, dispersed, output):
        counts = spectra[i] * sens * dlam_angstroms
        counts, _ = pipeline.rebin_counts(counts, wavelengths, sed_rebin)
        return render_star_segments(
            psf_payload, optical_payload, xsca[i], ysca[i], wl_deposit,
            counts, output, plan, dispersed=dispersed)
    return pipeline._source_loop(geometry_fn, body_fn, n_sources,
                                 xsca.shape[0], output)


@partial(jax.jit, static_argnames=("static", "plan", "sed_rebin"),
         donate_argnames=("output",))
def run_galaxy_batch_segment(arrays, sens, wavelengths, dlam_angstroms,
                             n_sources, spectra, xsca, ysca, images, output,
                             *, static, plan, sed_rebin=1):
    """Segment-render the first ``n_sources`` galaxies of a padded batch.

    Same contract as ``pipeline.run_galaxy_batch``.
    """
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    _, wl_deposit = pipeline.rebin_counts(sens, wavelengths, sed_rebin)
    node_wl = psf_payload["wavelengths"][jnp.asarray(plan.node_idx)]

    def geometry_fn(i):
        jac = galaxy_disperser.galaxy_trace_geometry(
            optical_payload, xsca[i], ysca[i], node_wl)[2]
        return jac, star_disperser._compute_dispersed_positions(
            optical_payload, xsca[i], ysca[i], wl_deposit)

    def body_fn(i, geom, output):
        counts = spectra[i] * sens * dlam_angstroms
        counts, _ = pipeline.rebin_counts(counts, wavelengths, sed_rebin)
        return render_galaxy_segments(
            psf_payload, optical_payload, images[i], xsca[i], ysca[i],
            counts, wl_deposit, output, plan, dispersed=geom[1],
            jacobians=geom[0])
    return pipeline._source_loop(geometry_fn, body_fn, n_sources,
                                 xsca.shape[0], output)


def make_batched_star_segment(psf_payload, optical_payload, sens,
                              wavelengths_jax, dlam_angstroms, plan,
                              sed_rebin=1):
    """Batch function with the signature of ``pipeline.make_batched_star_fori``'s
    result: ``run(n_sources, spectra, xsca, ysca, output) -> output``, so it
    drops into ``pipeline.disperse_batched_stars``."""
    arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
    dlam = jnp.float32(dlam_angstroms)

    def run(n_sources, spectra, xsca, ysca, output):
        return run_star_batch_segment(
            arrays, sens, wavelengths_jax, dlam, n_sources, spectra, xsca,
            ysca, output, static=static, plan=plan, sed_rebin=int(sed_rebin))
    return run


def make_batched_galaxy_segment(psf_payload, optical_payload, sens,
                                wavelengths_jax, dlam_angstroms, plan,
                                sed_rebin=1):
    """Galaxy analogue of :func:`make_batched_star_segment`:
    ``run(n_sources, spectra, xsca, ysca, images, output) -> output``, for
    ``pipeline.disperse_batched_galaxies`` /
    ``pipeline.disperse_batched_sersic_galaxies``."""
    arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
    dlam = jnp.float32(dlam_angstroms)

    def run(n_sources, spectra, xsca, ysca, images, output):
        return run_galaxy_batch_segment(
            arrays, sens, wavelengths_jax, dlam, n_sources, spectra, xsca,
            ysca, images, output, static=static, plan=plan,
            sed_rebin=int(sed_rebin))
    return run

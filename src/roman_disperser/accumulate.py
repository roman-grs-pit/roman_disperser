"""Per-source windows and deterministic accumulation (prototype).

The production batch functions (``pipeline.run_star_batch`` /
``run_galaxy_batch`` and the segment variants) add every source straight
into one full-frame [4088, 4088] accumulator, one source after another,
with scatter-adds whose index lists contain duplicates (successive
wavelength samples overlap). That has two costs:

* the loop over sources is serial, so a batch cannot use parallel hardware
  beyond what one source offers;
* on GPU a scatter-add with duplicate indices is lowered to atomic float
  adds, whose order varies from run to run, so the frame differs run to run
  at the float32 rounding level (F4 / I26).

This module renders each source into its own small dense window and adds
the windows to the frame in a fixed order, with no scatter that has
duplicate indices anywhere on the forward path:

**Window rendering** (:func:`render_star_window`, :func:`render_galaxy_window`).
A source's window covers its whole trace plus the stamp (static shape from
a :class:`WindowPlan`, built on the host from the trace geometry); its
origin is the (dynamic) lower corner. Two renderers:

* ``"deposit"``: the production ``interp``/``floor`` stamp deposit
  (``star_disperser.native_stamps``, same arithmetic), into the window
  instead of the frame;
* ``"segment"``: the trace-aligned segment render
  (``segment_render.*_node_windows``), whose node windows are added into the
  source window in node order.

Inside a window, the per-sample stamps (or the segment render's line-image
tents) still overlap. With ``inner="ordered"`` the samples are split into
``stride`` interleaved classes (sample s in class s mod stride); the plan
chooses the stride so that two samples of one class never touch the same
pixel, so each class is one scatter with *unique* indices (``unique_indices``
lets XLA write without atomics) and the classes are added in a fixed order.
``inner="scatter"`` keeps the production duplicate-index scatter (faster on
CPU, deterministic there, not on GPU).

**Frame accumulation.** Two modes:

* ``"window"`` (:func:`run_star_batch_window` / :func:`run_galaxy_batch_window`):
  same interface as ``pipeline.run_star_batch`` (drop-in for
  ``disperse_batched_stars`` / ``disperse_batched_sersic_galaxies``). The
  batch is processed in groups of ``vmap_width`` sources; a group is
  rendered with ``vmap`` (parallel), then its windows are added to the frame
  one after another (each add a unique-index scatter: a dense window, no
  atomics). The frame sum order is the source order; sort sources with
  :func:`canonical_order` to make it independent of catalogue order.
  The frame does not depend on the batch size. It does depend, at the
  float32 rounding level, on ``vmap_width`` on GPU (each width is its own
  compiled program); ``vectorize=False`` removes that too, at the cost of
  rendering the group's sources one after another.
* ``"tiled"`` (:func:`run_star_tiles`, :func:`stitch_tiles`,
  :class:`TiledDisperser`): the detector is split into ``n_tiles x n_tiles``
  tiles; each source belongs to the tile holding its window origin; each
  tile accumulates its sources (canonical order) into its own buffer (tile
  plus window-sized halo), tiles in parallel (``lax.map`` / ``vmap`` over
  tiles, or ``shard_map`` over devices); the buffers are then added to the
  frame in tile order. Each pixel's sum order is fixed by the tile layout
  alone, so the result does not depend on catalogue order, batch sizes or
  the number of devices.

Both are reverse-differentiable (scan + cond + vmap + unique scatters; the
group / round steps are checkpointed).
"""

from functools import partial
from typing import NamedTuple
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from . import galaxy_disperser
from . import jit_payload
from . import optical_model_jax as omj
from . import pipeline
from . import segment_render as sr
from . import star_disperser

RENDERERS = ("deposit", "segment")
INNER_MODES = ("ordered", "scatter")
ACCUMULATE_MODES = ("window", "tiled")
KINDS = ("star", "galaxy")


class WindowPlan(NamedTuple):
    """Static (hashable) description of a per-source window render.

    Attributes
    ----------
    kind : {"star", "galaxy"}
    renderer : {"deposit", "segment"}
    window_shape : (int, int)
        Native window per source (covers the trace and the stamp).
    oversample : int
    stamp_shape : (int, int)
        Oversampled stamp shape of the deposit (PSF for stars, PSF (*)
        galaxy image for galaxies); unused by the segment renderer.
    deposit : {"interp", "floor"}
    inner : {"ordered", "scatter"}
    stride : int
        Number of interleaved sample classes for ``inner="ordered"``.
    n_samples : int
        Deposit samples per source (after count rebinning) the stride was
        computed for.
    segment : SegmentPlan or None
    vmap_width : int
        Sources rendered in parallel per group (``"window"`` mode).
    vectorize : bool
        True: the sources of a group are rendered with ``vmap`` (one
        program over the group; the GPU setting); False: with ``lax.map``
        (one after another, as the production loop; faster on CPU).
    chunk_size : int
        Samples per chunk for ``renderer="deposit"``, ``inner="scatter"``.
    """
    kind: str
    renderer: str
    window_shape: tuple
    oversample: int
    stamp_shape: tuple
    deposit: str = "interp"
    inner: str = "ordered"
    stride: int = 1
    n_samples: int = 0
    segment: object = None
    vmap_width: int = 8
    vectorize: bool = True
    chunk_size: int = 2000


# ---------------------------------------------------------------------------
# Plan (host side)
# ---------------------------------------------------------------------------

def _probe_trace(optical_payload, wavelengths, probe=9, pad=300.0):
    """Dispersed positions [n_probe, N] (float64 numpy) of the samples at a
    ``probe x probe`` grid of source positions over the SCA (+- ``pad``)."""
    naxis = (int(optical_payload["det"]["naxis1"]),
             int(optical_payload["det"]["naxis2"]))
    px = np.linspace(-pad, naxis[0] + pad, probe)
    py = np.linspace(-pad, naxis[1] + pad, probe)
    xs = np.repeat(px[None, :], probe, 0).ravel().astype(np.float32)
    ys = np.repeat(py[:, None], probe, 1).ravel().astype(np.float32)
    wl = np.asarray(wavelengths, np.float32)
    xf, yf = omj.sca_to_fpa(optical_payload, jnp.asarray(xs), jnp.asarray(ys))
    n_p = xs.size
    xf = jnp.broadcast_to(jnp.asarray(xf)[:, None], (n_p, wl.size)).ravel()
    yf = jnp.broadcast_to(jnp.asarray(yf)[:, None], (n_p, wl.size)).ravel()
    wl_b = jnp.broadcast_to(jnp.asarray(wl)[None, :], (n_p, wl.size)).ravel()
    xm, ym = omj.trace_beam(optical_payload, xf, yf, wl_b)
    xd, yd = omj.mpa_to_sca(optical_payload, xm, ym)
    return (np.asarray(xd, np.float64).reshape(n_p, wl.size),
            np.asarray(yd, np.float64).reshape(n_p, wl.size))


def min_stride(xd, yd, thr_y, thr_x):
    """Smallest class stride g such that any two samples g, 2g, ... apart
    are separated by at least ``thr_y`` in y or ``thr_x`` in x (native px)
    at every probed position; ``N`` (one sample per class) if none is."""
    n = xd.shape[1]
    for g in range(1, n):
        ok = True
        for k in range(1, (n - 1) // g + 1):
            d = k * g
            sep = np.maximum(np.abs(yd[:, d:] - yd[:, :-d]) - thr_y,
                             np.abs(xd[:, d:] - xd[:, :-d]) - thr_x)
            if sep.min() < 0:
                ok = False
                break
        if ok:
            return g
    return max(n, 1)


def make_window_plan(psf_payload, optical_payload, kind, wavelengths, *,
                     renderer="deposit", deposit="interp", n_nodes=None,
                     galaxy_npix=None, psf_at="source", inner="ordered",
                     vmap_width=8, vectorize=True, chunk_size=2000, probe=9,
                     extent_factor=1.05, margin_px=2, stride_margin_px=1.0):
    """Build the static :class:`WindowPlan` for one (SCA, order, kind).

    Parameters
    ----------
    psf_payload, optical_payload : dict
    kind : {"star", "galaxy"}
    wavelengths : array [N]
        The deposit sample wavelengths (microns; after count rebinning),
        host values. Window extent and ordered stride are computed from them.
    renderer : {"deposit", "segment"}
    deposit : {"interp", "floor"}
        Deposit scheme for ``renderer="deposit"``.
    n_nodes : int
        Segment nodes (``renderer="segment"``).
    galaxy_npix : int
        Oversampled galaxy image size (galaxies).
    psf_at : {"source", "landing"}
        Segment PSF lookup position.
    inner : {"ordered", "scatter"}
        Accumulation inside a window (see the module docstring).
    vmap_width : int
        Sources rendered in parallel per group.
    probe, extent_factor, margin_px, stride_margin_px
        Safety margins of the window size (probed trace extents x factor +
        margin) and of the ordered stride (separation threshold +
        ``stride_margin_px``).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if renderer not in RENDERERS:
        raise ValueError(f"renderer must be one of {RENDERERS}, got {renderer!r}")
    if inner not in INNER_MODES:
        raise ValueError(f"inner must be one of {INNER_MODES}, got {inner!r}")
    if deposit not in star_disperser.DEPOSIT_MODES:
        raise ValueError(f"deposit must be one of {star_disperser.DEPOSIT_MODES}")
    if kind == "galaxy" and galaxy_npix is None:
        raise ValueError("galaxy_npix is required for kind='galaxy'")
    os_ = int(psf_payload["oversample"])
    psf_shape = tuple(int(s) for s in psf_payload["psf_grid"].shape[-2:])
    if kind == "galaxy":
        stamp_shape = (psf_shape[0] + int(galaxy_npix) - 1,
                       psf_shape[1] + int(galaxy_npix) - 1)
    else:
        stamp_shape = psf_shape
    wl = np.asarray(wavelengths, np.float64)
    xd, yd = _probe_trace(optical_payload, wl, probe=probe)
    ext = (float((yd.max(1) - yd.min(1)).max()),
           float((xd.max(1) - xd.min(1)).max()))
    span = [int(np.ceil(e * extent_factor)) for e in ext]
    seg_plan = None
    if renderer == "segment":
        if n_nodes is None:
            raise ValueError("n_nodes is required for renderer='segment'")
        seg_plan = sr.make_segment_plan(
            psf_payload, optical_payload, n_nodes, kind,
            galaxy_npix=galaxy_npix, psf_at=psf_at,
            wl_range=(float(wl.min()), float(wl.max())))
        # node origins lie within [o, o + span + 1] of the source origin o
        window = tuple(span[a] + 2 + seg_plan.out_shape[a] + margin_px
                       for a in range(2))
        thr = (3.0 / os_ + stride_margin_px / 2, 3.0 / os_ + stride_margin_px / 2)
    else:
        n_y, n_x = star_disperser._native_stamp_size(*stamp_shape, os_)
        grow = 1 if deposit == "interp" else 0
        # stamp base m in [floor(u) - 1, floor(u)], origin = floor(min u) - 1
        window = (span[0] + 3 + n_y + grow + margin_px,
                  span[1] + 3 + n_x + grow + margin_px)
        thr = (n_y + grow + 2 + stride_margin_px,
               n_x + grow + 2 + stride_margin_px)
    stride = min_stride(xd, yd, thr[0], thr[1]) if inner == "ordered" else 1
    return WindowPlan(kind=kind, renderer=renderer, window_shape=window,
                      oversample=os_, stamp_shape=stamp_shape,
                      deposit=deposit, inner=inner, stride=int(stride),
                      n_samples=int(wl.size), segment=seg_plan,
                      vmap_width=int(vmap_width), vectorize=bool(vectorize),
                      chunk_size=int(chunk_size))


# ---------------------------------------------------------------------------
# Traced pieces: window origin, ordered scatter, renderers
# ---------------------------------------------------------------------------

def window_origin(plan, xd, yd):
    """Native (0-based) lower corner (oy, ox) of a source's window from its
    dispersed sample positions; the same arithmetic as the stamp indices."""
    os_ = plan.oversample
    if plan.renderer == "segment":
        s_y, s_x = plan.segment.kernel_shape
        q_y = jnp.floor((yd - 0.5) * os_ - s_y / 2.0).astype(jnp.int32)
        q_x = jnp.floor((xd - 0.5) * os_ - s_x / 2.0).astype(jnp.int32)
        return (jnp.floor_divide(jnp.min(q_y), os_),
                jnp.floor_divide(jnp.min(q_x), os_))
    s_y, s_x = plan.stamp_shape
    u_y = yd - 0.5 + (-(s_y - 1) / (2.0 * os_))
    u_x = xd - 0.5 + (-(s_x - 1) / (2.0 * os_))
    return (jnp.floor(jnp.min(u_y)).astype(jnp.int32) - 1,
            jnp.floor(jnp.min(u_x)).astype(jnp.int32) - 1)


def add_window(target, window, oy, ox):
    """``target[oy:oy+h, ox:ox+w] += window`` (parts outside dropped) as a
    scatter with unique, sorted indices: no atomics on GPU."""
    h, w = window.shape[-2:]
    iy = (oy + jnp.arange(h))[:, None]
    ix = (ox + jnp.arange(w))[None, :]
    return target.at[iy, ix].add(window, mode="drop", unique_indices=True,
                                 indices_are_sorted=True,
                                 wrap_negative_indices=False)


def add_windows_ordered(target, windows, oy, ox):
    """Add [P, h, w] windows at origins [P] one after another (fixed order)."""
    def step(target, args):
        return add_window(target, *args), None
    target, _ = jax.lax.scan(step, target, (windows, oy, ox))
    return target


def _class_layout(n, stride):
    """Sample index [J, G] of class c (column) member j (row); -1 = padding."""
    g = int(stride)
    j = -(-n // g)
    idx = np.arange(j * g)
    idx[idx >= n] = -1
    return jnp.asarray(idx.reshape(j, g))


def _deposit_window(plan, binned, grid_wl, wavelengths, counts, xd, yd):
    """Deposit renderer: (window, oy, ox)."""
    oy, ox = window_origin(plan, xd, yd)
    w_y, w_x = plan.window_shape
    window = jnp.zeros((w_y, w_x), counts.dtype)
    os_ = plan.oversample
    n_y, n_x = binned.shape[-2:]
    if plan.deposit == "interp":
        n_y, n_x = n_y + 1, n_x + 1
    k_y = jnp.arange(n_y)
    k_x = jnp.arange(n_x)

    def stamps(idx):
        nat, m_y, m_x = star_disperser.native_stamps(
            binned, plan.stamp_shape, grid_wl, wavelengths[idx], counts[idx],
            xd[idx], yd[idx], os_, plan.deposit)
        return nat, m_y - oy, m_x - ox

    n = wavelengths.shape[0]
    if plan.inner == "scatter":
        idx = jnp.arange(n)
        nat, ly, lx = stamps(idx)
        iy = jnp.broadcast_to(ly[:, None, None] + k_y[None, :, None], nat.shape)
        ix = jnp.broadcast_to(lx[:, None, None] + k_x[None, None, :], nat.shape)
        window = window.at[iy.ravel(), ix.ravel()].add(
            nat.ravel(), mode="drop", wrap_negative_indices=False)
        return window, oy, ox

    if n != plan.n_samples:
        raise ValueError(f"plan made for {plan.n_samples} samples, got {n}")
    layout = _class_layout(n, plan.stride)       # [J, G]
    # Stamps of all samples at once (as the scatter path), laid out
    # [J, G, ...]; class c = column c, scattered with unique indices.
    valid = layout >= 0
    nat, ly, lx = stamps(jnp.where(valid, layout, 0).ravel())
    j, g = layout.shape
    nat = nat.reshape((j, g) + nat.shape[1:])
    ly = jnp.where(valid, ly.reshape(j, g), w_y)  # padding -> dropped
    lx = lx.reshape(j, g)

    def one_class(window, c):
        nat_c, ly_c, lx_c = nat[:, c], ly[:, c], lx[:, c]
        iy = jnp.broadcast_to(ly_c[:, None, None] + k_y[None, :, None],
                              nat_c.shape)
        ix = jnp.broadcast_to(lx_c[:, None, None] + k_x[None, None, :],
                              nat_c.shape)
        window = window.at[iy.ravel(), ix.ravel()].add(
            nat_c.ravel(), mode="drop", unique_indices=True,
            wrap_negative_indices=False)
        return window, None

    window, _ = jax.lax.scan(one_class, window, jnp.arange(g))
    return window, oy, ox


def _ordered_line_fn(stride):
    """``segment_render._line_images`` replacement whose tents are added in
    ``stride`` classes of unique-index scatters."""
    def line_fn(plan, wavelengths, counts, xd, yd, node_wl):
        (i_n, i_y, i_x), vals, oy, ox = sr._line_entries(
            plan, wavelengths, counts, xd, yd, node_wl)
        n = counts.shape[0]
        l_y = plan.line_shape[0]

        def per_sample(a):                       # [8N] -> [N, 8]
            return a.reshape(8, n).T
        i_n, i_y, i_x, vals = map(per_sample, (i_n, i_y, i_x, vals))
        layout = _class_layout(n, stride)
        lines = jnp.zeros((plan.n_nodes,) + tuple(plan.line_shape),
                          counts.dtype)

        def one_class(lines, col):
            valid = (col >= 0)[:, None]
            idx = jnp.where(col >= 0, col, 0)
            yy = jnp.where(valid, i_y[idx], l_y)
            lines = lines.at[i_n[idx].ravel(), yy.ravel(),
                             i_x[idx].ravel()].add(
                vals[idx].ravel(), mode="drop", unique_indices=True,
                wrap_negative_indices=False)
            return lines, None

        lines, _ = jax.lax.scan(one_class, lines, layout.T)
        return lines, oy, ox
    return line_fn


def _segment_window(plan, node_windows, xd, yd):
    """Assemble the node windows into the source window (node order)."""
    wins, noy, nox = node_windows
    oy, ox = window_origin(plan, xd, yd)
    window = jnp.zeros(plan.window_shape, wins.dtype)
    window = add_windows_ordered(window, wins, noy - oy, nox - ox)
    return window, oy, ox


def _line_fn(plan):
    return _ordered_line_fn(plan.stride) if plan.inner == "ordered" else None


def render_star_window(psf_payload, optical_payload, xsca, ysca, wavelengths,
                       counts, plan, dispersed=None):
    """One star's native window: ``(window [W_y, W_x], oy, ox)``.

    ``counts`` per sample (spectrum x sensitivity x dlam, rebinned) at
    ``wavelengths``; ``dispersed`` = precomputed dispersed positions.
    Adding the window at (oy, ox) gives the production image of
    ``disperse_star_psf`` (``renderer="deposit"``) or of
    ``segment_render.render_star_segments`` up to float32 summation order.
    """
    if dispersed is None:
        dispersed = star_disperser._compute_dispersed_positions(
            optical_payload, xsca, ysca, wavelengths)
    xd, yd = dispersed
    if plan.renderer == "segment":
        return _segment_window(plan, sr.star_node_windows(
            psf_payload, optical_payload, xsca, ysca, wavelengths, counts,
            plan.segment, dispersed=dispersed, line_fn=_line_fn(plan)), xd, yd)
    os_ = plan.oversample
    if 'psf_grid_binned' in psf_payload:
        binned = psf_model_interp(psf_payload, xsca, ysca, binned=True)
    else:
        binned = star_disperser.bin_stack_phases(
            psf_model_interp(psf_payload, xsca, ysca), os_)
    return _deposit_window(plan, binned, psf_payload['wavelengths'],
                           wavelengths, counts, xd, yd)


def psf_model_interp(psf_payload, x, y, binned=False):
    from . import psf_model
    if binned:
        return psf_model.interpolate_psf_spatial(
            psf_payload, x, y, grid_key='psf_grid_binned')
    return psf_model.interpolate_psf_spatial(psf_payload, x, y)


def render_galaxy_window(psf_payload, optical_payload, image, x0, y0, counts,
                         wavelengths, plan, geometry=None, dispersed=None):
    """One galaxy's native window: ``(window, oy, ox)``.

    ``geometry``: for ``renderer="deposit"`` the
    ``galaxy_trace_geometry`` at the PSF grid wavelengths; for
    ``"segment"`` the [M, 2, 2] Jacobians at the node wavelengths.
    """
    if dispersed is None:
        dispersed = star_disperser._compute_dispersed_positions(
            optical_payload, x0, y0, wavelengths)
    xd, yd = dispersed
    if plan.renderer == "segment":
        return _segment_window(plan, sr.galaxy_node_windows(
            psf_payload, optical_payload, image, x0, y0, counts, wavelengths,
            plan.segment, dispersed=dispersed, jacobians=geometry,
            line_fn=_line_fn(plan)), xd, yd)
    os_ = plan.oversample
    convolved, _, _, grid_wl = galaxy_disperser.prepare_galaxy_images(
        optical_payload, psf_payload, image, x0, y0, 1.0 / os_, 1.0 / os_,
        geometry=geometry)
    binned = star_disperser.bin_stack_phases(convolved, os_)
    return _deposit_window(plan, binned, grid_wl, wavelengths, counts, xd, yd)


def _galaxy_geometry(plan, psf_payload, optical_payload, x0, y0):
    if plan.renderer == "segment":
        node_wl = psf_payload["wavelengths"][jnp.asarray(plan.segment.node_idx)]
        return galaxy_disperser.galaxy_trace_geometry(
            optical_payload, x0, y0, node_wl)[2]
    return galaxy_disperser.galaxy_trace_geometry(
        optical_payload, x0, y0, psf_payload['wavelengths'])


# ---------------------------------------------------------------------------
# "window" mode: batch functions with the pipeline.run_*_batch interface
# ---------------------------------------------------------------------------

def _groups(a, v):
    """Pad the leading axis of ``a`` to a multiple of ``v``; [G, v, ...]."""
    b = a.shape[0]
    g = -(-b // v)
    pad = g * v - b
    if pad:
        a = jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], a.dtype)])
    return a.reshape((g, v) + a.shape[1:])


def _window_loop(geometry_fn, render_fn, n_sources, batch_args, output, width,
                 vectorize=True):
    """Groups of ``width`` sources: geometry (vmapped, outside the cond),
    render (vmapped), then the group's windows are added in source order.
    Groups past ``n_sources`` are skipped; padded rows inside the last group
    get zero windows."""
    # The group width is the plan's alone, never the batch size: the vmapped
    # program depends on its width (XLA fuses and orders float ops per
    # shape; on an L40S the phase-binned PSF of a vmapped group differs from
    # a mapped one by ~8e-8 of peak, and width 2 differs from widths 3/4/8
    # by ~1.3e-7 of the frame peak). With a fixed width the same program
    # renders every source, so the frame does not depend on the batch size
    # (short batches are padded). >= 2: a width-1 vmap compiles differently
    # again. Across *different* widths the frame is bit-identical only with
    # vectorize=False (lax.map) or on CPU.
    v = max(2, int(width))
    grouped = [_groups(a, v) for a in batch_args]
    n_groups = grouped[0].shape[0]

    def step(output, gi):
        args = [a[gi] for a in grouped]
        rows = gi * v + jnp.arange(v)
        valid = rows < n_sources
        # lax.map, not vmap: XLA:CPU evaluates the vmapped trace
        # polynomials differently at the float32 rounding level (positions
        # off by up to ~1e-3 px, i.e. ~4e-4 of a star's peak with interp);
        # mapped they are bit-identical to the production loop's.
        geom = jax.lax.map(lambda a: geometry_fn(*a), args)

        def body(output):
            if vectorize:
                win, oy, ox = jax.vmap(render_fn)(geom, *args)
            else:
                win, oy, ox = jax.lax.map(lambda a: render_fn(*a),
                                          (geom, *args))
            win = jnp.where(valid[:, None, None], win, 0.0)
            return add_windows_ordered(output, win, oy, ox)
        output = jax.lax.cond(gi * v < n_sources, body, lambda o: o, output)
        return output, None

    output, _ = jax.lax.scan(jax.checkpoint(step), output,
                             jnp.arange(n_groups))
    return output


@partial(jax.jit, static_argnames=("static", "plan", "sed_rebin"),
         donate_argnames=("output",))
def run_star_batch_window(arrays, sens, wavelengths, dlam_angstroms,
                          n_sources, spectra, xsca, ysca, output, *,
                          static, plan, sed_rebin=1):
    """Window-mode star batch; same contract as ``pipeline.run_star_batch``
    (``plan`` from :func:`make_window_plan`, ``kind="star"``)."""
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    _, wl_dep = pipeline.rebin_counts(sens, wavelengths, sed_rebin)

    def geometry_fn(spec, x, y):
        return star_disperser._compute_dispersed_positions(
            optical_payload, x, y, wl_dep)

    def render_fn(dispersed, spec, x, y):
        counts = spec * sens * dlam_angstroms
        counts, _ = pipeline.rebin_counts(counts, wavelengths, sed_rebin)
        return render_star_window(psf_payload, optical_payload, x, y, wl_dep,
                                  counts, plan, dispersed=dispersed)
    return _window_loop(geometry_fn, render_fn, n_sources,
                        (spectra, xsca, ysca), output, plan.vmap_width,
                        plan.vectorize)


@partial(jax.jit, static_argnames=("static", "plan", "sed_rebin"),
         donate_argnames=("output",))
def run_galaxy_batch_window(arrays, sens, wavelengths, dlam_angstroms,
                            n_sources, spectra, xsca, ysca, images, output, *,
                            static, plan, sed_rebin=1):
    """Window-mode galaxy batch; same contract as
    ``pipeline.run_galaxy_batch``."""
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    _, wl_dep = pipeline.rebin_counts(sens, wavelengths, sed_rebin)

    def geometry_fn(spec, x, y, img):
        return (_galaxy_geometry(plan, psf_payload, optical_payload, x, y),
                star_disperser._compute_dispersed_positions(
                    optical_payload, x, y, wl_dep))

    def render_fn(geom, spec, x, y, img):
        counts = spec * sens * dlam_angstroms
        counts, _ = pipeline.rebin_counts(counts, wavelengths, sed_rebin)
        return render_galaxy_window(psf_payload, optical_payload, img, x, y,
                                    counts, wl_dep, plan, geometry=geom[0],
                                    dispersed=geom[1])
    return _window_loop(geometry_fn, render_fn, n_sources,
                        (spectra, xsca, ysca, images), output,
                        plan.vmap_width, plan.vectorize)


def _arrays(psf_payload, optical_payload, plan, hoist_psf):
    arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
    if hoist_psf and plan.kind == "star" and plan.renderer == "deposit":
        pa, oa = arrays
        pa = dict(pa, psf_grid_binned=star_disperser.bin_psf_grid(
            pa['psf_grid'], plan.oversample))
        arrays = (pa, oa)
    return arrays, static


def make_batched_star_window(psf_payload, optical_payload, sens,
                             wavelengths_jax, dlam_angstroms, plan,
                             sed_rebin=1, hoist_psf=True):
    """``run(n_sources, spectra, xsca, ysca, output)`` for
    ``pipeline.disperse_batched_stars`` (window mode)."""
    arrays, static = _arrays(psf_payload, optical_payload, plan, hoist_psf)
    dlam = jnp.float32(dlam_angstroms)

    def run(n_sources, spectra, xsca, ysca, output):
        return run_star_batch_window(
            arrays, sens, wavelengths_jax, dlam, n_sources, spectra, xsca,
            ysca, output, static=static, plan=plan, sed_rebin=int(sed_rebin))
    return run


def make_batched_galaxy_window(psf_payload, optical_payload, sens,
                               wavelengths_jax, dlam_angstroms, plan,
                               sed_rebin=1):
    """``run(n_sources, spectra, xsca, ysca, images, output)`` for
    ``pipeline.disperse_batched_sersic_galaxies`` (window mode)."""
    arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
    dlam = jnp.float32(dlam_angstroms)

    def run(n_sources, spectra, xsca, ysca, images, output):
        return run_galaxy_batch_window(
            arrays, sens, wavelengths_jax, dlam, n_sources, spectra, xsca,
            ysca, images, output, static=static, plan=plan,
            sed_rebin=int(sed_rebin))
    return run


def canonical_order(xsca, ysca, spectra, extra=()):
    """Permutation that sorts sources by (x, y, spectrum checksum, extra):
    a sum order that does not depend on the catalogue order."""
    spectra = np.ascontiguousarray(np.asarray(spectra, np.float32))
    crc = np.array([zlib.crc32(r.tobytes()) for r in spectra], np.int64)
    keys = [np.asarray(e) for e in reversed(tuple(extra))]
    keys += [crc, np.asarray(ysca, np.float32), np.asarray(xsca, np.float32)]
    return np.lexsort(keys)


# ---------------------------------------------------------------------------
# "tiled" mode
# ---------------------------------------------------------------------------

class TileLayout(NamedTuple):
    """Detector tiling: ``n_tiles x n_tiles`` tiles of ``tile`` px; each
    tile buffer holds the tile plus a window-sized halo above/right and
    ``margin`` px on every side."""
    tile: int
    n_tiles: int
    margin: int
    buffer_shape: tuple

    @property
    def count(self):
        return self.n_tiles * self.n_tiles


def make_tile_layout(window_shape, n_tiles=8, naxis=4088, margin=4):
    tile = -(-int(naxis) // int(n_tiles))
    buf = (tile + int(window_shape[0]) + 2 * margin,
           tile + int(window_shape[1]) + 2 * margin)
    return TileLayout(tile=tile, n_tiles=int(n_tiles), margin=int(margin),
                      buffer_shape=buf)


def tile_origins(layout):
    """[n_tiles^2] tile lower corners (ty, tx), row-major."""
    t = np.arange(layout.n_tiles) * layout.tile
    ty, tx = np.meshgrid(t, t, indexing="ij")
    return ty.ravel().astype(np.int32), tx.ravel().astype(np.int32)


def assign_tiles(layout, oy, ox):
    """Tile index of each source from its window origin (clipped to the
    detector's tiles; the parts of off-detector windows that fall outside a
    buffer are off the detector too)."""
    iy = np.clip(np.asarray(oy) // layout.tile, 0, layout.n_tiles - 1)
    ix = np.clip(np.asarray(ox) // layout.tile, 0, layout.n_tiles - 1)
    return (iy * layout.n_tiles + ix).astype(np.int64)


def tile_table(tiles, order, n_tile_count, rounds_multiple=1):
    """[n_tiles, R] source indices per tile in ``order`` (-1 = empty);
    R is a multiple of ``rounds_multiple``."""
    tiles = np.asarray(tiles)
    lists = [[] for _ in range(n_tile_count)]
    for i in np.asarray(order):
        lists[tiles[i]].append(int(i))
    r = max([len(l) for l in lists] + [1])
    r = -(-r // rounds_multiple) * rounds_multiple
    table = np.full((n_tile_count, r), -1, np.int64)
    for t, l in enumerate(lists):
        table[t, :len(l)] = l
    return table


@partial(jax.jit, static_argnames=("static", "plan", "sed_rebin"))
def source_origins(arrays, wavelengths, xsca, ysca, *, static, plan,
                   sed_rebin=1):
    """Window origins (oy, ox) [N] of many sources (geometry only)."""
    _, optical_payload = jit_payload.merge_payloads(arrays, static)
    wl_dep = pipeline.rebin_counts(wavelengths, wavelengths, sed_rebin)[1]

    def one(x, y):
        xd, yd = star_disperser._compute_dispersed_positions(
            optical_payload, x, y, wl_dep)
        return window_origin(plan, xd, yd)
    return jax.vmap(one)(xsca, ysca)


def _tile_rounds(geometry_fn, render_fn, layout, tile_args, buffers, toy, tox,
                 tile_batch):
    """Per tile: scan over its rounds (render, add to the tile buffer at
    origin - tile corner + margin). Tiles are independent (lax.map with
    ``tile_batch`` tiles vectorised)."""
    m = layout.margin
    # Geometry of every (tile, round) first, unvectorised (see _window_loop).
    n_t, n_r = tile_args[0].shape[:2]
    src_args = tuple(tile_args[1:])
    flat = [a.reshape((n_t * n_r,) + a.shape[2:]) for a in src_args]
    geom = jax.lax.map(lambda a: geometry_fn(*a), flat)
    geom = jax.tree.map(lambda g: g.reshape((n_t, n_r) + g.shape[1:]), geom)

    def tile_fn(args):
        buf, ty, tx, valid, g_t, *src = args

        def step(buf, r):
            row = [a[r] for a in src]
            geom = jax.tree.map(lambda g: g[r], g_t)

            def body(buf):
                win, oy, ox = render_fn(geom, *row)
                return add_window(buf, win, oy - ty + m, ox - tx + m)
            buf = jax.lax.cond(valid[r], body, lambda b: b, buf)
            return buf, None
        buf, _ = jax.lax.scan(jax.checkpoint(step), buf,
                              jnp.arange(valid.shape[0]))
        return buf

    return jax.lax.map(tile_fn, (buffers, toy, tox, tile_args[0], geom)
                       + src_args, batch_size=int(tile_batch))


@partial(jax.jit, static_argnames=("static", "plan", "layout", "sed_rebin",
                                   "tile_batch"),
         donate_argnames=("buffers",))
def run_star_tiles(arrays, sens, wavelengths, dlam_angstroms, valid, spectra,
                   xsca, ysca, buffers, toy, tox, *, static, plan, layout,
                   sed_rebin=1, tile_batch=8):
    """Tiled star rounds: ``valid`` [T, R], ``spectra`` [T, R, N_wl],
    ``xsca``/``ysca`` [T, R], ``buffers`` [T, *layout.buffer_shape],
    ``toy``/``tox`` [T] tile corners. Returns the updated buffers."""
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    _, wl_dep = pipeline.rebin_counts(sens, wavelengths, sed_rebin)

    def geometry_fn(spec, x, y):
        return star_disperser._compute_dispersed_positions(
            optical_payload, x, y, wl_dep)

    def render_fn(dispersed, spec, x, y):
        counts = spec * sens * dlam_angstroms
        counts, _ = pipeline.rebin_counts(counts, wavelengths, sed_rebin)
        return render_star_window(psf_payload, optical_payload, x, y, wl_dep,
                                  counts, plan, dispersed=dispersed)
    return _tile_rounds(geometry_fn, render_fn, layout,
                        (valid, spectra, xsca, ysca), buffers, toy, tox,
                        tile_batch)


@partial(jax.jit, static_argnames=("static", "plan", "layout", "sed_rebin",
                                   "tile_batch"),
         donate_argnames=("buffers",))
def run_galaxy_tiles(arrays, sens, wavelengths, dlam_angstroms, valid,
                     spectra, xsca, ysca, images, buffers, toy, tox, *,
                     static, plan, layout, sed_rebin=1, tile_batch=8):
    """Tiled galaxy rounds; as :func:`run_star_tiles` plus ``images``
    [T, R, npix, npix]."""
    psf_payload, optical_payload = jit_payload.merge_payloads(arrays, static)
    _, wl_dep = pipeline.rebin_counts(sens, wavelengths, sed_rebin)

    def geometry_fn(spec, x, y, img):
        return (_galaxy_geometry(plan, psf_payload, optical_payload, x, y),
                star_disperser._compute_dispersed_positions(
                    optical_payload, x, y, wl_dep))

    def render_fn(geom, spec, x, y, img):
        counts = spec * sens * dlam_angstroms
        counts, _ = pipeline.rebin_counts(counts, wavelengths, sed_rebin)
        return render_galaxy_window(psf_payload, optical_payload, img, x, y,
                                    counts, wl_dep, plan, geometry=geom[0],
                                    dispersed=geom[1])
    return _tile_rounds(geometry_fn, render_fn, layout,
                        (valid, spectra, xsca, ysca, images), buffers, toy,
                        tox, tile_batch)


@partial(jax.jit, static_argnames=("layout",), donate_argnames=("output",))
def stitch_tiles(buffers, toy, tox, output, *, layout):
    """Add the tile buffers to the frame in tile order (fixed)."""
    m = layout.margin
    return add_windows_ordered(output, buffers, toy - m, tox - m)


class TiledDisperser:
    """Host driver of the tiled mode for one (SCA, order, kind).

    ``stars(spectra, xsca, ysca, output)`` /
    ``galaxies(spectra, xsca, ysca, sersic_params, npix, output)`` add all
    sources of the order to ``output`` (same role as
    ``pipeline.disperse_batched_stars`` / ``disperse_batched_sersic_galaxies``).

    Parameters
    ----------
    rounds : int
        Rounds (sources per tile) per device call; bounds the device memory
        of the per-call inputs ([n_tiles, rounds, N_wl] spectra).
    tile_batch : int
        Tiles rendered in parallel (``lax.map`` batch size).
    mesh : jax.sharding.Mesh, optional
        1-D mesh (axis ``"t"``) to shard the tiles over devices
        (``shard_map``); tile count must be divisible by its size.
    """

    def __init__(self, psf_payload, optical_payload, sens, wavelengths_jax,
                 dlam_angstroms, plan, sed_rebin=1, hoist_psf=True,
                 n_tiles=8, rounds=4, tile_batch=8, mesh=None, naxis=4088):
        self.plan = plan
        self.layout = make_tile_layout(plan.window_shape, n_tiles, naxis)
        self.arrays, self.static = _arrays(psf_payload, optical_payload,
                                           plan, hoist_psf)
        self.sens = sens
        self.wl = wavelengths_jax
        self.dlam = jnp.float32(dlam_angstroms)
        self.sed_rebin = int(sed_rebin)
        self.rounds = int(rounds)
        self.tile_batch = max(2, int(tile_batch))   # see _window_loop
        self.mesh = mesh
        ty, tx = tile_origins(self.layout)
        self.toy, self.tox = jnp.asarray(ty), jnp.asarray(tx)
        self.last_table = None

    def _origins(self, xsca, ysca):
        oy, ox = source_origins(self.arrays, self.wl, jnp.asarray(xsca),
                                jnp.asarray(ysca), static=self.static,
                                plan=self.plan, sed_rebin=self.sed_rebin)
        return np.asarray(oy), np.asarray(ox)

    def _call(self, fn, args, toy, tox):
        kw = dict(static=self.static, plan=self.plan, layout=self.layout,
                  sed_rebin=self.sed_rebin, tile_batch=self.tile_batch)
        if self.mesh is None:
            return fn(self.arrays, self.sens, self.wl, self.dlam, *args,
                      toy, tox, **kw)
        from jax.sharding import PartitionSpec as P
        tiled = P("t")
        n_tiled = len(args) + 2

        def local(arrays, sens, wl, dlam, *rest):
            return fn(arrays, sens, wl, dlam, *rest, **kw)
        sm = jax.shard_map(local, mesh=self.mesh,
                           in_specs=(P(), P(), P(), P()) + (tiled,) * n_tiled,
                           out_specs=tiled, check_vma=False)
        return sm(self.arrays, self.sens, self.wl, self.dlam, *args, toy, tox)

    def _run(self, xsca, ysca, spectra, extra, make_round_inputs, fn, output):
        xsca = np.asarray(xsca, np.float32)
        ysca = np.asarray(ysca, np.float32)
        spectra = np.asarray(spectra, np.float32)
        oy, ox = self._origins(xsca, ysca)
        tiles = assign_tiles(self.layout, oy, ox)
        order = canonical_order(xsca, ysca, spectra, extra)
        table = tile_table(tiles, order, self.layout.count, self.rounds)
        self.last_table = table
        n_t, r_tot = table.shape
        # Row n_t is a scratch buffer for the padding of the active-tile
        # list (never valid, never stitched).
        buffers = jnp.zeros((n_t + 1,) + self.layout.buffer_shape,
                            jnp.float32)
        quantum = self.tile_batch * (1 if self.mesh is None
                                     else self.mesh.devices.size)
        toy_np = np.append(np.asarray(self.toy), 0).astype(np.int32)
        tox_np = np.append(np.asarray(self.tox), 0).astype(np.int32)
        table = np.vstack([table, np.full((1, r_tot), -1, table.dtype)])
        for r0 in range(0, r_tot, self.rounds):
            sub_all = table[:, r0:r0 + self.rounds]
            # Only tiles with sources in these rounds; the list is padded to
            # a multiple of the tile batch (few distinct shapes = compiles).
            active = np.flatnonzero((sub_all >= 0).any(axis=1))
            if active.size == 0:
                continue
            n_act = -(-active.size // quantum) * quantum
            active = np.concatenate(
                [active, np.full(n_act - active.size, n_t, active.dtype)])
            sub = sub_all[active]
            valid = sub >= 0
            idx = np.where(valid, sub, 0)
            args = (jnp.asarray(valid), jnp.asarray(spectra[idx]),
                    jnp.asarray(xsca[idx]), jnp.asarray(ysca[idx]))
            act = jnp.asarray(active)
            args = args + make_round_inputs(idx) + (buffers[act],)
            upd = self._call(fn, args, jnp.asarray(toy_np[active]),
                             jnp.asarray(tox_np[active]))
            buffers = buffers.at[act].set(upd)
        output = stitch_tiles(buffers[:n_t], self.toy, self.tox, output,
                              layout=self.layout)
        output.block_until_ready()
        return output

    def stars(self, spectra, xsca, ysca, output):
        return self._run(xsca, ysca, spectra, (), lambda idx: (),
                         run_star_tiles, output)

    def galaxies(self, spectra, xsca, ysca, sersic_params, npix, output):
        params = [np.asarray(p, np.float32) for p in sersic_params]

        def images(idx):
            flat = [jnp.asarray(p[idx.ravel()]) for p in params]
            img = pipeline._make_sersic_images_jit(*flat, npix=npix)
            return (img.reshape(idx.shape + img.shape[-2:]),)
        return self._run(xsca, ysca, spectra, params, images,
                         run_galaxy_tiles, output)

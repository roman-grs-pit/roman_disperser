"""Matrix-free linear operator ``image = A @ spectra`` for one SCA (experimental).

Prototype for LINEAR-style spectral extraction (Ryan, Casertano & Pirzkal
2018, arXiv:1801.03936) and forward-model reconstruction with the
roman_disperser dispersers. For fixed source positions and morphologies the
dispersed image is linear in the source spectra, so the simulator *is* a
linear operator ``A``. This module exposes

* ``forward(theta) -> image`` (``A``),
* ``transpose(image) -> theta`` (``A^T``, from :func:`jax.linear_transpose`;
  :meth:`SceneOperator.transpose_vjp` gives the same via :func:`jax.vjp`),
* ``render(theta, dx, dy)``, the same model with per-source position offsets
  as differentiable inputs (needs an interpolating deposit; the floor
  deposit has zero position gradients and is refused there),
* ``model_fn()``, the fitting entry point ``f(theta, dxy, sersic)``:
  spectra, position offsets and Sérsic shape parameters all differentiable
  (galaxy images rendered in-graph by ``sersic.make_sersic_images``).

For position fits with curvature-based methods (HMC/NUTS, Newton, Laplace)
use ``deposit="cubic"`` and ``psf_spatial="cubic"``: both are C1 in source
position, where the production ``interp`` deposit and bilinear PSF blend
are piecewise linear (gradient jumps at every 1/os px and PSF node, zero
curvature in between). Sky positions enter through :func:`sky_dxy`
(``optical_model_jax.perturbed_sky_to_sca``).

``theta`` is ``[n_sources, n_par]``: either the spectra on the catalogue
wavelength grid (``basis=None``) or the node values of a piecewise-linear
spectral basis (:class:`LinearBasis`), in units of ``flux_unit``
(default 1e-17 erg/s/cm^2/A). The image is in e-/s, as the production MODEL.

Everything is built from the existing single-source pieces
(``psf_model.interpolate_psf_spatial``, ``galaxy_disperser.prepare_galaxy_images``,
``star_disperser.bin_stack_phases`` / ``_compute_dispersed_positions`` /
``deposit_binned``, ``pipeline.rebin_counts``); no existing function is
changed. Sources are looped with ``lax.scan`` over a static number of rows
(pad with zero-flux rows to reuse a compile), so reverse mode works; the
production ``pipeline.run_*_batch`` loops are reverse-differentiable too
(checkpointed ``lax.scan`` over the padded batch).

The model can be evaluated in a sub-window ``(y0, x0, ny, nx)`` of the SCA
(0-indexed array rows/columns); positions are shifted by the integer window
offset before the deposit, which leaves the deposit phases unchanged, so the
window is an exact crop of the full-frame image (light falling outside the
window is dropped, as at the detector edge).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import (galaxy_disperser, jit_payload, optical_model_jax, psf_model,
               sersic, star_disperser)
from .pipeline import rebin_counts

FULL_SCA = (0, 0, 4088, 4088)


# ---------------------------------------------------------------------------
# Spectral basis
# ---------------------------------------------------------------------------

class LinearBasis:
    """Piecewise-linear (hat-function) basis on uniform nodes.

    ``spectrum(lambda) = sum_k c_k * hat_k(lambda)``, i.e. linear
    interpolation of node values ``c`` (constant beyond the end nodes).

    Parameters
    ----------
    wavelengths : array [N_wl]
        Catalogue grid in microns (where the spectra live).
    spacing_angstroms : float
        Node spacing in A.
    lo, hi : float, optional
        Node range in microns (default: the grid range).
    """

    def __init__(self, wavelengths, spacing_angstroms, lo=None, hi=None):
        wl = np.asarray(wavelengths, np.float64)
        lo = wl[0] if lo is None else lo
        hi = wl[-1] if hi is None else hi
        step = spacing_angstroms / 1e4
        n = int(np.ceil((hi - lo) / step - 1e-9)) + 1
        self.nodes = jnp.asarray(lo + step * np.arange(n), jnp.float32)
        self.wavelengths = jnp.asarray(wl, jnp.float32)
        self.spacing_angstroms = float(spacing_angstroms)
        self.n = n

    def __call__(self, c):
        """[..., n_nodes] -> [..., N_wl]."""
        f = lambda v: jnp.interp(self.wavelengths, self.nodes, v)
        for _ in range(c.ndim - 1):
            f = jax.vmap(f)
        return f(c)

    def matrix(self):
        """Dense [N_wl, n_nodes] basis matrix (for tests / small problems)."""
        return np.asarray(jax.vmap(self.__call__, in_axes=1, out_axes=1)(
            jnp.eye(self.n, dtype=jnp.float32)))


# ---------------------------------------------------------------------------
# Per-order setup
# ---------------------------------------------------------------------------

class OrderSetup(NamedTuple):
    """Everything one spectral order needs (arrays only, plus a static key)."""
    arrays: tuple          # jit_payload arrays, PSF dict incl. psf_grid_binned
    static: tuple          # jit_payload static
    sens: jax.Array        # [N_wl] on the catalogue grid
    wl_deposit: jax.Array  # [N_dep] deposit wavelengths (after rebinning)
    rebin: int


def make_order_setup(psf_payload, optical_payload, sens, wavelengths, rebin=1):
    """Bundle one order's payloads, sensitivity and deposit grid.

    The star PSF grid is phase-binned once here (as with
    ``make_star_disperser(hoist_psf=True)``), so stars only blend four
    binned nodes per evaluation.
    """
    arrays, static = jit_payload.split_payloads(psf_payload, optical_payload)
    psf_arrays, opt_arrays = arrays
    psf_arrays = dict(psf_arrays, psf_grid_binned=star_disperser.bin_psf_grid(
        psf_arrays['psf_grid'], psf_payload['oversample']))
    wl = jnp.asarray(wavelengths, jnp.float32)
    _, wl_dep = rebin_counts(jnp.asarray(sens, jnp.float32), wl, int(rebin))
    return OrderSetup((psf_arrays, opt_arrays), static,
                      jnp.asarray(sens, jnp.float32), wl_dep, int(rebin))


# ---------------------------------------------------------------------------
# Per-source pieces (pure functions)
# ---------------------------------------------------------------------------

def _payloads(order):
    return jit_payload.merge_payloads(order.arrays, order.static)


def star_stack(order, x, y, psf_spatial="linear"):
    """Phase-binned PSF stack [os, os, N_grid, n, n] of a star at (x, y)."""
    psf, _ = _payloads(order)
    return psf_model.interpolate_psf_spatial(psf, x, y,
                                             grid_key='psf_grid_binned',
                                             method=psf_spatial)


def galaxy_stack(order, x, y, image, psf_spatial="linear"):
    """Phase-binned warped+convolved galaxy stack [os, os, N_grid, n, n]."""
    psf, opt = _payloads(order)
    os_ = psf['oversample']
    conv, _, _, _ = galaxy_disperser.prepare_galaxy_images(
        opt, psf, image, x, y, 1.0 / os_, 1.0 / os_, psf_spatial=psf_spatial)
    return star_disperser.bin_stack_phases(conv, os_)


def sersic_galaxy_stack(order, x, y, params, npix, psf_spatial="linear",
                        subsample=1):
    """:func:`galaxy_stack` of a Sérsic galaxy rendered directly in the
    warped frame (``galaxy_disperser.prepare_sersic_galaxy_images``);
    ``params`` = [4] (r_eff_pix, n, ba, theta)."""
    psf, opt = _payloads(order)
    os_ = psf['oversample']
    conv, _, _, _ = galaxy_disperser.prepare_sersic_galaxy_images(
        opt, psf, tuple(params[i] for i in range(4)), npix, x, y,
        1.0 / os_, 1.0 / os_, psf_spatial=psf_spatial, subsample=subsample)
    return star_disperser.bin_stack_phases(conv, os_)


def dispersed_positions(order, x, y):
    _, opt = _payloads(order)
    return star_disperser._compute_dispersed_positions(
        opt, x, y, order.wl_deposit)


def _deposit_one(order, binned, stamp_shape, xd, yd, spectrum, out, window,
                 dlam, chunk_size, deposit):
    psf, _ = _payloads(order)
    counts = spectrum * order.sens * dlam
    # Bin wavelengths are order.wl_deposit (precomputed); only the summed
    # counts are needed here, so a placeholder grid is passed.
    counts, _ = rebin_counts(counts, jnp.zeros(counts.shape[-1], counts.dtype),
                             order.rebin)
    return star_disperser.deposit_binned(
        binned, stamp_shape, psf['wavelengths'], order.wl_deposit, counts,
        xd - window[1], yd - window[0], out,
        oversample=psf['oversample'], chunk_size=chunk_size, deposit=deposit)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class SceneOperator:
    """``image = A @ theta`` for a fixed scene on one SCA.

    Parameters
    ----------
    orders : dict {order: OrderSetup}
    wavelengths : array [N_wl]
        Catalogue grid (microns); the spectra / basis live on it.
    dlam_angstroms : float
        Catalogue spacing (counts = f_lambda * sens * dlam).
    star_xy : array [n_star, 2] or None
        (x, y) SCA positions (1-indexed FITS) of point sources.
    galaxy_xy : array [n_gal, 2] or None
    galaxy_images : array [n_gal, npix, npix] or None
        Normalised oversampled images (``sersic.make_sersic_image``).
    window : (y0, x0, ny, nx), optional
        Model sub-window (default: full SCA).
    basis : LinearBasis, optional
        If given, ``theta`` holds basis coefficients.
    flux_unit : float
        Units of ``theta`` in erg/s/cm^2/A.
    precompute : bool
        Cache the per-source binned stacks and dispersed positions (fast
        repeated application, memory ~8 MB per star and ~21 MB per galaxy
        per order at production PSF size). Otherwise they are recomputed in
        every application (cheap for stars; one FFT stack per galaxy).
    checkpoint : bool
        ``jax.checkpoint`` the per-source scan body (default True). Without
        it the transpose (both :func:`jax.linear_transpose` and vjp) keeps
        per-source scan residuals, ~72 MB per star per order at production
        size (7.2 GB XLA temp for 100 stars); with it 0.14 GB and ~25 %
        faster on CPU (batch-5 benchmark).
    deposit : {"interp", "cubic", "floor"}
        ``"cubic"`` for curvature-based position fits (C1 in position);
        ``"floor"`` only for the linear operator (no position gradients:
        :meth:`render` and :meth:`model_fn` refuse it).
    chunk_size : int
    psf_spatial : {"linear", "cubic"}
        Spatial PSF interpolation of the sources (``"cubic"``: C1 in
        position, see ``psf_model.interpolate_psf_spatial``).
    galaxy_npix : int, optional
        Galaxy image size; needed with ``galaxy_images=None``, when the
        images are only rendered from Sérsic parameters by :meth:`model_fn`
        (then ``precompute`` must be False).
    sersic_subsample : int
        Sub-pixel sampling of the in-graph Sérsic images (``sersic.
        make_sersic_images``); 1 matches production.
    sersic_warp : {"analytic", "splat"}
        How :meth:`model_fn` applies the dispersion warp to Sérsic
        parameters (see there).

    Sources are ordered stars first, then galaxies, in ``theta``.
    """

    def __init__(self, orders, wavelengths, dlam_angstroms, star_xy=None,
                 galaxy_xy=None, galaxy_images=None, window=None, basis=None,
                 flux_unit=1e-17, precompute=True, checkpoint=True,
                 deposit="interp", chunk_size=2000, psf_spatial="linear",
                 galaxy_npix=None, sersic_subsample=1,
                 sersic_warp="analytic"):
        self.orders = dict(orders)
        self.order_keys = tuple(sorted(self.orders))
        self.wavelengths = jnp.asarray(wavelengths, jnp.float32)
        self.dlam = float(dlam_angstroms)
        self.star_xy = jnp.zeros((0, 2), jnp.float32) if star_xy is None \
            else jnp.asarray(star_xy, jnp.float32).reshape(-1, 2)
        self.galaxy_xy = jnp.zeros((0, 2), jnp.float32) if galaxy_xy is None \
            else jnp.asarray(galaxy_xy, jnp.float32).reshape(-1, 2)
        self.n_star = int(self.star_xy.shape[0])
        self.n_gal = int(self.galaxy_xy.shape[0])
        self.n_sources = self.n_star + self.n_gal
        if self.n_gal and galaxy_images is not None:
            self.galaxy_images = jnp.asarray(galaxy_images, jnp.float32)
            assert self.galaxy_images.shape[0] == self.n_gal
            self.galaxy_npix = int(self.galaxy_images.shape[-1])
        else:
            self.galaxy_images = None
            self.galaxy_npix = None if galaxy_npix is None else int(galaxy_npix)
            if self.n_gal and self.galaxy_npix is None:
                raise ValueError("galaxies need galaxy_images or galaxy_npix")
            if self.n_gal and precompute:
                raise ValueError("precompute needs galaxy_images")
        if psf_spatial not in ("linear", "cubic"):
            raise ValueError(f"psf_spatial must be 'linear' or 'cubic', "
                             f"got {psf_spatial!r}")
        self.psf_spatial = psf_spatial
        self.sersic_subsample = int(sersic_subsample)
        if sersic_warp not in ("analytic", "splat"):
            raise ValueError(f"sersic_warp must be 'analytic' or 'splat', "
                             f"got {sersic_warp!r}")
        self.sersic_warp = sersic_warp
        self.window = tuple(int(v) for v in (window or FULL_SCA))
        self.shape = self.window[2:]
        self.basis = basis
        self.n_par = basis.n if basis is not None else int(self.wavelengths.size)
        self.theta_shape = (self.n_sources, self.n_par)
        self.flux_unit = float(flux_unit)
        self.precompute = bool(precompute)
        self.checkpoint = bool(checkpoint)
        if deposit not in star_disperser.DEPOSIT_MODES:
            raise ValueError(f"unknown deposit {deposit!r}")
        self.deposit = deposit
        self.chunk_size = int(chunk_size)

        self._consts = self._build_consts()
        self._fwd = jax.jit(self._forward_impl)
        self._T = jax.jit(self._transpose_impl)
        self._Tvjp = jax.jit(self._transpose_vjp_impl)
        self._render = jax.jit(self._render_impl)

    # -- constants ----------------------------------------------------------

    def _stamp_shapes(self, key):
        psf_grid = self.orders[key].arrays[0]['psf_grid']
        s = psf_grid.shape[-2:]
        star = (int(s[0]), int(s[1]))
        if self.n_gal:
            npix = (self.galaxy_npix, self.galaxy_npix)
            gal = (int(npix[0]) + star[0] - 1, int(npix[1]) + star[1] - 1)
        else:
            gal = None
        return star, gal

    def _build_consts(self):
        consts = {"orders": {k: (self.orders[k].arrays, self.orders[k].sens,
                                 self.orders[k].wl_deposit)
                             for k in self.order_keys},
                  "star_xy": self.star_xy, "galaxy_xy": self.galaxy_xy,
                  "galaxy_images": self.galaxy_images}
        if self.precompute:
            pre = {}
            for k in self.order_keys:
                pre[k] = self._precompute_order(k, consts["orders"][k])
            consts["pre"] = pre
        return consts

    def _order(self, k, c):
        """OrderSetup from the traced constants ``c`` of order ``k``."""
        o = self.orders[k]
        return OrderSetup(c[0], o.static, c[1], c[2], o.rebin)

    def _precompute_order(self, k, c):
        out = {}
        if self.n_star:
            f = jax.jit(lambda c, xy: jax.lax.map(lambda p, o=self._order(k, c):
                (star_stack(o, p[0], p[1], self.psf_spatial),) +
                dispersed_positions(o, p[0], p[1]), xy))
            out["star"] = f(c, self.star_xy)
        if self.n_gal:
            f = jax.jit(lambda c, xy, im: jax.lax.map(
                lambda a, o=self._order(k, c):
                (galaxy_stack(o, a[0][0], a[0][1], a[1], self.psf_spatial),) +
                dispersed_positions(o, a[0][0], a[0][1]), (xy, im)))
            out["gal"] = f(c, self.galaxy_xy, self.galaxy_images)
        return out

    # -- model --------------------------------------------------------------

    def spectra(self, theta):
        """theta [n_src, n_par] -> f_lambda spectra [n_src, N_wl] (erg/s/cm^2/A)."""
        s = self.basis(theta) if self.basis is not None else theta
        return s * self.flux_unit

    def _scan_group(self, order, key, kind, stamp_shape, xs, out, pre):
        window = self.window
        dlam = self.dlam

        def body(out, row):
            if pre:
                binned, xd, yd, spec = row
            elif kind == "star":
                x, y, spec = row
                binned = star_stack(order, x, y, self.psf_spatial)
                xd, yd = dispersed_positions(order, x, y)
            elif kind == "gal_sersic":
                x, y, params, spec = row
                binned = sersic_galaxy_stack(
                    order, x, y, params, self.galaxy_npix, self.psf_spatial,
                    self.sersic_subsample)
                xd, yd = dispersed_positions(order, x, y)
            else:
                x, y, image, spec = row
                binned = galaxy_stack(order, x, y, image, self.psf_spatial)
                xd, yd = dispersed_positions(order, x, y)
            out = _deposit_one(order, binned, stamp_shape, xd, yd, spec, out,
                               window, dlam, self.chunk_size, self.deposit)
            return out, None

        if self.checkpoint:
            body = jax.checkpoint(body)
        out, _ = jax.lax.scan(body, out, xs)
        return out

    def _model(self, consts, theta, dxy=None, use_pre=None,
               galaxy_images=None, sersic_params=None):
        """Image for coefficients theta and optional position offsets dxy
        [n_src, 2], galaxy images, or Sérsic parameters [n_gal, 4] rendered
        in the warped frame (all force on-the-fly stacks)."""
        use_pre = self.precompute if use_pre is None else use_pre
        if galaxy_images is None:
            galaxy_images = consts["galaxy_images"]
        spec = self.spectra(theta)
        out = jnp.zeros(self.shape, jnp.float32)
        s_spec, g_spec = spec[:self.n_star], spec[self.n_star:]
        sxy, gxy = consts["star_xy"], consts["galaxy_xy"]
        if dxy is not None:
            sxy = sxy + dxy[:self.n_star]
            gxy = gxy + dxy[self.n_star:]
        for k in self.order_keys:
            order = self._order(k, consts["orders"][k])
            st_shape, gal_shape = self._stamp_shapes(k)
            if self.n_star:
                if use_pre:
                    b, xd, yd = consts["pre"][k]["star"]
                    xs = (b, xd, yd, s_spec)
                else:
                    xs = (sxy[:, 0], sxy[:, 1], s_spec)
                out = self._scan_group(order, k, "star", st_shape, xs, out,
                                       use_pre)
            if self.n_gal:
                kind = "gal"
                if use_pre:
                    b, xd, yd = consts["pre"][k]["gal"]
                    xs = (b, xd, yd, g_spec)
                elif sersic_params is not None:
                    xs = (gxy[:, 0], gxy[:, 1], sersic_params, g_spec)
                    kind = "gal_sersic"
                else:
                    xs = (gxy[:, 0], gxy[:, 1], galaxy_images, g_spec)
                out = self._scan_group(order, k, kind, gal_shape, xs, out,
                                       use_pre)
        return out

    def _forward_impl(self, consts, theta):
        return self._model(consts, theta)

    def _transpose_impl(self, consts, image):
        f = lambda th: self._model(consts, th)
        (ct,) = jax.linear_transpose(
            f, jax.ShapeDtypeStruct(self.theta_shape, jnp.float32))(image)
        return ct

    def _transpose_vjp_impl(self, consts, image):
        f = lambda th: self._model(consts, th)
        _, vjp = jax.vjp(f, jnp.zeros(self.theta_shape, jnp.float32))
        return vjp(image)[0]

    def _render_impl(self, consts, theta, dxy):
        return self._model(consts, theta, dxy=dxy, use_pre=False)

    # -- public -------------------------------------------------------------

    def forward(self, theta):
        """A @ theta: [n_src, n_par] -> [ny, nx] e-/s."""
        return self._fwd(self._consts, jnp.asarray(theta, jnp.float32))

    def transpose(self, image):
        """A^T @ image via jax.linear_transpose: [ny, nx] -> [n_src, n_par]."""
        return self._T(self._consts, jnp.asarray(image, jnp.float32))

    def transpose_vjp(self, image):
        """A^T @ image via jax.vjp (same result; reverse-mode residuals)."""
        return self._Tvjp(self._consts, jnp.asarray(image, jnp.float32))

    def _require_position_gradients(self):
        if self.deposit == "floor":
            raise ValueError("deposit='floor' has zero position gradients; "
                             "use 'interp' or 'cubic' for render/model_fn")

    def render(self, theta, dxy):
        """Model with per-source position offsets dxy [n_src, 2] (px);
        differentiable in theta and dxy (pure function of its inputs)."""
        self._require_position_gradients()
        return self._render(self._consts, jnp.asarray(theta, jnp.float32),
                            jnp.asarray(dxy, jnp.float32))

    def render_fn(self):
        """Un-jitted ``f(theta, dxy) -> image`` with constants bound, for
        composing with jax.grad / jit by the caller."""
        self._require_position_gradients()
        return lambda theta, dxy: self._model(self._consts, theta, dxy=dxy,
                                              use_pre=False)


    def model_fn(self):
        """Fitting entry point: un-jitted ``f(theta, dxy=None, sersic_params=None)
        -> image`` with the constants bound.

        ``theta`` [n_src, n_par] spectra / basis coefficients, ``dxy``
        [n_src, 2] position offsets in px (None: zero), ``sersic_params`` a tuple
        ``(r_eff_pix, n, ba, theta)`` of [n_gal] arrays (None: the fixed
        ``galaxy_images``). Differentiable in all three; compose with
        ``transforms.sersic_from_unconstrained`` for unconstrained shape
        parameters and with :func:`sky_dxy` for sky offsets.

        With ``sersic_warp="analytic"`` (default) the Sérsic profile is
        evaluated directly in the dispersion-warped frame at every PSF grid
        wavelength (no resampling; smooth in the shape parameters and the
        position); ``"splat"`` renders the image once and warps it with
        the production bilinear forward scatter (piecewise linear in the
        warp).
        """
        self._require_position_gradients()
        consts = self._consts
        npix = self.galaxy_npix
        sub = self.sersic_subsample
        analytic = self.sersic_warp == "analytic"

        def f(theta, dxy=None, sersic_params=None):
            if dxy is None:
                dxy = jnp.zeros((self.n_sources, 2), jnp.float32)
            images = params = None
            if sersic_params is not None:
                if analytic:
                    params = jnp.stack([jnp.asarray(p, jnp.float32)
                                        for p in sersic_params], axis=-1)
                else:
                    images = sersic.make_sersic_images(
                        *sersic_params, npix=npix, subsample=sub)
            elif self.n_gal and consts["galaxy_images"] is None:
                raise ValueError("no galaxy_images: pass sersic parameters")
            return self._model(consts, theta, dxy=dxy, use_pre=False,
                               galaxy_images=images, sersic_params=params)
        return f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sky_dxy(ref, payload, d_xi=0.0, d_eta=0.0, d_ra=0.0, d_dec=0.0,
            d_pa=0.0):
    """SCA position offsets [N, 2] (px) of sources from sky perturbations.

    ``ref`` is ``optical_model_jax.make_sky_reference`` of the operator's
    sources (same order; ``star_xy`` / ``galaxy_xy`` are its nominal SCA
    positions), ``payload`` the SCA's optical payload. ``d_xi, d_eta``
    move sources in the tangent plane, ``d_ra, d_dec, d_pa`` the pointing
    (degrees). The result is the float32 *change*, so it keeps full
    precision; pass it as ``dxy`` to :meth:`SceneOperator.model_fn`.
    """
    dx, dy = optical_model_jax._fpa_change(ref, d_xi, d_eta, d_ra, d_dec, d_pa)
    k = jnp.float32(3600.0) / payload["det"]["pixel_scale"]
    if "xsca0" not in ref:
        x0, y0 = optical_model_jax.fpa_to_sca(payload, ref["xfpa0"], ref["yfpa0"])
        x1, y1 = optical_model_jax.fpa_to_sca(payload, ref["xfpa0"] + dx,
                                              ref["yfpa0"] + dy)
        return jnp.stack([x1 - x0, y1 - y0], axis=-1)
    return jnp.stack([-k * dx, k * dy], axis=-1)

def auto_window(orders, xy, margin=40):
    """Bounding window (y0, x0, ny, nx) of all traces of sources ``xy``
    [n, 2] in all ``orders``, padded by ``margin`` px, clipped to the SCA.
    ``margin`` must cover half the native stamp (24 px star, 39 px galaxy)."""
    ys, xs = [], []
    for k, o in orders.items():
        for x, y in np.asarray(xy, np.float32):
            xd, yd = dispersed_positions(o, jnp.float32(x), jnp.float32(y))
            xs.append(np.asarray(xd)); ys.append(np.asarray(yd))
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    x0 = max(int(np.floor(np.nanmin(xs))) - margin, 0)
    x1 = min(int(np.ceil(np.nanmax(xs))) + margin, 4088)
    y0 = max(int(np.floor(np.nanmin(ys))) - margin, 0)
    y1 = min(int(np.ceil(np.nanmax(ys))) + margin, 4088)
    return (y0, x0, y1 - y0, x1 - x0)


def _dot(a, b):
    return jnp.vdot(a, b, precision=jax.lax.Precision.HIGHEST)


def conjugate_gradient(matvec, b, x0=None, maxiter=100, tol=1e-6,
                       callback=None):
    """Plain CG on a symmetric positive (semi)definite operator (host loop,
    so each iteration can be timed / logged). Returns (x, history)."""
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    p = r
    rs = _dot(r, r)
    bnorm = float(jnp.sqrt(_dot(b, b)))
    hist = [float(jnp.sqrt(rs)) / bnorm]
    for it in range(maxiter):
        Ap = matvec(p)
        alpha = rs / _dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _dot(r, r)
        hist.append(float(jnp.sqrt(rs_new)) / bnorm)
        if callback is not None:
            callback(it, x, hist[-1])
        if hist[-1] < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, hist

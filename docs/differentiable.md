# Differentiable forward modelling

The simulator is a JAX program, so gradients flow through most of it. This
note lists what is differentiable, what is not (and why), and the pieces
added for gradient-based fitting and inference (2026-11).

## Fitting entry point

```python
from roman_disperser import linear_operator as lo, transforms, likelihood

orders = {"1": lo.make_order_setup(psf_payload, opt_payload, sens, wl)}
op = lo.SceneOperator(orders, wl, dlam, star_xy=stars, galaxy_xy=gals,
                      galaxy_npix=npix, window=lo.auto_window(orders, xy),
                      precompute=False,
                      deposit="cubic", psf_spatial="cubic",   # C1 in position
                      sersic_subsample=3)
f = op.model_fn()                       # f(theta, dxy, sersic_params) -> e-/s

def log_post(theta, d_xi, d_eta, u_shape):
    dxy = lo.sky_dxy(ref, opt_payload, d_xi=d_xi, d_eta=d_eta)
    shape, logdet = transforms.sersic_from_unconstrained(u_shape)
    model = f(theta, dxy, shape)
    return likelihood.poisson_loglike(model, data, exptime, bkg) + logdet.sum()
```

* `theta`: spectra or `LinearBasis` coefficients (linear).
* `dxy`: per-source SCA offsets, or sky offsets through `sky_dxy`, which
  uses `optical_model_jax.perturbed_sky_to_sca` (float64 reference plus a
  float32 change).
* `sersic_params`: `(r_eff_pix, n, ba, theta)`, rendered in the graph.

## What is and is not differentiable

| Piece | Status | Notes |
|---|---|---|
| Trace polynomials, sky→SCA perturbations | smooth | `trace_sca`, `perturbed_sky_to_*` |
| `deposit="interp"` (production) | C0 in position | gradient jumps every 1/os px; zero curvature between |
| `deposit="cubic"` | C1 in position | Catmull-Rom across 4 phases/axis; equals `interp` at aligned positions; flux-conserving; ~4x table/reads. Not in the window/tiled accumulators |
| `deposit="floor"` | zero position gradient | refused by `render` / `model_fn` |
| PSF spatial, `method="linear"` | C0 | bilinear across the 4x4 nodes |
| PSF spatial, `method="cubic"` | C1 | Catmull-Rom on the node index; edge value held outside the grid |
| PSF wavelength interpolation | linear in λ | does not depend on fitted parameters |
| Sérsic images | smooth inside the clip ranges | fit through `transforms` (clips never active); `subsample` > 1 for compact profiles; `b_n(n)` is a 2000-point linear table (gradient steps of ~1e-4 relative) |
| Galaxy Jacobian warp | piecewise linear | bilinear scatter of the warped image |
| Batch loops `run_*_batch` | reverse mode | checkpointed `lax.scan` |
| Segment / window renderers | correct inside the static windows | turn on `segment_render.set_window_check(True)` (or `ROMAN_DISPERSER_CHECK_WINDOWS=1`) to raise `WindowOverflowError` instead of dropping light silently |
| Spectra from catalogue (host) | not traced | use `photometry` (AB mags, normalisation, redshift) in JAX |
| Source selection and magnitude cuts (host masks) | discrete | `selection` gives smooth `log S` with the production cuts as the `width → 0` limit |
| Poisson noise, romanisim detector stage | not differentiable | fit with `likelihood.poisson_loglike` / `gaussian_loglike`; IPC via `likelihood.ipc_convolve` |
| PSF content (STPSF cache) | constant | PSF fitting would need a JAX optical model or a PSF basis |

## Selection terms

`roman_disperser.selection` gives the per-source log selection
probabilities for a likelihood of a selected sample
(`likelihood.selected_loglike`: `log L + Σ log S_i − N log Z`):

* `trace_overlap_logp`: `catalog.select_sources` with soft box edges;
* `mag_logp`: the per-order F158 limit, with Gaussian scatter;
* `on_chip_fraction`: the counts-weighted fraction of a trace on the chip;
* `footprint_logp`: the undispersed position on the chip;
* `log_normaliser`: Monte Carlo `log Z` from reparametrised draws.

With `width=0` each function returns the production mask exactly
(`log S` = 0 or −∞); the tests check this against `catalog.select_sources`
and `pipeline.apply_order_mag_limits`.

## Precision

The model is float32. Gradient checks against finite differences are
reliable in x64 (`jax.enable_x64(True)`; see
`tests/test_differentiable.py`). On GPU, scatter-add order makes images
(and gradients) non-deterministic at the few-1e-2-of-peak level for
overlapping sources; prefer the deterministic `--accumulate window|tiled`
renderers when comparing gradients across runs.

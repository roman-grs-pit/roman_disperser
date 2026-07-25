"""Dispersing-element modes: grism (G150) and prism (P127).

Roman's WFI carries two dispersing elements, and essentially every quantity that
differs between them is a *constant* rather than an algorithm: which optical
model YAML to read, which sensitivity curves, which spectral orders exist, which
STPSF filter to ask for, and what wavelength range the element is calibrated
over. The physics — trace polynomials, PSF interpolation, dispersion, flux
conservation — is identical.

This module is the one place those constants live. It exists because the
alternative was tried and did not work: the original ``prism`` branch switched
elements by editing ``pipeline.ORDERS``, ``LAM_MIN``/``LAM_MAX`` and
``psf_model``'s STPSF filter map *in place*, which produced a checkout that could
do prism **or** grism but never both, and no way to merge the two. A module-level
constant is a single-mode global; a mode record passed as an argument is not.

Mode selection mirrors :func:`roman_disperser.paths.data_dir`:

    explicit argument  >  ``$ROMAN_DISPERSER_MODE``  >  ``"grism"``

Grism is the default so that every pre-existing call, script, notebook and SLURM
driver keeps its original behaviour without being touched.

Wavelength convention: ``lam_min``/``lam_max`` are in **microns**, matching the
optical model YAML (``optical_model.wl_min``/``wl_max``) and every wavelength
argument in ``psf_model`` and ``star_disperser``. The catalog and the sensitivity
FITS files speak Angstroms; conversion happens at those boundaries, not here.
"""

import os
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "DisperserMode", "GRISM", "PRISM", "MODES",
    "get_mode", "mode_for_optical_model", "check_band",
]


@dataclass(frozen=True)
class DisperserMode:
    """Everything that differs between the grism and the prism.

    Attributes
    ----------
    name : str
        ``"grism"`` or ``"prism"``. Matches ``meta.optical_element`` in the
        optical-model YAML, which is what :func:`mode_for_optical_model` keys on.
    optical_model_file : str
        Basename of the optical-model YAML within the vendored data directory.
        The ``v0.8`` in these names is the **upstream** (IPAC/SSC) model version,
        not ours — never renumber it locally.
    sensitivities_subdir : str
        Subdirectory of the data directory holding the sensitivity FITS files
        and their ``sensitivity_map.yaml``.
    orders : tuple of str
        Spectral orders the pipeline renders. This is a *choice*, not a property
        of the data: the grism YAML defines ``["1", "0", "2", "m1"]`` but we
        render only the first three, because ``m1`` is far off-detector for
        essentially all sources. Validated as a subset of the model's
        ``orders_defined`` by :func:`check_band`.
    stpsf_filter : dict str -> str
        Order -> STPSF ``WFI.filter`` name, for orders whose PSF STPSF can
        actually compute. Orders absent from this mapping have no STPSF filter
        of their own and must appear in ``psf_order_alias``. **Treat as
        read-only** — these are plain dicts rather than mapping proxies only so
        that a mode stays picklable, which the multiprocessing PSF-cache
        generator needs.
    psf_order_alias : dict str -> str
        Order -> the order whose PSF payload to reuse for it. For the grism,
        ``{"2": "1"}``: STPSF ships only ``GRISM0`` and ``GRISM1``, so order 2
        borrows the order-1 PSF. That is a genuine physical approximation — the
        second-order PSF is broader than the first — and it is tolerable only
        because the order-2 sensitivity is ~0.3% of order 1. The prism has a
        single beam and needs no alias.
    lam_min, lam_max : float
        Wavelength range in **microns** over which we disperse. Must lie within
        the optical model's own ``wl_min``/``wl_max`` validity range; see
        :func:`check_band`, which raises otherwise.
    romanisim_bandpass : str
        Value for ``romanisim --bandpass`` when wrapping our count-rate images
        into L1/L2 products.
    """

    name: str
    optical_model_file: str
    sensitivities_subdir: str
    orders: tuple
    stpsf_filter: dict
    psf_order_alias: dict
    lam_min: float
    lam_max: float
    romanisim_bandpass: str

    @property
    def psf_orders(self):
        """Orders for which a PSF cache is generated (one per STPSF filter)."""
        return tuple(self.stpsf_filter)

    def filter_for(self, order):
        """STPSF filter name for ``order``.

        Raises
        ------
        ValueError
            If this mode has no STPSF filter for that order. The message names
            the orders that are valid, because the failure is otherwise
            indistinguishable from a typo.
        """
        try:
            return self.stpsf_filter[order]
        except KeyError:
            raise ValueError(
                f"{self.name} mode has no STPSF filter for order {order!r}; "
                f"valid orders are {list(self.stpsf_filter)}"
            ) from None

    def psf_order(self, order):
        """Order whose PSF payload should be used to render ``order``.

        Identity for orders STPSF can compute directly; follows
        ``psf_order_alias`` otherwise (grism order 2 -> order 1).
        """
        if order in self.stpsf_filter:
            return order
        if order in self.psf_order_alias:
            return self.psf_order_alias[order]
        raise ValueError(
            f"{self.name} mode has no PSF for order {order!r}; "
            f"valid orders are {sorted(set(self.stpsf_filter) | set(self.psf_order_alias))}"
        )


GRISM = DisperserMode(
    name="grism",
    optical_model_file="Roman_grism_OpticalModel_v0.8.yaml",
    sensitivities_subdir="sensitivities",
    orders=("0", "1", "2"),
    stpsf_filter={"0": "GRISM0", "1": "GRISM1"},
    psf_order_alias={"2": "1"},
    lam_min=0.9,
    lam_max=2.0,
    romanisim_bandpass="GRISM",
)

PRISM = DisperserMode(
    name="prism",
    optical_model_file="Roman_prism_OpticalModel_v0.8.yaml",
    sensitivities_subdir="sensitivities_prism",
    orders=("1",),
    # The prism has one beam. The optical model labels it "1" for symmetry with
    # the grism's first order; it is not a diffraction order.
    stpsf_filter={"1": "PRISM"},
    psf_order_alias={},
    lam_min=0.75,
    lam_max=1.85,
    romanisim_bandpass="PRISM",
)

MODES = MappingProxyType({m.name: m for m in (GRISM, PRISM)})

DEFAULT_MODE = "grism"
MODE_ENV_VAR = "ROMAN_DISPERSER_MODE"


def get_mode(explicit=None):
    """Resolve a dispersing-element mode.

    Parameters
    ----------
    explicit : str or DisperserMode or None
        A mode name, an already-resolved :class:`DisperserMode` (returned
        unchanged, so callers can thread either through), or ``None`` to fall
        back to ``$ROMAN_DISPERSER_MODE`` and then to grism.

    Returns
    -------
    DisperserMode
    """
    if isinstance(explicit, DisperserMode):
        return explicit
    name = explicit or os.environ.get(MODE_ENV_VAR) or DEFAULT_MODE
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(
            f"unknown disperser mode {name!r}; choose from {list(MODES)}"
        ) from None


def mode_for_optical_model(model):
    """Return the mode matching a loaded optical model's ``optical_element``.

    Used by :func:`check_band` to catch a mode paired with the wrong YAML, which
    is otherwise a silent and very confusing failure: the trace polynomials load
    fine and produce plausible, wrong positions.
    """
    element = str(model.optical_element).lower()
    try:
        return MODES[element]
    except KeyError:
        raise ValueError(
            f"optical model declares optical_element={element!r}, which matches "
            f"no known mode {list(MODES)}"
        ) from None


def check_band(mode, model, padding=0.0):
    """Verify ``mode`` is consistent with a loaded optical ``model``. Raises.

    Three failure modes, all of which are silent-wrong-answer paths rather than
    crashes, which is why this raises rather than warns:

    1. **Mismatched element.** ``mode`` is prism but the YAML is the grism model
       (or vice versa). The polynomials evaluate happily and give wrong traces.
    2. **Extrapolation.** ``mode``'s band extends outside the model's
       ``wl_min``/``wl_max`` validity range, so the trace and dispersion
       polynomials are evaluated beyond the wavelengths they were fitted over.
       Polynomials diverge gracefully — plausible numbers, no warning.
    3. **Undefined order.** ``mode.orders`` contains an order the model does not
       define.

    Narrowing is *allowed*: a band strictly inside the model's validity range is
    a legitimate deliberate choice, and is not flagged. Only extrapolation is an
    error. Both modes agree exactly with their models today, so ``padding``
    defaults to 0; widen it only if a delivered model forces it.

    Parameters
    ----------
    mode : DisperserMode
    model : RomanOpticalModel
        A loaded model — this reads ``optical_element``, ``wl_min``, ``wl_max``
        and ``orders_defined``.
    padding : float
        Tolerance in **microns** by which ``mode``'s band may exceed the model's
        validity range before this raises.

    Raises
    ------
    ValueError
        On any of the three inconsistencies above.
    """
    model_mode = mode_for_optical_model(model)
    if model_mode.name != mode.name:
        raise ValueError(
            f"mode/model mismatch: mode is {mode.name!r} but the optical model "
            f"{getattr(model, 'config_file', '<unknown>')} declares "
            f"optical_element={model.optical_element!r}"
        )

    if mode.lam_min < model.wl_min - padding or mode.lam_max > model.wl_max + padding:
        raise ValueError(
            f"{mode.name} band [{mode.lam_min}, {mode.lam_max}] um extends outside "
            f"the optical model's validity range [{model.wl_min}, {model.wl_max}] um "
            f"(padding={padding} um). Dispersing there extrapolates the trace "
            f"polynomials beyond their fit and yields plausible but wrong positions."
        )

    undefined = [o for o in mode.orders if o not in model.orders_defined]
    if undefined:
        raise ValueError(
            f"{mode.name} mode renders order(s) {undefined} that the optical model "
            f"does not define; it defines {list(model.orders_defined)}"
        )

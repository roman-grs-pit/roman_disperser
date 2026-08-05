"""Dispersing-element definitions: the WFI grism (G150) and prism (P127).

The pipeline historically hard-coded grism constants (spectral orders, band,
STPSF filter names) at module level; prism support meant editing them in
place, so a checkout could simulate only one element. This module replaces
those constants with an explicit, *passed-around* value: a
:class:`DispersingElement` is nothing but a frozen bundle of the constants
that differ between the two elements. There is deliberately no behavior here
beyond validation — it is configuration, not a class hierarchy.

JAX note: an element is host-side configuration, consumed while building
payloads, file paths, and wavelength grids *before* any tracing. Nothing from
this module is passed into jit-compiled functions, so JIT compatibility is
unaffected.

Conventions
-----------
- ``name`` matches ``meta.optical_element`` in the IPAC/SSC optical-model
  YAML (``"grism"`` / ``"prism"``), and is what appears in FITS provenance.
- The grism is the default everywhere an element argument is optional; the
  prism must be asked for explicitly.
- Spectral orders are strings, as everywhere else in the package. The grism
  order ``"2"`` has no dedicated STPSF filter, so it reuses the order-1 PSF
  (``GRISM1``) — encoded in ``stpsf_filters``, which is why that mapping is
  per-element rather than a psf_model constant.
- Band edges (``lam_min``/``lam_max``, microns) must agree with the
  optical-model YAML; :func:`validate_against_model` enforces this at load
  time so a mismatched element/model pairing fails immediately rather than
  silently simulating the wrong band.
"""

from dataclasses import dataclass, field

# Tolerance for comparing element band edges against the optical-model YAML
# (microns). The YAML values are exact decimals, so this only absorbs float
# representation noise — any real disagreement is far larger.
BAND_TOL_UM = 0.005


@dataclass(frozen=True)
class DispersingElement:
    """Constants describing one WFI dispersing element (see module docstring)."""

    name: str                # matches YAML meta.optical_element
    orders: tuple            # spectral orders the pipeline simulates (strings)
    lam_min: float           # simulated band, microns (== YAML wl_min)
    lam_max: float           # simulated band, microns (== YAML wl_max)
    # dict is unhashable, so exclude it from the generated __hash__ (elements
    # are only hashed if someone keys a dict on one; keying on .name is better)
    stpsf_filters: dict = field(hash=False)  # order -> STPSF filter name
    optical_model_file: str    # YAML filename in the data dir
    sensitivities_subdir: str  # sensitivity FITS subdir in the data dir
    bandpass: str              # BANDPASS value in APT ECSV pointing tables


GRISM = DispersingElement(
    name="grism",
    orders=("0", "1", "2"),
    lam_min=0.9,
    lam_max=2.0,
    stpsf_filters={"0": "GRISM0", "1": "GRISM1", "2": "GRISM1"},
    optical_model_file="Roman_grism_OpticalModel_v0.8.yaml",
    sensitivities_subdir="sensitivities",
    bandpass="GRISM",
)

PRISM = DispersingElement(
    name="prism",
    orders=("1",),
    lam_min=0.75,
    lam_max=1.85,
    stpsf_filters={"1": "PRISM"},
    optical_model_file="Roman_prism_OpticalModel_v0.8.yaml",
    sensitivities_subdir="sensitivities_prism",
    bandpass="PRISM",
)

ELEMENTS = {"grism": GRISM, "prism": PRISM}


def get_element(element=None):
    """Return a :class:`DispersingElement` from a name (default: grism).

    Accepts an element instance (returned unchanged) or a name string.
    Raises ``ValueError`` on an unknown name so a typo in a config fails
    loudly instead of falling back to the default.
    """
    if element is None:
        return GRISM
    if isinstance(element, DispersingElement):
        return element
    try:
        return ELEMENTS[element.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"Unknown dispersing element {element!r}; "
            f"expected one of {sorted(ELEMENTS)}"
        ) from None


def validate_against_model(element, model):
    """Raise ``ValueError`` unless ``model`` is the right model for ``element``.

    ``model`` is a loaded ``RomanOpticalModel``. Three checks, all against
    what the YAML itself declares:

    1. ``meta.optical_element`` equals ``element.name`` — catches pointing a
       grism run at the prism YAML (or vice versa) outright;
    2. band edges agree within :data:`BAND_TOL_UM` — catches an element whose
       simulated band disagrees with the model's calibrated band (the
       silent-band-mismatch failure this module exists to prevent);
    3. every simulated order is defined in the YAML.

    Silence is not acceptable here (per the design decision): any mismatch is
    an error, never a warning.
    """
    element = get_element(element)
    problems = []
    if model.optical_element != element.name:
        problems.append(
            f"optical_element is {model.optical_element!r}, "
            f"element is {element.name!r}"
        )
    if abs(model.wl_min - element.lam_min) > BAND_TOL_UM or \
       abs(model.wl_max - element.lam_max) > BAND_TOL_UM:
        problems.append(
            f"model band [{model.wl_min}, {model.wl_max}] um does not match "
            f"element band [{element.lam_min}, {element.lam_max}] um "
            f"(tolerance {BAND_TOL_UM} um)"
        )
    missing = [o for o in element.orders if o not in model.orders_defined]
    if missing:
        problems.append(
            f"orders {missing} not in model orders_defined "
            f"{model.orders_defined}"
        )
    if problems:
        raise ValueError(
            f"Optical model {model.config_file!r} is inconsistent with "
            f"dispersing element {element.name!r}: " + "; ".join(problems)
        )
    return element

"""End-to-end golden-frame regression test.

Renders the fixed scene in ``tests/golden_frame.py`` through the production
dispersal path and compares the full detector frame against the pinned
reference asset (``golden_frame.GOLDEN_VERSION`` in the vendored data dir).
This is the durable equivalence guard for refactorings of the dispersal
internals — see the ``golden_frame`` module docstring for the design
rationale (scene choice, full-frame storage, the two tiers, per-backend
references, and the code/reference atomicity rule for intentional results
changes).

Gates (all thresholds from measurements on this scene, 2026-08-24, a10g +
CPU, jax 0.7.2 — see the golden_frame docstring and the research log):

* **Flux** (always, backend-insensitive): ``|sum(new) - sum(ref)| /
  |sum(ref)| <= 1e-7``, sums in float64. Measured floor ~3e-8 even *across*
  backends; one decade of headroom.
* **Per-pixel, same-backend reference** (CPU on CPU; a GPU model with its
  own blessed frames): ``allclose(rtol=1e-5, atol=1e-5 * max|ref|)``. CPU
  renders are deterministic; GPU run-to-run repeats measured <=6e-7 of
  peak (issue #22 scatter-order floor), so this gate has >=10x headroom.
* **Per-pixel, cross-backend fallback** (a GPU model without blessed
  frames, compared against the CPU reference): ``atol = 0.1 * max|ref|``.
  Cross-backend pixel differences reach 3.2e-2 of peak from benign
  position-rounding boundary flips (flux-conserving row swaps — see
  golden_frame docstring), so this tier only catches gross breakage
  (shifted traces, missing objects, wrong flux scale); the flux gate does
  the fine-grained work there.

A tight-gate failure whose diff shows the paired-adjacent-row signature
(equal-and-opposite net flux, total flux unchanged) may be the boundary-flip
mechanism resurfacing after a jax/driver change rather than a code bug —
diagnose before re-blessing or "fixing".
"""

import numpy as np
import pytest

from tests import golden_frame

RTOL = 1e-5
ATOL_SCALE_TIGHT = 1e-5     # x max|ref|, same-backend reference
ATOL_SCALE_CROSS = 1e-1     # x max|ref|, cross-backend fallback
REL_SUM_TOL = 1e-7


def _reference_for_backend(element_name, order, tier):
    """Pick the blessed reference for this backend: exact match if published,
    else the CPU frames with the loose cross-backend gate."""
    subdir = golden_frame.backend_subdir()
    try:
        ref, prov = golden_frame.load_reference(element_name, order, tier,
                                                subdir=subdir)
        return ref, prov, ATOL_SCALE_TIGHT, subdir
    except FileNotFoundError:
        if subdir == "cpu":
            raise
        ref, prov = golden_frame.load_reference(element_name, order, tier,
                                                subdir="cpu")
        return ref, prov, ATOL_SCALE_CROSS, f"cpu (no {subdir} frames; loose gate)"


def _compare(element_name, order, tier, dlam_A):
    try:
        ref, prov, atol_scale, ref_desc = _reference_for_backend(
            element_name, order, tier)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert prov["dlam_A"] == dlam_A, (
        f"reference {prov['golden_version']} was generated at "
        f"{prov['dlam_A']} A but the test expects {dlam_A} A — "
        "reference and code are out of step")

    new = golden_frame.render_frame(element_name, order, dlam_A)

    ref_sum = float(np.sum(ref, dtype=np.float64))
    new_sum = float(np.sum(new, dtype=np.float64))
    rel_sum = abs(new_sum - ref_sum) / max(abs(ref_sum), 1e-30)
    assert rel_sum <= REL_SUM_TOL, (
        f"total flux moved (vs {ref_desc}): ref {ref_sum:.9e} vs new "
        f"{new_sum:.9e} (rel {rel_sum:.3e} > {REL_SUM_TOL})")

    atol = atol_scale * float(np.abs(ref).max())
    max_abs = float(np.abs(new - ref).max())
    assert np.allclose(new, ref, rtol=RTOL, atol=atol), (
        f"per-pixel mismatch vs {ref_desc}: max|diff| = {max_abs:.3e} "
        f"(atol {atol:.3e}, rtol {RTOL}) — if the diff is paired "
        "adjacent rows with opposite sign and the flux gate passed, see "
        "the boundary-flip note in tests/golden_frame.py")


@pytest.mark.parametrize("element_name,order", golden_frame.CONFIGS_COARSE)
def test_golden_frame_coarse(element_name, order):
    _compare(element_name, order, "coarse", golden_frame.DLAM_COARSE_A)


@pytest.mark.slow
@pytest.mark.parametrize("element_name,order", golden_frame.CONFIGS_FULL)
def test_golden_frame_full(element_name, order):
    _compare(element_name, order, "full", golden_frame.DLAM_FULL_A)

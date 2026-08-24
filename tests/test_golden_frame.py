"""End-to-end golden-frame regression test.

Renders the fixed scene in ``tests/golden_frame.py`` through the production
dispersal path and compares the full detector frame against the pinned
reference asset (``golden_frame.GOLDEN_VERSION`` in the vendored data dir).
This is the durable equivalence guard for refactorings of the dispersal
internals — see the ``golden_frame`` module docstring for the design
rationale (scene choice, full-frame storage, the two tiers, and the
code/reference atomicity rule for intentional results changes).

Gate
----
The standing equivalence gate from the perf/native-deposit work (the
issue #22 CPU/GPU noise floor):

* per-pixel: ``allclose(new, ref, rtol=1e-5, atol=1e-5 * max|ref|)`` — the
  frame-scaled atol keeps empty and PSF-wing pixels from failing on
  meaningless relative differences;
* total flux: ``|sum(new) - sum(ref)| / |sum(ref)| <= 1e-7``, sums taken in
  float64. The measured CPU-vs-GPU floor is ~1e-8 per SCA frame
  (workbench/20260818-perf-exploration); one decade of headroom keeps the
  gate insensitive to benign accumulation-order changes while still
  catching real flux errors.

References are generated on CPU; this test passes on CPU and GPU backends
(the gate absorbs the cross-backend difference).

The ``full`` tier (production 2 A sampling) is marked ``slow``: it is
required before merging any PR that touches the dispersal path (see
CLAUDE.md), alongside the ``benchmarks/`` perf suite.
"""

import numpy as np
import pytest

from tests import golden_frame

RTOL = 1e-5
ATOL_SCALE = 1e-5   # x max|ref|
REL_SUM_TOL = 1e-7


def _compare(element_name, order, tier, dlam_A):
    try:
        ref, prov = golden_frame.load_reference(element_name, order, tier)
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
        f"total flux moved: ref {ref_sum:.9e} vs new {new_sum:.9e} "
        f"(rel {rel_sum:.3e} > {REL_SUM_TOL})")

    atol = ATOL_SCALE * float(np.abs(ref).max())
    max_abs = float(np.abs(new - ref).max())
    assert np.allclose(new, ref, rtol=RTOL, atol=atol), (
        f"per-pixel mismatch: max|diff| = {max_abs:.3e} "
        f"(atol {atol:.3e}, rtol {RTOL})")


@pytest.mark.parametrize("element_name,order", golden_frame.CONFIGS_COARSE)
def test_golden_frame_coarse(element_name, order):
    _compare(element_name, order, "coarse", golden_frame.DLAM_COARSE_A)


@pytest.mark.slow
@pytest.mark.parametrize("element_name,order", golden_frame.CONFIGS_FULL)
def test_golden_frame_full(element_name, order):
    _compare(element_name, order, "full", golden_frame.DLAM_FULL_A)

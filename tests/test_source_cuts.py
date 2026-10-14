"""Declarative source cuts in scripts/build_dispersed_image.py (thread xmm-scale-perf)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def bdi():
    spec = importlib.util.spec_from_file_location(
        "build_dispersed_image", SCRIPTS / "build_dispersed_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta():
    # mag 20, 26.9, 27.1 galaxies; one r_hl=0 bright galaxy; one NaN-flux galaxy;
    # two stars (r_hl=0, one faint). Non-default index, as after any filtering.
    mag = np.array([20.0, 26.9, 27.1, 21.0, np.nan, 18.0, 27.5])
    return pd.DataFrame({
        "type": ["SER", "SER", "SER", "SER", "SER", "PSF", "PSF"],
        "F158": (10 ** (-0.4 * mag)).astype(np.float32),
        "half_light_radius": np.array([0.3, 0.2, 0.2, 0.0, 0.3, 0.0, 0.0], np.float32),
    }, index=[10, 11, 12, 13, 14, 15, 16])


def _log(_):
    pass


def test_no_cuts_is_identity(bdi):
    m = _meta()
    for cuts in (None, []):
        out, summary = bdi.apply_source_cuts(m, cuts, log=_log)
        assert out is m and summary == []


def test_keith_filter_shape(bdi):
    """H <= 27 for all types, plus r_hl > 0 for galaxies only."""
    cuts = [{"column": "F158", "op": "mag_le", "value": 27.0},
            {"column": "half_light_radius", "op": "gt", "value": 0.0, "types": ["SER"]}]
    out, summary = bdi.apply_source_cuts(_meta(), cuts, log=_log)
    # kept: SER mag 20, 26.9; star mag 18. Removed: 27.1, NaN (mag rule),
    # star 27.5 (mag rule); r_hl=0 galaxy (size rule). Star with r_hl=0 kept.
    assert list(out.index) == [10, 11, 15]               # original index preserved
    assert [s["removed"] for s in summary] == [3, 1]     # sequential, no double count
    assert summary[1]["types"] == ["SER"]


def test_ops(bdi):
    m = _meta()
    r = lambda c: list(bdi.apply_source_cuts(m, [c], log=_log)[0].index)
    assert r({"column": "half_light_radius", "op": "between", "value": [0.1, 0.25]}) == [11, 12]
    assert r({"column": "type", "op": "isin", "value": ["PSF"]}) == [15, 16]
    assert r({"column": "F158", "op": "mag_ge", "value": 27.0, "types": ["SER"]}) == [12, 15, 16]
    assert r({"column": "half_light_radius", "op": "le", "value": 0.0}) == [13, 15, 16]


@pytest.mark.parametrize("cuts,match", [
    ({"column": "F158"}, "list"),
    ([{"column": "F158", "op": "mag_le"}], "column/op/value"),
    ([{"column": "F158", "op": "magle", "value": 27}], "not in"),
    ([{"column": "nope", "op": "gt", "value": 0}], "not in catalog"),
    ([{"column": "F158", "op": "gt", "value": 0, "typs": ["SER"]}], "unknown keys"),
    ([{"column": "F158", "op": "between", "value": [1, 2, 3]}], "between"),
])
def test_bad_rules_raise(bdi, cuts, match):
    with pytest.raises(ValueError, match=match):
        bdi.apply_source_cuts(_meta(), cuts, log=_log)


def test_describe(bdi):
    assert (bdi.describe_source_cut({"column": "F158", "op": "mag_le", "value": 27.0,
                                     "types": ["SER"]}) == "F158 mag_le 27.0 [SER]")

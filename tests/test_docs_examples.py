"""The Python examples in docs/optical_model.md run as written (cluster T19)."""

import re
from pathlib import Path

import pytest

DOC = Path(__file__).parent.parent / "docs" / "optical_model.md"


def test_optical_model_doc_examples_run():
    from roman_disperser import paths
    try:
        paths.optical_model_path()
    except Exception as exc:  # noqa: BLE001 - no hydrated data
        pytest.skip(f"optical model not hydrated: {exc}")
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(), re.S)
    assert len(blocks) >= 4
    ns = {}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{DOC.name}[block {i}]", "exec"), ns)
        except Exception as exc:
            raise AssertionError(f"block {i} of {DOC.name} failed: "
                                 f"{type(exc).__name__}: {exc}") from exc
    assert ns["xmpa"].shape == (10, 20)
    assert ns["J"].shape == (2, 2, 3)

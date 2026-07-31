"""Enforce the matmul precision convention by static analysis.

Why this test exists
--------------------
With ``jax_enable_x64`` off, XLA:GPU serves an unannotated float32
``dot_general`` as TF32 on Ampere and later — a 10-bit mantissa, eps about
4.9e-4 — while the identical op on CPU is exact. In July 2026 a single
unannotated ``rot_matrix @ xy`` in ``optical_model_jax.get_fpa_pos`` displaced
every source in the shipped SSC line-grid package by a median 1.84 px and up to
7.08 px. It survived two days of searching because every reproduction attempt
ran on CPU, where the line is exact.

So the defect class is: *correct on the machine you test on, wrong on the
machine you run on*. Three independent defences failed simultaneously —
``tests/test_disperser_gpu.py`` skips silently on a GPU-less head node, the
oracle it compared against shared the same bug, and its tolerance was stated in
degrees (1e-3 deg = 33 px). This test is the one defence that cannot skip: it
is pure AST analysis, needs no GPU, and runs in milliseconds.

It is deliberately *static* rather than a runtime check. A global
``JAX_DEFAULT_MATMUL_PRECISION`` would make the numerics right while leaving
the source wrong, so the next unannotated op would be silently absorbed instead
of reported — and library code has no business setting a process-global that
also governs unrelated JAX work in the same interpreter, some of which may
legitimately want TF32.
"""

import ast
from pathlib import Path

import pytest

# Package source under enforcement. Vendored modules are excluded: they are
# NumPy (host, float64), carry no JAX ops, and are maintained upstream.
SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "roman_disperser"
VENDORED = {"optical_model.py", "optical_model_utils.py"}

# jnp/lax callables whose float32 form XLA may lower to TF32.
MATMUL_FUNCS = {
    "matmul", "dot", "vdot", "inner", "outer", "tensordot", "einsum",
    "dot_general", "conv", "conv_general_dilated",
}


class _MatmulVisitor(ast.NodeVisitor):
    """Collect matmul-class ops lacking an explicit ``precision=`` keyword."""

    def __init__(self, filename):
        self.filename = filename
        self.violations = []

    def visit_BinOp(self, node):
        # `a @ b` is jnp.matmul at default precision and takes no keyword, so
        # it can never satisfy the convention. Use jnp.matmul(..., precision=)
        # instead. NumPy operands are host float64 and unaffected, but the AST
        # cannot tell them apart, hence the vendored-file exclusion above.
        if isinstance(node.op, ast.MatMult):
            self.violations.append(
                (node.lineno, "`@` operator (takes no precision= argument)")
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        if name in MATMUL_FUNCS:
            if not any(kw.arg == "precision" for kw in node.keywords):
                self.violations.append(
                    (node.lineno, f"{name}() without precision=")
                )
        self.generic_visit(node)


def _scan(source, filename="<test>"):
    visitor = _MatmulVisitor(filename)
    visitor.visit(ast.parse(source))
    return visitor.violations


def _source_files():
    return sorted(
        p for p in SRC_DIR.glob("*.py") if p.name not in VENDORED
    )


@pytest.mark.parametrize(
    "path", _source_files(), ids=lambda p: p.name
)
def test_matmul_ops_declare_precision(path):
    """Every matmul-class JAX op in the package declares precision=."""
    violations = _scan(path.read_text(), path.name)
    assert not violations, (
        f"{path.name}: matmul-class op(s) without precision='highest'. On "
        f"XLA:GPU these run as TF32 (eps ~ 4.9e-4) while being exact on CPU:\n"
        + "\n".join(f"  line {ln}: {what}" for ln, what in violations)
    )


# -- Self-tests: an AST checker that matches nothing is worse than none -------
#
# These pin the checker's own behaviour. Without them a refactor that broke the
# visitor would turn the test above green for the wrong reason -- the exact
# silent-pass failure mode this file exists to prevent.

BAD_CASES = [
    ("bare @ operator", "xy = rot_matrix @ xy"),
    ("jnp.matmul, no precision", "z = jnp.matmul(a, b)"),
    ("jnp.einsum, no precision", "z = jnp.einsum('ij,jk->ik', a, b)"),
    ("jnp.dot, no precision", "z = jnp.dot(a, b)"),
    ("jnp.tensordot, no precision", "z = jnp.tensordot(a, b, axes=1)"),
    ("lax.dot_general, no precision",
     "z = lax.dot_general(a, b, dimension_numbers=dn)"),
    ("nested inside a call", "z = jnp.sum(jnp.matmul(a, b))"),
    ("inside a comprehension", "zs = [jnp.matmul(a, b) for b in bs]"),
    ("augmented assignment", "acc += jnp.einsum('ij,jk->ik', a, b)"),
]

GOOD_CASES = [
    ("matmul with precision", "z = jnp.matmul(a, b, precision='highest')"),
    ("einsum with precision",
     "z = jnp.einsum('ij,jk->ik', a, b, precision='highest')"),
    ("dot with precision", "z = jnp.dot(a, b, precision='highest')"),
    ("elementwise multiply is not a matmul", "z = a * b"),
    ("fftconvolve is FFT-based, not a dot_general",
     "z = jax.scipy.signal.fftconvolve(a, b, mode='full')"),
    ("unrelated call", "z = jnp.sum(a)"),
    ("decorator matrix-multiply-free", "@jax.jit\ndef f(x):\n    return x + 1"),
]


@pytest.mark.parametrize(
    "label,source", BAD_CASES, ids=[c[0] for c in BAD_CASES]
)
def test_checker_catches_violations(label, source):
    """The checker must flag each known-bad form."""
    assert _scan(source), f"checker failed to flag: {label}"


@pytest.mark.parametrize(
    "label,source", GOOD_CASES, ids=[c[0] for c in GOOD_CASES]
)
def test_checker_accepts_compliant_code(label, source):
    """The checker must not flag compliant or unrelated code."""
    assert not _scan(source), f"checker false-positived on: {label}"


def test_checker_reports_correct_line_number():
    """Violations carry a usable line number, not just a boolean."""
    source = "import jax.numpy as jnp\nx = 1\nz = jnp.matmul(a, b)\n"
    violations = _scan(source)
    assert len(violations) == 1
    assert violations[0][0] == 3


def test_regression_get_fpa_pos_rotation_is_annotated():
    """Pin the specific line that shipped wrong (see PR #18).

    Named separately from the sweep above so that a future exclusion of
    optical_model_jax.py from the scan cannot quietly drop coverage of the one
    site known to have caused a science defect.
    """
    source = (SRC_DIR / "optical_model_jax.py").read_text()
    tree = ast.parse(source)

    matmuls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "matmul"
    ]
    assert matmuls, "expected a jnp.matmul in the sky->FPA rotation"
    for node in matmuls:
        kwargs = {kw.arg: kw for kw in node.keywords}
        assert "precision" in kwargs, (
            f"line {node.lineno}: rotation matmul lost its precision= "
            "annotation; this is the TF32 defect that displaced the 2026-07 "
            "SSC line-grid package by up to 7.08 px"
        )
        value = kwargs["precision"].value
        assert isinstance(value, ast.Constant) and value.value == "highest", (
            f"line {node.lineno}: precision must be the literal 'highest'"
        )

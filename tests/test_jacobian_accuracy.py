"""
Test Jacobian accuracy for the Sobol disperser algorithm.

Validates that the linear Jacobian approximation is accurate enough
to replace full trace_beam computation within cells.

Cell size: 10 × 10 SCA pixels × 100Å
Threshold: max error < 0.01 pixel
"""

import os
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from roman_disperser.optical_model import RomanOpticalModel
import roman_disperser.optical_model_jax as omj

# Test configuration
CELL_SIZE_X = 10.0       # SCA pixels
CELL_SIZE_Y = 10.0       # SCA pixels
CELL_SIZE_LAM = 0.01     # microns (100 Angstrom)

X_MIN, X_MAX = -500.0, 5500.0
Y_MIN, Y_MAX = -500.0, 5500.0
LAM_MIN, LAM_MAX = 0.9, 2.0

N_SAMPLES = 1000
RANDOM_SEED = 42
MAX_ERROR_THRESHOLD = 0.01  # pixels


# --- Helper functions (module-level for reuse) ---

def trace_sca_to_sca(payload, xsca, ysca, wavelength):
    """Full trace from SCA input to SCA output coordinates."""
    xfpa, yfpa = omj.sca_to_fpa(payload, xsca, ysca)
    xmpa, ympa = omj.trace_beam(payload, xfpa, yfpa, wavelength)
    xsca_out, ysca_out = omj.mpa_to_sca(payload, xmpa, ympa)
    return jnp.stack([xsca_out, ysca_out])


def compute_jacobian_at_point(payload, xsca, ysca, wavelength):
    """Compute 2x3 Jacobian at a single point."""
    def trace_single(inputs):
        return trace_sca_to_sca(
            payload, inputs[0:1], inputs[1:2], inputs[2:3]
        ).squeeze()
    inputs = jnp.array([xsca, ysca, wavelength])
    return jax.jacobian(trace_single)(inputs)


# Corner offsets for 8 corners of 3D cell
CORNER_OFFSETS = jnp.array([
    [-CELL_SIZE_X/2, -CELL_SIZE_Y/2, -CELL_SIZE_LAM/2],
    [-CELL_SIZE_X/2, -CELL_SIZE_Y/2, +CELL_SIZE_LAM/2],
    [-CELL_SIZE_X/2, +CELL_SIZE_Y/2, -CELL_SIZE_LAM/2],
    [-CELL_SIZE_X/2, +CELL_SIZE_Y/2, +CELL_SIZE_LAM/2],
    [+CELL_SIZE_X/2, -CELL_SIZE_Y/2, -CELL_SIZE_LAM/2],
    [+CELL_SIZE_X/2, -CELL_SIZE_Y/2, +CELL_SIZE_LAM/2],
    [+CELL_SIZE_X/2, +CELL_SIZE_Y/2, -CELL_SIZE_LAM/2],
    [+CELL_SIZE_X/2, +CELL_SIZE_Y/2, +CELL_SIZE_LAM/2],
])


def measure_cell_error(payload, xc, yc, lamc):
    """Compute max Jacobian approximation error at cell corners."""
    center_out = trace_sca_to_sca(
        payload, jnp.array([xc]), jnp.array([yc]), jnp.array([lamc])
    ).squeeze()
    J = compute_jacobian_at_point(payload, xc, yc, lamc)

    max_error = 0.0
    for offset in CORNER_OFFSETS:
        corner_x, corner_y, corner_lam = xc + offset[0], yc + offset[1], lamc + offset[2]
        full_out = trace_sca_to_sca(
            payload, jnp.array([corner_x]), jnp.array([corner_y]), jnp.array([corner_lam])
        ).squeeze()
        approx_out = center_out + J @ offset
        error = jnp.sqrt(jnp.sum((full_out - approx_out)**2))
        max_error = jnp.maximum(max_error, error)
    return max_error


def make_vectorized_error_fn(payload):
    """Create JIT-compiled vectorized error function."""
    @jax.jit
    def measure_many_errors(x_arr, y_arr, lam_arr):
        def single_error(x, y, lam):
            return measure_cell_error(payload, x, y, lam)
        return jax.vmap(single_error)(x_arr, y_arr, lam_arr)
    return measure_many_errors


# --- Fixtures ---

@pytest.fixture(scope="module")
def model():
    """Load optical model once for all tests."""
    pixi_root_path = os.environ.get("PIXI_PROJECT_ROOT", ".")
    fn = os.path.join(pixi_root_path, "data/Roman_grism_OpticalModel_v0.8.yaml")
    return RomanOpticalModel(fn)


@pytest.fixture(scope="module")
def random_samples():
    """Generate random cell centers (shared across all tests)."""
    np.random.seed(RANDOM_SEED)
    return {
        'x': jnp.array(np.random.uniform(X_MIN, X_MAX, N_SAMPLES).astype(np.float32)),
        'y': jnp.array(np.random.uniform(Y_MIN, Y_MAX, N_SAMPLES).astype(np.float32)),
        'lam': jnp.array(np.random.uniform(LAM_MIN, LAM_MAX, N_SAMPLES).astype(np.float32)),
    }


# --- Tests ---

class TestJacobianAccuracy:
    """Test Jacobian approximation accuracy for Sobol disperser."""

    @pytest.mark.parametrize("sca", range(1, 19))
    @pytest.mark.parametrize("order", ["0", "1", "2"])
    def test_jacobian_accuracy(self, model, random_samples, sca, order):
        """Verify Jacobian approximation error < 0.01 pixel for all SCAs/orders."""
        payload = omj.make_sca_payload(model, sca=sca, order=order)
        error_fn = make_vectorized_error_fn(payload)

        errors = error_fn(random_samples['x'], random_samples['y'], random_samples['lam'])
        errors_np = np.array(errors)

        max_error = errors_np.max()
        assert max_error < MAX_ERROR_THRESHOLD, (
            f"SCA {sca}, order {order}: max error {max_error:.6f} >= {MAX_ERROR_THRESHOLD} pixel"
        )

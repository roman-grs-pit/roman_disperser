"""
GPU-specific tests for the disperser module.

These tests verify that CPU and GPU produce identical results
when using the disperser functions.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roman_disperser.optical_model_jax as omj
from roman_disperser.optical_model import RomanOpticalModel
from roman_disperser.disperser import disperse_2d1d_sca

# Tolerances for float32 precision
# Note: disperser tests may have slightly larger differences due to
# bilinear scatter-add accumulation order differences between CPU/GPU
RTOL = 1e-5
ATOL = 2e-3


def has_gpu():
    """Check if GPU is available."""
    try:
        devices = jax.devices('gpu')
        return len(devices) > 0
    except RuntimeError:
        return False


def move_payload_to_device(payload, device):
    """Move all JAX arrays in payload to specified device."""
    def move_value(v):
        if isinstance(v, jnp.ndarray):
            return jax.device_put(v, device)
        elif isinstance(v, dict):
            return {k: move_value(vv) for k, vv in v.items()}
        else:
            return v

    return {k: move_value(v) for k, v in payload.items()}


# Skip entire module if no GPU available
pytestmark = pytest.mark.skipif(
    not has_gpu(),
    reason="GPU not available"
)


@pytest.fixture(scope="module")
def model():
    """Load optical model once for all tests."""
    pixi_root_path = os.environ.get("PIXI_PROJECT_ROOT", ".")
    fn = os.path.join(pixi_root_path, "data/Roman_prism_OpticalModel_v0.8.yaml")
    return RomanOpticalModel(fn)


class TestCPUvsGPU:
    """Test that CPU and GPU produce identical results."""

    def test_disperser_cpu_vs_gpu(self, model):
        """Verify CPU and GPU disperser results match."""
        # Create payload
        payload = omj.make_sca_payload(model, sca=1, order="1")

        # Create test data (as numpy, will be moved to device)
        image_np = np.ones((10, 10), dtype=np.float32)
        spec_np = np.ones(50, dtype=np.float32)
        x0, y0 = 2000.0, 2000.0
        dx, dy = 1.0, 1.0
        lam0 = float(model.wl_grid[0])
        dlam = float(model.wl_grid[1] - model.wl_grid[0])

        # Run on CPU
        cpu_device = jax.devices('cpu')[0]
        payload_cpu = move_payload_to_device(payload, cpu_device)
        image_cpu = jax.device_put(jnp.array(image_np), cpu_device)
        spec_cpu = jax.device_put(jnp.array(spec_np), cpu_device)
        output_cpu = jax.device_put(jnp.zeros((4088, 4088), dtype=jnp.float32), cpu_device)

        result_cpu = disperse_2d1d_sca(
            payload_cpu, image_cpu, x0, y0, dx, dy, spec_cpu, lam0, dlam, output_cpu
        )
        result_cpu.block_until_ready()
        result_cpu_np = np.array(result_cpu)

        # Run on GPU
        gpu_device = jax.devices('gpu')[0]
        payload_gpu = move_payload_to_device(payload, gpu_device)
        image_gpu = jax.device_put(jnp.array(image_np), gpu_device)
        spec_gpu = jax.device_put(jnp.array(spec_np), gpu_device)
        output_gpu = jax.device_put(jnp.zeros((4088, 4088), dtype=jnp.float32), gpu_device)

        result_gpu = disperse_2d1d_sca(
            payload_gpu, image_gpu, x0, y0, dx, dy, spec_gpu, lam0, dlam, output_gpu
        )
        result_gpu.block_until_ready()
        result_gpu_np = np.array(result_gpu)

        # Compare
        np.testing.assert_allclose(
            result_cpu_np, result_gpu_np,
            rtol=RTOL, atol=ATOL,
            err_msg="CPU and GPU disperser results do not match"
        )

    def test_trace_beam_cpu_vs_gpu(self, model):
        """Verify CPU and GPU trace_beam results match."""
        # Create payload
        payload = omj.make_sca_payload(model, sca=1, order="1")

        # Create test data - random FPA coordinates
        # Note: trace_beam expects wavelength array length to match number of FPA points
        # (the einsum contracts over matching 'n' dimensions)
        wavelength_np = np.array(model.wl_grid, dtype=np.float32)
        n_points = len(wavelength_np)
        np.random.seed(42)
        xfpa_np = np.random.uniform(-0.5, 0.5, n_points).astype(np.float32)
        yfpa_np = np.random.uniform(-0.5, 0.5, n_points).astype(np.float32)

        # Run on CPU
        cpu_device = jax.devices('cpu')[0]
        payload_cpu = move_payload_to_device(payload, cpu_device)
        xfpa_cpu = jax.device_put(jnp.array(xfpa_np), cpu_device)
        yfpa_cpu = jax.device_put(jnp.array(yfpa_np), cpu_device)
        wavelength_cpu = jax.device_put(jnp.array(wavelength_np), cpu_device)

        xmpa_cpu, ympa_cpu = omj.trace_beam(payload_cpu, xfpa_cpu, yfpa_cpu, wavelength_cpu)
        xmpa_cpu.block_until_ready()
        ympa_cpu.block_until_ready()
        xmpa_cpu_np = np.array(xmpa_cpu)
        ympa_cpu_np = np.array(ympa_cpu)

        # Run on GPU
        gpu_device = jax.devices('gpu')[0]
        payload_gpu = move_payload_to_device(payload, gpu_device)
        xfpa_gpu = jax.device_put(jnp.array(xfpa_np), gpu_device)
        yfpa_gpu = jax.device_put(jnp.array(yfpa_np), gpu_device)
        wavelength_gpu = jax.device_put(jnp.array(wavelength_np), gpu_device)

        xmpa_gpu, ympa_gpu = omj.trace_beam(payload_gpu, xfpa_gpu, yfpa_gpu, wavelength_gpu)
        xmpa_gpu.block_until_ready()
        ympa_gpu.block_until_ready()
        xmpa_gpu_np = np.array(xmpa_gpu)
        ympa_gpu_np = np.array(ympa_gpu)

        # Compare
        np.testing.assert_allclose(
            xmpa_cpu_np, xmpa_gpu_np,
            rtol=RTOL, atol=ATOL,
            err_msg="CPU and GPU trace_beam xmpa results do not match"
        )
        np.testing.assert_allclose(
            ympa_cpu_np, ympa_gpu_np,
            rtol=RTOL, atol=ATOL,
            err_msg="CPU and GPU trace_beam ympa results do not match"
        )

    def test_get_trace_coeffs_cpu_vs_gpu(self, model):
        """Verify CPU and GPU get_trace_coeffs results match."""
        # Create payload
        payload = omj.make_sca_payload(model, sca=1, order="1")

        # Create test data
        np.random.seed(42)
        xfpa_np = np.random.uniform(-0.5, 0.5, 50).astype(np.float32)
        yfpa_np = np.random.uniform(-0.5, 0.5, 50).astype(np.float32)

        # Run on CPU
        cpu_device = jax.devices('cpu')[0]
        payload_cpu = move_payload_to_device(payload, cpu_device)
        xfpa_cpu = jax.device_put(jnp.array(xfpa_np), cpu_device)
        yfpa_cpu = jax.device_put(jnp.array(yfpa_np), cpu_device)

        crv_cpu, ids_cpu = omj.get_trace_coeffs(payload_cpu, xfpa_cpu, yfpa_cpu)
        crv_cpu.block_until_ready()
        ids_cpu.block_until_ready()
        crv_cpu_np = np.array(crv_cpu)
        ids_cpu_np = np.array(ids_cpu)

        # Run on GPU
        gpu_device = jax.devices('gpu')[0]
        payload_gpu = move_payload_to_device(payload, gpu_device)
        xfpa_gpu = jax.device_put(jnp.array(xfpa_np), gpu_device)
        yfpa_gpu = jax.device_put(jnp.array(yfpa_np), gpu_device)

        crv_gpu, ids_gpu = omj.get_trace_coeffs(payload_gpu, xfpa_gpu, yfpa_gpu)
        crv_gpu.block_until_ready()
        ids_gpu.block_until_ready()
        crv_gpu_np = np.array(crv_gpu)
        ids_gpu_np = np.array(ids_gpu)

        # Compare
        np.testing.assert_allclose(
            crv_cpu_np, crv_gpu_np,
            rtol=RTOL, atol=ATOL,
            err_msg="CPU and GPU get_trace_coeffs crv results do not match"
        )
        np.testing.assert_allclose(
            ids_cpu_np, ids_gpu_np,
            rtol=RTOL, atol=ATOL,
            err_msg="CPU and GPU get_trace_coeffs ids results do not match"
        )

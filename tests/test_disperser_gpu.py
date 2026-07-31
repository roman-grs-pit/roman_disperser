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
    fn = os.path.join(pixi_root_path, "data/Roman_grism_OpticalModel_v0.8.yaml")
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


# Roman WFI plate scale, for stating position tolerances in pixels rather than
# degrees. ATOL above is 2e-3 *degrees* = 65 px, which is a sensible tolerance
# for an image comparison and a useless one for a position.
PIXEL_SCALE_ARCSEC = 0.11


def _deg_to_px(x):
    return np.asarray(x) * 3600.0 / PIXEL_SCALE_ARCSEC


class TestSkyToFPACPUvsGPU:
    """The sky -> FPA transform must agree across devices, and with float64.

    This is the regression test for the July 2026 TF32 defect. The rotation in
    `get_fpa_pos` was an unannotated float32 matmul, which XLA:GPU lowered to
    TF32 (10-bit mantissa) on Ampere while running it exactly on CPU. Sources
    were placed a median 1.84 px -- up to 7.08 px -- from where the catalogue
    said they were.

    Two properties are asserted, and the second is the one that matters:

    1. CPU and GPU agree. Catches a device-dependent lowering.
    2. Both agree with an independent float64 NumPy evaluation. This is the
       load-bearing check: if a future change made *both* devices wrong in the
       same way, property 1 would still pass.

    Tolerances are in pixels.
    """

    # Deliberately at a high absolute RA. Placement error from a float32
    # downcast scales with |RA|, so a test at RA ~ 10 (the SSC pointing) is
    # ~60x less sensitive to that failure mode than one at RA ~ 260.
    POINTING = (260.0, -10.0, 120.0)

    @staticmethod
    def _reference_float64(ra, dec, pointing_ra, pointing_dec, pointing_pa):
        """Evaluate the same transform in float64 NumPy, independently.

        The projection is the *textbook* gnomonic formula (xi = cos d sin dra
        / D etc.), a different derivation from the rotate-to-pole form in the
        live code, so agreement checks the arithmetic and not just the
        transcription. (Was the flat-sky formula until the gnomonic projection
        landed; at this Dec -10 pointing the flat-sky reference differs from
        the live code by ~1 px, far over the 1e-2 px bound below.)
        """
        rad = np.deg2rad
        dra = rad(np.float64(ra)) - rad(np.float64(pointing_ra))
        d, d0 = rad(np.float64(dec)), rad(np.float64(pointing_dec))
        D = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(dra)
        dx = np.rad2deg(np.cos(d) * np.sin(dra) / D)
        dy = np.rad2deg(
            (np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(dra)) / D
        )
        theta = np.deg2rad(np.float64(pointing_pa) + 180.0 - 60.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        xy = rot @ np.stack([dx, dy])
        return -xy[0], -xy[1]

    def _sources(self, n=512):
        pointing_ra, pointing_dec, _ = self.POINTING
        np.random.seed(2026)
        ra = pointing_ra + np.random.uniform(-0.4, 0.4, size=n)
        dec = pointing_dec + np.random.uniform(-0.4, 0.4, size=n)
        return ra, dec

    def test_cpu_and_gpu_agree(self):
        """Same offsets rotated on CPU and on GPU must match sub-milli-pixel."""
        pointing_ra, pointing_dec, pointing_pa = self.POINTING
        ra, dec = self._sources()

        dx, dy = omj.sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec)

        cpu = jax.devices('cpu')[0]
        gpu = jax.devices('gpu')[0]

        x_cpu, y_cpu = omj.get_fpa_pos_from_offsets(
            jax.device_put(jnp.asarray(dx), cpu),
            jax.device_put(jnp.asarray(dy), cpu),
            pointing_pa,
        )
        x_gpu, y_gpu = omj.get_fpa_pos_from_offsets(
            jax.device_put(jnp.asarray(dx), gpu),
            jax.device_put(jnp.asarray(dy), gpu),
            pointing_pa,
        )

        diff_px = _deg_to_px(
            np.hypot(np.asarray(x_cpu) - np.asarray(x_gpu),
                     np.asarray(y_cpu) - np.asarray(y_gpu))
        )
        # Tolerance is ~10 float32 ulp, not bit-identity. At the field radius
        # (~0.4 deg) one float32 ulp is 9.8e-4 px, and CPU and GPU legitimately
        # differ by 0-2 ulp because their libm cos/sin disagree in the last bit
        # when building the rotation matrix. Measured on an a10g: max 0.0020 px,
        # quantised in exact ulp steps.
        #
        # This still separates the two regimes by ~180x: TF32 has eps 4.9e-4
        # against float32's 1.2e-7, and reproduced the original defect at
        # 1.84 px median / 7.08 px max.
        assert diff_px.max() < 1e-2, (
            f"CPU and GPU sky->FPA differ by {diff_px.max():.4f} px. If this "
            "is ~1 px or more, the rotation matmul has lost precision='highest' "
            "and is running as TF32 on the GPU."
        )

    @pytest.mark.parametrize("device_kind", ["cpu", "gpu"])
    def test_matches_float64_reference(self, device_kind):
        """Each device must match an independent float64 evaluation.

        Guards against both devices being wrong in the same way, which a pure
        CPU-vs-GPU comparison cannot see.
        """
        pointing_ra, pointing_dec, pointing_pa = self.POINTING
        ra, dec = self._sources()

        dx, dy = omj.sky_to_tangent_offsets(ra, dec, pointing_ra, pointing_dec)
        device = jax.devices(device_kind)[0]

        x, y = omj.get_fpa_pos_from_offsets(
            jax.device_put(jnp.asarray(dx), device),
            jax.device_put(jnp.asarray(dy), device),
            pointing_pa,
        )

        x_ref, y_ref = self._reference_float64(
            ra, dec, pointing_ra, pointing_dec, pointing_pa
        )
        diff_px = _deg_to_px(
            np.hypot(np.asarray(x) - x_ref, np.asarray(y) - y_ref)
        )
        assert diff_px.max() < 1e-2, (
            f"{device_kind} sky->FPA differs from the float64 reference by "
            f"{diff_px.max():.4f} px"
        )

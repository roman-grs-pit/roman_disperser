# GPU Verification Checklist for Runpod

This checklist documents the steps to verify that the JAX-based disperser code runs correctly on a GPU (tested on Runpod with NVIDIA GPUs).

## 1. Setup & Installation

- [x] Clone/sync repo to `/workspace` (persists across container restarts)
- [x] Install pixi:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
- [x] Add pixi to PATH (add to `/workspace/.bashrc` for persistence):
  ```bash
  export PATH="$HOME/.pixi/bin:$PATH"
  ```
- [x] Run `pixi install` in the repo directory
- [x] Activate CUDA environment: `pixi shell cuda` (or use `pixi run -e cuda <command>`)

## 2. GPU Verification

Run these checks before running any tests or notebooks.

- [x] Verify JAX sees the GPU:
  ```bash
  pixi run -e cuda check-jax
  ```
  Expected output: `Backend: gpu` and `GpuDevice(id=0)`

- [x] Verify CUDA version compatibility:
  ```bash
  nvidia-smi
  ```
  Should show CUDA 12.x (matches `pixi.toml` requirement)

- [x] Quick Python sanity check:
  ```python
  import jax
  import jax.numpy as jnp
  x = jnp.ones(1000)
  print(f"Backend: {jax.default_backend()}")
  print(f"Device: {x.devices()}")
  ```
  Should show `gpu` backend and `GpuDevice`

## 3. Run Tests

- [x] Run full test suite:
  ```bash
  pixi run -e cuda pytest -q tests
  ```
- [x] All tests should pass
- [x] Compare results to CPU runs (should be identical within tolerances)
- [x] Note any numerical differences (float32 GPU vs CPU can differ slightly)

## 4. Notebook Verification

- [x] Copy notebook for GPU testing:
  ```bash
  cp notebooks/demos/multi_galaxy_demo.ipynb notebooks/demos/multi_galaxy_demo_gpu.ipynb
  ```

- [x] Launch Jupyter:
  ```bash
  pixi run -e cuda jupyter lab --ip=0.0.0.0 --port=8888
  ```

- [x] Add GPU verification cell at the top of the notebook:
  ```python
  import jax
  print(f"Backend: {jax.default_backend()}")
  print(f"Devices: {jax.devices()}")
  assert jax.default_backend() == 'gpu', "Not using GPU!"
  ```

- [x] Run notebook with default `N_GALAXIES=15`
- [x] Compare timing to CPU runs (expect 5-10x speedup)
- [x] Verify flux conservation metrics match CPU results

## 5. Scale-Up Testing

- [ ] Increase `N_GALAXIES` progressively: 100 → 500 → 1000
- [ ] Monitor GPU memory in another terminal:
  ```bash
  watch -n 1 nvidia-smi
  ```
- [ ] Track timing per galaxy as you scale
- [ ] Note if any OOM errors occur and at what scale

## 6. Additional Verification

### Numerical Consistency

- [ ] Save CPU results to a file (on Mac)
- [ ] Compare GPU results to CPU results (should match within `rtol=1e-4`)
- [ ] Watch for any NaN or Inf values in output

### JIT Compilation Verification

- [x] First run includes compilation overhead
- [x] Second run should be noticeably faster
- [x] Use proper timing with synchronization:
  ```python
  import time
  output.block_until_ready()  # Force sync before timing
  start = time.time()
  result = disperse_jit(...)
  result.block_until_ready()  # Force sync after
  print(f"Time: {time.time() - start:.3f}s")
  ```

### Memory Profiling (Optional)

- [ ] Track peak GPU memory during multi-galaxy runs
- [ ] Try different `WAVELENGTH_CHUNK_SIZE` values (50, 100, 200)
- [ ] Document memory/speed tradeoffs observed

## 7. Results Log

**Test Environment:** NVIDIA RTX A5000, CUDA 12.4, Driver 550.127.05

| Test | Status | Notes |
|------|--------|-------|
| `check-jax` shows GPU | ✅ | Backend: gpu, CudaDevice(id=0) |
| All pytest tests pass | ✅ | 134 passed in 24s |
| Single galaxy notebook runs | ✅ | |
| Multi-galaxy (N=15) runs | ✅ | 0.286s total (3 orders, cached) |
| Multi-galaxy (N=100) runs | — | Not tested |
| Multi-galaxy (N=500) runs | — | Not tested |
| Multi-galaxy (N=1000) runs | — | Not tested |

### Timing Comparison

**Configuration:** 15 galaxies, 150×150 images (3× oversampled), 1000 wavelengths, 3 orders

| Configuration | CPU | GPU (RTX A5000) | Speedup |
|---------------|-----|-----------------|---------|
| 15 galaxies, 3 orders (cached) | 14.58s | 0.29s | **51x** |
| Per galaxy average | 0.324s | 0.006s | **54x** |

### JIT Compilation Speedup (GPU)

| Order | First Call | Cached Call | JIT Speedup |
|-------|------------|-------------|-------------|
| Order 1 | 0.691s | 0.064s | 10.8x |
| Order 0 | 0.654s | 0.171s | 3.8x |
| Order 2 | 0.534s | 0.052s | 10.3x |

### Flux Conservation

All orders show identical flux conservation between CPU and GPU (within float32 precision):
- Order 1: 99.99% conservation
- Order 0: 66.33% conservation (spectra extend off detector)
- Order 2: 72.79% conservation (spectra extend off detector)

---

## Appendix: Claude Code on Runpod

To persist Claude Code across container restarts:

```bash
# Create persistent directory on /workspace
mkdir -p /workspace/.claude

# Symlink from home directory
ln -s /workspace/.claude ~/.claude

# Add to /workspace/.bashrc for persistence
echo 'export PATH="/workspace/.local/share/claude-code/bin:$PATH"' >> /workspace/.bashrc
```

Alternative: Set XDG directories before installation:
```bash
export XDG_CONFIG_HOME=/workspace/.config
export XDG_DATA_HOME=/workspace/.local/share
```

# GPU Verification Checklist for Runpod

This checklist documents the steps to verify that the JAX-based disperser code runs correctly on a GPU (tested on Runpod with NVIDIA GPUs).

## 1. Setup & Installation

- [ ] Clone/sync repo to `/workspace` (persists across container restarts)
- [ ] Install pixi:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
- [ ] Add pixi to PATH (add to `/workspace/.bashrc` for persistence):
  ```bash
  export PATH="$HOME/.pixi/bin:$PATH"
  ```
- [ ] Run `pixi install` in the repo directory
- [ ] Activate CUDA environment: `pixi shell cuda` (or use `pixi run -e cuda <command>`)

## 2. GPU Verification

Run these checks before running any tests or notebooks.

- [ ] Verify JAX sees the GPU:
  ```bash
  pixi run -e cuda check-jax
  ```
  Expected output: `Backend: gpu` and `GpuDevice(id=0)`

- [ ] Verify CUDA version compatibility:
  ```bash
  nvidia-smi
  ```
  Should show CUDA 12.x (matches `pixi.toml` requirement)

- [ ] Quick Python sanity check:
  ```python
  import jax
  import jax.numpy as jnp
  x = jnp.ones(1000)
  print(f"Backend: {jax.default_backend()}")
  print(f"Device: {x.devices()}")
  ```
  Should show `gpu` backend and `GpuDevice`

## 3. Run Tests

- [ ] Run full test suite:
  ```bash
  pixi run -e cuda pytest -q tests
  ```
- [ ] All tests should pass
- [ ] Compare results to CPU runs (should be identical within tolerances)
- [ ] Note any numerical differences (float32 GPU vs CPU can differ slightly)

## 4. Notebook Verification

- [ ] Copy notebook for GPU testing:
  ```bash
  cp notebooks/demos/multi_galaxy_demo.ipynb notebooks/demos/multi_galaxy_demo_gpu.ipynb
  ```

- [ ] Launch Jupyter:
  ```bash
  pixi run -e cuda jupyter lab --ip=0.0.0.0 --port=8888
  ```

- [ ] Add GPU verification cell at the top of the notebook:
  ```python
  import jax
  print(f"Backend: {jax.default_backend()}")
  print(f"Devices: {jax.devices()}")
  assert jax.default_backend() == 'gpu', "Not using GPU!"
  ```

- [ ] Run notebook with default `N_GALAXIES=15`
- [ ] Compare timing to CPU runs (expect 5-10x speedup)
- [ ] Verify flux conservation metrics match CPU results

## 5. Scale-Up Testing

- [ ] Increase `N_GALAXIES` progressively: 100 → 500 → 1000
- [ ] Monitor GPU memory in another terminal:
  ```bash
  watch -n 1 nvidia-smi
  ```
- [ ] Track timing per galaxy as you scale
- [ ] Note if any OOM errors occur and at what scale
- [ ] Consider removing diagnostic zoom plots for large runs (reduces memory/time)

## 6. Additional Verification

### Numerical Consistency

- [ ] Save CPU results to a file (on Mac)
- [ ] Compare GPU results to CPU results (should match within `rtol=1e-4`)
- [ ] Watch for any NaN or Inf values in output

### JIT Compilation Verification

- [ ] First run includes compilation overhead
- [ ] Second run should be noticeably faster
- [ ] Use proper timing with synchronization:
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

| Test | Status | Notes |
|------|--------|-------|
| `check-jax` shows GPU | | |
| All pytest tests pass | | |
| Single galaxy notebook runs | | |
| Multi-galaxy (N=15) runs | | |
| Multi-galaxy (N=100) runs | | |
| Multi-galaxy (N=500) runs | | |
| Multi-galaxy (N=1000) runs | | |

### Timing Comparison

| Configuration | CPU (Mac) | GPU (Runpod) | Speedup |
|---------------|-----------|--------------|---------|
| Single galaxy, 1 order | | | |
| 15 galaxies, 3 orders | | | |
| 100 galaxies, 1 order | | | |
| 1000 galaxies, 1 order | | | |

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

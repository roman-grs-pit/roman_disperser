The code aims to disperse the flux in a box in x-y-lambda into a region x'-y' following a coded optical model. The box is divided into a set of cells defined by a set of grid points. Currently the code calculates the dispersed position of the center of each cell, and then distributes the flux in the cell by a bilinear interpolation onto the dispersed grid. This isn't correct, since the flux could fall completely into one cell - the input cell might be a lot smaller.

The first part of the fix is to replace the bilinear interpolation with a simpler grid deposition.

To be more careful with how the flux gets distributed, one needs relatively smaller grid cells, and computing the full dispersion solution would get expensive for these points. So instead we propose the following algorithm :

1. Divide the x-y-lambda into cells; these could be "relatively large cells".
2. Calculate the dispersion solution for the central point, but also calculate the Jacobian of that transformation. This is easy since the code is all in JAX.
3. Allocate points using a Sobol sequence within the cell.
4. Calculate the flux at each point using the input image and spectrum.
5. Disperse the points using the Jacobian. Put these points in a smaller subgrid, also save the position of the subgrid.
6. Repeat 3-5 for all cells.
7. Normalize all the dispersed points by the full flux in the input box.
8. Deposit each of the subgrids into the full 4088x4088 grid.


## Implementation Plan

### Step 0: Jacobian Accuracy Validation ✓ COMPLETE

**Goal:** Test that the Jacobian calculation is accurate enough for the proposed algorithm.

**Test parameters:**
- Cell size: 10 × 10 SCA pixels × 100Å (0.01 μm)
- Spatial range: -500 to 5500 SCA pixels (extends beyond detector)
- Wavelength range: 0.9 to 2.0 μm
- All 18 SCAs, orders "0", "1", "2" (54 configurations)
- 1000 random cells sampled per configuration

**Method:** For each cell, compute dispersion at the 8 corners using both:
1. Full solution: `omj.trace_sca_to_sca(payload, x, y, λ)`
2. Jacobian approximation: `center_output + J @ [dx, dy, dλ]`

Measure the maximum Euclidean error in output (x', y') pixels.

**Results:**
| Metric | Value |
|--------|-------|
| Worst max error | 0.0079 pixels |
| Best max error | 0.0020 pixels |
| Mean of max errors | 0.0047 pixels |
| Worst 99th percentile | 0.0054 pixels |

**Conclusion:** ✓ EXCELLENT - All errors < 0.01 pixel. The Jacobian approximation is validated for 10×10×100Å cells.

**Artifacts:**
- `notebooks/demos/jacobian_accuracy_test.ipynb` - exploration notebook with visualizations
- `notebooks/demos/jacobian_accuracy_results.json` - full validation results (54 configs)
- `tests/test_jacobian_accuracy.py` - regression test (54 parametrized tests)


## Implementation Notes

### Key Functions

**`omj.trace_sca_to_sca(payload, xsca, ysca, wavelength)`**
- Computes full dispersion: (xsca, ysca, λ) → (xsca', ysca')
- Chains: `sca_to_fpa` → `trace_beam` → `mpa_to_sca`
- Returns tuple `(xsca_out, ysca_out)`

**Jacobian computation:**
```python
def compute_jacobian_at_point(payload, xsca, ysca, wavelength):
    def trace_single(inputs):
        xout, yout = omj.trace_sca_to_sca(payload, inputs[0:1], inputs[1:2], inputs[2:3])
        return jnp.stack([xout, yout]).squeeze()
    return jax.jacobian(trace_single)(jnp.array([xsca, ysca, wavelength]))
```

**Typical Jacobian values (at detector center, λ=1.5μm):**
- ∂x'/∂x ≈ 0.98, ∂x'/∂y ≈ 0, ∂x'/∂λ ≈ -6 pix/μm (small cross-dispersion)
- ∂y'/∂x ≈ 0, ∂y'/∂y ≈ 1, ∂y'/∂λ ≈ 913 pix/μm (main dispersion direction)

### JIT Compilation Pattern

The payload contains non-traceable strings (`wl_transform`), so use the closure pattern:
```python
payload = omj.make_sca_payload(model, sca=5, order="1")

@jax.jit
def my_jitted_function(x, y, lam):
    return omj.trace_sca_to_sca(payload, x, y, lam)  # payload captured in closure
```

See `docs/jit_compilation.md` for full details.


## Next Steps

### Steps 1-8: Sobol Disperser Implementation

1. **Divide x-y-lambda into cells** - "relatively large cells" (e.g., 10×10×100Å validated)
2. **Calculate dispersion + Jacobian at cell center**
3. **Allocate points using Sobol sequence within cell**
4. **Calculate flux at each point** using input image and spectrum
5. **Disperse points using Jacobian** - put in smaller subgrid, save position
6. **Repeat 3-5 for all cells**
7. **Normalize** dispersed points by full flux in input box
8. **Deposit subgrids** into full 4088×4088 detector grid


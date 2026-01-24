# Goal

Disperse a set of stars throughout the grism.

# Proposed Algorithm (for one star)

1. Take the star's position in SCA coordinates.
2. Loop over wavelengths from 0.9 to 2.0 microns in say 100 Angstrom steps.
3. Find the new SCA position for the dispersed position using the optical model.
4. Interpolate the PSF at this position and wavelength.
5. Deposit/accumulate the PSF position into the output image.

# Design Phases

## Phase 1 : Build data model for PSF intepolation

- Explore the enclosed energy and determine how many pixels we will need.
- Oversampling of ~4 (parameter, but pick a good default)
- Needs to be GPU friendly (in terms of memory, access patterns, and parallelism)
- Determine the PSF interpolation method (I think we want trilinear) and implement it efficiently for GPU execution.
- what spatial and wavelength grid do we need here? Define some validation tests.
- write functions to build grids for different grism orders/detectors and do the necessary interpolations.

## Phase 2 : Single star

Implement the proposed algorithm
- For now, use GRISM0 for zeroth order, GRISM1 for all others (1,2)

## Phase 3 : Scaling tests

- See how efficiently this runs with large numbers of stars.

## Phase 4 : Incorporate grism efficiencies + spectra 

- start with a single spectrum.



# Requirements

- Use JAX, write for efficiency and ability to jit, and run efficiently on a GPU.




# References

@docs/disperser_design.md
@docs/optical_model.md
@docs/stpsf.md
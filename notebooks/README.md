# Notebooks

Demonstration notebooks for the roman_disperser package.

## Demo Notebooks (`demos/`)

- **`single_galaxy_demo.ipynb`**: Single galaxy dispersion demonstration
  - Creates synthetic galaxy profile and spectrum
  - Disperses onto detector using the optical model
  - Demonstrates JIT compilation for performance

- **`multi_galaxy_demo.ipynb`**: Multi-galaxy batch dispersion
  - Generates multiple galaxies at random positions
  - Batch dispersion using `disperse_galaxies_sequential`
  - Shows how to process ~1000s of galaxies efficiently

## Archive (`archive/`)

Legacy development and testing notebooks:
- **`quicklook_jax.ipynb`**: Visualization of spectral traces for orders 0, +/-1

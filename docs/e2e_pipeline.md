# End-to-end pipeline: catalog + spectra → realistic dispersed WFI image

```
sources table + spectra ──make_catalog.py──▶ catalog dir (metadata.parquet + seds.zarr)
                                                │
                     config.yaml + pointings.ecsv
                                                ▼
  stage 1 (GPU, L40S)   build_dispersed_image.py  ──▶  per-SCA FITS: MODEL (e/s, noiseless), ISIM
                                                │
  stage 2 (CPU)         wrap_with_romanisim.py  ──▶  per-SCA L1/L2 ASDF (romanisim)
                        (MODEL injected as a romanisim source)
```

`scripts/slurm_e2e.sh CONFIG POINTINGS` submits both stages on a Slurm cluster
(stage 2 waits for stage 1 and runs as a CPU job array).

## 1. Inputs

### Source table (ECSV/CSV/FITS)

| column | unit | required | notes |
|---|---|---|---|
| `id` | – | yes | unique label |
| `ra`, `dec` | deg | yes | ICRS |
| `type` | – | no | `PSF` (default) or `SER` (Sérsic) |
| `mag` | AB | no | F158 magnitude; the spectrum is rescaled to it. Omit to use the spectrum's absolute FLAM. |
| `n`, `half_light_radius` | –, arcsec | for `SER` | Sérsic index, half-light radius |
| `pa`, `ba` | deg, – | no | position angle E of N (0), axis ratio (1) |
| `spectrum` | – | no | key into the spectra file (default: `id`) |
| `z` | – | no | bookkeeping only |

### Spectra

A table with `wavelength` (Å, increasing) and one column per spectrum key in
FLAM (erg s⁻¹ cm⁻² Å⁻¹), or an `.npz` with `wavelength` plus one array per key.
They are interpolated onto the catalog grid (7500–21000 Å, 2 Å), which serves
both the grism (9000–20000 Å) and the prism (7500–18500 Å). Give spectra at
≤ 2 Å sampling where lines matter.

```bash
python scripts/make_catalog.py --sources src.ecsv --spectra spec.fits --out my_cat/
python scripts/make_catalog.py --mock --pointing 10.0 0.0 0.0 --sca 5 --out mock_cat/   # test field
```

The mock field puts stars (17–21 mag, blackbodies), emission-line galaxies
(z = 1–1.8, Hα/[NII]/[OIII]/Hβ, 12 Å FWHM lines) and continuum galaxies on one
SCA, and writes `truth.ecsv` (SCA positions, magnitudes, line wavelengths and
EWs). Existing catalogs in the same format (`catalog-v2`, the Galacticus /
XMM-LSS builds from `build_source_catalog.py`) are used as they are.

### Config and pointings

As for `build_dispersed_image.py` (`docs/grism_pipeline.md`): a YAML config
(`element: grism|prism`, `scas`, `catalog_dir`, `output_dir`, `seed`, …) and
an APT-style pointing ECSV (`RA DEC PA BANDPASS MA_TABLE_NUMBER …`).

## 2. Stage 1: dispersion (GPU)

`build_dispersed_image.py --config C --pointings P`. Per SCA it writes
`MODEL` (noiseless count rate, e⁻/s, sources only) and `ISIM` (a Poisson draw
of `MODEL × EXPTIME`), a source manifest and provenance. Run it on an
**L40S**: it is 1.5–2.5× faster than A40/A5000 on dense fields and never slower.

## 3. Stage 2: detector simulation (romanisim, CPU)

`wrap_with_romanisim.py --input-dir OUT --output-dir L2 --level 2` runs
`romanisim-make-image` per SCA product, at the product's pointing, SCA, MA
table and seed. It needs a romanisim with the IPC fix (#384) on `PYTHONPATH`.

**Counts mode `inject` (default).** `romanisim_inject_rate.py` adds
`MODEL × exptime` at the point where romanisim renders its own sources. The
dispersed light therefore gets exactly what a romanisim source gets:

- Poisson noise;
- QE thinning (relative QE from the CRDS flat; the grism and prism have no
  flat, so romanisim derives it from the gain map and pixel area);
- sky, dark, IPC, cosmic rays, non-linearity, read noise, saturation;
- the L2 ramp fit.

`exptime` is romanisim's MA-table time. The romanisim PSF, which nothing
uses with `--nobj 0`, is not built.

**Counts mode `extra-counts` (legacy).** The old `--extra-counts FILE ISIM`
adds the disperser's Poisson draw *after* QE and noise. The mean response is
the same as `inject`, but the light skips romanisim's QE, and the Poisson
noise is drawn twice for a source that is also in the sky term. Kept only for
comparison.

Cost: ~260–290 s and ~8 GB per SCA, single-threaded. Most of that (~200 s)
is romanisim's read-by-read electron apportioning (`l1.apportion_counts_to_resultants`).
Run SCAs in parallel: `--num-threads` concurrent SCAs per worker, workers as
a job array.

## 4. Output units (romanisim L2)

L2 `data` responds to every input (sky and injected sources alike) with the
same factor: ≈ 0.87 L2 unit per e⁻/s at the SCA centre (5 e⁻/s →
4.36, sky 2.32 e⁻/s → 2.03; SCA 5, MA table 1036). This is romanisim's L2
convention, not a disperser effect; convert with the L2 `meta.photometry`.
See the notes repo `agent_work/tests_cluster/E2E_PIPELINE.md` for the
measurements.

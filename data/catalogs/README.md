# Source Catalog Format

Unified catalog format for Roman grism simulations, storing both stars and
galaxies in a common structure. Designed for fast spatial queries and efficient
SED lookup.

## Directory Layout

```
data/catalogs/
  README.md              # This file
  metadata.parquet       # Source metadata (stars + galaxies)
  seds.zarr/             # Zarr store containing:
    star_seds/           #   Stellar SED templates (24 × 5501)
    galaxy_seds/         #   Group with per-sim arrays:
      sim_001/           #     SEDs from galacticus sub_1 (N₁ × 5501)
      sim_002/           #     SEDs from galacticus sub_2 (N₂ × 5501)
      ...                #     (one array per simulation sub-file)
```

## Wavelength Grid

All SED arrays share a common wavelength grid covering the Roman grism range:

```python
import numpy as np
wavelengths = np.linspace(9000, 20000, 5501)  # Angstroms, 2 Å spacing
```

This is trimmed from the full Galacticus grid (`np.linspace(2000, 40000, 19001)`)
to the grism-relevant range (0.9–2.0 microns).

## Metadata (Parquet)

A single Parquet file with one row per source (stars and galaxies together).
Column metadata (units, descriptions) is embedded in the Parquet schema.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `ra` | float64 | deg | Right ascension (ICRS) |
| `dec` | float64 | deg | Declination (ICRS) |
| `type` | string | — | Source type: `"PSF"` (point source) or `"SER"` (Sérsic profile) |
| `n` | float32 | — | Sérsic index. 1.0 = exponential disk, 0 = point source |
| `half_light_radius` | float32 | arcsec | Half-light radius. 0 for point sources |
| `pa` | float32 | deg | Position angle (E of N). 0 for point sources |
| `ba` | float32 | — | Minor-to-major axis ratio (0–1). 1 for point sources |
| `F158` | float32 | maggies | F158 flux. AB mag = −2.5 × log10(F158) |
| `z_obs` | float32 | — | Observed redshift (includes peculiar velocity). 0 for stars |
| `z_cosmo` | float32 | — | Cosmological redshift. 0 for stars |
| `sed_index` | int32 | — | Row index into the SED Zarr array (per `type` and `sim`) |
| `flux_scale` | float32 | — | SED multiplier. Galaxy SEDs: 1.0. Star SEDs: `10^(−0.4 × mag)` |
| `sim` | int16 | — | Galacticus sub-file number (1–100). 0 for stars |

For galaxies, `sed_index` is the row index within the per-sim array
`seds.zarr/galaxy_seds/sim_{sim:03d}` (identical to the original HDF5 row
index). For stars, `sed_index` is the template index into `seds.zarr/star_seds`.

The column structure follows the
[romanisim catalog convention](https://romanisim.readthedocs.io/en/latest/romanisim/catalog.html),
extended with SED lookup columns.

### Reconstructing a source spectrum

```python
import pyarrow.parquet as pq
import zarr
import numpy as np

# Load metadata
meta = pq.read_table("data/catalogs/metadata.parquet").to_pandas()

# Open SED store
store = zarr.open("data/catalogs/seds.zarr", mode="r")

# Look up one source
row = meta.iloc[42]
if row["type"] == "PSF":
    sed = np.array(store["star_seds"][row["sed_index"]]) * row["flux_scale"]
else:
    key = f"galaxy_seds/sim_{row['sim']:03d}"
    sed = np.array(store[key][row["sed_index"]]) * row["flux_scale"]
```

## Star SEDs (Zarr)

**Path:** `seds.zarr/star_seds`
**Shape:** `(N_templates, 5501)` — typically 24 unique stellar templates.

Templates are normalized to 0 AB magnitude in the F158 band. The per-source
`flux_scale` in the metadata applies the actual magnitude:
`flux_scale = 10^(−0.4 × mag_F158)`.

Multiple stars share the same `sed_index` (template). The 87,039 stars in the
catalog map to only 24 unique spectral shapes.

**Source:** Resampled from the Pickles stellar atlas templates in
`data/stars/SEDtemplates/` onto the common wavelength grid.

## Galaxy SEDs (Zarr)

**Path:** `seds.zarr/galaxy_seds/sim_{NNN}`
**Shape per sim:** `(N_sources, 5501)` — one row per galaxy in that sub-file.

Galaxy SEDs are stored in per-simulation arrays that mirror the original
Galacticus HDF5 file structure. Each `sim_NNN` array corresponds to
`galacticus_FOV_EVERY100_sub_{N}.hdf5`, and `sed_index` equals the original
HDF5 row index.

This structure allows:
- **Incremental extraction** — process one sim file at a time
- **Parallel extraction** — one worker per sim file, no coordination needed
- **Append without rewriting** — adding sim_002 doesn't touch sim_001

SEDs are observed-frame (redshifted), with dust attenuation applied
(Calzetti model, Av = 1.6523). Emission lines are included in the continuum.
The SED units are the native Galacticus output (absolute luminosity density in
AB zeropoint units); `flux_scale = 1.0` for all galaxies.

**Source:** Extracted from the Galacticus 4 deg² mock
(`galacticus_FOV_EVERY100_sub_*.hdf5`), trimmed to grism wavelength range.

## Zarr Compression

All Zarr arrays use:
- **Codec:** blosc + zstd (level 3) with byte shuffle
- **Chunks:** `(1000, 5501)` — each chunk holds 1000 sources × all wavelengths
- **Dtype:** float32

This achieves ~3.2× compression on galaxy SEDs. Star templates are too small
to matter.

Chosen over HDF5 for faster I/O (~5× read speed via blosc multi-threaded
decompression vs single-threaded gzip) and cloud compatibility (each chunk is
a separate file, works with S3/GCS via fsspec).

## File Sizes (single simulation, sub_1)

| File | Sources | Raw (f32) | Compressed |
|------|---------|-----------|------------|
| `metadata.parquet` | ~117,000 | — | ~3 MB |
| `seds.zarr/star_seds` | 24 templates | 528 KB | ~200 KB |
| `seds.zarr/galaxy_seds/sim_001` | 29,956 | 659 MB | ~206 MB |
| **Total** | | | **~209 MB** |

For the full 100-simulation catalog: ~41 GB compressed for galaxy SEDs,
metadata and star templates remain negligible.

## Reading and Writing

### Dependencies

```
pip install pyarrow zarr h5py
```

Or via pixi (this repo):
```
pixi install  # h5py, pyarrow, zarr are in pyproject.toml dependencies
```

### Reading with PyArrow + Zarr

```python
import pyarrow.parquet as pq
import zarr
import numpy as np

# --- Read metadata ---
meta = pq.read_table("data/catalogs/metadata.parquet")

# Inspect schema and column metadata
for field in meta.schema:
    md = field.metadata or {}
    unit = md.get(b"unit", b"").decode()
    desc = md.get(b"description", b"").decode()
    print(f"{field.name}: {field.type}, unit={unit}, desc={desc}")

# Convert to pandas for convenience
df = meta.to_pandas()

# Cone search example
ra0, dec0, radius = 10.0, 0.5, 0.1  # degrees
mask = (df.ra - ra0)**2 + (df.dec - dec0)**2 < radius**2  # approximate
sources = df[mask]

# --- Read SEDs ---
store = zarr.open("data/catalogs/seds.zarr", mode="r")

# Load SEDs for selected sources
wavelengths = np.linspace(9000, 20000, 5501)
for _, row in sources.iterrows():
    if row["type"] == "PSF":
        sed = np.array(store["star_seds"][row["sed_index"]])
    else:
        sed = np.array(store[f"galaxy_seds/sim_{row['sim']:03d}"][row["sed_index"]])
    sed = sed * row["flux_scale"]
```

### Writing with PyArrow + Zarr

```python
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BytesCodec, BloscCodec

# --- Write metadata ---
# Define schema with column metadata
fields = [
    pa.field("ra", pa.float64(), metadata={"unit": "deg", "description": "Right ascension (ICRS)"}),
    pa.field("dec", pa.float64(), metadata={"unit": "deg", "description": "Declination (ICRS)"}),
    pa.field("type", pa.string(), metadata={"description": "PSF (star) or SER (galaxy)"}),
    # ... etc for all columns
]
schema = pa.schema(fields)
table = pa.table({"ra": ra_array, "dec": dec_array, ...}, schema=schema)
pq.write_table(table, "data/catalogs/metadata.parquet")

# --- Write SEDs ---
codecs = [BytesCodec(), BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]
store = zarr.open("data/catalogs/seds.zarr", mode="w")

# Star templates (tiny)
store.create_array("star_seds", data=star_template_array, dtype="float32",
                   chunks=(24, 5501), codecs=codecs)

# Galaxy SEDs — one array per sim file
for sim_num in range(1, 101):
    galaxy_data = extract_from_hdf5(sim_num)  # (N_sources, 5501) float32
    store.create_array(f"galaxy_seds/sim_{sim_num:03d}",
                       data=galaxy_data, dtype="float32",
                       chunks=(1000, 5501), codecs=codecs)
```

## Provenance

- **Galaxy SEDs and metadata:** Galacticus 4 deg² mock
  (semi-analytical model on UNIT N-body merger trees).
  Cosmology: Planck 2016 (H0=67.74, Om0=0.3089).
  Dust: Calzetti model, Av=1.6523 (calibrated to WISP Hα number counts).
  See `Readme_4sqdeg.txt` in the raw data for full details.

- **Star catalog:** `data/stars/sim_star_cat_galacticus.txt` with Pickles
  stellar atlas SED templates from `data/stars/SEDtemplates/`.

- **Extraction script:** `scripts/build_source_catalog.py`

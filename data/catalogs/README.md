# Source Catalog Format

Unified catalog format for Roman grism simulations, storing both stars and
galaxies in a common structure. Designed for fast spatial queries and efficient
SED lookup.

**Requires:** Zarr v3 (zarr-python ≥ 3.0) for sharding support.

## Directory Layout

```
data/catalogs/
  README.md              # This file
  metadata.parquet       # Source metadata (stars + galaxies)
  seds.zarr/             # Zarr v3 store containing:
    wavelengths           #   Common wavelength grid (5501,) float64
    star_seds/            #   Stellar SED templates (24 × 5501)
    galaxy_seds/          #   Group with per-sim arrays:
      sim_001/            #     SEDs from galacticus sub_1 (N₁ × 5501)
      sim_002/            #     SEDs from galacticus sub_2 (N₂ × 5501)
      ...                 #     (one sharded array per simulation sub-file)
```

## Wavelength Grid

All SED arrays share a common wavelength grid stored in the Zarr store at
`seds.zarr/wavelengths`. The grid covers the Roman grism range:

```python
import numpy as np
wavelengths = np.linspace(9000, 20000, 5501)  # Angstroms, 2 Å spacing
```

This is trimmed from the full Galacticus grid (`np.linspace(2000, 40000, 19001)`)
to the grism-relevant range (0.9–2.0 microns).

## SED Units

All SED arrays store **f_λ** (flux density per unit wavelength) in float32.

- **Star SEDs:** Normalized to 0 AB magnitude in the F158 band. The per-source
  `flux_scale` in the metadata applies the actual magnitude. After scaling,
  units are f_λ in the AB system (maggies per Angstrom).

- **Galaxy SEDs:** Observed-frame (redshifted), with dust attenuation applied
  (Calzetti model, Av = 1.6523). Emission lines are included. Units are the
  native Galacticus output — absolute luminosity density (f_λ) in AB zeropoint
  units. `flux_scale = 1.0` for all galaxies.

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

# Load metadata and SED store
meta = pq.read_table("data/catalogs/metadata.parquet").to_pandas()
store = zarr.open("data/catalogs/seds.zarr", mode="r")
wavelengths = np.array(store["wavelengths"])  # (5501,) Angstroms

# Look up one source
row = meta.iloc[42]
if row["type"] == "PSF":
    sed = np.array(store["star_seds"][row["sed_index"]]) * row["flux_scale"]
else:
    key = f"galaxy_seds/sim_{row['sim']:03d}"
    sed = np.array(store[key][row["sed_index"]]) * row["flux_scale"]
# sed is f_lambda in AB units, wavelengths in Angstroms
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

Galaxy SEDs are stored in per-simulation sharded arrays that mirror the original
Galacticus HDF5 file structure. Each `sim_NNN` array corresponds to
`galacticus_FOV_EVERY100_sub_{N}.hdf5`, and `sed_index` equals the original
HDF5 row index.

This structure allows:
- **Incremental extraction** — process one sim file at a time
- **Parallel extraction** — one worker per sim file, no coordination needed
- **Append without rewriting** — adding sim_002 doesn't touch sim_001
- **Efficient random access** — sharding enables single-source reads without
  loading full chunks (see Zarr Storage below)

**Source:** Extracted from the Galacticus 4 deg² mock
(`galacticus_FOV_EVERY100_sub_*.hdf5`), trimmed to grism wavelength range.

## Zarr Storage

**Format:** Zarr v3 with sharding.

All Zarr arrays use:
- **Compressor:** blosc + zstd (level 3) with byte shuffle
- **Dtype:** float32

### Sharding

Galaxy SED arrays use Zarr v3 sharding to enable efficient random access to
non-consecutive sources (e.g., all galaxies on a given SCA):

- **Shard (outer chunk):** `(N_sources, 5501)` — one shard file per sim array
- **Inner chunk:** `(10, 5501)` — random access unit (10 sources × all wavelengths)

This means each per-sim array is stored as a single file on disk with an
internal index. Reading one source requires decompressing only 10 rows (~220 KB
compressed), not the entire array. This is critical for:
- **Non-consecutive access patterns** — gathering scattered `sed_index` values
- **Cloud/S3 access** — one HTTP range request per 10-source chunk via fsspec
- **Reasonable file count** — one file per sim, not thousands of chunk files

Without sharding (flat chunks of 1000), gathering 1000 random sources takes
~21s. With sharding (inner chunks of 10), the same gather takes ~1.6s.

Star templates are too small to benefit from sharding and use a single chunk.

### Compression Performance

Measured on sub_1 (29,956 sources):

| Configuration | Compressed size | Ratio |
|---------------|-----------------|-------|
| shuffle + zstd, inner chunk=10 | 413 MB | 1.60× |

The ~1.6× compression ratio is typical for float32 scientific data with high
dynamic range (1e-7 to 1e5). Shuffle + zstd was chosen for its robustness
across chunk sizes; bitshuffle compresses poorly at small inner chunk sizes.

### Zarr Metadata (Attributes)

The Zarr store includes self-describing metadata on groups and arrays:

- **Root group:** format version, description, provenance, cosmology
- **`wavelengths`:** units (Angstrom), grid definition
- **`star_seds`:** units (f_λ, AB zeropoint, normalized to 0 mag F158),
  axis labels
- **`galaxy_seds` group:** source file pattern, number of sims
- **`galaxy_seds/sim_NNN`:** units (f_λ, AB zeropoint, native Galacticus),
  dust model, frame (observed), axis labels

## File Sizes (single simulation, sub_1)

| File | Sources | Raw (f32) | Compressed |
|------|---------|-----------|------------|
| `metadata.parquet` | ~117,000 | — | ~3 MB |
| `seds.zarr/wavelengths` | 5,501 | 44 KB | ~44 KB |
| `seds.zarr/star_seds` | 24 templates | 528 KB | ~200 KB |
| `seds.zarr/galaxy_seds/sim_001` | 29,956 | 659 MB | ~413 MB |
| **Total** | | | **~416 MB** |

For the full 100-simulation catalog: ~41 GB compressed for galaxy SEDs,
metadata and star templates remain negligible. A magnitude cut (e.g., F158 ≤ 25
or 26) substantially reduces this by excluding faint sources below detection
threshold.

## Reading and Writing

### Dependencies

```
pip install pyarrow zarr>=3.0 h5py
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
wavelengths = np.array(store["wavelengths"])  # (5501,) Angstroms

# Load SEDs for selected sources (random access via sharding)
for _, row in sources.iterrows():
    if row["type"] == "PSF":
        sed = np.array(store["star_seds"][row["sed_index"]])
    else:
        sed = np.array(store[f"galaxy_seds/sim_{row['sim']:03d}"][row["sed_index"]])
    sed = sed * row["flux_scale"]
    # sed is f_lambda, wavelengths in Angstroms
```

### Writing with PyArrow + Zarr

```python
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import BloscCodec

compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

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
store = zarr.open("data/catalogs/seds.zarr", mode="w")

# Wavelength grid
store.create_array("wavelengths", data=wavelengths, compressors=compressor,
                   attributes={"units": "Angstrom",
                               "description": "Common wavelength grid for all SEDs",
                               "grid_definition": "np.linspace(9000, 20000, 5501)"})

# Star templates (tiny, no sharding needed)
store.create_array("star_seds", data=star_template_array,
                   chunks=(24, 5501), compressors=compressor,
                   attributes={"units": "f_lambda (AB zeropoint, normalized to 0 mag F158)",
                               "axes": ["template_index", "wavelength"]})

# Galaxy SEDs — one sharded array per sim file
# chunks = inner chunk (random access unit), shards = outer shard (file on disk)
for sim_num in range(1, 101):
    galaxy_data = extract_from_hdf5(sim_num)  # (N_sources, 5501) float32
    n_src = galaxy_data.shape[0]
    # Round shard up to multiple of inner chunk size
    shard_rows = ((n_src + 9) // 10) * 10
    store.create_array(
        f"galaxy_seds/sim_{sim_num:03d}",
        data=galaxy_data,
        chunks=(10, 5501),               # inner chunk: 10 sources
        shards=(shard_rows, 5501),        # outer shard: whole array
        compressors=compressor,
        attributes={"units": "f_lambda (AB zeropoint, native Galacticus)",
                    "axes": ["sed_index", "wavelength"],
                    "dust_model": "Calzetti, Av=1.6523",
                    "frame": "observed (redshifted)"},
    )

# Group-level metadata
store["galaxy_seds"].attrs.update({
    "source_files": "galacticus_FOV_EVERY100_sub_*.hdf5",
    "n_sims": 100,
})
store.attrs.update({
    "format_version": "1.0",
    "description": "Roman grism source catalog SEDs",
    "provenance": "Galacticus 4 deg² mock + Pickles stellar atlas",
    "cosmology": "Planck 2016 (H0=67.74, Om0=0.3089)",
})
```

**Note:** The `shards` parameter requires zarr-python ≥ 3.0 (Zarr v3 format).
The shard size must be a multiple of the inner chunk size; the write example
rounds up and zero-pads trailing rows.

## Provenance

- **Galaxy SEDs and metadata:** Galacticus 4 deg² mock
  (semi-analytical model on UNIT N-body merger trees).
  Cosmology: Planck 2016 (H0=67.74, Om0=0.3089).
  Dust: Calzetti model, Av=1.6523 (calibrated to WISP Hα number counts).
  See `Readme_4sqdeg.txt` in the raw data for full details.

- **Star catalog:** `data/stars/sim_star_cat_galacticus.txt` with Pickles
  stellar atlas SED templates from `data/stars/SEDtemplates/`.

- **Extraction script:** `scripts/build_source_catalog.py`

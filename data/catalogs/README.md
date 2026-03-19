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
    star_seds/            #   Stellar SED templates (N_templates × 5501)
    galaxy_seds/          #   Group with per-partition arrays:
      sim_001/            #     Galaxy SEDs, partition 1 (N₁ × 5501)
      sim_002/            #     Galaxy SEDs, partition 2 (N₂ × 5501)
      ...                 #     (one sharded array per partition)
```

## Wavelength Grid

All SED arrays share a common wavelength grid stored in the Zarr store at
`seds.zarr/wavelengths`. The grid covers the Roman grism range:

```python
import numpy as np
wavelengths = np.linspace(9000, 20000, 5501)  # Angstroms, 2 Å spacing
```

## SED Units

All SED arrays store **f_λ (FLAM)** — apparent flux density per unit wavelength
in units of **erg/s/cm²/Å**, stored as float32. These are the physical units
expected by the grism disperser: the count rate in a detector pixel is

    counts/s = f_λ × sensitivity × Δλ

where sensitivity is the grism sensitivity curve and Δλ is the wavelength bin
width in Angstroms.

- **Star SEDs:** Normalized to 0 AB magnitude in the F158 band. The per-source
  `flux_scale` in the metadata applies the actual magnitude:
  `flux_scale = 10^(−0.4 × mag_F158)`. After scaling, the SED is in FLAM.

- **Galaxy SEDs:** Apparent f_λ (FLAM) in the observer frame, normalized to
  the catalog F158 apparent magnitude. `flux_scale = 1.0` for all galaxies —
  no further scaling needed.

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
| `sim` | int16 | — | Partition number for galaxy SED lookup. 0 for stars |

For galaxies, `sed_index` is the row index within the partition array
`seds.zarr/galaxy_seds/sim_{sim:03d}`. For stars, `sed_index` is the template
index into `seds.zarr/star_seds`.

The `sim` column partitions galaxies into groups for SED storage. Each partition
is stored as a separate Zarr array — this is an implementation detail for
efficient I/O and need not correspond to any physical grouping. A catalog with
all galaxies in a single partition (`sim=1`) is valid.

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
# sed is f_lambda in FLAM (erg/s/cm^2/Å), wavelengths in Angstroms
```

## Star SEDs (Zarr)

**Path:** `seds.zarr/star_seds`
**Shape:** `(N_templates, 5501)` — typically 24 unique stellar templates.

Templates are normalized to 0 AB magnitude in the F158 band. The per-source
`flux_scale` in the metadata applies the actual magnitude:
`flux_scale = 10^(−0.4 × mag_F158)`.

Multiple stars may share the same `sed_index` (template).

## Galaxy SEDs (Zarr)

**Path:** `seds.zarr/galaxy_seds/sim_{NNN}`
**Shape per partition:** `(N_sources, 5501)` — one row per galaxy.

Galaxy SEDs are stored in per-partition sharded arrays. Each galaxy's
`sed_index` is its row index within its partition's array.

This structure allows:
- **Incremental building** — add one partition at a time
- **Parallel building** — one worker per partition, no coordination needed
- **Append without rewriting** — adding a new partition doesn't touch existing ones
- **Efficient random access** — sharding enables single-source reads without
  loading full chunks (see Zarr Storage below)

## Zarr Storage

**Format:** Zarr v3 with sharding.

All Zarr arrays use:
- **Compressor:** blosc + zstd (level 3) with byte shuffle
- **Dtype:** float32

### Sharding

Galaxy SED arrays use Zarr v3 sharding to enable efficient random access to
non-consecutive sources (e.g., all galaxies on a given SCA):

- **Shard (outer chunk):** `(N_sources, 5501)` — one shard file per partition
- **Inner chunk:** `(10, 5501)` — random access unit (10 sources × all wavelengths)

Each partition is stored as a single file on disk with an internal index.
Reading one source requires decompressing only 10 rows (~220 KB compressed),
not the entire array. This is critical for:
- **Non-consecutive access patterns** — gathering scattered `sed_index` values
- **Cloud/S3 access** — one HTTP range request per 10-source chunk via fsspec
- **Reasonable file count** — one file per partition, not thousands of chunk files

Without sharding (flat chunks of 1000), gathering 1000 random sources takes
~21s. With sharding (inner chunks of 10), the same gather takes ~1.6s.

Star templates are too small to benefit from sharding and use a single chunk.

### Compression Performance

Measured on a partition of ~30,000 galaxy SEDs:

| Configuration | Compressed size | Ratio |
|---------------|-----------------|-------|
| shuffle + zstd, inner chunk=10 | 413 MB | 1.60× |

The ~1.6× compression ratio is typical for float32 scientific data with high
dynamic range. Shuffle + zstd was chosen for its robustness across chunk sizes;
bitshuffle compresses poorly at small inner chunk sizes.

### Zarr Metadata (Attributes)

The Zarr store includes self-describing metadata on groups and arrays:

- **Root group:** format version, description, provenance
- **`wavelengths`:** units (Angstrom), grid definition
- **`star_seds`:** units (FLAM, normalized to 0 mag F158), axis labels
- **`galaxy_seds` group:** number of partitions
- **`galaxy_seds/sim_NNN`:** units (FLAM, apparent), frame (observed),
  axis labels

## Magnitude Cut

The catalog applies a magnitude cut of **F158 ≤ 26 AB** to exclude sources below
the grism detection threshold. At mag 26, a flat AB source yields ~2.5 counts/s
integrated over the full 1st-order trace, corresponding to SNR/resolution-element
~0.3 in the deep survey (32 exposures) and ~0.1 in the wide survey (8 exposures).
This is sufficient for contamination modeling while excluding sources that
contribute negligibly to the detector signal. See `scripts/magnitude_cutoff.py`
for the full SNR analysis.

## Reading and Writing

### Dependencies

```
pip install pyarrow zarr>=3.0
```

Or via pixi (this repo):
```
pixi install  # pyarrow, zarr are in pyproject.toml dependencies
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
    # sed is f_lambda in FLAM (erg/s/cm^2/Å), wavelengths in Angstroms
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
                   attributes={"units": "FLAM (erg/s/cm^2/Å, normalized to 0 mag F158)",
                               "axes": ["template_index", "wavelength"]})

# Galaxy SEDs — one sharded array per partition
# chunks = inner chunk (random access unit), shards = outer shard (file on disk)
for sim_num in range(1, n_partitions + 1):
    galaxy_data = ...  # (N_sources, 5501) float32, FLAM
    n_src = galaxy_data.shape[0]
    # Round shard up to multiple of inner chunk size
    shard_rows = ((n_src + 9) // 10) * 10
    store.create_array(
        f"galaxy_seds/sim_{sim_num:03d}",
        data=galaxy_data,
        chunks=(10, 5501),               # inner chunk: 10 sources
        shards=(shard_rows, 5501),        # outer shard: whole array
        compressors=compressor,
        attributes={"units": "FLAM (erg/s/cm^2/Å, apparent)",
                    "axes": ["sed_index", "wavelength"],
                    "frame": "observed"},
    )

# Group-level metadata
store["galaxy_seds"].attrs.update({
    "n_partitions": n_partitions,
})
store.attrs.update({
    "format_version": "1.0",
    "description": "Roman grism source catalog SEDs",
})
```

**Note:** The `shards` parameter requires zarr-python ≥ 3.0 (Zarr v3 format).
The shard size must be a multiple of the inner chunk size; the write example
rounds up and zero-pads trailing rows.

## Provenance (Galacticus mock)

The reference catalog shipped with this repository is extracted from the
Galacticus 4 deg² mock, a semi-analytical galaxy catalog built on UNIT N-body
merger trees. The catalog contains 1/100th of the full simulation (one
sub-sample); each sub-sample covers the entire 4 deg² field, so `sim` values
1–100 correspond to independent random sub-samples of the same volume, not
spatial tiles.

- **Galaxy SEDs and metadata:** Galacticus 4 deg² mock.
  Cosmology: Planck 2016 (H0=67.74, Om0=0.3089).
  Dust: Calzetti model, Av=1.6523 (calibrated to WISP Hα number counts).
  The raw Galacticus SEDs are f_ν in internal absolute units, sampled on a
  wavelength grid; the extraction script converts to apparent FLAM by
  applying the f_ν → f_λ transformation and normalizing to the catalog F158
  apparent magnitude via synphot.
  See `Readme_4sqdeg.txt` in the raw data for full details.

- **Star catalog:** `data/stars/sim_star_cat_galacticus.txt` with Pickles
  stellar atlas SED templates from `data/stars/SEDtemplates/`.

- **Extraction script:** `scripts/build_source_catalog.py`

#!/usr/bin/env python
"""
Pad a source catalog by periodically replicating in RA.

Creates a new catalog directory with:
- A padded ``metadata.parquet`` (duplicated rows with shifted RA)
- A symlink to the original ``seds.zarr`` (SEDs are unchanged)

The user specifies the periodic box boundaries (``--ra-box-min``,
``--ra-box-max``) and the desired output range (``--ra-min``, ``--ra-max``).
Sources are replicated in whole-period shifts to cover the requested range.

Example::

    pixi run python scripts/pad_catalog.py \
        --input data/catalogs \
        --output data/catalogs_padded \
        --ra-box-min 9.0 --ra-box-max 11.0 \
        --ra-min 8.0 --ra-max 12.0

This is a temporary workaround for pointing lists that extend beyond the
catalog's native RA coverage.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def pad_catalog(input_dir, output_dir, ra_min, ra_max, ra_box_min, ra_box_max):
    """Pad a catalog by periodic RA replication.

    Parameters
    ----------
    input_dir : str or Path
        Input catalog directory (must contain metadata.parquet and seds.zarr).
    output_dir : str or Path
        Output directory for the padded catalog.
    ra_min, ra_max : float
        Desired RA range in degrees (output).
    ra_box_min, ra_box_max : float
        Periodic box boundaries in RA (degrees).  The period is
        ``ra_box_max - ra_box_min``.

    Returns
    -------
    n_copies : int
        Number of catalog copies in the padded output (including the original).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if ra_min >= ra_max:
        raise ValueError(f"ra_min ({ra_min}) must be < ra_max ({ra_max})")
    if ra_box_min >= ra_box_max:
        raise ValueError(
            f"ra_box_min ({ra_box_min}) must be < ra_box_max ({ra_box_max})"
        )

    period = ra_box_max - ra_box_min

    # Load metadata
    meta = pq.read_table(input_dir / "metadata.parquet").to_pandas()
    schema = pq.read_schema(input_dir / "metadata.parquet")

    cat_ra_min = meta["ra"].min()
    cat_ra_max = meta["ra"].max()

    # Sanity check: sources should fall within the declared box
    if cat_ra_min < ra_box_min - 0.01 or cat_ra_max > ra_box_max + 0.01:
        raise ValueError(
            f"Source RA range [{cat_ra_min:.4f}, {cat_ra_max:.4f}] extends "
            f"beyond declared box [{ra_box_min}, {ra_box_max}]"
        )

    print(f"Input catalog: {len(meta)} sources, "
          f"RA [{cat_ra_min:.4f}, {cat_ra_max:.4f}]")
    print(f"Periodic box: [{ra_box_min}, {ra_box_max}], period={period} deg")
    print(f"Requested range: RA [{ra_min}, {ra_max}] "
          f"({ra_max - ra_min:.4f} deg)")

    # Compute integer shifts needed to cover [ra_min, ra_max)
    k_min = int(np.floor((ra_min - ra_box_min) / period))
    k_max = int(np.ceil((ra_max - ra_box_max) / period))
    shifts = list(range(k_min, k_max + 1))

    print(f"Shifts: {shifts} ({len(shifts)} copies)")

    # Build padded dataframe
    import pandas as pd
    frames = []
    for k in shifts:
        shifted = meta.copy()
        shifted["ra"] = meta["ra"] + k * period
        frames.append(shifted)

    padded = pd.concat(frames, ignore_index=True)

    # Trim to requested range [ra_min, ra_max)
    # Strict upper bound avoids double-counting at period boundaries.
    mask = (padded["ra"] >= ra_min) & (padded["ra"] < ra_max)
    padded = padded[mask].reset_index(drop=True)

    print(f"Padded catalog: {len(padded)} sources, "
          f"RA [{padded['ra'].min():.4f}, {padded['ra'].max():.4f}]")

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write padded metadata (preserve original schema/column metadata)
    import pyarrow as pa
    table = pa.Table.from_pandas(padded, schema=schema, preserve_index=False)
    pq.write_table(table, output_dir / "metadata.parquet")

    # Symlink seds.zarr to original
    zarr_link = output_dir / "seds.zarr"
    zarr_target = (input_dir / "seds.zarr").resolve()
    if zarr_link.exists() or zarr_link.is_symlink():
        zarr_link.unlink()
    os.symlink(zarr_target, zarr_link)

    print(f"Output: {output_dir}")
    print(f"  metadata.parquet: {len(padded)} rows")
    print(f"  seds.zarr -> {zarr_target}")

    return len(shifts)


def main():
    parser = argparse.ArgumentParser(
        description="Pad a source catalog by periodic RA replication.",
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input catalog directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for padded catalog")
    parser.add_argument("--ra-box-min", type=float, required=True,
                        help="RA lower bound of the periodic box (degrees)")
    parser.add_argument("--ra-box-max", type=float, required=True,
                        help="RA upper bound of the periodic box (degrees)")
    parser.add_argument("--ra-min", type=float, required=True,
                        help="Minimum RA of desired output range (degrees)")
    parser.add_argument("--ra-max", type=float, required=True,
                        help="Maximum RA of desired output range (degrees)")

    args = parser.parse_args()
    pad_catalog(args.input, args.output, args.ra_min, args.ra_max,
                args.ra_box_min, args.ra_box_max)


if __name__ == "__main__":
    main()

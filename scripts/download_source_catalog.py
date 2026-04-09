#!/usr/bin/env python
"""Download pre-built source catalog from GitHub releases.

Downloads the Parquet metadata and Zarr SED store (~155 MB) from the public
roman-grs-pit/roman_disperser_data repository.

Usage:
    python scripts/download_source_catalog.py           # skip if exists
    python scripts/download_source_catalog.py --force   # re-download
"""

import argparse
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# To upload/update the catalog asset:
#   # Build the catalog:
#   pixi run python scripts/build_source_catalog.py --sims 1
#   # Create tarball:
#   cd data/catalogs && tar czf /tmp/source_catalog_sim001.tar.gz metadata.parquet seds.zarr/
#   # Upload:
#   gh release create catalog-v1 --repo roman-grs-pit/roman_disperser_data \
#     --title "Source catalog v1 (sim 1)" \
#     --notes "Galaxy + star catalog (1 sim, F158 ≤ 26). See data/catalogs/README.md for format." \
#     /tmp/source_catalog_sim001.tar.gz
CATALOG_RELEASE = "catalog-v1"
BASE_URL = f"https://github.com/roman-grs-pit/roman_disperser_data/releases/download/{CATALOG_RELEASE}"
TARBALL = "source_catalog_sim001.tar.gz"


def download(force=False):
    catalog_dir = Path(__file__).resolve().parent.parent / "data" / "catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    parquet = catalog_dir / "metadata.parquet"
    zarr_dir = catalog_dir / "seds.zarr"

    if parquet.exists() and zarr_dir.exists() and not force:
        print(f"Catalog already present at {catalog_dir}")
        print("Use --force to re-download.")
        return

    url = f"{BASE_URL}/{TARBALL}"
    print(f"Downloading {TARBALL} (~155 MB)...")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urllib.request.urlretrieve(url, tmp_path)
        print("Extracting...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=catalog_dir, filter="data")
        print(f"Catalog installed to {catalog_dir}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download source catalog from GitHub releases.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    download(force=args.force)

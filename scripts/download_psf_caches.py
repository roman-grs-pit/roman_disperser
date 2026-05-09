#!/usr/bin/env python
"""Download pre-generated PSF caches from GitHub releases.

Downloads 36 PSF cache files (~4.3 GB total) from the public
roman-grs-pit/roman_disperser_data repository.

Usage:
    python scripts/download_psf_caches.py           # skip existing files
    python scripts/download_psf_caches.py --force    # re-download all
"""

import argparse
import urllib.request
from pathlib import Path

# NOTE: Prism PSF caches have not yet been uploaded to a public release.
# Until the release exists, this script will 404; generate caches locally with
# scripts/generate_psf_caches.py instead.
PSF_RELEASE = "psf-prism-v1"
BASE_URL = f"https://github.com/roman-grs-pit/roman_disperser_data/releases/download/{PSF_RELEASE}"

PSF_FILES = [
    f"psf_WFI{sca:02d}_PRISM_4x4x56_0.75-1.85um_fov5.0_os4.npz"
    for sca in range(1, 19)
]


def download(force=False):
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "psf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = len(PSF_FILES)
    downloaded = 0

    for i, fname in enumerate(PSF_FILES, 1):
        dest = cache_dir / fname
        if dest.exists() and not force:
            print(f"[{i}/{n}] {fname} — exists, skipping")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"[{i}/{n}] Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        downloaded += 1

    if downloaded == 0:
        print(f"All {n} files already present. Use --force to re-download.")
    else:
        print(f"Downloaded {downloaded}/{n} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PSF caches from GitHub releases.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()
    download(force=args.force)

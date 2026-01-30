#!/usr/bin/env python
"""
Migrate PSF cache files from meters to microns.

This script converts the internal `wavelengths` array in existing PSF cache files
from meters to microns (multiply by 1e6).

Note: Cache filenames already display wavelengths in microns (e.g., "0.90-2.00um"),
so only the internal data needs conversion.

Usage:
    python scripts/migrate_psf_caches.py [--cache-dir PATH] [--dry-run]

Arguments:
    --cache-dir PATH    Path to PSF cache directory (default: data/psf_cache)
    --dry-run           Show what would be migrated without making changes
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def detect_wavelength_units(wavelengths):
    """Detect whether wavelengths are in meters or microns.

    Args:
        wavelengths: Array of wavelength values

    Returns:
        'meters' if values look like meters (< 1e-4)
        'microns' if values look like microns (0.1 < val < 10)
        'unknown' otherwise
    """
    min_wl = wavelengths.min()
    max_wl = wavelengths.max()

    # Meter values: ~0.9e-6 to ~2e-6
    if min_wl < 1e-4 and max_wl < 1e-4:
        return 'meters'

    # Micron values: ~0.9 to ~2.0
    if 0.1 < min_wl < 10 and 0.1 < max_wl < 10:
        return 'microns'

    return 'unknown'


def migrate_cache_file(filepath, dry_run=False):
    """Migrate a single PSF cache file from meters to microns.

    Args:
        filepath: Path to the .npz file
        dry_run: If True, only report what would be done

    Returns:
        tuple: (status, message)
            status: 'migrated', 'skipped', or 'error'
            message: Description of what happened
    """
    filepath = Path(filepath)

    try:
        # Load the cache file
        with np.load(filepath, allow_pickle=True) as data:
            # Get wavelengths
            if 'wavelengths' not in data:
                return 'error', f"No 'wavelengths' key found"

            wavelengths = data['wavelengths']
            units = detect_wavelength_units(wavelengths)

            if units == 'microns':
                return 'skipped', f"Already in microns ({wavelengths.min():.2f}-{wavelengths.max():.2f})"

            if units == 'unknown':
                return 'error', f"Unknown units (range {wavelengths.min():.2e}-{wavelengths.max():.2e})"

            # Units are in meters - need to convert
            if dry_run:
                new_min = wavelengths.min() * 1e6
                new_max = wavelengths.max() * 1e6
                return 'would_migrate', f"Would convert {wavelengths.min():.2e}-{wavelengths.max():.2e} m to {new_min:.2f}-{new_max:.2f} μm"

            # Convert and save
            # Load all data
            cache_data = {key: data[key] for key in data.files}

            # Convert wavelengths
            cache_data['wavelengths'] = wavelengths * 1e6

            # Also convert wl_grid if present (should be the same)
            if 'wl_grid' in cache_data:
                cache_data['wl_grid'] = cache_data['wl_grid'] * 1e6

        # Write back (outside the context manager to close the file first)
        np.savez(filepath, **cache_data)

        new_wavelengths = cache_data['wavelengths']
        return 'migrated', f"Converted to {new_wavelengths.min():.2f}-{new_wavelengths.max():.2f} μm"

    except Exception as e:
        return 'error', str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Migrate PSF cache files from meters to microns'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/psf_cache',
        help='Path to PSF cache directory (default: data/psf_cache)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        print("Nothing to migrate.")
        return 0

    # Find all .npz files
    npz_files = list(cache_dir.glob('*.npz'))

    if not npz_files:
        print(f"No .npz files found in {cache_dir}")
        print("Nothing to migrate.")
        return 0

    print(f"Found {len(npz_files)} cache files in {cache_dir}")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")
    print()

    # Process each file
    results = {'migrated': 0, 'would_migrate': 0, 'skipped': 0, 'error': 0}

    for filepath in sorted(npz_files):
        status, message = migrate_cache_file(filepath, dry_run=args.dry_run)
        results[status] += 1

        # Format output
        status_icons = {
            'migrated': '✓',
            'would_migrate': '→',
            'skipped': '-',
            'error': '✗'
        }
        icon = status_icons.get(status, '?')
        print(f"  {icon} {filepath.name}: {message}")

    # Summary
    print()
    print("Summary:")
    if args.dry_run:
        print(f"  Would migrate: {results['would_migrate']}")
    else:
        print(f"  Migrated: {results['migrated']}")
    print(f"  Skipped (already microns): {results['skipped']}")
    print(f"  Errors: {results['error']}")

    if args.dry_run and results['would_migrate'] > 0:
        print()
        print("Run without --dry-run to perform the migration.")

    return 0 if results['error'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
